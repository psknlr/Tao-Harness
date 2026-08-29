"""The agent runtime.

One loop serves every model and every condition.  The five experimental arms
differ only in which parts of this file are switched on:

===== ================= ============== ============== =================
Arm    structured prompt  KG evidence    model-chosen   verification pass
                                         tool calls
===== ================= ============== ============== =================
M0     no                no             no             no
M1     yes               no             no             no
M2     yes               static         no             no
M3     yes               agentic        yes            no
M4     yes               agentic        yes            yes
===== ================= ============== ============== =================

Everything else -- prompts, tool descriptions, retrieval weights, top-k, graph
hops, context budget, tool budget, decode parameters, output schema, the
verifier -- is identical, and :meth:`FrameworkConfig.framework_hash` proves it.
Two runs whose hashes differ are not comparable, and the analysis CLI refuses
to pool them.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tcm_kg.index import KGRetriever, RetrievalParams
from tcm_kg.schema import Domain
from tcm_kg.store import KGStore
from tcm_models.base import Completion, DecodeParams, LLMClient, Message
from tcm_tools.base import REGISTRY, ToolBudget, ToolContext, ToolRegistry, ToolResult

from .parsing import ParseOutcome, coerce_str, extract_json_object
from .prompts import load_prompt, prompt_fingerprint
from .tasks import ContextBudget, Task
from .trace import LLMStep, ToolStep, Trace

CONDITIONS = ("M0", "M1", "M2", "M3", "M4")

#: Topics the knowledge graph does not encode.  Used by the PA verification
#: pass to catch a model citing graph support it cannot have had.
UNCOVERED_TOPICS: Mapping[str, Tuple[str, ...]] = {
    "dose": ("剂量", "用量", "超量", "克", "g", "用法用量"),
    "incompatibility": ("十八反", "十九畏", "配伍禁忌", "相反", "相畏"),
    "pregnancy": ("妊娠", "孕妇", "孕期"),
    "drug_interaction": ("相互作用", "中西药联用"),
}


@dataclass
class FrameworkConfig:
    """Everything that must be identical across models, in one hashable object."""

    retrieval: RetrievalParams = field(default_factory=RetrievalParams)
    tool_budget: ToolBudget = field(default_factory=ToolBudget)
    context_budget: ContextBudget = field(default_factory=ContextBudget)
    decode: DecodeParams = field(default_factory=DecodeParams)
    max_agent_turns: int = 16
    #: how many times a model may emit unparseable output before the case is
    #: recorded as a format failure rather than retried indefinitely
    max_format_retries: int = 2
    registry: ToolRegistry = field(default=REGISTRY)

    def framework_hash(self) -> str:
        payload = json.dumps(
            {
                "prompts": prompt_fingerprint(),
                "tools": self.registry.fingerprint(),
                "retrieval": self.retrieval.fingerprint(),
                "tool_budget": self.tool_budget.fingerprint(),
                "context_budget": self.context_budget.fingerprint(),
                "decode": self.decode.fingerprint(),
                "max_agent_turns": self.max_agent_turns,
                "max_format_retries": self.max_format_retries,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> Dict[str, Any]:
        return {
            "framework_hash": self.framework_hash(),
            "prompt_fingerprint": prompt_fingerprint(),
            "tool_fingerprint": self.registry.fingerprint(),
            "retrieval": self.retrieval.fingerprint(),
            "tool_budget": self.tool_budget.fingerprint(),
            "context_budget": self.context_budget.fingerprint(),
            "decode": json.loads(self.decode.fingerprint()),
            "max_agent_turns": self.max_agent_turns,
        }


class AgentRuntime:
    """Runs one case under one condition with one model."""

    def __init__(
        self,
        kg: KGStore,
        retriever: KGRetriever,
        task: Task,
        model: LLMClient,
        config: Optional[FrameworkConfig] = None,
        *,
        run_id: Optional[str] = None,
    ):
        self.kg = kg
        self.retriever = retriever
        self.task = task
        self.model = model
        self.config = config or FrameworkConfig()
        self.run_id = run_id or uuid.uuid4().hex[:12]

    # ------------------------------------------------------------------ entry
    def run(
        self, item: Mapping[str, Any], condition: str, *, sample: int = 0
    ) -> Trace:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}; known: {CONDITIONS}")
        trace = Trace(
            run_id=self.run_id,
            case_id=str(item.get("id") or item.get("case_id") or ""),
            dataset=self.task.name,
            condition=condition,
            model_key=self.model.name,
            framework_hash=self.config.framework_hash(),
            sample=sample,
            started_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        )
        started = time.perf_counter()
        try:
            if condition in {"M0", "M1", "M2"}:
                self._run_single_shot(item, condition, trace, sample)
            else:
                self._run_agent(item, condition, trace, sample)
        except Exception as exc:  # one bad case must not lose the other 299
            trace.error = f"{type(exc).__name__}: {exc}"
        trace.wall_ms = (time.perf_counter() - started) * 1000
        return trace

    # ------------------------------------------------------- M0 / M1 / M2
    def _run_single_shot(
        self, item: Mapping[str, Any], condition: str, trace: Trace, sample: int
    ) -> None:
        system = self.task.system_prompt(condition)
        user = self.task.user_message(item)
        if condition == "M2":
            context = self.task.static_context(item)
            rendered = self.task.render_context(context)
            trace.static_context_chars = len(rendered)
            user = f"{user}\n\n{rendered}"
        messages = [Message("system", system), Message("user", user)]
        outcome = self._ask(messages, trace, phase="answer", sample=sample)
        self._finalise(outcome, trace, item)

    # ------------------------------------------------------------ M3 / M4
    def _run_agent(
        self, item: Mapping[str, Any], condition: str, trace: Trace, sample: int
    ) -> None:
        ctx = ToolContext(self.kg, self.retriever, self.task.domain, self.config.tool_budget)
        system = self.task.system_prompt(condition) + "\n" + self._tool_protocol()
        messages: List[Message] = [
            Message("system", system),
            Message("user", self.task.user_message(item)),
        ]

        format_failures = 0
        result: Optional[Dict[str, Any]] = None

        for turn in range(self.config.max_agent_turns):
            outcome = self._ask(messages, trace, phase="reason", sample=sample)
            if outcome.value is None:
                format_failures += 1
                if format_failures > self.config.max_format_retries:
                    trace.error = "model did not emit parseable JSON"
                    break
                messages.append(Message("assistant", outcome.raw[:2000]))
                messages.append(
                    Message(
                        "user",
                        "你的上一轮输出无法解析为 JSON。请只输出一个 JSON 对象，"
                        '形如 {"action": "tool", ...} 或 {"action": "answer", ...}，'
                        "不要输出任何其他文字。",
                    )
                )
                continue

            action = str(outcome.value.get("action") or "").lower()
            trace.llm_steps[-1].action = action or "unparsed"

            if action == "answer" or "result" in outcome.value:
                payload = outcome.value.get("result")
                result = payload if isinstance(payload, Mapping) else outcome.value
                break

            if action == "tool":
                messages.append(Message("assistant", json.dumps(outcome.value, ensure_ascii=False)))
                tool_result = self._call_tool(ctx, outcome.value, trace)
                messages.append(
                    Message(
                        "user",
                        "TOOL_RESULT:\n"
                        + tool_result.to_model_text(
                            max_chars=self.config.tool_budget.max_result_chars
                        )
                        + self._budget_note(ctx),
                    )
                )
                if ctx.remaining() == 0:
                    messages.append(
                        Message(
                            "user",
                            "工具预算已用尽。请立即输出最终答案，"
                            '格式为 {"action": "answer", "result": {...}}。',
                        )
                    )
                continue

            # a well-formed JSON object that is neither a tool call nor an
            # answer: treat it as the answer rather than burning a turn
            result = dict(outcome.value)
            break

        if result is None and trace.error is None:
            trace.error = "agent exhausted its turn budget without answering"

        if result is not None and condition == "M4":
            result = self._verify_and_revise(item, result, ctx, trace, sample)

        if result is not None:
            trace.final = self.task.normalise_result(result, item)
            trace.final_raw = json.dumps(result, ensure_ascii=False)

    # ----------------------------------------------------------- verification
    def _verify_and_revise(
        self,
        item: Mapping[str, Any],
        result: Mapping[str, Any],
        ctx: ToolContext,
        trace: Trace,
        sample: int,
    ) -> Dict[str, Any]:
        """M4: deterministic re-check of the answer, then one revision turn."""
        report = self._verification_report(item, result, ctx, trace)
        if report is None:
            return dict(result)
        messages = [
            Message("system", load_prompt("verifier")),
            Message(
                "user",
                "【你的结论】\n"
                + json.dumps(result, ensure_ascii=False)
                + "\n\n【校验结果】\n"
                + json.dumps(report, ensure_ascii=False),
            ),
        ]
        outcome = self._ask(messages, trace, phase="verify_revise", sample=sample)
        if outcome.value is None:
            revised = dict(result)
            revised["revision"] = "unchanged"
            revised["revision_reason"] = "校验后模型输出无法解析，保留原结论。"
            return revised
        payload = outcome.value
        # the model has been answering in the {"action","result"} envelope all
        # along and often keeps using it here; unwrap rather than nest
        if isinstance(payload.get("result"), Mapping):
            payload = payload["result"]
        revised = dict(payload)
        revised.setdefault("revision", "unchanged")
        revised["verification_report"] = report
        # a revision turn must not silently drop required answer fields
        for key in self.task.answer_fields():
            if key not in revised and key in result:
                revised[key] = result[key]
        return revised

    def _verification_report(
        self,
        item: Mapping[str, Any],
        result: Mapping[str, Any],
        ctx: ToolContext,
        trace: Trace,
    ) -> Optional[Dict[str, Any]]:
        args = self.task.verify_arguments(result, item)
        if args is not None:
            # SDT: graph consistency of the claimed syndrome
            verify_ctx = ToolContext(
                self.kg, self.retriever, self.task.domain, ToolBudget(max_calls=2, max_calls_per_tool=2)
            )
            tool_result = self.config.registry.call("verify_tcm_decision", verify_ctx, args)
            self._record_tool(tool_result, verify_ctx, trace)
            return tool_result.to_dict()
        return self._coverage_audit(result, trace)

    def _coverage_audit(
        self, result: Mapping[str, Any], trace: Trace
    ) -> Optional[Dict[str, Any]]:
        """PA: did the answer lean on the graph where the graph has nothing?

        Deterministic and cheap, and it targets the specific failure mode this
        knowledge graph invites -- a model reading ``not_covered`` as
        ``no problem found`` and asserting graph support for a dose or a
        compatibility rule the graph never encoded.
        """
        reasoning = " ".join(
            coerce_str(result.get(key)) for key in ("reasoning", "option_analysis")
        )
        if not reasoning:
            return None
        flagged: List[Dict[str, Any]] = []
        for topic, keywords in UNCOVERED_TOPICS.items():
            if any(word in reasoning for word in keywords):
                flagged.append(
                    {
                        "topic": topic,
                        "status": "not_covered_by_graph",
                        "note": (
                            "该结论涉及图谱不收录的知识；若你在推理中引用了图谱作为依据，"
                            "请改为依据自身知识并说明图谱无据。"
                        ),
                    }
                )
        coverage_counts = trace.coverage_counts()
        return {
            "check": "evidence_coverage_audit",
            "tool_coverage_observed": coverage_counts,
            "uncovered_topics_referenced": flagged,
            "n_tool_calls": trace.n_tool_calls,
            "verdict": "review_recommended" if flagged else "no_coverage_conflict",
        }

    # ---------------------------------------------------------------- helpers
    def _ask(
        self, messages: Sequence[Message], trace: Trace, *, phase: str, sample: int
    ) -> ParseOutcome:
        completion = self.model.generate(messages, self.config.decode, sample=sample)
        outcome = extract_json_object(completion.text)
        trace.llm_steps.append(
            LLMStep(
                index=len(trace.llm_steps),
                phase=phase,
                prompt_chars=sum(len(m.content) for m in messages),
                completion=completion.to_dict(),
                parse_strategy=outcome.strategy,
            )
        )
        if completion.error and trace.error is None:
            trace.error = completion.error
        return outcome

    def _call_tool(
        self, ctx: ToolContext, action: Mapping[str, Any], trace: Trace
    ) -> ToolResult:
        name = str(action.get("tool") or "")
        raw_args = action.get("arguments")
        arguments = dict(raw_args) if isinstance(raw_args, Mapping) else {}
        result = self.config.registry.call(name, ctx, arguments)
        self._record_tool(result, ctx, trace)
        return result

    @staticmethod
    def _record_tool(result: ToolResult, ctx: ToolContext, trace: Trace) -> None:
        record = ctx.calls[-1]
        trace.tool_steps.append(
            ToolStep(
                index=len(trace.tool_steps),
                tool=record.tool,
                arguments=dict(record.arguments),
                coverage=record.coverage,
                ok=record.ok,
                latency_ms=record.latency_ms,
                n_results=record.n_results,
                error=record.error,
                result_chars=len(result.to_model_text()),
            )
        )

    def _tool_protocol(self) -> str:
        specs = self.config.registry.specs_for(self.task.domain)
        tool_list = "\n".join(spec.prompt_block() for spec in specs)
        return (
            load_prompt("tool_protocol")
            .replace("{max_calls}", str(self.config.tool_budget.max_calls))
            .replace("{max_calls_per_tool}", str(self.config.tool_budget.max_calls_per_tool))
            .replace("{tool_list}", tool_list)
        )

    @staticmethod
    def _budget_note(ctx: ToolContext) -> str:
        return f"\n[剩余工具预算: {ctx.remaining()} 次]"

    def _finalise(
        self, outcome: ParseOutcome, trace: Trace, item: Mapping[str, Any]
    ) -> None:
        trace.parse_strategy = outcome.strategy
        trace.final_raw = outcome.raw
        if outcome.value is None:
            if trace.error is None:
                trace.error = "model did not emit parseable JSON"
            return
        payload = outcome.value
        if "result" in payload and isinstance(payload["result"], Mapping):
            payload = payload["result"]
        trace.final = self.task.normalise_result(payload, item)
