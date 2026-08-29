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

#: ``M2C`` and ``M3C`` are compute-matched controls, not ablations of the
#: knowledge graph. Without them, M2→M3 confounds "the agent used the graph"
#: with "the model got more turns to think", and M3→M4 confounds "verification
#: helped" with "one more revision helped". Each control spends the same
#: number of model calls as the arm it is matched to, and receives no graph
#: evidence in those extra calls.
CONDITIONS = ("M0", "M1", "M2", "M3", "M4", "M2C", "M3C")

#: control -> the arm whose test-time compute it matches
COMPUTE_MATCHED: Mapping[str, str] = {"M2C": "M3", "M3C": "M4"}

#: Topics the knowledge graph does not encode.  Used by the PA verification
#: pass to catch a model citing graph support it cannot have had.
#: Ordered worst-first: one contradiction outweighs any number of passes.
_VERDICT_PRIORITY = (
    "contradicted",
    "contradicted_by_graph",
    "partially_supported",
    "not_in_graph",
    "not_covered",
    "supported",
)

UNCOVERED_TOPICS: Mapping[str, Tuple[str, ...]] = {
    "dose": ("剂量", "用量", "超量", "克", "g", "用法用量"),
    "incompatibility": ("十八反", "十九畏", "配伍禁忌", "相反", "相畏"),
    "pregnancy": ("妊娠", "孕妇", "孕期"),
    "drug_interaction": ("相互作用", "中西药联用"),
}


def _summarise_verdicts(checks: Sequence[Mapping[str, Any]]) -> str:
    """Worst verdict across every deterministic check.

    Worst-first because verification is a safety gate: an answer with one
    contradicted component and three supported ones needs review, and
    averaging would hide exactly the case the pass exists to catch.
    """
    seen: set = set()
    for check in checks:
        data = check.get("data")
        if isinstance(data, Mapping):
            if data.get("overall"):
                seen.add(str(data["overall"]))
            for finding in data.get("findings") or []:
                if not isinstance(finding, Mapping):
                    continue
                if finding.get("claim_verdict"):
                    seen.add(str(finding["claim_verdict"]))
                elif finding.get("attested_requirements") or finding.get("restrictions"):
                    # the checker found grounded facts but was given no claim to
                    # adjudicate; that is evidence, not a silent no-op
                    seen.add("supported")
            if data.get("alias_duplicates") or data.get("formula_overlap"):
                seen.add("contradicted")
        if check.get("coverage") == "not_covered":
            seen.add("not_covered")
    for verdict in _VERDICT_PRIORITY:
        if verdict in seen:
            return verdict
    return "no_verdict"


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
    #: What the run is measuring. Two arms are only comparable if these match
    #: too: an SDT run and a PA run share every budget and prompt file, so
    #: without these the hash collided and certified as identical two runs that
    #: read different sub-graphs and answered different questions.
    task: str = ""
    domain: str = ""
    #: Content fingerprints of the inputs, filled in by the runner.
    kg_hash: str = ""
    dataset_hash: str = ""

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
                "task": self.task,
                "domain": self.domain,
                "kg": self.kg_hash,
                "dataset": self.dataset_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> Dict[str, Any]:
        return {
            "framework_hash": self.framework_hash(),
            "task": self.task,
            "domain": self.domain,
            "kg_hash": self.kg_hash[:16],
            "dataset_hash": self.dataset_hash[:16],
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
            elif condition == "M2C":
                self._run_iterative_control(item, trace, sample)
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

    # ------------------------------------------------- M2C: compute control
    def _run_iterative_control(
        self, item: Mapping[str, Any], trace: Trace, sample: int
    ) -> None:
        """M2C: as many model turns as M3 gets, but no knowledge graph.

        The point is to hold test-time compute constant while removing the only
        thing M3 adds -- graph access. If M2C matches M3, the agentic gain was
        thinking time rather than knowledge, and the honest contrast to report
        is M2C→M3 rather than M2→M3.

        The model is told it may think for several turns; each turn it is
        prompted onward with no new information, so no evidence enters. Turns
        are capped at the tool budget M3 would have had, so the two arms cost
        the same number of calls.
        """
        system = (
            self.task.system_prompt("M1")
            + "\n"
            + load_prompt("control_iterative").replace(
                "{max_calls}", str(self.config.tool_budget.max_calls)
            )
        )
        messages: List[Message] = [
            Message("system", system),
            Message("user", self.task.user_message(item)),
        ]
        budget = self.config.tool_budget.max_calls
        result: Optional[Dict[str, Any]] = None
        format_failures = 0

        for turn in range(self.config.max_agent_turns):
            outcome = self._ask(messages, trace, phase="reason", sample=sample)
            if outcome.value is None:
                format_failures += 1
                if format_failures > self.config.max_format_retries:
                    trace.error = "model did not emit parseable JSON"
                    break
                messages.append(Message("assistant", outcome.raw[:2000]))
                messages.append(
                    Message("user", '请只输出一个 JSON 对象，形如 {"action": "think", ...} 或 {"action": "answer", ...}。')
                )
                continue

            action = str(outcome.value.get("action") or "").lower()
            trace.llm_steps[-1].action = action or "unparsed"
            if action == "answer" or "result" in outcome.value:
                payload = outcome.value.get("result")
                result = payload if isinstance(payload, Mapping) else outcome.value
                break
            if action == "think" and budget > 0:
                budget -= 1
                messages.append(Message("assistant", json.dumps(outcome.value, ensure_ascii=False)))
                messages.append(
                    Message(
                        "user",
                        f"已记录。请继续推理或给出最终答案。[剩余思考轮次: {budget}]"
                        + ("\n思考轮次已用尽，请立即给出最终答案。" if budget == 0 else ""),
                    )
                )
                continue
            result = dict(outcome.value)
            break

        if result is None and trace.error is None:
            trace.error = "control arm exhausted its turn budget without answering"
        if result is not None:
            trace.final = self.task.normalise_result(result, item)
            trace.final_raw = json.dumps(result, ensure_ascii=False)

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

        if result is not None and condition in {"M4", "M3C"}:
            result = self._verify_and_revise(
                item, result, ctx, trace, sample, sham=(condition == "M3C")
            )

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
        *,
        sham: bool = False,
    ) -> Dict[str, Any]:
        """M4: deterministic re-check then one revision turn.

        ``sham=True`` is the M3C control: the same extra revision turn, with no
        verification evidence in it. M4 minus M3C is the effect of the
        verification *content*; M4 minus M3 also contains the effect of simply
        being asked to look again.
        """
        if sham:
            report = None
        else:
            report = self._verification_report(item, result, ctx, trace)
            if report is None:
                return dict(result)

        if sham:
            messages = [
                Message("system", load_prompt("control_sham_revision")),
                Message("user", "【你的结论】\n" + json.dumps(result, ensure_ascii=False)),
            ]
        else:
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
        if report is not None:
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
        """Re-run deterministic checks over the answer the model just gave.

        The task decides *what* to verify; this runs it. A task may return a
        list of argument sets -- one per selected syndrome for SDT, one per
        applicable rule checker for PA -- and every one is executed, because a
        verification pass that checks only the first element of a multi-select
        answer verifies almost nothing.

        The coverage audit remains as the fallback for items no deterministic
        checker can adjudicate (over half the PA set), where the honest
        verification is "the graph cannot speak to this; did you claim it
        could?".
        """
        plans = self.task.verify_arguments(result, item)
        if plans is None:
            return self._coverage_audit(result, trace)
        if isinstance(plans, Mapping):
            plans = [plans]

        # a budget of its own, so verification cannot be starved by an agent
        # that spent everything during reasoning
        verify_ctx = ToolContext(
            self.kg,
            self.retriever,
            self.task.domain,
            ToolBudget(max_calls=max(4, 2 * len(plans)), max_calls_per_tool=max(4, len(plans))),
        )
        checks: List[Dict[str, Any]] = []
        for plan in plans:
            arguments = {k: v for k, v in plan.items() if not k.startswith("_")}
            tool_name = str(plan.get("_tool") or "verify_tcm_decision")
            tool_result = self.config.registry.call(tool_name, verify_ctx, arguments)
            self._record_tool(tool_result, verify_ctx, trace)
            entry = tool_result.to_dict()
            if plan.get("_option"):
                entry["option"] = plan["_option"]
            checks.append(entry)

        report: Dict[str, Any] = {
            "check": "deterministic_verification",
            "n_checks": len(checks),
            "checks": checks,
            "verdict": _summarise_verdicts(checks),
        }
        # the coverage audit is complementary, not alternative: a deterministic
        # check can pass while the prose still over-claims graph support
        audit = self._coverage_audit(result, trace)
        if audit:
            report["coverage_audit"] = audit
        return report

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
