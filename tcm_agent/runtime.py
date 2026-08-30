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
from tcm_tools.base import (
    REGISTRY,
    ToolBudget,
    ToolContext,
    ToolPhase,
    ToolRegistry,
    ToolResult,
)

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

#: Arms that share one agent phase and differ only in what follows it.
BRANCHABLE: Tuple[str, ...] = ("M3", "M3C", "M4")
_BRANCHABLE = BRANCHABLE


def _stratum(report: Mapping[str, Any]) -> str:
    """How much the verification pass could actually say about this item."""
    if report.get("verdict") == "not_applicable":
        return "not_applicable"
    if int(report.get("n_checks") or 0) > 0:
        return "deterministic"
    return "audit_only"


def _check_compute_parity(branches: Mapping[str, Trace]) -> None:
    """Confirm each control spent exactly what the arm it matches spent.

    The controls exist so that "M4 beat M3C" cannot be read as "M4 got another
    turn". That argument only holds if the turn counts are equal *per case*,
    not on average: a control that skips its revision turn on the 12% of items
    no checker adjudicates still averages close, while the contrast on exactly
    those items is a second-turn effect wearing a verification label.

    A mismatch is recorded on both traces rather than raised. Losing the whole
    case would be worse than scoring it and saying it is unusable, and the
    report drops flagged cases from the paired contrast.
    """
    for control, arm in COMPUTE_MATCHED.items():
        left, right = branches.get(control), branches.get(arm)
        if left is None or right is None:
            continue
        if left.n_llm_calls == right.n_llm_calls:
            continue
        note = (
            f"compute parity broken: {control} used {left.n_llm_calls} LLM calls, "
            f"{arm} used {right.n_llm_calls}"
        )
        left.parity_error = note
        right.parity_error = note


def _copy_trace(source: Trace, condition: str) -> Trace:
    """A trace carrying the shared prefix, ready for its own branch."""
    branch = Trace(
        run_id=source.run_id,
        case_id=source.case_id,
        dataset=source.dataset,
        condition=condition,
        model_key=source.model_key,
        framework_hash=source.framework_hash,
        run_signature=source.run_signature,
        sample=source.sample,
        started_at=source.started_at,
    )
    branch.llm_steps = list(source.llm_steps)
    branch.tool_steps = list(source.tool_steps)
    branch.static_context_chars = source.static_context_chars
    branch.parse_strategy = source.parse_strategy
    branch.error = source.error
    branch.wall_ms = source.wall_ms
    branch.branch_group = source.branch_group
    return branch

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
    #: The full run signature (framework + inputs + model + code), filled in by
    #: the runner and stamped on every trace.  Deliberately *not* part of
    #: ``framework_hash``: the framework hash certifies that two arms shared a
    #: scaffold and must stay equal across models, while the signature also
    #: pins the model and the code and so differs between them.
    run_signature: str = ""

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
            "run_signature": self.run_signature,
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
            run_signature=self.config.run_signature,
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
        """M2C: M2's knowledge, M3's thinking budget, no adaptive tools.

        This control exists so that ``M2C→M3`` isolates *one* thing: whether
        letting the model choose its own queries beats a fixed retrieval, at
        equal compute. It therefore has to match M2 on knowledge and M3 on
        turns.

        An earlier version built it from the M1 prompt with no graph evidence
        at all, which meant ``M2C→M3`` moved four variables together -- KG or
        no KG, static or adaptive retrieval, tools or no tools, one turn or
        many -- and could not isolate agency. It now receives **the same static
        KG context block M2 receives**, and is given the turn budget M3 gets,
        with no tool access. Each turn it is prompted onward with no new
        information, so no further evidence enters while the compute matches.
        """
        system = (
            self.task.system_prompt("M2")
            + "\n"
            + load_prompt("control_iterative").replace(
                "{max_calls}", str(self.config.tool_budget.max_calls)
            )
        )
        context = self.task.static_context(item)
        rendered = self.task.render_context(context)
        trace.static_context_chars = len(rendered)
        messages: List[Message] = [
            Message("system", system),
            Message("user", f"{self.task.user_message(item)}\n\n{rendered}"),
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
    def run_branch_group(
        self,
        item: Mapping[str, Any],
        conditions: Sequence[str],
        *,
        sample: int = 0,
    ) -> Dict[str, Trace]:
        """Run the agent phase **once** and fork it into the verification arms.

        M3, M3C and M4 share an identical agent phase -- same prompt, same
        tools, same budget -- and differ only in what happens afterwards:
        nothing, a sham revision, or a real verification plus revision. Running
        them as three independent invocations meant each got its own
        trajectory, so ``M3C→M4`` carried whatever the model happened to do
        differently in the agent phase on top of the verification difference.
        Provider seed support is inconsistent, so that is not something a seed
        can be relied on to remove.

        Here the shared prefix is computed once and copied into each branch, so
        the *only* difference between M3C and M4 is the verification evidence
        in the revision turn. That makes M3C→M4 the cleanest contrast in the
        study rather than the muddiest.
        """
        wanted = [c for c in conditions if c in _BRANCHABLE]
        if not wanted:
            return {}

        shared = Trace(
            run_id=self.run_id,
            case_id=str(item.get("id") or item.get("case_id") or ""),
            dataset=self.task.name,
            condition="M3",
            model_key=self.model.name,
            framework_hash=self.config.framework_hash(),
            run_signature=self.config.run_signature,
            sample=sample,
            started_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        )
        shared.branch_group = f"{shared.case_id}#{sample}"
        started = time.perf_counter()
        try:
            result, ctx = self._agent_phase(item, "M3", shared, sample)
        except Exception as exc:  # one bad case must not lose the branch group
            shared.error = f"{type(exc).__name__}: {exc}"
            result, ctx = None, ToolContext(
                self.kg,
                self.retriever,
                self.task.domain,
                self.config.tool_budget,
                phase=ToolPhase.AGENT,
            )
        shared.wall_ms = (time.perf_counter() - started) * 1000

        out: Dict[str, Trace] = {}
        for condition in wanted:
            branch = _copy_trace(shared, condition)
            branch_result = dict(result) if result else None
            if branch_result is not None and condition in {"M4", "M3C"}:
                branch_started = time.perf_counter()
                branch_result = self._verify_and_revise(
                    item, branch_result, ctx, branch, sample, sham=(condition == "M3C")
                )
                branch.wall_ms += (time.perf_counter() - branch_started) * 1000
            if branch_result is not None:
                branch.final = self.task.normalise_result(branch_result, item)
                branch.final_raw = json.dumps(branch_result, ensure_ascii=False)
            out[condition] = branch
        _check_compute_parity(out)
        return out

    def _run_agent(
        self, item: Mapping[str, Any], condition: str, trace: Trace, sample: int
    ) -> None:
        result, ctx = self._agent_phase(item, condition, trace, sample)
        if result is not None and condition in {"M4", "M3C"}:
            result = self._verify_and_revise(
                item, result, ctx, trace, sample, sham=(condition == "M3C")
            )
        if result is not None:
            trace.final = self.task.normalise_result(result, item)
            trace.final_raw = json.dumps(result, ensure_ascii=False)

    def _agent_phase(
        self, item: Mapping[str, Any], condition: str, trace: Trace, sample: int
    ) -> Tuple[Optional[Dict[str, Any]], ToolContext]:
        """The tool-using reasoning loop, up to and including the first answer."""
        # Agent phase: the registry refuses a verification-phase tool here even
        # if the model names one, so M3 is unverified by construction rather
        # than by the tool list happening not to mention the checkers.
        ctx = ToolContext(
            self.kg,
            self.retriever,
            self.task.domain,
            self.config.tool_budget,
            phase=ToolPhase.AGENT,
        )
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
        return (dict(result) if result else None), ctx

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
                # No checker adjudicates this item and the answer carried no
                # prose to audit.  Returning here would let M4 skip the
                # revision turn that M3C always takes, so on exactly these
                # cases M4 would have one LLM call fewer and an answer that
                # was never reconsidered.  Any M4-minus-M3C difference over
                # them would then measure *a second turn*, not verification.
                # Take the turn, and say honestly that there was nothing to
                # check -- which is also the more informative stimulus: a
                # model told "no automatic check applies" is being asked to
                # rely on its own knowledge, and whether it then wobbles is
                # itself a finding.
                report = self._not_applicable_report(trace)
        if not sham:
            trace.verification_stratum = _stratum(report)

        # Both branches re-receive the full original case, question and option
        # list. Without them a revision turn is asked to change its answer
        # while unable to see what the alternatives say: told only that option
        # C is unsupported, a model has no basis to choose A over B or D. That
        # is not a verification effect, it is an amnesia effect -- and it
        # differs between the arms only by accident. Re-supplying the item in
        # both branches keeps the verification report the single difference.
        original = self.task.user_message(item)
        if sham:
            messages = [
                Message("system", load_prompt("control_sham_revision")),
                Message(
                    "user",
                    original
                    + "\n\n【你的结论】\n"
                    + json.dumps(result, ensure_ascii=False),
                ),
            ]
        else:
            messages = [
                Message("system", load_prompt("verifier")),
                Message(
                    "user",
                    original
                    + "\n\n【你的结论】\n"
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
            phase=ToolPhase.VERIFICATION,
        )
        checks: List[Dict[str, Any]] = []
        for plan in plans:
            arguments = {k: v for k, v in plan.items() if not k.startswith("_")}
            tool_name = str(plan.get("_tool") or "verify_tcm_decision")
            tool_result = self.config.registry.call(tool_name, verify_ctx, arguments)
            # the harness chose this call, not the model: tag it so tool-selection
            # metrics do not credit M4 for checkers its verifier ran for it
            self._record_tool(tool_result, verify_ctx, trace, phase="verification")
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

    def _not_applicable_report(self, trace: Trace) -> Dict[str, Any]:
        """The verification report for an item no deterministic check covers.

        Emitted rather than skipping the turn, so every M4 case costs exactly
        one revision turn and pairs one-to-one with its M3C twin.  It is
        marked ``not_applicable`` so the scorer can report the M4 gain
        separately over adjudicated and unadjudicated items -- if the gain
        lives entirely in the latter, it is a second-turn effect wearing a
        verification label.
        """
        return {
            "check": "deterministic_verification",
            "n_checks": 0,
            "checks": [],
            "verdict": "not_applicable",
            "note": (
                "本题没有可自动核验的确定性规则，图谱亦无相关记录。"
                "请依据自身医学知识复核上述结论；若无需修改，请原样保留。"
            ),
            "tool_coverage_observed": trace.coverage_counts(phase="agent"),
        }

    def _coverage_audit(
        self, result: Mapping[str, Any], trace: Trace
    ) -> Optional[Dict[str, Any]]:
        """PA: did the answer lean on the graph where the graph has nothing?

        Deterministic and cheap, and it targets the specific failure mode this
        knowledge graph invites -- a model reading ``not_covered`` as
        ``no problem found`` and asserting graph support for a dose or a
        compatibility rule the graph never encoded.
        """
        # ``.strip()`` is load-bearing: joining two absent fields with a space
        # yields " ", which is truthy, so without it the guard never fired and
        # every answer got an audit report -- including answers with no prose
        # to over-claim in.  That in turn meant the ``not_applicable`` stratum
        # was unreachable and the parity fix below it was never exercised.
        reasoning = " ".join(
            coerce_str(result.get(key)) for key in ("reasoning", "option_analysis")
        ).strip()
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
        # what the *agent* saw, not what the verifier has just looked up: the
        # audit asks whether the model over-claimed given its own evidence
        coverage_counts = trace.coverage_counts(phase="agent")
        return {
            "check": "evidence_coverage_audit",
            "tool_coverage_observed": coverage_counts,
            "uncovered_topics_referenced": flagged,
            "n_tool_calls": trace.n_agent_tool_calls,
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
    def _record_tool(
        result: ToolResult, ctx: ToolContext, trace: Trace, *, phase: str = "agent"
    ) -> None:
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
                phase=phase,
            )
        )

    def _tool_protocol(self) -> str:
        """The tool list an agent sees.

        Verification-phase tools are withheld -- which now means the five
        deterministic PA checkers as well as ``verify_tcm_decision``. Hiding
        only the latter left ``check_dose`` and friends callable by the agent,
        so M3 was an optionally-self-checking arm and M3→M4 contrasted
        *optional* with *mandatory* verification rather than absent with
        present. Those checkers are exactly what the M4 pass runs, so they
        belong to it alone.
        """
        specs = self.config.registry.specs_for(self.task.domain, phase=ToolPhase.AGENT)
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
