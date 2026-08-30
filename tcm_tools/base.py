"""Tool contract for the TCM-KG agent.

Three properties are load-bearing for the experiment:

1. **No raw query language.**  Exposing ``run_cypher`` would turn "how well
   does this model write Cypher" into a confound that varies by model.  Every
   model instead sees the same eight high-level tools with byte-identical
   descriptions.
2. **Domain enforcement at the tool boundary.**  A tool physically cannot
   return a node type its domain forbids, so the SDT agent cannot recover a
   syndrome by looking at which formula treats it.
3. **Explicit coverage.**  When the graph does not encode what a tool was asked
   for, the tool returns ``Coverage.NOT_COVERED`` with a reason.  It never
   returns an empty result that a model could read as "no contraindication
   exists".  Silence and absence are different answers, and conflating them is
   how a prescription-audit system produces a dangerous false negative.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from tcm_kg.index import KGRetriever
from tcm_kg.schema import Domain, DomainPolicy, DomainViolation, policy_for
from tcm_kg.store import KGStore


class Coverage(str, Enum):
    """Whether the graph could actually answer the question that was asked."""

    #: The graph encodes this and the answer below is grounded in it.
    SUPPORTED = "supported"
    #: The graph encodes part of it; the gap is named in ``ToolResult.caveats``.
    PARTIAL = "partial"
    #: The graph does not encode this class of fact at all. The model must fall
    #: back on its own knowledge and should say so.
    NOT_COVERED = "not_covered"
    #: The graph encodes this class of fact, and for this query there is none.
    #: Distinct from NOT_COVERED: here absence really is evidence of absence.
    EMPTY = "empty"


@dataclass
class ToolResult:
    """Uniform tool return value."""

    tool: str
    ok: bool
    coverage: Coverage
    data: Any = None
    evidence: List[Mapping[str, Any]] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    error: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "tool": self.tool,
            "ok": self.ok,
            "coverage": self.coverage.value,
        }
        if self.data is not None:
            payload["data"] = self.data
        if self.evidence:
            payload["evidence"] = self.evidence
        if self.caveats:
            payload["caveats"] = self.caveats
        if self.error:
            payload["error"] = self.error
        return payload

    def to_model_text(self, *, max_chars: int = 4000) -> str:
        """Render for the model's context window, deterministically truncated."""
        text = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=None)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 20] + '..."<truncated>"}'


class ToolPhase(str, Enum):
    """Which stage of a run may issue a tool.

    ``verification=True`` alone was too coarse. It hid ``verify_tcm_decision``
    from the agent but left the five deterministic PA checkers visible, and
    those are exactly the tools the M4 verification pass uses. An M3 agent
    could therefore call ``check_dose`` itself, which makes M3 an
    *optionally-self-checking* arm and turns M3→M4 into "optional versus
    mandatory verification" rather than "absent versus present" -- a weaker
    claim than the tables imply.
    """

    #: the agent may call it; the verification pass may not
    AGENT = "agent"
    #: only the post-hoc verification pass may call it
    VERIFICATION = "verification"
    #: available to both (retrieval that either stage legitimately needs)
    BOTH = "both"


@dataclass(frozen=True)
class ToolSpec:
    """Frozen declaration of one tool.

    ``description`` and ``parameters`` are part of the framework contract and
    feed the framework hash; changing either invalidates cross-model
    comparability and the runner will refuse to merge runs across the change.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    domains: Tuple[Domain, ...]
    #: deterministic rule-engine tools do not consult the graph for ranking and
    #: are reported separately in the trace metrics
    deterministic: bool = False
    #: verification tools may read verification-only node types
    verification: bool = False
    #: which run stage may issue this tool
    phase: ToolPhase = ToolPhase.AGENT

    def openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }

    def prompt_block(self) -> str:
        """Plain-text rendering for models without native function calling."""
        props = self.parameters.get("properties", {})
        required = set(self.parameters.get("required", []))
        lines = [f"- {self.name}: {self.description}", "  参数:"]
        for key, spec in props.items():
            flag = "必填" if key in required else "可选"
            desc = spec.get("description", "")
            lines.append(f"    - {key} ({spec.get('type', 'string')}, {flag}): {desc}")
        return "\n".join(lines)


@dataclass
class ToolBudget:
    """Hard caps applied identically to every model."""

    max_calls: int = 12
    max_calls_per_tool: int = 4
    max_result_chars: int = 4000

    def fingerprint(self) -> str:
        return f"{self.max_calls}/{self.max_calls_per_tool}/{self.max_result_chars}"


class BudgetExceeded(RuntimeError):
    """Raised when an agent exhausts its identical, frozen tool budget."""


@dataclass
class ToolCallRecord:
    """One entry of the trace -- the unit the trace metrics are computed over."""

    index: int
    tool: str
    arguments: Mapping[str, Any]
    coverage: str
    ok: bool
    latency_ms: float
    n_results: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "coverage": self.coverage,
            "ok": self.ok,
            "latency_ms": round(self.latency_ms, 2),
            "n_results": self.n_results,
            **({"error": self.error} if self.error else {}),
        }


class ToolContext:
    """Everything a tool may touch, plus the budget it must respect."""

    def __init__(
        self,
        kg: KGStore,
        retriever: KGRetriever,
        domain: Domain | str,
        budget: Optional[ToolBudget] = None,
        phase: ToolPhase = ToolPhase.AGENT,
    ):
        self.kg = kg
        self.retriever = retriever
        self.domain: Domain = Domain(domain) if isinstance(domain, str) else domain
        self.policy: DomainPolicy = policy_for(self.domain)
        self.budget = budget or ToolBudget()
        #: Which side of the agent/verifier boundary this context sits on.
        #: Enforced in ``ToolRegistry.call``: withholding a tool from the
        #: prompt is not isolation, it is obscurity -- a model that names
        #: ``check_dose`` anyway would otherwise execute it, and M3 would be
        #: an optionally self-checking arm after all.
        self.phase: ToolPhase = phase
        self.calls: List[ToolCallRecord] = []
        self._per_tool: Dict[str, int] = {}

    # -------------------------------------------------------------- budgeting
    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def remaining(self) -> int:
        return max(0, self.budget.max_calls - self.n_calls)

    def check_budget(self, tool_name: str) -> Optional[str]:
        if self.n_calls >= self.budget.max_calls:
            return f"tool budget exhausted ({self.budget.max_calls} calls)"
        if self._per_tool.get(tool_name, 0) >= self.budget.max_calls_per_tool:
            return (
                f"per-tool budget exhausted for {tool_name} "
                f"({self.budget.max_calls_per_tool} calls)"
            )
        return None

    def record(self, record: ToolCallRecord) -> None:
        self.calls.append(record)
        self._per_tool[record.tool] = self._per_tool.get(record.tool, 0) + 1

    # ------------------------------------------------------------- guardrails
    def assert_visible(self, node_type: str, *, verification: bool = False) -> None:
        if not self.policy.may_return(node_type, verification=verification):
            raise DomainViolation(
                f"node type {node_type!r} is not readable in domain "
                f"{self.domain.value!r}: {self.policy.rationale}"
            )

    def visible_types(self, requested: Optional[Iterable[str]] = None, *, verification: bool = False) -> List[str]:
        allowed = self.policy.visible_types(verification=verification)
        if requested is None:
            return sorted(allowed)
        return sorted(t for t in requested if t in allowed)


ToolFn = Callable[[ToolContext, Mapping[str, Any]], ToolResult]


class ToolRegistry:
    """Plugin registry: tools register themselves and are then frozen.

    Borrowed from the DeepSeek-harness "everything is a plugin" arrangement --
    models, datasets, scorers and tools all resolve through registries, so a new
    benchmark or a new provider is an addition rather than an edit.
    """

    def __init__(self) -> None:
        self._specs: Dict[str, ToolSpec] = {}
        self._fns: Dict[str, ToolFn] = {}

    def register(self, spec: ToolSpec) -> Callable[[ToolFn], ToolFn]:
        def decorator(fn: ToolFn) -> ToolFn:
            if spec.name in self._specs:
                raise ValueError(f"tool {spec.name!r} already registered")
            self._specs[spec.name] = spec
            self._fns[spec.name] = fn
            return fn

        return decorator

    def spec(self, name: str) -> ToolSpec:
        return self._specs[name]

    def specs_for(
        self, domain: Domain | str, *, phase: Optional[ToolPhase] = None
    ) -> List[ToolSpec]:
        domain = Domain(domain) if isinstance(domain, str) else domain
        specs = [s for s in self._specs.values() if domain in s.domains]
        if phase is not None:
            specs = [s for s in specs if s.phase in (phase, ToolPhase.BOTH)]
        return specs

    def names_for(self, domain: Domain | str) -> List[str]:
        return [s.name for s in self.specs_for(domain)]

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def call(
        self, name: str, ctx: ToolContext, arguments: Mapping[str, Any]
    ) -> ToolResult:
        """Invoke a tool with budget, domain and error handling applied."""
        started = time.perf_counter()
        index = ctx.n_calls

        def _fail(message: str, coverage: Coverage = Coverage.NOT_COVERED) -> ToolResult:
            result = ToolResult(
                tool=name,
                ok=False,
                coverage=coverage,
                error=message,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            ctx.record(
                ToolCallRecord(
                    index=index,
                    tool=name,
                    arguments=dict(arguments),
                    coverage=result.coverage.value,
                    ok=False,
                    latency_ms=result.latency_ms,
                    n_results=0,
                    error=message,
                )
            )
            return result

        if name not in self._specs:
            return _fail(f"unknown tool {name!r}; available: {sorted(self._specs)}")
        spec = self._specs[name]
        if ctx.domain not in spec.domains:
            return _fail(
                f"tool {name!r} is not available in domain {ctx.domain.value!r}"
            )
        # Phase is a real boundary, checked here rather than only omitted from
        # the prompt. The agent never sees the PA checkers or the verifier in
        # its tool list, but a model can produce a name it was never shown --
        # from its own priors, or from an earlier turn of a long context -- and
        # before this check it was executed. The M3C->M4 contrast claims the
        # verification evidence is the only difference between the arms; that
        # only holds if an M3 agent cannot obtain it by asking.
        if spec.phase is not ToolPhase.BOTH and spec.phase is not ctx.phase:
            return _fail(
                f"tool {name!r} belongs to the {spec.phase.value} phase and cannot "
                f"be called from the {ctx.phase.value} phase"
            )
        breach = ctx.check_budget(name)
        if breach:
            return _fail(breach)

        missing = [
            key
            for key in spec.parameters.get("required", [])
            if arguments.get(key) in (None, "", [], {})
        ]
        if missing:
            return _fail(f"missing required argument(s): {missing}")

        try:
            result = self._fns[name](ctx, arguments)
        except DomainViolation as exc:
            return _fail(str(exc))
        except Exception as exc:  # a buggy tool must not abort a 300-case run
            return _fail(f"{type(exc).__name__}: {exc}")

        result.latency_ms = (time.perf_counter() - started) * 1000
        ctx.record(
            ToolCallRecord(
                index=index,
                tool=name,
                arguments=dict(arguments),
                coverage=result.coverage.value,
                ok=result.ok,
                latency_ms=result.latency_ms,
                n_results=_count(result.data),
            )
        )
        return result

    def fingerprint(self, domain: Optional[Domain | str] = None) -> str:
        """Hash of the tool contract, folded into the framework hash."""
        specs = self.specs_for(domain) if domain is not None else list(self._specs.values())
        payload = json.dumps(
            [
                {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.parameters,
                    "domains": sorted(d.value for d in s.domains),
                    "deterministic": s.deterministic,
                    "phase": s.phase.value,
                }
                for s in sorted(specs, key=lambda s: s.name)
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _count(data: Any) -> int:
    if data is None:
        return 0
    if isinstance(data, list):
        return len(data)
    if isinstance(data, Mapping):
        for key in ("results", "candidates", "items", "entries", "findings"):
            value = data.get(key)
            if isinstance(value, list):
                return len(value)
        return 1
    return 1


#: The single global registry.  Importing :mod:`tcm_tools` populates it.
REGISTRY = ToolRegistry()
