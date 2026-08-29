"""MCP-style tool surface over the TCM knowledge graph.

Importing this package registers all 16 tools:

* 8 knowledge tools (``search_tcm_entities`` .. ``verify_tcm_decision``)
* 5 deterministic prescription-audit checkers
* 3 clinical-pathway tools (stage detail, deterministic transition
  evaluation, treatment planning), available only in the pathway domain

Nothing here exposes a query language; every model sees the same descriptions.
"""

from . import checkers, clinical, medication, pathway, safety, verify  # noqa: F401  (registration)
from .base import (
    REGISTRY,
    BudgetExceeded,
    Coverage,
    ToolBudget,
    ToolCallRecord,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from .verify import Verdict

__all__ = [
    "REGISTRY",
    "BudgetExceeded",
    "Coverage",
    "ToolBudget",
    "ToolCallRecord",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "Verdict",
]
