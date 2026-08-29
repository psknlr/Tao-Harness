"""MCP-style tool surface over the TCM knowledge graph.

Importing this package registers all 13 tools:

* 8 knowledge tools (``search_tcm_entities`` .. ``verify_tcm_decision``)
* 5 deterministic prescription-audit checkers

Nothing here exposes a query language; every model sees the same descriptions.
"""

from . import checkers, clinical, medication, safety, verify  # noqa: F401  (registration)
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
