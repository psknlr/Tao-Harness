"""Content fingerprints of the code and data a run depended on.

A framework hash over tool *descriptions* proves that every arm saw the same
tool contract. It does not prove they ran the same tool *implementation*:
rewriting the body of ``check_dose`` while leaving its ``ToolSpec`` untouched
changes every PA verification result and leaves the hash identical. The same
holds for scorers -- a changed scoring rule silently reinterprets recorded
traces.

These helpers hash the source of the modules that decide outcomes, so a run
manifest records not just *what was configured* but *what code ran*.

Hashing source text rather than bytecode keeps the value stable across Python
versions and independent of comments' absence, and keeps it readable in a diff.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

#: Modules whose implementation determines a run's outputs.
TOOL_MODULES: Sequence[str] = (
    "tcm_tools.base",
    "tcm_tools.clinical",
    "tcm_tools.medication",
    "tcm_tools.safety",
    "tcm_tools.checkers",
    "tcm_tools.verify",
    "tcm_tools.pathway",
    "tcm_tools._common",
)

SCORER_MODULES: Sequence[str] = (
    "tcm_eval.scorers",
    "tcm_eval.official_sdt",
    "tcm_eval.metrics",
    "tcm_eval.stats",
)

RETRIEVAL_MODULES: Sequence[str] = (
    "tcm_kg.index",
    "tcm_kg.store",
    "tcm_kg.normalize",
    "tcm_kg.schema",
)

RUNTIME_MODULES: Sequence[str] = (
    "tcm_agent.runtime",
    "tcm_agent.tasks",
    "tcm_agent.parsing",
)


def _module_source(name: str) -> str:
    try:
        module = __import__(name, fromlist=["__name__"])
        return inspect.getsource(module)
    except Exception:
        return f"<unavailable:{name}>"


def source_hash(module_names: Iterable[str]) -> str:
    """SHA-256 over the concatenated source of the named modules."""
    digest = hashlib.sha256()
    for name in sorted(module_names):
        digest.update(name.encode("utf-8"))
        digest.update(_module_source(name).encode("utf-8"))
    return digest.hexdigest()


def case_set_hash(case_ids: Sequence[str]) -> str:
    """SHA-256 over the exact case list a run covered.

    Guards a real failure mode: run with ``limit: 500``, later drop to
    ``limit: 200``, and without clearing ``runs/`` the scorer happily scores
    all 500 recorded traces while the config -- and any reader -- believes the
    run was 200 cases. Freezing the case list makes the mismatch detectable
    instead of invisible.
    """
    digest = hashlib.sha256()
    for case_id in case_ids:  # order-sensitive on purpose: the split's order
        digest.update(str(case_id).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def code_fingerprints() -> Dict[str, str]:
    """All implementation hashes, short form, for the manifest."""
    return {
        "tools_impl_sha256": source_hash(TOOL_MODULES),
        "scorers_impl_sha256": source_hash(SCORER_MODULES),
        "retrieval_impl_sha256": source_hash(RETRIEVAL_MODULES),
        "runtime_impl_sha256": source_hash(RUNTIME_MODULES),
    }


def compare_fingerprints(
    frozen: Mapping[str, Any], current: Mapping[str, Any]
) -> List[str]:
    """Differences between a frozen manifest and the code running now.

    Returned as warnings rather than raised: re-scoring recorded traces with a
    fixed scorer is a legitimate and expected operation -- that is the point of
    separating generation from scoring. What is not legitimate is doing it
    without noticing, so the drift is reported every time.
    """
    drift: List[str] = []
    for key, frozen_value in frozen.items():
        if not key.endswith("_sha256"):
            continue
        current_value = current.get(key)
        if current_value and frozen_value and current_value != frozen_value:
            drift.append(
                f"{key}: manifest {str(frozen_value)[:12]} != current {str(current_value)[:12]}"
            )
    return drift
