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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

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


#: Manifest keys that describe *what the run was*.  A resumed run may add
#: models or extend a case list; it may not change any of these and still be
#: the same experiment.
IMMUTABLE_MANIFEST_KEYS: Sequence[str] = (
    "task",
    "domain",
    "framework_hash",
    "design_signature",
    "kg_content_sha256",
    "dataset_sha256",
    # The gold file, resolved rather than configured: replacing it changes the
    # answers a run is graded against, which is a different experiment even
    # though the input JSON is byte-identical.
    "dataset_results_sha256",
    "case_set_sha256",
)


def run_signature(
    *,
    framework_hash: str,
    kg_hash: str,
    dataset_hash: str,
    model_fingerprint: str,
    case_set: str = "",
    code: Optional[Mapping[str, str]] = None,
) -> str:
    """One value identifying the apparatus that produced a trace.

    ``framework_hash`` is deliberately blind to the model and the code: it has
    to stay equal across arms, which is what makes "every model saw the same
    scaffold" checkable.  That blindness is also a hole -- traces generated
    before and after a tool rewrite, or by two different model snapshots, carry
    the same framework hash and pool together silently.

    The signature closes it by covering the model spec, the implementation
    hashes and the frozen case set as well.  It is stamped on every trace, so
    the check does not depend on a manifest file still being present or still
    describing the traces beside it.  Condition is *not* included: conditions
    are the independent variable and must share a signature to be comparable.
    """
    payload = {
        "framework": framework_hash,
        "kg": kg_hash,
        "dataset": dataset_hash,
        "model": model_fingerprint,
        "case_set": case_set,
        "code": dict(sorted((code or code_fingerprints()).items())),
    }
    import json as _json

    return hashlib.sha256(
        _json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def design_signature(
    *,
    run_signature: str,
    conditions: Sequence[str],
    samples: int,
    limit: Optional[int] = None,
    stratify: Optional[str] = None,
) -> str:
    """What the experiment *is*, on top of the apparatus that runs it.

    ``run_signature`` deliberately omits the condition, so the arms of one
    experiment share it and can be pooled. That leaves a hole one level up: the
    condition list, the sample count and the sampling rule are not in any
    fingerprint, so narrowing a config does not change what a recorded trace
    matches.

    Run seven arms, then edit the config down to ``[M0, M1]``: the M2..M4
    traces still carry a matching signature, resume keeps them, and the
    manifest says the experiment has two arms while the trace file holds
    seven. Drop ``samples`` from 3 to 1 and the extra samples survive as
    ordinary items -- scoring no longer runs consensus over them, so they are
    counted individually and one case gets three times the weight of its
    neighbours.

    Neither is detectable from the apparatus signature, because neither
    changes the apparatus. This is the fingerprint for the *design*: it selects
    the output directory's identity, gates resume, and is recorded in the
    manifest so a report can state which experiment produced it.
    """
    payload = {
        "apparatus": run_signature,
        "conditions": sorted(str(c) for c in conditions),
        "samples": int(samples),
        "limit": limit,
        "stratify": stratify or "",
    }
    import json as _json

    return hashlib.sha256(
        _json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def manifest_conflicts(
    frozen: Mapping[str, Any], current: Mapping[str, Any]
) -> List[str]:
    """Immutable manifest fields that a re-run would be changing.

    Non-empty means the output directory already holds a *different*
    experiment, and writing over its manifest would destroy the only record of
    what the traces beside it were generated under.
    """
    out: List[str] = []
    for key in IMMUTABLE_MANIFEST_KEYS:
        before, after = frozen.get(key), current.get(key)
        if before and after and before != after:
            out.append(f"{key}: {str(before)[:12]} -> {str(after)[:12]}")
    return out


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
