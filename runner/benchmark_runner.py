#!/usr/bin/env python3
"""TCM-KG Agent benchmark runner.

Subcommands
-----------
``inspect``   report the KG ontology, the tool contract and the dataset schema
``coverage``  audit which PA rule families the graph can actually ground
``run``       generate traces for a config (resumable; caches every request)
``score``     score recorded traces -- no generation, so re-scoring is free
``judge``     run the LLM judge over recorded SDT traces
``report``    build the Markdown report from scored traces
``compare``   paired A/B between two runs, win/lose/tie with signed deltas
``submit``    write an official TCMEval-SDT ``@``-separated submission file

The split between ``run`` and ``score`` is the point: a scorer fix never
re-bills a single token, and a run recorded once replays in CI forever.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tcm_tools  # noqa: F401  (registers the tool surface)
from tcm_agent import AgentRuntime, Trace, build_task, read_traces, write_traces
from tcm_agent.runtime import BRANCHABLE, GROUPED
from tcm_eval import (
    PA_METRICS,
    PA_PRIMARY,
    CP_METRICS,
    CP_PRIMARY,
    SDT_METRICS,
    SDT_PRIMARY,
    ScoredItem,
    aggregate_trace_metrics,
    build_report,
    inspect_dataset,
    load_dataset,
    score_cp,
    score_pa,
    score_sdt,
    trace_metrics,
)
from tcm_eval.judge import SDTJudge, aggregate_judge
from tcm_eval.metrics import pathogenesis_probe_rate, tool_selection_accuracy
from tcm_eval.official_sdt import write_submission
from tcm_eval.provenance import (
    case_set_hash,
    code_fingerprints,
    compare_fingerprints,
    design_signature,
    manifest_conflicts,
    run_signature,
)
from tcm_eval.stats import mcnemar, paired_bootstrap
from tcm_kg import load_kg
from tcm_kg.index import KGRetriever
from tcm_agent.parsing import extract_json_object
from tcm_eval.report import _md_table
from tcm_models.base import DecodeParams, Message
from tcm_models.registry import build_client
from tcm_tools.base import REGISTRY

from .config import ExperimentConfig, judge_model_key, load_experiment, load_models


# --------------------------------------------------------------------------- #
# shared setup
# --------------------------------------------------------------------------- #


def _load_graph(kg_path: Optional[str], config: Optional[ExperimentConfig] = None):
    kg = load_kg(kg_path)
    params = config.framework.retrieval if config else None
    retriever = KGRetriever(kg, params)
    cache_dir = REPO_ROOT / "runs" / ".index"
    # Key the index cache on the graph's *contents*, not its node count: an
    # edit to a thousand relations leaves the count unchanged and would
    # otherwise silently reuse a stale index.
    cache_file = cache_dir / f"{retriever.cache_key(kg.content_hash())}.pkl"
    if not retriever.load(cache_file):
        retriever.warm()
        retriever.save(cache_file)
    return kg, retriever


_KG_HASH_CACHE: Dict[str, str] = {}


def _kg_hash_for(config: ExperimentConfig, args: argparse.Namespace) -> str:
    """Content hash of the graph this config would use, without warming an index.

    ``score`` needs it to check the traces against the frozen manifest, but for
    PA and CP it does not otherwise load the graph, and building the retrieval
    index just to compute a hash would make every score run minutes slower.
    """
    key = str(getattr(args, "kg", None) or "")
    if key not in _KG_HASH_CACHE:
        try:
            _KG_HASH_CACHE[key] = load_kg(getattr(args, "kg", None)).content_hash()
        except Exception:
            _KG_HASH_CACHE[key] = ""
    return _KG_HASH_CACHE[key]


def _file_hash(path: Path) -> str:
    """SHA-256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def write_manifest(
    config: ExperimentConfig,
    kg,
    specs: Mapping[str, Any],
    *,
    n_items: int,
    extra: Optional[Mapping[str, Any]] = None,
    allow_conflict: bool = False,
    gold_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Record everything needed to reproduce or audit this run.

    A framework hash proves two arms shared a scaffold; it does not tell a
    reader a year later which graph, which dataset file, which code revision or
    which model snapshots produced a number. The manifest does, and it is
    written before generation starts so an interrupted run still leaves a
    record of what it was doing.
    """
    manifest: Dict[str, Any] = {
        "experiment": config.name,
        "task": config.task,
        "domain": config.domain.value,
        "framework_hash": config.framework.framework_hash(),
        "kg_content_sha256": kg.content_hash(),
        "dataset_path": str(config.dataset_path),
        "dataset_sha256": _file_hash(config.dataset_path),
        "n_items": n_items,
        "conditions": config.conditions,
        "samples": config.samples,
        "git_commit": _git_commit(),
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "models": {
            key: {
                "provider": specs[key].provider,
                "model_id": specs[key].model_id,
                "base_url": specs[key].base_url,
                "extra_body": dict(specs[key].extra_body),
                "fingerprint_sha256": specs[key].fingerprint(),
            }
            for key in config.models
            if key in specs
        },
        "framework": config.framework.describe(),
        **code_fingerprints(),
    }
    # The gold file the loader *resolved*, which for SDT it discovers rather
    # than being told. Hashing only a configured path left the discovered one
    # unfrozen: swap Results/*.txt after generation and `score` would re-grade
    # recorded predictions against different answers with every other
    # fingerprint intact.
    resolved_gold = gold_path or config.dataset_results_path
    if resolved_gold:
        manifest["dataset_results_path"] = str(resolved_gold)
        manifest["dataset_results_sha256"] = _file_hash(Path(resolved_gold))
        manifest["dataset_results_resolved_by"] = (
            "config" if config.dataset_results_path else "loader discovery"
        )
    if extra:
        manifest.update(extra)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / f"manifest.{config.task}.json"

    # A manifest is the only record of what the traces beside it were produced
    # under. Overwriting it on every `run` meant a resumed run with an edited
    # config silently replaced that record: the traces stayed, the description
    # of how they were made became the new config, and nothing was left to say
    # they disagreed. So a conflicting manifest is preserved, never replaced.
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
        if isinstance(previous, Mapping):
            conflicts = manifest_conflicts(previous, manifest)
            if not conflicts:
                # Same experiment, fewer models: `run --models gpt` used to
                # rewrite the model map to just gpt while the other four
                # traces.*.jsonl stayed on disk and kept being scored, so the
                # manifest described a one-model experiment beside a five-model
                # report. Per-model maps merge; everything else is the current
                # run's, which the conflict check has already agreed matches.
                for key in ("models", "run_signatures", "design_signatures"):
                    merged = dict(previous.get(key) or {})
                    merged.update(manifest.get(key) or {})
                    if merged:
                        manifest[key] = merged
                added = sorted(
                    set(manifest.get("models") or {}) - set(previous.get("models") or {})
                )
                kept = sorted(
                    set(previous.get("models") or {}) - set(manifest.get("models") or {})
                )
                if added or kept:
                    print(
                        f"manifest: {len(manifest.get('models') or {})} model(s) "
                        f"recorded" + (f", added {added}" if added else ""),
                        file=sys.stderr,
                    )
            if conflicts:
                if not allow_conflict:
                    raise SystemExit(
                        f"{path} describes a different experiment:\n  "
                        + "\n  ".join(conflicts)
                        + "\n\nThe traces in this directory were generated under those "
                        "settings. Point --config at a fresh output_dir, or pass "
                        "--new-run to archive the old manifest and start a new one "
                        "here (existing traces will not resume)."
                    )
                archive = (
                    config.output_dir
                    / f"manifest.{config.task}.{str(previous.get('run_signature') or previous.get('created_at') or 'previous')}.json"
                )
                archive.write_text(
                    json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"archived the previous manifest to {archive.name}", file=sys.stderr)

    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _trace_path(config: ExperimentConfig, model: str) -> Path:
    return config.output_dir / f"traces.{config.task}.{model}.jsonl"


def _cache_path(config: ExperimentConfig, model: str) -> Path:
    return config.output_dir / "cache" / f"{model}.jsonl"


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def cmd_run(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    if args.models:
        config.models = args.models
    if args.conditions:
        config.conditions = [c.upper() for c in args.conditions]
    if args.limit is not None:
        config.dataset_limit = args.limit

    dataset = load_dataset(config.dataset_path, config.dataset_kind, **config.loader_kwargs())
    print(dataset.mapping.report(), file=sys.stderr)
    if dataset.mapping.missing:
        print(
            f"WARNING: unmapped dataset fields {dataset.mapping.missing}; "
            f"scoring for those fields will be skipped.",
            file=sys.stderr,
        )
    items = dataset.subset(
        config.dataset_limit, stratify=config.dataset_stratify
    ).items
    if not items:
        print("no items to run", file=sys.stderr)
        return 1

    kg, retriever = _load_graph(args.kg, config)
    specs = load_models(args.models_config)
    # content fingerprints participate in the framework hash, so they must be
    # set before it is computed or recorded anywhere
    config.framework.kg_hash = kg.content_hash()
    config.framework.dataset_hash = _file_hash(config.dataset_path)
    framework_hash = config.framework.framework_hash()

    case_ids = [str(item["id"]) for item in items]
    case_set = case_set_hash(case_ids)
    code = code_fingerprints()

    def _signature(model_key: str) -> str:
        spec = specs.get(model_key)
        return run_signature(
            framework_hash=framework_hash,
            kg_hash=config.framework.kg_hash,
            dataset_hash=config.framework.dataset_hash,
            model_fingerprint=spec.fingerprint() if spec else "",
            case_set=case_set,
            code=code,
        )

    def _design(model_key: str) -> str:
        return design_signature(
            run_signature=_signature(model_key),
            conditions=config.conditions,
            samples=config.samples,
            limit=config.dataset_limit,
            stratify=config.dataset_stratify,
        )

    manifest = write_manifest(
        config,
        kg,
        specs,
        gold_path=dataset.gold_path,
        n_items=len(items),
        extra={
            "case_ids": case_ids,
            "case_set_sha256": case_set,
            "n_unique_case_ids": len(set(case_ids)),
            # non-zero means the source file shipped duplicate keys and the
            # loader had to impose uniqueness; a reader should know
            "n_renamed_case_ids": dataset.n_renamed_ids,
            "run_signatures": {k: _signature(k) for k in config.models if k in specs},
            # Two levels: the apparatus, which the arms share, and the design,
            # which they do not -- a narrowed condition list or a changed
            # sample count is a different experiment in the same apparatus.
            "design_signatures": {k: _design(k) for k in config.models if k in specs},
            "design_signature": design_signature(
                run_signature=framework_hash,
                conditions=config.conditions,
                samples=config.samples,
                limit=config.dataset_limit,
                stratify=config.dataset_stratify,
            ),
        },
        allow_conflict=bool(getattr(args, "new_run", False)),
    )
    print(
        f"framework_hash={framework_hash}  kg={manifest['kg_content_sha256'][:12]}  "
        f"dataset={manifest['dataset_sha256'][:12]}  n_items={len(items)}",
        file=sys.stderr,
    )
    (config.output_dir / "run_config.json").write_text(
        json.dumps(config.describe(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_written = 0
    for model_key in config.models:
        if model_key not in specs:
            print(f"skipping unknown model {model_key!r}", file=sys.stderr)
            continue
        trace_file = _trace_path(config, model_key)
        signature = _signature(model_key)
        config.framework.run_signature = signature

        # Resume on the *signature*, not the framework hash. The framework hash
        # is designed to stay equal across models and says nothing about the
        # code, so resuming on it silently kept traces from a different model
        # snapshot or from before a tool rewrite -- exactly the traces a resume
        # must regenerate. Traces predating the signature carry "" and are
        # never resumed: they cannot prove what produced them.
        design = _design(model_key)
        config.framework.design_signature = design

        recorded = read_traces(trace_file) if not args.overwrite else []
        existing = {
            (t.case_id, t.condition, t.sample): t
            for t in recorded
            # Both levels. The apparatus signature alone let a trace from a
            # condition this run no longer declares, or a sample index this run
            # no longer produces, survive into the output -- the manifest then
            # described one experiment and the trace file held another.
            if t.run_signature == signature
            and t.design_signature == design
            and t.condition in config.conditions
            and t.sample < config.samples
        }
        stale = len(recorded) - len(existing)
        if existing:
            print(f"{model_key}: resuming, {len(existing)} traces already recorded", file=sys.stderr)
        if stale:
            print(
                f"{model_key}: discarding {stale} trace(s) from a different run "
                f"signature; they will be regenerated",
                file=sys.stderr,
            )

        client = build_client(
            specs[model_key],
            cache_path=_cache_path(config, model_key),
            replay=args.replay,
            script=args.echo_script,
        )
        task = build_task(config.task, kg, retriever, config.framework.context_budget)

        # M3/M3C/M4 share one agent phase and are generated together, so the
        # only difference between them is what happens after the first answer.
        # M2C joins them because it has to spend the number of model calls M3
        # spent *on this case*, which is only known once M3 has run.
        group_conditions = [c for c in config.conditions if c in GROUPED]
        solo_conditions = [c for c in config.conditions if c not in GROUPED]

        jobs: List[Tuple[Mapping[str, Any], Tuple[str, ...], int]] = []
        regenerated = 0
        for item in items:
            for sample in range(config.samples):
                for condition in solo_conditions:
                    if (str(item["id"]), condition, sample) not in existing:
                        jobs.append((item, (condition,), sample))
                if not group_conditions:
                    continue
                present = [
                    c
                    for c in group_conditions
                    if (str(item["id"]), c, sample) in existing
                ]
                if len(present) == len(group_conditions):
                    continue
                # Regenerate the whole group, not just the missing arm.
                # Filling in only M4 gave it a fresh agent prefix while the
                # surviving M3C kept the old one, so the pair no longer shared
                # a trajectory -- and M3C->M4, whose entire claim is that the
                # verification report is the only difference, silently became
                # a comparison of two independent runs.
                for condition in present:
                    existing.pop((str(item["id"]), condition, sample), None)
                    regenerated += 1
                jobs.append((item, tuple(group_conditions), sample))
        if regenerated:
            print(
                f"{model_key}: discarding {regenerated} arm(s) of partially "
                f"recorded groups so each group keeps one shared trajectory",
                file=sys.stderr,
            )
        n_traces = sum(len(group) for _i, group, _s in jobs)
        print(
            f"{model_key}: {n_traces} traces to generate "
            f"({len(jobs)} invocations; {len(group_conditions)} arms generated as a group)",
            file=sys.stderr,
        )

        traces: List[Trace] = list(existing.values())
        started = time.time()

        def _one(job: Tuple[Mapping[str, Any], Tuple[str, ...], int]) -> List[Trace]:
            item, group, sample = job
            runtime = AgentRuntime(
                kg, retriever, task, client, config.framework, run_id=config.name
            )
            if len(group) == 1 and group[0] not in GROUPED:
                return [runtime.run(item, group[0], sample=sample)]
            return list(runtime.run_branch_group(item, group, sample=sample).values())

        if config.concurrency > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
                futures = {pool.submit(_one, job): job for job in jobs}
                for done, future in enumerate(as_completed(futures), 1):
                    traces.extend(future.result())
                    if done % 20 == 0 or done == len(jobs):
                        print(
                            f"  {model_key}: {done}/{len(jobs)} "
                            f"({time.time() - started:.0f}s)",
                            file=sys.stderr,
                        )
                        write_traces(trace_file, traces)
        else:
            for done, job in enumerate(jobs, 1):
                traces.extend(_one(job))
                if done % 20 == 0 or done == len(jobs):
                    print(f"  {model_key}: {done}/{len(jobs)}", file=sys.stderr)
                    write_traces(trace_file, traces)

        traces.sort(key=lambda t: (t.case_id, t.condition, t.sample))
        write_traces(trace_file, traces)
        total_written += len(traces)
        errors = sum(1 for t in traces if t.error)
        print(
            f"{model_key}: wrote {len(traces)} traces to {trace_file} ({errors} with errors)",
            file=sys.stderr,
        )

    print(f"done: {total_written} traces", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #


def _consensus_traces(traces: Sequence[Trace]) -> List[Trace]:
    """Collapse the samples of each case into one self-consistent trace.

    Only runs when a case actually has several samples. The combined trace
    keeps the union of tool steps so behavioural metrics still reflect the work
    done, and is marked ``sample=-1`` so it cannot be confused with a real one.
    """
    from tcm_eval.scorers import consensus_prediction

    groups: Dict[Tuple[str, str, str], List[Trace]] = {}
    for trace in traces:
        groups.setdefault((trace.model_key, trace.condition, trace.case_id), []).append(trace)

    out: List[Trace] = []
    for (_model, _condition, _case), bucket in groups.items():
        if len(bucket) == 1:
            out.append(bucket[0])
            continue
        bucket.sort(key=lambda t: t.sample)
        merged = consensus_prediction([t.final for t in bucket])
        head = bucket[0]
        combined = Trace(
            run_id=head.run_id,
            case_id=head.case_id,
            dataset=head.dataset,
            condition=head.condition,
            model_key=head.model_key,
            framework_hash=head.framework_hash,
            # Provenance must survive the merge. Dropping it made a combined
            # trace unattributable -- the branch group the M3C/M4 pairing needs,
            # the parity flag that removes an unusable pair, and the two
            # signatures that prove what produced it all vanished, and the
            # analysis silently treated a consensus trace as clean.
            run_signature=head.run_signature,
            design_signature=head.design_signature,
            sample=-1,
            started_at=head.started_at,
        )
        combined.branch_group = head.branch_group
        combined.verification_stratum = head.verification_stratum
        # Any sample whose compute did not match invalidates the merged case:
        # the consensus answer was voted on by an arm that spent differently.
        broken = [t.parity_error for t in bucket if t.parity_error]
        combined.parity_error = broken[0] if broken else ""
        for trace in bucket:
            combined.llm_steps.extend(trace.llm_steps)
            combined.tool_steps.extend(trace.tool_steps)
            combined.wall_ms += trace.wall_ms
            combined.static_context_chars = max(
                combined.static_context_chars, trace.static_context_chars
            )
        combined.final = merged
        combined.parse_strategy = head.parse_strategy
        combined.error = None if merged else "no sample produced an answer"
        out.append(combined)
    return out


def _score_traces(
    config: ExperimentConfig,
    traces: Sequence[Trace],
    gold: Mapping[str, Mapping[str, Any]],
    kg,
    pricing: Optional[Mapping[str, Sequence[float]]] = None,
    contamination: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> List[ScoredItem]:
    scored: List[ScoredItem] = []
    # The audit, already validated against this run's identity by cmd_score.
    # Every scored item carries its stratum, so the clean-subset sensitivity
    # analysis needs no separate bookkeeping at report time.
    contamination_cases = dict(contamination or {})
    if config.samples > 1:
        traces = _consensus_traces(traces)
    for trace in traces:
        reference = gold.get(trace.case_id)
        if reference is None:
            continue
        if not reference.get("_has_gold"):
            continue  # generated but unscoreable (a split with blank answers)
        if config.task == "sdt":
            metrics = score_sdt(trace.final, reference, kg=kg)
            probe = pathogenesis_probe_rate(trace, reference.get("pathogenesis_options"))
            if probe is not None:
                metrics["pathogenesis_probe_rate"] = probe
        elif config.task == "cp":
            metrics = score_cp(trace.final, reference)
            metrics["disease"] = reference.get("disease")  # cluster label
        else:
            metrics = score_pa(trace.final, reference)
            metrics["rule_id"] = reference.get("rule_id")
            metrics["category"] = reference.get("category")
            accuracy = tool_selection_accuracy(trace, reference.get("rule_id"))
            if accuracy is not None:
                metrics["tool_selection_accuracy"] = accuracy
        stratum = contamination_cases.get(trace.case_id)
        if stratum:
            metrics["contamination_stratum"] = stratum["stratum"]
            metrics["contamination_score"] = max(
                float(stratum.get("ngram_jaccard") or 0.0),
                float(stratum.get("containment") or 0.0),
            )
        scored.append(
            ScoredItem(
                case_id=trace.case_id,
                dataset=trace.dataset,
                condition=trace.condition,
                model_key=trace.model_key,
                sample=trace.sample,
                metrics=metrics,
                trace_metrics={
                    **trace_metrics(
                        trace, cost_per_mtok=(pricing or {}).get(trace.model_key, (0.0, 0.0))
                    ),
                    **(
                        {"pathogenesis_probe_rate": metrics["pathogenesis_probe_rate"]}
                        if "pathogenesis_probe_rate" in metrics
                        else {}
                    ),
                },
            )
        )
    return scored


def _gold_hash(dataset: Any, config: ExperimentConfig) -> str:
    """SHA of the gold file the loader *actually* resolved.

    The SDT config sets no ``results_path``, so the loader discovers
    ``Results/*.txt`` on its own -- and only an explicitly configured path was
    ever hashed. Replacing that file after generation would let `score`
    re-grade recorded predictions against different answers with the dataset
    hash unchanged, and every fail-closed check would pass.
    """
    resolved = getattr(dataset, "gold_path", None) or config.dataset_results_path
    return _file_hash(Path(resolved)) if resolved else ""


def _load_contamination(config: ExperimentConfig) -> Dict[str, Dict[str, Any]]:
    """The audit for this experiment, if one has been run.

    Absent is not an error here -- ``score`` should work while developing --
    but the report says loudly when the stratification is missing, because a
    KG result with no contamination audit behind it has not ruled out the one
    explanation a paired design cannot rule out by itself.
    """
    path = config.output_dir / f"contamination.{config.task}.json"
    if not path.exists():
        return {}
    try:
        from tcm_eval.contamination import load_report

        return load_report(path)
    except (OSError, json.JSONDecodeError, KeyError):
        return {}


def _check_contamination_identity(
    audit: Mapping[str, Any],
    dataset: Any,
    config: ExperimentConfig,
    kg_hash: str,
    case_ids: Sequence[str],
) -> List[str]:
    """Is this audit about the run being scored, or an older one?"""
    from tcm_eval.contamination import audit_identity, identity_conflicts

    frozen = audit.get("identity") or {}
    if not frozen:
        return ["the audit predates identity stamping; re-run `contaminate`"]
    current = audit_identity(
        kg_hash=kg_hash,
        dataset_hash=_file_hash(config.dataset_path),
        gold_hash=_gold_hash(dataset, config),
        case_set_hash=case_set_hash([str(c) for c in case_ids]),
    )
    return identity_conflicts(frozen, current)


def _load_manifest(config: ExperimentConfig) -> Optional[Dict[str, Any]]:
    path = config.output_dir / f"manifest.{config.task}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def cmd_score(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    dataset = load_dataset(config.dataset_path, config.dataset_kind, **config.loader_kwargs())
    gold = {str(item["id"]): item for item in dataset.items}

    # The frozen manifest -- not the config as it reads today -- defines what
    # the run actually was. Scoring against the current config would let a
    # later edit (a smaller `limit`, a repointed model, a changed dataset)
    # silently reinterpret traces produced under different conditions.
    allow_drift = bool(getattr(args, "allow_drift", False))
    #: Every reason this score run is not what the manifest describes.  Carried
    #: into the scored manifest so a number produced under --allow-drift can
    #: never be mistaken later for a clean one.
    drift_notes: List[str] = []

    def _refuse_or_note(reason: str, detail: str) -> Optional[int]:
        """Fail closed, unless the operator has explicitly taken the risk.

        Warnings were the wrong shape for these: they scroll past, they do not
        reach the report, and a scored file produced despite one is
        indistinguishable afterwards from a clean one.  Scoring the wrong
        traces silently is worse than not scoring.
        """
        drift_notes.append(f"{reason}: {detail}")
        if allow_drift:
            print(f"--allow-drift: {reason} — {detail}", file=sys.stderr)
            return None
        print(
            f"REFUSING to score: {reason}.\n  {detail}\n"
            f"  Re-run generation, or pass --allow-drift to score anyway "
            f"(the drift is then recorded in scored_manifest and the report).",
            file=sys.stderr,
        )
        return 2

    manifest = _load_manifest(config)
    if manifest is None:
        # A scored file with no manifest has no experimental identity: nothing
        # records which graph, dataset, code or models produced it, and the
        # config it is scored against is whatever the file says today. Fine
        # while developing, not fine for a number that goes in a paper.
        code = _refuse_or_note(
            "no frozen manifest in the output directory",
            f"{config.output_dir} has no manifest.{config.task}.json, so there is "
            f"no record of what produced these traces. Run `run` first.",
        )
        if code is not None:
            return code
    else:
        drift = compare_fingerprints(manifest, code_fingerprints())
        if drift:
            # Scoring code changing is the expected case -- that is the point of
            # separating generation from scoring -- so it is a note. Tool,
            # retrieval or runtime code changing means the traces could not be
            # reproduced by this checkout, and that fails closed.
            scoring_only = all(k.startswith("scorers_impl") for k in (d.split(":")[0] for d in drift))
            if scoring_only:
                print(
                    "NOTE: scoring code has changed since generation — "
                    + "; ".join(drift)
                    + ". Re-scoring recorded traces with fixed scorers is expected.",
                    file=sys.stderr,
                )
                drift_notes.append("scoring code changed since generation: " + "; ".join(drift))
            else:
                code = _refuse_or_note(
                    "code that produced the traces has changed",
                    "; ".join(drift),
                )
                if code is not None:
                    return code
        current_inputs = {
            "kg_content_sha256": _kg_hash_for(config, args),
            "dataset_sha256": _file_hash(config.dataset_path),
            # The gold answers themselves. An unchanged input JSON with a
            # swapped Results file re-grades the same predictions against
            # different answers, and every other check passes.
            "dataset_results_sha256": _gold_hash(dataset, config),
        }
        for key, value in current_inputs.items():
            frozen_value = manifest.get(key)
            if value and frozen_value and value != frozen_value:
                code = _refuse_or_note(
                    f"{key} differs from the frozen manifest",
                    f"manifest {str(frozen_value)[:12]} != current {str(value)[:12]}",
                )
                if code is not None:
                    return code
        frozen_cases = manifest.get("case_ids")
        if frozen_cases:
            expected = case_set_hash([str(c) for c in frozen_cases])
            if expected != manifest.get("case_set_sha256"):
                # The manifest is the immutable record of what this run was.
                # If it contradicts itself, it is not a record of anything, and
                # scoring against it certifies a run that cannot be described.
                code = _refuse_or_note(
                    "the manifest contradicts its own case-set hash",
                    f"case_ids hash to {expected[:12]} but case_set_sha256 says "
                    f"{str(manifest.get('case_set_sha256'))[:12]}; the file has been "
                    f"edited or truncated.",
                )
                if code is not None:
                    return code
            gold = {cid: gold[cid] for cid in map(str, frozen_cases) if cid in gold}
            print(
                f"scoring the {len(gold)} cases frozen in the manifest "
                f"(case_set {str(manifest.get('case_set_sha256'))[:12]})",
                file=sys.stderr,
            )
    # A stale contamination audit is worse than none. None is visible -- the
    # report says so loudly. A stale one silently stratifies this run by a
    # previous graph's contamination, so the clean-subset analysis, whose whole
    # job is to show the gain is not retrieval, gets computed against a graph
    # that no longer exists.
    audit = _load_contamination(config)
    contamination_cases: Dict[str, Any] = {}
    if audit:
        stale = _check_contamination_identity(
            audit, dataset, config, _kg_hash_for(config, args), sorted(gold)
        )
        if stale:
            code = _refuse_or_note(
                "the contamination audit does not describe this run",
                "; ".join(stale) + ". Re-run `benchmark_runner contaminate`.",
            )
            if code is not None:
                return code
        contamination_cases = audit.get("cases") or {}

    kg, _ = _load_graph(args.kg, config) if config.task == "sdt" else (None, None)
    if kg is not None:
        config.framework.kg_hash = kg.content_hash()
    config.framework.dataset_hash = _file_hash(config.dataset_path)

    # Real pricing, so cost_usd is a number rather than a structural zero.
    specs = load_models(args.models_config)
    pricing = {
        key: (spec.input_usd_per_mtok, spec.output_usd_per_mtok)
        for key, spec in specs.items()
    }

    all_scored: List[ScoredItem] = []
    trace_summaries: Dict[str, Any] = {}
    for trace_file in sorted(config.output_dir.glob(f"traces.{config.task}.*.jsonl")):
        traces = read_traces(trace_file)
        if not traces:
            continue
        hashes = {t.framework_hash for t in traces}
        if len(hashes) > 1:
            code = _refuse_or_note(
                f"{trace_file.name} mixes framework hashes",
                f"{sorted(hashes)} — these traces were produced by different "
                f"frameworks and are not comparable. Re-run the older ones.",
            )
            if code is not None:
                return code

        # The framework hash is equal across models by design, so it cannot
        # catch a file that mixes model snapshots or pre/post-rewrite code.
        # The signature can.
        signatures = {t.run_signature for t in traces}
        if len(signatures) > 1:
            code = _refuse_or_note(
                f"{trace_file.name} mixes run signatures",
                f"{sorted(s or '(unsigned)' for s in signatures)} — different "
                f"models, code revisions or case sets produced these traces.",
            )
            if code is not None:
                return code
        model_key = traces[0].model_key
        frozen_signature = (manifest or {}).get("run_signatures", {}).get(model_key)
        observed = next(iter(signatures))
        if frozen_signature and observed and frozen_signature != observed:
            code = _refuse_or_note(
                f"{trace_file.name} was not produced by the run this manifest describes",
                f"manifest {str(frozen_signature)[:12]} != traces {str(observed)[:12]}",
            )
            if code is not None:
                return code

        if manifest is not None:
            # Score what the manifest describes, not what happens to be in the
            # directory. `run --models gpt` rewrote the model list while the
            # other four trace files stayed on disk, and score globbed them all
            # back in -- so the manifest said one model and the report showed
            # five. A file the manifest does not list is from another run.
            # ``or {}`` would have made an *empty* model map -- exactly what a
            # subset re-run leaves behind -- disable the check it exists for.
            declared_models = (
                set(manifest["models"]) if "models" in manifest else None
            )
            if declared_models is not None and model_key not in declared_models:
                code = _refuse_or_note(
                    f"{trace_file.name} is for a model this manifest does not list",
                    f"{model_key!r} is not in {sorted(declared_models)}; it belongs to "
                    f"a different run. Move or delete the stale trace file.",
                )
                if code is not None:
                    return code

            # The design fingerprint, per trace. run_signature deliberately
            # omits the condition so the arms can be pooled, which leaves a
            # trace from a dropped condition or a retired sample index matching
            # on signature alone.
            frozen_design = (manifest.get("design_signatures") or {}).get(model_key)
            declared_conditions = set(manifest.get("conditions") or config.conditions)
            declared_samples = int(manifest.get("samples") or config.samples)
            bad_design = {
                t.design_signature for t in traces if t.design_signature
            } - ({frozen_design} if frozen_design else set())
            if frozen_design and bad_design:
                code = _refuse_or_note(
                    f"{trace_file.name} mixes experiment designs",
                    f"manifest {str(frozen_design)[:12]} != {sorted(s[:12] for s in bad_design)}; "
                    f"the condition list, sample count or sampling rule changed after "
                    f"these traces were generated.",
                )
                if code is not None:
                    return code
            stray_conditions = {t.condition for t in traces} - declared_conditions
            if stray_conditions:
                code = _refuse_or_note(
                    f"{trace_file.name} holds conditions this experiment does not declare",
                    f"{sorted(stray_conditions)} not in {sorted(declared_conditions)}",
                )
                if code is not None:
                    return code
            stray_samples = {t.sample for t in traces if 0 <= t.sample >= declared_samples}
            if stray_samples:
                code = _refuse_or_note(
                    f"{trace_file.name} holds sample indices beyond samples={declared_samples}",
                    f"{sorted(stray_samples)}; scoring would weight those cases "
                    f"more heavily than their neighbours.",
                )
                if code is not None:
                    return code

        all_scored.extend(
            _score_traces(config, traces, gold, kg, pricing, contamination_cases)
        )
        by_condition: Dict[str, List[Trace]] = {}
        for trace in traces:
            by_condition.setdefault(trace.condition, []).append(trace)
        for condition, bucket in by_condition.items():
            # Keyed by dataset as well: `report` merges several configs into one
            # document, and a bare "model/condition" key let the PA summary for
            # a model overwrite its SDT summary, so the behavioural table showed
            # one dataset's numbers under both headings.
            key = f"{bucket[0].dataset or config.task}/{bucket[0].model_key}/{condition}"
            trace_summaries[key] = aggregate_trace_metrics(
                bucket, cost_per_mtok=pricing.get(bucket[0].model_key, (0.0, 0.0))
            )

    if not all_scored:
        print(f"no traces found under {config.output_dir}", file=sys.stderr)
        return 1

    out_path = config.output_dir / f"scores.{config.task}.jsonl"
    with open(out_path, "w", encoding="utf-8") as handle:
        for item in all_scored:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    (config.output_dir / f"trace_summary.{config.task}.json").write_text(
        json.dumps(trace_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if manifest is not None:
        (config.output_dir / f"scored_manifest.{config.task}.json").write_text(
            json.dumps(
                {
                    "generation_manifest": manifest,
                    "scored_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                    "scoring_code": code_fingerprints(),
                    "n_scored": len(all_scored),
                    # a score run that had to be forced says so permanently
                    "allow_drift": allow_drift,
                    "drift": drift_notes,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"scored {len(all_scored)} traces -> {out_path}", file=sys.stderr)

    metrics = {"sdt": SDT_METRICS, "pa": PA_METRICS, "cp": CP_METRICS}.get(
        config.task, PA_METRICS
    )
    from tcm_eval.report import main_table

    print(main_table(all_scored, config.task, metrics))
    return 0


def _load_scores(path: Path) -> List[ScoredItem]:
    items: List[ScoredItem] = []
    if not path.exists():
        return items
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        items.append(
            ScoredItem(
                case_id=payload["case_id"],
                dataset=payload["dataset"],
                condition=payload["condition"],
                model_key=payload["model_key"],
                sample=int(payload.get("sample", 0)),
                metrics=payload.get("metrics") or {},
                trace_metrics=payload.get("trace_metrics") or {},
            )
        )
    return items


# --------------------------------------------------------------------------- #
# judge / report / compare / inspect / coverage
# --------------------------------------------------------------------------- #


def cmd_judge(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    if config.task != "sdt":
        print("the judge scores SDT free-text steps only", file=sys.stderr)
        return 1
    dataset = load_dataset(config.dataset_path, config.dataset_kind, **config.loader_kwargs())
    gold = {str(item["id"]): item for item in dataset.items}

    specs = load_models(args.models_config)
    judge_key = args.judge_model or judge_model_key(args.models_config)
    if judge_key not in specs:
        print(f"judge model {judge_key!r} not in models config", file=sys.stderr)
        return 1
    client = build_client(
        specs[judge_key],
        cache_path=config.output_dir / "cache" / f"judge.{judge_key}.jsonl",
        replay=args.replay,
        script=args.echo_script,
    )
    judge = SDTJudge(client)

    rows: List[Dict[str, Any]] = []
    for trace_file in sorted(config.output_dir.glob(f"traces.{config.task}.*.jsonl")):
        for trace in read_traces(trace_file):
            reference = gold.get(trace.case_id)
            if reference is None:
                continue
            score = judge.score(trace.case_id, trace.final, reference)
            rows.append(
                {
                    "model_key": trace.model_key,
                    "condition": trace.condition,
                    "sample": trace.sample,
                    **score.to_dict(),
                }
            )
    out_path = config.output_dir / f"judge.{config.task}.jsonl"
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"judged {len(rows)} traces -> {out_path}", file=sys.stderr)
    return 0


#: Manifest fields the report reproduces verbatim, so a reader can check any
#: number in it against the run that produced it.
_MANIFEST_REPORT_KEYS = (
    "framework_hash", "task", "domain", "git_commit",
    "kg_content_sha256", "dataset_sha256", "case_set_sha256",
    "n_items", "created_at", "models", "run_signatures",
    "tools_impl_sha256", "scorers_impl_sha256", "retrieval_impl_sha256",
    "runtime_impl_sha256",
)


def cmd_report(args: argparse.Namespace) -> int:
    items: List[ScoredItem] = []
    summaries: Dict[str, Any] = {}
    # One provenance block per dataset. Keeping only the first config's block
    # meant a three-dataset report carried the SDT hashes and silently
    # attested that the PA and CP numbers came from the same run -- the one
    # claim the block exists to make, and the only one it could not support.
    framework: Dict[str, Any] = {}
    for config_path in args.configs:
        config = load_experiment(config_path)
        items.extend(_load_scores(config.output_dir / f"scores.{config.task}.jsonl"))
        summary_path = config.output_dir / f"trace_summary.{config.task}.json"
        if summary_path.exists():
            summaries.update(json.loads(summary_path.read_text(encoding="utf-8")))

        # Read the frozen manifest, never recompute: at report time the
        # config's kg_hash and dataset_hash are unset, so a recomputed
        # framework hash would differ from the one the traces were actually
        # generated under and the report would attest to a run that never
        # happened.
        manifest = _load_manifest(config)
        block: Dict[str, Any] = (
            {k: manifest[k] for k in _MANIFEST_REPORT_KEYS if k in manifest}
            if manifest
            else {
                "note": "no frozen manifest found; run `run` first",
                **config.framework.describe(),
            }
        )
        scored_manifest = config.output_dir / f"scored_manifest.{config.task}.json"
        if scored_manifest.exists():
            try:
                payload = json.loads(scored_manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            # A number scored under --allow-drift must carry that label into
            # the report, not only into a file nobody opens.
            if payload.get("allow_drift"):
                block["scored_with_allow_drift"] = True
                block["drift"] = payload.get("drift") or []
        framework[config.task] = block

    if not items:
        print("no scores found; run `score` first", file=sys.stderr)
        return 1
    report = build_report(items, trace_summaries=summaries, framework=framework)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    print(f"report -> {target}", file=sys.stderr)
    print(report)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Paired A/B between two score files (dsh-eval `eval compare`)."""
    a_items = _load_scores(Path(args.base))
    b_items = _load_scores(Path(args.candidate))
    if not a_items or not b_items:
        print("both score files must exist and be non-empty", file=sys.stderr)
        return 1

    metric = args.metric
    a_index = {(i.model_key, i.condition, i.case_id): i for i in a_items}
    b_index = {(i.model_key, i.condition, i.case_id): i for i in b_items}
    shared = sorted(set(a_index) & set(b_index))
    if not shared:
        print("no shared (model, condition, case) keys between the two runs", file=sys.stderr)
        return 1

    xs = [float(a_index[k].metrics.get(metric, 0.0)) for k in shared]
    ys = [float(b_index[k].metrics.get(metric, 0.0)) for k in shared]
    binary = all(v in (0.0, 1.0) for v in xs + ys)
    result = mcnemar(xs, ys) if binary else paired_bootstrap(xs, ys)

    payload = {
        "metric": metric,
        "n_shared": len(shared),
        "base": args.base,
        "candidate": args.candidate,
        **result.to_dict(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    """Write an official SDT submission file from recorded traces.

    Emitting the benchmark's own format means a run can be scored by the
    benchmark's own evaluator, rather than only by this harness's arithmetic.
    """
    config = load_experiment(args.config)
    if config.task != "sdt":
        print("submission files are defined for SDT only", file=sys.stderr)
        return 1
    dataset = load_dataset(config.dataset_path, config.dataset_kind, **config.loader_kwargs())
    order = [item["id"] for item in dataset.items]

    trace_file = config.output_dir / f"traces.{config.task}.{args.model}.jsonl"
    traces = [
        t
        for t in read_traces(trace_file)
        if t.condition == args.condition and t.sample == args.sample
    ]
    if not traces:
        print(f"no traces for {args.model}/{args.condition} in {trace_file}", file=sys.stderr)
        return 1
    by_case = {t.case_id: t.final for t in traces}
    rows = [(case_id, by_case.get(case_id)) for case_id in order]

    target = Path(args.out or config.output_dir / f"submission.{args.model}.{args.condition}.txt")
    write_submission(target, rows)
    missing = sum(1 for _c, prediction in rows if not prediction)
    print(
        f"wrote {len(rows)} lines to {target}"
        + (f" ({missing} cases had no answer and are blank)" if missing else ""),
        file=sys.stderr,
    )
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    kg, retriever = _load_graph(args.kg)
    print("== knowledge graph ==")
    print(json.dumps(kg.summary(), ensure_ascii=False, indent=2))

    from tcm_kg.schema import Domain, policy_for

    print("\n== access domains ==")
    for domain in (Domain.CLINICAL, Domain.SAFETY):
        policy = policy_for(domain)
        print(f"\n[{domain.value}]")
        print(f"  visible node types: {sorted(policy.allowed_nodes)}")
        if policy.verification_only_nodes:
            print(f"  verification only : {sorted(policy.verification_only_nodes)}")
        print(f"  tools             : {REGISTRY.names_for(domain)}")
        print(f"  rationale         : {policy.rationale}")

    print(f"\n== tool contract (fingerprint {REGISTRY.fingerprint()}) ==")
    for spec in sorted(REGISTRY.specs_for(Domain.FULL), key=lambda s: s.name):
        marker = " [deterministic]" if spec.deterministic else ""
        print(f"\n{spec.name}{marker}")
        print(f"  {spec.description}")

    if args.dataset:
        print("\n== dataset ==")
        print(inspect_dataset(args.dataset, args.dataset_kind))
    return 0


#: One structured answer, in the schema every task uses. Short enough to cost
#: almost nothing across five providers, strict enough that a provider which
#: cannot hold the output contract fails here rather than mid-run.
_SMOKE_PROMPT = (
    "请只输出一个 JSON 对象，不要输出任何其他内容，格式为："
    '{"action": "answer", "result": {"syndrome_answer": ["A"], "note": "ok"}}'
)


def cmd_smoke(args: argparse.Namespace) -> int:
    """Check every configured provider before spending a real run on it.

    A five-model comparison is only a comparison of models if the adapters
    behave the same. Each one is asked the same short question twice at
    temperature 0 with a fixed seed, and the reply is checked for the things
    that would otherwise turn into a silent per-provider confound: whether the
    JSON contract holds, whether token usage is reported (a provider that
    returns none makes its cost and its context growth unmeasurable), and
    whether two identical seeded calls agree.

    Determinism is *reported, not required*. Providers differ in whether they
    honour a seed at all, and the run is still valid if they do not -- but with
    ``samples > 1`` a provider that ignores the seed is not drawing the
    independent samples self-consistency assumes, and that has to be known
    before the run rather than inferred from the results.
    """
    specs = load_models(args.models_config)
    wanted = args.models or sorted(specs)
    decode = DecodeParams(temperature=0.0, top_p=1.0, max_tokens=256, seed=20260829)
    rows: List[List[Any]] = []
    failures: List[str] = []

    for key in wanted:
        spec = specs.get(key)
        if spec is None:
            failures.append(f"{key}: not in {args.models_config}")
            continue
        client = build_client(spec)
        messages = [Message("user", _SMOKE_PROMPT)]
        try:
            first = client.generate(messages, decode, sample=0)
            second = client.generate(messages, decode, sample=0)
        except Exception as exc:  # a provider that cannot answer at all
            failures.append(f"{key}: {type(exc).__name__}: {exc}")
            rows.append([key, spec.model_id, "ERROR", "-", "-", "-", "-", str(exc)[:40]])
            continue

        parsed = extract_json_object(first.text)
        usage = first.to_dict().get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        deterministic = first.text.strip() == second.text.strip()

        problems = []
        if parsed.value is None:
            problems.append("no parseable JSON")
        if not prompt_tokens or not completion_tokens:
            problems.append("no token usage")
        if first.error:
            problems.append(f"error: {first.error}")
        if problems:
            failures.append(f"{key}: " + "; ".join(problems))

        rows.append([
            key,
            spec.model_id,
            "ok" if not problems else "FAIL",
            "yes" if parsed.value is not None else "no",
            prompt_tokens or None,
            completion_tokens or None,
            "yes" if deterministic else "no",
            spec.fingerprint()[:12],
        ])

    print(_md_table(
        ["model", "model_id", "status", "JSON", "prompt tok", "completion tok",
         "seed→identical", "fingerprint"],
        rows,
    ))
    if any(r[6] == "no" for r in rows if r[2] != "ERROR"):
        print(
            "\nNOTE: a provider whose two seeded calls differ does not pin its "
            "sampling. Fine at samples=1; with samples>1 its self-consistency "
            "draws are not comparable to a seeded provider's, and the paper has "
            "to say so.",
            file=sys.stderr,
        )
    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        print(
            "\nFix these before generating: an adapter defect becomes a "
            "per-model confound that no paired test can remove.",
            file=sys.stderr,
        )
        return 2
    print("\nall providers ready", file=sys.stderr)
    return 0


def cmd_contaminate(args: argparse.Namespace) -> int:
    """Audit the benchmark against the graph's own text.

    Run this before generation, not after: the stratum it assigns each case is
    what makes the clean-subset sensitivity analysis possible, and that is the
    only argument that separates "the graph improved reasoning" from "the graph
    contained the answer".
    """
    from tcm_eval.contamination import GraphText, audit_dataset, format_report

    config = load_experiment(args.config)
    dataset = load_dataset(
        config.dataset_path, config.dataset_kind, **config.loader_kwargs()
    )
    items = dataset.subset(
        config.dataset_limit, stratify=config.dataset_stratify
    ).items

    kg = load_kg(args.kg)
    print(f"indexing graph text ...", file=sys.stderr)
    graph = GraphText.from_kg(kg)
    print(f"  {len(graph)} passages", file=sys.stderr)

    from tcm_eval.contamination import audit_identity

    report = audit_dataset(items, graph, config.dataset_kind)
    report["identity"] = audit_identity(
        kg_hash=kg.content_hash(),
        dataset_hash=_file_hash(config.dataset_path),
        gold_hash=_gold_hash(dataset, config),
        case_set_hash=case_set_hash([str(i["id"]) for i in items]),
    )

    out = Path(args.out or config.output_dir / f"contamination.{config.task}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    markdown = format_report(report)
    Path(str(out).replace(".json", ".md")).write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\n(written to {out})", file=sys.stderr)
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    from scripts.kg_coverage import coverage_report

    kg, retriever = _load_graph(args.kg)
    report = coverage_report(kg, retriever)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n(written to {target})", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_runner",
        description="TCM-KG Agent benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--kg", default=None, help="path to the knowledge graph artefact")
    parser.add_argument(
        "--models-config", default="configs/models.yaml", help="model registry"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="generate traces")
    run.add_argument("config")
    run.add_argument("--models", nargs="*", help="override the model list")
    run.add_argument("--conditions", nargs="*", help="override the condition list")
    run.add_argument("--limit", type=int, default=None, help="first N dataset items")
    run.add_argument("--overwrite", action="store_true", help="ignore recorded traces")
    run.add_argument(
        "--new-run",
        action="store_true",
        help="this output_dir holds a different experiment: archive its manifest "
        "and start a new one here",
    )
    run.add_argument("--replay", action="store_true", help="offline: cache only, no API key")
    run.add_argument("--echo-script", nargs="*", default=None, help="scripted offline responses")
    run.set_defaults(func=cmd_run)

    smoke = sub.add_parser(
        "smoke", help="check every provider adapter before a real run"
    )
    smoke.add_argument("--models", nargs="*", help="subset to check")
    smoke.set_defaults(func=cmd_smoke)

    contaminate = sub.add_parser(
        "contaminate",
        help="audit the benchmark against the graph's text before running",
    )
    contaminate.add_argument("config")
    contaminate.add_argument("--out", default=None)
    contaminate.set_defaults(func=cmd_contaminate)

    score = sub.add_parser("score", help="score recorded traces")
    score.add_argument("config")
    score.add_argument(
        "--allow-drift",
        action="store_true",
        help="score anyway when the traces do not match the frozen manifest "
        "(mixed run signatures, changed KG, dataset or framework). Reports "
        "produced this way must say so.",
    )
    score.set_defaults(func=cmd_score)

    judge = sub.add_parser("judge", help="LLM-judge the SDT free-text steps")
    judge.add_argument("config")
    judge.add_argument("--judge-model", default=None)
    judge.add_argument("--replay", action="store_true")
    judge.add_argument("--echo-script", nargs="*", default=None)
    judge.set_defaults(func=cmd_judge)

    report = sub.add_parser("report", help="build the Markdown report")
    report.add_argument("configs", nargs="+")
    report.add_argument("--out", default="runs/report.md")
    report.set_defaults(func=cmd_report)

    compare = sub.add_parser("compare", help="paired A/B between two score files")
    compare.add_argument("base")
    compare.add_argument("candidate")
    compare.add_argument("--metric", default="syndrome_exact")
    compare.set_defaults(func=cmd_compare)

    submit = sub.add_parser("submit", help="write an official SDT submission file")
    submit.add_argument("config")
    submit.add_argument("--model", required=True)
    submit.add_argument("--condition", default="M3")
    submit.add_argument("--sample", type=int, default=0)
    submit.add_argument("--out", default=None)
    submit.set_defaults(func=cmd_submit)

    inspect = sub.add_parser("inspect", help="print the graph, domain and tool contract")
    inspect.add_argument("--dataset", default=None)
    inspect.add_argument(
        "--dataset-kind", default="sdt", choices=["sdt", "pa", "tcmsd", "cp"]
    )
    inspect.set_defaults(func=cmd_inspect)

    coverage = sub.add_parser(
        "coverage", help="audit which PA rule families the graph can ground"
    )
    coverage.add_argument("--out", default="docs/kg_coverage.md")
    coverage.set_defaults(func=cmd_coverage)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
