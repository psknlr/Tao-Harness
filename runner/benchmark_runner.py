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
import json
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
from tcm_eval import (
    PA_METRICS,
    PA_PRIMARY,
    SDT_METRICS,
    SDT_PRIMARY,
    ScoredItem,
    aggregate_trace_metrics,
    build_report,
    inspect_dataset,
    load_dataset,
    score_pa,
    score_sdt,
    trace_metrics,
)
from tcm_eval.judge import SDTJudge, aggregate_judge
from tcm_eval.metrics import tool_selection_accuracy
from tcm_eval.official_sdt import write_submission
from tcm_eval.stats import mcnemar, paired_bootstrap
from tcm_kg import load_kg
from tcm_kg.index import KGRetriever
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
    cache_file = cache_dir / f"{retriever.cache_key(str(len(kg.nodes)))}.pkl"
    if not retriever.load(cache_file):
        retriever.warm()
        retriever.save(cache_file)
    return kg, retriever


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
    items = dataset.subset(config.dataset_limit).items
    if not items:
        print("no items to run", file=sys.stderr)
        return 1

    kg, retriever = _load_graph(args.kg, config)
    specs = load_models(args.models_config)
    framework_hash = config.framework.framework_hash()
    print(f"framework_hash={framework_hash}  n_items={len(items)}", file=sys.stderr)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "run_config.json").write_text(
        json.dumps(config.describe(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_written = 0
    for model_key in config.models:
        if model_key not in specs:
            print(f"skipping unknown model {model_key!r}", file=sys.stderr)
            continue
        trace_file = _trace_path(config, model_key)
        existing = {
            (t.case_id, t.condition, t.sample): t
            for t in read_traces(trace_file)
            if t.framework_hash == framework_hash and not args.overwrite
        }
        if existing:
            print(f"{model_key}: resuming, {len(existing)} traces already recorded", file=sys.stderr)

        client = build_client(
            specs[model_key],
            cache_path=_cache_path(config, model_key),
            replay=args.replay,
            script=args.echo_script,
        )
        task = build_task(config.task, kg, retriever, config.framework.context_budget)

        jobs: List[Tuple[Mapping[str, Any], str, int]] = [
            (item, condition, sample)
            for item in items
            for condition in config.conditions
            for sample in range(config.samples)
            if (str(item["id"]), condition, sample) not in existing
        ]
        print(f"{model_key}: {len(jobs)} traces to generate", file=sys.stderr)

        traces: List[Trace] = list(existing.values())
        started = time.time()

        def _one(job: Tuple[Mapping[str, Any], str, int]) -> Trace:
            item, condition, sample = job
            runtime = AgentRuntime(
                kg, retriever, task, client, config.framework, run_id=config.name
            )
            return runtime.run(item, condition, sample=sample)

        if config.concurrency > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
                futures = {pool.submit(_one, job): job for job in jobs}
                for done, future in enumerate(as_completed(futures), 1):
                    traces.append(future.result())
                    if done % 20 == 0 or done == len(jobs):
                        print(
                            f"  {model_key}: {done}/{len(jobs)} "
                            f"({time.time() - started:.0f}s)",
                            file=sys.stderr,
                        )
                        write_traces(trace_file, traces)
        else:
            for done, job in enumerate(jobs, 1):
                traces.append(_one(job))
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


def _score_traces(
    config: ExperimentConfig, traces: Sequence[Trace], gold: Mapping[str, Mapping[str, Any]], kg
) -> List[ScoredItem]:
    scored: List[ScoredItem] = []
    for trace in traces:
        reference = gold.get(trace.case_id)
        if reference is None:
            continue
        if not reference.get("_has_gold"):
            continue  # generated but unscoreable (a split with blank answers)
        if config.task == "sdt":
            metrics = score_sdt(trace.final, reference, kg=kg)
        else:
            metrics = score_pa(trace.final, reference)
            metrics["rule_id"] = reference.get("rule_id")
            metrics["category"] = reference.get("category")
            accuracy = tool_selection_accuracy(trace, reference.get("rule_id"))
            if accuracy is not None:
                metrics["tool_selection_accuracy"] = accuracy
        scored.append(
            ScoredItem(
                case_id=trace.case_id,
                dataset=trace.dataset,
                condition=trace.condition,
                model_key=trace.model_key,
                sample=trace.sample,
                metrics=metrics,
                trace_metrics=trace_metrics(trace),
            )
        )
    return scored


def cmd_score(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    dataset = load_dataset(config.dataset_path, config.dataset_kind, **config.loader_kwargs())
    gold = {str(item["id"]): item for item in dataset.items}
    kg, _ = _load_graph(args.kg, config) if config.task == "sdt" else (None, None)

    all_scored: List[ScoredItem] = []
    trace_summaries: Dict[str, Any] = {}
    for trace_file in sorted(config.output_dir.glob(f"traces.{config.task}.*.jsonl")):
        traces = read_traces(trace_file)
        if not traces:
            continue
        hashes = {t.framework_hash for t in traces}
        if len(hashes) > 1:
            print(
                f"REFUSING {trace_file.name}: mixed framework hashes {sorted(hashes)}. "
                f"These traces were produced by different frameworks and are not "
                f"comparable. Re-run the older ones.",
                file=sys.stderr,
            )
            return 2
        all_scored.extend(_score_traces(config, traces, gold, kg))
        by_condition: Dict[str, List[Trace]] = {}
        for trace in traces:
            by_condition.setdefault(trace.condition, []).append(trace)
        for condition, bucket in by_condition.items():
            trace_summaries[f"{bucket[0].model_key}/{condition}"] = aggregate_trace_metrics(bucket)

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
    print(f"scored {len(all_scored)} traces -> {out_path}", file=sys.stderr)

    metrics = SDT_METRICS if config.task == "sdt" else PA_METRICS
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


def cmd_report(args: argparse.Namespace) -> int:
    items: List[ScoredItem] = []
    summaries: Dict[str, Any] = {}
    framework: Optional[Mapping[str, Any]] = None
    for config_path in args.configs:
        config = load_experiment(config_path)
        items.extend(_load_scores(config.output_dir / f"scores.{config.task}.jsonl"))
        summary_path = config.output_dir / f"trace_summary.{config.task}.json"
        if summary_path.exists():
            summaries.update(json.loads(summary_path.read_text(encoding="utf-8")))
        if framework is None:
            framework = config.framework.describe()
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
    run.add_argument("--replay", action="store_true", help="offline: cache only, no API key")
    run.add_argument("--echo-script", nargs="*", default=None, help="scripted offline responses")
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="score recorded traces")
    score.add_argument("config")
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
    inspect.add_argument("--dataset-kind", default="sdt", choices=["sdt", "pa", "tcmsd"])
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
