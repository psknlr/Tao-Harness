"""Evaluation: datasets, scorers, judge, trace metrics, statistics, reporting."""

from .datasets import Dataset, FieldMapping, inspect_dataset, load_dataset
from .judge import JudgeScore, SDTJudge, aggregate_judge
from .metrics import (
    aggregate_trace_metrics,
    coverage_honesty,
    tool_selection_accuracy,
    trace_metrics,
)
from .report import build_report, contrast_table, main_table
from .scorers import (
    PA_METRICS,
    PA_PRIMARY,
    SDT_METRICS,
    SDT_PRIMARY,
    ScoredItem,
    aggregate,
    majority_vote,
    score_pa,
    score_sdt,
)
from .stats import holm_bonferroni, mcnemar, paired_bootstrap, pearson, spearman

__all__ = [
    "Dataset",
    "FieldMapping",
    "JudgeScore",
    "PA_METRICS",
    "PA_PRIMARY",
    "SDTJudge",
    "SDT_METRICS",
    "SDT_PRIMARY",
    "ScoredItem",
    "aggregate",
    "aggregate_judge",
    "aggregate_trace_metrics",
    "build_report",
    "contrast_table",
    "coverage_honesty",
    "holm_bonferroni",
    "inspect_dataset",
    "load_dataset",
    "main_table",
    "majority_vote",
    "mcnemar",
    "paired_bootstrap",
    "pearson",
    "score_pa",
    "score_sdt",
    "spearman",
    "tool_selection_accuracy",
    "trace_metrics",
]
