"""Evaluation: datasets, scorers, judge, trace metrics, statistics, reporting."""

from .datasets import (
    Dataset,
    FieldMapping,
    inspect_dataset,
    load_dataset,
    load_syndrome_knowledge,
    parse_letters,
    parse_options,
    parse_result_file,
)
from .judge import JudgeScore, SDTJudge, aggregate_judge
from .metrics import (
    aggregate_trace_metrics,
    coverage_honesty,
    tool_selection_accuracy,
    trace_metrics,
)
from .official_sdt import (
    clinical_info_extraction_eval,
    rouge_l,
    score_proportional,
    to_submission_line,
    write_submission,
)
from .report import build_report, contrast_table, main_table, pa_rule_table
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
    "clinical_info_extraction_eval",
    "contrast_table",
    "coverage_honesty",
    "holm_bonferroni",
    "inspect_dataset",
    "load_dataset",
    "load_syndrome_knowledge",
    "pa_rule_table",
    "parse_letters",
    "parse_options",
    "parse_result_file",
    "rouge_l",
    "score_proportional",
    "to_submission_line",
    "write_submission",
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
