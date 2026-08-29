"""Dataset loading for TCMEval-SDT and TCMEval-PA.

The published record schemas are described in prose ("medical record ID,
clinical data, explanatory summary, TCM syndrome, clinical information, TCM
pathogenesis" for SDT; 297 single- plus 31 multiple-choice items for PA) but
the concrete JSON keys vary between the distributed files and any local
re-export.  Rather than hard-coding one guess, each field declares an alias
list and the loader reports which key it actually bound to, so a schema
mismatch surfaces as a loud mapping report instead of a silent column of
zeros.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: field -> candidate keys, in priority order
SDT_ALIASES: Mapping[str, Sequence[str]] = {
    "id": ("id", "case_id", "medical_record_id", "record_id", "ID", "编号", "病案号", "医案编号"),
    "clinical_data": (
        "clinical_data",
        "clinicalData",
        "medical_record",
        "case",
        "case_text",
        "text",
        "临床资料",
        "病例",
        "医案",
        "四诊信息",
    ),
    "clinical_information": (
        "clinical_information",
        "clinicalInformation",
        "clinical_info",
        "临床信息",
        "临床资料提取",
    ),
    "pathogenesis": (
        "TCM_pathogenesis",
        "tcm_pathogenesis",
        "pathogenesis",
        "TCMPathogenesis",
        "病机",
        "中医病机",
    ),
    "syndrome": (
        "TCM_syndrome",
        "tcm_syndrome",
        "syndrome",
        "TCMSyndrome",
        "证候",
        "中医证候",
        "证型",
        "辨证",
    ),
    "explanation": (
        "explanatory_summary",
        "explanation",
        "summary",
        "explain",
        "辨证分析",
        "解释",
        "分析",
    ),
}

PA_ALIASES: Mapping[str, Sequence[str]] = {
    "id": ("id", "question_id", "qid", "ID", "编号", "题号"),
    "question": ("question", "stem", "query", "题干", "问题", "content"),
    "options": ("options", "choices", "选项", "option"),
    "answer": ("answer", "answers", "label", "gold", "correct_answer", "正确答案", "标准答案", "答案"),
    "question_type": ("question_type", "type", "题型", "q_type", "category_type"),
    "rule_id": ("rule_id", "rule", "rule_code", "规则编号", "规则", "knowledge_point", "考点", "category"),
    "explanation": ("explanation", "analysis", "解析", "答案解析"),
}


@dataclass
class FieldMapping:
    """Which source key each logical field bound to."""

    bound: Dict[str, str] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    unmapped_keys: List[str] = field(default_factory=list)

    def report(self) -> str:
        lines = ["field mapping:"]
        for logical, source in sorted(self.bound.items()):
            lines.append(f"  {logical:22s} <- {source}")
        if self.missing:
            lines.append(f"  MISSING (no alias matched): {self.missing}")
        if self.unmapped_keys:
            lines.append(f"  unused source keys: {self.unmapped_keys[:12]}")
        return "\n".join(lines)


@dataclass
class Dataset:
    name: str
    items: List[Dict[str, Any]]
    mapping: FieldMapping
    path: Optional[Path] = None

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def subset(self, limit: Optional[int] = None, ids: Optional[Iterable[str]] = None) -> "Dataset":
        items = self.items
        if ids is not None:
            wanted = {str(i) for i in ids}
            items = [i for i in items if str(i.get("id")) in wanted]
        if limit is not None:
            items = items[:limit]
        return Dataset(self.name, items, self.mapping, self.path)


def _first_present(record: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    lowered = {str(k).lower(): k for k in record}
    for key in keys:
        if key in record:
            return key
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def _load_records(path: Path) -> List[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in ("data", "items", "records", "questions", "cases"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        # a dict keyed by case id
        if all(isinstance(v, Mapping) for v in payload.values()):
            return [{"id": k, **v} for k, v in payload.items()]
    raise ValueError(f"unrecognised dataset structure in {path}")


def load_dataset(
    path: str | Path, kind: str, *, aliases: Optional[Mapping[str, Sequence[str]]] = None
) -> Dataset:
    """Load and normalise SDT or PA records."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"dataset not found: {target}")
    alias_map = aliases or (SDT_ALIASES if kind == "sdt" else PA_ALIASES)
    records = _load_records(target)
    if not records:
        raise ValueError(f"{target} contains no records")

    # Bind over a sample rather than the first record alone: a re-export that
    # mixes key styles would otherwise bind to whichever style happens to be
    # first and silently null out every other record.
    probe_records = records[: min(len(records), 20)]
    mapping = FieldMapping()
    per_field_sources: Dict[str, List[str]] = {}
    for logical, candidates in alias_map.items():
        counts: Dict[str, int] = {}
        for record in probe_records:
            source = _first_present(record, candidates)
            if source is not None and record.get(source) not in (None, ""):
                counts[source] = counts.get(source, 0) + 1
        if not counts:
            mapping.missing.append(logical)
            continue
        ordered = sorted(counts, key=lambda k: (-counts[k], list(candidates).index(k) if k in candidates else 99))
        mapping.bound[logical] = ordered[0]
        per_field_sources[logical] = ordered
    seen_sources = {s for sources in per_field_sources.values() for s in sources}
    mapping.unmapped_keys = sorted(
        {k for record in probe_records for k in record} - seen_sources
    )

    items: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        item: Dict[str, Any] = {}
        for logical in mapping.bound:
            value = None
            for source in per_field_sources.get(logical, ()):
                if record.get(source) not in (None, ""):
                    value = record.get(source)
                    break
            item[logical] = value
        item.setdefault("id", None)
        if item.get("id") in (None, ""):
            item["id"] = f"{kind}_{index:04d}"
        item["id"] = str(item["id"])
        item["_raw"] = dict(record)
        items.append(item)

    if kind == "pa":
        for item in items:
            item["options"] = _normalise_options(item.get("options"))
    return Dataset(name=kind, items=items, mapping=mapping, path=target)


def _normalise_options(options: Any) -> Dict[str, str]:
    """Coerce list / dict / newline-string option blocks into ``{letter: text}``."""
    if isinstance(options, Mapping):
        return {str(k).strip().upper()[:1] or str(k): str(v) for k, v in options.items()}
    out: Dict[str, str] = {}
    if isinstance(options, str):
        options = [line for line in options.splitlines() if line.strip()]
    if isinstance(options, (list, tuple)):
        for index, entry in enumerate(options):
            text = str(entry).strip()
            if len(text) > 1 and text[0].isalpha() and text[1] in ".、:：)） ":
                out[text[0].upper()] = text[2:].strip()
            else:
                out[chr(ord("A") + index)] = text
    return out


def inspect_dataset(path: str | Path, kind: str) -> str:
    """Human-readable schema report, used by ``benchmark_runner.py inspect``."""
    dataset = load_dataset(path, kind)
    lines = [
        f"dataset: {kind}  path: {dataset.path}",
        f"records: {len(dataset)}",
        dataset.mapping.report(),
    ]
    if dataset.items:
        sample = {k: v for k, v in dataset.items[0].items() if k != "_raw"}
        lines.append("first record (mapped):")
        lines.append(json.dumps(sample, ensure_ascii=False, indent=2)[:1500])
    if kind == "pa":
        from collections import Counter

        counts = Counter(
            len(str(i.get("answer") or "").replace(" ", "")) > 1 for i in dataset.items
        )
        lines.append(f"multi-answer items (by answer length heuristic): {counts}")
    return "\n".join(lines)
