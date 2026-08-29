"""Experiment configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from tcm_agent.runtime import CONDITIONS, FrameworkConfig
from tcm_agent.tasks import ContextBudget
from tcm_kg.index import RetrievalParams
from tcm_kg.schema import Domain
from tcm_models.base import DecodeParams, ModelSpec
from tcm_models.registry import spec_from_config
from tcm_tools.base import REGISTRY, ToolBudget

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ExperimentConfig:
    name: str
    task: str
    domain: Domain
    dataset_path: Path
    dataset_kind: str
    dataset_limit: Optional[int]
    models: List[str]
    conditions: List[str]
    samples: int
    framework: FrameworkConfig
    output_dir: Path
    concurrency: int
    raw: Mapping[str, Any] = field(default_factory=dict)

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "domain": self.domain.value,
            "dataset": str(self.dataset_path),
            "models": self.models,
            "conditions": self.conditions,
            "samples": self.samples,
            **self.framework.describe(),
        }


def _resolve(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else REPO_ROOT / target


def load_models(path: str | Path = "configs/models.yaml") -> Dict[str, ModelSpec]:
    payload = yaml.safe_load(_resolve(path).read_text(encoding="utf-8")) or {}
    return {
        key: spec_from_config(key, config)
        for key, config in (payload.get("models") or {}).items()
    }


def judge_model_key(path: str | Path = "configs/models.yaml") -> Optional[str]:
    payload = yaml.safe_load(_resolve(path).read_text(encoding="utf-8")) or {}
    return (payload.get("judge") or {}).get("model")


def load_experiment(path: str | Path) -> ExperimentConfig:
    payload = yaml.safe_load(_resolve(path).read_text(encoding="utf-8")) or {}

    dataset = payload.get("dataset") or {}
    conditions = [str(c).upper() for c in (payload.get("conditions") or list(CONDITIONS))]
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(f"unknown conditions {unknown}; known: {list(CONDITIONS)}")

    decode_raw = payload.get("decode") or {}
    retrieval_raw = payload.get("retrieval") or {}
    tool_raw = payload.get("tool_budget") or {}
    context_raw = payload.get("context_budget") or {}

    framework = FrameworkConfig(
        retrieval=RetrievalParams(
            **{k: v for k, v in retrieval_raw.items() if k in RetrievalParams.__dataclass_fields__}
        ),
        tool_budget=ToolBudget(
            **{k: v for k, v in tool_raw.items() if k in ToolBudget.__dataclass_fields__}
        ),
        context_budget=ContextBudget(
            **{k: v for k, v in context_raw.items() if k in ContextBudget.__dataclass_fields__}
        ),
        decode=DecodeParams(
            **{k: v for k, v in decode_raw.items() if k in DecodeParams.__dataclass_fields__}
        ),
        max_agent_turns=int(payload.get("max_agent_turns", 16)),
        max_format_retries=int(payload.get("max_format_retries", 2)),
        registry=REGISTRY,
    )

    return ExperimentConfig(
        name=str(payload.get("name") or Path(path).stem),
        task=str(payload.get("task") or dataset.get("kind") or "sdt"),
        domain=Domain(str(payload.get("domain") or Domain.CLINICAL.value)),
        dataset_path=_resolve(dataset.get("path") or ""),
        dataset_kind=str(dataset.get("kind") or "sdt"),
        dataset_limit=dataset.get("limit"),
        models=[str(m) for m in (payload.get("models") or ["echo"])],
        conditions=conditions,
        samples=int(payload.get("samples", 1)),
        framework=framework,
        output_dir=_resolve(payload.get("output_dir") or "runs"),
        concurrency=int(payload.get("concurrency", 4)),
        raw=payload,
    )
