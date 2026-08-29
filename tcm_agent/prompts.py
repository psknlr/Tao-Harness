"""Frozen prompt registry.

Prompts live as plain text files so they can be diffed, reviewed and printed in
a paper appendix.  Their combined SHA-256 is folded into the framework hash: if
a prompt changes, runs made before and after are no longer comparable and the
runner refuses to pool them.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"prompt {name!r} not found at {path}")
    return path.read_text(encoding="utf-8").rstrip("\n")


@lru_cache(maxsize=1)
def all_prompts() -> Dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(PROMPT_DIR.glob("*.txt"))}


@lru_cache(maxsize=1)
def prompt_fingerprint() -> str:
    digest = hashlib.sha256()
    for name, body in sorted(all_prompts().items()):
        digest.update(name.encode("utf-8"))
        digest.update(body.encode("utf-8"))
    return digest.hexdigest()[:16]
