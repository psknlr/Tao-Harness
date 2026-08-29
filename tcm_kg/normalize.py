"""Text normalisation for Chinese clinical strings.

Everything here is deterministic and dependency-free so that retrieval results
are byte-identical across machines and across models -- a precondition for the
frozen-framework comparison in :mod:`tcm_agent.runtime`.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Character-level normalisation
# --------------------------------------------------------------------------- #

_PUNCT_MAP = str.maketrans(
    {
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
        "，": ",",
        "、": ",",
        "；": ";",
        "：": ":",
        "。": ".",
        "！": "!",
        "？": "?",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "～": "~",
        "－": "-",
        "—": "-",
        "　": " ",
    }
)

_WS_RE = re.compile(r"\s+")
#: Enumeration prefixes used throughout the protocols: ``1.`` ``（3）`` ``③`` ``二、``
_ENUM_PREFIX_RE = re.compile(
    r"^\s*(?:[\(\[（［]?\s*(?:\d{1,2}|[一二三四五六七八九十]{1,3})\s*[\)\]）］]?\s*[.,;:、，；：。]?"
    r"|[①-⑳㈠-㈩]|[▲△●○※\*\-])+\s*"
)
_TRAILING_ETC_RE = re.compile(r"[,;，、；]?\s*(?:等|等等)\s*[.。]?\s*$")


def normalize_text(value: object) -> str:
    """Fold width, unify punctuation, collapse whitespace, lowercase ASCII."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.translate(_PUNCT_MAP)
    text = _WS_RE.sub(" ", text).strip()
    return text.lower()


def strip_enumeration(text: str) -> str:
    """Drop a leading list marker (``1.``/``（3）``/``③``) from a protocol line."""
    prev = None
    out = text.strip()
    while prev != out:
        prev = out
        out = _ENUM_PREFIX_RE.sub("", out).strip()
    return out


def strip_trailing_etc(text: str) -> str:
    """Drop a trailing ``等`` that would otherwise pollute an entity name."""
    return _TRAILING_ETC_RE.sub("", text).strip()


# --------------------------------------------------------------------------- #
# Departments: the graph carries six pairs of synonymous department labels.
# --------------------------------------------------------------------------- #

DEPARTMENT_CANONICAL: Dict[str, str] = {
    "传染科": "传染病科",
    "传染病科": "传染病科",
    "脾胃科": "脾胃病科",
    "脾胃病科": "脾胃病科",
    "呼吸科": "肺病科",
    "肺病科": "肺病科",
    "肝病科": "肝胆病科",
    "肝胆病科": "肝胆病科",
    "血液科": "血液病科",
    "血液病科": "血液病科",
    "风湿科": "风湿病科",
    "风湿病科": "风湿病科",
    "感染科": "传染病科",
}


def canonical_department(name: str) -> str:
    """Map a department label onto its canonical form (identity if unknown)."""
    return DEPARTMENT_CANONICAL.get(name.strip(), name.strip())


# --------------------------------------------------------------------------- #
# Herb names: the graph stores inline preparation annotations inside the name,
# e.g. ``麝香（冲服，或白芷代）`` or ``生石膏(先煎)``.  Splitting them out is what
# makes the N-003 decoction checker groundable.
# --------------------------------------------------------------------------- #

#: Preparation requirements recognised in herb names and in edge evidence.
DECOCTION_MARKERS: Tuple[str, ...] = (
    "先煎",
    "后下",
    "包煎",
    "另煎",
    "烊化",
    "冲服",
    "另炖",
    "兑服",
    "泡服",
    "研末",
    "调服",
)

_PAREN_RE = re.compile(r"[(（]([^)）]*)[)）]")


def split_herb_annotation(raw_name: str) -> Tuple[str, List[str]]:
    """Split ``生石膏(先煎)`` into ``("生石膏", ["先煎"])``.

    Any parenthetical that contains no recognised preparation marker is kept as
    part of the base name, because for diseases and formulas the parenthetical
    is the western-medicine gloss (``心悸（心律失常-室性早搏）``) and dropping it
    would merge distinct entities.
    """
    markers: List[str] = []
    base = raw_name

    def _replace(match: "re.Match[str]") -> str:
        inner = match.group(1)
        hits = [m for m in DECOCTION_MARKERS if m in inner]
        if hits:
            markers.extend(hits)
            return ""
        return match.group(0)

    base = _PAREN_RE.sub(_replace, base).strip()
    base = strip_trailing_etc(base).strip(" ,;，、；")
    # de-duplicate while preserving order
    seen: set = set()
    ordered = [m for m in markers if not (m in seen or seen.add(m))]
    return (base or raw_name.strip()), ordered


def extract_decoction_markers(text: str) -> List[str]:
    """Return preparation markers mentioned anywhere in a free-text string."""
    seen: set = set()
    out: List[str] = []
    for marker in DECOCTION_MARKERS:
        if marker in text and marker not in seen:
            seen.add(marker)
            out.append(marker)
    return out


#: A herb name may be preceded by a processing prefix in running text
#: (``生石膏`` for ``石膏``) but the annotation always follows the name directly.
_HERB_PREFIXES = ("生", "炒", "焦", "炙", "煅", "醋", "酒", "盐", "蜜", "制", "熟", "清", "姜")


def markers_for_herb_in_sentence(sentence: str, herb_names: Iterable[str]) -> List[str]:
    """Preparation markers that a sentence attributes *to this herb specifically*.

    A recipe line lists many herbs and annotates only some of them --
    ``"生地、生石膏、地榆炭、生大黄(后下)等"`` must yield ``后下`` for 大黄 and
    nothing for 石膏.  Attribution is therefore positional: the parenthetical
    has to open immediately after an occurrence of the herb's name (allowing a
    trailing processing character), not merely appear in the same sentence.
    """
    out: List[str] = []
    seen: set = set()
    for name in herb_names:
        name = (name or "").strip()
        if len(name) < 2:
            continue
        start = 0
        while True:
            idx = sentence.find(name, start)
            if idx < 0:
                break
            start = idx + len(name)
            tail = sentence[start:]
            # allow "石膏 (先煎)" and "石膏粉(先煎)" but not "石膏、大黄(后下)"
            offset = 0
            while offset < len(tail) and tail[offset] in " \t":
                offset += 1
            if offset < len(tail) and tail[offset] in "(（":
                closing = tail.find(")", offset)
                closing_full = tail.find("）", offset)
                if closing < 0 or (0 <= closing_full < closing):
                    closing = closing_full
                if closing > offset:
                    for marker in extract_decoction_markers(tail[offset + 1 : closing]):
                        if marker not in seen:
                            seen.add(marker)
                            out.append(marker)
    return out


# --------------------------------------------------------------------------- #
# Syndrome names
# --------------------------------------------------------------------------- #

_SYNDROME_SPLIT_RE = re.compile(r"[,;/|，、；／｜]|兼夹|夹杂")
_SYNDROME_DESC_RE = re.compile(r"^([^:：]{2,20}?证)\s*[:：]")


def canonical_syndrome(name: str) -> str:
    """Canonical surface form of a syndrome name.

    Removes list markers and a trailing description, keeps the ``证`` suffix if
    present.  ``"1.心虚胆怯证：心悸，善惊易恐"`` -> ``"心虚胆怯证"``.
    """
    text = strip_enumeration(str(name).strip())
    match = _SYNDROME_DESC_RE.match(text)
    if match:
        text = match.group(1)
    text = re.split(r"[:：]", text)[0]
    return strip_trailing_etc(text).strip(" ,.;，。；、")


def syndrome_atoms(name: str) -> List[str]:
    """Split a compound syndrome into comparable atoms.

    ``"痰阻血瘀，湿郁化热证"`` -> ``["痰阻血瘀证", "湿郁化热证"]``.  Atoms are what
    the SDT scorer awards partial credit over, because Chinese syndrome names
    compose additively and an answer that recovers one of two conjuncts is
    genuinely closer than one that recovers neither.
    """
    canonical = canonical_syndrome(name)
    parts = [p.strip() for p in _SYNDROME_SPLIT_RE.split(canonical) if p.strip()]
    if not parts:
        return []
    atoms: List[str] = []
    for part in parts:
        part = part.strip(" ,.;，。；、")
        if not part:
            continue
        if not part.endswith("证") and len(parts) > 1:
            part = part + "证"
        atoms.append(part)
    seen: set = set()
    return [a for a in atoms if not (a in seen or seen.add(a))]


# --------------------------------------------------------------------------- #
# Character n-grams -- the tokenisation used by the lexical index.
# --------------------------------------------------------------------------- #

_CJK_RE = re.compile(r"[一-鿿]")
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+")


def char_ngrams(text: str, sizes: Sequence[int] = (1, 2, 3)) -> List[str]:
    """Tokenise Chinese text into character n-grams plus ASCII word tokens.

    Character n-grams avoid a segmenter dependency: they are deterministic,
    need no model download, and handle the graph's heavily compounded clinical
    vocabulary (``舌质暗红``, ``苔黄腻``) better than a general-purpose segmenter
    that was not trained on TCM text.
    """
    normalized = normalize_text(text)
    if not normalized:
        return []
    tokens: List[str] = list(_ASCII_TOKEN_RE.findall(normalized))
    cjk_runs = re.findall(r"[一-鿿]+", normalized)
    for run in cjk_runs:
        for size in sizes:
            if size > len(run):
                if size == min(sizes):
                    tokens.append(run)
                continue
            for i in range(len(run) - size + 1):
                tokens.append(run[i : i + size])
    return tokens


def has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def dedupe(items: Iterable[str]) -> List[str]:
    seen: set = set()
    return [x for x in items if x and not (x in seen or seen.add(x))]
