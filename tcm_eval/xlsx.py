"""Minimal read-only XLSX reader built on the standard library.

TCMEval-PA is distributed as a workbook. Rather than take a hard dependency on
openpyxl or pandas -- which would break the promise that this harness installs
and runs from a clean Python years from now -- an ``.xlsx`` is just a zip of
XML, and the subset needed to read a flat sheet of strings is small.

Handles the two things that actually bite: the shared-string table (Excel
stores repeated text once and cells reference it by index) and sparse rows
(Excel omits empty cells entirely, so column position must come from the cell
reference, not from enumeration order).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_COL_RE = re.compile(r"([A-Z]+)")


def _column_index(reference: str) -> int:
    """``"AB12"`` -> 27 (zero-based column index)."""
    match = _COL_RE.match(reference or "")
    if not match:
        return 0
    index = 0
    for char in match.group(1):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _shared_strings(archive: zipfile.ZipFile) -> List[str]:
    try:
        payload = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(payload)
    return [
        "".join(node.text or "" for node in si.iter(f"{_NS}t"))
        for si in root.findall(f"{_NS}si")
    ]


def read_sheet(path: str | Path, sheet: str = "xl/worksheets/sheet1.xml") -> List[List[str]]:
    """Return the sheet as a dense list of rows of strings."""
    with zipfile.ZipFile(Path(path)) as archive:
        strings = _shared_strings(archive)
        root = ET.fromstring(archive.read(sheet))

    rows: List[List[str]] = []
    for row in root.iter(f"{_NS}row"):
        cells: Dict[int, str] = {}
        for cell in row.findall(f"{_NS}c"):
            index = _column_index(cell.get("r", ""))
            kind = cell.get("t")
            value_node = cell.find(f"{_NS}v")
            inline = cell.find(f"{_NS}is")
            if kind == "s" and value_node is not None:
                position = int(value_node.text or 0)
                value = strings[position] if 0 <= position < len(strings) else ""
            elif inline is not None:
                value = "".join(node.text or "" for node in inline.iter(f"{_NS}t"))
            else:
                value = value_node.text if value_node is not None else ""
            cells[index] = (value or "").strip()
        width = max(cells) + 1 if cells else 0
        rows.append([cells.get(i, "") for i in range(width)])
    return rows


def read_records(path: str | Path, sheet: str = "xl/worksheets/sheet1.xml") -> List[Dict[str, str]]:
    """Read a sheet whose first row is a header into a list of dicts."""
    rows = read_sheet(path, sheet)
    if not rows:
        return []
    header = [h or f"col_{i}" for i, h in enumerate(rows[0])]
    records: List[Dict[str, str]] = []
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        record = {header[i]: (row[i] if i < len(row) else "") for i in range(len(header))}
        records.append(record)
    return records
