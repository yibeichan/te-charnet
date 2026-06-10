# tests/test_verify_stage2_network.py
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import verify_stage2_network as v  # noqa: E402


def _write_table(path: Path, rows: list[dict]) -> None:
    cols = ["scene_id", "start", "end", "speaker"]
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "")) for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_table_rows_filters_and_coerces(tmp_path):
    p = tmp_path / "t.tsv"
    _write_table(p, [
        {"scene_id": "1", "start": "0", "end": "1", "speaker": "A"},   # keep
        {"scene_id": "1", "start": "", "end": "2", "speaker": "B"},    # drop: empty start (scene-marker)
        {"scene_id": "", "start": "3", "end": "4", "speaker": "C"},    # drop: empty scene_id
        {"scene_id": "2", "start": "5", "end": "6", "speaker": ""},    # drop: empty speaker
        {"scene_id": "2", "start": "x", "end": "7", "speaker": "D"},   # drop: non-numeric start
    ])
    rows = v.read_table_rows(p)
    assert rows == [{"scene_id": 1, "start": 0.0, "end": 1.0, "speaker": "A"}]


def test_read_table_rows_blank_cells_are_not_nan(tmp_path):
    # Regression: default pd.read_csv would turn the blank speaker into NaN,
    # str(NaN)=="nan" (non-empty) -> bogus row survives. Must be dropped.
    p = tmp_path / "t.tsv"
    _write_table(p, [{"scene_id": "1", "start": "0", "end": "1", "speaker": ""}])
    assert v.read_table_rows(p) == []


def test_read_table_rows_missing_column_raises(tmp_path):
    p = tmp_path / "bad.tsv"
    # header missing the 'speaker' column entirely
    p.write_text("scene_id\tstart\tend\n1\t0\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        v.read_table_rows(p)
