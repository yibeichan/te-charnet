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


def test_reconstruct_scenes_clean_fixture():
    # Scene 1: A,B,A -> adjacency(A,B)=2, proximity(A,B)=2.0, cop=1, weight=3.25
    # Scene 2: single speaker A -> one node, no edges
    rows = [
        {"scene_id": 1, "start": 0.0, "end": 1.0, "speaker": "A"},
        {"scene_id": 1, "start": 1.0, "end": 2.0, "speaker": "B"},
        {"scene_id": 1, "start": 2.0, "end": 3.0, "speaker": "A"},
        {"scene_id": 2, "start": 10.0, "end": 11.0, "speaker": "A"},
    ]
    scenes = v.reconstruct_scenes(rows, v.DEFAULTS)
    assert len(scenes) == 2
    s1, s2 = scenes
    assert s1["scene_id"] == 1 and s1["start"] == 0.0 and s1["end"] == 3.0
    assert s1["nodes"] == ["A", "B"]
    assert s1["edges"] == {
        ("A", "B"): {"weight": 3.25, "adjacency": 2.0, "proximity": 2.0, "copresence": 1.0}
    }
    assert s2["scene_id"] == 2 and s2["nodes"] == ["A"] and s2["edges"] == {}


def test_reconstruct_scene_proximity_rounds_to_4dp():
    # speakers_seq [A,C,D,B]: A-B at distance 3 -> proximity 1/3 -> stored 0.3333
    rows = [
        {"scene_id": 1, "start": float(i), "end": float(i + 1), "speaker": s}
        for i, s in enumerate(["A", "C", "D", "B"])
    ]
    scenes = v.reconstruct_scenes(rows, v.DEFAULTS)
    ab = scenes[0]["edges"][("A", "B")]
    assert ab["proximity"] == 0.3333
    assert ab["adjacency"] == 0.0
    assert ab["weight"] == 0.4167   # round(0 + 0.5*(1/3) + 0.25, 4)
