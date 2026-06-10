# tests/test_verify_stage2_network.py
from __future__ import annotations

import json
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


def test_aggregate_episode_sums_rounded_scene_values():
    # Two identical [A,C,D,B] scenes. Per-scene A-B proximity stored 0.3333.
    # Correct aggregate sums ROUNDED scene values: 0.3333+0.3333 -> 0.6666.
    # Raw-summed-once would give round(1/3+1/3,4)=0.6667. Verifier must give 0.6666.
    def scene_rows(scene_id, t0):
        return [
            {"scene_id": scene_id, "start": float(t0 + i), "end": float(t0 + i + 1), "speaker": s}
            for i, s in enumerate(["A", "C", "D", "B"])
        ]
    rows = scene_rows(1, 0) + scene_rows(2, 100)
    scenes = v.reconstruct_scenes(rows, v.DEFAULTS)
    epi = v.aggregate_episode(scenes)
    assert epi["n_scenes"] == 2
    assert epi["start"] == 0.0 and epi["end"] == 104.0
    assert epi["nodes"] == ["A", "B", "C", "D"]
    assert epi["edges"][("A", "B")]["proximity"] == 0.6666   # NOT 0.6667
    assert epi["edges"][("A", "B")]["weight"] == 0.8334
    assert epi["edges"][("A", "B")]["copresence"] == 2.0


def test_aggregate_episode_empty():
    epi = v.aggregate_episode([])
    assert epi == {"start": 0.0, "end": 0.0, "n_scenes": 0, "nodes": [], "edges": {}}


def test_aggregate_episode_partial_overlap_pair():
    # Scene 1 has an A-B edge; scene 2 has only A-C (A and B both appear as nodes
    # somewhere, but A-B exists in just one scene). A-B must reflect scene 1 only.
    rows = [
        # scene 1: A,B,A -> A-B edge (adjacency 2, proximity 2.0, weight 3.25)
        {"scene_id": 1, "start": 0.0, "end": 1.0, "speaker": "A"},
        {"scene_id": 1, "start": 1.0, "end": 2.0, "speaker": "B"},
        {"scene_id": 1, "start": 2.0, "end": 3.0, "speaker": "A"},
        # scene 2: A,C,A -> A-C edge only (no B at all here)
        {"scene_id": 2, "start": 10.0, "end": 11.0, "speaker": "A"},
        {"scene_id": 2, "start": 11.0, "end": 12.0, "speaker": "C"},
        {"scene_id": 2, "start": 12.0, "end": 13.0, "speaker": "A"},
    ]
    scenes = v.reconstruct_scenes(rows, v.DEFAULTS)
    epi = v.aggregate_episode(scenes)
    assert epi["nodes"] == ["A", "B", "C"]
    # A-B only in scene 1 -> not doubled
    assert epi["edges"][("A", "B")] == {"weight": 3.25, "adjacency": 2.0, "proximity": 2.0, "copresence": 1.0}
    # A-C only in scene 2 -> present
    assert ("A", "C") in epi["edges"]
    # B-C never co-present -> absent
    assert ("B", "C") not in epi["edges"]


def _build_fixture(tmp_path):
    """Hand-built table + matching committed JSON (NOT generated by charnet)."""
    rows = [
        {"scene_id": "1", "start": "0", "end": "1", "speaker": "A"},
        {"scene_id": "1", "start": "1", "end": "2", "speaker": "B"},
        {"scene_id": "1", "start": "2", "end": "3", "speaker": "A"},
        {"scene_id": "2", "start": "10", "end": "11", "speaker": "A"},  # one-speaker scene
    ]
    table = tmp_path / "friends_s00e00a_sentence_speaker_table.tsv"
    _write_table(table, rows)
    net_dir = tmp_path / "friends_s00e00a"
    net_dir.mkdir()
    temporal = [
        {"scene_id": 1, "start": 0.0, "end": 3.0, "nodes": ["A", "B"],
         "edges": [{"source": "A", "target": "B", "weight": 3.25,
                    "adjacency": 2.0, "proximity": 2.0, "copresence": 1.0}]},
        {"scene_id": 2, "start": 10.0, "end": 11.0, "nodes": ["A"], "edges": []},
    ]
    episode = {"episode": "friends_s00e00a", "start": 0.0, "end": 11.0, "n_scenes": 2,
               "nodes": ["A", "B"],
               "edges": [{"source": "A", "target": "B", "weight": 3.25,
                          "adjacency": 2.0, "proximity": 2.0, "copresence": 1.0}]}
    (net_dir / "temporal_network.json").write_text(json.dumps(temporal), encoding="utf-8")
    (net_dir / "episode_network.json").write_text(json.dumps(episode), encoding="utf-8")
    return table, net_dir / "temporal_network.json", net_dir / "episode_network.json"


def test_check_episode_clean_fixture_passes(tmp_path):
    table, temporal, episode = _build_fixture(tmp_path)
    fails = v.check_episode("s00e00a", table, temporal, episode, v.DEFAULTS, tol=1e-6)
    assert fails == [], fails


def test_check_episode_perturbed_weight_fails(tmp_path):
    table, temporal, episode = _build_fixture(tmp_path)
    data = json.loads(temporal.read_text())
    data[0]["edges"][0]["weight"] = 9.99
    temporal.write_text(json.dumps(data), encoding="utf-8")
    fails = v.check_episode("s00e00a", table, temporal, episode, v.DEFAULTS, tol=1e-6)
    assert any("weight" in f for f in fails)


def test_check_episode_dropped_edge_fails(tmp_path):
    table, temporal, episode = _build_fixture(tmp_path)
    data = json.loads(temporal.read_text())
    data[0]["edges"] = []
    temporal.write_text(json.dumps(data), encoding="utf-8")
    fails = v.check_episode("s00e00a", table, temporal, episode, v.DEFAULTS, tol=1e-6)
    assert any("MISSING from committed JSON" in f for f in fails), fails


def test_check_episode_corrupted_aggregate_fails(tmp_path):
    table, temporal, episode = _build_fixture(tmp_path)
    data = json.loads(episode.read_text())
    data["edges"][0]["proximity"] = 1.234
    episode.write_text(json.dumps(data), encoding="utf-8")
    fails = v.check_episode("s00e00a", table, temporal, episode, v.DEFAULTS, tol=1e-6)
    assert any("episode" in f.lower() for f in fails)


def test_check_episode_extra_committed_scene_fails(tmp_path):
    table, temporal, episode = _build_fixture(tmp_path)
    data = json.loads(temporal.read_text())
    # add a scene that the table (and thus reconstruction) does not contain
    data.append({"scene_id": 99, "start": 50.0, "end": 51.0, "nodes": ["A"], "edges": []})
    temporal.write_text(json.dumps(data), encoding="utf-8")
    fails = v.check_episode("s00e00a", table, temporal, episode, v.DEFAULTS, tol=1e-6)
    assert any("99" in f and "not reconstructed" in f for f in fails), fails


def _build_tables_root(tmp_path):
    """tables-root/<season>/<ep>_..._table.tsv + network-root/friends_<ep>/*.json."""
    tables_root = tmp_path / "sentences"
    season = tables_root / "s0"
    season.mkdir(parents=True)
    rows = [
        {"scene_id": "1", "start": "0", "end": "1", "speaker": "A"},
        {"scene_id": "1", "start": "1", "end": "2", "speaker": "B"},
        {"scene_id": "1", "start": "2", "end": "3", "speaker": "A"},
    ]
    _write_table(season / "friends_s00e00a_sentence_speaker_table.tsv", rows)
    net_root = tmp_path / "02_build_network"
    net_dir = net_root / "friends_s00e00a"
    net_dir.mkdir(parents=True)
    temporal = [{"scene_id": 1, "start": 0.0, "end": 3.0, "nodes": ["A", "B"],
                 "edges": [{"source": "A", "target": "B", "weight": 3.25,
                            "adjacency": 2.0, "proximity": 2.0, "copresence": 1.0}]}]
    episode = {"episode": "friends_s00e00a", "start": 0.0, "end": 3.0, "n_scenes": 1,
               "nodes": ["A", "B"],
               "edges": [{"source": "A", "target": "B", "weight": 3.25,
                          "adjacency": 2.0, "proximity": 2.0, "copresence": 1.0}]}
    (net_dir / "temporal_network.json").write_text(json.dumps(temporal), encoding="utf-8")
    (net_dir / "episode_network.json").write_text(json.dumps(episode), encoding="utf-8")
    return tables_root, net_root


def test_main_exit_0_on_clean(tmp_path):
    tables_root, net_root = _build_tables_root(tmp_path)
    rc = v.main(["--tables-root", str(tables_root), "--network-root", str(net_root)])
    assert rc == 0


def test_main_exit_1_on_mismatch(tmp_path):
    tables_root, net_root = _build_tables_root(tmp_path)
    p = net_root / "friends_s00e00a" / "temporal_network.json"
    data = json.loads(p.read_text())
    data[0]["edges"][0]["weight"] = 9.99
    p.write_text(json.dumps(data), encoding="utf-8")
    rc = v.main(["--tables-root", str(tables_root), "--network-root", str(net_root)])
    assert rc == 1


def test_main_exit_2_when_nothing_checkable(tmp_path):
    tables_root, _ = _build_tables_root(tmp_path)
    empty_net = tmp_path / "empty_net"
    empty_net.mkdir()
    rc = v.main(["--tables-root", str(tables_root), "--network-root", str(empty_net)])
    assert rc == 2   # table found but stage-2 absent -> skipped -> nothing checked
