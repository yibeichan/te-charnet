# tests/test_network_export.py
import sys
from pathlib import Path

import pytest  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from charnet.models import EdgeData, SceneGraph  # noqa: E402
from charnet import network_export as nx_exp  # noqa: E402


def _scene(scene_id, start, end):
    return SceneGraph(
        scene_id=scene_id, start=start, end=end,
        nodes=["A", "B", "C"],
        edges=[
            EdgeData(source="A", target="B", weight=2.0, adjacency=1.0, proximity=0.0, copresence=0.0),
            EdgeData(source="B", target="C", weight=1.0, adjacency=1.0, proximity=0.0, copresence=0.0),
        ],
    )


def test_scene_network_trace_columns_and_rows():
    df = nx_exp.scene_network_trace([_scene(1, 0.0, 10.0), _scene(2, 10.0, 20.0)])
    assert list(df.columns) == nx_exp.SCENE_NETWORK_COLUMNS
    assert len(df) == 2
    row = df.iloc[0]
    assert row["scene_id"] == 1
    assert row["start"] == 0.0 and row["end"] == 10.0
    assert row["n_nodes"] == 3 and row["n_edges"] == 2
    assert row["n_components"] == 1
    assert 0.0 < row["density"] <= 1.0
    assert row["interaction_entropy"] >= 0.0


def test_scene_network_trace_empty_keeps_columns():
    df = nx_exp.scene_network_trace([])
    assert list(df.columns) == nx_exp.SCENE_NETWORK_COLUMNS
    assert len(df) == 0


def test_character_centrality_trace_columns_and_rows():
    df = nx_exp.character_centrality_trace([_scene(1, 0.0, 10.0)], measures=["degree"])
    assert list(df.columns) == ["scene_id", "start", "end", "character", "degree"]
    assert sorted(df["character"]) == ["A", "B", "C"]
    assert (df["scene_id"] == 1).all()


def test_character_centrality_trace_multi_measure_order():
    df = nx_exp.character_centrality_trace([_scene(1, 0.0, 10.0)],
                                           measures=["betweenness", "degree"])
    assert list(df.columns) == ["scene_id", "start", "end", "character",
                                "betweenness", "degree"]


def test_character_centrality_trace_empty_keeps_columns():
    df = nx_exp.character_centrality_trace([], measures=["degree", "eigenvector"])
    assert list(df.columns) == ["scene_id", "start", "end", "character",
                                "degree", "eigenvector"]
    assert len(df) == 0


def test_character_centrality_trace_rejects_unknown_measure():
    with pytest.raises(ValueError, match="unknown"):
        nx_exp.character_centrality_trace([_scene(1, 0.0, 10.0)], measures=["bogus"])


def test_data_dictionaries_cover_columns():
    # every column is described; coverage guards silent schema drift
    for col in nx_exp.SCENE_NETWORK_COLUMNS:
        assert col in nx_exp.SCENE_NETWORK_DD
    for col in nx_exp.CHARACTER_CENTRALITY_BASE_COLUMNS:
        assert col in nx_exp.CHARACTER_CENTRALITY_DD
    # start/end documented as network-coverage windows, not full scene spans
    assert "coverage" in nx_exp.SCENE_NETWORK_DD["start"]["Description"].lower()
