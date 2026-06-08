# tests/test_export_network_metrics.py
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import export_network_metrics as E  # noqa: E402
from charnet.io import save_temporal_network  # noqa: E402
from charnet.models import EdgeData, SceneGraph  # noqa: E402


def nx_exp_columns():
    import export_network_metrics as _E
    return _E.nx_exp.SCENE_NETWORK_COLUMNS


def _write_network(network_root: Path, dirname: str, ep_scene_id=1):
    sg = SceneGraph(
        scene_id=ep_scene_id, start=0.0, end=10.0,
        nodes=["A", "B", "C"],
        edges=[
            EdgeData(source="A", target="B", weight=2.0, adjacency=1.0, proximity=0.0, copresence=0.0),
            EdgeData(source="B", target="C", weight=1.0, adjacency=1.0, proximity=0.0, copresence=0.0),
        ],
    )
    path = network_root / dirname / "temporal_network.json"
    save_temporal_network([sg], path)
    return path


def test_resolve_network_path_prefers_friends_prefix(tmp_path):
    root = tmp_path / "02_build_network"
    _write_network(root, "friends_s01e01a")
    resolved = E.resolve_network_path(root, "s01e01a")
    assert resolved == root / "friends_s01e01a" / "temporal_network.json"


def test_resolve_network_path_falls_back_to_bare(tmp_path):
    root = tmp_path / "02_build_network"
    _write_network(root, "s01e01a")
    resolved = E.resolve_network_path(root, "s01e01a")
    assert resolved == root / "s01e01a" / "temporal_network.json"


def test_resolve_network_path_missing_returns_none(tmp_path):
    root = tmp_path / "02_build_network"
    assert E.resolve_network_path(root, "s09e99z") is None


def _make_scenes_dir(tmp_path):
    # so expand_episode_spec can resolve the episode id
    scenes = tmp_path / "scenes" / "s1"
    scenes.mkdir(parents=True)
    (scenes / "friends_s01e01a_scene_summary.tsv").write_text("scene_id\tstart\tend\n")
    return tmp_path / "scenes"


def _run_main(monkeypatch, tmp_path, episodes, network_root, out_dir):
    scenes_in = _make_scenes_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "export_network_metrics.py", "--episodes", episodes,
        "--scenes-in", str(scenes_in),
        "--network-root", str(network_root),
        "--out-dir", str(out_dir),
        "--measures", "degree,betweenness",
    ])
    E.main()


def test_main_writes_tsvs_and_sidecars(tmp_path, monkeypatch):
    root = tmp_path / "02_build_network"
    _write_network(root, "friends_s01e01a")
    out_dir = tmp_path / "network_metrics"
    _run_main(monkeypatch, tmp_path, "s01e01a", root, out_dir)

    scene_tsv = out_dir / "s1" / "friends_s01e01a_scene_network.tsv"
    char_tsv = out_dir / "s1" / "friends_s01e01a_character_centrality.tsv"
    assert scene_tsv.exists() and char_tsv.exists()

    sdf = pd.read_csv(scene_tsv, sep="\t")
    assert list(sdf.columns) == nx_exp_columns()
    cdf = pd.read_csv(char_tsv, sep="\t")
    assert list(cdf.columns) == ["scene_id", "start", "end", "character",
                                 "degree", "betweenness"]

    assert (out_dir / "scene_network.json").exists()
    assert (out_dir / "character_centrality.json").exists()
    desc = json.loads((out_dir.parent / "dataset_description.json").read_text())
    assert desc["DatasetType"] == "derivative"


def test_main_explicit_missing_episode_errors(tmp_path, monkeypatch):
    root = tmp_path / "02_build_network"  # empty: no episode dirs
    out_dir = tmp_path / "network_metrics"
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, tmp_path, "s01e01a", root, out_dir)
