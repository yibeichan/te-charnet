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


def _run_main_with_measures(monkeypatch, tmp_path, episodes, network_root, out_dir, measures):
    scenes_in = _make_scenes_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "export_network_metrics.py", "--episodes", episodes,
        "--scenes-in", str(scenes_in),
        "--network-root", str(network_root),
        "--out-dir", str(out_dir),
        "--measures", measures,
    ])
    E.main()


def _run_main(monkeypatch, tmp_path, episodes, network_root, out_dir):
    _run_main_with_measures(monkeypatch, tmp_path, episodes, network_root, out_dir, "degree,betweenness")


def test_main_writes_tsvs_and_sidecars(tmp_path, monkeypatch):
    root = tmp_path / "02_build_network"
    _write_network(root, "friends_s01e01a")
    out_dir = tmp_path / "network_metrics"
    _run_main(monkeypatch, tmp_path, "s01e01a", root, out_dir)

    scene_tsv = out_dir / "s1" / "friends_s01e01a_scene_network.tsv"
    char_tsv = out_dir / "s1" / "friends_s01e01a_character_centrality.tsv"
    assert scene_tsv.exists() and char_tsv.exists()

    sdf = pd.read_csv(scene_tsv, sep="\t")
    assert list(sdf.columns) == E.nx_exp.SCENE_NETWORK_COLUMNS
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
    with pytest.raises(SystemExit, match=r"s01e01a"):
        _run_main(monkeypatch, tmp_path, "s01e01a", root, out_dir)


def test_main_resolves_bare_episode_dir(tmp_path, monkeypatch):
    root = tmp_path / "02_build_network"
    _write_network(root, "s01e01a")  # bare dir, not friends_-prefixed
    out_dir = tmp_path / "network_metrics"
    _run_main(monkeypatch, tmp_path, "s01e01a", root, out_dir)
    assert (out_dir / "s1" / "friends_s01e01a_scene_network.tsv").exists()
    assert (out_dir / "s1" / "friends_s01e01a_character_centrality.tsv").exists()


def _write_multiscene_network(network_root: Path, dirname: str):
    scenes = []
    for sid, start in [(1, 0.0), (2, 10.0)]:
        scenes.append(SceneGraph(
            scene_id=sid, start=start, end=start + 5.0,
            nodes=["A", "B", "C"],
            edges=[
                EdgeData(source="A", target="B", weight=2.0, adjacency=1.0, proximity=0.0, copresence=0.0),
                EdgeData(source="B", target="C", weight=1.0, adjacency=1.0, proximity=0.0, copresence=0.0),
            ],
        ))
    path = network_root / dirname / "temporal_network.json"
    save_temporal_network(scenes, path)
    return path


def test_main_multiscene_row_counts(tmp_path, monkeypatch):
    root = tmp_path / "02_build_network"
    _write_multiscene_network(root, "friends_s01e01a")
    out_dir = tmp_path / "network_metrics"
    _run_main(monkeypatch, tmp_path, "s01e01a", root, out_dir)
    sdf = pd.read_csv(out_dir / "s1" / "friends_s01e01a_scene_network.tsv", sep="\t")
    cdf = pd.read_csv(out_dir / "s1" / "friends_s01e01a_character_centrality.tsv", sep="\t")
    assert len(sdf) == 2          # 2 scenes
    assert len(cdf) == 6          # 2 scenes x 3 characters
    assert sorted(sdf["scene_id"]) == [1, 2]


def test_main_season_spec_skips_missing_without_error(tmp_path, monkeypatch):
    # two episodes discoverable by expand_episode_spec, only one has a network dir
    scenes = tmp_path / "scenes" / "s1"
    scenes.mkdir(parents=True)
    (scenes / "friends_s01e01a_scene_summary.tsv").write_text("scene_id\tstart\tend\n")
    (scenes / "friends_s01e01b_scene_summary.tsv").write_text("scene_id\tstart\tend\n")
    root = tmp_path / "02_build_network"
    _write_network(root, "friends_s01e01a")  # only e01a present; e01b missing
    out_dir = tmp_path / "network_metrics"
    monkeypatch.setattr(sys, "argv", [
        "export_network_metrics.py", "--episodes", "s1",
        "--scenes-in", str(tmp_path / "scenes"),
        "--network-root", str(root),
        "--out-dir", str(out_dir),
        "--measures", "degree,betweenness",
    ])
    E.main()  # must NOT raise — season spec, partial coverage is allowed
    assert (out_dir / "s1" / "friends_s01e01a_scene_network.tsv").exists()
    assert not (out_dir / "s1" / "friends_s01e01b_scene_network.tsv").exists()


def test_main_zero_written_exits(tmp_path, monkeypatch):
    # non-explicit ALL spec but no network dirs at all -> zero written -> nonzero exit,
    # so a bad --network-root / SCRATCH_DIR cannot masquerade as a successful empty run
    root = tmp_path / "02_build_network"  # never created -> no episode resolves
    out_dir = tmp_path / "network_metrics"
    with pytest.raises(SystemExit, match="0 episodes written"):
        _run_main(monkeypatch, tmp_path, "ALL", root, out_dir)


def test_main_invalid_measure_raises(tmp_path, monkeypatch):
    # an unsupported --measures value surfaces as an error, not a silent empty column
    root = tmp_path / "02_build_network"
    _write_network(root, "friends_s01e01a")
    out_dir = tmp_path / "network_metrics"
    scenes_in = _make_scenes_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "export_network_metrics.py", "--episodes", "s01e01a",
        "--scenes-in", str(scenes_in),
        "--network-root", str(root),
        "--out-dir", str(out_dir),
        "--measures", "bogus",
    ])
    with pytest.raises(ValueError, match="unknown"):
        E.main()


def test_main_invalid_measure_validated_up_front(tmp_path, monkeypatch):
    # bad --measures must raise even when NO episode resolves (validation precedes
    # the loop) — not silently fall through to the "0 episodes written" exit — and
    # must fail before any sidecar is written
    root = tmp_path / "02_build_network"  # empty: nothing resolves
    out_dir = tmp_path / "network_metrics"
    with pytest.raises(ValueError, match="unknown"):
        _run_main_with_measures(monkeypatch, tmp_path, "ALL", root, out_dir, "bogus")
    assert not (out_dir / "scene_network.json").exists()
    assert not (out_dir.parent / "dataset_description.json").exists()
