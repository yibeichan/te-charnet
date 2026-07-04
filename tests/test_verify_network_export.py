"""Tests for the network-export verifier itself.

A verifier with no test can silently stop catching regressions. These build a
real stage-2 temporal_network.json, generate the export TSVs through the actual
export code path (charnet.network_export), and confirm the independent
recomputation in verify_network_export agrees — then that it FAILS on a
perturbed value, a dropped character row, and reports the expected skips.
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import verify_network_export as V  # noqa: E402
from charnet import network_export as nx_exp  # noqa: E402
from charnet.io import load_temporal_network, save_temporal_network  # noqa: E402
from charnet.models import EdgeData, SceneGraph  # noqa: E402

MEASURES = ["degree", "betweenness", "eigenvector"]


def _scene(scene_id, start):
    return SceneGraph(
        scene_id=scene_id, start=start, end=start + 10.0,
        nodes=["A", "B", "C"],
        edges=[
            EdgeData(source="A", target="B", weight=2.0, adjacency=1.0, proximity=0.0, copresence=0.0),
            EdgeData(source="B", target="C", weight=1.0, adjacency=1.0, proximity=0.0, copresence=0.0),
        ],
    )


def _build_fixture(tmp_path, ep="s01e01a", n_scenes=2):
    """Real temporal_network.json + export TSVs produced by the export code path."""
    network_root = tmp_path / "02_build_network"
    njson = network_root / f"friends_{ep}" / "temporal_network.json"
    scene_graphs = [_scene(i + 1, 10.0 * i) for i in range(n_scenes)]
    save_temporal_network(scene_graphs, njson)

    loaded = load_temporal_network(njson)
    export = tmp_path / "network_metrics" / "s1"
    export.mkdir(parents=True)
    nx_exp.scene_network_trace(loaded).to_csv(
        export / f"friends_{ep}_scene_network.tsv", sep="\t", index=False)
    nx_exp.character_centrality_trace(loaded, measures=MEASURES).to_csv(
        export / f"friends_{ep}_character_centrality.tsv", sep="\t", index=False)
    return tmp_path / "network_metrics", network_root


def _run(export_dir, network_root, extra=()):
    return V.main(["--export-dir", str(export_dir),
                   "--network-root", str(network_root), *extra])


def test_clean_fixture_exits_zero(tmp_path):
    assert _run(*_build_fixture(tmp_path)) == 0


def test_perturbed_scene_value_fails(tmp_path):
    export_dir, network_root = _build_fixture(tmp_path)
    tsv = export_dir / "s1" / "friends_s01e01a_scene_network.tsv"
    df = pd.read_csv(tsv, sep="\t")
    df.loc[0, "density"] = df.loc[0, "density"] + 0.5
    df.to_csv(tsv, sep="\t", index=False)
    assert _run(export_dir, network_root) == 1


def test_perturbed_centrality_value_fails(tmp_path):
    export_dir, network_root = _build_fixture(tmp_path)
    tsv = export_dir / "s1" / "friends_s01e01a_character_centrality.tsv"
    df = pd.read_csv(tsv, sep="\t")
    df.loc[0, "degree"] = df.loc[0, "degree"] + 0.5
    df.to_csv(tsv, sep="\t", index=False)
    assert _run(export_dir, network_root) == 1


def test_dropped_character_row_fails_completeness(tmp_path):
    # a silently missing character row must fail, not pass as "present rows matched"
    export_dir, network_root = _build_fixture(tmp_path)
    tsv = export_dir / "s1" / "friends_s01e01a_character_centrality.tsv"
    df = pd.read_csv(tsv, sep="\t")
    df = df.drop(index=df.index[0]).reset_index(drop=True)
    df.to_csv(tsv, sep="\t", index=False)
    assert _run(export_dir, network_root) == 1


def test_extra_scene_row_fails(tmp_path):
    # a scene row with no counterpart in the JSON must fail
    export_dir, network_root = _build_fixture(tmp_path)
    tsv = export_dir / "s1" / "friends_s01e01a_scene_network.tsv"
    df = pd.read_csv(tsv, sep="\t")
    ghost = df.iloc[[0]].copy()
    ghost.loc[:, "scene_id"] = 999
    pd.concat([df, ghost], ignore_index=True).to_csv(tsv, sep="\t", index=False)
    assert _run(export_dir, network_root) == 1


def test_missing_network_json_is_skip(tmp_path):
    export_dir, network_root = _build_fixture(tmp_path)
    # remove the network json -> episode unverifiable -> skipped; nothing else to check -> exit 2
    (network_root / "friends_s01e01a" / "temporal_network.json").unlink()
    assert _run(export_dir, network_root) == 2


def test_missing_character_tsv_is_skip(tmp_path):
    export_dir, network_root = _build_fixture(tmp_path)
    (export_dir / "s1" / "friends_s01e01a_character_centrality.tsv").unlink()
    assert _run(export_dir, network_root) == 2


def test_no_export_tsvs_exits_two(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _run(empty, tmp_path / "net") == 2
