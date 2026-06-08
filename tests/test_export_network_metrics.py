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
