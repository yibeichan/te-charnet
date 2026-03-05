"""Tests for charnet.network — current aligned-rows based graph construction."""
from __future__ import annotations

import pytest
import networkx as nx

from charnet.models import SceneGraph, EdgeData
from charnet.network import (
    _adjacency_count,
    _proximity_scores,
    aggregate_episode_graph,
    build_temporal_network_from_aligned_rows,
    to_networkx,
)


class TestAdjacencyCount:
    def test_counts_turns(self):
        seq = ["A", "B", "A", "B"]
        counts = _adjacency_count(seq)
        key = tuple(sorted(["A", "B"]))
        assert counts[key] == 3

    def test_same_speaker_not_counted(self):
        seq = ["A", "A", "B"]
        counts = _adjacency_count(seq)
        assert ("A", "A") not in counts


class TestProximityScores:
    def test_closer_pairs_score_higher(self):
        seq = ["A", "B", "C"]
        scores = _proximity_scores(seq, window=2)
        key_ab = tuple(sorted(["A", "B"]))
        key_ac = tuple(sorted(["A", "C"]))
        assert scores.get(key_ab, 0) > scores.get(key_ac, 0)


class TestBuildTemporalNetworkFromAlignedRows:
    def test_builds_scene_graphs_from_scene_id(self):
        rows = [
            {
                "start": "0.0",
                "end": "1.0",
                "scene_id": "1",
                "speaker": "Monica",
                "utterance": "Hi",
                "scene_desc": "",
            },
            {
                "start": "1.1",
                "end": "2.0",
                "scene_id": "1",
                "speaker": "Ross",
                "utterance": "Hello",
                "scene_desc": "",
            },
            {
                "start": "",
                "end": "",
                "scene_id": "2",
                "speaker": "",
                "utterance": "",
                "scene_desc": "New scene",
            },
            {
                "start": "3.0",
                "end": "4.0",
                "scene_id": "2",
                "speaker": "Rachel",
                "utterance": "Hey",
                "scene_desc": "",
            },
        ]
        graphs = build_temporal_network_from_aligned_rows(rows)
        assert len(graphs) == 2
        assert graphs[0].scene_id == 1
        assert graphs[1].scene_id == 2
        assert "Monica" in graphs[0].nodes
        assert "Rachel" in graphs[1].nodes


class TestToNetworkx:
    def test_converts_correctly(self):
        sg = SceneGraph(
            scene_id=1,
            start=0.0,
            end=5.0,
            nodes=["A", "B"],
            edges=[EdgeData("A", "B", 2.0, 1.0, 1.5, 1.0)],
        )
        G = to_networkx(sg)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1
        assert G["A"]["B"]["weight"] == 2.0


class TestAggregateEpisodeGraph:
    def test_aggregates_weights(self):
        sg1 = SceneGraph(
            scene_id=1,
            start=0.0,
            end=5.0,
            nodes=["A", "B"],
            edges=[EdgeData("A", "B", 2.0, 1.0, 1.5, 1.0)],
        )
        sg2 = SceneGraph(
            scene_id=2,
            start=5.0,
            end=10.0,
            nodes=["A", "B"],
            edges=[EdgeData("A", "B", 3.0, 2.0, 1.0, 1.0)],
        )
        G = aggregate_episode_graph([sg1, sg2])
        assert G["A"]["B"]["weight"] == pytest.approx(5.0)
        assert G["A"]["B"]["adjacency"] == pytest.approx(3.0)
