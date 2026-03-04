"""Tests for charnet.network — graph construction."""
from __future__ import annotations

import pytest
import networkx as nx

from charnet.models import Utterance, Scene, SceneGraph, EdgeData
from charnet.network import (
    build_scene_graph, build_temporal_network,
    to_networkx, aggregate_episode_graph,
    _adjacency_count, _proximity_scores,
)


class TestAdjacencyCount:
    def test_counts_turns(self):
        seq = ["A", "B", "A", "B"]
        counts = _adjacency_count(seq)
        key = tuple(sorted(["A", "B"]))
        assert counts[key] == 3  # A->B, B->A, A->B

    def test_same_speaker_not_counted(self):
        seq = ["A", "A", "B"]
        counts = _adjacency_count(seq)
        # A->A should not appear
        assert ("A", "A") not in counts

    def test_empty_sequence(self):
        assert _adjacency_count([]) == {}


class TestProximityScores:
    def test_closer_pairs_score_higher(self):
        seq = ["A", "B", "C"]
        scores = _proximity_scores(seq, window=2)
        key_ab = tuple(sorted(["A", "B"]))
        key_ac = tuple(sorted(["A", "C"]))
        # A-B distance 1, A-C distance 2
        assert scores.get(key_ab, 0) > scores.get(key_ac, 0)

    def test_empty_sequence(self):
        assert _proximity_scores([]) == {}


class TestBuildSceneGraph:
    def test_builds_graph(self, sample_scene, sample_utterances):
        sg = build_scene_graph(sample_scene, sample_utterances)
        assert isinstance(sg, SceneGraph)
        assert "Monica" in sg.nodes
        assert "Ross" in sg.nodes

    def test_edges_have_positive_weight(self, sample_scene, sample_utterances):
        sg = build_scene_graph(sample_scene, sample_utterances)
        for e in sg.edges:
            assert e.weight > 0

    def test_empty_scene_returns_empty_graph(self):
        sc = Scene(
            scene_id=0, start=0.0, end=10.0, speakers=[],
            n_shots=0, n_utterances=0, utterance_indices=[],
        )
        sg = build_scene_graph(sc, [])
        assert sg.nodes == []
        assert sg.edges == []


class TestToNetworkx:
    def test_converts_correctly(self, sample_scene, sample_utterances):
        sg = build_scene_graph(sample_scene, sample_utterances)
        G = to_networkx(sg)
        assert isinstance(G, nx.Graph)
        assert "Monica" in G.nodes
        assert G.number_of_edges() >= 1

    def test_edge_weights_preserved(self, sample_scene, sample_utterances):
        sg = build_scene_graph(sample_scene, sample_utterances)
        G = to_networkx(sg)
        for u, v, d in G.edges(data=True):
            assert d["weight"] > 0


class TestAggregateEpisodeGraph:
    def test_aggregates_weights(self, sample_scene, sample_utterances):
        sg1 = build_scene_graph(sample_scene, sample_utterances)
        sg2 = build_scene_graph(sample_scene, sample_utterances)
        G = aggregate_episode_graph([sg1, sg2])
        # Weights should be doubled compared to single scene
        sg_single = build_scene_graph(sample_scene, sample_utterances)
        G_single = to_networkx(sg_single)
        for u, v in G.edges():
            assert G[u][v]["weight"] == pytest.approx(G_single[u][v]["weight"] * 2)

    def test_empty_list(self):
        G = aggregate_episode_graph([])
        assert G.number_of_nodes() == 0


class TestBuildTemporalNetwork:
    def test_returns_one_graph_per_scene(self, sample_scene, sample_utterances):
        scenes = [sample_scene]
        graphs = build_temporal_network(scenes, sample_utterances)
        assert len(graphs) == 1
        assert isinstance(graphs[0], SceneGraph)
