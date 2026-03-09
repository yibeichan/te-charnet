"""Tests for charnet.metrics helper functions."""
from __future__ import annotations

import networkx as nx

from charnet.metrics import (
    betweenness_centrality,
    degree_centrality,
    degree_unweighted_centrality,
    episode_graph_from_dict,
    episode_metrics_from_graph,
)


def test_episode_graph_from_dict_builds_graph():
    payload = {
        "nodes": ["A", "B"],
        "edges": [
            {"source": "A", "target": "B", "weight": 2.5, "adjacency": 2, "proximity": 1.0, "copresence": 1}
        ],
    }
    G = episode_graph_from_dict(payload)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1
    assert G["A"]["B"]["weight"] == 2.5


def test_episode_metrics_from_graph_basic():
    G = nx.Graph()
    G.add_edge("A", "B", weight=1.0, adjacency=1.0, proximity=0.0, copresence=1.0)
    metrics = episode_metrics_from_graph(G, centrality_measures=["degree"], community_method="girvan_newman")
    assert metrics["n_nodes"] == 2
    assert metrics["n_edges"] == 1
    assert "degree" in metrics["centralities"]


def test_degree_centrality_is_weighted_strength_share():
    G = nx.Graph()
    G.add_edge("A", "B", weight=4.0)
    G.add_edge("A", "C", weight=1.0)
    G.add_edge("B", "C", weight=1.0)

    cent = degree_centrality(G)
    assert abs(sum(cent.values()) - 1.0) < 1e-9
    assert cent["A"] > cent["C"]
    assert cent["B"] > cent["C"]


def test_degree_unweighted_singleton_is_zero():
    G = nx.Graph()
    G.add_node("A")
    cent = degree_unweighted_centrality(G)
    assert cent == {"A": 0.0}


def test_betweenness_uses_inverse_weight_distance():
    G = nx.Graph()
    # Strong A-B-C path and weak direct A-C tie: B should mediate shortest path.
    G.add_edge("A", "B", weight=5.0)
    G.add_edge("B", "C", weight=5.0)
    G.add_edge("A", "C", weight=0.1)

    cent = betweenness_centrality(G)
    assert cent["B"] > 0.0
