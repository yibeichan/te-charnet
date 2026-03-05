"""Tests for charnet.metrics helper functions."""
from __future__ import annotations

import networkx as nx

from charnet.metrics import episode_graph_from_dict, episode_metrics_from_graph


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
