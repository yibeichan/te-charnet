"""Network analysis: centrality, community detection, temporal dynamics."""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

import networkx as nx
import pandas as pd

from charnet.models import SceneGraph
from charnet.network import to_networkx

logger = logging.getLogger(__name__)


def degree_centrality(G: nx.Graph) -> dict[str, float]:
    return nx.degree_centrality(G)


def betweenness_centrality(G: nx.Graph) -> dict[str, float]:
    return nx.betweenness_centrality(G, weight="weight")


def eigenvector_centrality(G: nx.Graph) -> dict[str, float]:
    if G.number_of_nodes() == 0:
        return {}
    try:
        return nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        logger.warning("Eigenvector centrality did not converge — using degree centrality as fallback")
        return degree_centrality(G)


def compute_centralities(
    G: nx.Graph, measures: list[str]
) -> dict[str, dict[str, float]]:
    """Compute requested centrality measures."""
    results: dict[str, dict[str, float]] = {}
    dispatch = {
        "degree": degree_centrality,
        "betweenness": betweenness_centrality,
        "eigenvector": eigenvector_centrality,
    }
    for m in measures:
        if m in dispatch:
            results[m] = dispatch[m](G)
        else:
            logger.warning("Unknown centrality measure: %s", m)
    return results


def scene_metrics(scene_graph: SceneGraph) -> dict[str, Any]:
    """Compute per-scene metrics."""
    G = to_networkx(scene_graph)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    components = nx.number_connected_components(G)

    result: dict[str, Any] = {
        "scene_id": scene_graph.scene_id,
        "start": scene_graph.start,
        "end": scene_graph.end,
        "duration": scene_graph.end - scene_graph.start,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "n_components": components,
    }

    # Scene-level entropy (how distributed is speech)
    weights = [e.weight for e in scene_graph.edges]
    if weights:
        total = sum(weights)
        probs = [w / total for w in weights]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    else:
        entropy = 0.0
    result["interaction_entropy"] = entropy

    return result


def episode_graph_from_dict(episode_network: dict[str, Any]) -> nx.Graph:
    """Build a NetworkX graph from stage-2 episode_network.json content."""
    G = nx.Graph()
    for node in episode_network.get("nodes", []):
        G.add_node(node)
    for edge in episode_network.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            continue
        G.add_edge(
            source,
            target,
            weight=float(edge.get("weight", 0.0)),
            adjacency=float(edge.get("adjacency", 0.0)),
            proximity=float(edge.get("proximity", 0.0)),
            copresence=float(edge.get("copresence", 0.0)),
        )
    return G


def episode_metrics_from_graph(
    G: nx.Graph,
    centrality_measures: list[str] | None = None,
    community_method: str = "louvain",
) -> dict[str, Any]:
    """Compute episode-level aggregate metrics from an episode graph."""
    if centrality_measures is None:
        centrality_measures = ["degree", "betweenness", "eigenvector"]

    centralities = compute_centralities(G, centrality_measures)

    # Community detection
    communities: Optional[list[set]] = None
    community_map: dict[str, int] = {}
    if community_method == "louvain":
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G, weight="weight")
            community_map = partition
            # Group into sets
            n_comm = max(partition.values()) + 1 if partition else 0
            communities = [set() for _ in range(n_comm)]
            for node, c in partition.items():
                communities[c].add(node)
        except ImportError:
            logger.warning("python-louvain not installed — skipping community detection")
    elif community_method == "girvan_newman":
        if G.number_of_edges() > 0:
            comp = nx.community.girvan_newman(G)
            communities_tuple = next(comp)
            communities = [set(c) for c in communities_tuple]
            for i, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = i

    # Edge weight distribution
    edge_weights = [d["weight"] for _, _, d in G.edges(data=True)]

    # Character pair stats
    pair_stats = []
    for u, v, d in G.edges(data=True):
        pair_stats.append({
            "char_a": u,
            "char_b": v,
            "total_weight": d.get("weight", 0),
            "adjacency": d.get("adjacency", 0),
            "proximity": d.get("proximity", 0),
        })

    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "centralities": centralities,
        "community_map": community_map,
        "n_communities": len(communities) if communities else 0,
        "edge_weights": edge_weights,
        "pair_stats": pair_stats,
    }


def centrality_timeseries(
    scene_graphs: list[SceneGraph],
    measures: list[str] | None = None,
) -> pd.DataFrame:
    """Compute centrality time series across scenes.

    Returns DataFrame with columns: scene_id, start, end, character, <measure1>, <measure2>, ...
    """
    if measures is None:
        measures = ["degree", "betweenness", "eigenvector"]

    rows = []
    for sg in scene_graphs:
        G = to_networkx(sg)
        if G.number_of_nodes() == 0:
            continue
        cents = compute_centralities(G, measures)
        for node in G.nodes():
            row: dict[str, Any] = {
                "scene_id": sg.scene_id,
                "start": sg.start,
                "end": sg.end,
                "character": node,
            }
            for m in measures:
                row[m] = cents.get(m, {}).get(node, 0.0)
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def edge_birth_death(scene_graphs: list[SceneGraph]) -> pd.DataFrame:
    """Track when each character pair first and last interacts."""
    first_seen: dict[tuple[str, str], int] = {}
    last_seen: dict[tuple[str, str], int] = {}

    for sg in scene_graphs:
        for e in sg.edges:
            key = tuple(sorted([e.source, e.target]))
            if key not in first_seen:
                first_seen[key] = sg.scene_id
            last_seen[key] = sg.scene_id

    rows = [
        {
            "char_a": k[0],
            "char_b": k[1],
            "first_scene": first_seen[k],
            "last_scene": last_seen[k],
        }
        for k in first_seen
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame()
