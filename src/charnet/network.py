"""Graph construction: scenes → weighted interaction graphs."""
from __future__ import annotations

import itertools
import logging
from typing import Any, Optional

import networkx as nx

from charnet.models import Utterance, Scene, SceneGraph, EdgeData

logger = logging.getLogger(__name__)


def _adjacency_count(speakers_seq: list[str]) -> dict[tuple[str, str], float]:
    """Count how many times speaker A speaks immediately after speaker B."""
    counts: dict[tuple[str, str], float] = {}
    for i in range(1, len(speakers_seq)):
        a, b = speakers_seq[i - 1], speakers_seq[i]
        if a != b and a and b:
            key = tuple(sorted([a, b]))
            counts[key] = counts.get(key, 0.0) + 1.0
    return counts


def _proximity_scores(
    speakers_seq: list[str], window: int = 3
) -> dict[tuple[str, str], float]:
    """Compute proximity score: sum of 1/distance for pairs within window."""
    scores: dict[tuple[str, str], float] = {}
    for i, spk_a in enumerate(speakers_seq):
        if not spk_a:
            continue
        for j in range(i + 1, min(i + window + 1, len(speakers_seq))):
            spk_b = speakers_seq[j]
            if not spk_b or spk_b == spk_a:
                continue
            key = tuple(sorted([spk_a, spk_b]))
            dist = j - i
            scores[key] = scores.get(key, 0.0) + 1.0 / dist
    return scores


def build_scene_graph(
    scene: Scene,
    utterances: list[Utterance],
    weight_adjacency: float = 1.0,
    weight_proximity: float = 0.5,
    weight_copresence: float = 0.25,
    proximity_window: int = 3,
) -> SceneGraph:
    """Build a weighted interaction graph for a single scene."""
    # Get utterances for this scene in order
    scene_utts = sorted(
        [u for u in utterances if u.index in set(scene.utterance_indices)],
        key=lambda u: u.start,
    )

    speakers_seq = [u.speaker for u in scene_utts if u.speaker]
    unique_speakers = scene.speakers  # already computed

    adj = _adjacency_count(speakers_seq)
    prox = _proximity_scores(speakers_seq, window=proximity_window)

    # All pairs of unique speakers
    all_pairs = set()
    for a, b in itertools.combinations(sorted(unique_speakers), 2):
        all_pairs.add((a, b))
    # Also add pairs from adjacency/proximity even if not in unique_speakers list
    for key in list(adj.keys()) + list(prox.keys()):
        all_pairs.add(key)

    edges: list[EdgeData] = []
    for pair in all_pairs:
        a, b = pair
        adj_val = adj.get(pair, 0.0)
        prox_val = prox.get(pair, 0.0)
        cop_val = 1.0 if (a in unique_speakers and b in unique_speakers) else 0.0

        w = weight_adjacency * adj_val + weight_proximity * prox_val + weight_copresence * cop_val
        if w > 0:
            edges.append(EdgeData(
                source=a, target=b,
                weight=round(w, 4),
                adjacency=adj_val,
                proximity=round(prox_val, 4),
                copresence=cop_val,
            ))

    return SceneGraph(
        scene_id=scene.scene_id,
        start=scene.start,
        end=scene.end,
        nodes=list(unique_speakers),
        edges=edges,
    )


def build_temporal_network(
    scenes: list[Scene],
    utterances: list[Utterance],
    weight_adjacency: float = 1.0,
    weight_proximity: float = 0.5,
    weight_copresence: float = 0.25,
    proximity_window: int = 3,
) -> list[SceneGraph]:
    """Build per-scene graphs for all scenes."""
    graphs: list[SceneGraph] = []
    for scene in scenes:
        g = build_scene_graph(
            scene, utterances,
            weight_adjacency=weight_adjacency,
            weight_proximity=weight_proximity,
            weight_copresence=weight_copresence,
            proximity_window=proximity_window,
        )
        graphs.append(g)
    logger.info("Built temporal network: %d scene graphs", len(graphs))
    return graphs


def to_networkx(scene_graph: SceneGraph) -> nx.Graph:
    """Convert a SceneGraph to a NetworkX graph."""
    G = nx.Graph()
    G.add_nodes_from(scene_graph.nodes)
    for e in scene_graph.edges:
        G.add_edge(e.source, e.target, weight=e.weight,
                   adjacency=e.adjacency, proximity=e.proximity, copresence=e.copresence)
    return G


def aggregate_episode_graph(scene_graphs: list[SceneGraph]) -> nx.Graph:
    """Aggregate all scene graphs into a single episode-level graph."""
    G = nx.Graph()
    for sg in scene_graphs:
        G.add_nodes_from(sg.nodes)
        for e in sg.edges:
            if G.has_edge(e.source, e.target):
                G[e.source][e.target]["weight"] += e.weight
                G[e.source][e.target]["adjacency"] += e.adjacency
                G[e.source][e.target]["proximity"] += e.proximity
                G[e.source][e.target]["copresence"] += e.copresence
            else:
                G.add_edge(e.source, e.target, weight=e.weight,
                           adjacency=e.adjacency, proximity=e.proximity,
                           copresence=e.copresence)
    return G


def rolling_window_aggregation(
    scene_graphs: list[SceneGraph], window_seconds: float = 300.0
) -> list[dict]:
    """Produce rolling-window aggregated graphs.

    Returns list of dicts with keys: window_start, window_end, graph (nx.Graph).
    """
    if not scene_graphs:
        return []

    episode_start = scene_graphs[0].start
    episode_end = scene_graphs[-1].end
    results = []

    t = episode_start
    while t < episode_end:
        t_end = t + window_seconds
        window_graphs = [sg for sg in scene_graphs if sg.start < t_end and sg.end > t]
        G = aggregate_episode_graph(window_graphs)
        results.append({"window_start": t, "window_end": t_end, "graph": G})
        t += window_seconds

    return results
