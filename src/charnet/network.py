"""Graph construction: scenes → weighted interaction graphs."""
from __future__ import annotations

import itertools
import logging
from typing import Any

import networkx as nx

from charnet.models import SceneGraph, EdgeData

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


def build_temporal_network_from_aligned_rows(
    aligned_rows: list[dict[str, Any]],
    weight_adjacency: float = 1.0,
    weight_proximity: float = 0.5,
    weight_copresence: float = 0.25,
    proximity_window: int = 3,
) -> list[SceneGraph]:
    """Build per-scene graphs using stage-1 aligned rows with scene_id labels."""
    grouped: dict[int, list[tuple[float, float, str]]] = {}

    for row in aligned_rows:
        scene_id_raw = str(row.get("scene_id", "")).strip()
        speaker = str(row.get("speaker", "")).strip()
        if not scene_id_raw or not speaker:
            continue
        try:
            scene_id = int(float(scene_id_raw))
            start = float(row.get("start", ""))
            end = float(row.get("end", ""))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(scene_id, []).append((start, end, speaker))

    scene_graphs: list[SceneGraph] = []
    for scene_id in sorted(grouped):
        turns = sorted(grouped[scene_id], key=lambda x: x[0])
        if not turns:
            continue

        scene_start = turns[0][0]
        scene_end = max(t[1] for t in turns)
        speakers_seq = [t[2] for t in turns]
        unique_speakers = sorted(set(speakers_seq))

        adj = _adjacency_count(speakers_seq)
        prox = _proximity_scores(speakers_seq, window=proximity_window)

        all_pairs = set(itertools.combinations(unique_speakers, 2))
        for key in list(adj.keys()) + list(prox.keys()):
            all_pairs.add(key)

        edges: list[EdgeData] = []
        for pair in sorted(all_pairs):
            a, b = pair
            adj_val = adj.get(pair, 0.0)
            prox_val = prox.get(pair, 0.0)
            cop_val = 1.0 if (a in unique_speakers and b in unique_speakers) else 0.0
            w = weight_adjacency * adj_val + weight_proximity * prox_val + weight_copresence * cop_val
            if w > 0:
                edges.append(
                    EdgeData(
                        source=a,
                        target=b,
                        weight=round(w, 4),
                        adjacency=adj_val,
                        proximity=round(prox_val, 4),
                        copresence=cop_val,
                    )
                )

        scene_graphs.append(
            SceneGraph(
                scene_id=scene_id,
                start=scene_start,
                end=scene_end,
                nodes=unique_speakers,
                edges=edges,
            )
        )

    logger.info("Built temporal network from aligned rows: %d scene graphs", len(scene_graphs))
    return scene_graphs


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
