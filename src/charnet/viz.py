"""Visualization: static plots and network export."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd

from charnet.models import SceneGraph
from charnet.network import to_networkx

logger = logging.getLogger(__name__)

# Color palette for characters
_PALETTE = list(mcolors.TABLEAU_COLORS.values())


def _node_sizes(G: nx.Graph, centrality: dict[str, float], scale: float = 3000.0) -> list[float]:
    vals = [centrality.get(n, 0.1) for n in G.nodes()]
    if max(vals) > 0:
        vals = [v / max(vals) * scale for v in vals]
    return [max(v, 100) for v in vals]


def _edge_widths(G: nx.Graph, scale: float = 5.0) -> list[float]:
    weights = [d.get("weight", 1.0) for _, _, d in G.edges(data=True)]
    if not weights:
        return []
    max_w = max(weights) or 1.0
    return [w / max_w * scale for w in weights]


def plot_episode_graph(
    G: nx.Graph,
    centrality: Optional[dict[str, float]] = None,
    community_map: Optional[dict[str, int]] = None,
    layout: str = "spring",
    title: str = "Episode Interaction Network",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot the full-episode interaction network."""
    fig, ax = plt.subplots(figsize=(12, 10))

    if layout == "spring":
        pos = nx.spring_layout(G, weight="weight", seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G, weight="weight")
    else:
        pos = nx.circular_layout(G)

    # Node colors by community
    node_colors = [
        _PALETTE[community_map.get(n, 0) % len(_PALETTE)]
        if community_map else "#4C72B0"
        for n in G.nodes()
    ]

    # Node sizes by centrality
    if centrality is None:
        centrality = nx.degree_centrality(G)
    sizes = _node_sizes(G, centrality)
    widths = _edge_widths(G)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=sizes, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
    if widths:
        nx.draw_networkx_edges(G, pos, width=widths, ax=ax, alpha=0.6, edge_color="#888888")
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, edge_color="#888888")

    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved episode graph to %s", output_path)

    return fig


def plot_interaction_heatmap(
    G: nx.Graph,
    title: str = "Character Interaction Heatmap",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot a heatmap of pairwise interaction weights."""
    import seaborn as sns

    nodes = sorted(G.nodes())
    n = len(nodes)
    matrix = np.zeros((n, n))
    idx = {node: i for i, node in enumerate(nodes)}

    for u, v, d in G.edges(data=True):
        w = d.get("weight", 0)
        matrix[idx[u], idx[v]] = w
        matrix[idx[v], idx[u]] = w

    fig, ax = plt.subplots(figsize=(max(8, n), max(7, n - 1)))
    sns.heatmap(
        matrix, xticklabels=nodes, yticklabels=nodes,
        annot=n <= 15, fmt=".1f", cmap="Blues", ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved heatmap to %s", output_path)

    return fig


def plot_centrality_timeseries(
    df: pd.DataFrame,
    measure: str = "degree",
    title: str = "Centrality Over Time",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot centrality over time for each character."""
    if df.empty or measure not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    characters = df["character"].unique()
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, char in enumerate(sorted(characters)):
        sub = df[df["character"] == char].sort_values("start")
        ax.plot(sub["start"] / 60, sub[measure], label=char,
                marker="o", markersize=3, linewidth=1.5,
                color=_PALETTE[i % len(_PALETTE)])

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(measure.replace("_", " ").title())
    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved centrality timeseries to %s", output_path)

    return fig


def plot_scene_timeline(
    scene_graphs: list[SceneGraph],
    community_map: Optional[dict[str, int]] = None,
    title: str = "Scene Timeline",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot a horizontal strip showing scene boundaries and dominant characters."""
    fig, ax = plt.subplots(figsize=(16, 3))

    for sg in scene_graphs:
        dominant = None
        G = to_networkx(sg)
        if G.number_of_nodes() > 0:
            deg = nx.degree_centrality(G)
            dominant = max(deg, key=deg.get)

        color_idx = 0
        if dominant and community_map:
            color_idx = community_map.get(dominant, 0)

        color = _PALETTE[color_idx % len(_PALETTE)]
        ax.barh(0, (sg.end - sg.start) / 60.0, left=sg.start / 60.0, height=0.8,
                color=color, alpha=0.7, edgecolor="white")

        if dominant:
            mid = (sg.start + sg.end) / 2 / 60
            ax.text(mid, 0, dominant[:4], ha="center", va="center", fontsize=6, color="white", fontweight="bold")

    ax.set_xlabel("Time (minutes)")
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved scene timeline to %s", output_path)

    return fig


def export_gexf(G: nx.Graph, path: Path) -> None:
    """Export graph to GEXF format for Gephi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, str(path))
    logger.info("Saved GEXF to %s", path)


def export_json(G: nx.Graph, path: Path) -> None:
    """Export graph to JSON (D3-compatible node-link format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(G)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved JSON graph to %s", path)
