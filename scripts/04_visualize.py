#!/usr/bin/env python3
"""Stage 4: Visualize scene/episode networks, scene segmentation, and network metrics."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from charnet.io import load_temporal_network
from charnet.metrics import centrality_timeseries, episode_graph_from_dict
from charnet.network import to_networkx
from charnet.viz import (
    export_gexf,
    export_json,
    plot_centrality_timeseries,
    plot_episode_graph,
    plot_interaction_heatmap,
    plot_scene_timeline,
)

logger = logging.getLogger(__name__)
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")


def _safe_close(fig: plt.Figure) -> None:
    if fig is not None:
        plt.close(fig)


def _plot_scene_duration_bars(per_scene_df: pd.DataFrame, output_path: Path) -> None:
    if per_scene_df.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(per_scene_df["scene_id"], per_scene_df["duration"], color="#4C72B0")
    ax.set_xlabel("Scene ID")
    ax.set_ylabel("Duration (s)")
    ax.set_title("Scene Segment Durations")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _safe_close(fig)


def _plot_per_scene_metrics(per_scene_df: pd.DataFrame, output_path: Path) -> None:
    if per_scene_df.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(per_scene_df["scene_id"], per_scene_df["n_nodes"], marker="o", color="#1f77b4")
    axes[0].set_ylabel("n_nodes")
    axes[0].set_title("Per-Scene Network Metrics")

    axes[1].plot(per_scene_df["scene_id"], per_scene_df["n_edges"], marker="o", color="#2ca02c")
    axes[1].set_ylabel("n_edges")

    axes[2].plot(per_scene_df["scene_id"], per_scene_df["density"], marker="o", color="#d62728")
    if "interaction_density" in per_scene_df.columns:
        axes[2].plot(
            per_scene_df["scene_id"],
            per_scene_df["interaction_density"],
            marker="o",
            linestyle="--",
            color="#9467bd",
            label="interaction_density",
        )
        axes[2].legend(loc="best")
    axes[2].set_ylabel("density")
    axes[2].set_xlabel("scene_id")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _safe_close(fig)


def _plot_episode_centrality_bars(metrics_data: dict, output_path: Path) -> None:
    centralities = metrics_data.get("episode", {}).get("centralities", {})
    degree = centralities.get("degree", {})
    if not degree:
        return
    items = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(names, vals, color="#9467bd")
    ax.set_ylabel("Degree centrality")
    ax.set_title("Episode Centrality Ranking (Degree)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _safe_close(fig)


def _plot_edge_birth_death(edge_df: pd.DataFrame, output_path: Path) -> None:
    if edge_df.empty:
        return
    edge_df = edge_df.copy()
    edge_df["pair"] = edge_df["char_a"] + " - " + edge_df["char_b"]
    edge_df = edge_df.sort_values(["first_scene", "last_scene", "pair"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, max(4, 0.3 * len(edge_df) + 1)))
    y = range(len(edge_df))
    ax.hlines(y, edge_df["first_scene"], edge_df["last_scene"], color="#7f7f7f", linewidth=2)
    ax.scatter(edge_df["first_scene"], y, color="#1f77b4", s=24, label="first")
    ax.scatter(edge_df["last_scene"], y, color="#d62728", s=24, label="last")
    ax.set_yticks(list(y))
    ax.set_yticklabels(edge_df["pair"])
    ax.set_xlabel("Scene ID")
    ax.set_title("Edge Birth/Death by Scene")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _safe_close(fig)


@click.command()
@click.option("--episode", "-e", required=True)
@click.option("--network-dir", default=None)
@click.option("--analyze-dir", default=None)
@click.option("--output-dir", "-o", default=None)
@click.option("--layout", default="spring", show_default=True,
              type=click.Choice(["spring", "kamada_kawai", "circular"]))
@click.option("--export-formats", default="png,gexf,json", show_default=True,
              help="Comma-separated export formats.")
@click.option("--max-scene-plots", default=None, type=int,
              help="Optional cap on number of per-scene network plots.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(episode, network_dir, analyze_dir, output_dir, layout, export_formats, max_scene_plots, verbose):
    """Generate all requested visualizations."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    if network_dir is None:
        network_dir = Path(SCRATCH_DIR) / "output" / "02_build_network" / episode
    else:
        network_dir = Path(network_dir)

    if analyze_dir is None:
        analyze_dir = Path(SCRATCH_DIR) / "output" / "03_analyze" / episode
    else:
        analyze_dir = Path(analyze_dir)

    if output_dir is None:
        output_dir = Path(SCRATCH_DIR) / "output" / "04_visualize" / episode
    else:
        output_dir = Path(output_dir)

    temporal_network_path = network_dir / "temporal_network.json"
    episode_network_path = network_dir / "episode_network.json"
    if not temporal_network_path.exists():
        raise click.ClickException(f"temporal_network.json not found: {temporal_network_path}")
    if not episode_network_path.exists():
        raise click.ClickException(f"episode_network.json not found: {episode_network_path}")

    scene_graphs = load_temporal_network(temporal_network_path)
    with open(episode_network_path, "r", encoding="utf-8") as f:
        episode_network = json.load(f)
    G_episode = episode_graph_from_dict(episode_network)

    formats = [f.strip() for f in export_formats.split(",") if f.strip()]
    do_png = "png" in formats

    # Optional analysis inputs
    metrics_data: dict = {}
    metrics_path = analyze_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    community_map = metrics_data.get("episode", {}).get("community_map")
    if community_map:
        community_map = {k: int(v) for k, v in community_map.items()}

    ts_path = analyze_dir / "centrality_timeseries.csv"
    if ts_path.exists():
        ts_df = pd.read_csv(ts_path)
    else:
        ts_df = centrality_timeseries(scene_graphs)

    edge_bd_path = analyze_dir / "edge_birth_death.csv"
    edge_bd_df = pd.read_csv(edge_bd_path) if edge_bd_path.exists() else pd.DataFrame()

    per_scene_df = pd.DataFrame(metrics_data.get("per_scene", []))

    # Output folders by requested plot type
    figures_dir = output_dir / "figures"
    scene_networks_dir = figures_dir / "scene_networks"
    episode_dir = figures_dir / "episode"
    segments_dir = figures_dir / "scene_segments"
    metrics_dir = figures_dir / "metrics"
    for p in [scene_networks_dir, episode_dir, segments_dir, metrics_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # 1) Network per scene
    scene_graphs_sorted = sorted(scene_graphs, key=lambda sg: sg.scene_id)
    if max_scene_plots is not None:
        scene_graphs_sorted = scene_graphs_sorted[:max_scene_plots]
    if do_png:
        for sg in scene_graphs_sorted:
            G_scene = to_networkx(sg)
            if G_scene.number_of_nodes() == 0:
                continue
            try:
                cent = nx.eigenvector_centrality(G_scene, weight="weight", max_iter=1000)
            except Exception:
                cent = nx.degree_centrality(G_scene)
            fig = plot_episode_graph(
                G_scene,
                centrality=cent,
                community_map=community_map,
                layout=layout,
                title=f"Scene Network — {episode} — scene {sg.scene_id}",
                output_path=scene_networks_dir / f"scene_{sg.scene_id:04d}.png",
            )
            _safe_close(fig)

    # 2) Aggregated network per episode
    if do_png:
        try:
            centrality = nx.eigenvector_centrality(G_episode, weight="weight", max_iter=1000)
        except Exception:
            centrality = nx.degree_centrality(G_episode) if G_episode.number_of_nodes() else {}

        fig = plot_episode_graph(
            G_episode,
            centrality=centrality,
            community_map=community_map,
            layout=layout,
            title=f"Episode Network — {episode}",
            output_path=episode_dir / "episode_network.png",
        )
        _safe_close(fig)

        fig = plot_interaction_heatmap(
            G_episode,
            title=f"Interaction Heatmap — {episode}",
            output_path=episode_dir / "interaction_heatmap.png",
        )
        _safe_close(fig)

    if "gexf" in formats:
        export_gexf(G_episode, episode_dir / f"{episode}_network.gexf")
    if "json" in formats:
        export_json(G_episode, episode_dir / f"{episode}_network.json")

    # 3) Scene segment plots
    if do_png:
        fig = plot_scene_timeline(
            scene_graphs,
            community_map=community_map,
            title=f"Scene Timeline — {episode}",
            output_path=segments_dir / "scene_timeline.png",
        )
        _safe_close(fig)
        if not per_scene_df.empty and {"scene_id", "duration"}.issubset(per_scene_df.columns):
            _plot_scene_duration_bars(per_scene_df, segments_dir / "scene_durations.png")

    # 4) Network metrics
    if do_png:
        if not ts_df.empty and "degree" in ts_df.columns:
            fig = plot_centrality_timeseries(
                ts_df,
                measure="degree",
                title=f"Degree Centrality Over Scenes — {episode}",
                output_path=metrics_dir / "centrality_timeseries_degree.png",
            )
            _safe_close(fig)
        if not per_scene_df.empty and {"scene_id", "n_nodes", "n_edges", "density"}.issubset(
            per_scene_df.columns
        ):
            _plot_per_scene_metrics(per_scene_df, metrics_dir / "per_scene_network_metrics.png")
        if metrics_data:
            _plot_episode_centrality_bars(metrics_data, metrics_dir / "episode_centrality_ranking.png")
        if not edge_bd_df.empty and {"char_a", "char_b", "first_scene", "last_scene"}.issubset(edge_bd_df.columns):
            _plot_edge_birth_death(edge_bd_df, metrics_dir / "edge_birth_death.png")

    logger.info("Visualization complete. Output: %s", figures_dir)
    click.echo(f"Figures written to: {figures_dir}")


if __name__ == "__main__":
    main()
