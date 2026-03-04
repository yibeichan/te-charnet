#!/usr/bin/env python3
"""Stage 4: Visualize — generate static plots and export graph files."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from charnet.io import load_temporal_network
from charnet.metrics import episode_metrics, centrality_timeseries
from charnet.network import aggregate_episode_graph
from charnet.viz import (
    plot_episode_graph, plot_interaction_heatmap,
    plot_centrality_timeseries, plot_scene_timeline,
    export_gexf, export_json,
)

logger = logging.getLogger(__name__)
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")


@click.command()
@click.option("--episode", "-e", required=True)
@click.option("--network-dir", default=None)
@click.option("--analyze-dir", default=None)
@click.option("--output-dir", "-o", default=None)
@click.option("--layout", default="spring", show_default=True,
              type=click.Choice(["spring", "kamada_kawai", "circular"]))
@click.option("--export-formats", default="png,gexf,json", show_default=True,
              help="Comma-separated export formats.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(episode, network_dir, analyze_dir, output_dir, layout, export_formats, verbose):
    """Generate visualizations for the episode interaction network."""
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

    network_path = network_dir / "temporal_network.json"
    if not network_path.exists():
        raise click.ClickException(f"temporal_network.json not found: {network_path}")

    scene_graphs = load_temporal_network(network_path)
    G_episode = aggregate_episode_graph(scene_graphs)

    formats = [f.strip() for f in export_formats.split(",")]

    # Load analysis results if available
    community_map = None
    metrics_path = analyze_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics_data = json.load(f)
        community_map = metrics_data.get("episode", {}).get("community_map")
        # community_map keys might be strings; values are community IDs
        if community_map:
            community_map = {k: int(v) for k, v in community_map.items()}

    ts_df = centrality_timeseries(scene_graphs)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Episode network graph
    import networkx as nx
    centrality = nx.eigenvector_centrality(G_episode, weight="weight") if G_episode.number_of_nodes() > 0 else {}
    try:
        centrality = nx.eigenvector_centrality(G_episode, weight="weight", max_iter=1000)
    except Exception:
        centrality = nx.degree_centrality(G_episode)

    fig = plot_episode_graph(
        G_episode, centrality=centrality, community_map=community_map,
        layout=layout, title=f"Episode Network — {episode}",
        output_path=figures_dir / "episode_network.png" if "png" in formats else None,
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

    # 2. Interaction heatmap
    fig = plot_interaction_heatmap(
        G_episode, title=f"Interaction Heatmap — {episode}",
        output_path=figures_dir / "heatmap.png" if "png" in formats else None,
    )
    plt.close(fig)

    # 3. Centrality time series
    fig = plot_centrality_timeseries(
        ts_df, measure="degree", title=f"Degree Centrality — {episode}",
        output_path=figures_dir / "centrality_timeseries.png" if "png" in formats else None,
    )
    plt.close(fig)

    # 4. Scene timeline
    fig = plot_scene_timeline(
        scene_graphs, community_map=community_map,
        title=f"Scene Timeline — {episode}",
        output_path=figures_dir / "scene_timeline.png" if "png" in formats else None,
    )
    plt.close(fig)

    # Export graph formats
    if "gexf" in formats:
        export_gexf(G_episode, figures_dir / f"{episode}_network.gexf")
    if "json" in formats:
        export_json(G_episode, figures_dir / f"{episode}_network.json")

    logger.info("Visualization complete. Output: %s", figures_dir)
    click.echo(f"Figures written to: {figures_dir}")


if __name__ == "__main__":
    main()
