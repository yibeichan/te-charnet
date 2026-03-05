#!/usr/bin/env python3
"""Stage 3: Analyze — compute network metrics and centrality time series."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from charnet.io import load_temporal_network
from charnet.metrics import (
    centrality_timeseries,
    edge_birth_death,
    episode_graph_from_dict,
    episode_metrics_from_graph,
    scene_metrics,
)

logger = logging.getLogger(__name__)
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")


@click.command()
@click.option("--episode", "-e", required=True)
@click.option("--network-dir", default=None)
@click.option("--output-dir", "-o", default=None)
@click.option("--community-method", default="louvain", show_default=True,
              type=click.Choice(["louvain", "girvan_newman"]))
@click.option("--centrality-measures", default="degree,betweenness,eigenvector", show_default=True,
              help="Comma-separated list of centrality measures.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(episode, network_dir, output_dir, community_method, centrality_measures, verbose):
    """Compute network metrics at scene and episode level."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    if network_dir is None:
        network_dir = Path(SCRATCH_DIR) / "output" / "02_build_network" / episode
    else:
        network_dir = Path(network_dir)

    if output_dir is None:
        output_dir = Path(SCRATCH_DIR) / "output" / "03_analyze" / episode
    else:
        output_dir = Path(output_dir)

    temporal_network_path = network_dir / "temporal_network.json"
    episode_network_path = network_dir / "episode_network.json"
    if not temporal_network_path.exists():
        raise click.ClickException(f"temporal_network.json not found: {temporal_network_path}")
    if not episode_network_path.exists():
        raise click.ClickException(
            f"episode_network.json not found: {episode_network_path}. "
            "Run 02_build_network.py (aligned-rows mode) first."
        )

    scene_graphs = load_temporal_network(temporal_network_path)
    with open(episode_network_path, "r", encoding="utf-8") as f:
        episode_network_dict = json.load(f)
    episode_graph = episode_graph_from_dict(episode_network_dict)
    measures = [m.strip() for m in centrality_measures.split(",")]

    # Per-scene metrics
    per_scene = [scene_metrics(sg) for sg in scene_graphs]

    # Episode metrics
    ep_metrics = episode_metrics_from_graph(
        episode_graph,
        centrality_measures=measures,
        community_method=community_method,
    )

    # Centrality time series
    ts_df = centrality_timeseries(scene_graphs, measures=measures)

    # Edge birth/death
    bd_df = edge_birth_death(scene_graphs)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics.json
    metrics_out = {
        "episode_id": episode,
        "per_scene": per_scene,
        "episode": {
            "start": episode_network_dict.get("start"),
            "end": episode_network_dict.get("end"),
            "n_scenes": episode_network_dict.get("n_scenes"),
            "n_nodes": ep_metrics["n_nodes"],
            "n_edges": ep_metrics["n_edges"],
            "n_communities": ep_metrics["n_communities"],
            "community_map": ep_metrics["community_map"],
            "pair_stats": ep_metrics["pair_stats"],
            "centralities": ep_metrics["centralities"],
        },
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    # Save centrality timeseries CSV
    if not ts_df.empty:
        ts_df.to_csv(output_dir / "centrality_timeseries.csv", index=False)

    # Save edge birth/death CSV
    if not bd_df.empty:
        bd_df.to_csv(output_dir / "edge_birth_death.csv", index=False)

    logger.info("Analysis complete. Output: %s", output_dir)
    click.echo(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
