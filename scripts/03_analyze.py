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
from charnet.metrics import scene_metrics, episode_metrics, centrality_timeseries, edge_birth_death

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

    network_path = network_dir / "temporal_network.json"
    if not network_path.exists():
        raise click.ClickException(f"temporal_network.json not found: {network_path}")

    scene_graphs = load_temporal_network(network_path)
    measures = [m.strip() for m in centrality_measures.split(",")]

    # Per-scene metrics
    per_scene = [scene_metrics(sg) for sg in scene_graphs]

    # Episode metrics
    ep_metrics = episode_metrics(scene_graphs, centrality_measures=measures,
                                  community_method=community_method)

    # Centrality time series
    ts_df = centrality_timeseries(scene_graphs, measures=measures)

    # Edge birth/death
    bd_df = edge_birth_death(scene_graphs)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics.json
    metrics_out = {
        "episode": episode,
        "per_scene": per_scene,
        "episode": {
            "n_nodes": ep_metrics["n_nodes"],
            "n_edges": ep_metrics["n_edges"],
            "n_communities": ep_metrics["n_communities"],
            "community_map": ep_metrics["community_map"],
            "pair_stats": ep_metrics["pair_stats"],
            "centralities": ep_metrics["centralities"],
        },
    }
    with open(output_dir / "metrics.json", "w") as f:
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
