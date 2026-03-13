#!/usr/bin/env python3
"""Stage 2: Build network from aligned_rows scene_id labels."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from charnet.io import load_records, load_corrected_speaker_rows, save_temporal_network
from charnet.network import (
    aggregate_episode_graph,
    build_temporal_network_from_aligned_rows,
)

logger = logging.getLogger(__name__)
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")


@click.command()
@click.option("--episode", "-e", required=True)
@click.option("--segment-dir", default=None)
@click.option("--output-dir", "-o", default=None)
@click.option("--corrected-speaker-tsv", default=None, type=click.Path(exists=True, path_type=Path),
              help="Speaker-filled TSV from 01b (optional). When provided, used instead of aligned_rows.json.")
@click.option("--speaker-col", default="speaker", show_default=True,
              help="Column to use as speaker source when loading --corrected-speaker-tsv.")
@click.option("--weight-adjacency", default=1.0, show_default=True)
@click.option("--weight-proximity", default=0.5, show_default=True)
@click.option("--weight-copresence", default=0.25, show_default=True)
@click.option("--proximity-window", default=3, show_default=True)
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    episode, segment_dir, output_dir, corrected_speaker_tsv, speaker_col,
    weight_adjacency, weight_proximity, weight_copresence, proximity_window, verbose,
):
    """Build per-scene and half-episode interaction networks."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    if segment_dir is None:
        segment_dir = Path(SCRATCH_DIR) / "output" / "01a_segment_scenes" / episode
    else:
        segment_dir = Path(segment_dir)

    if output_dir is None:
        output_dir = Path(SCRATCH_DIR) / "output" / "02_build_network" / episode
    else:
        output_dir = Path(output_dir)

    if corrected_speaker_tsv is not None:
        aligned_rows = load_corrected_speaker_rows(corrected_speaker_tsv, speaker_col=speaker_col)
        logger.info("Building network from speaker TSV (%d rows)", len(aligned_rows))
    else:
        aligned_rows_path = segment_dir / "aligned_rows.json"
        if not aligned_rows_path.exists():
            raise click.ClickException(
                f"Required file not found: {aligned_rows_path}. "
                "Run 01a_extract_annotations.py first."
            )
        aligned_rows = load_records(aligned_rows_path)
        logger.info("Building network from aligned rows (%d rows)", len(aligned_rows))
    temporal_network = build_temporal_network_from_aligned_rows(
        aligned_rows,
        weight_adjacency=weight_adjacency,
        weight_proximity=weight_proximity,
        weight_copresence=weight_copresence,
        proximity_window=proximity_window,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_temporal_network(temporal_network, output_dir / "temporal_network.json")
    episode_graph = aggregate_episode_graph(temporal_network)

    edges = []
    for u, v, d in sorted(episode_graph.edges(data=True), key=lambda x: (x[0], x[1])):
        edges.append(
            {
                "source": u,
                "target": v,
                "weight": round(float(d.get("weight", 0.0)), 4),
                "adjacency": round(float(d.get("adjacency", 0.0)), 4),
                "proximity": round(float(d.get("proximity", 0.0)), 4),
                "copresence": round(float(d.get("copresence", 0.0)), 4),
            }
        )
    if temporal_network:
        episode_start = min(sg.start for sg in temporal_network)
        episode_end = max(sg.end for sg in temporal_network)
    else:
        episode_start = 0.0
        episode_end = 0.0
    episode_network = {
        "episode": episode,
        "start": episode_start,
        "end": episode_end,
        "n_scenes": len(temporal_network),
        "nodes": sorted(episode_graph.nodes()),
        "edges": edges,
    }
    with open(output_dir / "episode_network.json", "w", encoding="utf-8") as f:
        json.dump(episode_network, f, indent=2)

    # Summary
    n_scene_edges = sum(len(sg.edges) for sg in temporal_network)
    n_episode_edges = len(edges)
    logger.info(
        "Built %d scene graphs (%d scene-edges total) + half-episode graph (%d edges). Output: %s",
        len(temporal_network),
        n_scene_edges,
        n_episode_edges,
        output_dir,
    )
    click.echo(
        f"Scene graphs: {len(temporal_network)}  Scene-edges: {n_scene_edges}  "
        f"Episode-edges: {n_episode_edges}  Output: {output_dir}"
    )


if __name__ == "__main__":
    main()
