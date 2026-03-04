#!/usr/bin/env python3
"""Stage 2: Build network — scenes → weighted interaction graphs."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from charnet.io import load_utterances, load_scenes, save_temporal_network
from charnet.network import build_temporal_network

logger = logging.getLogger(__name__)
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")


@click.command()
@click.option("--episode", "-e", required=True)
@click.option("--preprocess-dir", default=None)
@click.option("--segment-dir", default=None)
@click.option("--output-dir", "-o", default=None)
@click.option("--weight-adjacency", default=1.0, show_default=True)
@click.option("--weight-proximity", default=0.5, show_default=True)
@click.option("--weight-copresence", default=0.25, show_default=True)
@click.option("--proximity-window", default=3, show_default=True)
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    episode, preprocess_dir, segment_dir, output_dir,
    weight_adjacency, weight_proximity, weight_copresence, proximity_window, verbose,
):
    """Convert scenes into weighted temporal interaction graphs."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    if preprocess_dir is None:
        preprocess_dir = Path(SCRATCH_DIR) / "output" / "00_preprocess" / episode
    else:
        preprocess_dir = Path(preprocess_dir)

    if segment_dir is None:
        segment_dir = Path(SCRATCH_DIR) / "output" / "01_segment_scenes" / episode
    else:
        segment_dir = Path(segment_dir)

    if output_dir is None:
        output_dir = Path(SCRATCH_DIR) / "output" / "02_build_network" / episode
    else:
        output_dir = Path(output_dir)

    utt_path = preprocess_dir / "utterances.json"
    scenes_path = segment_dir / "scenes.json"

    for p in [utt_path, scenes_path]:
        if not p.exists():
            raise click.ClickException(f"Required file not found: {p}")

    utterances = load_utterances(utt_path)
    scenes = load_scenes(scenes_path)
    logger.info("Building network for %d scenes", len(scenes))

    temporal_network = build_temporal_network(
        scenes, utterances,
        weight_adjacency=weight_adjacency,
        weight_proximity=weight_proximity,
        weight_copresence=weight_copresence,
        proximity_window=proximity_window,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_temporal_network(temporal_network, output_dir / "temporal_network.json")

    # Summary
    n_edges = sum(len(sg.edges) for sg in temporal_network)
    logger.info("Built %d scene graphs with %d total edges. Output: %s", len(temporal_network), n_edges, output_dir)
    click.echo(f"Scene graphs: {len(temporal_network)}  Edges: {n_edges}  Output: {output_dir}")


if __name__ == "__main__":
    main()
