#!/usr/bin/env python3
"""Stage 1: Segment scenes — triangulate shots + transcript into narrative scenes."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from charnet.io import load_utterances, load_shots_json, save_scenes
from charnet.scene_segment import segment_with_shots, segment_transcript_only

logger = logging.getLogger(__name__)
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")


@click.command()
@click.option("--episode", "-e", required=True,
              help="Episode name (e.g. friends_s06e01a).")
@click.option("--preprocess-dir", default=None,
              help="Directory containing utterances.json and shots.json from stage 0. "
                   "Defaults to $SCRATCH_DIR/output/00_preprocess/<episode>/")
@click.option("--output-dir", "-o", default=None,
              help="Output directory. Defaults to $SCRATCH_DIR/output/01_segment_scenes/<episode>/")
@click.option("--jaccard-threshold", default=0.3, show_default=True)
@click.option("--max-shot-gap", default=2.0, show_default=True)
@click.option("--min-scene-duration", default=10.0, show_default=True)
@click.option("--max-scene-duration", default=300.0, show_default=True)
@click.option("--silence-lookahead", default=3, show_default=True)
@click.option("--transcript-only", "transcript_only_mode", is_flag=True, default=False,
              help="Use transcript-only segmentation even if shots.json exists.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    episode, preprocess_dir, output_dir,
    jaccard_threshold, max_shot_gap, min_scene_duration, max_scene_duration,
    silence_lookahead, transcript_only_mode, verbose,
):
    """Segment transcript + shots into narrative scenes."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    if preprocess_dir is None:
        preprocess_dir = Path(SCRATCH_DIR) / "output" / "00_preprocess" / episode
    else:
        preprocess_dir = Path(preprocess_dir)

    if output_dir is None:
        output_dir = Path(SCRATCH_DIR) / "output" / "01_segment_scenes" / episode
    else:
        output_dir = Path(output_dir)

    utt_path = preprocess_dir / "utterances.json"
    shots_path = preprocess_dir / "shots.json"

    if not utt_path.exists():
        raise click.ClickException(f"utterances.json not found: {utt_path}. Run 00_preprocess.py first.")

    utterances = load_utterances(utt_path)
    logger.info("Loaded %d utterances", len(utterances))

    shots = []
    if not transcript_only_mode and shots_path.exists():
        shots = load_shots_json(shots_path)
        logger.info("Loaded %d shots", len(shots))

    if shots and not transcript_only_mode:
        scenes = segment_with_shots(
            utterances, shots,
            jaccard_threshold=jaccard_threshold,
            max_shot_gap=max_shot_gap,
            min_scene_duration=min_scene_duration,
            max_scene_duration=max_scene_duration,
            silence_lookahead=silence_lookahead,
        )
    else:
        logger.info("Using transcript-only segmentation")
        scenes = segment_transcript_only(
            utterances,
            min_scene_duration=min_scene_duration,
            max_scene_duration=max_scene_duration,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_scenes(scenes, output_dir / "scenes.json")

    logger.info("Segmented into %d scenes. Output: %s", len(scenes), output_dir)
    click.echo(f"Scenes: {len(scenes)}  Output: {output_dir}")


if __name__ == "__main__":
    main()
