#!/usr/bin/env python3
"""Stage 1: align community transcript to timed utterances + assign shot ids."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from charnet.community_align import (
    align_monotonic,
    assign_scene_ids_from_scene_desc,
    build_alignment_rows,
    save_alignment_rows_tsv,
)
from charnet.io import load_utterances, load_shots_json, load_records, save_records

logger = logging.getLogger(__name__)
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")


@click.command()
@click.option("--episode", "-e", required=True,
              help="Episode name (e.g. friends_s06e01a).")
@click.option("--preprocess-dir", default=None,
              help="Directory containing stage-0 outputs. "
                   "Defaults to $SCRATCH_DIR/output/00_preprocess/<episode>/")
@click.option("--output-dir", "-o", default=None,
              help="Output directory. Defaults to $SCRATCH_DIR/output/01_segment_scenes/<episode>/")
@click.option("--min-similarity", default=0.52, show_default=True, type=float,
              help="Minimum similarity for accepting mapped community dialogue labels.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    episode,
    preprocess_dir,
    output_dir,
    min_similarity,
    verbose,
):
    """Align corrected timed utterances to community transcript + annotate scene_desc and shot_id."""
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

    sentence_path = preprocess_dir / "sentences.json"
    if not sentence_path.exists():
        raise click.ClickException(
            f"sentences.json not found: {sentence_path}. "
            "Run 00_preprocess.py with the updated script first."
        )

    community_events_path = preprocess_dir / "community_events.json"
    community_dialogues_path = preprocess_dir / "community_dialogues.json"
    if not community_events_path.exists() or not community_dialogues_path.exists():
        raise click.ClickException(
            "Community transcript artifacts not found in preprocess output. "
            "Run 00_preprocess.py with --community-transcript (or ensure inferable path)."
        )

    timed_utterances = load_utterances(sentence_path)
    community_events = load_records(community_events_path)
    community_dialogues = load_records(community_dialogues_path)
    logger.info(
        "Loaded %d timed sentence utterances + %d community dialogues",
        len(timed_utterances),
        len(community_dialogues),
    )

    shots_path = preprocess_dir / "shots.json"
    shots = load_shots_json(shots_path) if shots_path.exists() else []
    if shots:
        logger.info("Loaded %d shots", len(shots))

    mapping, sim_cache = align_monotonic(timed_utterances, community_dialogues)
    rows, matched = build_alignment_rows(
        timed_utterances=timed_utterances,
        community_events=community_events,
        community_dialogues=community_dialogues,
        mapping=mapping,
        sim_cache=sim_cache,
        shots=shots,
        min_similarity=min_similarity,
    )
    rows, n_scene_ids = assign_scene_ids_from_scene_desc(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_tsv = output_dir / "aligned_rows.tsv"
    aligned_json = output_dir / "aligned_rows.json"
    save_alignment_rows_tsv(rows, aligned_tsv)
    save_records(rows, aligned_json, label="aligned rows")
    logger.info(
        "Aligned rows saved: %s (matches: %d / %d utterances, scene_ids: %d)",
        aligned_tsv,
        matched,
        len(timed_utterances),
        n_scene_ids,
    )

    click.echo(
        f"Aligned rows: {len(rows)}  Matched: {matched}/{len(timed_utterances)}  "
        f"Scene IDs: {n_scene_ids}  Output: {output_dir}"
    )


if __name__ == "__main__":
    main()
