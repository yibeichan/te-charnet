#!/usr/bin/env python3
"""Stage 0: Preprocess — normalize transcript JSON and shot CSV inputs."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

# Allow running as script without installing package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json

from charnet.io import (
    load_transcript, load_shots, load_speaker_map,
    load_word_transcript, load_pyscene_tsv,
    estimate_missing_end_times, save_utterances, save_shots,
)

logger = logging.getLogger(__name__)
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")


@click.command()
@click.option("--transcript", "-t", type=click.Path(exists=True), required=True,
              help="Path to transcript JSON file.")
@click.option("--shots", "-s", type=click.Path(exists=True), default=None,
              help="Path to PySceneDetect CSV file (optional).")
@click.option("--speaker-map", "-m", type=click.Path(exists=True), default=None,
              help="Path to speaker map JSON file (optional).")
@click.option("--episode", "-e", default=None,
              help="Episode name (e.g. friends_s06e01a). Inferred from transcript filename if not set.")
@click.option("--output-dir", "-o", default=None,
              help="Output directory. Defaults to $SCRATCH_DIR/output/00_preprocess/<episode>/")
@click.option("--min-utterance-duration", default=0.1, show_default=True,
              help="Discard utterances shorter than this many seconds.")
@click.option("--estimate-end-times/--no-estimate-end-times", default=True, show_default=True,
              help="Estimate missing end times from next utterance start.")
@click.option("--word-gap-threshold", default=0.5, show_default=True, type=float,
              help="Max gap (seconds) between same-speaker words before starting a new utterance (word-level JSON only).")
@click.option("--inspect-only", is_flag=True, default=False,
              help="Only inspect inputs and print summary; do not write output.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Enable debug logging.")
def main(
    transcript, shots, speaker_map, episode, output_dir,
    min_utterance_duration, estimate_end_times, word_gap_threshold, inspect_only, verbose,
):
    """Normalize transcript JSON and PySceneDetect CSV into canonical format."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    transcript_path = Path(transcript)

    # Infer episode name from filename if not provided
    if episode is None:
        episode = transcript_path.stem
        # Strip common suffixes like _model-AA_desc-wSpeaker_transcript
        for suffix in ["_model-AA_desc-wSpeaker_transcript", "_transcript", "_desc-wSpeaker_transcript"]:
            if episode.endswith(suffix):
                episode = episode[: -len(suffix)]
                break
    logger.info("Episode: %s", episode)

    # Output directory
    if output_dir is None:
        output_dir = Path(SCRATCH_DIR) / "output" / "00_preprocess" / episode
    else:
        output_dir = Path(output_dir)

    # Load speaker map
    spk_map = None
    if speaker_map:
        spk_map = load_speaker_map(Path(speaker_map))
        logger.info("Loaded speaker map with %d entries", len(spk_map))

    # Load transcript — auto-detect word-level vs utterance-level format
    logger.info("Loading transcript: %s", transcript_path)
    with open(transcript_path) as _f:
        _probe = json.load(_f)
    if (
        isinstance(_probe, dict)
        and "words" in _probe
        and isinstance(_probe["words"], list)
        and _probe["words"]
        and "word" in _probe["words"][0]
    ):
        logger.info("Detected word-level transcript format")
        utterances = load_word_transcript(
            transcript_path, word_gap_threshold=word_gap_threshold, speaker_map=spk_map
        )
    else:
        utterances = load_transcript(transcript_path, speaker_map=spk_map)
    logger.info("Loaded %d utterances", len(utterances))

    # Estimate missing end times
    if estimate_end_times:
        utterances = estimate_missing_end_times(utterances, min_duration=min_utterance_duration)

    # Filter too-short utterances
    before = len(utterances)
    utterances = [u for u in utterances if (u.end - u.start) >= min_utterance_duration]
    if len(utterances) < before:
        logger.warning("Filtered %d utterances shorter than %.2fs", before - len(utterances), min_utterance_duration)

    # Load shots — dispatch on file extension
    shot_list = []
    if shots:
        logger.info("Loading shots: %s", shots)
        shots_path = Path(shots)
        if shots_path.suffix == ".tsv":
            shot_list = load_pyscene_tsv(shots_path)
        else:
            shot_list = load_shots(shots_path)
        logger.info("Loaded %d shots", len(shot_list))

        # Validate timestamp alignment
        if utterances and shot_list:
            utt_end = utterances[-1].end
            shot_end = shot_list[-1].end
            if utt_end > shot_end * 1.1:
                logger.warning(
                    "Transcript extends to %.1fs but shots end at %.1fs — possible mismatch",
                    utt_end, shot_end,
                )

    # Inspect summary
    speakers = sorted(set(u.speaker for u in utterances if u.speaker))
    logger.info("Speakers (%d): %s", len(speakers), ", ".join(speakers))
    if utterances:
        logger.info(
            "Transcript span: %.2fs — %.2fs (%.1f min)",
            utterances[0].start, utterances[-1].end,
            (utterances[-1].end - utterances[0].start) / 60,
        )

    if inspect_only:
        click.echo(f"Episode:    {episode}")
        click.echo(f"Utterances: {len(utterances)}")
        click.echo(f"Shots:      {len(shot_list)}")
        click.echo(f"Speakers:   {', '.join(speakers)}")
        return

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    save_utterances(utterances, output_dir / "utterances.json")
    if shot_list:
        save_shots(shot_list, output_dir / "shots.json")

    logger.info("Preprocessing complete. Output: %s", output_dir)
    click.echo(f"Output written to: {output_dir}")


if __name__ == "__main__":
    main()
