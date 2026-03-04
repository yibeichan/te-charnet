#!/usr/bin/env python3
"""Orchestrator: run all pipeline stages end-to-end for a single episode."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")
SCRIPTS_DIR = Path(__file__).parent


def run_stage(script: str, args: list[str], verbose: bool = False) -> None:
    cmd = [sys.executable, str(SCRIPTS_DIR / script)] + args
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    if result.returncode != 0:
        if result.stderr:
            click.echo(result.stderr, err=True)
        raise click.ClickException(f"Stage {script} failed with exit code {result.returncode}")
    if result.stdout:
        click.echo(result.stdout.strip())


@click.command()
@click.option("--transcript", "-t", required=True, type=click.Path(exists=True),
              help="Path to transcript JSON file.")
@click.option("--shots", "-s", default=None, type=click.Path(exists=True),
              help="Path to PySceneDetect CSV (optional).")
@click.option("--speaker-map", "-m", default=None, type=click.Path(exists=True),
              help="Path to speaker map JSON (optional).")
@click.option("--episode", "-e", default=None,
              help="Episode name. Inferred from transcript filename if not set.")
@click.option("--config", "-c", default=None, type=click.Path(exists=True),
              help="Config YAML file (not yet used; thresholds passed as CLI args).")
@click.option("--output-base", default=None,
              help="Base output directory. Defaults to $SCRATCH_DIR/output/")
@click.option("--skip-stages", default="", show_default=True,
              help="Comma-separated stage numbers to skip (e.g. '3,4').")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(transcript, shots, speaker_map, episode, config, output_base, skip_stages, verbose):
    """Run the full charnet pipeline for one episode."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    skip = set(skip_stages.split(",")) if skip_stages else set()

    transcript_path = Path(transcript)
    if episode is None:
        episode = transcript_path.stem
        for suffix in ["_model-AA_desc-wSpeaker_transcript", "_transcript", "_desc-wSpeaker_transcript"]:
            if episode.endswith(suffix):
                episode = episode[: -len(suffix)]
                break

    click.echo(f"=== charnet pipeline — episode: {episode} ===")

    # Stage 00
    if "0" not in skip:
        click.echo("\n[Stage 00] Preprocessing...")
        args = ["--transcript", transcript, "--episode", episode]
        if shots:
            args += ["--shots", shots]
        if speaker_map:
            args += ["--speaker-map", speaker_map]
        if verbose:
            args.append("--verbose")
        run_stage("00_preprocess.py", args, verbose=verbose)

    # Stage 01
    if "1" not in skip:
        click.echo("\n[Stage 01] Segmenting scenes...")
        args = ["--episode", episode]
        if verbose:
            args.append("--verbose")
        run_stage("01_segment_scenes.py", args, verbose=verbose)

    # Stage 02
    if "2" not in skip:
        click.echo("\n[Stage 02] Building network...")
        args = ["--episode", episode]
        if verbose:
            args.append("--verbose")
        run_stage("02_build_network.py", args, verbose=verbose)

    # Stage 03
    if "3" not in skip:
        click.echo("\n[Stage 03] Analyzing...")
        args = ["--episode", episode]
        if verbose:
            args.append("--verbose")
        run_stage("03_analyze.py", args, verbose=verbose)

    # Stage 04
    if "4" not in skip:
        click.echo("\n[Stage 04] Visualizing...")
        args = ["--episode", episode]
        if verbose:
            args.append("--verbose")
        run_stage("04_visualize.py", args, verbose=verbose)

    click.echo(f"\n=== Pipeline complete for {episode} ===")
    click.echo(f"Outputs in: {Path(SCRATCH_DIR) / 'output'}")


if __name__ == "__main__":
    main()
