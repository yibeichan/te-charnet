#!/usr/bin/env python3
"""Orchestrator: run all pipeline stages for one episode or a full season."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")
SCRIPTS_DIR = Path(__file__).parent
REPO_ROOT = SCRIPTS_DIR.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "friends_annotations" / "annotation_results"


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


def normalize_episode_key(value: str) -> str:
    key = value.strip()
    if re.fullmatch(r"s\d{2}e\d{2}[a-z]*", key, flags=re.IGNORECASE):
        key = f"friends_{key}"
    return key.lower()


def normalize_season_id(value: str) -> str:
    token = value.strip().lower()
    token = re.sub(r"^friends_", "", token)
    match = re.fullmatch(r"s?0*(\d+)", token)
    if not match:
        raise click.ClickException(f"Invalid season id: {value}. Use values like 's6' or '6'.")
    return f"s{int(match.group(1))}"


def episode_from_transcript_filename(path: Path) -> str:
    episode = path.stem
    for suffix in [
        "_model-AA_desc-wSpeaker_transcript",
        "_desc-wSpeaker_transcript",
        "_model-AA_desc-wUtter_transcript",
        "_desc-wUtter_transcript",
        "_transcript",
    ]:
        if episode.endswith(suffix):
            return episode[: -len(suffix)]
    return episode


def discover_season_episodes(season_dir: str, data_root: Path) -> list[str]:
    transcript_dir = data_root / "Speech2Text" / season_dir
    if not transcript_dir.exists():
        raise click.ClickException(f"Transcript directory not found: {transcript_dir}")

    episodes = {
        normalize_episode_key(episode_from_transcript_filename(path))
        for path in transcript_dir.glob("friends_s*e*_model-AA_desc-wSpeaker_transcript.json")
    }
    if not episodes:
        raise click.ClickException(f"No transcripts found in {transcript_dir}")
    return sorted(episodes)


def infer_season_from_episode_arg(episode_value: str) -> str | None:
    token = episode_value.strip().lower()
    token = re.sub(r"^friends_", "", token)
    return normalize_season_id(token) if re.fullmatch(r"s?0*\d+", token) else None


@click.command()
@click.option("--episode", "-e", default=None,
              help="Episode key (e.g. friends_s06e01b or s06e01b).")
@click.option("--season", default=None,
              help="Season id (e.g. s6 or 6). Runs all episode parts in that season.")
@click.option("--data-root", type=click.Path(exists=True), default=str(DEFAULT_DATA_ROOT),
              show_default=True,
              help="Root folder containing Speech2Text, TSVpyscene, and community_based.")
@click.option("--transcript", "-t", default=None, type=click.Path(exists=True),
              help="Transcript override (single-episode mode only).")
@click.option("--shots", "-s", default=None, type=click.Path(exists=True),
              help="Shots override (single-episode mode only).")
@click.option("--speaker-map", "-m", default=None, type=click.Path(exists=True),
              help="Path to speaker map JSON (optional).")
@click.option("--community-transcript", "-u", default=None, type=click.Path(exists=True),
              help="Community transcript override (single-episode mode only).")
@click.option("--output-base", default=None,
              help="Base output directory. Defaults to $SCRATCH_DIR/output/")
@click.option("--skip-stages", default="", show_default=True,
              help="Comma-separated stage numbers to skip (e.g. '3,4').")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    episode,
    season,
    data_root,
    transcript,
    shots,
    speaker_map,
    community_transcript,
    output_base,
    skip_stages,
    verbose,
):
    """Run the full charnet pipeline for one episode or a whole season."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    # Accept season-like values passed via --episode, e.g. --episode friends_s06.
    if episode and not season:
        inferred_season = infer_season_from_episode_arg(episode)
        if inferred_season:
            season = inferred_season
            episode = None

    if episode and season:
        raise click.ClickException("Use either --episode or --season, not both.")
    if season and (transcript or shots or community_transcript):
        raise click.ClickException(
            "--transcript/--shots/--community-transcript overrides are only supported in --episode mode."
        )
    if not episode and not season and not transcript:
        raise click.ClickException("Provide one of --episode, --season, or --transcript.")

    data_root_path = Path(data_root)
    output_root = Path(output_base) if output_base else (Path(SCRATCH_DIR) / "output")
    skip = {s.strip() for s in skip_stages.split(",") if s.strip()}

    if season:
        episodes = discover_season_episodes(normalize_season_id(season), data_root_path)
    elif episode:
        episodes = [normalize_episode_key(episode)]
    else:
        episodes = [normalize_episode_key(episode_from_transcript_filename(Path(transcript)))]

    click.echo(f"=== charnet pipeline — processing {len(episodes)} episode(s) ===")

    for idx, ep in enumerate(episodes, start=1):
        click.echo(f"\n=== Episode {idx}/{len(episodes)}: {ep} ===")

        stage00_dir = output_root / "00_preprocess" / ep
        stage01_dir = output_root / "01_segment_scenes" / ep
        stage02_dir = output_root / "02_build_network" / ep
        stage03_dir = output_root / "03_analyze" / ep
        stage04_dir = output_root / "04_visualize" / ep

        # Stage 00
        if "0" not in skip:
            click.echo("[Stage 00] Preprocessing...")
            args = [
                "--episode", ep,
                "--data-root", str(data_root_path),
                "--output-dir", str(stage00_dir),
            ]
            if transcript and len(episodes) == 1:
                args += ["--transcript", transcript]
            if shots and len(episodes) == 1:
                args += ["--shots", shots]
            if speaker_map:
                args += ["--speaker-map", speaker_map]
            if community_transcript and len(episodes) == 1:
                args += ["--community-transcript", community_transcript]
            if verbose:
                args.append("--verbose")
            run_stage("00_preprocess.py", args, verbose=verbose)

        # Stage 01
        if "1" not in skip:
            click.echo("[Stage 01] Segmenting scenes...")
            args = [
                "--episode", ep,
                "--preprocess-dir", str(stage00_dir),
                "--output-dir", str(stage01_dir),
            ]
            if verbose:
                args.append("--verbose")
            run_stage("01_segment_scenes.py", args, verbose=verbose)

        # Stage 02
        if "2" not in skip:
            click.echo("[Stage 02] Building network...")
            args = [
                "--episode", ep,
                "--segment-dir", str(stage01_dir),
                "--output-dir", str(stage02_dir),
            ]
            if verbose:
                args.append("--verbose")
            run_stage("02_build_network.py", args, verbose=verbose)

        # Stage 03
        if "3" not in skip:
            click.echo("[Stage 03] Analyzing...")
            args = [
                "--episode", ep,
                "--network-dir", str(stage02_dir),
                "--output-dir", str(stage03_dir),
            ]
            if verbose:
                args.append("--verbose")
            run_stage("03_analyze.py", args, verbose=verbose)

        # Stage 04
        if "4" not in skip:
            click.echo("[Stage 04] Visualizing...")
            args = [
                "--episode", ep,
                "--network-dir", str(stage02_dir),
                "--analyze-dir", str(stage03_dir),
                "--output-dir", str(stage04_dir),
            ]
            if verbose:
                args.append("--verbose")
            run_stage("04_visualize.py", args, verbose=verbose)

    click.echo(f"\n=== Pipeline complete ({len(episodes)} episode(s)) ===")
    click.echo(f"Outputs in: {output_root}")


if __name__ == "__main__":
    main()
