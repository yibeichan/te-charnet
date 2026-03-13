#!/usr/bin/env python3
"""Orchestrator: run the charnet pipeline for one episode or a full season.

Pipeline flow: 01a (align) → 01b (fill speakers) → 02 (network) → 03 (analyze) → 04 (visualize)
"""
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

sys.path.insert(0, str(REPO_ROOT / "src"))

from charnet.transcript_align import (  # noqa: E402
    discover_episodes_in_season,
    episode_to_season_dir,
    normalize_episode_key,
    normalize_season_id,
)


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


def infer_season_from_episode_arg(episode_value: str) -> str | None:
    token = episode_value.strip().lower()
    token = re.sub(r"^friends_", "", token)
    return normalize_season_id(token) if re.fullmatch(r"s?0*\d+", token) else None


def discover_episodes_from_tsv(season_dir: str, tsv_root: Path) -> list[str]:
    """Discover episode keys from sentence_speaker_table TSVs produced by 01a."""
    season_path = tsv_root / season_dir
    if not season_path.exists():
        return []
    episodes = set()
    for tsv_path in season_path.glob("*_sentence_speaker_table.tsv"):
        ep = tsv_path.stem.replace("_sentence_speaker_table", "")
        episodes.add(normalize_episode_key(ep))
    return sorted(episodes)


def find_speaker_tsv(episode: str, output_root: Path) -> Path | None:
    """Find the best available speaker-filled TSV for an episode.

    Prefers final-cleaned (from global QA), falls back to enhanced.
    """
    season_dir = episode_to_season_dir(episode)

    # Final cleaned (global QA output)
    final_cleaned = (
        output_root / "map_speaker_final" / "global_qa_work" / "final_cleaned"
        / f"{episode}_sentence_speaker_table_final_cleaned.tsv"
    )
    if final_cleaned.exists():
        return final_cleaned

    # Enhanced (01b output)
    enhanced = (
        output_root / "map_speaker_enhanced" / season_dir
        / f"{episode}_sentence_speaker_table_enhanced.tsv"
    )
    if enhanced.exists():
        return enhanced

    return None


@click.command()
@click.option("--episode", "-e", default=None,
              help="Episode key (e.g. friends_s06e01b or s06e01b).")
@click.option("--season", default=None,
              help="Season id (e.g. s6 or 6). Runs all episode parts in that season.")
@click.option("--annotation-root", type=click.Path(exists=True), default=str(DEFAULT_DATA_ROOT),
              show_default=True,
              help="Root folder containing Speech2Text, TSVpyscene, and community_based.")
@click.option("--output-base", default=None,
              help="Base output directory. Defaults to $SCRATCH_DIR/output/")
@click.option("--skip-stages", default="", show_default=True,
              help="Comma-separated stage numbers to skip (e.g. '3,4').")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    episode,
    season,
    annotation_root,
    output_base,
    skip_stages,
    verbose,
):
    """Run the charnet pipeline: 01a → 01b → 02 → 03 → 04."""
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
    if not episode and not season:
        raise click.ClickException("Provide one of --episode or --season.")

    annotation_root_path = Path(annotation_root)
    output_root = Path(output_base) if output_base else (Path(SCRATCH_DIR) / "output")
    skip = {s.strip() for s in skip_stages.split(",") if s.strip()}

    if season:
        season_dir = normalize_season_id(season)
        episodes = discover_episodes_in_season(season_dir, annotation_root_path)
    else:
        episodes = [normalize_episode_key(episode)]

    click.echo(f"=== charnet pipeline — processing {len(episodes)} episode(s) ===")

    # Stage 01a — extract annotations + scene summary (per-season or per-episode)
    if "1a" not in skip:
        click.echo("\n[Stage 01a] Extracting annotations...")
        stage01a_args = [
            "--annotation-root", str(annotation_root_path),
            "--output-dir", str(output_root / "map_speaker"),
        ]
        if season:
            stage01a_args += ["--season", season]
        else:
            stage01a_args += ["--episode", episodes[0]]
        if verbose:
            stage01a_args.append("--verbose")
        run_stage("01a_extract_annotations.py", stage01a_args, verbose=verbose)

    # Stage 01b — fill missing speakers (per-season)
    if "1b" not in skip:
        click.echo("\n[Stage 01b] Filling missing speaker annotations...")
        stage01b_args = [
            "--input-dir", str(output_root / "map_speaker"),
            "--output-dir", str(output_root / "map_speaker_enhanced"),
            "--final-dir", str(output_root / "map_speaker_final"),
        ]
        if season:
            stage01b_args += ["--season", season]
        if verbose:
            stage01b_args.append("--verbose")
        run_stage("01b_fill_speakers.py", stage01b_args, verbose=verbose)

    for idx, ep in enumerate(episodes, start=1):
        click.echo(f"\n=== Episode {idx}/{len(episodes)}: {ep} ===")

        stage02_dir = output_root / "02_build_network" / ep
        stage03_dir = output_root / "03_analyze" / ep
        stage04_dir = output_root / "04_visualize" / ep

        # Stage 02
        if "2" not in skip:
            click.echo("[Stage 02] Building network...")
            speaker_tsv = find_speaker_tsv(ep, output_root)
            if speaker_tsv is None:
                click.echo(f"  [SKIP] No speaker-filled TSV found for {ep}")
                continue
            args = [
                "--episode", ep,
                "--corrected-speaker-tsv", str(speaker_tsv),
                "--speaker-col", "speaker",
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
