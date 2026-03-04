#!/usr/bin/env python3
"""Batch runner: discover and process all Season 6 episode parts via friends_annotations submodule."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Generator

import click

logger = logging.getLogger(__name__)
SCRIPTS_DIR = Path(__file__).parent
REPO_ROOT = SCRIPTS_DIR.parent


def discover_s6_episodes(
    submodule_root: Path,
) -> Generator[tuple[str, Path, Path], None, None]:
    """Yield (episode_key, transcript_path, tsv_path) for each S6 episode part."""
    pyscene_dir = submodule_root / "annotation_results" / "TSVpyscene" / "s6"
    transcript_dir = submodule_root / "annotation_results" / "Speech2Text" / "s6"

    for tsv_path in sorted(pyscene_dir.glob("friends_s06e*_pyscene.tsv")):
        episode_key = tsv_path.stem.replace("_pyscene", "")  # e.g. "friends_s06e01a"
        transcript_path = transcript_dir / f"{episode_key}_model-AA_desc-wSpeaker_transcript.json"
        if transcript_path.exists():
            yield (episode_key, transcript_path, tsv_path)
        else:
            logger.warning("No transcript found for %s (expected %s)", episode_key, transcript_path)


@click.command()
@click.option("--config", "-c", default=str(REPO_ROOT / "config" / "default.yaml"),
              type=click.Path(), show_default=True,
              help="Config YAML file.")
@click.option("--submodule-root", default=str(REPO_ROOT / "data" / "friends_annotations"),
              type=click.Path(), show_default=True,
              help="Path to friends_annotations submodule root.")
@click.option("--output-base", default=str(REPO_ROOT / "output" / "s6"),
              show_default=True,
              help="Base output directory for all episodes.")
@click.option("--episodes", "-e", default=None, multiple=True,
              help="Filter to specific episode keys (e.g. friends_s06e01a). Repeat for multiple.")
@click.option("--skip-existing", is_flag=True, default=False,
              help="Skip episodes whose output directory already exists.")
@click.option("--dry-run", is_flag=True, default=False,
              help="List episodes that would be processed without running anything.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Enable debug logging.")
def main(config, submodule_root, output_base, episodes, skip_existing, dry_run, verbose):
    """Discover and process all Season 6 Friends episode parts."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    submodule_path = Path(submodule_root)
    output_base_path = Path(output_base)

    if not submodule_path.exists():
        raise click.ClickException(
            f"Submodule not found: {submodule_path}\n"
            "Run: git submodule update --init data/friends_annotations"
        )

    all_episodes = list(discover_s6_episodes(submodule_path))
    if not all_episodes:
        raise click.ClickException(f"No S6 episodes found under {submodule_path}")

    # Filter if --episodes provided
    if episodes:
        filter_set = set(episodes)
        all_episodes = [(k, t, s) for k, t, s in all_episodes if k in filter_set]
        if not all_episodes:
            raise click.ClickException(f"No matching episodes found for: {episodes}")

    click.echo(f"Found {len(all_episodes)} episode part(s) to process")

    if dry_run:
        for episode_key, transcript_path, tsv_path in all_episodes:
            click.echo(f"  {episode_key}")
            click.echo(f"    transcript: {transcript_path}")
            click.echo(f"    shots:      {tsv_path}")
        return

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for episode_key, transcript_path, tsv_path in all_episodes:
        episode_out = output_base_path / episode_key

        if skip_existing and episode_out.exists():
            logger.info("Skipping %s (output exists)", episode_key)
            n_skip += 1
            continue

        click.echo(f"\n=== Processing {episode_key} ===")

        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "run_pipeline.py"),
            "--transcript", str(transcript_path),
            "--shots", str(tsv_path),
            "--episode", episode_key,
            "--output-base", str(output_base_path),
        ]
        if config:
            cmd += ["--config", config]
        if verbose:
            cmd.append("--verbose")

        try:
            result = subprocess.run(cmd, text=True)
            if result.returncode != 0:
                logger.error("Episode %s failed with exit code %d", episode_key, result.returncode)
                n_fail += 1
            else:
                n_ok += 1
        except Exception as exc:
            logger.error("Episode %s raised an exception: %s", episode_key, exc)
            n_fail += 1

    click.echo(f"\n=== Batch complete: {n_ok} ok, {n_skip} skipped, {n_fail} failed ===")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
