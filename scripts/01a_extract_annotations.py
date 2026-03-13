#!/usr/bin/env python3
"""Stage 01a: extract sentence-level speaker annotations and scene summary."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

REPO_ROOT = Path(__file__).parent.parent
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")
DEFAULT_ANNOTATION_ROOT = REPO_ROOT / "data" / "friends_annotations" / "annotation_results"

sys.path.insert(0, str(REPO_ROOT / "src"))

from charnet.io import load_shots  # noqa: E402
from charnet.transcript_align import (  # noqa: E402
    discover_episodes_in_season,
    episode_to_season_dir,
    normalize_episode_key,
    normalize_season_id,
    process_episode,
)

logger = logging.getLogger(__name__)


def _load_shots_for_episode(episode: str, annotation_root: Path) -> list | None:
    """Try to load shot boundaries from TSVpyscene for an episode."""
    season_dir = episode_to_season_dir(episode)
    shot_path = annotation_root / "TSVpyscene" / season_dir / f"{episode}_pyscene.tsv"
    if shot_path.exists():
        return load_shots(shot_path)
    return None


@click.command()
@click.option("--episode", "-e", default=None, help="Single episode key, e.g. friends_s01e01a.")
@click.option("--season", default=None, help="Season id, e.g. s1 or 1.")
@click.option(
    "--annotation-root",
    type=click.Path(path_type=Path, exists=True),
    default=DEFAULT_ANNOTATION_ROOT,
    show_default=True,
    help="Root containing Speech2Text/, community_based/, and TSVpyscene/ folders.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output root. Defaults to $SCRATCH_DIR/output/map_speaker.",
)
@click.option(
    "--wutter-json",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Single-episode override for transcript JSON path.",
)
@click.option(
    "--community-txt",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Single-episode override for community transcript path.",
)
@click.option("--min-similarity", type=float, default=0.52, show_default=True)
@click.option("--length-ratio-cap", type=float, default=5.0, show_default=True)
@click.option("--neighbor-min-similarity", type=float, default=0.7, show_default=True)
@click.option("--anchor-min-similarity", type=float, default=0.6, show_default=True)
@click.option("--anchor-expand-min-similarity", type=float, default=0.72, show_default=True)
@click.option("--anchor-short-sent-tokens", type=int, default=4, show_default=True)
@click.option("--anchor-short-sent-similarity", type=float, default=0.40, show_default=True)
@click.option("--scene-iter-fill-similarity", type=float, default=0.72, show_default=True)
@click.option("--scene-iter-context-similarity", type=float, default=0.65, show_default=True)
@click.option("--scene-iter-ambiguity-margin", type=float, default=0.05, show_default=True)
@click.option("--scene-iter-max-rounds", type=int, default=8, show_default=True)
@click.option("--overwrite-speaker", is_flag=True, default=False, show_default=True)
@click.option("--scene-summary-only", is_flag=True, default=False, help="Only generate scene summary TSVs, skip sentence tables.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    episode: str | None,
    season: str | None,
    annotation_root: Path,
    output_dir: Path | None,
    wutter_json: Path | None,
    community_txt: Path | None,
    min_similarity: float,
    length_ratio_cap: float,
    neighbor_min_similarity: float,
    anchor_min_similarity: float,
    anchor_expand_min_similarity: float,
    anchor_short_sent_tokens: int,
    anchor_short_sent_similarity: float,
    scene_iter_fill_similarity: float,
    scene_iter_context_similarity: float,
    scene_iter_ambiguity_margin: float,
    scene_iter_max_rounds: int,
    overwrite_speaker: bool,
    scene_summary_only: bool,
    verbose: bool,
) -> None:
    """Extract speaker annotations and scene summary for one episode or a full season."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    if bool(episode) == bool(season):
        raise click.ClickException("Provide exactly one of --episode or --season.")
    if season and (wutter_json or community_txt):
        raise click.ClickException("--wutter-json/--community-txt are only allowed in --episode mode.")

    if output_dir is None:
        output_dir = Path(SCRATCH_DIR) / "output" / "map_speaker"

    if episode:
        episodes = [normalize_episode_key(episode)]
    else:
        season_dir = normalize_season_id(season or "")
        episodes = discover_episodes_in_season(season_dir, annotation_root)

    ok = 0
    failed = 0

    for ep in episodes:
        try:
            shots = _load_shots_for_episode(ep, annotation_root)
            out_tsv, stats = process_episode(
                episode=ep,
                annotation_root=annotation_root,
                output_dir=output_dir,
                min_similarity=min_similarity,
                neighbor_min_similarity=neighbor_min_similarity,
                anchor_min_similarity=anchor_min_similarity,
                anchor_expand_min_similarity=anchor_expand_min_similarity,
                scene_iter_fill_similarity=scene_iter_fill_similarity,
                scene_iter_context_similarity=scene_iter_context_similarity,
                scene_iter_ambiguity_margin=scene_iter_ambiguity_margin,
                scene_iter_max_rounds=scene_iter_max_rounds,
                overwrite_speaker=overwrite_speaker,
                length_ratio_cap=length_ratio_cap,
                anchor_short_sent_tokens=anchor_short_sent_tokens,
                anchor_short_sent_similarity=anchor_short_sent_similarity,
                wutter_override=wutter_json if len(episodes) == 1 else None,
                community_override=community_txt if len(episodes) == 1 else None,
                shots=shots,
                scene_summary_only=scene_summary_only,
            )
            ok += 1
            click.echo(f"[OK] {ep}")
            click.echo(f"  output: {out_tsv}")
            if not scene_summary_only:
                click.echo(f"  stats: {stats}")
        except Exception as exc:  # pragma: no cover - CLI guardrail
            failed += 1
            click.echo(f"[FAIL] {ep}: {exc}")

    click.echo(f"Completed. ok={ok} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
