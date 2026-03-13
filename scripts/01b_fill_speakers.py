#!/usr/bin/env python3
"""Stage 01b: fill missing speaker annotations using cascading inference rules."""
from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from pathlib import Path

import click
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")

sys.path.insert(0, str(REPO_ROOT / "src"))

from charnet.speaker_fill import (  # noqa: E402
    FINAL_COLUMNS,
    PipelineConfig,
    global_qa,
    process_tsv,
    zip_dir,
)
import tempfile

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = REPO_ROOT / "src" / "charnet" / "pipeline_config.yaml"
DEFAULT_INPUT_DIR = Path(SCRATCH_DIR) / "output" / "map_speaker"
DEFAULT_OUTPUT_DIR = Path(SCRATCH_DIR) / "output" / "map_speaker_enhanced"
DEFAULT_FINAL_DIR = Path(SCRATCH_DIR) / "output" / "map_speaker_final"

# Enhanced output keeps timing columns alongside the pipeline's schema.
ENHANCED_COLUMNS = FINAL_COLUMNS[:2] + ["start", "end"] + FINAL_COLUMNS[2:]


def _restore_timing(enhanced_df: pd.DataFrame, original_path: Path) -> pd.DataFrame:
    """Insert start/end timing columns from the original TSV back into the enhanced frame."""
    orig = pd.read_csv(original_path, sep="\t")
    assert len(enhanced_df) == len(orig), (
        f"Row count mismatch: enhanced={len(enhanced_df)}, original={len(orig)} for {original_path.name}"
    )
    enhanced_df = enhanced_df.copy()
    enhanced_df.insert(2, "start", orig["start"].values)
    enhanced_df.insert(3, "end", orig["end"].values)
    return enhanced_df


def _restore_timing_from_enhanced(final_df: pd.DataFrame, enhanced_path: Path) -> pd.DataFrame:
    """Merge start/end back into a global-QA final-cleaned frame using the enhanced TSV."""
    enh = pd.read_csv(enhanced_path, sep="\t")
    assert len(final_df) == len(enh), (
        f"Row count mismatch: final={len(final_df)}, enhanced={len(enh)} for {enhanced_path.name}"
    )
    final_df = final_df.copy()
    final_df.insert(2, "start", enh["start"].values)
    final_df.insert(3, "end", enh["end"].values)
    return final_df


def _discover_seasons(input_dir: Path, season_filter: str | None) -> list[str]:
    """Return sorted list of season directory names (e.g. ['s1', 's2', ...])."""
    if season_filter:
        token = season_filter.strip().lower()
        match = re.fullmatch(r"s?0*(\d+)", token)
        if not match:
            raise click.ClickException(f"Invalid season id: {season_filter}")
        return [f"s{int(match.group(1))}"]
    return sorted(
        d.name for d in input_dir.iterdir()
        if d.is_dir() and re.fullmatch(r"s\d+", d.name)
    )


@click.command()
@click.option("--season", default=None, help="Season id (e.g. s1 or 1). Omit to process all.")
@click.option(
    "--input-dir",
    type=click.Path(path_type=Path, exists=True),
    default=DEFAULT_INPUT_DIR,
    show_default=True,
    help="Directory containing per-season subdirectories of TSVs from 01a.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Output root for enhanced TSVs.",
)
@click.option(
    "--final-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_FINAL_DIR,
    show_default=True,
    help="Output root for global-QA final cleaned TSVs.",
)
@click.option(
    "--config",
    type=click.Path(path_type=Path, exists=True),
    default=DEFAULT_CONFIG,
    show_default=True,
    help="Path to pipeline_config.yaml.",
)
@click.option("--skip-qa", is_flag=True, default=False, help="Skip the global QA pass.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    season: str | None,
    input_dir: Path,
    output_dir: Path,
    final_dir: Path,
    config: Path,
    skip_qa: bool,
    verbose: bool,
) -> None:
    """Fill missing speaker annotations for TSVs produced by 01a."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    cfg = PipelineConfig.from_yaml(config)
    seasons = _discover_seasons(input_dir, season)
    if not seasons:
        raise click.ClickException(f"No season directories found in {input_dir}")

    for season_label in seasons:
        season_in = input_dir / season_label
        tsv_files = sorted(season_in.glob("*.tsv"))
        if not tsv_files:
            click.echo(f"[SKIP] {season_label}: no TSV files found")
            continue

        enhanced_dir = output_dir / season_label
        review_dir = output_dir / f"{season_label}_review"
        enhanced_dir.mkdir(parents=True, exist_ok=True)
        review_dir.mkdir(parents=True, exist_ok=True)

        summary_rows = []

        for tsv_path in tsv_files:
            try:
                enhanced_df, summary = process_tsv(tsv_path, cfg)
                enhanced_df = _restore_timing(enhanced_df, tsv_path)

                out_name = tsv_path.stem + "_enhanced.tsv"
                out_path = enhanced_dir / out_name
                enhanced_df[ENHANCED_COLUMNS].to_csv(out_path, sep="\t", index=False)

                review_df = enhanced_df[enhanced_df["review_flag"]]
                if len(review_df) > 0:
                    review_path = review_dir / (tsv_path.stem + "_review.tsv")
                    review_df[ENHANCED_COLUMNS].to_csv(review_path, sep="\t", index=False)

                summary_rows.append(summary)
                click.echo(f"  [OK] {tsv_path.name}  filled={summary['filled_rows']}  review={summary['review_rows']}")
            except Exception as exc:
                click.echo(f"  [FAIL] {tsv_path.name}: {exc}")
                summary_rows.append({"file": tsv_path.name, "error": str(exc)})

        # Season summary
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / f"{season_label}_annotation_summary.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)

        total_filled = sum(s.get("filled_rows", 0) for s in summary_rows)
        total_review = sum(s.get("review_rows", 0) for s in summary_rows)
        click.echo(f"[OK] {season_label}: {len(tsv_files)} files, filled={total_filled}, review={total_review}")

    if skip_qa or not seasons:
        click.echo("Done (global QA skipped).")
        return

    # Global QA pass — global_qa() requires zip inputs, so build temp zips.
    click.echo("\nRunning global QA...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        enhanced_zips: list[Path] = []
        for season_label in seasons:
            enhanced_dir = output_dir / season_label
            enh_files = sorted(enhanced_dir.glob("*_enhanced.tsv"))
            if not enh_files:
                continue
            summary_file = output_dir / f"{season_label}_annotation_summary.tsv"
            files_to_zip = ([summary_file] if summary_file.exists() else []) + enh_files
            zip_path = tmp_path / f"{season_label}_enhanced_outputs.zip"
            zip_dir(zip_path, files_to_zip)
            enhanced_zips.append(zip_path)

        if enhanced_zips:
            global_qa(enhanced_zips, final_dir)

    # Clean up artifacts produced by global_qa inside final_dir
    for zf in final_dir.glob("*.zip"):
        zf.unlink()
    qa_input_dir = final_dir / "global_qa_work" / "input"
    if qa_input_dir.exists():
        shutil.rmtree(qa_input_dir)

    # Restore timing columns in final cleaned TSVs
    final_cleaned_dir = final_dir / "global_qa_work" / "final_cleaned"
    if final_cleaned_dir.exists():
        for final_tsv in sorted(final_cleaned_dir.glob("*_final_cleaned.tsv")):
            enhanced_name = final_tsv.name.replace("_final_cleaned.tsv", "_enhanced.tsv")
            for season_label in seasons:
                candidate = output_dir / season_label / enhanced_name
                if candidate.exists():
                    final_df = pd.read_csv(final_tsv, sep="\t")
                    final_df = _restore_timing_from_enhanced(final_df, candidate)
                    final_df.to_csv(final_tsv, sep="\t", index=False)
                    break

    click.echo("Done.")


if __name__ == "__main__":
    main()
