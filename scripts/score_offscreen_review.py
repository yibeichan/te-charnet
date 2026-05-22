#!/usr/bin/env python3
"""Score the manual speaker_offscreen review against the sealed key.

Usage:
    .venv/bin/python scripts/score_offscreen_review.py \
        --review-tsv output/manual_review/speaker_offscreen_sample_YYYYMMDD.tsv \
        --key-tsv    output/manual_review/speaker_offscreen_sample_YYYYMMDD_KEY.tsv

SC-002 passes if confirmation rate >= 0.90.
"""
from __future__ import annotations

import re
from pathlib import Path

import click
import pandas as pd


NONE_TOKENS = {"", "none", "nobody", "no one", "empty", "na", "n/a", "-"}


def normalize_tokens(cell: str | float | None) -> set[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return set()
    raw = str(cell).strip().lower()
    if raw in NONE_TOKENS:
        return set()
    parts = re.split(r"[,/;|]+", raw)
    return {p.strip() for p in parts if p.strip() and p.strip() not in NONE_TOKENS}


def verdict_row(labeled_speaker: str, visible_cell: str) -> str:
    visible = normalize_tokens(visible_cell)
    if not visible:
        return "unreviewed" if visible_cell is None or (isinstance(visible_cell, float) and pd.isna(visible_cell)) else "confirmed"
    target = str(labeled_speaker).strip().lower()
    if target in visible:
        return "disconfirmed"
    return "confirmed"


@click.command()
@click.option("--review-tsv", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--key-tsv", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out-tsv", type=click.Path(path_type=Path), default=None,
              help="Optional scored output TSV. Defaults to <review-tsv>.scored.tsv")
@click.option("--threshold", type=float, default=0.90, show_default=True,
              help="SC-002 pass threshold (fraction confirmed).")
def main(review_tsv: Path, key_tsv: Path, out_tsv: Path | None, threshold: float):
    review = pd.read_csv(review_tsv, sep="\t", dtype=str, keep_default_na=False)
    key = pd.read_csv(key_tsv, sep="\t", dtype=str, keep_default_na=False)

    merged = review.merge(key, on="review_id", how="inner", suffixes=("", "_key"))
    if len(merged) != len(review):
        raise click.ClickException(
            f"Join produced {len(merged)} rows but review has {len(review)} — review_id mismatch."
        )

    visible_col = merged["visible_during_clip"].where(merged["visible_during_clip"].astype(bool), None)
    merged["verdict"] = [
        verdict_row(spk, vis) for spk, vis in zip(merged["labeled_speaker"], visible_col)
    ]

    counts = merged["verdict"].value_counts()
    n_total = len(merged)
    n_unreviewed = int(counts.get("unreviewed", 0))
    n_reviewed = n_total - n_unreviewed
    n_conf = int(counts.get("confirmed", 0))
    n_disc = int(counts.get("disconfirmed", 0))
    rate = (n_conf / n_reviewed) if n_reviewed else float("nan")

    season_re = re.compile(r"friends_(s\d+)")
    merged["__season__"] = merged["episode"].apply(
        lambda e: (season_re.match(e or "").group(1) if season_re.match(e or "") else "?")
    )

    click.echo(f"{'-'*72}")
    click.echo(f"Review file : {review_tsv}")
    click.echo(f"Key file    : {key_tsv}")
    click.echo(f"Rows total  : {n_total}")
    click.echo(f"Reviewed    : {n_reviewed}   Unreviewed: {n_unreviewed}")
    click.echo(f"Confirmed   : {n_conf}")
    click.echo(f"Disconfirmed: {n_disc}")
    click.echo(f"Confirmation rate: {rate:.3f}   threshold: {threshold:.2f}   "
               f"{'PASS' if (rate >= threshold) else 'FAIL'}")
    click.echo(f"{'-'*72}")
    click.echo("Per-season breakdown:")
    by_season = merged.groupby("__season__")["verdict"].value_counts().unstack(fill_value=0)
    for col in ["confirmed", "disconfirmed", "unreviewed"]:
        if col not in by_season.columns:
            by_season[col] = 0
    by_season["reviewed"] = by_season["confirmed"] + by_season["disconfirmed"]
    by_season["rate"] = by_season["confirmed"] / by_season["reviewed"].replace(0, pd.NA)
    click.echo(by_season[["confirmed", "disconfirmed", "unreviewed", "reviewed", "rate"]].to_string())

    out_path = out_tsv or review_tsv.with_suffix(".scored.tsv")
    keep_cols = [
        "review_id", "episode", "sentence_id", "start_seconds", "end_seconds",
        "labeled_speaker", "visible_during_clip", "audible_speaker", "verdict",
        "system_visual_presence", "system_speaker_visual_ratio",
        "system_visual_presence_chars", "system_annotation_confidence", "notes",
    ]
    have = [c for c in keep_cols if c in merged.columns]
    merged[have].to_csv(out_path, sep="\t", index=False)
    click.echo(f"\nScored TSV written: {out_path}")


if __name__ == "__main__":
    main()
