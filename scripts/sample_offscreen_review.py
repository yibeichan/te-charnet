#!/usr/bin/env python3
"""Generate a stratified blind sample of speaker_offscreen flags for manual review.

Reads `output/annotations/sentences/sN/*_sentence_speaker_table.tsv`, filters to
sentences where `annotation_review_reason == 'speaker_offscreen'`, samples
N rows per season (seed: --seed + season_number), and emits two TSVs:

  * blind review file (reviewer-facing): episode + timecodes + blank verdict cols
  * sealed answer key: labeled speaker + system claims, joined by review_id

See docs/manual_review_speaker_offscreen.md for the reviewer protocol.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import click
import pandas as pd


def hms(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = t - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


@click.command()
@click.option("--sentences-root", type=click.Path(exists=True, path_type=Path),
              default=Path("output/annotations/sentences"), show_default=True)
@click.option("--out-dir", type=click.Path(path_type=Path),
              default=Path("output/manual_review"), show_default=True)
@click.option("--per-season", type=int, default=20, show_default=True,
              help="Samples drawn per season.")
@click.option("--seed", type=int, default=20260522, show_default=True,
              help="Base seed; per-season seed = seed + season_number.")
@click.option("--stamp", default=None,
              help="Filename stamp (default: today's YYYYMMDD).")
def main(sentences_root: Path, out_dir: Path, per_season: int, seed: int, stamp: str | None):
    seasons = sorted(d.name for d in sentences_root.iterdir() if d.is_dir())
    samples = []
    for s in seasons:
        dfs = []
        for tsv in sorted((sentences_root / s).glob("*_sentence_speaker_table.tsv")):
            d = pd.read_csv(tsv, sep="\t")
            d["__episode__"] = tsv.stem.replace("_sentence_speaker_table", "")
            dfs.append(d)
        df = pd.concat(dfs, ignore_index=True)
        pool = df[df["annotation_review_reason"] == "speaker_offscreen"].copy()
        n = min(per_season, len(pool))
        season_n = int(s.lstrip("s"))
        sample = pool.sample(n=n, random_state=seed + season_n)
        samples.append(sample)
        click.echo(f"{s}: pool={len(pool):>5}, sampled={n}")

    review = pd.concat(samples, ignore_index=True)
    blind = pd.DataFrame({
        "review_id": [f"R{i:03d}" for i in range(1, len(review) + 1)],
        "episode": review["__episode__"].values,
        "sentence_id": review["sentence_id"].values,
        "start_seconds": review["start"].round(3).values,
        "end_seconds": review["end"].round(3).values,
        "start_hms": [hms(t) for t in review["start"].values],
        "end_hms": [hms(t) for t in review["end"].values],
        "duration_seconds": (review["end"] - review["start"]).round(3).values,
        "visible_during_clip": "",
        "audible_speaker": "",
        "notes": "",
    })
    key = pd.DataFrame({
        "review_id": blind["review_id"].values,
        "episode": blind["episode"].values,
        "sentence_id": blind["sentence_id"].values,
        "labeled_speaker": review["speaker"].values,
        "system_visual_presence": review["visual_presence"].values,
        "system_speaker_visual_presence": review["speaker_visual_presence"].values,
        "system_speaker_visual_ratio": review["speaker_visual_ratio"].round(3).values,
        "system_visual_presence_chars": review["visual_presence_chars"].values,
        "system_annotation_confidence": review["annotation_confidence"].values,
        "system_review_reason": review["annotation_review_reason"].values,
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or date.today().strftime("%Y%m%d")
    blind_path = out_dir / f"speaker_offscreen_sample_{stamp}.tsv"
    key_path = out_dir / f"speaker_offscreen_sample_{stamp}_KEY.tsv"
    blind.to_csv(blind_path, sep="\t", index=False)
    key.to_csv(key_path, sep="\t", index=False)

    click.echo()
    click.echo(f"Blind review TSV : {blind_path}  ({len(blind)} rows)")
    click.echo(f"Sealed answer key: {key_path}  ({len(key)} rows)")


if __name__ == "__main__":
    main()
