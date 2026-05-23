"""Evaluate our LLM scene segmentation against manual annotations (s1-s6).

Reports, per episode and aggregated:
  - boundary detection P/R/F1 at ±2 / ±5 / ±10 s tolerance, at two units:
      * "scene" — manual segments collapsed into scenes (location-stable blocks)
      * "segment" — manual segments left as-is (story beats)
  - per-gold-unit IoU against best-overlapping predicted scene
  - boundary-miss diagnostic: each gold segment-boundary tagged with ONbond_*
    reasons (location / character entry / character leave / time jump /
    goal change / music transit / theme song) and whether a predicted boundary
    fell within ±5 s.

Comparison is restricted to the overlapping coverage window per episode
(t in [max(manual_start, our_start), min(manual_end, our_end)]) so the
theme-song / pre-roll that our pipeline skips doesn't count as misses.

Outputs (default OUT_DIR=output/evaluation/scene_segmentation/):
  - per_episode.tsv  (long: one row per episode×unit)
  - per_episode_counts.tsv  (wide: one row per episode, counts only)
  - aggregate.json
  - boundary_diagnostics.tsv  (one row per gold segment-boundary in window)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MANUAL_DIR = REPO / "data/friends_annotations/annotation_results/manual_segmentation"
DEFAULT_OURS_DIR = REPO / "output/annotations/scenes"
OURS_DIR = DEFAULT_OURS_DIR  # mutable; overridden by --ours-dir
OUT_DIR = REPO / "output/evaluation/scene_segmentation"

TOLERANCES = (2.0, 5.0, 10.0)
DIAG_TOL = 5.0

ONBOND_COLS = [
    "ONbond_location",
    "ONbond_charact_entry",
    "ONbond_charact_leave",
    "ONbond_time_jump",
    "ONbond_goal_change",
    "ONbond_music_transit",
    "ONbond_theme_song",
]

EP_RE = re.compile(r"^s(\d{2})e(\d{2})[a-z]$")


def manual_path(episode: str) -> Path:
    return MANUAL_DIR / f"s{int(episode[1:3])}" / f"friends_{episode}_manualseg.tsv"


def ours_path(episode: str) -> Path:
    return OURS_DIR / f"s{int(episode[1:3])}" / f"friends_{episode}_scene_summary.tsv"


def set_ours_dir(path: Path) -> None:
    global OURS_DIR
    OURS_DIR = path


def discover_episodes() -> list[str]:
    """Half-eps present in BOTH manual and ours (i.e., s1-s6 intersection)."""
    manual_eps = set()
    for p in MANUAL_DIR.rglob("friends_*_manualseg.tsv"):
        m = re.match(r"friends_(s\d{2}e\d{2}[a-z])_manualseg\.tsv", p.name)
        if m:
            manual_eps.add(m.group(1))
    ours_eps = set()
    for p in OURS_DIR.rglob("friends_*_scene_summary.tsv"):
        m = re.match(r"friends_(s\d{2}e\d{2}[a-z])_scene_summary\.tsv", p.name)
        if m:
            ours_eps.add(m.group(1))
    return sorted(manual_eps & ours_eps)


def load_manual_raw(episode: str) -> pd.DataFrame:
    """Raw manual TSV, one row per segment, with all ONbond_* columns."""
    return pd.read_csv(manual_path(episode), sep="\t")


def collapse_to_scenes(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("scene", as_index=False)
        .agg(start=("onset", "min"), end=("offset", "max"))
        .sort_values("start")
        .reset_index(drop=True)
    )


def as_segments(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.drop(columns=["scene"])
        .rename(columns={"segment": "scene", "onset": "start", "offset": "end"})[
            ["scene", "start", "end"]
        ]
        .sort_values("start")
        .reset_index(drop=True)
    )


def load_ours(episode: str) -> pd.DataFrame:
    df = pd.read_csv(ours_path(episode), sep="\t")
    return df.rename(columns={"scene_id": "scene"})[["scene", "start", "end"]].copy()


def boundaries(scenes: pd.DataFrame) -> list[float]:
    if len(scenes) <= 1:
        return []
    return scenes["end"].iloc[:-1].tolist()


def clip_to_window(scenes: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    keep = scenes[(scenes["end"] > lo) & (scenes["start"] < hi)].copy()
    keep["start"] = keep["start"].clip(lower=lo)
    keep["end"] = keep["end"].clip(upper=hi)
    return keep.reset_index(drop=True)


def match_boundaries(gold: list[float], pred: list[float], tol: float) -> dict:
    """Greedy nearest-first 1-to-1 match within tolerance."""
    gold_sorted = sorted(gold)
    pred_sorted = sorted(pred)
    used = [False] * len(pred_sorted)
    tp = 0
    matched_for_gold: list[float | None] = []
    for g in gold_sorted:
        best_j, best_d = -1, tol + 1
        for j, p in enumerate(pred_sorted):
            if used[j]:
                continue
            d = abs(p - g)
            if d <= tol and d < best_d:
                best_j, best_d = j, d
        if best_j >= 0:
            used[best_j] = True
            tp += 1
            matched_for_gold.append(pred_sorted[best_j])
        else:
            matched_for_gold.append(None)
    fn = len(gold_sorted) - tp
    fp = len(pred_sorted) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "matched_for_gold": matched_for_gold,
        "gold_sorted": gold_sorted,
    }


def iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def per_unit_iou(gold: pd.DataFrame, pred: pd.DataFrame) -> pd.Series:
    ious = []
    for _, g in gold.iterrows():
        best = 0.0
        for _, p in pred.iterrows():
            v = iou(g["start"], g["end"], p["start"], p["end"])
            if v > best:
                best = v
        ious.append(best)
    return pd.Series(ious, dtype=float)


def evaluate_episode(episode: str) -> dict:
    raw = load_manual_raw(episode)
    pred = load_ours(episode)

    # window from full (uncollapsed) coverage
    gold_full_start = raw["onset"].min()
    gold_full_end = raw["offset"].max()
    lo = max(gold_full_start, pred["start"].min())
    hi = min(gold_full_end, pred["end"].max())

    pred_w = clip_to_window(pred, lo, hi)
    pred_b = boundaries(pred_w)

    out = {
        "episode": episode,
        "season": int(episode[1:3]),
        "window": (lo, hi),
        "pred_full_n": int(len(pred)),
        "pred_window_n": int(len(pred_w)),
        "pred_window_n_b": len(pred_b),
    }

    for unit_name, builder in (("scene", collapse_to_scenes), ("segment", as_segments)):
        gold = builder(raw)
        gold_w = clip_to_window(gold, lo, hi)
        gold_b = boundaries(gold_w)

        unit_out = {
            "gold_full_n": int(len(gold)),
            "gold_window_n": int(len(gold_w)),
            "gold_window_n_b": len(gold_b),
            "ious": per_unit_iou(gold_w, pred_w),
        }
        for tol in TOLERANCES:
            m = match_boundaries(gold_b, pred_b, tol)
            unit_out[f"tol_{int(tol)}s"] = {
                k: m[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")
            }
        out[unit_name] = unit_out

    # diagnostic: each gold segment-boundary, tagged by ONbond reasons
    raw_sorted = raw.sort_values("onset").reset_index(drop=True)
    seg_bounds_full = raw_sorted["offset"].iloc[:-1].tolist()
    seg_bounds_in_win = [(i, b) for i, b in enumerate(seg_bounds_full) if lo <= b <= hi]
    pred_b_sorted = sorted(pred_b)
    diag_rows = []
    for idx, b in seg_bounds_in_win:
        # bond reasons come from the row STARTING at this boundary (next segment's ONbond)
        if idx + 1 >= len(raw_sorted):
            reasons = []
        else:
            nxt = raw_sorted.iloc[idx + 1]
            reasons = [c.replace("ONbond_", "") for c in ONBOND_COLS if bool(nxt.get(c, False))]
        nearest_pred = min((abs(p - b) for p in pred_b_sorted), default=float("inf"))
        diag_rows.append({
            "episode": episode,
            "season": int(episode[1:3]),
            "boundary_time": round(b, 3),
            "bond_reasons": ";".join(reasons) if reasons else "none",
            "nearest_pred_dist": round(nearest_pred, 3) if nearest_pred != float("inf") else None,
            "matched_at_5s": bool(nearest_pred <= DIAG_TOL),
        })
    out["diagnostics"] = diag_rows
    return out


def long_rows(r: dict) -> list[dict]:
    rows = []
    for unit in ("scene", "segment"):
        u = r[unit]
        ious = u["ious"]
        row = {
            "episode": r["episode"],
            "season": r["season"],
            "unit": unit,
            "window_lo": round(r["window"][0], 3),
            "window_hi": round(r["window"][1], 3),
            "gold_full_n": u["gold_full_n"],
            "gold_window_n": u["gold_window_n"],
            "gold_window_n_b": u["gold_window_n_b"],
            "pred_full_n": r["pred_full_n"],
            "pred_window_n": r["pred_window_n"],
            "pred_window_n_b": r["pred_window_n_b"],
            "iou_mean": round(ious.mean(), 4) if len(ious) else 0.0,
            "iou_median": round(ious.median(), 4) if len(ious) else 0.0,
            "iou_ge_0.5": round((ious >= 0.5).mean(), 4) if len(ious) else 0.0,
        }
        for tol in TOLERANCES:
            m = u[f"tol_{int(tol)}s"]
            row[f"tp@{int(tol)}s"] = m["tp"]
            row[f"fp@{int(tol)}s"] = m["fp"]
            row[f"fn@{int(tol)}s"] = m["fn"]
            row[f"P@{int(tol)}s"] = round(m["precision"], 4)
            row[f"R@{int(tol)}s"] = round(m["recall"], 4)
            row[f"F1@{int(tol)}s"] = round(m["f1"], 4)
        rows.append(row)
    return rows


def counts_row(r: dict) -> dict:
    return {
        "episode": r["episode"],
        "season": r["season"],
        "gold_n_scenes": r["scene"]["gold_full_n"],
        "gold_n_segments": r["segment"]["gold_full_n"],
        "pred_n_scenes": r["pred_full_n"],
        "gold_n_scenes_win": r["scene"]["gold_window_n"],
        "gold_n_segments_win": r["segment"]["gold_window_n"],
        "pred_n_scenes_win": r["pred_window_n"],
    }


def aggregate(long_df: pd.DataFrame, counts_df: pd.DataFrame) -> dict:
    agg: dict = {"n_episodes": int(counts_df.shape[0])}
    for unit in ("scene", "segment"):
        sub = long_df[long_df["unit"] == unit]
        agg[unit] = {
            "F1@2s_mean": round(sub["F1@2s"].mean(), 4),
            "F1@5s_mean": round(sub["F1@5s"].mean(), 4),
            "F1@10s_mean": round(sub["F1@10s"].mean(), 4),
            "P@5s_mean": round(sub["P@5s"].mean(), 4),
            "R@5s_mean": round(sub["R@5s"].mean(), 4),
            "iou_mean_mean": round(sub["iou_mean"].mean(), 4),
            "iou_ge_0.5_mean": round(sub["iou_ge_0.5"].mean(), 4),
            "by_season": {
                int(s): {
                    "n_eps": int(g.shape[0]),
                    "F1@5s_mean": round(g["F1@5s"].mean(), 4),
                    "iou_mean_mean": round(g["iou_mean"].mean(), 4),
                }
                for s, g in sub.groupby("season")
            },
        }
    agg["counts"] = {
        "pred_n_scenes_mean": round(counts_df["pred_n_scenes"].mean(), 2),
        "pred_n_scenes_std": round(counts_df["pred_n_scenes"].std(), 2),
        "gold_n_scenes_mean": round(counts_df["gold_n_scenes"].mean(), 2),
        "gold_n_scenes_std": round(counts_df["gold_n_scenes"].std(), 2),
        "gold_n_segments_mean": round(counts_df["gold_n_segments"].mean(), 2),
        "gold_n_segments_std": round(counts_df["gold_n_segments"].std(), 2),
        "corr_pred_vs_gold_scenes": round(
            counts_df["pred_n_scenes"].corr(counts_df["gold_n_scenes"]), 4
        ),
        "corr_pred_vs_gold_segments": round(
            counts_df["pred_n_scenes"].corr(counts_df["gold_n_segments"]), 4
        ),
    }
    return agg


def diag_summary(diag_df: pd.DataFrame) -> dict:
    """Tally match-rate at ±5s by ONbond reason, breaking out single-reason boundaries
    from multi-reason ones."""
    out: dict = {"n_boundaries": int(diag_df.shape[0])}
    single = diag_df.copy()
    single["reasons_split"] = single["bond_reasons"].str.split(";")
    single["n_reasons"] = single["reasons_split"].apply(len)
    out["overall_match_rate@5s"] = round(diag_df["matched_at_5s"].mean(), 4)

    by_single = {}
    for reason in [c.replace("ONbond_", "") for c in ONBOND_COLS] + ["none"]:
        mask = (single["n_reasons"] == 1) & (single["bond_reasons"] == reason)
        if mask.sum() > 0:
            by_single[reason] = {
                "n": int(mask.sum()),
                "match_rate@5s": round(single.loc[mask, "matched_at_5s"].mean(), 4),
            }
    out["single_reason"] = by_single

    # any-reason marginals (boundary counted once per reason it carries)
    any_reason: dict = {}
    for reason in [c.replace("ONbond_", "") for c in ONBOND_COLS]:
        mask = single["reasons_split"].apply(lambda lst: reason in lst)
        if mask.sum() > 0:
            any_reason[reason] = {
                "n": int(mask.sum()),
                "match_rate@5s": round(single.loc[mask, "matched_at_5s"].mean(), 4),
            }
    out["any_reason"] = any_reason
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--episodes",
        default="ALL",
        help="Comma-separated episode ids, or 'ALL' (s1-s6 intersection).",
    )
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--ours-dir", default=str(DEFAULT_OURS_DIR),
                    help="Directory of predicted scene_summary.tsv files "
                         "(default: output/annotations/scenes).")
    args = ap.parse_args()

    set_ours_dir(Path(args.ours_dir).resolve())
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.episodes == "ALL":
        episodes = discover_episodes()
    else:
        episodes = [e.strip() for e in args.episodes.split(",") if e.strip()]

    print(f"Evaluating {len(episodes)} episodes...")

    long_rows_all: list[dict] = []
    counts_rows: list[dict] = []
    diag_rows_all: list[dict] = []
    for ep in episodes:
        r = evaluate_episode(ep)
        long_rows_all.extend(long_rows(r))
        counts_rows.append(counts_row(r))
        diag_rows_all.extend(r["diagnostics"])

    long_df = pd.DataFrame(long_rows_all)
    counts_df = pd.DataFrame(counts_rows)
    diag_df = pd.DataFrame(diag_rows_all)

    long_path = out_dir / "per_episode.tsv"
    counts_path = out_dir / "per_episode_counts.tsv"
    diag_path = out_dir / "boundary_diagnostics.tsv"
    agg_path = out_dir / "aggregate.json"

    long_df.to_csv(long_path, sep="\t", index=False)
    counts_df.to_csv(counts_path, sep="\t", index=False)
    diag_df.to_csv(diag_path, sep="\t", index=False)

    agg = aggregate(long_df, counts_df)
    agg["diagnostics"] = diag_summary(diag_df)
    agg_path.write_text(json.dumps(agg, indent=2))

    print(f"\nWrote {long_path.relative_to(REPO)} ({long_df.shape[0]} rows)")
    print(f"Wrote {counts_path.relative_to(REPO)} ({counts_df.shape[0]} rows)")
    print(f"Wrote {diag_path.relative_to(REPO)} ({diag_df.shape[0]} rows)")
    print(f"Wrote {agg_path.relative_to(REPO)}")

    print("\n=== Aggregate (means across {} episodes) ===".format(agg["n_episodes"]))
    for unit in ("scene", "segment"):
        u = agg[unit]
        print(f"  [{unit}]  F1@2s={u['F1@2s_mean']:.3f}  F1@5s={u['F1@5s_mean']:.3f}  "
              f"F1@10s={u['F1@10s_mean']:.3f}  P@5s={u['P@5s_mean']:.3f}  R@5s={u['R@5s_mean']:.3f}  "
              f"iou_mean={u['iou_mean_mean']:.3f}  iou≥0.5={u['iou_ge_0.5_mean']:.3f}")
    c = agg["counts"]
    print(f"\nCounts: pred={c['pred_n_scenes_mean']:.1f}±{c['pred_n_scenes_std']:.1f}  "
          f"gold_scenes={c['gold_n_scenes_mean']:.1f}±{c['gold_n_scenes_std']:.1f}  "
          f"gold_segments={c['gold_n_segments_mean']:.1f}±{c['gold_n_segments_std']:.1f}")
    print(f"corr(pred_n, gold_scenes)   = {c['corr_pred_vs_gold_scenes']:.3f}")
    print(f"corr(pred_n, gold_segments) = {c['corr_pred_vs_gold_segments']:.3f}")

    d = agg["diagnostics"]
    print(f"\nBoundary diagnostic: {d['n_boundaries']} gold segment-boundaries, "
          f"overall match@5s = {d['overall_match_rate@5s']:.3f}")
    print("  By ANY reason (boundary counted under each reason it carries):")
    for reason, stats in sorted(d["any_reason"].items(), key=lambda kv: -kv[1]["match_rate@5s"]):
        print(f"    {reason:>16}  n={stats['n']:4d}  match@5s={stats['match_rate@5s']:.3f}")
    print("  By SINGLE reason (boundary carries exactly one reason):")
    for reason, stats in sorted(d["single_reason"].items(), key=lambda kv: -kv[1]["match_rate@5s"]):
        print(f"    {reason:>16}  n={stats['n']:4d}  match@5s={stats['match_rate@5s']:.3f}")


if __name__ == "__main__":
    main()
