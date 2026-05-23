"""Augment fan-transcript scene boundaries with character-presence subdivisions.

For each scene in the input scene_summary.tsv, scan the char-tracker stage-05
per-second grid restricted to that scene. Divide the scene into fixed-width
tiles, compute each tile's "active main-cast set" (chars present in
>= PRESENCE_FRAC of tile seconds), and propose a sub-boundary at the tile
boundary where the Jaccard distance between adjacent tiles exceeds
JACCARD_THRESH — gated by a minimum-spacing rule against existing scene
boundaries (and against each other).

Outputs an augmented scene_summary.tsv with the same column schema. New
sub-scene rows inherit the parent scene_desc with a "[char_aug N]" suffix
and synthesised shot_ids (empty for now).

Usage:
  python scripts/augment_scenes_char_presence.py \\
      --episodes s01e01a,s02e10a \\
      --out-dir output/annotations/scenes_char_aug
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from charnet.visual_presence import (
    DEFAULT_CHAR_TRACKER_DIR,
    char_tracker_csv_path,
    load_char_tracker_grid,
    resolve_char_tracker_dir,
)

DEFAULT_SCENES_IN = REPO / "output/annotations/scenes"
DEFAULT_SCENES_OUT = REPO / "output/annotations/scenes_char_aug"
DEFAULT_SHOTS_DIR = REPO / "data/friends_annotations/annotation_results/TSVpyscene"

TILE_SECS = 5.0
PRESENCE_FRAC = 0.20
JACCARD_THRESH = 0.5
MIN_SPACING_SECS = 15.0
PERSISTENCE_TILES = 2  # new set must hold for >=N tiles after the boundary
SHOT_SNAP_WINDOW = 3.0  # snap to nearest shot transition within ±W; drop if none
MIN_SCENE_LENGTH = 0.0  # only subdivide scenes >= this length (seconds)


def jaccard_distance(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def load_shot_transitions(episode: str, shots_dir: Path) -> list[float]:
    """Return shot transition times (onsets after the first) in seconds."""
    season = int(episode[1:3])
    path = shots_dir / f"s{season}" / f"friends_{episode}_pyscene.tsv"
    if not path.exists():
        return []
    df = pd.read_csv(path, sep="\t")
    if "onset" not in df.columns:
        return []
    return [float(o) for o in df["onset"].iloc[1:].tolist()]


def snap_to_shot(boundary: float, shot_times: list[float], snap_window: float) -> float | None:
    """Nearest shot transition within ±snap_window. None if none."""
    if not shot_times:
        return None
    best, best_d = None, snap_window + 1.0
    for st in shot_times:
        d = abs(st - boundary)
        if d <= snap_window and d < best_d:
            best, best_d = st, d
    return best


def tile_active_set(rows: list[list[int]], chars: list[str],
                    tile_start: int, tile_end: int, presence_frac: float) -> set[str]:
    """Set of chars active in >= presence_frac of seconds in [tile_start, tile_end)."""
    tile_end = min(tile_end, len(rows))
    n = max(1, tile_end - tile_start)
    counts = [0] * len(chars)
    for sec in range(tile_start, tile_end):
        for ci, v in enumerate(rows[sec]):
            counts[ci] += v
    return {chars[ci] for ci, c in enumerate(counts) if c / n >= presence_frac}


def propose_sub_boundaries(
    scene_start: float,
    scene_end: float,
    grid_chars: list[str],
    grid_rows: list[list[int]],
    *,
    tile_secs: float = TILE_SECS,
    presence_frac: float = PRESENCE_FRAC,
    jaccard_thresh: float = JACCARD_THRESH,
    min_spacing: float = MIN_SPACING_SECS,
    persistence_tiles: int = PERSISTENCE_TILES,
    shot_times: list[float] | None = None,
    shot_snap_window: float = SHOT_SNAP_WINDOW,
    shot_snap_required: bool = True,
    min_scene_length: float = MIN_SCENE_LENGTH,
) -> list[float]:
    """Return list of sub-boundary times inside (scene_start, scene_end)."""
    if scene_end - scene_start < max(2 * min_spacing, min_scene_length):
        return []
    s0 = int(scene_start)
    s1 = min(int(scene_end), len(grid_rows))
    if s1 - s0 < int(2 * tile_secs):
        return []

    # tile the scene
    tiles = []  # list of (tile_start_sec, tile_end_sec, active_set)
    cursor = s0
    while cursor + tile_secs <= s1:
        te = int(cursor + tile_secs)
        active = tile_active_set(grid_rows, grid_chars, cursor, te, presence_frac)
        tiles.append((cursor, te, active))
        cursor += int(tile_secs)
    if len(tiles) < 2:
        return []

    # find tile-to-tile jumps, gated by persistence
    candidates: list[float] = []
    for i in range(1, len(tiles)):
        prev_set = tiles[i - 1][2]
        cur_set = tiles[i][2]
        if not prev_set and not cur_set:
            continue
        d = jaccard_distance(prev_set, cur_set)
        if d < jaccard_thresh:
            continue
        # persistence: the new set must remain "close" to cur_set over the next
        # persistence_tiles tiles (mean Jaccard distance from cur_set < jaccard_thresh).
        future = tiles[i : i + persistence_tiles]
        if len(future) < persistence_tiles:
            continue
        distances = [jaccard_distance(cur_set, ft[2]) for ft in future]
        if max(distances) >= jaccard_thresh:
            # the new set didn't hold — likely a transient flicker
            continue
        candidates.append(float(tiles[i][0]))

    # shot-snap: replace each candidate with the nearest shot transition within
    # ±shot_snap_window; drop if shot_snap_required and no transition in range.
    if shot_times is not None:
        snapped: list[float] = []
        for c in candidates:
            s = snap_to_shot(c, shot_times, shot_snap_window)
            if s is not None:
                snapped.append(s)
            elif not shot_snap_required:
                snapped.append(c)
        candidates = snapped

    # gate: must be at least min_spacing from scene endpoints and each other
    accepted: list[float] = []
    for b in sorted(candidates):
        if b - scene_start < min_spacing or scene_end - b < min_spacing:
            continue
        if accepted and b - accepted[-1] < min_spacing:
            continue
        accepted.append(b)
    return accepted


def augment_episode(
    episode: str,
    scenes_in_dir: Path,
    scenes_out_dir: Path,
    char_tracker_dir: Path,
    shots_dir: Path,
    *,
    tile_secs: float = TILE_SECS,
    presence_frac: float = PRESENCE_FRAC,
    jaccard_thresh: float = JACCARD_THRESH,
    min_spacing: float = MIN_SPACING_SECS,
    persistence_tiles: int = PERSISTENCE_TILES,
    shot_snap_window: float = SHOT_SNAP_WINDOW,
    shot_snap_required: bool = True,
    min_scene_length: float = MIN_SCENE_LENGTH,
) -> dict:
    season = int(episode[1:3])
    in_path = scenes_in_dir / f"s{season}" / f"friends_{episode}_scene_summary.tsv"
    out_dir = scenes_out_dir / f"s{season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"friends_{episode}_scene_summary.tsv"

    df = pd.read_csv(in_path, sep="\t")
    grid_path = char_tracker_csv_path(char_tracker_dir, f"friends_{episode}")
    if grid_path is None:
        # No char-tracker data → just copy through
        df.to_csv(out_path, sep="\t", index=False)
        return {"episode": episode, "n_input_scenes": len(df), "n_output_scenes": len(df),
                "n_new_boundaries": 0, "char_tracker_present": False}
    chars, rows = load_char_tracker_grid(grid_path)
    shot_times = load_shot_transitions(episode, shots_dir)

    new_rows: list[dict] = []
    n_new = 0
    next_id = 1
    for _, row in df.iterrows():
        start = float(row["start"])
        end = float(row["end"])
        desc = str(row.get("scene_desc", "") or "")
        shot_ids = str(row.get("shot_ids", "") or "")
        subs = propose_sub_boundaries(
            start, end, chars, rows,
            tile_secs=tile_secs,
            presence_frac=presence_frac,
            jaccard_thresh=jaccard_thresh,
            min_spacing=min_spacing,
            persistence_tiles=persistence_tiles,
            shot_times=shot_times if shot_times else None,
            shot_snap_window=shot_snap_window,
            shot_snap_required=shot_snap_required,
            min_scene_length=min_scene_length,
        )
        if not subs:
            new_rows.append({
                "scene_id": next_id,
                "scene_desc": desc,
                "start": f"{start:.2f}",
                "end": f"{end:.2f}",
                "shot_ids": shot_ids,
            })
            next_id += 1
            continue
        n_new += len(subs)
        bounds = [start] + subs + [end]
        for k in range(len(bounds) - 1):
            sub_start, sub_end = bounds[k], bounds[k + 1]
            sub_desc = desc if k == 0 else f"{desc} [char_aug {k}]"
            new_rows.append({
                "scene_id": next_id,
                "scene_desc": sub_desc,
                "start": f"{sub_start:.2f}",
                "end": f"{sub_end:.2f}",
                "shot_ids": shot_ids if k == 0 else "",
            })
            next_id += 1

    out_df = pd.DataFrame(new_rows, columns=["scene_id", "scene_desc", "start", "end", "shot_ids"])
    out_df.to_csv(out_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    return {
        "episode": episode,
        "n_input_scenes": len(df),
        "n_output_scenes": len(out_df),
        "n_new_boundaries": n_new,
        "char_tracker_present": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--episodes",
        required=True,
        help="Comma-separated episode ids, or 'ALL' for every scene_summary.tsv under --scenes-in.",
    )
    ap.add_argument("--scenes-in", default=str(DEFAULT_SCENES_IN))
    ap.add_argument("--scenes-out", default=str(DEFAULT_SCENES_OUT))
    ap.add_argument("--char-tracker-dir", default=None,
                    help="Override stage-05 dir (default: env CHAR_TRACKER_DIR or scratch).")
    ap.add_argument("--tile-secs", type=float, default=TILE_SECS)
    ap.add_argument("--presence-frac", type=float, default=PRESENCE_FRAC)
    ap.add_argument("--jaccard-thresh", type=float, default=JACCARD_THRESH)
    ap.add_argument("--min-spacing", type=float, default=MIN_SPACING_SECS)
    ap.add_argument("--persistence-tiles", type=int, default=PERSISTENCE_TILES)
    ap.add_argument("--shots-dir", default=str(DEFAULT_SHOTS_DIR))
    ap.add_argument("--shot-snap-window", type=float, default=SHOT_SNAP_WINDOW)
    ap.add_argument("--no-shot-snap-required", action="store_true",
                    help="If set, keep candidates that have no nearby shot transition.")
    ap.add_argument("--min-scene-length", type=float, default=MIN_SCENE_LENGTH,
                    help="Only subdivide scenes whose length >= this many seconds.")
    args = ap.parse_args()

    scenes_in = Path(args.scenes_in)
    scenes_out = Path(args.scenes_out)
    shots_dir = Path(args.shots_dir)
    ct_dir = resolve_char_tracker_dir(args.char_tracker_dir) or Path(DEFAULT_CHAR_TRACKER_DIR)

    if args.episodes == "ALL":
        episodes = sorted(
            re.match(r"friends_(s\d{2}e\d{2}[a-z])_scene_summary\.tsv", p.name).group(1)
            for p in scenes_in.rglob("friends_*_scene_summary.tsv")
        )
    else:
        episodes = [e.strip() for e in args.episodes.split(",") if e.strip()]

    print(f"Augmenting {len(episodes)} episodes")
    print(f"  scenes_in: {scenes_in}")
    print(f"  scenes_out: {scenes_out}")
    print(f"  char_tracker_dir: {ct_dir}")
    print(f"  shots_dir: {shots_dir}")
    print(f"  params: tile_secs={args.tile_secs} presence_frac={args.presence_frac} "
          f"jaccard_thresh={args.jaccard_thresh} min_spacing={args.min_spacing} "
          f"persistence_tiles={args.persistence_tiles} "
          f"shot_snap_window={args.shot_snap_window} "
          f"shot_snap_required={not args.no_shot_snap_required}")
    print()

    total_in = total_out = total_new = total_no_ct = 0
    for ep in episodes:
        r = augment_episode(
            ep, scenes_in, scenes_out, ct_dir, shots_dir,
            tile_secs=args.tile_secs,
            presence_frac=args.presence_frac,
            jaccard_thresh=args.jaccard_thresh,
            min_spacing=args.min_spacing,
            persistence_tiles=args.persistence_tiles,
            shot_snap_window=args.shot_snap_window,
            shot_snap_required=not args.no_shot_snap_required,
            min_scene_length=args.min_scene_length,
        )
        total_in += r["n_input_scenes"]
        total_out += r["n_output_scenes"]
        total_new += r["n_new_boundaries"]
        total_no_ct += 0 if r["char_tracker_present"] else 1
        ct_flag = " " if r["char_tracker_present"] else "*"
        print(f"  {ct_flag} {ep}: {r['n_input_scenes']:>3} → {r['n_output_scenes']:>3} "
              f"(+{r['n_new_boundaries']})")
    print()
    print(f"Totals: scenes {total_in} → {total_out} (+{total_new} sub-boundaries) "
          f"over {len(episodes)} eps ({total_no_ct} without char-tracker)")


if __name__ == "__main__":
    main()
