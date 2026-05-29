# src/charnet/char_presence.py
"""Character-presence sub-boundary proposer (prototype #1).

Lifted from scripts/augment_scenes_char_presence.py so the hybrid augmenter
can import propose_sub_boundaries. Logic is unchanged; see
docs/scene_segmentation_evaluation.md "Prototype #1 results".
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_SHOTS_DIR = REPO / "data/friends_annotations/annotation_results/TSVpyscene"

TILE_SECS = 5.0
PRESENCE_FRAC = 0.20
JACCARD_THRESH = 0.5
MIN_SPACING_SECS = 15.0
PERSISTENCE_TILES = 2  # new set must hold for >=N tiles after the boundary
SHOT_SNAP_WINDOW = 3.0  # snap to nearest shot transition within ±W; drop if none
MIN_SCENE_LENGTH = 0.0


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
