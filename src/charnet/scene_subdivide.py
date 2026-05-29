"""Generic scene-subdivision plumbing shared by all augmenters.

Reads a fan-transcript ``scene_summary.tsv``, asks a per-scene ``propose``
callback for interior sub-boundary times, and rewrites the table with the
new boundaries (renumbered scene ids, inherited descriptions tagged with an
augmentation suffix).
"""
from __future__ import annotations

import csv
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SCENE_GLOB = "friends_*_scene_summary.tsv"
_EP_RE = re.compile(r"friends_(s\d{2}e\d{2}[a-z])_scene_summary\.tsv")


def _all_episodes(scenes_in_dir: Path) -> list[str]:
    """Return sorted episode ids discovered under *scenes_in_dir*.

    *scenes_in_dir* must be the *root* of the per-season scene tree (e.g.
    ``output/scenes/``); the function discovers episodes recursively via
    ``rglob``, so callers should pass the root, not a season subdirectory.
    """
    eps = []
    for p in scenes_in_dir.rglob(SCENE_GLOB):
        m = _EP_RE.match(p.name)
        if m:
            eps.append(m.group(1))
    return sorted(eps)


@dataclass(frozen=True)
class Scene:
    scene_id: int
    scene_desc: str
    start: float
    end: float
    shot_ids: str


ProposeFn = Callable[[Scene], list[float]]


def subdivide_episode(
    episode: str,
    scenes_in_dir: Path,
    scenes_out_dir: Path,
    propose: ProposeFn,
    *,
    aug_tag: str,
) -> dict:
    """Rewrite one episode's scene table, inserting proposed sub-boundaries.

    *propose* receives each input :class:`Scene` and returns a list of strictly
    interior boundary times. Output rows are renumbered 1..N; sub-scenes after
    the first inherit ``scene_desc`` with a ``[<aug_tag> k]`` suffix and an
    empty ``shot_ids``.

    Returns a stats dict; "n_new_boundaries" is the total count of interior
    cut-points inserted across all scenes.
    """
    season = int(episode[1:3])
    in_path = scenes_in_dir / f"s{season}" / f"friends_{episode}_scene_summary.tsv"
    out_dir = scenes_out_dir / f"s{season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"friends_{episode}_scene_summary.tsv"

    df = pd.read_csv(in_path, sep="\t")
    new_rows: list[dict] = []
    n_new = 0
    next_id = 1
    for _, row in df.iterrows():
        scene = Scene(
            scene_id=int(row["scene_id"]),
            scene_desc=str(row.get("scene_desc", "") or ""),
            start=float(row["start"]),
            end=float(row["end"]),
            shot_ids=str(row.get("shot_ids", "") or ""),
        )
        subs = [b for b in sorted(set(propose(scene))) if scene.start < b < scene.end]
        if not subs:
            new_rows.append({
                "scene_id": next_id, "scene_desc": scene.scene_desc,
                "start": f"{scene.start:.2f}", "end": f"{scene.end:.2f}",
                "shot_ids": scene.shot_ids,
            })
            next_id += 1
            continue
        n_new += len(subs)
        bounds = [scene.start] + subs + [scene.end]
        for k in range(len(bounds) - 1):
            desc = scene.scene_desc if k == 0 else f"{scene.scene_desc} [{aug_tag} {k}]"
            new_rows.append({
                "scene_id": next_id, "scene_desc": desc,
                "start": f"{bounds[k]:.2f}", "end": f"{bounds[k + 1]:.2f}",
                "shot_ids": scene.shot_ids if k == 0 else "",
            })
            next_id += 1

    out_df = pd.DataFrame(new_rows, columns=["scene_id", "scene_desc", "start", "end", "shot_ids"])
    out_df.to_csv(out_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    return {
        "episode": episode,
        "n_input_scenes": len(df),
        "n_output_scenes": len(out_df),
        "n_new_boundaries": n_new,
    }


def expand_episode_spec(spec: str, scenes_in_dir: Path) -> list[str]:
    """Expand an episode spec into a sorted episode-id list.

    Accepts: ``ALL``; a single season ``s3``; a season range ``s3-s6``
    (inclusive); or a comma-separated explicit list ``s01e01a,s02e03b``.
    Season specs filter the episodes actually present under *scenes_in_dir*.
    """
    spec = spec.strip()
    if spec == "ALL":
        return _all_episodes(scenes_in_dir)
    range_m = re.fullmatch(r"s(\d+)-s(\d+)", spec)
    if range_m:
        lo, hi = int(range_m.group(1)), int(range_m.group(2))
        return [e for e in _all_episodes(scenes_in_dir) if lo <= int(e[1:3]) <= hi]
    single_m = re.fullmatch(r"s(\d+)", spec)
    if single_m:
        n = int(single_m.group(1))
        return [e for e in _all_episodes(scenes_in_dir) if int(e[1:3]) == n]
    _ID_RE = re.compile(r"^s\d{2}e\d{2}[a-z]$")
    ids = [e.strip() for e in spec.split(",") if e.strip()]
    for e in ids:
        if not _ID_RE.match(e):
            raise ValueError(f"Invalid episode id: {e!r}")
    return ids
