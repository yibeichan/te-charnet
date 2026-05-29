"""Generic scene-subdivision plumbing shared by all augmenters.

Reads a fan-transcript ``scene_summary.tsv``, asks a per-scene ``propose``
callback for interior sub-boundary times, and rewrites the table with the
new boundaries (renumbered scene ids, inherited descriptions tagged with an
augmentation suffix).
"""
from __future__ import annotations

import csv  # noqa: F401
import re
from collections.abc import Callable  # noqa: F401
from dataclasses import dataclass  # noqa: F401
from pathlib import Path

import pandas as pd  # noqa: F401

SCENE_GLOB = "friends_*_scene_summary.tsv"
_EP_RE = re.compile(r"friends_(s\d{2}e\d{2}[a-z])_scene_summary\.tsv")


def _all_episodes(scenes_in_dir: Path) -> list[str]:
    eps = []
    for p in scenes_in_dir.rglob(SCENE_GLOB):
        m = _EP_RE.match(p.name)
        if m:
            eps.append(m.group(1))
    return sorted(eps)


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
    return [e.strip() for e in spec.split(",") if e.strip()]
