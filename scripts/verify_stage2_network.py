#!/usr/bin/env python3
"""Independent cross-check of the stage-2 networks.

Reconstructs each episode's per-scene and aggregate interaction graphs *from
scratch* off the tracked speaker tables (a different code path than
``charnet.network`` — nothing is imported from it) and confirms the committed
``output/02_build_network/<ep>/temporal_network.json`` and
``episode_network.json`` match, plus a layer of structural invariants. Together
with ``verify_network_export.py`` this independently verifies the whole chain:
tracked speaker table -> stage-2 graph -> network-metric export.

Usage:
    python scripts/verify_stage2_network.py \
        --tables-root output/annotations/sentences \
        --network-root output/02_build_network

Exit 0 iff every reconstruction value, every aggregate, and every invariant
holds; 1 on any mismatch/violation; 2 if nothing could be checked.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULTS = {
    "weight_adjacency": 1.0,
    "weight_proximity": 0.5,
    "weight_copresence": 0.25,
    "proximity_window": 3,
}

# Half-ulp of 4-decimal rounding; used to size the weight-formula invariant slack.
HALF_ULP_4DP = 5e-5


def _round4(x: float) -> float:
    return round(float(x), 4)


def read_table_rows(path: Path) -> list[dict]:
    """Read a speaker TSV, mirroring stage-2's combined two-phase row filter.

    keep_default_na=False keeps blank cells as "" (default pd.read_csv would
    make them NaN, which str()/float() would silently let through).
    Preserves TSV row order (matters for stable sort on tied start times).
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    rows: list[dict] = []
    for _, r in df.iterrows():
        scene_raw = str(r.get("scene_id", "")).strip()
        start_raw = str(r.get("start", "")).strip()
        end_raw = str(r.get("end", "")).strip()
        speaker = str(r.get("speaker", "")).strip()
        if not start_raw:                      # phase (a): scene-marker rows
            continue
        if not scene_raw or not speaker:       # phase (b): empty scene_id/speaker
            continue
        try:
            scene_id = int(float(scene_raw))
            start = float(start_raw)
            end = float(end_raw)
        except (TypeError, ValueError):        # phase (b): coercion failures
            continue
        rows.append({"scene_id": scene_id, "start": start, "end": end, "speaker": speaker})
    return rows
