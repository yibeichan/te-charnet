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

import itertools
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

    Replicates the combined production filter:
    load_corrected_speaker_rows (empty start) +
    build_temporal_network_from_aligned_rows (empty scene_id/speaker,
    coercion failures).

    keep_default_na=False keeps blank cells as "" (default pd.read_csv would
    make them NaN, which str()/float() would silently let through).
    Preserves TSV row order (matters for stable sort on tied start times).
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"scene_id", "start", "end", "speaker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required column(s): {sorted(missing)}")
    rows: list[dict] = []
    for _, r in df.iterrows():
        scene_raw = str(r.get("scene_id", "")).strip()
        start_raw = str(r.get("start", "")).strip()
        end_raw = str(r.get("end", "")).strip()
        speaker = str(r.get("speaker", "")).strip()
        if not start_raw:                      # phase (a): load_corrected_speaker_rows drops empty-start (scene-marker) rows
            continue
        if not scene_raw or not speaker:       # phase (b): build_temporal_network_from_aligned_rows drops empty scene_id/speaker
            continue
        try:
            scene_id = int(float(scene_raw))   # mirrors network.py's int(float(...)) coercion
            start = float(start_raw)
            end = float(end_raw)
        except (TypeError, ValueError):        # phase (b): network.py skips rows whose scene_id/start/end fail coercion
            continue
        rows.append({"scene_id": scene_id, "start": start, "end": end, "speaker": speaker})
    return rows


def _adjacency(speakers_seq: list[str]) -> dict[tuple[str, str], float]:
    counts: dict[tuple[str, str], float] = {}
    for i in range(1, len(speakers_seq)):
        a, b = speakers_seq[i - 1], speakers_seq[i]
        if a != b and a and b:
            key = tuple(sorted([a, b]))
            counts[key] = counts.get(key, 0.0) + 1.0
    return counts


def _proximity(speakers_seq: list[str], window: int) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for i, a in enumerate(speakers_seq):
        if not a:
            continue
        for j in range(i + 1, min(i + window + 1, len(speakers_seq))):
            b = speakers_seq[j]
            if not b or b == a:
                continue
            key = tuple(sorted([a, b]))
            scores[key] = scores.get(key, 0.0) + 1.0 / (j - i)
    return scores


def reconstruct_scenes(rows: list[dict], params: dict) -> list[dict]:
    """Rebuild per-scene graphs independently of charnet.network.

    `rows` must be in TSV order so the stable sort-by-start reproduces stage-2's
    tie handling. Returns scene dicts with edges keyed by sorted (a, b) pair.
    """
    grouped: dict[int, list[tuple[float, float, str]]] = {}
    for row in rows:
        grouped.setdefault(row["scene_id"], []).append(
            (row["start"], row["end"], row["speaker"])
        )

    wa, wp, wc = params["weight_adjacency"], params["weight_proximity"], params["weight_copresence"]
    window = params["proximity_window"]

    scenes: list[dict] = []
    for scene_id in sorted(grouped):
        turns = sorted(grouped[scene_id], key=lambda t: t[0])  # stable, start only
        if not turns:
            continue
        speakers_seq = [t[2] for t in turns]
        unique = sorted(set(speakers_seq))
        adj = _adjacency(speakers_seq)
        prox = _proximity(speakers_seq, window)

        pairs = set(itertools.combinations(unique, 2)) | set(adj) | set(prox)
        edges: dict[tuple[str, str], dict] = {}
        for pair in sorted(pairs):
            a, b = pair
            adj_v = adj.get(pair, 0.0)
            prox_v = prox.get(pair, 0.0)
            cop_v = 1.0 if (a in unique and b in unique) else 0.0
            w = wa * adj_v + wp * prox_v + wc * cop_v
            if w > 0:
                edges[pair] = {
                    "weight": _round4(w),
                    "adjacency": adj_v,
                    "proximity": _round4(prox_v),
                    "copresence": cop_v,
                }
        scenes.append({
            "scene_id": scene_id,
            "start": turns[0][0],
            "end": max(t[1] for t in turns),
            "nodes": unique,
            "edges": edges,
            "n_turns": len(turns),   # carried for the adjacency-bound invariant (Task 4)
        })
    return scenes
