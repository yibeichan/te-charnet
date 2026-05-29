# src/charnet/topic_shift.py
"""Topic-shift sub-boundary proposer (improvement direction #2).

Pipeline per scene: group sentence rows into community-transcript "turns",
embed each turn (MiniLM, cached), score each inter-turn gap by the cosine
distance between mean-pooled blocks of W turns on either side, and propose
boundaries at local-maximum gaps whose TextTiling depth exceeds a threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Turn:
    text: str
    start: float
    end: float


def _clean_str(val) -> str:
    """NaN-safe strip: returns '' for None/NaN, else str(val).strip().

    Per the repo gotcha, never let NaN become the string 'nan'.
    """
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    return str(val).strip()


def build_text(utterance_ct, utterance) -> str:
    """Best-available turn text: community transcript, else Speech2Text.

    NaN-safe per the repo gotcha — never stringify NaN to "nan".
    """
    for val in (utterance_ct, utterance):
        s = _clean_str(val)
        if s:
            return s
    return ""


def group_turns_for_scene(scene_rows: pd.DataFrame) -> list[Turn]:
    """Collapse consecutive rows sharing the same ``utterance_ct`` into turns.

    Rows are assumed already ordered by time within one scene. A turn's text is
    its (shared) ct text, or the first row's ``utterance`` fallback when ct is
    blank; blank-ct rows never merge with neighbours.
    """
    turns: list[Turn] = []
    prev_ct_key: str | None = None
    for _, row in scene_rows.iterrows():
        ct_raw = row.get("utterance_ct")
        ct = _clean_str(ct_raw)
        text = build_text(ct_raw, row.get("utterance"))
        start, end = float(row["start"]), float(row["end"])
        mergeable = ct != "" and ct == prev_ct_key
        if mergeable and turns:
            last = turns[-1]
            turns[-1] = Turn(text=last.text, start=last.start, end=max(last.end, end))  # text shared by all rows with same ct key
        else:
            turns.append(Turn(text=text, start=start, end=end))
        prev_ct_key = ct if ct != "" else None
    return turns


def _mean_block(vecs: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Mean of rows [lo, hi); caller guarantees lo < hi within bounds."""
    return vecs[lo:hi].mean(axis=0)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def block_distance_trace(vecs: np.ndarray, w: int) -> np.ndarray:
    """Cosine distance between W-turn blocks on either side of each gap.

    Returns an array of length ``len(vecs) - 1``; entry ``i`` scores the gap
    between turn ``i`` and turn ``i+1``. Blocks are truncated to available
    turns at the sequence edges (no padding).
    """
    n = len(vecs)
    trace = np.zeros(max(0, n - 1))
    for i in range(n - 1):
        left = _mean_block(vecs, max(0, i - w + 1), i + 1)
        right = _mean_block(vecs, i + 1, min(n, i + 1 + w))
        trace[i] = _cosine_distance(left, right)
    return trace


def peak_depths(trace: np.ndarray) -> list[tuple[int, float]]:
    """Local maxima of *trace* with their TextTiling depth, deepest first.

    Depth of a peak = (peak − nearest lower-or-equal valley on the left)
    + (peak − nearest lower-or-equal valley on the right). Endpoints use the
    available side only.
    """
    n = len(trace)
    out: list[tuple[int, float]] = []
    for i in range(n):
        left_ok = i == 0 or trace[i] >= trace[i - 1]
        right_ok = i == n - 1 or trace[i] >= trace[i + 1]
        if not (left_ok and right_ok):
            continue
        # walk left to the local valley
        lv = trace[i]
        j = i
        while j > 0 and trace[j - 1] <= trace[j]:
            j -= 1
            lv = min(lv, trace[j])
        rv = trace[i]
        k = i
        while k < n - 1 and trace[k + 1] <= trace[k]:
            k += 1
            rv = min(rv, trace[k])
        depth = (trace[i] - lv) + (trace[i] - rv)
        if depth > 0:
            out.append((i, depth))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def propose_topic_boundaries(
    turns: list[Turn],
    vecs: np.ndarray,
    *,
    w: int,
    tau_depth: float,
    min_spacing: float,
) -> list[float]:
    """Interior boundary times for one scene's turn sequence.

    A boundary is a local-maximum gap with depth ≥ *tau_depth*; accepted
    greedily by descending depth subject to *min_spacing* from previously
    accepted boundaries. Each boundary is placed at the end time of the turn
    before the gap (a sentence end).
    """
    if len(turns) < 2 * w + 1 or len(vecs) != len(turns):
        return []
    trace = block_distance_trace(vecs, w)
    accepted_idx: list[int] = []
    for gap_i, depth in peak_depths(trace):
        if depth < tau_depth:
            continue
        t = turns[gap_i].end
        if any(abs(t - turns[j].end) < min_spacing for j in accepted_idx):
            continue
        accepted_idx.append(gap_i)
    return sorted(turns[i].end for i in accepted_idx)
