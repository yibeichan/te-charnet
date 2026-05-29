# src/charnet/topic_shift.py
"""Topic-shift sub-boundary proposer (improvement direction #2).

Pipeline per scene: group sentence rows into community-transcript "turns",
embed each turn (MiniLM, cached), score each inter-turn gap by the cosine
distance between mean-pooled blocks of W turns on either side, and propose
boundaries at local-maximum gaps whose TextTiling depth exceeds a threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

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
