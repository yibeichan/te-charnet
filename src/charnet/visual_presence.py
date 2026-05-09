"""Visual-presence helpers for sentence-level annotation QA.

Deepgaze data is saliency data, not character identity. These helpers expose it
as a generic visual-attention proxy so downstream stages can distinguish
"visual signal available" from "no visual corroboration" without pretending to
know which character is visible.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


DEFAULT_DEEPGAZE_FPS = 29.97
VISUAL_COLUMNS = [
    "visual_presence",
    "visual_presence_source",
    "visual_presence_ratio",
    "visual_presence_frames",
    "visual_presence_total_frames",
    "visual_presence_note",
    "annotation_confidence",
    "annotation_review_reason",
]


def deepgaze_path_for_episode(annotation_root: Path, episode: str) -> Path | None:
    """Return the Deepgaze maxpeak TSV path for *episode*, if present."""
    season_match = None
    for part in episode.split("_"):
        if part.startswith("s") and "e" in part:
            season_match = part.split("e", 1)[0]
            break
    if not season_match:
        return None

    season_dir = f"s{int(season_match[1:])}"
    path = annotation_root / "DeepgazeMr" / season_dir / f"{episode}_maxpeak_coord.tsv"
    return path if path.exists() else None


def load_deepgaze_maxpeak(path: Path) -> list[bool]:
    """Load a Deepgaze maxpeak TSV as per-frame visual-signal booleans."""
    frames: list[bool] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            h = _safe_float(row.get("h"))
            w = _safe_float(row.get("w"))
            p = _safe_float(row.get("p"))
            frames.append(
                h is not None
                and w is not None
                and p is not None
                and math.isfinite(h)
                and math.isfinite(w)
                and math.isfinite(p)
                and p > 0
            )
    return frames


def visual_presence_for_window(
    frames: list[bool] | None,
    start: Any,
    end: Any,
    *,
    fps: float = DEFAULT_DEEPGAZE_FPS,
) -> dict[str, str]:
    """Aggregate frame-level visual signal over a sentence time window."""
    if frames is None:
        return {
            "visual_presence": "unavailable",
            "visual_presence_source": "none",
            "visual_presence_ratio": "",
            "visual_presence_frames": "",
            "visual_presence_total_frames": "",
            "visual_presence_note": "visual_presence_data_unavailable",
        }

    start_f = _safe_float(start)
    end_f = _safe_float(end)
    if start_f is None:
        return {
            "visual_presence": "unknown",
            "visual_presence_source": "deepgaze_maxpeak",
            "visual_presence_ratio": "",
            "visual_presence_frames": "",
            "visual_presence_total_frames": "",
            "visual_presence_note": "missing_sentence_start",
        }
    if end_f is None or end_f <= start_f:
        end_f = start_f + (1.0 / fps)

    start_idx = max(0, int(math.floor(start_f * fps)))
    end_idx = min(len(frames), max(start_idx + 1, int(math.ceil(end_f * fps))))
    if start_idx >= len(frames) or end_idx <= start_idx:
        return {
            "visual_presence": "unknown",
            "visual_presence_source": "deepgaze_maxpeak",
            "visual_presence_ratio": "",
            "visual_presence_frames": "0",
            "visual_presence_total_frames": "0",
            "visual_presence_note": "sentence_outside_visual_data_range",
        }

    window = frames[start_idx:end_idx]
    present = sum(1 for value in window if value)
    total = len(window)
    ratio = present / total if total else 0.0
    if ratio >= 0.20:
        presence = "present"
    elif present > 0:
        presence = "partial"
    else:
        presence = "absent"

    return {
        "visual_presence": presence,
        "visual_presence_source": "deepgaze_maxpeak",
        "visual_presence_ratio": f"{ratio:.4f}",
        "visual_presence_frames": str(present),
        "visual_presence_total_frames": str(total),
        "visual_presence_note": "",
    }


def add_visual_presence_to_sentences(
    sentences: list[dict],
    frames: list[bool] | None,
    *,
    fps: float = DEFAULT_DEEPGAZE_FPS,
) -> dict[str, int]:
    """Append visual-presence fields to sentence dictionaries in-place."""
    counts = {"present": 0, "partial": 0, "absent": 0, "unknown": 0, "unavailable": 0}
    for sent in sentences:
        fields = visual_presence_for_window(frames, sent.get("start"), sent.get("end"), fps=fps)
        sent.update(fields)
        counts[fields["visual_presence"]] = counts.get(fields["visual_presence"], 0) + 1
    return counts


def assess_annotation_confidence(
    speaker: str,
    speaker_ct: str,
    speaker_confidence: str,
    visual_presence: str,
) -> tuple[str, str]:
    """Return composite annotation confidence and a visual-specific review reason."""
    speaker = str(speaker or "").strip()
    speaker_ct = str(speaker_ct or "").strip()
    speaker_confidence = str(speaker_confidence or "").strip()
    visual_presence = str(visual_presence or "").strip()

    base_rank = {"high": 3, "medium": 2, "low": 1, "unresolved": 0, "": 1}
    rank = base_rank.get(speaker_confidence, 1)
    reason = ""

    if speaker and speaker_ct and speaker != speaker_ct:
        rank = min(rank, 1)
        reason = "speaker_conflict"

    if speaker and visual_presence == "absent":
        rank = min(rank, 2)
        reason = "speaker_no_visual_presence"
    elif speaker and visual_presence in {"unknown", "unavailable"}:
        rank = min(rank, 2)
        reason = "visual_presence_unavailable" if visual_presence == "unavailable" else "visual_presence_unknown"
    elif speaker and visual_presence == "partial":
        rank = min(rank, 2)
        reason = "partial_visual_presence"

    if not speaker:
        rank = min(rank, 1)
        reason = reason or "missing_speaker"

    if rank >= 3:
        return "high", ""
    if rank == 2:
        return "medium", reason
    return "low", reason or "low_speaker_confidence"


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
