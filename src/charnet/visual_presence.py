"""Visual-presence helpers for sentence-level annotation QA.

Uses char-tracker stage-05 per-character per-second timestamps. Each episode
ships a CSV with columns ``second,chandler,joey,monica,phoebe,rachel,ross``
holding 0/1 per main-cast character per integer second. Aggregating to a
sentence's ``[start, end]`` window yields a character-identified visual signal
(not generic saliency), so downstream logic can distinguish "speaker on
screen" from "someone else on screen" from "no main-cast face".
"""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any

MAIN_CHARACTERS: tuple[str, ...] = (
    "chandler",
    "joey",
    "monica",
    "phoebe",
    "rachel",
    "ross",
)
DEFAULT_CHAR_TRACKER_DIR = Path(
    "/orcd/scratch/bcs/002/yibei/friends-char-track/output/05_character_timestamps"
)
CHAR_TRACKER_ENV_VAR = "CHAR_TRACKER_STAGE5_DIR"
PRESENCE_THRESHOLD = 0.5

# CharTrackerGrid = (chars_in_header_order, [[0/1 per char] for each second])
CharTrackerGrid = tuple[list[str], list[list[int]]]


VISUAL_COLUMNS = [
    "visual_presence",
    "visual_presence_source",
    "visual_presence_chars",
    "speaker_visual_presence",
    "speaker_visual_ratio",
    "visual_presence_note",
    "annotation_confidence",
    "annotation_review_reason",
]

VISUAL_COLUMN_DEFAULTS: dict[str, str] = {
    "visual_presence": "unavailable",
    "visual_presence_source": "none",
    "visual_presence_chars": "",
    "speaker_visual_presence": "unavailable",
    "speaker_visual_ratio": "",
    "visual_presence_note": "visual_presence_data_unavailable",
    "annotation_confidence": "",
    "annotation_review_reason": "",
}


def resolve_char_tracker_dir(override: Path | str | None = None) -> Path | None:
    """Resolve the stage-05 directory: explicit override → env var → default scratch path.

    Returns ``None`` if none of the candidates point at an existing directory.
    """
    candidates: list[Path] = []
    if override is not None:
        candidates.append(Path(override))
    env = os.environ.get(CHAR_TRACKER_ENV_VAR)
    if env:
        candidates.append(Path(env))
    candidates.append(DEFAULT_CHAR_TRACKER_DIR)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def char_tracker_csv_path(char_tracker_dir: Path, episode: str) -> Path | None:
    """Locate ``<episode>_timestamps.csv`` inside *char_tracker_dir*, if present."""
    path = char_tracker_dir / f"{episode}_timestamps.csv"
    return path if path.exists() else None


def load_char_tracker_grid(path: Path) -> CharTrackerGrid:
    """Load a stage-05 CSV into ``(chars, rows)``.

    ``chars`` lowercases the header characters; ``rows[sec]`` is the per-char
    0/1 list for integer second ``sec``. Non-integer or out-of-order seconds
    are tolerated; missing seconds are filled with zeros.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header or header[0].strip().lower() != "second":
            raise ValueError(f"{path}: expected first header column 'second', got {header[:1]}")
        chars = [h.strip().lower() for h in header[1:]]
        by_second: dict[int, list[int]] = {}
        for row in reader:
            if not row:
                continue
            try:
                sec = int(row[0])
            except ValueError:
                continue
            values = [_int_or_zero(v) for v in row[1:]]
            if len(values) < len(chars):
                values.extend([0] * (len(chars) - len(values)))
            by_second[sec] = values[: len(chars)]
    if not by_second:
        return chars, []
    last = max(by_second)
    rows = [by_second.get(i, [0] * len(chars)) for i in range(last + 1)]
    return chars, rows


def visual_presence_for_window(
    grid: CharTrackerGrid | None,
    start: Any,
    end: Any,
    *,
    threshold: float = PRESENCE_THRESHOLD,
) -> dict[str, Any]:
    """Aggregate per-second character presence over a sentence window.

    Returns a dict with the public TSV columns plus ``_char_ratios`` (a
    ``{char: fraction-of-seconds}`` map used by callers to compute
    ``speaker_visual_presence``; the leading underscore signals it's not a
    persisted column).
    """
    if grid is None:
        return {
            "visual_presence": "unavailable",
            "visual_presence_source": "none",
            "visual_presence_chars": "",
            "_char_ratios": {},
            "visual_presence_note": "visual_presence_data_unavailable",
        }

    chars, rows = grid
    start_f = _safe_float(start)
    end_f = _safe_float(end)
    if start_f is None:
        return {
            "visual_presence": "unknown",
            "visual_presence_source": "char_tracker_stage05",
            "visual_presence_chars": "",
            "_char_ratios": {},
            "visual_presence_note": "missing_sentence_start",
        }
    if end_f is None or end_f <= start_f:
        end_f = start_f + 1.0

    s_idx = max(0, int(math.floor(start_f)))
    e_idx = min(len(rows), max(s_idx + 1, int(math.ceil(end_f))))
    if s_idx >= len(rows) or e_idx <= s_idx:
        return {
            "visual_presence": "unknown",
            "visual_presence_source": "char_tracker_stage05",
            "visual_presence_chars": "",
            "_char_ratios": {},
            "visual_presence_note": "sentence_outside_visual_data_range",
        }

    window = rows[s_idx:e_idx]
    total = len(window)
    counts = [0] * len(chars)
    for sec_row in window:
        for i, v in enumerate(sec_row):
            counts[i] += v
    ratios = {chars[i]: counts[i] / total for i in range(len(chars))}
    chars_present = sorted(c for c, r in ratios.items() if r >= threshold)
    any_partial = any(r > 0 for r in ratios.values())
    if chars_present:
        presence = "present"
    elif any_partial:
        presence = "partial"
    else:
        presence = "absent"
    return {
        "visual_presence": presence,
        "visual_presence_source": "char_tracker_stage05",
        "visual_presence_chars": "|".join(chars_present),
        "_char_ratios": ratios,
        "visual_presence_note": "",
    }


def _speaker_visual_state(
    speaker: str,
    ratios: dict[str, float],
    source: str,
    *,
    threshold: float = PRESENCE_THRESHOLD,
) -> tuple[str, str]:
    """Return ``(state, ratio_str)`` for the named speaker against *ratios*.

    States:
      - ``unavailable`` — no visual source loaded
      - ``unknown`` — visual data loaded but speaker name empty
      - ``unobserved`` — speaker is not a main-cast char tracked by stage-05
      - ``present`` / ``partial`` / ``absent`` — derived from the per-second ratio
    """
    if source == "none" or not ratios:
        return "unavailable", ""
    if not speaker:
        return "unknown", ""
    key = speaker.strip().lower()
    if key not in ratios:
        return "unobserved", ""
    r = ratios[key]
    if r >= threshold:
        state = "present"
    elif r > 0:
        state = "partial"
    else:
        state = "absent"
    return state, f"{r:.4f}"


def add_visual_presence_to_sentences(
    sentences: list[dict],
    grid: CharTrackerGrid | None,
    *,
    threshold: float = PRESENCE_THRESHOLD,
) -> dict[str, int]:
    """Append visual-presence + speaker_visual_* fields to each sentence dict."""
    counts = {"present": 0, "partial": 0, "absent": 0, "unknown": 0, "unavailable": 0}
    for sent in sentences:
        fields = visual_presence_for_window(grid, sent.get("start"), sent.get("end"), threshold=threshold)
        ratios = fields.pop("_char_ratios", {})
        speaker_for_visual = sent.get("speaker_ct") or sent.get("speaker") or ""
        state, ratio_str = _speaker_visual_state(
            str(speaker_for_visual), ratios, fields["visual_presence_source"], threshold=threshold
        )
        sent.update(fields)
        sent["speaker_visual_presence"] = state
        sent["speaker_visual_ratio"] = ratio_str
        counts[fields["visual_presence"]] = counts.get(fields["visual_presence"], 0) + 1
    return counts


def assess_annotation_confidence(
    speaker: str,
    speaker_ct: str,
    speaker_confidence: str,
    visual_presence: str,
    visual_presence_chars: str = "",
    speaker_visual_presence: str = "",
) -> tuple[str, str]:
    """Return ``(composite_confidence, review_reason)``.

    The visual signal is character-identified (char-tracker stage-05), so this
    function distinguishes three failure modes:

    - ``speaker_offscreen`` — someone is on screen, but not the alleged speaker.
      Strongest negative; downgrades to ``low``.
    - ``speaker_no_face`` — no main-cast face anywhere in the window.
      Could be off-camera dialogue, side profile, or a guest scene; downgrade
      to ``medium``.
    - ``visual_unobserved_speaker`` — speaker is a guest character that
      char-tracker doesn't track; neutral.
    """
    speaker = str(speaker or "").strip()
    speaker_ct = str(speaker_ct or "").strip()
    speaker_confidence = str(speaker_confidence or "").strip()
    visual_presence = str(visual_presence or "").strip()
    speaker_visual_presence = str(speaker_visual_presence or "").strip()

    base_rank = {"high": 3, "medium": 2, "low": 1, "unresolved": 0, "": 1}
    rank = base_rank.get(speaker_confidence, 1)
    reason = ""

    if speaker and speaker_ct and speaker != speaker_ct:
        rank = min(rank, 1)
        reason = "speaker_conflict"

    if speaker:
        if speaker_visual_presence == "present":
            pass  # visual confirms speaker — keep rank
        elif speaker_visual_presence == "absent" and visual_presence == "present":
            # someone IS on screen, just not them — strong negative signal
            rank = min(rank, 1)
            reason = "speaker_offscreen"
        elif speaker_visual_presence == "absent":
            rank = min(rank, 2)
            reason = reason or "speaker_no_face"
        elif speaker_visual_presence == "partial":
            rank = min(rank, 2)
            reason = reason or "partial_visual_presence"
        elif speaker_visual_presence == "unobserved":
            pass  # guest character outside char-tracker scope — neutral
        elif speaker_visual_presence in {"", "unknown"} and visual_presence == "unknown":
            rank = min(rank, 2)
            reason = reason or "visual_presence_unknown"
        elif speaker_visual_presence in {"", "unavailable"} and visual_presence == "unavailable":
            rank = min(rank, 2)
            reason = reason or "visual_presence_unavailable"

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


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
