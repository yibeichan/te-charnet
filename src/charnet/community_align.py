"""Alignment utilities: map timed transcript to community transcript + scene markers."""
from __future__ import annotations

import csv
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from charnet.models import Shot, Utterance

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - optional fallback
    fuzz = None


def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def normalize_for_match(text: str) -> str:
    text = _normalize_unicode(text).lower().strip()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_speaker(name: str) -> str:
    if not name:
        return ""
    text = _normalize_unicode(name).lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if fuzz is not None:
        ratio = fuzz.ratio(a, b) / 100.0
        token_ratio = fuzz.token_set_ratio(a, b) / 100.0
        base = 0.7 * ratio + 0.3 * token_ratio
    else:
        base = SequenceMatcher(None, a, b).ratio()
    ta = a.split()
    tb = b.split()
    if not ta or not tb:
        return base
    overlap = len(set(ta) & set(tb)) / max(1, min(len(set(ta)), len(set(tb))))
    return 0.65 * base + 0.35 * overlap


def parse_community_transcript(path: Path) -> tuple[list[dict], list[dict]]:
    """Parse community transcript text into event stream and dialogue subset."""
    events: list[dict] = []
    dialogues: list[dict] = []
    dialogue_idx = 0

    for line_idx, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = raw.strip()
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            desc = line[1:-1].strip()
            desc = re.sub(r"^\s*scene\s*:\s*", "", desc, flags=re.IGNORECASE)
            events.append({"idx": line_idx, "kind": "scene", "scene_desc": desc})
            continue

        if line.startswith("(") and line.endswith(")"):
            events.append({"idx": line_idx, "kind": "scene", "scene_desc": line[1:-1].strip()})
            continue

        if line.lower() in {"opening credits", "closing credits"}:
            events.append({"idx": line_idx, "kind": "scene", "scene_desc": line})
            continue

        match = re.match(r"^([^:]{1,80}):\s*(.+)$", line)
        if match:
            speaker = match.group(1).strip()
            utterance = match.group(2).strip()
            rec = {
                "idx": line_idx,
                "kind": "dialogue",
                "dialogue_idx": dialogue_idx,
                "speaker_ct": speaker,
                "utterance_ct": utterance,
                "norm": normalize_for_match(utterance),
            }
            events.append(rec)
            dialogues.append(rec)
            dialogue_idx += 1
            continue

        events.append({"idx": line_idx, "kind": "scene", "scene_desc": line})

    return events, dialogues


def align_monotonic(
    timed_utterances: list[Utterance],
    community_dialogues: list[dict],
    skip_timed_penalty: float = 0.6,
    skip_comm_penalty: float = 0.2,
) -> tuple[dict[int, int], dict[tuple[int, int], float]]:
    """Return timed index -> community dialogue index mapping and similarity cache."""
    n = len(timed_utterances)
    m = len(community_dialogues)
    neg_inf = -10**12

    dp = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]  # 1=diag, 2=up, 3=left

    dp[0][0] = 0.0
    for j in range(1, m + 1):
        dp[0][j] = 0.0
        bt[0][j] = 3
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - skip_timed_penalty
        bt[i][0] = 2

    sim_cache: dict[tuple[int, int], float] = {}
    for i in range(1, n + 1):
        timed = timed_utterances[i - 1]
        timed_norm = normalize_for_match(timed.text)
        timed_spk = normalize_speaker(timed.speaker)
        for j in range(1, m + 1):
            comm = community_dialogues[j - 1]
            comm_idx = int(comm["dialogue_idx"])
            comm_spk = normalize_speaker(str(comm.get("speaker_ct", "")))

            sim = text_similarity(timed_norm, str(comm.get("norm", "")))
            sim_cache[(timed.index, comm_idx)] = sim

            diag_score = (sim - 0.45) * 2.2
            if timed_spk and comm_spk:
                if timed_spk == comm_spk:
                    diag_score += 0.35
                else:
                    diag_score -= 0.05

            diag = dp[i - 1][j - 1] + diag_score
            up = dp[i - 1][j] - skip_timed_penalty
            left = dp[i][j - 1] - skip_comm_penalty

            if diag >= up and diag >= left:
                dp[i][j] = diag
                bt[i][j] = 1
            elif up >= left:
                dp[i][j] = up
                bt[i][j] = 2
            else:
                dp[i][j] = left
                bt[i][j] = 3

    end_j = max(range(m + 1), key=lambda j: dp[n][j])
    i, j = n, end_j
    mapping: dict[int, int] = {}
    while i > 0 and j > 0:
        move = bt[i][j]
        if move == 1:
            timed = timed_utterances[i - 1]
            comm = community_dialogues[j - 1]
            mapping[timed.index] = int(comm["dialogue_idx"])
            i -= 1
            j -= 1
        elif move == 2:
            i -= 1
        else:
            j -= 1

    return mapping, sim_cache


def scene_by_dialogue(events: list[dict]) -> dict[int, str]:
    current = ""
    mapping: dict[int, str] = {}
    for ev in events:
        if ev.get("kind") == "scene":
            current = str(ev.get("scene_desc", ""))
        elif ev.get("kind") == "dialogue":
            mapping[int(ev["dialogue_idx"])] = current
    return mapping


def find_shot_id_at_time(shots: list[Shot], timestamp: Optional[float]) -> Optional[int]:
    if timestamp is None or not shots:
        return None
    for shot in shots:
        if shot.start <= timestamp < shot.end:
            return shot.shot_id
    last = shots[-1]
    if timestamp == last.end:
        return last.shot_id
    return None


_HIGH_THRESHOLD = 0.85
_REVIEW_THRESHOLD = 0.60


def assign_confidence(sim: float, matched: bool) -> tuple[str, str]:
    """Return (confidence_level, method) for an alignment result.

    confidence_level: "high" | "medium" | "low" | "unresolved"
    method: "exact_align" | "fuzzy_align" | "none"
    """
    if not matched:
        return "unresolved", "none"
    if sim >= _HIGH_THRESHOLD:
        return "high", "exact_align"
    if sim >= _REVIEW_THRESHOLD:
        return "medium", "fuzzy_align"
    return "low", "fuzzy_align"


def build_alignment_rows(
    timed_utterances: list[Utterance],
    community_events: list[dict],
    community_dialogues: list[dict],
    mapping: dict[int, int],
    sim_cache: dict[tuple[int, int], float],
    shots: Optional[list[Shot]] = None,
    min_similarity: float = 0.52,
) -> tuple[list[dict[str, str]], int]:
    """Build final rows with optional scene-only rows and shot_id."""
    by_dialogue = {int(d["dialogue_idx"]): d for d in community_dialogues}
    scene_map = scene_by_dialogue(community_events)

    rows: list[dict[str, str]] = []
    matched = 0
    last_scene = None

    for utt in sorted(timed_utterances, key=lambda u: u.start):
        shot_id = find_shot_id_at_time(shots or [], utt.start)
        shot_id_str = str(shot_id) if shot_id is not None else ""

        comm_idx = mapping.get(utt.index)
        comm = by_dialogue.get(comm_idx) if comm_idx is not None else None

        keep_match = False
        sim = 0.0
        if comm is not None:
            sim = sim_cache.get((utt.index, comm_idx), 0.0)
            spk_match = normalize_speaker(utt.speaker) == normalize_speaker(str(comm["speaker_ct"]))
            keep_match = sim >= min_similarity or (sim >= (min_similarity - 0.08) and spk_match)

        confidence, method = assign_confidence(sim, keep_match)

        if keep_match and comm is not None:
            scene = scene_map.get(int(comm["dialogue_idx"]), "")
            if scene and scene != last_scene:
                rows.append(
                    {
                        "start": "",
                        "end": "",
                        "shot_id": "",
                        "speaker": "",
                        "utterance": "",
                        "speaker_ct": "",
                        "utterance_ct": "",
                        "scene_desc": scene,
                        "alignment_score": "",
                        "speaker_confidence": "",
                        "speaker_method": "",
                    }
                )
                last_scene = scene

            rows.append(
                {
                    "start": f"{utt.start:.2f}",
                    "end": f"{utt.end:.2f}",
                    "shot_id": shot_id_str,
                    "speaker": utt.speaker,
                    "utterance": utt.text,
                    "speaker_ct": str(comm["speaker_ct"]),
                    "utterance_ct": str(comm["utterance_ct"]),
                    "scene_desc": "",
                    "alignment_score": f"{sim:.4f}",
                    "speaker_confidence": confidence,
                    "speaker_method": method,
                }
            )
            matched += 1
        else:
            rows.append(
                {
                    "start": f"{utt.start:.2f}",
                    "end": f"{utt.end:.2f}",
                    "shot_id": shot_id_str,
                    "speaker": utt.speaker,
                    "utterance": utt.text,
                    "speaker_ct": "",
                    "utterance_ct": "",
                    "scene_desc": "",
                    "alignment_score": f"{sim:.4f}" if sim else "",
                    "speaker_confidence": confidence,
                    "speaker_method": method,
                }
            )

    return rows, matched


def assign_scene_ids_from_scene_desc(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Assign sequential scene_id values from scene_desc boundary rows.

    Scene rows (scene_desc != "") start a new scene id; subsequent dialogue rows inherit it.
    If dialogue rows appear before the first scene row, they are backfilled to the first scene id.
    """
    scene_id = 0
    first_scene_id: Optional[int] = None
    first_scene_row_idx: Optional[int] = None

    for idx, row in enumerate(rows):
        if row.get("scene_desc", ""):
            scene_id += 1
            row["scene_id"] = str(scene_id)
            if first_scene_id is None:
                first_scene_id = scene_id
                first_scene_row_idx = idx
        else:
            row["scene_id"] = str(scene_id) if scene_id > 0 else ""

    if first_scene_id is not None and first_scene_row_idx is not None:
        for idx in range(first_scene_row_idx):
            if not rows[idx].get("scene_id"):
                rows[idx]["scene_id"] = str(first_scene_id)

    return rows, scene_id


def save_alignment_rows_tsv(rows: list[dict[str, str]], path: Path) -> None:
    """Save aligned rows to TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "start",
        "end",
        "shot_id",
        "scene_id",
        "speaker",
        "utterance",
        "speaker_ct",
        "utterance_ct",
        "scene_desc",
        "alignment_score",
        "speaker_confidence",
        "speaker_method",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_review_queue(
    rows: list[dict[str, str]],
    path: Path,
    context_lines: int = 2,
) -> int:
    """Export rows needing manual review to a TSV file.

    Rows with speaker_confidence in {"low", "unresolved"} are flagged.
    Each flagged row is written with up to *context_lines* dialogue rows
    before and after it (scene-only rows are skipped from context).

    Returns the number of flagged rows written.
    """
    review_confidences = {"low", "unresolved"}
    dialogue_indices = [i for i, r in enumerate(rows) if not r.get("scene_desc")]
    flagged_dialogue_pos = {
        pos
        for pos, i in enumerate(dialogue_indices)
        if rows[i].get("speaker_confidence") in review_confidences
    }

    included: set[int] = set()
    for pos in flagged_dialogue_pos:
        for offset in range(-context_lines, context_lines + 1):
            neighbour = pos + offset
            if 0 <= neighbour < len(dialogue_indices):
                included.add(dialogue_indices[neighbour])

    review_rows = [rows[i] for i in sorted(included)]

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "start",
        "end",
        "speaker",
        "utterance",
        "speaker_ct",
        "utterance_ct",
        "alignment_score",
        "speaker_confidence",
        "speaker_method",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(review_rows)

    return len(flagged_dialogue_pos)


def find_episode_window(
    timed_utterances: list[Utterance],
    community_dialogues: list[dict],
    window_size: Optional[int] = None,
) -> tuple[int, int]:
    """Find the region in community_dialogues that best matches timed_utterances.

    Useful when timed_utterances cover only part of the full episode (e.g. a
    half-episode local transcript).  Returns (start_idx, end_idx) — a slice
    of community_dialogues to pass to align_monotonic().

    Args:
        timed_utterances: Local transcript utterances.
        community_dialogues: Full-episode community transcript dialogues.
        window_size: Number of community dialogues in the sliding window.
            Defaults to len(timed_utterances) + 20% padding.

    Returns:
        (start_idx, end_idx) into community_dialogues (end_idx exclusive).
    """
    n_timed = len(timed_utterances)
    n_comm = len(community_dialogues)

    if n_comm == 0:
        return 0, 0
    if n_timed == 0:
        return 0, n_comm

    if window_size is None:
        window_size = min(n_comm, int(n_timed * 1.2) + 5)
    elif window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")

    window_size = min(window_size, n_comm)

    if window_size >= n_comm:
        return 0, n_comm

    timed_norms = [normalize_for_match(u.text) for u in timed_utterances]
    comm_norms = [str(d.get("norm", "")) for d in community_dialogues]

    # Sparse sample: compare every ~4th timed utterance for speed
    sample_step = max(1, n_timed // 25)
    sampled = list(range(0, n_timed, sample_step))

    best_score = -1.0
    best_start = 0

    for start in range(n_comm - window_size + 1):
        window_norms = comm_norms[start : start + window_size]
        score = 0.0
        for ti in sampled:
            # Best similarity of this timed utterance against any window item
            best_local = max(
                text_similarity(timed_norms[ti], cn) for cn in window_norms
            )
            score += best_local
        avg = score / len(sampled)
        if avg > best_score:
            best_score = avg
            best_start = start

    return best_start, best_start + window_size
