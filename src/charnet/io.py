"""I/O utilities: loading and parsing."""
from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from charnet.models import EdgeData, Scene, SceneGraph, Shot, Utterance

logger = logging.getLogger(__name__)

# PySceneDetect CSV columns
SHOT_START_COL = "Start Time (seconds)"
SHOT_END_COL = "End Time (seconds)"
SHOT_NUM_COL = "Scene Number"


def load_transcript(path: Path, speaker_map: Optional[dict[str, str]] = None) -> list[Utterance]:
    """Load and normalize a transcript JSON file.

    For dict-shaped transcripts, read only the `words` list and map each
    word dict to one Utterance using fields: word/start/end/speaker.
    Applies speaker_map if provided.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Dict transcripts: only accept `words` and treat each word as one utterance.
    if isinstance(raw, dict):
        words = raw.get("words")
        if not isinstance(words, list):
            raise ValueError(
                f"Transcript JSON must contain a list under 'words'. Available keys: {list(raw.keys())}"
            )
        raw = words

    if not raw:
        logger.warning("Transcript is empty: %s", path)
        return []

    # friends_annotations word entries use fixed keys
    first = raw[0]
    missing = [k for k in ("word", "start", "end", "speaker") if k not in first]
    if missing:
        raise ValueError(
            f"Word record is missing required keys {missing}. Fields found: {list(first.keys())}"
        )

    utterances: list[Utterance] = []
    n_missing_speaker = 0
    n_missing_end = 0

    for i, record in enumerate(raw):
        speaker = str(record.get("speaker", ""))
        if not speaker:
            n_missing_speaker += 1

        start = float(record["start"])

        end: Optional[float] = None
        if record.get("end") is not None:
            end = float(record["end"])
        else:
            n_missing_end += 1
            # Will be estimated later
            end = start  # placeholder

        text = str(record.get("word", ""))

        # Apply speaker map
        if speaker_map and speaker in speaker_map:
            speaker = speaker_map[speaker]

        utterances.append(Utterance(speaker=speaker, start=start, end=end, text=text, index=i))

    if n_missing_speaker:
        logger.warning("%d utterances have no speaker label", n_missing_speaker)
    if n_missing_end:
        logger.warning("%d utterances have no end time (will be estimated)", n_missing_end)

    # Sort by start time
    utterances.sort(key=lambda u: u.start)
    return utterances


def estimate_missing_end_times(utterances: list[Utterance], min_duration: float = 0.1) -> list[Utterance]:
    """Fill in missing/zero end times using the next utterance's start time."""
    for i, utt in enumerate(utterances):
        if utt.end <= utt.start:
            if i + 1 < len(utterances):
                # Use next utterance's start, clamped to at least min_duration
                utt.end = max(utt.start + min_duration, utterances[i + 1].start)
            else:
                utt.end = utt.start + min_duration
    return utterances


def load_shots(path: Path) -> list[Shot]:
    """Load shot boundaries and return list of Shot objects.

    Preferred format (friends_annotations TSV):
      onset, duration, onset_frame
    Converted as:
      start = onset
      end = onset + duration
      shot_id = sequential 1-based row index

    Also supports legacy PySceneDetect CSVs with:
      Scene Number, Start Time (seconds), End Time (seconds)
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    if not lines:
        logger.warning("Shots file is empty: %s", path)
        return []

    # friends_annotations TSV: onset/duration/onset_frame
    delimiter = "\t" if "\t" in lines[0] else ","
    reader = csv.DictReader(lines, delimiter=delimiter)
    fieldnames = [f.strip() for f in (reader.fieldnames or [])]
    shots: list[Shot] = []

    if {"onset", "duration"}.issubset(fieldnames):
        for i, row in enumerate(reader, start=1):
            try:
                onset = float(row["onset"])
                duration = float(row["duration"])
                shots.append(Shot(shot_id=i, start=onset, end=onset + duration))
            except (KeyError, TypeError, ValueError) as e:
                logger.warning("Skipping malformed TSV shot row: %s — %s", dict(row), e)
        logger.info("Loaded %d shots from TSV %s", len(shots), path)
        return sorted(shots, key=lambda s: s.start)

    # Legacy PySceneDetect CSV fallback with metadata rows before header.
    header_idx = None
    for i, line in enumerate(lines):
        if "Scene Number" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            f"Unsupported shots format in {path}. Expected TSV columns "
            f"('onset', 'duration', 'onset_frame') or legacy PySceneDetect CSV columns "
            f"('{SHOT_NUM_COL}', '{SHOT_START_COL}', '{SHOT_END_COL}')."
        )

    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        try:
            shot_id = int(row[SHOT_NUM_COL])
            start = float(row[SHOT_START_COL])
            end = float(row[SHOT_END_COL])
            shots.append(Shot(shot_id=shot_id, start=start, end=end))
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Skipping malformed legacy shot row: %s — %s", dict(row), e)

    logger.info("Loaded %d shots from %s", len(shots), path)
    return sorted(shots, key=lambda s: s.start)


def _words_to_utterance(
    words: list[dict],
    index: int,
    speaker_map: Optional[dict[str, str]],
) -> Utterance:
    """Build a single Utterance from a list of word dicts (same speaker, continuous)."""
    speaker = words[0].get("speaker", "")
    speaker = speaker.strip().title() if speaker else ""
    if speaker_map and speaker in speaker_map:
        speaker = speaker_map[speaker]
    start = float(words[0]["start"])
    end = float(words[-1]["end"])
    text = " ".join(w.get("word", "") for w in words)
    return Utterance(speaker=speaker, start=start, end=end, text=text, index=index)


def load_word_transcript(
    path: Path,
    word_gap_threshold: float = 0.5,
    speaker_map: Optional[dict[str, str]] = None,
) -> list[Utterance]:
    """Load a word-level transcript JSON and group words into utterances.

    Accepted formats:
    - {"words": [{"word": "Okay,", "start": 4.8, "end": 5.08, "speaker": "chandler", ...}]}
    - [{"word": "Okay,", "start": 4.8, "end": 5.08, "speaker": "chandler", ...}]

    Grouping: new utterance when speaker changes OR gap between consecutive words >= word_gap_threshold.
    Speaker normalization: .strip().title() unless overridden by speaker_map.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if word_gap_threshold < 0:
        raise ValueError(f"word_gap_threshold must be >= 0, got {word_gap_threshold}")

    if isinstance(raw, dict):
        words = raw.get("words", [])
    elif isinstance(raw, list):
        words = raw
    else:
        raise ValueError(
            f"Unsupported transcript format in {path}. Expected dict with 'words' or a list of word dicts."
        )

    if not isinstance(words, list):
        raise ValueError(f"'words' in {path} must be a list.")

    if not words:
        logger.warning("No words found in word-level transcript: %s", path)
        return []

    first = words[0]
    missing = [k for k in ("word", "start", "end", "speaker") if k not in first]
    if missing:
        raise ValueError(
            f"Word record is missing required keys {missing}. Fields found: {list(first.keys())}"
        )

    utterances: list[Utterance] = []
    current_group: list[dict] = [words[0]]

    for word in words[1:]:
        prev = current_group[-1]
        gap = float(word["start"]) - float(prev["end"])
        speaker_changed = word.get("speaker", "") != prev.get("speaker", "")

        if speaker_changed or gap >= word_gap_threshold:
            utterances.append(_words_to_utterance(current_group, len(utterances), speaker_map))
            current_group = [word]
        else:
            current_group.append(word)

    utterances.append(_words_to_utterance(current_group, len(utterances), speaker_map))

    logger.info("Loaded %d words → %d utterances from %s", len(words), len(utterances), path)
    return utterances


def load_sentence_transcript(
    path: Path,
    speaker_map: Optional[dict[str, str]] = None,
) -> list[Utterance]:
    """Load sentence-level transcript entries into Utterance objects.

    Accepted formats:
    - {"sentences": [{"text": "...", "start": 1.2, "end": 2.3, "speaker": "ross"}]}
    - [{"text": "...", "start": 1.2, "end": 2.3, "speaker": "ross"}]
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        sentences = raw.get("sentences", [])
    elif isinstance(raw, list):
        sentences = raw
    else:
        raise ValueError(
            f"Unsupported sentence transcript format in {path}. "
            "Expected dict with 'sentences' or a list of sentence dicts."
        )

    if not isinstance(sentences, list):
        raise ValueError(f"'sentences' in {path} must be a list.")

    utterances: list[Utterance] = []
    n_missing_end = 0
    for i, rec in enumerate(sentences):
        if not isinstance(rec, dict):
            logger.warning("Skipping non-dict sentence record at index %d in %s", i, path)
            continue
        text = str(rec.get("text", "")).strip()
        if not text:
            continue
        speaker = str(rec.get("speaker", "")).strip()
        if speaker_map and speaker in speaker_map:
            speaker = speaker_map[speaker]

        start = float(rec.get("start", 0.0))
        end = rec.get("end")
        if end is None:
            n_missing_end += 1
            end_f = start
        else:
            end_f = float(end)

        utterances.append(Utterance(speaker=speaker, start=start, end=end_f, text=text, index=i))

    if n_missing_end:
        logger.warning("%d sentence utterances have no end time (will be estimated)", n_missing_end)

    utterances.sort(key=lambda u: u.start)
    logger.info("Loaded %d sentence utterances from %s", len(utterances), path)
    return utterances


def infer_community_transcript_path(transcript_path: Path, episode: str) -> Optional[Path]:
    """Infer community transcript path for a transcript episode key.

    Example:
      transcript: .../Speech2Text/s6/friends_s06e01a_model-AA_desc-wSpeaker_transcript.json
      episode:    friends_s06e01a
      -> .../community_based/s6/friends_s06e01_ufs.txt
    """
    match = re.match(r"^(friends_s\d{2}e\d{2})([a-z]+)?$", episode)
    if not match:
        return None
    episode_root = match.group(1)
    season_match = re.search(r"s0*(\d+)e", episode_root)
    if not season_match:
        return None
    season_dir = f"s{int(season_match.group(1))}"
    filename = f"{episode_root}_ufs.txt"

    candidates: list[Path] = []

    parts = list(transcript_path.parts)
    if "Speech2Text" in parts:
        i = parts.index("Speech2Text")
        base = Path(*parts[:i]) if i > 0 else Path(".")
        candidates.append(base / "community_based" / season_dir / filename)

    candidates.append(
        Path("data")
        / "friends_annotations"
        / "annotation_results"
        / "community_based"
        / season_dir
        / filename
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def save_records(records: list[dict[str, Any]], path: Path, label: str = "records") -> None:
    """Save list-of-dict records to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d %s to %s", len(records), label, path)


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load list-of-dict records from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, found {type(data).__name__}")
    return data


def load_speaker_map(path: Path) -> dict[str, str]:
    """Load optional speaker map JSON (SPEAKER_01 -> 'Monica')."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_utterances(utterances: list[Utterance], path: Path) -> None:
    """Save normalized utterances to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {"speaker": u.speaker, "start": u.start, "end": u.end, "text": u.text, "index": u.index}
        for u in utterances
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %d utterances to %s", len(utterances), path)


def load_utterances(path: Path) -> list[Utterance]:
    """Load normalized utterances JSON produced by preprocess stage."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Utterance(**d) for d in data]


def save_shots(shots: list[Shot], path: Path) -> None:
    """Save normalized shots to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"shot_id": s.shot_id, "start": s.start, "end": s.end} for s in shots]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %d shots to %s", len(shots), path)


def load_shots_json(path: Path) -> list[Shot]:
    """Load normalized shots JSON produced by preprocess stage."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Shot(**d) for d in data]


def save_temporal_network(scene_graphs: list[SceneGraph | dict[str, Any]], path: Path) -> None:
    """Save temporal network JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [sg.to_dict() if hasattr(sg, "to_dict") else sg for sg in scene_graphs]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved temporal network (%d scenes) to %s", len(scene_graphs), path)


def load_temporal_network(path: Path) -> list[SceneGraph]:
    """Load temporal network JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = []
    for d in data:
        edges = [EdgeData(**e) for e in d.get("edges", [])]
        result.append(SceneGraph(
            scene_id=d["scene_id"],
            start=d["start"],
            end=d["end"],
            nodes=d.get("nodes", []),
            edges=edges,
        ))
    return result


_ALIGNED_ROWS_FIELDNAMES = [
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


def save_alignment_rows_tsv(rows: list[dict[str, str]], path: Path) -> None:
    """Save aligned rows to TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_ALIGNED_ROWS_FIELDNAMES, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_alignment_rows_tsv(path: Path) -> list[dict[str, str]]:
    """Load a saved aligned_rows TSV back into a list of dicts."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


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


def load_corrected_speaker_rows(path: Path, speaker_col: str = "speaker_ct") -> list[dict]:
    """Load a speaker TSV and normalise column names for network building.

    Works with speaker-filled TSVs from 01b (enhanced or final-cleaned)
    as well as legacy aligned_rows TSVs.
    Required columns: scene_id, start, end, and *speaker_col*.
    """
    rows = load_alignment_rows_tsv(path)
    result = []
    for row in rows:
        if not row.get("start"):  # skip scene-marker rows
            continue
        normalised = dict(row)
        if speaker_col != "speaker":
            normalised["speaker"] = row.get(speaker_col, "")
        result.append(normalised)
    return result


def load_scenes(path: Path) -> list[Scene]:
    """Load scenes.json produced by Stage 1."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Scene(
            scene_id=d["scene_id"],
            start=d["start"],
            end=d["end"],
            speakers=d.get("speakers", []),
            n_shots=d.get("n_shots", 0),
            n_utterances=d.get("n_utterances", 0),
            utterance_indices=d.get("utterance_indices", []),
        )
        for d in data
    ]
