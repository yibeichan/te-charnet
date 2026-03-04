"""I/O utilities: loading, parsing, and field auto-detection."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Optional

from charnet.models import Utterance, Shot

logger = logging.getLogger(__name__)

# --- Field name variants ---
SPEAKER_FIELDS = ["speaker", "Speaker", "speaker_label", "speakerLabel", "SPEAKER"]
START_FIELDS = ["start", "start_time", "begin", "Start", "start_sec"]
END_FIELDS = ["end", "end_time", "End", "end_sec"]
TEXT_FIELDS = ["text", "transcript", "content", "Text", "words"]

# PySceneDetect CSV columns
SHOT_START_COL = "Start Time (seconds)"
SHOT_END_COL = "End Time (seconds)"
SHOT_NUM_COL = "Scene Number"


def _detect_field(record: dict, candidates: list[str]) -> Optional[str]:
    """Return the first candidate key present in record, or None."""
    for key in candidates:
        if key in record:
            return key
    return None


def load_transcript(path: Path, speaker_map: Optional[dict[str, str]] = None) -> list[Utterance]:
    """Load and normalize a transcript JSON file.

    Handles heterogeneous field names. Applies speaker_map if provided.
    Returns list of Utterance objects sorted by start time.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Handle both list-of-dicts and dict-with-list
    if isinstance(raw, dict):
        # Try common wrapper keys
        for key in ("utterances", "segments", "words", "transcript", "data"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
        else:
            raise ValueError(f"Transcript JSON is a dict but no list found under known keys: {list(raw.keys())}")

    if not raw:
        logger.warning("Transcript is empty: %s", path)
        return []

    # Auto-detect field names from first record
    first = raw[0]
    speaker_key = _detect_field(first, SPEAKER_FIELDS)
    start_key = _detect_field(first, START_FIELDS)
    end_key = _detect_field(first, END_FIELDS)
    text_key = _detect_field(first, TEXT_FIELDS)

    if speaker_key is None:
        logger.warning("No speaker field detected. Fields found: %s", list(first.keys()))
    if start_key is None:
        raise ValueError(f"No start-time field detected. Fields found: {list(first.keys())}")

    logger.info(
        "Transcript field mapping — speaker:%s  start:%s  end:%s  text:%s",
        speaker_key, start_key, end_key, text_key,
    )

    utterances: list[Utterance] = []
    n_missing_speaker = 0
    n_missing_end = 0

    for i, record in enumerate(raw):
        speaker = str(record.get(speaker_key, "")) if speaker_key else ""
        if not speaker:
            n_missing_speaker += 1

        start = float(record[start_key])

        end: Optional[float] = None
        if end_key and end_key in record and record[end_key] is not None:
            end = float(record[end_key])
        else:
            n_missing_end += 1
            # Will be estimated later
            end = start  # placeholder

        text = str(record.get(text_key, "")) if text_key else ""

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
    """Load PySceneDetect CSV and return list of Shot objects.

    PySceneDetect CSVs have a metadata row before the header. We skip rows
    until we find the header row containing 'Scene Number'.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "Scene Number" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not find 'Scene Number' header in {path}")

    reader = csv.DictReader(lines[header_idx:])
    shots: list[Shot] = []
    for row in reader:
        try:
            shot_id = int(row[SHOT_NUM_COL])
            start = float(row[SHOT_START_COL])
            end = float(row[SHOT_END_COL])
            shots.append(Shot(shot_id=shot_id, start=start, end=end))
        except (KeyError, ValueError) as e:
            logger.warning("Skipping malformed shot row: %s — %s", dict(row), e)

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
    """Load a word-level transcript JSON (friends_annotations format) and group into utterances.

    Format: {"words": [{"word": "Okay,", "start": 4.8, "end": 5.08, "speaker": "chandler", ...}]}

    Grouping: new utterance when speaker changes OR gap between consecutive words >= word_gap_threshold.
    Speaker normalization: .strip().title() unless overridden by speaker_map.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    words: list[dict] = raw.get("words", [])
    if not words:
        logger.warning("No words found in word-level transcript: %s", path)
        return []

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


def load_pyscene_tsv(path: Path) -> list[Shot]:
    """Load a friends_annotations TSV pyscene file.

    Format: onset\\tduration\\tonset_frame
    End = onset + duration. Shot IDs are sequential (1-based).
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        shots: list[Shot] = []
        for i, row in enumerate(reader, start=1):
            onset = float(row["onset"])
            duration = float(row["duration"])
            shots.append(Shot(shot_id=i, start=onset, end=onset + duration))

    logger.info("Loaded %d shots from TSV %s", len(shots), path)
    return sorted(shots, key=lambda s: s.start)


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


def save_scenes(scenes: list, path: Path) -> None:
    """Save scenes list to JSON."""
    from charnet.models import Scene
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [s.to_dict() if hasattr(s, "to_dict") else s for s in scenes]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %d scenes to %s", len(scenes), path)


def load_scenes(path: Path):
    """Load scenes JSON produced by segment stage."""
    from charnet.models import Scene
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scenes = []
    for d in data:
        scenes.append(Scene(
            scene_id=d["scene_id"],
            start=d["start"],
            end=d["end"],
            speakers=d.get("speakers", []),
            n_shots=d.get("n_shots", 0),
            n_utterances=d.get("n_utterances", 0),
            utterance_indices=d.get("utterance_indices", []),
        ))
    return scenes


def save_temporal_network(scene_graphs: list, path: Path) -> None:
    """Save temporal network JSON."""
    from charnet.models import SceneGraph
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [sg.to_dict() if hasattr(sg, "to_dict") else sg for sg in scene_graphs]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved temporal network (%d scenes) to %s", len(scene_graphs), path)


def load_temporal_network(path: Path) -> list:
    """Load temporal network JSON."""
    from charnet.models import SceneGraph, EdgeData
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
