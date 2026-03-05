"""Shared test fixtures for charnet tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from charnet.models import Scene, Shot, Utterance


# --- Synthetic utterances ---
# 5 utterances, two speakers (Monica and Ross), spanning ~30 seconds

SAMPLE_UTTERANCES = [
    Utterance(speaker="Monica", start=0.0, end=3.5, text="Hey Ross!", index=0),
    Utterance(speaker="Ross", start=4.0, end=7.0, text="Hey Monica, what's up?", index=1),
    Utterance(speaker="Monica", start=7.5, end=10.0, text="Nothing much.", index=2),
    Utterance(speaker="Ross", start=10.5, end=13.0, text="Want to grab coffee?", index=3),
    Utterance(speaker="Monica", start=13.5, end=16.0, text="Sure!", index=4),
]

# 3 shots covering the same span
SAMPLE_SHOTS = [
    Shot(shot_id=1, start=0.0, end=8.0),
    Shot(shot_id=2, start=8.0, end=14.0),
    Shot(shot_id=3, start=14.0, end=18.0),
]


@pytest.fixture
def sample_utterances() -> list[Utterance]:
    return [Utterance(**vars(u)) for u in SAMPLE_UTTERANCES]


@pytest.fixture
def sample_shots() -> list[Shot]:
    return [Shot(**vars(s)) for s in SAMPLE_SHOTS]


@pytest.fixture
def sample_scene(sample_utterances) -> Scene:
    return Scene(
        scene_id=0,
        start=0.0,
        end=16.0,
        speakers=["Monica", "Ross"],
        n_shots=3,
        n_utterances=5,
        utterance_indices=[0, 1, 2, 3, 4],
    )


@pytest.fixture
def transcript_json_file(tmp_path, sample_utterances) -> Path:
    """A temporary word-level transcript JSON file."""
    data = [
        {"speaker": u.speaker, "start": u.start, "end": u.end, "word": u.text}
        for u in sample_utterances
    ]
    path = tmp_path / "transcript.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def transcript_json_variant_fields(tmp_path, sample_utterances) -> Path:
    """Transcript JSON with unsupported field names."""
    data = [
        {"speaker_label": u.speaker, "start_time": u.start, "end_time": u.end, "content": u.text}
        for u in sample_utterances
    ]
    path = tmp_path / "transcript_variant.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def shots_csv_file(tmp_path, sample_shots) -> Path:
    """A temporary PySceneDetect-style CSV file."""
    lines = [
        "Timecode List:,00:00:00.000",  # metadata row (to be skipped)
        "Scene Number,Start Frame,Start Timecode,Start Time (seconds),End Frame,End Timecode,End Time (seconds),Length (frames),Length (timecode),Length (seconds)",
    ]
    for s in sample_shots:
        lines.append(
            f"{s.shot_id},0,00:00:00.000,{s.start},100,00:00:08.000,{s.end},100,00:00:08.000,{s.end - s.start}"
        )
    path = tmp_path / "shots.csv"
    path.write_text("\n".join(lines))
    return path


@pytest.fixture
def speaker_map_file(tmp_path) -> Path:
    data = {"SPEAKER_01": "Monica", "SPEAKER_02": "Ross"}
    path = tmp_path / "speaker_map.json"
    path.write_text(json.dumps(data))
    return path
