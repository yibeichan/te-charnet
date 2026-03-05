"""Tests for charnet.community_align."""
from __future__ import annotations

from charnet.community_align import (
    parse_community_transcript,
    align_monotonic,
    assign_scene_ids_from_scene_desc,
    build_alignment_rows,
)
from charnet.models import Shot, Utterance


def test_parse_community_transcript(tmp_path):
    path = tmp_path / "community.txt"
    path.write_text(
        "[Scene: Test room]\n\nRoss: Hi there.\n\n(Rachel enters)\n\nRachel: Hello!\n",
        encoding="utf-8",
    )
    events, dialogues = parse_community_transcript(path)
    assert len(events) == 4
    assert len(dialogues) == 2
    assert dialogues[0]["speaker_ct"] == "Ross"


def test_alignment_rows_include_scene_and_shot_id():
    timed = [
        Utterance(speaker="ross", start=0.0, end=1.0, text="Hi there.", index=0),
        Utterance(speaker="rachel", start=2.0, end=3.0, text="Hello!", index=1),
    ]
    community_events = [
        {"kind": "scene", "scene_desc": "Test room"},
        {
            "kind": "dialogue",
            "dialogue_idx": 0,
            "speaker_ct": "Ross",
            "utterance_ct": "Hi there.",
            "norm": "hi there",
        },
        {
            "kind": "dialogue",
            "dialogue_idx": 1,
            "speaker_ct": "Rachel",
            "utterance_ct": "Hello!",
            "norm": "hello",
        },
    ]
    community_dialogues = [community_events[1], community_events[2]]
    mapping, sim_cache = align_monotonic(timed, community_dialogues)
    shots = [Shot(shot_id=1, start=0.0, end=1.5), Shot(shot_id=2, start=1.5, end=4.0)]

    rows, matched = build_alignment_rows(
        timed_utterances=timed,
        community_events=community_events,
        community_dialogues=community_dialogues,
        mapping=mapping,
        sim_cache=sim_cache,
        shots=shots,
        min_similarity=0.45,
    )

    rows, n_scene_ids = assign_scene_ids_from_scene_desc(rows)

    assert matched == 2
    assert n_scene_ids == 1
    assert rows[0]["scene_desc"] == "Test room"
    assert rows[0]["scene_id"] == "1"
    dialogue_rows = [r for r in rows if r["speaker"]]
    assert dialogue_rows[0]["shot_id"] == "1"
    assert dialogue_rows[1]["shot_id"] == "2"
    assert dialogue_rows[0]["scene_id"] == "1"
    assert dialogue_rows[1]["scene_id"] == "1"
