from __future__ import annotations

import pandas as pd

from charnet.speaker_fill import PipelineConfig, infer_speaker, process_tsv
from charnet.visual_presence import (
    add_visual_presence_to_sentences,
    assess_annotation_confidence,
    load_char_tracker_grid,
    resolve_char_tracker_dir,
    visual_presence_for_window,
)


def _cfg() -> PipelineConfig:
    return PipelineConfig(
        song_markers=["singing"],
        continuation_prefixes=["and", "but"],
        high_bridge_score=0.95,
        ct_match_score=1.0,
        direction_score=0.9,
        name_address_score=0.75,
        short_turn_score=0.65,
        scene_context_score=0.6,
        long_ambiguous_score=0.5,
        weak_ambiguous_score=0.35,
        scene_majority_score=0.25,
    )


def _grid(chars=("rachel", "ross"), per_second=None):
    rows = list(per_second or [])
    return list(chars), rows


# ---------------------------------------------------------------------------
# visual_presence_for_window
# ---------------------------------------------------------------------------


def test_visual_presence_for_window_reports_unavailable_without_grid():
    fields = visual_presence_for_window(None, 0.0, 1.0)

    assert fields["visual_presence"] == "unavailable"
    assert fields["visual_presence_source"] == "none"
    assert fields["visual_presence_chars"] == ""
    assert fields["visual_presence_note"] == "visual_presence_data_unavailable"


def test_visual_presence_for_window_aggregates_per_second_grid():
    # rachel on for seconds 0–1, ross off; window 0..2 (s_idx=0, e_idx=2).
    grid = _grid(("rachel", "ross"), [[1, 0], [1, 0]])

    fields = visual_presence_for_window(grid, 0.0, 1.5)

    assert fields["visual_presence"] == "present"
    assert fields["visual_presence_chars"] == "rachel"
    assert fields["visual_presence_source"] == "char_tracker_stage05"


def test_visual_presence_for_window_partial_when_only_below_threshold():
    # 1 second on of 4 → ratio 0.25, below 0.5 threshold → "partial"
    grid = _grid(("rachel", "ross"), [[1, 0], [0, 0], [0, 0], [0, 0]])

    fields = visual_presence_for_window(grid, 0.0, 4.0)

    assert fields["visual_presence"] == "partial"
    assert fields["visual_presence_chars"] == ""


def test_visual_presence_for_window_absent_when_grid_empty_in_window():
    grid = _grid(("rachel", "ross"), [[0, 0], [0, 0], [0, 0]])

    fields = visual_presence_for_window(grid, 0.0, 2.0)

    assert fields["visual_presence"] == "absent"
    assert fields["visual_presence_chars"] == ""


def test_visual_presence_for_window_unknown_when_outside_data():
    grid = _grid(("rachel", "ross"), [[1, 1]])

    fields = visual_presence_for_window(grid, 100.0, 101.0)

    assert fields["visual_presence"] == "unknown"
    assert fields["visual_presence_note"] == "sentence_outside_visual_data_range"


# ---------------------------------------------------------------------------
# add_visual_presence_to_sentences (per-sentence aggregator)
# ---------------------------------------------------------------------------


def test_add_visual_presence_marks_speaker_present_when_in_window():
    grid = _grid(("rachel", "ross"), [[1, 0], [1, 0], [0, 1], [0, 1]])
    sentences = [
        {"start": 0.0, "end": 2.0, "speaker_ct": "Rachel", "speaker": ""},
        {"start": 2.0, "end": 4.0, "speaker_ct": "Rachel", "speaker": ""},
        {"start": 2.0, "end": 4.0, "speaker_ct": "Ross", "speaker": ""},
    ]

    counts = add_visual_presence_to_sentences(sentences, grid)

    assert counts == {"present": 3, "partial": 0, "absent": 0, "unknown": 0, "unavailable": 0}
    assert sentences[0]["speaker_visual_presence"] == "present"
    assert sentences[1]["speaker_visual_presence"] == "absent"  # rachel off seconds 2..4
    assert sentences[2]["speaker_visual_presence"] == "present"


def test_add_visual_presence_marks_guest_speaker_unobserved():
    grid = _grid(("rachel", "ross"), [[1, 0]])
    sentences = [{"start": 0.0, "end": 1.0, "speaker_ct": "Janice", "speaker": ""}]

    add_visual_presence_to_sentences(sentences, grid)

    # Janice isn't in main cast → char-tracker has no signal for her
    assert sentences[0]["speaker_visual_presence"] == "unobserved"
    assert sentences[0]["visual_presence_chars"] == "rachel"


# ---------------------------------------------------------------------------
# assess_annotation_confidence
# ---------------------------------------------------------------------------


def test_annotation_confidence_high_when_speaker_visually_confirmed():
    confidence, reason = assess_annotation_confidence(
        speaker="Rachel",
        speaker_ct="Rachel",
        speaker_confidence="high",
        visual_presence="present",
        visual_presence_chars="rachel",
        speaker_visual_presence="present",
    )

    assert confidence == "high"
    assert reason == ""


def test_annotation_confidence_flags_speaker_offscreen_when_others_visible():
    # Someone IS on screen (Ross) but the alleged speaker (Rachel) is not.
    confidence, reason = assess_annotation_confidence(
        speaker="Rachel",
        speaker_ct="Rachel",
        speaker_confidence="high",
        visual_presence="present",
        visual_presence_chars="ross",
        speaker_visual_presence="absent",
    )

    assert confidence == "low"
    assert reason == "speaker_offscreen"


def test_annotation_confidence_flags_speaker_no_face_when_window_empty():
    # No main-cast face anywhere in the window — could be off-camera dialogue.
    confidence, reason = assess_annotation_confidence(
        speaker="Rachel",
        speaker_ct="Rachel",
        speaker_confidence="high",
        visual_presence="absent",
        visual_presence_chars="",
        speaker_visual_presence="absent",
    )

    assert confidence == "medium"
    assert reason == "speaker_no_face"


def test_annotation_confidence_neutral_for_guest_speaker():
    # Char-tracker only tracks 6 main cast — guest speakers don't trigger downgrade.
    confidence, reason = assess_annotation_confidence(
        speaker="Janice",
        speaker_ct="Janice",
        speaker_confidence="high",
        visual_presence="present",
        visual_presence_chars="chandler",
        speaker_visual_presence="unobserved",
    )

    assert confidence == "high"
    assert reason == ""


# ---------------------------------------------------------------------------
# Stage-05 CSV loader
# ---------------------------------------------------------------------------


def test_load_char_tracker_grid_parses_csv(tmp_path):
    path = tmp_path / "friends_s01e01a_timestamps.csv"
    path.write_text("second,chandler,joey,monica,phoebe,rachel,ross\n0,0,0,0,0,0,0\n1,1,0,0,0,0,0\n2,1,1,0,0,0,0\n")

    chars, rows = load_char_tracker_grid(path)

    assert chars == ["chandler", "joey", "monica", "phoebe", "rachel", "ross"]
    assert rows[0] == [0, 0, 0, 0, 0, 0]
    assert rows[1] == [1, 0, 0, 0, 0, 0]
    assert rows[2] == [1, 1, 0, 0, 0, 0]


def test_resolve_char_tracker_dir_uses_explicit_override(tmp_path):
    p = tmp_path / "stage05"
    p.mkdir()
    assert resolve_char_tracker_dir(p) == p


def test_resolve_char_tracker_dir_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("CHAR_TRACKER_STAGE5_DIR", raising=False)
    # Point at a definitely-nonexistent path so default scratch dir doesn't pick up.
    monkeypatch.setattr(
        "charnet.visual_presence.DEFAULT_CHAR_TRACKER_DIR",
        tmp_path / "definitely_not_here",
    )

    assert resolve_char_tracker_dir(tmp_path / "also_missing") is None


# ---------------------------------------------------------------------------
# process_tsv: confidence/review-reason recomputation
# ---------------------------------------------------------------------------


def test_process_tsv_flags_speaker_offscreen(tmp_path):
    path = tmp_path / "episode_sentence_speaker_table.tsv"
    pd.DataFrame(
        [
            {
                "scene_id": "1",
                "sentence_id": "1",
                "start": "0.0",
                "end": "1.0",
                "utterance": "Hi.",
                "speaker": "Rachel",
                "utterance_ct": "Hi.",
                "speaker_ct": "Rachel",
                "visual_presence": "present",
                "visual_presence_source": "char_tracker_stage05",
                "visual_presence_chars": "ross",  # Ross is on screen, not Rachel
                "speaker_visual_presence": "absent",
                "speaker_visual_ratio": "0.0000",
                "visual_presence_note": "",
                "annotation_confidence": "high",
                "annotation_review_reason": "",
            }
        ]
    ).to_csv(path, sep="\t", index=False)

    df, summary = process_tsv(path, _cfg())

    assert df.loc[0, "visual_presence"] == "present"
    assert df.loc[0, "speaker_visual_presence"] == "absent"
    assert df.loc[0, "annotation_confidence"] == "low"
    assert df.loc[0, "annotation_review_reason"] == "speaker_offscreen"
    assert df.loc[0, "review_flag"]
    assert "speaker_offscreen" in df.loc[0, "review_reason"]
    assert summary["speaker_offscreen_rows"] == 1


def test_process_tsv_defaults_missing_visual_data_without_review(tmp_path):
    path = tmp_path / "legacy_sentence_speaker_table.tsv"
    pd.DataFrame(
        [
            {
                "scene_id": "1",
                "sentence_id": "1",
                "start": "0.0",
                "end": "1.0",
                "utterance": "Hi.",
                "speaker": "Rachel",
                "utterance_ct": "Hi.",
                "speaker_ct": "Rachel",
            }
        ]
    ).to_csv(path, sep="\t", index=False)

    df, summary = process_tsv(path, _cfg())

    assert df.loc[0, "visual_presence"] == "unavailable"
    assert df.loc[0, "visual_presence_source"] == "none"
    assert df.loc[0, "annotation_confidence"] == "medium"
    assert df.loc[0, "annotation_review_reason"] == "visual_presence_unavailable"
    assert not df.loc[0, "review_flag"]
    assert summary["visual_unavailable_rows"] == 1


# ---------------------------------------------------------------------------
# infer_speaker: visual tiebreaker
# ---------------------------------------------------------------------------


def _ambiguous_scene_df(visual_presence_chars: str) -> pd.DataFrame:
    """Three-row scene: Rachel speaks, mystery line, Ross speaks.

    Mystery line is text-ambiguous between Rachel and Ross. visual_presence_chars
    on the mystery row determines who infer_speaker should pick.
    """
    return pd.DataFrame(
        [
            {
                "scene_id": 1,
                "utterance": "Hey, what's up?",
                "speaker": "Rachel",
                "speaker_ct": "Rachel",
                "visual_presence": "present",
                "visual_presence_chars": "rachel",
            },
            {
                "scene_id": 1,
                "utterance": "Maybe.",  # generic — no name mentions, mid-length
                "speaker": "",
                "speaker_ct": "",
                "visual_presence": "present",
                "visual_presence_chars": visual_presence_chars,
            },
            {
                "scene_id": 1,
                "utterance": "I know, right?",
                "speaker": "Ross",
                "speaker_ct": "Ross",
                "visual_presence": "present",
                "visual_presence_chars": "ross",
            },
        ]
    )


def test_infer_speaker_uses_visual_to_pick_between_two_candidates():
    df = _ambiguous_scene_df(visual_presence_chars="rachel")

    speaker, method, conf, _score, _review, _reason, _note = infer_speaker(df, 1, _cfg())

    assert speaker == "Rachel"
    assert method == "visual_disambiguation"
    assert conf == "medium"


def test_infer_speaker_skips_visual_tiebreaker_when_both_or_neither_present():
    # Both candidates on screen → ambiguous, fall through to text rule.
    df_both = _ambiguous_scene_df(visual_presence_chars="rachel|ross")
    speaker_both, method_both, *_ = infer_speaker(df_both, 1, _cfg())
    assert method_both != "visual_disambiguation"

    # Neither on screen → visual gives no signal, fall through to text rule.
    df_neither = _ambiguous_scene_df(visual_presence_chars="monica")
    speaker_neither, method_neither, *_ = infer_speaker(df_neither, 1, _cfg())
    assert method_neither != "visual_disambiguation"
    # Both fallthrough cases must still return *something* (long-ambiguous → prev_sp).
    assert speaker_both in {"Rachel", "Ross"}
    assert speaker_neither in {"Rachel", "Ross"}
