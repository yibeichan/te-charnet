from __future__ import annotations

import pandas as pd

from charnet.speaker_fill import PipelineConfig, process_tsv
from charnet.visual_presence import (
    add_visual_presence_to_sentences,
    assess_annotation_confidence,
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


def test_visual_presence_for_window_reports_unavailable_without_frames():
    fields = visual_presence_for_window(None, 0.0, 1.0)

    assert fields["visual_presence"] == "unavailable"
    assert fields["visual_presence_source"] == "none"
    assert fields["visual_presence_note"] == "visual_presence_data_unavailable"


def test_visual_presence_for_window_aggregates_frame_overlap():
    frames = [False, False, True, True, False, False]

    fields = visual_presence_for_window(frames, 0.0, 0.4, fps=10.0)

    assert fields["visual_presence"] == "present"
    assert fields["visual_presence_frames"] == "2"
    assert fields["visual_presence_total_frames"] == "4"
    assert fields["visual_presence_ratio"] == "0.5000"


def test_add_visual_presence_to_sentences_marks_absent_windows():
    sentences = [{"start": 0.0, "end": 0.2}, {"start": 0.2, "end": 0.4}]

    counts = add_visual_presence_to_sentences(sentences, [False, False, True, True], fps=10.0)

    assert counts == {"present": 1, "partial": 0, "absent": 1, "unknown": 0, "unavailable": 0}
    assert sentences[0]["visual_presence"] == "absent"
    assert sentences[1]["visual_presence"] == "present"


def test_annotation_confidence_flags_speaker_without_visual_presence():
    confidence, reason = assess_annotation_confidence(
        speaker="Rachel",
        speaker_ct="Rachel",
        speaker_confidence="high",
        visual_presence="absent",
    )

    assert confidence == "medium"
    assert reason == "speaker_no_visual_presence"


def test_process_tsv_preserves_visual_columns_and_sets_review_reason(tmp_path):
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
                "visual_presence": "absent",
                "visual_presence_source": "deepgaze_maxpeak",
                "visual_presence_ratio": "0.0000",
                "visual_presence_frames": "0",
                "visual_presence_total_frames": "30",
                "visual_presence_note": "",
                "annotation_confidence": "high",
                "annotation_review_reason": "",
            }
        ]
    ).to_csv(path, sep="\t", index=False)

    df, summary = process_tsv(path, _cfg())

    assert df.loc[0, "visual_presence"] == "absent"
    assert df.loc[0, "annotation_confidence"] == "medium"
    assert df.loc[0, "annotation_review_reason"] == "speaker_no_visual_presence"
    assert df.loc[0, "review_flag"]
    assert "speaker_no_visual_presence" in df.loc[0, "review_reason"]
    assert summary["visual_absent_rows"] == 1
    assert summary["annotation_medium_confidence_rows"] == 1
