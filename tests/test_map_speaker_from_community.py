"""Tests for speaker mapping helpers in map_speaker_from_community.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "friends_annotations"
    / "src"
    / "map_speaker_code"
    / "map_speaker_from_community.py"
)
SPEC = importlib.util.spec_from_file_location("map_speaker_from_community", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
map_speaker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(map_speaker)


def _sentence(
    text: str,
    speaker_ct: str | None = None,
    utterance_ct: str | None = None,
    *,
    scene_id: int = 4,
    start: float = 0.0,
    end: float = 0.0,
) -> dict:
    return {
        "text": text,
        "speaker_ct": speaker_ct,
        "speaker_mapped": speaker_ct,
        "utterance_ct": utterance_ct,
        "scene_id": scene_id,
        "start": start,
        "end": end,
        "comm_dialogue_idx": 17 if speaker_ct else None,
        "match_similarity": 0.93 if speaker_ct else 0.0,
    }


def test_anchor_chunk_fill_fills_left_and_right_neighbors():
    sentences = [
        _sentence("Oh, God."),
        _sentence("Well, it started about a half hour before the wedding."),
        _sentence(
            "I realized that I was more turned on by this gravy boat than by Barry.",
            speaker_ct="Rachel",
            utterance_ct=(
                "Oh God. Well, it started about a half hour before the wedding. "
                "I realized that I was more turned on by this gravy boat than by Barry! "
                "And then I got really freaked out."
            ),
        ),
        _sentence("And then I got really freaked out."),
    ]

    filled = map_speaker.fill_unmatched_from_anchor_chunks(
        sentences,
        anchor_min_similarity=0.6,
        expand_min_similarity=0.72,
    )

    assert filled == 3
    assert sentences[0]["speaker_ct"] == "Rachel"
    assert sentences[1]["speaker_ct"] == "Rachel"
    assert sentences[3]["speaker_ct"] == "Rachel"
    assert sentences[0]["filled_from_anchor_idx"] == 2
    assert sentences[3]["fill_method"] == "anchor_chunk"


def test_anchor_chunk_fill_stops_when_similarity_is_low():
    sentences = [
        _sentence("Completely unrelated sentence about zebras."),
        _sentence(
            "I realized that I was more turned on by this gravy boat than by Barry.",
            speaker_ct="Rachel",
            utterance_ct=(
                "Oh God. Well, it started about a half hour before the wedding. "
                "I realized that I was more turned on by this gravy boat than by Barry!"
            ),
        ),
    ]

    filled = map_speaker.fill_unmatched_from_anchor_chunks(
        sentences,
        anchor_min_similarity=0.6,
        expand_min_similarity=0.72,
    )

    assert filled == 0
    assert not sentences[0].get("speaker_ct")


def test_anchor_chunk_fill_never_overwrites_mapped_rows():
    sentences = [
        _sentence("Who wasn't invited to the wedding.", speaker_ct="Monica", utterance_ct="Who wasn't invited to the wedding."),
        _sentence(
            "I realized that I was more turned on by this gravy boat than by Barry.",
            speaker_ct="Rachel",
            utterance_ct=(
                "Oh God. Well, it started about a half hour before the wedding. "
                "I realized that I was more turned on by this gravy boat than by Barry! "
                "And then I got really freaked out."
            ),
        ),
        _sentence("And then I got really freaked out."),
    ]

    filled = map_speaker.fill_unmatched_from_anchor_chunks(
        sentences,
        anchor_min_similarity=0.6,
        expand_min_similarity=0.72,
    )

    assert filled == 1
    assert sentences[0]["speaker_ct"] == "Monica"
    assert sentences[1]["speaker_ct"] == "Rachel"
    assert sentences[2]["speaker_ct"] == "Rachel"


def test_scene_directional_residual_uses_closer_neighbor_and_updates_residual():
    sentences = [
        _sentence(
            "Oh God.",
            speaker_ct="Rachel",
            utterance_ct="Oh God.",
            start=0.0,
            end=1.0,
        ),
        _sentence(
            "Well, it started about a half hour before the wedding.",
            start=1.1,
            end=2.0,
        ),
        _sentence(
            "I realized that I was more turned on by this gravy boat than by Barry.",
            speaker_ct="Rachel",
            utterance_ct=(
                "Oh God. Well, it started about a half hour before the wedding. "
                "I realized that I was more turned on by this gravy boat than by Barry."
            ),
            start=2.05,
            end=3.0,
        ),
    ]

    filled = map_speaker.fill_unmatched_scene_directional_residual(
        sentences,
        fill_min_similarity=0.72,
        context_min_similarity=0.65,
        ambiguity_margin=0.05,
        max_rounds=4,
    )

    assert filled == 1
    assert sentences[1]["speaker_ct"] == "Rachel"
    assert sentences[1]["fill_method"] == "scene_directional_residual"
    assert sentences[1]["filled_from_neighbor_idx"] == 2
    assert (
        sentences[2]["utterance_ct"]
        == "I realized that I was more turned on by this gravy boat than by Barry"
    )


def test_scene_directional_residual_does_not_cross_scene_boundary():
    sentences = [
        _sentence(
            "Oh God.",
            speaker_ct="Rachel",
            utterance_ct="Oh God.",
            scene_id=4,
            start=0.0,
            end=1.0,
        ),
        _sentence(
            "Well, it started about a half hour before the wedding.",
            scene_id=4,
            start=1.1,
            end=2.0,
        ),
        _sentence(
            "I realized that I was more turned on by this gravy boat than by Barry.",
            speaker_ct="Rachel",
            utterance_ct=(
                "Oh God. Well, it started about a half hour before the wedding. "
                "I realized that I was more turned on by this gravy boat than by Barry."
            ),
            scene_id=5,
            start=2.05,
            end=3.0,
        ),
    ]

    filled = map_speaker.fill_unmatched_scene_directional_residual(sentences)

    assert filled == 0
    assert not sentences[1].get("speaker_ct")


def test_scene_directional_residual_can_use_left_neighbor_direction():
    sentences = [
        _sentence(
            "B line",
            speaker_ct="Rachel",
            utterance_ct="A line. B line. C line. D line.",
            start=0.0,
            end=1.0,
        ),
        _sentence(
            "C line",
            start=1.05,
            end=1.95,
        ),
        _sentence(
            "D line",
            speaker_ct="Rachel",
            utterance_ct="D line.",
            start=2.2,
            end=3.0,
        ),
    ]

    filled = map_speaker.fill_unmatched_scene_directional_residual(sentences)

    assert filled == 1
    assert sentences[1]["speaker_ct"] == "Rachel"
    assert sentences[1]["filled_from_neighbor_idx"] == 0
    assert sentences[0]["utterance_ct"] == "A line B line"


def test_anchor_chunk_fill_skips_ambiguous_anchor():
    """When two chunks score similarly for the anchor text, expansion should be skipped."""
    sentences = [
        _sentence("Hello world."),
        _sentence(
            "Hello world",
            speaker_ct="Rachel",
            utterance_ct="Hello world. Hello world. And then I got really freaked out.",
        ),
        _sentence("And then I got really freaked out."),
    ]

    filled = map_speaker.fill_unmatched_from_anchor_chunks(
        sentences,
        anchor_min_similarity=0.6,
        expand_min_similarity=0.72,
        ambiguity_margin=0.05,
    )

    # The first empty row matches both chunk 0 and chunk 1 nearly equally —
    # ambiguity check should block expansion, leaving it unfilled.
    assert sentences[0].get("speaker_ct") is None


def test_neighbor_fill_assigns_residual_to_empty_row():
    """Fallback neighbor fill should use the residual of the neighbour's utterance_ct."""
    sentences = [
        _sentence(
            "It started before the wedding.",
            speaker_ct="Rachel",
            utterance_ct="It started before the wedding. And I got really freaked out.",
            start=0.0,
            end=2.0,
        ),
        _sentence(
            "And I got really freaked out.",
            start=2.1,
            end=3.5,
        ),
    ]

    filled = map_speaker.fill_unmatched_from_neighbors(
        sentences,
        min_similarity=0.7,
    )

    assert filled == 1
    assert sentences[1]["speaker_ct"] == "Rachel"
    assert sentences[1]["fill_method"] == "neighbor_residual"


def test_neighbor_fill_skips_below_threshold():
    """When the residual does not match the empty row, it should not be filled."""
    sentences = [
        _sentence(
            "It started before the wedding.",
            speaker_ct="Rachel",
            utterance_ct="It started before the wedding. And I got really freaked out.",
            start=0.0,
            end=2.0,
        ),
        _sentence(
            "Completely unrelated zebra sentence.",
            start=2.1,
            end=3.5,
        ),
    ]

    filled = map_speaker.fill_unmatched_from_neighbors(
        sentences,
        min_similarity=0.7,
    )

    assert filled == 0
    assert not sentences[1].get("speaker_ct")


def test_length_ratio_filter_rejects_short_community_match():
    """A community utterance much shorter than the local sentence should be rejected."""
    # local: ~20 tokens, community: ~3 tokens → ratio ≈ 7, above cap of 5
    sentences = [
        {
            "text": "And I know that you and I have kind of drifted apart but you are the only person I knew who lived here.",
            "speaker_ct": None,
            "speaker_mapped": None,
            "utterance_ct": None,
            "scene_id": None,
            "comm_dialogue_idx": 42,
            "match_similarity": 0.55,
        }
    ]
    # Simulate what map_speakers_for_episode does with length_ratio_cap:
    local_tokens = len(map_speaker.normalize_for_match(sentences[0]["text"]).split())
    comm_tokens = len(map_speaker.normalize_for_match("Who wasn't invited.").split())
    length_ok = comm_tokens == 0 or (local_tokens / comm_tokens) <= 5.0
    assert not length_ok, "Expected ratio check to reject this match"


def test_length_ratio_filter_accepts_normal_match():
    """A community utterance of comparable length should pass the filter."""
    local_tokens = len(map_speaker.normalize_for_match("I realized I was more turned on by this gravy boat than by Barry.").split())
    comm_tokens = len(map_speaker.normalize_for_match("I realized that I was more turned on by this gravy boat than by Barry!").split())
    length_ok = comm_tokens == 0 or (local_tokens / comm_tokens) <= 5.0
    assert length_ok, "Expected ratio check to accept this match"


def test_short_sentence_expansion_fills_short_row():
    """A very short sentence should be filled when short_sent_min_similarity is applied."""
    sentences = [
        {  # anchor: Rachel with long utterance that contains both phrases
            "text": "I realized that I was more turned on by this gravy boat than by Barry.",
            "speaker_ct": "Rachel",
            "speaker_mapped": "Rachel",
            "utterance_ct": (
                "I realized that I was more turned on by this gravy boat than by Barry. "
                "You know."
            ),
            "scene_id": 4,
            "start": 0.0,
            "end": 2.0,
            "comm_dialogue_idx": 17,
            "match_similarity": 0.93,
        },
        {  # short sentence — would fail at 0.72 but pass at 0.40
            "text": "You know.",
            "speaker_ct": None,
            "speaker_mapped": None,
            "utterance_ct": None,
            "scene_id": 4,
            "start": 2.1,
            "end": 2.5,
            "comm_dialogue_idx": None,
            "match_similarity": 0.0,
        },
    ]

    filled = map_speaker.fill_unmatched_from_anchor_chunks(
        sentences,
        anchor_min_similarity=0.6,
        expand_min_similarity=0.72,
        short_sent_token_threshold=4,
        short_sent_min_similarity=0.40,
    )

    assert filled == 1
    assert sentences[1]["speaker_ct"] == "Rachel"
    assert sentences[1]["fill_method"] == "anchor_chunk"


def test_comm_dialogue_propagation_fills_stranded_rows():
    """Unmapped rows whose comm_dialogue_idx maps to exactly one speaker are filled."""
    sentences = [
        {
            "text": "I realized that I was more turned on by this gravy boat than by Barry.",
            "speaker_ct": "Rachel",
            "speaker_mapped": "Rachel",
            "utterance_ct": "...",
            "scene_id": 4,
            "comm_dialogue_idx": 42,
        },
        {
            "text": "You know.",
            "speaker_ct": None,
            "speaker_mapped": None,
            "utterance_ct": None,
            "scene_id": 4,
            "comm_dialogue_idx": 42,
        },
        {
            "text": "Anyway I just had to get out of there.",
            "speaker_ct": None,
            "speaker_mapped": None,
            "utterance_ct": None,
            "scene_id": 4,
            "comm_dialogue_idx": 42,
        },
    ]

    filled = map_speaker.fill_unmatched_by_dialogue_index(sentences)

    assert filled == 2
    assert sentences[1]["speaker_ct"] == "Rachel"
    assert sentences[1]["fill_method"] == "comm_dialogue_propagation"
    assert sentences[2]["speaker_ct"] == "Rachel"


def test_comm_dialogue_propagation_skips_ambiguous():
    """Rows whose comm_dialogue_idx is shared by multiple speakers are left empty."""
    sentences = [
        {
            "text": "Sure.",
            "speaker_ct": "Rachel",
            "speaker_mapped": "Rachel",
            "utterance_ct": "Sure.",
            "scene_id": 4,
            "comm_dialogue_idx": 99,
        },
        {
            "text": "Hi.",
            "speaker_ct": "Monica",
            "speaker_mapped": "Monica",
            "utterance_ct": "Hi.",
            "scene_id": 4,
            "comm_dialogue_idx": 99,
        },
        {
            "text": "Okay.",
            "speaker_ct": None,
            "speaker_mapped": None,
            "utterance_ct": None,
            "scene_id": 4,
            "comm_dialogue_idx": 99,
        },
    ]

    filled = map_speaker.fill_unmatched_by_dialogue_index(sentences)

    assert filled == 0
    assert sentences[2]["speaker_ct"] is None


def test_fill_scene_ids_propagates_forward():
    """fill_scene_ids should forward-fill scene_id from matched rows to unmatched ones."""
    sentences = [
        {"text": "A", "scene_id": 2, "speaker_ct": "Rachel"},
        {"text": "B", "scene_id": None, "speaker_ct": None},
        {"text": "C", "scene_id": None, "speaker_ct": None},
        {"text": "D", "scene_id": 5, "speaker_ct": "Monica"},
        {"text": "E", "scene_id": None, "speaker_ct": None},
    ]

    map_speaker.fill_scene_ids(sentences)

    assert sentences[0]["scene_id"] == 2
    assert sentences[1]["scene_id"] == 2
    assert sentences[2]["scene_id"] == 2
    assert sentences[3]["scene_id"] == 5
    assert sentences[4]["scene_id"] == 5
