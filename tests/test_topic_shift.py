import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from charnet.topic_shift import Turn, build_text, group_turns_for_scene, block_distance_trace, peak_depths, propose_topic_boundaries
from charnet.topic_shift import embed_texts_cached


def test_build_text_prefers_ct_falls_back_to_utterance():
    assert build_text("clean line", "asr noise") == "clean line"
    assert build_text("", "asr fallback") == "asr fallback"
    assert build_text(np.nan, "asr fallback") == "asr fallback"   # NaN, not "nan"
    assert build_text("   ", "asr fallback") == "asr fallback"


def test_group_turns_dedups_consecutive_identical_ct():
    df = pd.DataFrame([
        {"utterance_ct": "Hi there folks", "utterance": "hi", "start": 1.0, "end": 2.0},
        {"utterance_ct": "Hi there folks", "utterance": "there", "start": 2.0, "end": 3.5},
        {"utterance_ct": "Different turn", "utterance": "diff", "start": 3.5, "end": 5.0},
    ])
    turns = group_turns_for_scene(df)
    assert turns == [
        Turn(text="Hi there folks", start=1.0, end=3.5),
        Turn(text="Different turn", start=3.5, end=5.0),
    ]


def test_group_turns_blank_ct_uses_utterance_and_does_not_merge():
    df = pd.DataFrame([
        {"utterance_ct": "", "utterance": "first asr", "start": 1.0, "end": 2.0},
        {"utterance_ct": "", "utterance": "second asr", "start": 2.0, "end": 3.0},
    ])
    turns = group_turns_for_scene(df)
    assert [t.text for t in turns] == ["first asr", "second asr"]


def test_group_turns_nan_ct_uses_utterance_fallback():
    df = pd.DataFrame([
        {"utterance_ct": np.nan, "utterance": "asr one", "start": 1.0, "end": 2.0},
        {"utterance_ct": np.nan, "utterance": "asr two", "start": 2.0, "end": 3.0},
    ])
    turns = group_turns_for_scene(df)
    # NaN ct → no merge; each row falls back to its own utterance
    assert [t.text for t in turns] == ["asr one", "asr two"]


def test_block_distance_trace_flags_the_topic_change():
    # 6 turns: first 3 ≈ vector A, last 3 ≈ vector B. W=2 → gap at index 2 (turn2|turn3) is largest.
    A = np.array([1.0, 0.0])
    B = np.array([0.0, 1.0])
    vecs = np.stack([A, A, A, B, B, B])
    trace = block_distance_trace(vecs, w=2)
    assert len(trace) == 5                       # n_turns - 1
    assert np.argmax(trace) == 2                 # boundary between the two halves
    assert trace[2] > 0.9                         # near-orthogonal blocks


def test_peak_depths_picks_local_maxima():
    trace = np.array([0.1, 0.2, 0.9, 0.2, 0.15, 0.8, 0.1])
    peaks = dict(peak_depths(trace))             # {gap_index: depth}
    assert 2 in peaks and 5 in peaks
    assert peaks[2] > peaks[5]                    # deeper peak first


def test_propose_topic_boundaries_returns_turn_end_time():
    A, B = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    vecs = np.stack([A, A, A, B, B, B])
    turns = [Turn("t", float(i), float(i) + 1.0) for i in range(6)]  # each 1s, contiguous
    subs = propose_topic_boundaries(
        turns, vecs, w=2, tau_depth=0.3, min_spacing=0.5,
    )
    # gap index 2 → boundary at turns[2].end == 3.0
    assert subs == [3.0]


def test_propose_topic_boundaries_too_few_turns():
    vecs = np.zeros((3, 2))
    turns = [Turn("t", float(i), float(i) + 1.0) for i in range(3)]
    assert propose_topic_boundaries(turns, vecs, w=2, tau_depth=0.3, min_spacing=0.5) == []


def test_propose_topic_boundaries_min_spacing_suppresses_nearby_peak():
    A = np.array([1.0, 0.0])
    B = np.array([0.0, 1.0])
    C = np.array([1.0, 1.0]) / np.sqrt(2)
    vecs = np.stack([A, A, B, B, C, C])
    turns = [Turn("t", float(i), float(i) + 1.0) for i in range(6)]
    # trace = [0, 1.0, 0, ~0.293, 0]: deep peak at gap 1 (turns[1].end=2.0),
    # shallow peak at gap 3 (turns[3].end=4.0), 2.0 s apart.
    near = propose_topic_boundaries(turns, vecs, w=1, tau_depth=0.1, min_spacing=2.5)
    assert near == [2.0]            # shallower gap-3 peak suppressed (2.0s < 2.5s)
    far = propose_topic_boundaries(turns, vecs, w=1, tau_depth=0.1, min_spacing=0.5)
    assert far == [2.0, 4.0]        # loose spacing retains both


class _FakeEncoder:
    """Deterministic 2-d encoder: returns [len(text), n_vowels]. Counts calls."""
    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return np.array(
            [[float(len(t)), float(sum(c in "aeiou" for c in t.lower()))] for t in texts]
        )


def test_embed_texts_cached_round_trip_and_reuse(tmp_path):
    enc = _FakeEncoder()
    texts = ["hello world", "topic shift"]
    v1 = embed_texts_cached("s01e01a", texts, enc, tmp_path)
    assert v1.shape == (2, 2)
    assert enc.calls == 1
    # second call hits cache → no new encode
    v2 = embed_texts_cached("s01e01a", texts, enc, tmp_path)
    assert enc.calls == 1
    assert np.allclose(v1, v2)


def test_embed_texts_cached_invalidates_on_text_change(tmp_path):
    enc = _FakeEncoder()
    embed_texts_cached("s01e01a", ["a", "b"], enc, tmp_path)
    embed_texts_cached("s01e01a", ["a", "c"], enc, tmp_path)  # changed → re-encode
    assert enc.calls == 2
