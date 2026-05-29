import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from charnet.topic_shift import Turn, build_text, group_turns_for_scene


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
