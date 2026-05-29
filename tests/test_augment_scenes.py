import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from charnet.scene_subdivide import Scene  # noqa: E402
import augment_scenes as A  # noqa: E402


def _fake_encoder(texts):
    # deterministic 2-d vectors; distinct per text
    return np.array([[float(len(t)), float(sum(c in "aeiou" for c in t.lower()))] for t in texts])


def test_build_topic_propose_encode_once_and_slice(tmp_path):
    # two scenes, a few turns each
    sent = tmp_path / "sentences" / "s1"
    sent.mkdir(parents=True)
    pd.DataFrame([
        {"scene_id": 1, "utterance_ct": "alpha", "utterance": "alpha", "start": 0.0, "end": 1.0},
        {"scene_id": 1, "utterance_ct": "beta",  "utterance": "beta",  "start": 1.0, "end": 2.0},
        {"scene_id": 2, "utterance_ct": "gamma", "utterance": "gamma", "start": 2.0, "end": 3.0},
    ]).to_csv(sent / "friends_s01e01a_sentence_speaker_table.tsv", sep="\t", index=False)

    params = {"w": 1, "tau_depth": 0.0, "min_spacing": 0.5}
    propose = A._build_topic_propose("s01e01a", tmp_path / "sentences", _fake_encoder, tmp_path / "cache", params)
    assert propose is not None
    # scene 1 has 2 turns → propose returns a (possibly empty) list of floats without error
    out1 = propose(Scene(scene_id=1, scene_desc="", start=0.0, end=2.0, shot_ids=""))
    assert isinstance(out1, list)
    # scene absent from grouping → index.get fallback (0,0) → empty slice → no boundaries, no crash
    out_missing = propose(Scene(scene_id=99, scene_desc="", start=0.0, end=2.0, shot_ids=""))
    assert out_missing == []


def test_build_topic_propose_returns_none_when_no_sentences(tmp_path):
    propose = A._build_topic_propose("s09e99z", tmp_path / "sentences", _fake_encoder, tmp_path / "cache", {"w": 1, "tau_depth": 0.0, "min_spacing": 0.5})
    assert propose is None
