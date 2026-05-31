import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import export_topic_trace as E  # noqa: E402


def _fake_encoder(texts):
    return np.array([[float(len(t)), float(sum(c in "aeiou" for c in t.lower()))] for t in texts])


def test_episode_trace_end_to_end(tmp_path):
    sent = tmp_path / "sentences" / "s1"
    sent.mkdir(parents=True)
    rows = [{"scene_id": 1, "utterance_ct": w, "utterance": w, "start": float(i), "end": float(i) + 1.0}
            for i, w in enumerate(["alpha", "beta", "gamma", "delta", "epsilon"])]
    pd.DataFrame(rows).to_csv(sent / "friends_s01e01a_sentence_speaker_table.tsv", sep="\t", index=False)

    df = E._episode_trace("s01e01a", tmp_path / "sentences", _fake_encoder, tmp_path / "cache",
                          w=1, tau_depth=0.1, min_spacing=0.5)
    assert df is not None
    assert list(df.columns) == ["scene_id", "onset", "block_distance", "depth", "is_peak", "w", "tau_depth", "min_spacing"]
    assert len(df) == 4  # 5 turns -> 4 gaps
    assert df["onset"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_missing_sentence_table_returns_none(tmp_path):
    out = E._episode_trace("s09e99z", tmp_path / "sentences", _fake_encoder, tmp_path / "cache",
                           w=1, tau_depth=0.1, min_spacing=0.5)
    assert out is None
