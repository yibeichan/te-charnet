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
    assert len(df) == 4  # 5 turns -> 4 inter-turn gaps
    assert df["onset"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_main_writes_tsv_and_sidecars(tmp_path, monkeypatch):
    # tiny sentence table
    sent = tmp_path / "sentences" / "s1"
    sent.mkdir(parents=True)
    rows = [{"scene_id": 1, "utterance_ct": w, "utterance": w, "start": float(i), "end": float(i) + 1.0}
            for i, w in enumerate(["alpha", "beta", "gamma", "delta", "epsilon"])]
    pd.DataFrame(rows).to_csv(sent / "friends_s01e01a_sentence_speaker_table.tsv", sep="\t", index=False)
    # a scenes dir so expand_episode_spec can resolve the episode
    scenes = tmp_path / "scenes" / "s1"
    scenes.mkdir(parents=True)
    (scenes / "friends_s01e01a_scene_summary.tsv").write_text("scene_id\tstart\tend\n")

    out_dir = tmp_path / "topic_shift"
    # avoid loading the real model
    monkeypatch.setattr(E.ts, "minilm_encoder", lambda: _fake_encoder)
    monkeypatch.setattr(sys, "argv", [
        "export_topic_trace.py", "--episodes", "s01e01a",
        "--scenes-in", str(tmp_path / "scenes"),
        "--sentences-in", str(tmp_path / "sentences"),
        "--out-dir", str(out_dir),
        "--cache-dir", str(tmp_path / "cache"),
        "--w", "1", "--tau-depth", "0.1", "--min-spacing", "0.5",
    ])
    E.main()

    import json
    tsv = out_dir / "s1" / "friends_s01e01a_topic_trace.tsv"
    assert tsv.exists()
    df = pd.read_csv(tsv, sep="\t")
    assert list(df.columns) == ["scene_id", "onset", "block_distance", "depth", "is_peak", "w", "tau_depth", "min_spacing"]
    assert len(df) == 4  # 5 turns -> 4 inter-turn gaps
    dd = json.loads((out_dir / "topic_trace.json").read_text())
    assert "block_distance" in dd
    desc = json.loads((out_dir.parent / "dataset_description.json").read_text())
    assert desc["DatasetType"] == "derivative"


def test_missing_sentence_table_returns_none(tmp_path):
    out = E._episode_trace("s09e99z", tmp_path / "sentences", _fake_encoder, tmp_path / "cache",
                           w=1, tau_depth=0.1, min_spacing=0.5)
    assert out is None
