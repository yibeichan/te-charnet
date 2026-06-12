import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import export_dialogue_embeddings as E  # noqa: E402
from charnet import topic_shift as ts  # noqa: E402


def _fake_encoder(texts):
    # deterministic, dimension 4, distinct per text
    return np.array([[float(len(t)), float(sum(map(ord, t)) % 97), 1.0, 0.0] for t in texts],
                    dtype=np.float32)


def _write_table(tmp_path, rows, ep="s01e01a"):
    sent = tmp_path / "sentences" / "s1"
    sent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(sent / f"friends_{ep}_sentence_speaker_table.tsv", sep="\t", index=False)
    return tmp_path / "sentences"


BASIC_ROWS = [
    # scene 1: rows 0+1 merge (same ct) -> turn 0 (n=2); row 2 -> turn 1
    {"scene_id": 1, "utterance_ct": "hi there", "utterance": "hi", "start": 0.0, "end": 1.0},
    {"scene_id": 1, "utterance_ct": "hi there", "utterance": "there", "start": 1.0, "end": 2.5},
    {"scene_id": 1, "utterance_ct": "ok", "utterance": "ok", "start": 3.0, "end": 4.0},
    # scene 2: blank ct -> utterance fallback, two separate turns
    {"scene_id": 2, "utterance_ct": "", "utterance": "yes", "start": 10.0, "end": 11.0},
    {"scene_id": 2, "utterance_ct": "", "utterance": "no", "start": 11.0, "end": 12.0},
]


def test_episode_product_rows_and_alignment(tmp_path):
    sentences_in = _write_table(tmp_path, BASIC_ROWS)
    out = E._episode_product("s01e01a", sentences_in, _fake_encoder, tmp_path / "cache")
    assert out is not None
    df, vecs, key = out
    assert list(df.columns) == ["turn_id", "scene_id", "start", "end", "n_sentences"]
    assert df["turn_id"].tolist() == [0, 1, 2, 3]
    assert df["scene_id"].tolist() == [1, 1, 2, 2]
    assert df["start"].tolist() == [0.0, 3.0, 10.0, 11.0]
    assert df["end"].tolist() == [2.5, 4.0, 11.0, 12.0]
    assert df["n_sentences"].tolist() == [2, 1, 1, 1]
    assert vecs.shape == (4, 4) and vecs.dtype == np.float32
    # row i of vecs is the embedding of turn i's text
    expected = _fake_encoder(["hi there", "ok", "yes", "no"])
    assert np.array_equal(vecs, expected)
    assert key == ts._texts_hash(["hi there", "ok", "yes", "no"], E.MODEL_ID)


def test_missing_table_returns_none(tmp_path):
    assert E._episode_product("s09e99z", tmp_path / "sentences", _fake_encoder, tmp_path / "cache") is None


def test_cache_status_new_hit_reencode(tmp_path):
    sentences_in = _write_table(tmp_path, BASIC_ROWS)
    cache = tmp_path / "cache"
    texts = ["hi there", "ok", "yes", "no"]
    key = ts._texts_hash(texts, E.MODEL_ID)
    assert E._cache_status("s01e01a", key, cache) == "new"
    E._episode_product("s01e01a", sentences_in, _fake_encoder, cache)  # populates cache
    assert E._cache_status("s01e01a", key, cache) == "hit"
    # stale pre-model_id key on disk -> re-encoded
    stale = ts._texts_hash(texts, "")
    np.savez(cache / "s1" / "s01e01a.npz", vecs=np.zeros((4, 4), np.float32), key=np.array(stale))
    assert E._cache_status("s01e01a", key, cache) == "re-encoded"


def test_main_writes_products_and_sidecars(tmp_path, monkeypatch):
    sentences_in = _write_table(tmp_path, BASIC_ROWS)
    scenes = tmp_path / "scenes" / "s1"
    scenes.mkdir(parents=True)
    (scenes / "friends_s01e01a_scene_summary.tsv").write_text("scene_id\tstart\tend\n")
    out_dir = tmp_path / "dialogue_embeddings"
    monkeypatch.setattr(E.ts, "minilm_encoder", lambda: _fake_encoder)
    monkeypatch.setattr(sys, "argv", [
        "export_dialogue_embeddings.py", "--episodes", "s01e01a",
        "--scenes-in", str(tmp_path / "scenes"),
        "--sentences-in", str(sentences_in),
        "--out-dir", str(out_dir),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    E.main()

    tsv = out_dir / "s1" / "friends_s01e01a_dialogue_turns.tsv"
    npz = out_dir / "s1" / "friends_s01e01a_dialogue_embeddings.npz"
    assert tsv.exists() and npz.exists()
    df = pd.read_csv(tsv, sep="\t")
    prod = np.load(npz, allow_pickle=False)
    assert len(df) == prod["vecs"].shape[0] == 4
    # product NPZ matches what landed in the cache
    cached = np.load(tmp_path / "cache" / "s1" / "s01e01a.npz", allow_pickle=False)
    assert np.array_equal(prod["vecs"], cached["vecs"])
    assert str(prod["key"]) == str(cached["key"])
    dd = json.loads((out_dir / "dialogue_turns.json").read_text())
    assert set(dd) == {"turn_id", "scene_id", "start", "end", "n_sentences"}
    desc = json.loads((out_dir.parent / "dataset_description.json").read_text())
    assert desc["DatasetType"] == "derivative"


def test_main_skips_missing_and_reports(tmp_path, monkeypatch, capsys):
    # scenes dir lists an episode with no sentence table -> skip, not crash
    scenes = tmp_path / "scenes" / "s1"
    scenes.mkdir(parents=True)
    (scenes / "friends_s01e02a_scene_summary.tsv").write_text("scene_id\tstart\tend\n")
    monkeypatch.setattr(E.ts, "minilm_encoder", lambda: _fake_encoder)
    monkeypatch.setattr(sys, "argv", [
        "export_dialogue_embeddings.py", "--episodes", "s01e02a",
        "--scenes-in", str(tmp_path / "scenes"),
        "--sentences-in", str(tmp_path / "sentences"),
        "--out-dir", str(tmp_path / "out"),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    E.main()
    assert "1 missing sentence tables" in capsys.readouterr().out
