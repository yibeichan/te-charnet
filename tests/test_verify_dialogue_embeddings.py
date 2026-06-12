import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import verify_dialogue_embeddings as V  # noqa: E402


def _hand_key(texts, model_id="all-MiniLM-L6-v2"):
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(b"\xff")
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _build_fixture(tmp_path):
    """Hand-built table + matching product TSV/NPZ + cache NPZ. Returns roots."""
    sent = tmp_path / "sentences" / "s1"
    sent.mkdir(parents=True)
    rows = [
        {"scene_id": 1, "utterance_ct": "hi there", "utterance": "hi", "start": 0.0, "end": 1.0},
        {"scene_id": 1, "utterance_ct": "hi there", "utterance": "there", "start": 1.0, "end": 2.5},
        {"scene_id": 1, "utterance_ct": "ok", "utterance": "ok", "start": 3.0, "end": 4.0},
        {"scene_id": 2, "utterance_ct": "", "utterance": "yes", "start": 10.0, "end": 11.0},
    ]
    pd.DataFrame(rows).to_csv(sent / "friends_s01e01a_sentence_speaker_table.tsv", sep="\t", index=False)

    texts = ["hi there", "ok", "yes"]
    key = _hand_key(texts)
    vecs = np.arange(12, dtype=np.float32).reshape(3, 4)

    out = tmp_path / "product" / "s1"
    out.mkdir(parents=True)
    pd.DataFrame({
        "turn_id": [0, 1, 2], "scene_id": [1, 1, 2],
        "start": [0.0, 3.0, 10.0], "end": [2.5, 4.0, 11.0], "n_sentences": [2, 1, 1],
    }).to_csv(out / "friends_s01e01a_dialogue_turns.tsv", sep="\t", index=False)
    np.savez(out / "friends_s01e01a_dialogue_embeddings.npz", vecs=vecs, key=np.array(key))

    cache = tmp_path / "cache" / "s1"
    cache.mkdir(parents=True)
    np.savez(cache / "s01e01a.npz", vecs=vecs, key=np.array(key))
    return tmp_path / "sentences", tmp_path / "product", tmp_path / "cache"


def _run(sentences, product, cache, extra=()):
    argv = ["verify_dialogue_embeddings.py",
            "--tables-root", str(sentences), "--product-root", str(product),
            "--cache-root", str(cache), *extra]
    return V.run(argv[1:])


def test_clean_fixture_exits_zero(tmp_path):
    assert _run(*_build_fixture(tmp_path), extra=("--expected-dim", "4")) == 0


def test_dim_check_uses_expected_dim_flag(tmp_path):
    # fixture vectors are 4-d, not 384-d: default must fail, flag must pass
    roots = _build_fixture(tmp_path)
    assert _run(*roots, extra=("--expected-dim", "4")) == 0
    assert _run(*roots) == 1


def test_perturbed_vector_fails(tmp_path):
    sentences, product, cache = _build_fixture(tmp_path)
    npz = product / "s1" / "friends_s01e01a_dialogue_embeddings.npz"
    d = dict(np.load(npz, allow_pickle=False))
    d["vecs"] = d["vecs"].copy()
    d["vecs"][1, 2] += 0.5
    np.savez(npz, **d)
    assert _run(sentences, product, cache, extra=("--expected-dim", "4")) == 1


def test_permuted_vecs_with_valid_key_fails(tmp_path):
    sentences, product, cache = _build_fixture(tmp_path)
    npz = product / "s1" / "friends_s01e01a_dialogue_embeddings.npz"
    d = dict(np.load(npz, allow_pickle=False))
    d["vecs"] = d["vecs"][[1, 0, 2]]  # key stays valid; binding must catch this
    np.savez(npz, **d)
    assert _run(sentences, product, cache, extra=("--expected-dim", "4")) == 1


def test_wrong_tsv_timing_fails(tmp_path):
    sentences, product, cache = _build_fixture(tmp_path)
    tsv = product / "s1" / "friends_s01e01a_dialogue_turns.tsv"
    df = pd.read_csv(tsv, sep="\t")
    df.loc[0, "end"] = 99.0
    df.to_csv(tsv, sep="\t", index=False)
    assert _run(sentences, product, cache, extra=("--expected-dim", "4")) == 1


def test_wrong_n_sentences_fails(tmp_path):
    sentences, product, cache = _build_fixture(tmp_path)
    tsv = product / "s1" / "friends_s01e01a_dialogue_turns.tsv"
    df = pd.read_csv(tsv, sep="\t")
    df.loc[0, "n_sentences"] = 1
    df.to_csv(tsv, sep="\t", index=False)
    assert _run(sentences, product, cache, extra=("--expected-dim", "4")) == 1


def test_missing_product_npz_is_skip_not_failure(tmp_path):
    sentences, product, cache = _build_fixture(tmp_path)
    (product / "s1" / "friends_s01e01a_dialogue_embeddings.npz").unlink()
    assert _run(sentences, product, cache, extra=("--expected-dim", "4")) == 0


def test_missing_cache_with_product_npz_fails(tmp_path):
    sentences, product, cache = _build_fixture(tmp_path)
    (cache / "s1" / "s01e01a.npz").unlink()
    assert _run(sentences, product, cache, extra=("--expected-dim", "4")) == 1


def test_nothing_checkable_exits_two(tmp_path):
    empty = tmp_path / "none"
    assert _run(empty, empty, empty) == 2


def test_nan_scene_rows_are_accounted(tmp_path):
    sentences, product, cache = _build_fixture(tmp_path)
    tpath = sentences / "s1" / "friends_s01e01a_sentence_speaker_table.tsv"
    df = pd.read_csv(tpath, sep="\t")
    df.loc[len(df)] = {"scene_id": np.nan, "utterance_ct": "ghost", "utterance": "ghost",
                       "start": 50.0, "end": 51.0}
    df.to_csv(tpath, sep="\t", index=False)
    # dropped-row is reported but reconstruction still matches -> pass
    assert _run(sentences, product, cache, extra=("--expected-dim", "4")) == 0
