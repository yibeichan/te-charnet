"""Tests for the topic-trace verifier itself.

Builds a sentence table, runs the real export (export_topic_trace.main) with a
fake encoder to produce a self-consistent TSV + cache NPZ, then confirms the
independent recomputation in verify_topic_trace agrees — and FAILS on a
perturbed onset, a perturbed block_distance, a perturbed cache vector, and a
stale/missing cache. depth/is_peak are the documented negative-result audit
trail and are NOT recomputed: perturbing them must still PASS.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import export_topic_trace as EXP  # noqa: E402
import verify_topic_trace as V  # noqa: E402


def _fake_encoder(texts):
    # deterministic, 3-d, distinct per text so adjacent blocks differ
    return np.array([[float(len(t)), float(sum(map(ord, t)) % 13), 1.0] for t in texts],
                    dtype=np.float32)


def _build_fixture(tmp_path, ep="s01e01a", w=1):
    """Export TSV + cache NPZ produced by the real export path. Returns roots."""
    sent = tmp_path / "sentences" / "s1"
    sent.mkdir(parents=True)
    # one scene, 5 turns -> 4 gaps (needs >= 2w+1 = 3 turns for w=1)
    words = ["alpha", "bravo", "charlie", "delta", "echo"]
    rows = [{"scene_id": 1, "utterance_ct": word, "utterance": word,
             "start": float(i), "end": float(i) + 0.9} for i, word in enumerate(words)]
    pd.DataFrame(rows).to_csv(sent / f"friends_{ep}_sentence_speaker_table.tsv", sep="\t", index=False)

    scenes = tmp_path / "scenes" / "s1"
    scenes.mkdir(parents=True)
    (scenes / f"friends_{ep}_scene_summary.tsv").write_text("scene_id\tstart\tend\n")

    product = tmp_path / "topic_shift"
    cache = tmp_path / "cache"
    import unittest.mock as mock
    with mock.patch.object(EXP.ts, "minilm_encoder", lambda: _fake_encoder), \
         mock.patch.object(sys, "argv", [
             "export_topic_trace.py", "--episodes", ep,
             "--scenes-in", str(tmp_path / "scenes"),
             "--sentences-in", str(tmp_path / "sentences"),
             "--out-dir", str(product), "--cache-dir", str(cache),
             "--w", str(w), "--tau-depth", "0.1", "--min-spacing", "0.5"]):
        EXP.main()
    return tmp_path / "sentences", product, cache


def _run(sentences, product, cache, extra=()):
    return V.run(["--tables-root", str(sentences), "--product-root", str(product),
                  "--cache-root", str(cache), *extra])


def _tsv(product, ep="s01e01a"):
    return product / "s1" / f"friends_{ep}_topic_trace.tsv"


def _cache_npz(cache, ep="s01e01a"):
    return cache / "s1" / f"{ep}.npz"


def test_clean_fixture_exits_zero(tmp_path):
    assert _run(*_build_fixture(tmp_path)) == 0


def test_perturbed_block_distance_fails(tmp_path):
    sentences, product, cache = _build_fixture(tmp_path)
    df = pd.read_csv(_tsv(product), sep="\t")
    df.loc[0, "block_distance"] = df.loc[0, "block_distance"] + 0.3
    df.to_csv(_tsv(product), sep="\t", index=False)
    assert _run(sentences, product, cache) == 1


def test_perturbed_onset_fails(tmp_path):
    sentences, product, cache = _build_fixture(tmp_path)
    df = pd.read_csv(_tsv(product), sep="\t")
    df.loc[0, "onset"] = df.loc[0, "onset"] + 5.0
    df.to_csv(_tsv(product), sep="\t", index=False)
    assert _run(sentences, product, cache) == 1


def test_perturbed_cache_vector_fails(tmp_path):
    # cache vecs changed but TSV block_distance original -> independent recompute
    # disagrees (key still valid: only the recompute catches it)
    sentences, product, cache = _build_fixture(tmp_path)
    d = dict(np.load(_cache_npz(cache), allow_pickle=False))
    d["vecs"] = d["vecs"].copy()
    d["vecs"][0, 0] += 3.0
    np.savez(_cache_npz(cache), **d)
    assert _run(sentences, product, cache) == 1


def test_stale_cache_key_fails(tmp_path):
    sentences, product, cache = _build_fixture(tmp_path)
    d = dict(np.load(_cache_npz(cache), allow_pickle=False))
    np.savez(_cache_npz(cache), vecs=d["vecs"], key=np.array("deadbeef"))
    assert _run(sentences, product, cache) == 1


def test_missing_cache_fails(tmp_path):
    # block_distance cannot be vouched for without the cache vectors
    sentences, product, cache = _build_fixture(tmp_path)
    _cache_npz(cache).unlink()
    assert _run(sentences, product, cache) == 1


def test_corrupt_cache_fails_cleanly(tmp_path, capsys):
    sentences, product, cache = _build_fixture(tmp_path)
    _cache_npz(cache).write_bytes(b"\x00not a zip")
    rc = _run(sentences, product, cache)
    assert rc == 1
    assert "cache" in capsys.readouterr().out.lower()


def test_perturbed_depth_is_peak_still_passes(tmp_path):
    # depth/is_peak are the documented negative-result audit trail — NOT recomputed
    sentences, product, cache = _build_fixture(tmp_path)
    df = pd.read_csv(_tsv(product), sep="\t")
    df["is_peak"] = ~df["is_peak"].astype(bool)
    df["depth"] = 999.0
    df.to_csv(_tsv(product), sep="\t", index=False)
    assert _run(sentences, product, cache) == 0


def test_wrong_row_count_fails(tmp_path):
    sentences, product, cache = _build_fixture(tmp_path)
    df = pd.read_csv(_tsv(product), sep="\t")
    df.drop(index=df.index[-1]).to_csv(_tsv(product), sep="\t", index=False)
    assert _run(sentences, product, cache) == 1


def test_nothing_checkable_exits_two(tmp_path):
    empty = tmp_path / "none"
    assert _run(empty, empty, empty) == 2


def test_legitimately_empty_tsv_is_skip_not_fail(tmp_path, capsys):
    # w=3 needs 2w+1 = 7 turns per scene; the fixture scene has 5, so the export
    # legitimately writes a zero-row TSV. w is recorded only in rows, so an empty
    # TSV cannot be verified at any particular w — it must skip, not false-FAIL
    # against the w=1 fallback.
    sentences, product, cache = _build_fixture(tmp_path, w=3)
    assert pd.read_csv(_tsv(product), sep="\t").empty  # fixture sanity: export wrote 0 rows
    rc = _run(sentences, product, cache)
    out = capsys.readouterr().out
    assert rc == 2  # only episode is unverifiable -> nothing checkable
    assert "empty" in out.lower()
