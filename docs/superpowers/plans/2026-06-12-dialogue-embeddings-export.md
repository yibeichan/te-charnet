# Dialogue-Embeddings Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export per-dialogue-turn MiniLM embeddings with episode timing as a verified data product (NPZ + TSV pair per episode, 341 episodes), per the approved spec `docs/superpowers/specs/2026-06-12-dialogue-embeddings-export-design.md`.

**Architecture:** A counts-aware turn-grouping API is added to `charnet.topic_shift` (existing functions delegate, behavior unchanged except a stable tie-order sort). A new exporter script rebuilds turns from the tracked sentence tables, embeds via the existing `embed_texts_cached` cache contract, and writes an aligned TSV (tracked) + NPZ (untracked) per episode. A new independent verifier reconstructs turns/texts/hash with its own code and binds product vectors to the cache.

**Tech Stack:** Python ≥3.10, pandas, numpy, sentence-transformers (runtime only — tests use a fake encoder), pytest, ruff.

**Conventions that apply to every task:** run commands from the repo root (`/orcd/home/002/yibei/te-charnet`); tests run as `cd src && pytest` per CLAUDE.md but individual files also work as `pytest tests/test_X.py -v` from root (conftest handles paths — match how existing tests are invoked: `pytest tests/...` from root works because tests insert `src/` and `scripts/` on `sys.path` themselves). `ruff check .` must stay clean.

---

## File structure

- Modify: `src/charnet/topic_shift.py` — add `group_turns_with_counts`, `turns_by_scene_with_counts`; make `turns_by_scene`'s per-scene sort stable; existing functions delegate.
- Create: `scripts/export_dialogue_embeddings.py` — exporter CLI.
- Create: `scripts/verify_dialogue_embeddings.py` — independent verifier CLI.
- Modify: `tests/test_topic_shift.py` — counts + stable-ordering tests.
- Create: `tests/test_export_dialogue_embeddings.py` — exporter tests.
- Create: `tests/test_verify_dialogue_embeddings.py` — verifier tests.
- Modify: `.gitignore` — track the new TSVs/sidecars, keep NPZs untracked.
- Modify: `docs/data_products_catalog.md` — register the product.

---

### Task 1: Counts-aware turn grouping in `topic_shift`

**Files:**
- Modify: `src/charnet/topic_shift.py` (functions `group_turns_for_scene` ~line 51, `turns_by_scene` ~line 236)
- Test: `tests/test_topic_shift.py`

- [ ] **Step 1.1: Write the failing tests** — append to `tests/test_topic_shift.py`:

```python
from charnet.topic_shift import group_turns_with_counts, turns_by_scene_with_counts


def test_group_turns_with_counts_tracks_merged_rows():
    df = pd.DataFrame([
        {"utterance_ct": "hello there", "utterance": "x", "start": 0.0, "end": 1.0},
        {"utterance_ct": "hello there", "utterance": "y", "start": 1.0, "end": 2.0},
        {"utterance_ct": "hello there", "utterance": "z", "start": 2.0, "end": 3.0},
        {"utterance_ct": "bye", "utterance": "w", "start": 3.0, "end": 4.0},
    ])
    pairs = group_turns_with_counts(df)
    assert [(t.text, n) for t, n in pairs] == [("hello there", 3), ("bye", 1)]
    assert pairs[0][0].start == 0.0 and pairs[0][0].end == 3.0


def test_group_turns_with_counts_blank_ct_counts_one_each():
    df = pd.DataFrame([
        {"utterance_ct": "", "utterance": "a", "start": 0.0, "end": 1.0},
        {"utterance_ct": "", "utterance": "b", "start": 1.0, "end": 2.0},
    ])
    pairs = group_turns_with_counts(df)
    assert [(t.text, n) for t, n in pairs] == [("a", 1), ("b", 1)]


def test_group_turns_for_scene_matches_counts_variant():
    df = pd.DataFrame([
        {"utterance_ct": "s", "utterance": "a", "start": 0.0, "end": 1.0},
        {"utterance_ct": "s", "utterance": "b", "start": 1.0, "end": 2.0},
        {"utterance_ct": "", "utterance": "c", "start": 2.0, "end": 3.0},
    ])
    assert group_turns_for_scene(df) == [t for t, _ in group_turns_with_counts(df)]


def test_turns_by_scene_with_counts_stable_tie_order():
    # two rows with IDENTICAL start: original row order must decide turn order
    df = pd.DataFrame([
        {"scene_id": 1, "utterance_ct": "first", "utterance": "first", "start": 5.0, "end": 6.0},
        {"scene_id": 1, "utterance_ct": "second", "utterance": "second", "start": 5.0, "end": 7.0},
        {"scene_id": 1, "utterance_ct": "third", "utterance": "third", "start": 4.0, "end": 5.0},
    ])
    pairs = turns_by_scene_with_counts(df)[1]
    assert [t.text for t, _ in pairs] == ["third", "first", "second"]


def test_turns_by_scene_delegates_and_matches():
    df = pd.DataFrame([
        {"scene_id": 2, "utterance_ct": "b", "utterance": "b", "start": 1.0, "end": 2.0},
        {"scene_id": 1, "utterance_ct": "a", "utterance": "a", "start": 0.0, "end": 1.0},
    ])
    plain = turns_by_scene(df)
    counted = turns_by_scene_with_counts(df)
    assert set(plain) == set(counted) == {1, 2}
    for sid in plain:
        assert plain[sid] == [t for t, _ in counted[sid]]
```

- [ ] **Step 1.2: Run the new tests, verify they fail**

Run: `pytest tests/test_topic_shift.py -v -k "counts or stable_tie or delegates"`
Expected: FAIL / ERROR with `ImportError: cannot import name 'group_turns_with_counts'`

- [ ] **Step 1.3: Implement.** In `src/charnet/topic_shift.py`, replace the body of `group_turns_for_scene` with a delegate and add the two new functions. The merge logic moves verbatim into `group_turns_with_counts` (only the tuple bookkeeping is new):

```python
def group_turns_with_counts(scene_rows: pd.DataFrame) -> list[tuple[Turn, int]]:
    """Like ``group_turns_for_scene`` but each turn carries how many sentence
    rows merged into it (the dialogue-embeddings export's ``n_sentences``)."""
    out: list[tuple[Turn, int]] = []
    prev_ct_key: str | None = None
    for _, row in scene_rows.iterrows():
        ct_raw = row.get("utterance_ct")
        ct = _clean_str(ct_raw)
        text = build_text(ct_raw, row.get("utterance"))
        start, end = float(row["start"]), float(row["end"])
        mergeable = ct != "" and ct == prev_ct_key
        if mergeable and out:
            last, n = out[-1]
            out[-1] = (Turn(text=last.text, start=last.start, end=max(last.end, end)), n + 1)  # text shared by all rows with same ct key
        else:
            out.append((Turn(text=text, start=start, end=end), 1))
        prev_ct_key = ct if ct != "" else None
    return out


def group_turns_for_scene(scene_rows: pd.DataFrame) -> list[Turn]:
    """Collapse consecutive rows sharing the same ``utterance_ct`` into turns.

    Rows are assumed already ordered by time within one scene. A turn's text is
    its (shared) ct text, or the first row's ``utterance`` fallback when ct is
    blank; blank-ct rows never merge with neighbours.
    """
    return [t for t, _ in group_turns_with_counts(scene_rows)]
```

And replace `turns_by_scene` (~line 236) with:

```python
def turns_by_scene_with_counts(sentences: pd.DataFrame) -> dict[int, list[tuple[Turn, int]]]:
    """Per-scene (turn, merged-row-count) sequences from an episode's sentence table.

    Rows are STABLE-sorted by ``start`` within each scene (mergesort): for tied
    starts the original table row order decides — this ordering is the export
    contract that fixes turn_id, embedding row order, and the cache key.
    """
    out: dict[int, list[tuple[Turn, int]]] = {}
    for scene_id, grp in sentences.groupby("scene_id", sort=True):
        grp = grp.sort_values("start", kind="mergesort")
        out[int(scene_id)] = group_turns_with_counts(grp)
    return out


def turns_by_scene(sentences: pd.DataFrame) -> dict[int, list[Turn]]:
    """Group an episode's sentence table into per-scene turn sequences."""
    return {sid: [t for t, _ in pairs] for sid, pairs in turns_by_scene_with_counts(sentences).items()}
```

- [ ] **Step 1.4: Run the full suite**

Run: `pytest tests/ -q && ruff check .`
Expected: all tests pass (155 existing + 5 new), ruff clean. The stable-sort change must not break any existing topic-shift test (none pin tie order).

- [ ] **Step 1.5: Commit**

```bash
git add src/charnet/topic_shift.py tests/test_topic_shift.py
git commit -m "Add counts-aware turn grouping + stable tie ordering to topic_shift"
```

---

### Task 2: Exporter script

**Files:**
- Create: `scripts/export_dialogue_embeddings.py`
- Test: `tests/test_export_dialogue_embeddings.py`

- [ ] **Step 2.1: Write the failing tests** — create `tests/test_export_dialogue_embeddings.py`:

```python
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
```

- [ ] **Step 2.2: Run tests, verify they fail**

Run: `pytest tests/test_export_dialogue_embeddings.py -v`
Expected: FAIL at import with `ModuleNotFoundError: No module named 'export_dialogue_embeddings'`

- [ ] **Step 2.3: Implement** — create `scripts/export_dialogue_embeddings.py`:

```python
# scripts/export_dialogue_embeddings.py
"""Export per-turn dialogue embeddings: timing TSV (tracked) + vector NPZ (untracked).

  python scripts/export_dialogue_embeddings.py --episodes ALL

TSV row i describes NPZ ``vecs`` row i. Turn construction contract (stable
ordering, merge semantics) lives in charnet.topic_shift.turns_by_scene_with_counts;
the spec is docs/superpowers/specs/2026-06-12-dialogue-embeddings-export-design.md.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from charnet import topic_shift as ts  # noqa: E402
from charnet.bids_meta import write_data_dictionary, write_dataset_description  # noqa: E402
from charnet.scene_subdivide import expand_episode_spec  # noqa: E402

DEFAULT_SCENES_IN = REPO / "output/annotations/scenes"
DEFAULT_SENTENCES_IN = REPO / "output/annotations/sentences"
DEFAULT_OUT_DIR = REPO / "output/annotations/dialogue_embeddings"
DEFAULT_CACHE_DIR = REPO / "output/intermediate/sentence_embeddings"
MODEL_ID = "all-MiniLM-L6-v2"
COLUMNS = ["turn_id", "scene_id", "start", "end", "n_sentences"]

DATA_DICTIONARY = {
    "turn_id": {"Description": "0-based episode-wide turn index; equals the row index into the companion NPZ's 'vecs' matrix."},
    "scene_id": {"Description": "Fan-transcript scene index the turn belongs to."},
    "start": {"Description": "Turn onset: start of its first sentence row, relative to episode start. Mapping to fMRI run time / TRs is the consumer's responsibility.", "Units": "s"},
    "end": {"Description": "Turn offset: max end across its merged sentence rows.", "Units": "s"},
    "n_sentences": {"Description": "Number of sentence-table rows merged into this turn (consecutive rows sharing one community-transcript utterance)."},
}


def _git_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "describe", "--tags", "--always", "--dirty"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _sentences_path(sentences_in: Path, episode: str) -> Path:
    season = int(episode[1:3])
    return sentences_in / f"s{season}" / f"friends_{episode}_sentence_speaker_table.tsv"


def _cache_status(episode: str, key: str, cache_dir: Path) -> str:
    """'hit' | 're-encoded' (stale/corrupt key on disk) | 'new' (no cache file)."""
    path = Path(cache_dir) / f"s{int(episode[1:3])}" / f"{episode}.npz"
    if not path.exists():
        return "new"
    try:
        cached = np.load(path, allow_pickle=False)
        return "hit" if str(cached["key"]) == key else "re-encoded"
    except Exception:
        return "re-encoded"


def _episode_product(episode, sentences_in, encoder, cache_dir, status_counts=None):
    """Returns (turns_df, vecs, key) or None when the sentence table is missing.

    When *status_counts* is given, the cache state ('hit'/'re-encoded'/'new')
    is tallied BEFORE embed_texts_cached mutates the cache.
    """
    spath = _sentences_path(sentences_in, episode)
    if not spath.exists():
        return None
    sents = pd.read_csv(spath, sep="\t")
    if "scene_id" not in sents.columns:
        raise ValueError(f"{spath}: missing 'scene_id' column")
    n_no_scene = int(sents["scene_id"].isna().sum())
    if n_no_scene:
        print(f"  WARNING {episode}: {n_no_scene} rows with missing scene_id dropped")
    by_scene = ts.turns_by_scene_with_counts(sents)
    rows, texts = [], []
    for sid in by_scene:  # dict preserves groupby(sort=True) scene order
        for turn, n in by_scene[sid]:
            rows.append({"turn_id": len(texts), "scene_id": sid,
                         "start": turn.start, "end": turn.end, "n_sentences": n})
            texts.append(turn.text)
    key = ts._texts_hash(texts, MODEL_ID)
    if status_counts is not None:
        status_counts[_cache_status(episode, key, Path(cache_dir))] += 1
    vecs = ts.embed_texts_cached(episode, texts, encoder, Path(cache_dir), model_id=MODEL_ID)
    return pd.DataFrame(rows, columns=COLUMNS), vecs, key


def _write_atomic_tsv(df: pd.DataFrame, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, dest)


def _write_atomic_npz(vecs: np.ndarray, key: str, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".tmp.npz")  # np.savez appends .npz unless present
    np.savez(tmp, vecs=vecs, key=np.array(key))
    os.replace(tmp, dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="ALL", help="ALL | sN | sN-sM | comma-list")
    ap.add_argument("--scenes-in", default=str(DEFAULT_SCENES_IN), help="root used only to resolve episode specs")
    ap.add_argument("--sentences-in", default=str(DEFAULT_SENTENCES_IN))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    sentences_in = Path(args.sentences_in)
    cache_dir = Path(args.cache_dir)
    episodes = expand_episode_spec(args.episodes, Path(args.scenes_in))
    encoder = ts.minilm_encoder()

    # sidecars first so the output dir is self-describing even on partial runs
    write_data_dictionary(out_dir / "dialogue_turns.json", DATA_DICTIONARY)
    write_dataset_description(
        out_dir.parent / "dataset_description.json",
        name="charnet Friends stimulus annotations",
        version=_git_version(),
        source_datasets=[{"Description": "Courtois NeuroMod Friends fMRI stimulus episodes"}],
    )

    print(f"Exporting dialogue embeddings for {len(episodes)} eps → {out_dir}")
    n_written = n_skipped = 0
    status_counts = {"hit": 0, "re-encoded": 0, "new": 0}
    for ep in episodes:
        product = _episode_product(ep, sentences_in, encoder, cache_dir, status_counts)
        if product is None:
            n_skipped += 1
            continue
        df, vecs, key = product
        season = int(ep[1:3])
        ep_dir = out_dir / f"s{season}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_atomic_tsv(df, ep_dir / f"friends_{ep}_dialogue_turns.tsv")
        _write_atomic_npz(vecs, key, ep_dir / f"friends_{ep}_dialogue_embeddings.npz")
        n_written += 1
        print(f"  {ep}: {len(df)} turns")
    print(f"\nWrote {n_written} episodes ({n_skipped} missing sentence tables)")
    print(f"Cache: {status_counts['hit']} hits, {status_counts['re-encoded']} re-encoded, {status_counts['new']} new")


if __name__ == "__main__":
    main()
```

**Why `status_counts` lives inside `_episode_product`:** cache status must be sampled *before* `embed_texts_cached` runs — it overwrites stale keys, so judging afterwards would always report "hit". The optional parameter keeps one code path; the Step 2.1 tests call `_episode_product` without it and pass unchanged.

- [ ] **Step 2.4: Run the tests**

Run: `pytest tests/test_export_dialogue_embeddings.py -v`
Expected: all 5 PASS

- [ ] **Step 2.5: Full suite + lint**

Run: `pytest tests/ -q && ruff check .`
Expected: green, clean

- [ ] **Step 2.6: Commit**

```bash
git add scripts/export_dialogue_embeddings.py tests/test_export_dialogue_embeddings.py
git commit -m "Add dialogue-embeddings exporter (NPZ + timing TSV per episode)"
```

---

### Task 3: Independent verifier

**Files:**
- Create: `scripts/verify_dialogue_embeddings.py`
- Test: `tests/test_verify_dialogue_embeddings.py`

**Independence rule (from the spec):** the verifier imports **nothing** from `charnet` — its own row-cleaning, merge, ordering, and SHA256 logic. Tests hand-build fixtures rather than generating them through `charnet.topic_shift`.

- [ ] **Step 3.1: Write the failing tests** — create `tests/test_verify_dialogue_embeddings.py`:

```python
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
    assert _run(*_build_fixture(tmp_path)) == 0


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
```

- [ ] **Step 3.2: Run tests, verify they fail**

Run: `pytest tests/test_verify_dialogue_embeddings.py -v`
Expected: FAIL at import with `ModuleNotFoundError: No module named 'verify_dialogue_embeddings'`

- [ ] **Step 3.3: Implement** — create `scripts/verify_dialogue_embeddings.py`:

```python
#!/usr/bin/env python
# scripts/verify_dialogue_embeddings.py
"""Independently verify the dialogue-embeddings export against the tracked
sentence tables.

Imports NOTHING from charnet: own row cleaning, turn merge, stable ordering,
and SHA256 key derivation. Per episode it checks
  1. TSV correctness  — turn_id/scene_id/start/end/n_sentences match an
     independent reconstruction exactly;
  2. row accounting   — every usable sentence row lands in exactly one turn
     (sum of n_sentences == retained rows); NaN-scene_id drops are reported;
  3. key check        — product NPZ key == SHA256(model_id + rebuilt texts);
  4. vector binding   — cache NPZ key matches too, and product vecs are
     array_equal to cache vecs (a permuted matrix with a valid key fails);
  5. sanity           — float32, (n_turns, dim), finite, start <= end.

Product NPZ absent  -> SKIP (TSV checks 1-2 still run).
Cache absent/stale while product NPZ exists -> FAILURE (vectors unvouchable).
Exit 0 all pass; 1 any failure; 2 nothing checkable.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TABLES_ROOT = REPO / "output/annotations/sentences"
DEFAULT_PRODUCT_ROOT = REPO / "output/annotations/dialogue_embeddings"
DEFAULT_CACHE_ROOT = REPO / "output/intermediate/sentence_embeddings"
MODEL_ID = "all-MiniLM-L6-v2"
COLUMNS = ["turn_id", "scene_id", "start", "end", "n_sentences"]


def _clean(val) -> str:
    """NaN-safe strip — own copy, not charnet's (independence)."""
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    return str(val).strip()


def _texts_key(texts: list[str], model_id: str = MODEL_ID) -> str:
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(b"\xff")
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _reconstruct(sents: pd.DataFrame):
    """(rows, texts, n_dropped): independent turn reconstruction.

    Contract: scenes ascending; rows stable-sorted by start within scene
    (ties keep table order); consecutive rows sharing a non-blank
    utterance_ct merge; text = ct, else utterance fallback.
    """
    usable = sents[sents["scene_id"].notna()]
    n_dropped = len(sents) - len(usable)
    rows: list[dict] = []
    texts: list[str] = []
    for sid, grp in usable.groupby("scene_id", sort=True):
        grp = grp.sort_values("start", kind="mergesort")
        prev_ct: str | None = None
        first_turn_of_scene = True
        for _, r in grp.iterrows():
            ct = _clean(r.get("utterance_ct"))
            text = ct if ct else _clean(r.get("utterance"))
            start, end = float(r["start"]), float(r["end"])
            if ct != "" and ct == prev_ct and not first_turn_of_scene:
                rows[-1]["end"] = max(rows[-1]["end"], end)
                rows[-1]["n_sentences"] += 1
            else:
                rows.append({"turn_id": len(rows), "scene_id": int(sid),
                             "start": start, "end": end, "n_sentences": 1})
                texts.append(text)
                first_turn_of_scene = False
            prev_ct = ct if ct != "" else None
    return rows, texts, n_dropped


def _episode_id(table_path: Path) -> str:
    # friends_s01e01a_sentence_speaker_table.tsv -> s01e01a
    return table_path.name.removeprefix("friends_").split("_")[0]


def check_episode(ep: str, table_path: Path, product_root: Path, cache_root: Path,
                  expected_dim: int) -> tuple[list[str], bool]:
    """Returns (mismatches, skipped). skipped=True when the product NPZ is absent."""
    errs: list[str] = []
    season = f"s{int(ep[1:3])}"
    tsv_path = product_root / season / f"friends_{ep}_dialogue_turns.tsv"
    npz_path = product_root / season / f"friends_{ep}_dialogue_embeddings.npz"
    cache_path = cache_root / season / f"{ep}.npz"

    sents = pd.read_csv(table_path, sep="\t")
    rows, texts, n_dropped = _reconstruct(sents)
    if n_dropped:
        print(f"  note {ep}: {n_dropped} rows without scene_id (excluded from turns)")

    if not tsv_path.exists():
        return [f"{ep}: product TSV missing"], False
    got = pd.read_csv(tsv_path, sep="\t")
    if list(got.columns) != COLUMNS:
        errs.append(f"{ep}: TSV columns {list(got.columns)} != {COLUMNS}")
        return errs, False
    exp = pd.DataFrame(rows, columns=COLUMNS)
    if len(got) != len(exp):
        errs.append(f"{ep}: TSV has {len(got)} rows, reconstruction has {len(exp)}")
    else:
        for col in COLUMNS:
            if col in ("start", "end"):
                bad = (got[col] - exp[col]).abs() > 1e-9
            else:
                bad = got[col].astype(int) != exp[col].astype(int)
            if bad.any():
                i = int(bad.idxmax())
                errs.append(f"{ep}: {col} mismatch at turn {i}: "
                            f"tsv={got[col][i]} expected={exp[col][i]} "
                            f"({int(bad.sum())} rows differ)")
        if int(exp["n_sentences"].sum()) != len(sents) - n_dropped:
            errs.append(f"{ep}: row accounting broken: n_sentences sums to "
                        f"{int(exp['n_sentences'].sum())}, retained rows {len(sents) - n_dropped}")
        if (got["start"] > got["end"]).any():
            errs.append(f"{ep}: TSV has turns with start > end")

    if not npz_path.exists():
        return errs, True  # skip vector checks; TSV findings still count

    key = _texts_key(texts)
    prod = np.load(npz_path, allow_pickle=False)
    if str(prod["key"]) != key:
        errs.append(f"{ep}: product NPZ key != recomputed text hash")
    vecs = prod["vecs"]
    if vecs.dtype != np.float32:
        errs.append(f"{ep}: vecs dtype {vecs.dtype} != float32")
    if vecs.shape != (len(texts), expected_dim):
        errs.append(f"{ep}: vecs shape {vecs.shape} != ({len(texts)}, {expected_dim})")
    if not np.isfinite(vecs).all():
        errs.append(f"{ep}: vecs contain non-finite values")

    # vector binding: texts -> key -> cache vecs -> product vecs
    if not cache_path.exists():
        errs.append(f"{ep}: cache NPZ missing — vectors cannot be vouched for "
                    f"(regenerate via scripts/export_dialogue_embeddings.py)")
    else:
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["key"]) != key:
            errs.append(f"{ep}: cache NPZ key != recomputed text hash (stale cache)")
        elif not np.array_equal(vecs, cached["vecs"]):
            errs.append(f"{ep}: product vecs != cache vecs (binding broken)")
    return errs, False


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-root", default=str(DEFAULT_TABLES_ROOT))
    ap.add_argument("--product-root", default=str(DEFAULT_PRODUCT_ROOT))
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--expected-dim", type=int, default=384)
    args = ap.parse_args(argv)

    tables = sorted(Path(args.tables_root).glob("*/*_sentence_speaker_table.tsv"))
    if not tables:
        print(f"No sentence tables under {args.tables_root}: nothing to check")
        return 2

    all_errs: list[str] = []
    n_checked = n_skipped = 0
    for tpath in tables:
        ep = _episode_id(tpath)
        errs, skipped = check_episode(ep, tpath, Path(args.product_root),
                                      Path(args.cache_root), args.expected_dim)
        all_errs.extend(errs)
        if skipped and not errs:
            n_skipped += 1
            print(f"  skip {ep}: product NPZ absent (TSV checks only)")
        else:
            n_checked += 1

    print(f"\nChecked {n_checked} episodes ({n_skipped} NPZ-absent skips)")
    if all_errs:
        for e in all_errs[:50]:
            print(f"  FAIL {e}")
        if len(all_errs) > 50:
            print(f"  ... and {len(all_errs) - 50} more")
        return 1
    if n_checked == 0 and n_skipped == 0:
        return 2
    print("All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
```

**Implementation notes (read before coding):**
- `first_turn_of_scene` guards against a `prev_ct` carried over from the
  previous scene merging across a scene boundary — `prev_ct` is reset per
  scene anyway (it's re-initialized in the loop), so the guard's real job is
  to never merge into `rows[-1]` from a *different* scene. Keep both: reset
  `prev_ct = None` per scene AND the flag.
- A TSV missing while its NPZ exists is a failure (`product TSV missing`),
  not a skip — the TSV is the tracked half.
- Skip semantics: `skipped and not errs` — an episode with TSV errors AND a
  missing NPZ counts as checked-and-failed, not skipped.
- The spec's optional `--re-embed N` deep check is **deferred** (YAGNI until
  someone asks for it); the spec documents it as optional. Do not implement.

- [ ] **Step 3.4: Run the tests**

Run: `pytest tests/test_verify_dialogue_embeddings.py -v`
Expected: all 10 PASS

- [ ] **Step 3.5: Full suite + lint**

Run: `pytest tests/ -q && ruff check .`
Expected: green, clean

- [ ] **Step 3.6: Commit**

```bash
git add scripts/verify_dialogue_embeddings.py tests/test_verify_dialogue_embeddings.py
git commit -m "Add independent dialogue-embeddings verifier with vector binding"
```

---

### Task 4: gitignore + catalog registration

**Files:**
- Modify: `.gitignore` (after the network_metrics block, ~line "!output/annotations/network_metrics/**")
- Modify: `docs/data_products_catalog.md`

- [ ] **Step 4.1: Add gitignore rules.** Insert directly after the `!output/annotations/network_metrics/**` line:

```gitignore
# Track dialogue-embeddings timing TSVs + sidecars; the NPZ vector files stay
# untracked (~160 MB) — regenerable via scripts/export_dialogue_embeddings.py
# and bound to the tracked tables by scripts/verify_dialogue_embeddings.py.
!output/annotations/dialogue_embeddings/
!output/annotations/dialogue_embeddings/**
output/annotations/dialogue_embeddings/**/*.npz
```

(Order matters: re-include the directory, then its contents, then re-exclude the NPZs. Per the repo gotcha, verify with `git add -n`, **not** `git check-ignore -v`.)

- [ ] **Step 4.2: Verify the rules with a dry run**

```bash
mkdir -p output/annotations/dialogue_embeddings/s1
touch output/annotations/dialogue_embeddings/s1/friends_s01e01a_dialogue_turns.tsv \
      output/annotations/dialogue_embeddings/s1/friends_s01e01a_dialogue_embeddings.npz \
      output/annotations/dialogue_embeddings/dialogue_turns.json
git add -n output/annotations/dialogue_embeddings/
```

Expected: the dry run lists the `.tsv` and `.json`, NOT the `.npz`. Then remove the probe files:

```bash
rm output/annotations/dialogue_embeddings/s1/friends_s01e01a_dialogue_turns.tsv \
   output/annotations/dialogue_embeddings/s1/friends_s01e01a_dialogue_embeddings.npz \
   output/annotations/dialogue_embeddings/dialogue_turns.json
rmdir output/annotations/dialogue_embeddings/s1 output/annotations/dialogue_embeddings
```

- [ ] **Step 4.3: Update the catalog.** In `docs/data_products_catalog.md`:

(a) Row ~line 30 (`intermediate/sentence_embeddings/sN/<ep>.npz`): change the final status cell from `**available-not-yet-exported** (needs turn timestamps to be a product)` to `**exported** → \`annotations/dialogue_embeddings/\` (feature #3)`.

(b) In the numbered feature list (~line 98, "3. **Dialogue embeddings** *(available, needs export — heaviest)*"), mark it shipped and describe the product. Replace the entry's status phrasing with:

```markdown
3. **Dialogue embeddings** *(shipped — feature #3)* — per-episode pair under
   `output/annotations/dialogue_embeddings/sN/`:
   `friends_<ep>_dialogue_turns.tsv` (tracked; `turn_id`, `scene_id`, `start`,
   `end`, `n_sentences` — row i indexes NPZ row i) +
   `friends_<ep>_dialogue_embeddings.npz` (untracked; `vecs` (n_turns, 384)
   float32 MiniLM `all-MiniLM-L6-v2`, `key` = SHA256(model_id + texts)).
   Regenerate: `python scripts/export_dialogue_embeddings.py --episodes ALL`.
   Verify: `python scripts/verify_dialogue_embeddings.py` (independent turn
   reconstruction + key check + cache↔product vector binding; exit 0 on full
   pass). Data dictionary: `dialogue_turns.json`. Spec:
   `docs/superpowers/specs/2026-06-12-dialogue-embeddings-export-design.md`.
```

Adapt the surrounding prose minimally so the section still reads correctly (the catalog's exact current wording may differ — preserve its table/list structure; the content above is what must be conveyed).

- [ ] **Step 4.4: Commit**

```bash
git add .gitignore docs/data_products_catalog.md
git commit -m "Register dialogue-embeddings product: gitignore tracking rules + catalog entry"
```

---

### Task 5: Full export run, verification, ship the TSVs

This task runs the real encoder. Precondition: `sentence_transformers` importable in the project env (it is — the topic-shift trace used it). The first full run **re-encodes all 341 episodes** (the on-disk s1–s6 caches carry pre-model_id keys; expected, one-time). CPU-only, batch 64; expect tens of minutes — run it in the background and poll.

- [ ] **Step 5.1: Run the export**

```bash
python scripts/export_dialogue_embeddings.py --episodes ALL 2>&1 | tail -20
```

Expected final lines: `Wrote 341 episodes (0 missing sentence tables)` and a cache line of the shape `Cache: 0 hits, 292 re-encoded, 49 new` (numbers may differ slightly if any cache file was already refreshed; what matters is hits+re-encoded+new == 341).

- [ ] **Step 5.2: Spot-check one episode**

```bash
head -5 output/annotations/dialogue_embeddings/s1/friends_s01e01a_dialogue_turns.tsv
python -c "
import numpy as np, pandas as pd
df = pd.read_csv('output/annotations/dialogue_embeddings/s1/friends_s01e01a_dialogue_turns.tsv', sep='\t')
z = np.load('output/annotations/dialogue_embeddings/s1/friends_s01e01a_dialogue_embeddings.npz', allow_pickle=False)
print(len(df), z['vecs'].shape, z['vecs'].dtype)
assert len(df) == z['vecs'].shape[0]
"
```

Expected: row count equals `vecs` rows; dtype float32; dim 384. (For reference, s01e01a previously cached 315 turns — same ballpark expected, exact count comes from the current tables.)

- [ ] **Step 5.3: Run the verifier**

```bash
python scripts/verify_dialogue_embeddings.py; echo "exit: $?"
```

Expected: `Checked 341 episodes (0 NPZ-absent skips)`, `All checks passed`, `exit: 0`.

- [ ] **Step 5.4: Idempotence check** — re-run the exporter on one season and confirm all cache hits now:

```bash
python scripts/export_dialogue_embeddings.py --episodes s1 2>&1 | tail -2
```

Expected: `Cache: 48 hits, 0 re-encoded, 0 new`.

- [ ] **Step 5.5: Stage and inspect what ships**

```bash
git add output/annotations/dialogue_embeddings/
git status --short -- output/annotations/dialogue_embeddings/ | head
git status --short -- output/annotations/dialogue_embeddings/ | grep -c "\.tsv"
git status --short -- output/annotations/dialogue_embeddings/ | grep -c "\.npz" || true
```

Expected: 341 TSVs + `dialogue_turns.json` staged; `dataset_description.json` at `output/annotations/` staged or already-tracked-unchanged; **zero** NPZs staged.

- [ ] **Step 5.6: Commit the product**

```bash
git commit -m "Ship dialogue-turn timing TSVs (341 episodes, 7 seasons)

Companion NPZ vectors are untracked: regenerable via
scripts/export_dialogue_embeddings.py, independently verified against the
tracked sentence tables by scripts/verify_dialogue_embeddings.py (341/341)."
```

- [ ] **Step 5.7: Final full-suite check**

```bash
pytest tests/ -q && ruff check .
```

Expected: green, clean.

---

## Self-review notes (already applied)

- **Spec coverage:** turn contract + counts API → Task 1; exporter, atomic writes, cache-status logging, sidecars → Task 2; verifier checks 1–5, skip/fail semantics, exit codes → Task 3; gitignore split + catalog → Task 4; full 341 run, re-encode expectation, verification, shipping → Task 5. `--re-embed N` is deliberately deferred (documented in Task 3 notes; spec lists it as optional).
- **Type consistency:** `turns_by_scene_with_counts -> dict[int, list[tuple[Turn, int]]]` is used identically in Task 1 (definition), Task 2 (exporter), and the test files. Verifier shares no types (independence).
- **Known divergence from spec text:** spec names `group_turns_with_counts(scene_rows) -> list[tuple[Turn, int]]` — matched exactly. Spec's verifier exit-code semantics (skip vs fail) — matched in Task 3 tests.
