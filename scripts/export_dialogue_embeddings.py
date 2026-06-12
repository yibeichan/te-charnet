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
