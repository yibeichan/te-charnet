#!/usr/bin/env python
# scripts/verify_topic_trace.py
"""Independently verify the topic-shift trace export against the tracked
sentence tables and the shared sentence-embedding cache.

Imports NOTHING from charnet: own row cleaning, per-scene turn merge, stable
ordering, SHA256 key derivation, and a from-scratch block-distance recompute
(mean-pooled w-turn blocks, cosine distance). Agreement is real evidence the
exported numbers are correct, not just non-NaN.

Per episode it checks the two LOAD-BEARING columns:
  1. onset          — turn timing, recomputed from the sentence table alone
                      (no model): onset[i] == end of the turn before gap i.
  2. block_distance — cosine distance between mean-pooled w-turn blocks either
                      side of each gap, recomputed from the cached embeddings
                      using the ``w`` recorded in the TSV, within --tol.
Plus structure (row count, per-scene gap count for scenes with >= 2w+1 turns,
scene_id membership), provenance (w/tau_depth/min_spacing constant), and a
cache key check (cache NPZ key == SHA256(model_id + rebuilt turn texts)).

NOT CHECKED BY DESIGN: ``depth`` and ``is_peak``. The topic-shift detector is a
documented negative result (docs/scene_segmentation_evaluation.md, Prototype
#2); those columns are an audit trail, not validated boundaries, so this
verifier does not reimplement the detector to vouch for them.

Cache absent/stale/corrupt while a product TSV exists -> FAILURE
(block_distance unvouchable); onset/structure are still checked. Corrupt NPZ is
a clean per-episode FAILure, never a traceback.

Exit 0 all pass; 1 any failure; 2 nothing checkable.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TABLES_ROOT = REPO / "output/annotations/sentences"
DEFAULT_PRODUCT_ROOT = REPO / "output/annotations/topic_shift"
DEFAULT_CACHE_ROOT = REPO / "output/intermediate/sentence_embeddings"
MODEL_ID = "all-MiniLM-L6-v2"
COLUMNS = ["scene_id", "onset", "block_distance", "depth", "is_peak",
           "w", "tau_depth", "min_spacing"]


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


def _load_npz(path: Path, members: tuple[str, ...] = ("key", "vecs")):
    """(dict, None) on success, (None, reason) on unreadable/incomplete NPZ."""
    try:
        with np.load(path, allow_pickle=False) as npz:
            return {m: npz[m] for m in members}, None
    except KeyError as e:
        return None, f"missing member {e}"
    except (zipfile.BadZipFile, OSError, EOFError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"


def _reconstruct(sents: pd.DataFrame):
    """Per-scene turn reconstruction, independent of charnet.

    Contract (mirrors charnet.topic_shift.turns_by_scene_with_counts): scenes
    ascending; rows stable-sorted by start within scene (ties keep table
    order); consecutive rows sharing a non-blank utterance_ct merge; text = ct,
    else utterance fallback.

    Returns (scene_ends, flat_texts, scene_slices):
      scene_ends[sid]   -> list of turn end-times for that scene, in order
      flat_texts        -> episode-wide turn texts in scene-then-time order
      scene_slices[sid] -> (lo, hi) index range into flat_texts / cache vecs
    """
    usable = sents[sents["scene_id"].notna()]
    scene_ends: dict[int, list[float]] = {}
    scene_slices: dict[int, tuple[int, int]] = {}
    flat_texts: list[str] = []
    for sid, grp in usable.groupby("scene_id", sort=True):
        sid = int(sid)
        grp = grp.sort_values("start", kind="mergesort")
        lo = len(flat_texts)
        ends: list[float] = []
        prev_ct: str | None = None
        first = True
        for _, r in grp.iterrows():
            ct = _clean(r.get("utterance_ct"))
            text = ct if ct else _clean(r.get("utterance"))
            end = float(r["end"])
            if ct != "" and ct == prev_ct and not first:
                ends[-1] = max(ends[-1], end)
            else:
                ends.append(end)
                flat_texts.append(text)
                first = False
            prev_ct = ct if ct != "" else None
        scene_ends[sid] = ends
        scene_slices[sid] = (lo, len(flat_texts))
    return scene_ends, flat_texts, scene_slices


def _cos_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, 1.0 - float(np.dot(a, b)) / (na * nb))


def _block_distance(vecs: np.ndarray, w: int) -> list[float]:
    """Cosine distance between mean-pooled w-turn blocks on either side of each
    gap; entry i scores the gap between turn i and i+1. Independent recompute."""
    n = len(vecs)
    out: list[float] = []
    for i in range(n - 1):
        left = vecs[max(0, i - w + 1):i + 1].mean(axis=0)
        right = vecs[i + 1:min(n, i + 1 + w)].mean(axis=0)
        out.append(_cos_distance(left, right))
    return out


def _expected_rows(scene_ends, scene_slices, vecs, w):
    """(scene_id, onset, block_distance) per gap, in scene-then-time order.

    Scenes with fewer than 2w+1 turns contribute no rows (mirrors the export).
    """
    rows = []
    for sid in sorted(scene_ends):
        ends = scene_ends[sid]
        if len(ends) < 2 * w + 1:
            continue
        lo, hi = scene_slices[sid]
        dists = _block_distance(vecs[lo:hi], w)
        for i in range(len(ends) - 1):
            rows.append((sid, ends[i], dists[i]))
    return rows


def _episode_id(table_path: Path) -> str:
    return table_path.name.removeprefix("friends_").split("_")[0]


def check_episode(ep: str, table_path: Path, product_root: Path,
                  cache_root: Path, tol: float) -> tuple[list[str], bool]:
    """(mismatches, skipped). skipped=True when the product TSV is absent."""
    errs: list[str] = []
    season = f"s{int(ep[1:3])}"
    tsv_path = product_root / season / f"friends_{ep}_topic_trace.tsv"
    cache_path = cache_root / season / f"{ep}.npz"

    if not tsv_path.exists():
        return [], True

    got = pd.read_csv(tsv_path, sep="\t")
    if list(got.columns) != COLUMNS:
        return [f"{ep}: TSV columns {list(got.columns)} != {COLUMNS}"], False

    # provenance columns must be constant within the file
    for col in ("w", "tau_depth", "min_spacing"):
        if got[col].nunique() > 1:
            errs.append(f"{ep}: {col} not constant across rows: {sorted(got[col].unique())}")
    if errs:
        return errs, False
    w = int(got["w"].iloc[0]) if len(got) else 1

    sents = pd.read_csv(table_path, sep="\t")
    scene_ends, flat_texts, scene_slices = _reconstruct(sents)

    # cache: block_distance is unvouchable without the embeddings it was built from
    vecs = None
    if not cache_path.exists():
        errs.append(f"{ep}: cache NPZ missing — block_distance cannot be vouched for "
                    f"(regenerate via scripts/export_topic_trace.py)")
    else:
        cached, load_err = _load_npz(cache_path)
        if load_err:
            errs.append(f"{ep}: cache NPZ unreadable ({load_err}) — block_distance "
                        f"cannot be vouched for")
        elif str(cached["key"]) != _texts_key(flat_texts):
            errs.append(f"{ep}: cache NPZ key != recomputed text hash (stale cache)")
        elif cached["vecs"].shape[0] != len(flat_texts):
            errs.append(f"{ep}: cache vecs rows {cached['vecs'].shape[0]} != "
                        f"{len(flat_texts)} reconstructed turns")
        else:
            vecs = cached["vecs"]

    # onset + structure need no embeddings; block_distance needs vecs.
    if vecs is not None:
        exp = _expected_rows(scene_ends, scene_slices, vecs, w)
        if len(got) != len(exp):
            errs.append(f"{ep}: TSV has {len(got)} rows, reconstruction has {len(exp)}")
        else:
            for i, (sid, onset, dist) in enumerate(exp):
                if int(got["scene_id"].iloc[i]) != sid:
                    errs.append(f"{ep}: scene_id mismatch at row {i}: "
                                f"tsv={got['scene_id'].iloc[i]} expected={sid}")
                if abs(float(got["onset"].iloc[i]) - onset) > 1e-9:
                    errs.append(f"{ep}: onset mismatch at row {i}: "
                                f"tsv={got['onset'].iloc[i]} expected={onset}")
                if abs(float(got["block_distance"].iloc[i]) - dist) > tol + tol * abs(dist):
                    errs.append(f"{ep}: block_distance mismatch at row {i}: "
                                f"tsv={got['block_distance'].iloc[i]} expected={dist}")
    else:
        # cache unusable: still verify onset + row structure independently
        exp_onsets = [(sid, onset) for sid, onset, _ in
                      _expected_rows(scene_ends, scene_slices,
                                     np.zeros((len(flat_texts), 1)), w)]
        if len(got) != len(exp_onsets):
            errs.append(f"{ep}: TSV has {len(got)} rows, reconstruction has {len(exp_onsets)}")
        else:
            for i, (sid, onset) in enumerate(exp_onsets):
                if int(got["scene_id"].iloc[i]) != sid:
                    errs.append(f"{ep}: scene_id mismatch at row {i}")
                if abs(float(got["onset"].iloc[i]) - onset) > 1e-9:
                    errs.append(f"{ep}: onset mismatch at row {i}: "
                                f"tsv={got['onset'].iloc[i]} expected={onset}")
    return errs, False


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables-root", default=str(DEFAULT_TABLES_ROOT))
    ap.add_argument("--product-root", default=str(DEFAULT_PRODUCT_ROOT))
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--tol", type=float, default=1e-6)
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
                                      Path(args.cache_root), args.tol)
        all_errs.extend(errs)
        if skipped:
            n_skipped += 1
            print(f"  skip {ep}: product TSV absent")
        else:
            n_checked += 1

    print(f"\nChecked {n_checked} episodes ({n_skipped} TSV-absent skips)")
    if all_errs:
        for e in all_errs[:50]:
            print(f"  FAIL {e}")
        if len(all_errs) > 50:
            print(f"  ... and {len(all_errs) - 50} more")
        return 1
    if n_checked == 0:
        return 2
    print("All checks passed (onset + block_distance independently recomputed; "
          "depth/is_peak are the negative-result audit trail, not checked).")
    return 0


if __name__ == "__main__":
    sys.exit(run())
