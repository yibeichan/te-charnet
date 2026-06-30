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
  6. deep check     — with --re-embed N, re-encodes N sampled passing
     episodes with the real model (lazy import) and compares to product
     vecs within atol=1e-5; --seed makes the sample reproducible.

Corrupt or member-incomplete NPZ files (product or cache) are clean
per-episode FAILures (exit 1), never tracebacks.

Product NPZ absent  -> SKIP (TSV checks 1-2 still run).
Cache absent/stale while product NPZ exists -> FAILURE (vectors unvouchable).
Exit 0 all pass; 1 any failure; 2 nothing checkable.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
import zipfile
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


def _load_npz(path: Path, members: tuple[str, ...] = ("key", "vecs")):
    """(dict, None) on success, (None, reason) on unreadable/incomplete NPZ."""
    try:
        with np.load(path, allow_pickle=False) as npz:
            return {m: npz[m] for m in members}, None
    except KeyError as e:
        return None, f"missing member {e}"
    except (zipfile.BadZipFile, OSError, EOFError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"


def _build_real_encoder():
    """Lazy MiniLM encoder matching the export's settings exactly.

    Returns None when sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    model = SentenceTransformer(MODEL_ID, device="cpu")

    def encode(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, model.get_sentence_embedding_dimension()),
                            dtype=np.float32)
        return np.asarray(
            model.encode(texts, batch_size=64, show_progress_bar=False,
                         normalize_embeddings=False),
            dtype=np.float32,
        )

    return encode


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
                  expected_dim: int) -> tuple[list[str], bool, dict | None]:
    """Returns (mismatches, skipped, deep). skipped=True when the product NPZ is
    absent; deep carries texts+vecs for re-embedding when the episode fully passed."""
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
        return [f"{ep}: product TSV missing"], False, None
    got = pd.read_csv(tsv_path, sep="\t")
    if list(got.columns) != COLUMNS:
        errs.append(f"{ep}: TSV columns {list(got.columns)} != {COLUMNS}")
        return errs, False, None
    exp = pd.DataFrame(rows, columns=COLUMNS)
    if got[COLUMNS].isna().any().any():
        # corrupted export: blank/NaN cells would crash astype(int) below
        errs.append(f"{ep}: TSV contains blank/NaN cells")
        return errs, False, None
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
        return errs, True, None  # skip vector checks; TSV findings still count

    key = _texts_key(texts)
    prod, load_err = _load_npz(npz_path)
    if load_err:
        errs.append(f"{ep}: product NPZ unreadable ({load_err})")
        return errs, False, None
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
        cached, load_err = _load_npz(cache_path)
        if load_err:
            errs.append(f"{ep}: cache NPZ unreadable ({load_err}) — vectors "
                        f"cannot be vouched for (regenerate via "
                        f"scripts/export_dialogue_embeddings.py)")
        elif str(cached["key"]) != key:
            errs.append(f"{ep}: cache NPZ key != recomputed text hash (stale cache)")
        elif not np.array_equal(vecs, cached["vecs"]):
            errs.append(f"{ep}: product vecs != cache vecs (binding broken)")
    deep = {"texts": texts, "vecs": vecs} if not errs else None
    return errs, False, deep


def run(argv: list[str] | None = None, encoder_factory=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-root", default=str(DEFAULT_TABLES_ROOT))
    ap.add_argument("--product-root", default=str(DEFAULT_PRODUCT_ROOT))
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--expected-dim", type=int, default=384)
    ap.add_argument("--re-embed", type=int, default=0, metavar="N",
                    help="re-encode N sampled passing episodes with the real "
                         "model and compare to product vecs")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for --re-embed episode sampling")
    args = ap.parse_args(argv)

    tables = sorted(Path(args.tables_root).glob("*/*_sentence_speaker_table.tsv"))
    if not tables:
        print(f"No sentence tables under {args.tables_root}: nothing to check")
        return 2

    all_errs: list[str] = []
    pool: list[tuple[str, dict]] = []
    n_checked = n_skipped = 0
    for tpath in tables:
        ep = _episode_id(tpath)
        errs, skipped, deep = check_episode(ep, tpath, Path(args.product_root),
                                            Path(args.cache_root), args.expected_dim)
        all_errs.extend(errs)
        if deep is not None:
            pool.append((ep, deep))
        if skipped and not errs:
            n_skipped += 1
            print(f"  skip {ep}: product NPZ absent (TSV checks only)")
        else:
            n_checked += 1

    if args.re_embed > 0 and pool:
        rng = random.Random(args.seed)
        chosen = rng.sample(sorted(pool), min(args.re_embed, len(pool)))
        print(f"Re-embed deep check: {[ep for ep, _ in chosen]}"
              + (f" (seed={args.seed})" if args.seed is not None else ""))
        encoder = encoder_factory() if encoder_factory else _build_real_encoder()
        if encoder is None:
            print("  FAIL --re-embed requires sentence-transformers (not installed)")
            all_errs.append("--re-embed: sentence-transformers not installed")
        else:
            for ep, deep in chosen:
                fresh = encoder(deep["texts"])
                if fresh.shape != deep["vecs"].shape:
                    all_errs.append(f"{ep}: re-embedded shape {fresh.shape} != "
                                    f"product {deep['vecs'].shape}")
                elif not np.allclose(deep["vecs"], fresh, atol=1e-5, rtol=0):
                    diff = float(np.abs(deep["vecs"] - fresh).max())
                    all_errs.append(f"{ep}: re-embedded vecs differ from product "
                                    f"(max abs diff {diff:.3g})")
            print(f"  re-embedded {len(chosen)} episode(s)")

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
