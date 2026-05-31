# src/charnet/topic_shift.py
"""Topic-shift sub-boundary proposer (improvement direction #2).

Pipeline per scene: group sentence rows into community-transcript "turns",
embed each turn (MiniLM, cached), score each inter-turn gap by the cosine
distance between mean-pooled blocks of W turns on either side, and propose
boundaries at local-maximum gaps whose TextTiling depth exceeds a threshold.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Turn:
    text: str
    start: float
    end: float


def _clean_str(val) -> str:
    """NaN-safe strip: returns '' for None/NaN, else str(val).strip().

    Per the repo gotcha, never let NaN become the string 'nan'.
    """
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    return str(val).strip()


def build_text(utterance_ct, utterance) -> str:
    """Best-available turn text: community transcript, else Speech2Text.

    NaN-safe per the repo gotcha — never stringify NaN to "nan".
    """
    for val in (utterance_ct, utterance):
        s = _clean_str(val)
        if s:
            return s
    return ""


def group_turns_for_scene(scene_rows: pd.DataFrame) -> list[Turn]:
    """Collapse consecutive rows sharing the same ``utterance_ct`` into turns.

    Rows are assumed already ordered by time within one scene. A turn's text is
    its (shared) ct text, or the first row's ``utterance`` fallback when ct is
    blank; blank-ct rows never merge with neighbours.
    """
    turns: list[Turn] = []
    prev_ct_key: str | None = None
    for _, row in scene_rows.iterrows():
        ct_raw = row.get("utterance_ct")
        ct = _clean_str(ct_raw)
        text = build_text(ct_raw, row.get("utterance"))
        start, end = float(row["start"]), float(row["end"])
        mergeable = ct != "" and ct == prev_ct_key
        if mergeable and turns:
            last = turns[-1]
            turns[-1] = Turn(text=last.text, start=last.start, end=max(last.end, end))  # text shared by all rows with same ct key
        else:
            turns.append(Turn(text=text, start=start, end=end))
        prev_ct_key = ct if ct != "" else None
    return turns


def _mean_block(vecs: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Mean of rows [lo, hi); caller guarantees lo < hi within bounds."""
    return vecs[lo:hi].mean(axis=0)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(max(0.0, 1.0 - np.dot(a, b) / (na * nb)))


def block_distance_trace(vecs: np.ndarray, w: int) -> np.ndarray:
    """Cosine distance between W-turn blocks on either side of each gap.

    Returns an array of length ``len(vecs) - 1``; entry ``i`` scores the gap
    between turn ``i`` and turn ``i+1``. Blocks are truncated to available
    turns at the sequence edges (no padding).
    """
    n = len(vecs)
    trace = np.zeros(max(0, n - 1))
    for i in range(n - 1):
        left = _mean_block(vecs, max(0, i - w + 1), i + 1)
        right = _mean_block(vecs, i + 1, min(n, i + 1 + w))
        trace[i] = _cosine_distance(left, right)
    return trace


def peak_depths(trace: np.ndarray) -> list[tuple[int, float]]:
    """Local maxima of *trace* with their TextTiling depth, deepest first.

    Depth of a peak = (peak − nearest lower-or-equal valley on the left)
    + (peak − nearest lower-or-equal valley on the right). Endpoints use the
    available side only.
    """
    n = len(trace)
    out: list[tuple[int, float]] = []
    for i in range(n):
        left_ok = i == 0 or trace[i] >= trace[i - 1]
        right_ok = i == n - 1 or trace[i] >= trace[i + 1]
        if not (left_ok and right_ok):
            continue
        # at an endpoint the walk can't move, so lv stays == trace[i] and that side contributes 0 depth (one-sided)
        # walk left to the local valley
        lv = trace[i]
        j = i
        while j > 0 and trace[j - 1] <= trace[j]:
            j -= 1
            lv = min(lv, trace[j])
        rv = trace[i]
        k = i
        while k < n - 1 and trace[k + 1] <= trace[k]:
            k += 1
            rv = min(rv, trace[k])
        depth = (trace[i] - lv) + (trace[i] - rv)
        if depth > 0:
            out.append((i, depth))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _accepted_peak_indices(
    trace: np.ndarray,
    turns: list[Turn],
    *,
    tau_depth: float,
    min_spacing: float,
) -> list[int]:
    """Gap indices the detector accepts as boundaries, given a precomputed trace.

    Greedy by descending depth: keep a local-maximum gap if its depth ≥
    *tau_depth* and it is ≥ *min_spacing* seconds from every already-accepted
    gap. Returns the accepted gap indices, sorted ascending.
    """
    accepted: list[int] = []
    for gap_i, depth in peak_depths(trace):
        if depth < tau_depth:
            continue
        t = turns[gap_i].end
        if any(abs(t - turns[j].end) < min_spacing for j in accepted):
            continue
        accepted.append(gap_i)
    return sorted(accepted)


def propose_topic_boundaries(
    turns: list[Turn],
    vecs: np.ndarray,
    *,
    w: int,
    tau_depth: float,
    min_spacing: float,
) -> list[float]:
    """Interior boundary times for one scene's turn sequence.

    A boundary is a local-maximum gap with depth ≥ *tau_depth*; accepted
    greedily by descending depth subject to *min_spacing* from previously
    accepted boundaries. Each boundary is placed at the end time of the turn
    before the gap (a sentence end).
    """
    if len(turns) < 2 * w + 1 or len(vecs) != len(turns):  # require at least one full w-width block available on each side of some interior gap
        return []
    trace = block_distance_trace(vecs, w)
    idx = _accepted_peak_indices(trace, turns, tau_depth=tau_depth, min_spacing=min_spacing)
    return sorted(turns[i].end for i in idx)


Encoder = Callable[[list[str]], np.ndarray]


def _texts_hash(texts: list[str], model_id: str = "") -> str:
    # model_id="" → no prefix (caller supplies it); embed_texts_cached always passes one.
    h = hashlib.sha256()
    if model_id:
        h.update(model_id.encode("utf-8"))
        h.update(b"\xff")  # separator distinct from \x00 used between texts
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def embed_texts_cached(
    episode: str,
    texts: list[str],
    encoder: Encoder,
    cache_dir: Path,
    *,
    model_id: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Encode *texts*, caching by (episode, model_id, texts-hash) under *cache_dir*.

    Cache layout: ``<cache_dir>/<season>/<episode>.npz`` storing the vectors
    plus the text hash; a hash mismatch (texts changed or model changed) forces
    a re-encode. *model_id* is folded into the hash so swapping models never
    silently returns stale vectors. (Caches written before model_id was added
    to the key mismatch on first run and re-encode once — expected, one-time.)
    """
    cache_dir = Path(cache_dir)
    season = f"s{int(episode[1:3])}"
    path = cache_dir / season / f"{episode}.npz"
    key = _texts_hash(texts, model_id)
    if path.exists():
        try:
            cached = np.load(path, allow_pickle=False)
            if str(cached["key"]) == key and cached["vecs"].shape[0] == len(texts):
                return cached["vecs"]
        except Exception:
            pass  # corrupted/partial write — fall through to re-encode and overwrite
    vecs = np.asarray(encoder(texts), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vecs=vecs, key=np.array(key))
    return vecs


def intersect_within(char_times, topic_times, *, eps: float) -> list[float]:
    """Topic times t with some char time c such that |t - c| <= eps (closed); placed at the topic time. Result sorted."""
    out = [t for t in topic_times if any(abs(t - c) <= eps for c in char_times)]
    return sorted(out)


def turns_by_scene(sentences: pd.DataFrame) -> dict[int, list[Turn]]:
    """Group an episode's sentence table into per-scene turn sequences.

    Rows are sorted by ``start`` within each scene group before turn collapsing.
    """
    out: dict[int, list[Turn]] = {}
    for scene_id, grp in sentences.groupby("scene_id", sort=True):
        grp = grp.sort_values("start")
        out[int(scene_id)] = group_turns_for_scene(grp)
    return out


TRACE_COLUMNS = ["scene_id", "onset", "block_distance", "depth", "is_peak", "w", "tau_depth", "min_spacing"]


def _scene_trace_rows(
    scene_id: int,
    turns: list[Turn],
    vecs: np.ndarray,
    *,
    w: int,
    tau_depth: float,
    min_spacing: float,
) -> list[dict]:
    """Per-gap trace rows for one scene; empty if the scene is too short."""
    if len(turns) < 2 * w + 1 or len(vecs) != len(turns):
        return []
    trace = block_distance_trace(vecs, w)
    depth_by_idx = dict(peak_depths(trace))
    accepted = set(_accepted_peak_indices(trace, turns, tau_depth=tau_depth, min_spacing=min_spacing))
    rows: list[dict] = []
    for i in range(len(trace)):  # one gap per i in [0, n_turns-1)
        rows.append({
            "scene_id": scene_id,
            "onset": turns[i].end,
            "block_distance": float(trace[i]),
            "depth": float(depth_by_idx[i]) if i in depth_by_idx else float("nan"),
            "is_peak": i in accepted,
            "w": w,
            "tau_depth": tau_depth,
            "min_spacing": min_spacing,
        })
    return rows


def episode_topic_trace(
    turns_by_scene: dict[int, list[Turn]],
    vecs_by_scene: dict[int, np.ndarray],
    *,
    w: int,
    tau_depth: float,
    min_spacing: float,
) -> pd.DataFrame:
    """Assemble the per-gap topic-shift trace for one episode.

    One row per inter-turn gap within each scene, in scene then time order.
    Scenes with fewer than ``2*w + 1`` turns contribute no rows.
    """
    rows: list[dict] = []
    for scene_id in sorted(turns_by_scene):
        rows.extend(_scene_trace_rows(
            scene_id, turns_by_scene[scene_id], vecs_by_scene.get(scene_id, np.empty((0,))),
            w=w, tau_depth=tau_depth, min_spacing=min_spacing,
        ))
    return pd.DataFrame(rows, columns=TRACE_COLUMNS)


def minilm_encoder(model_name: str = "all-MiniLM-L6-v2") -> Encoder:
    """Build a CPU MiniLM encoder. Imported lazily so tests need no model."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")

    def encode(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
        return np.asarray(
            model.encode(texts, batch_size=64, show_progress_bar=False,
                         normalize_embeddings=False),
            dtype=np.float32,
        )

    return encode
