# Topic-Shift Trace Export — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the continuous topic-shift `block_distance` signal (plus a depth/`is_peak` audit trail) as a timestamped per-episode TSV with BIDS-inspired schema docs, for the `brain-states-friends` `08a`/`08b` pipeline.

**Architecture:** Reuse the existing `charnet.topic_shift` detector + embedding cache. Extract the greedy peak-acceptance into `_accepted_peak_indices` so `is_peak` matches the detector exactly; add `episode_topic_trace` to assemble per-gap rows; add a tiny `bids_meta` module for the data-dictionary + dataset_description JSON; add an `export_topic_trace.py` CLI mirroring `augment_scenes.py`.

**Tech Stack:** Python ≥3.10, pandas, numpy, sentence-transformers (cached, no re-encode), pytest.

**Spec:** `docs/brain_features/2026-05-29-topic-shift-trace-export-design.md`.

**Branch:** create `004-topic-trace-export` off `main` before Task 1.

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `src/charnet/topic_shift.py` | extract `_accepted_peak_indices`; add `episode_topic_trace` + `_scene_trace_rows` | Modify |
| `src/charnet/bids_meta.py` | `write_data_dictionary`, `write_dataset_description` (idempotent JSON) | Create |
| `scripts/export_topic_trace.py` | CLI: sentences → turns → cached embeddings → trace TSV + JSON docs | Create |
| `tests/test_topic_shift.py` | tests for `_accepted_peak_indices`, `episode_topic_trace` | Modify |
| `tests/test_bids_meta.py` | round-trip tests for the JSON writers | Create |
| `tests/test_export_topic_trace.py` | integration smoke (fake encoder, tiny TSV) | Create |

---

## Task 1: Extract `_accepted_peak_indices` (behavior-preserving refactor)

**Files:**
- Modify: `src/charnet/topic_shift.py`
- Test: `tests/test_topic_shift.py`

- [ ] **Step 1: Write the pinning test** (append to `tests/test_topic_shift.py`)

```python
from charnet.topic_shift import _accepted_peak_indices


def test_accepted_peak_indices_matches_propose_boundaries():
    A = np.array([1.0, 0.0]); B = np.array([0.0, 1.0]); C = np.array([1.0, 1.0]) / np.sqrt(2)
    vecs = np.stack([A, A, B, B, C, C])
    turns = [Turn("t", float(i), float(i) + 1.0) for i in range(6)]
    trace = block_distance_trace(vecs, 1)
    idx = _accepted_peak_indices(trace, turns, tau_depth=0.1, min_spacing=0.5)
    # accepted gap indices map to the same times propose_topic_boundaries returns
    times = sorted(turns[i].end for i in idx)
    assert times == propose_topic_boundaries(turns, vecs, w=1, tau_depth=0.1, min_spacing=0.5)
    # with tight spacing only the deepest survives
    idx_tight = _accepted_peak_indices(trace, turns, tau_depth=0.1, min_spacing=2.5)
    assert sorted(turns[i].end for i in idx_tight) == [2.0]
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_topic_shift.py::test_accepted_peak_indices_matches_propose_boundaries -q`
Expected: FAIL — cannot import `_accepted_peak_indices`.

- [ ] **Step 3: Extract the helper and rewire `propose_topic_boundaries`**

Add this function immediately above `propose_topic_boundaries` in `src/charnet/topic_shift.py`:

```python
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
```

Then replace the body of `propose_topic_boundaries` (keep its signature/docstring) so it delegates:

```python
    if len(turns) < 2 * w + 1 or len(vecs) != len(turns):  # require at least one full w-width block available on each side of some interior gap
        return []
    trace = block_distance_trace(vecs, w)
    idx = _accepted_peak_indices(trace, turns, tau_depth=tau_depth, min_spacing=min_spacing)
    return sorted(turns[i].end for i in idx)
```

- [ ] **Step 4: Run the full topic_shift test file**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_topic_shift.py -q`
Expected: all pass (existing `propose_topic_boundaries` tests still green + the new one). Confirms the refactor preserved behavior.

- [ ] **Step 5: Commit**

```bash
git add src/charnet/topic_shift.py tests/test_topic_shift.py
git commit -m "Extract _accepted_peak_indices from propose_topic_boundaries (no behavior change)"
```

---

## Task 2: `episode_topic_trace`

**Files:**
- Modify: `src/charnet/topic_shift.py`
- Test: `tests/test_topic_shift.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_topic_shift.py`)

```python
import math
from charnet.topic_shift import episode_topic_trace


def test_episode_topic_trace_rows_and_fields():
    A = np.array([1.0, 0.0]); B = np.array([0.0, 1.0]); C = np.array([1.0, 1.0]) / np.sqrt(2)
    turns_by_scene = {
        1: [Turn("t", float(i), float(i) + 1.0) for i in range(6)],   # 6 turns -> 5 gaps
        2: [Turn("x", 100.0, 101.0)],                                 # 1 turn -> no gaps, skipped
    }
    vecs_by_scene = {1: np.stack([A, A, B, B, C, C]), 2: np.zeros((1, 2))}
    df = episode_topic_trace(turns_by_scene, vecs_by_scene, w=1, tau_depth=0.1, min_spacing=0.5)

    # scene 2 (1 turn) contributes nothing; scene 1 contributes 5 gap rows
    assert list(df["scene_id"].unique()) == [1]
    assert len(df) == 5
    # onset = end of the turn before each gap: turns[0..4].end = 1..5
    assert list(df["onset"]) == [1.0, 2.0, 3.0, 4.0, 5.0]
    # block_distance trace = [0, 1.0, 0, ~0.293, 0]
    assert df.loc[df["onset"] == 2.0, "block_distance"].iloc[0] > 0.9
    assert df.loc[df["onset"] == 1.0, "block_distance"].iloc[0] == 0.0
    # depth present only at local maxima (gaps at onset 2.0 and 4.0), NaN elsewhere
    assert not math.isnan(df.loc[df["onset"] == 2.0, "depth"].iloc[0])
    assert math.isnan(df.loc[df["onset"] == 1.0, "depth"].iloc[0])
    # is_peak True where the detector accepts (onset 2.0 and 4.0 at loose spacing)
    assert bool(df.loc[df["onset"] == 2.0, "is_peak"].iloc[0]) is True
    assert bool(df.loc[df["onset"] == 1.0, "is_peak"].iloc[0]) is False
    # constant param columns
    assert set(df["w"]) == {1} and set(df["tau_depth"]) == {0.1} and set(df["min_spacing"]) == {0.5}


def test_episode_topic_trace_empty_when_all_scenes_too_short():
    turns_by_scene = {1: [Turn("a", 0.0, 1.0), Turn("b", 1.0, 2.0)]}  # 2 turns < 2*w+1 for w=1 (=3)
    vecs_by_scene = {1: np.zeros((2, 2))}
    df = episode_topic_trace(turns_by_scene, vecs_by_scene, w=1, tau_depth=0.1, min_spacing=0.5)
    assert list(df.columns) == ["scene_id", "onset", "block_distance", "depth", "is_peak", "w", "tau_depth", "min_spacing"]
    assert len(df) == 0
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_topic_shift.py::test_episode_topic_trace_rows_and_fields -q`
Expected: FAIL — cannot import `episode_topic_trace`.

- [ ] **Step 3: Implement** (add near the other detection functions in `src/charnet/topic_shift.py`)

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_topic_shift.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/charnet/topic_shift.py tests/test_topic_shift.py
git commit -m "Add episode_topic_trace: per-gap continuous topic-shift trace"
```

---

## Task 3: `bids_meta` — data dictionary + dataset_description writers

**Files:**
- Create: `src/charnet/bids_meta.py`
- Test: `tests/test_bids_meta.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bids_meta.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from charnet.bids_meta import write_data_dictionary, write_dataset_description


def test_write_data_dictionary_round_trip(tmp_path):
    cols = {"onset": {"Description": "gap time", "Units": "s"}}
    p = tmp_path / "topic_trace.json"
    write_data_dictionary(p, cols)
    loaded = json.loads(p.read_text())
    assert loaded["onset"]["Units"] == "s"
    # idempotent overwrite with new content
    write_data_dictionary(p, {"onset": {"Description": "changed", "Units": "s"}})
    assert json.loads(p.read_text())["onset"]["Description"] == "changed"


def test_write_dataset_description_required_keys(tmp_path):
    p = tmp_path / "dataset_description.json"
    write_dataset_description(p, name="charnet annotations", version="abc1234",
                              source_datasets=[{"Description": "NeuroMod Friends"}])
    d = json.loads(p.read_text())
    assert d["Name"] == "charnet annotations"
    assert d["DatasetType"] == "derivative"
    assert d["BIDSVersion"]
    assert d["GeneratedBy"][0]["Name"] == "charnet"
    assert d["GeneratedBy"][0]["Version"] == "abc1234"
    assert d["SourceDatasets"][0]["Description"] == "NeuroMod Friends"
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_bids_meta.py -q`
Expected: FAIL — `ModuleNotFoundError: charnet.bids_meta`.

- [ ] **Step 3: Implement**

```python
# src/charnet/bids_meta.py
"""BIDS-inspired sidecar writers for charnet annotation products.

Not full BIDS (stimulus-level, not subject-level) — only the data-dictionary
and dataset_description conventions that apply to a derivative annotation set.
"""
from __future__ import annotations

import json
from pathlib import Path

BIDS_VERSION = "1.9.0"


def write_data_dictionary(path: Path, columns: dict[str, dict]) -> None:
    """Write a column data dictionary (`{col: {Description, Units, Levels}}`).

    Idempotent: overwrites *path* with the given mapping.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(columns, indent=2) + "\n")


def write_dataset_description(
    path: Path,
    *,
    name: str,
    version: str,
    source_datasets: list[dict] | None = None,
) -> None:
    """Write a BIDS-style derivative `dataset_description.json`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = {
        "Name": name,
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "charnet", "Version": version}],
        "SourceDatasets": source_datasets or [],
    }
    path.write_text(json.dumps(desc, indent=2) + "\n")
```

- [ ] **Step 4: Run the tests**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_bids_meta.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/charnet/bids_meta.py tests/test_bids_meta.py
git commit -m "Add bids_meta: data-dictionary + dataset_description writers"
```

---

## Task 4: `export_topic_trace.py` CLI + integration smoke

**Files:**
- Create: `scripts/export_topic_trace.py`
- Test: `tests/test_export_topic_trace.py`

- [ ] **Step 1: Implement the CLI**

```python
# scripts/export_topic_trace.py
"""Export the continuous topic-shift trace per episode as a timestamped TSV.

  python scripts/export_topic_trace.py --episodes s3-s6 \
      --out-dir output/annotations/topic_shift
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from charnet import topic_shift as ts  # noqa: E402
from charnet.bids_meta import write_data_dictionary, write_dataset_description  # noqa: E402
from charnet.scene_subdivide import expand_episode_spec  # noqa: E402

DEFAULT_SCENES_IN = REPO / "output/annotations/scenes"
DEFAULT_SENTENCES_IN = REPO / "output/annotations/sentences"
DEFAULT_OUT_DIR = REPO / "output/annotations/topic_shift"
DEFAULT_CACHE_DIR = REPO / "output/intermediate/sentence_embeddings"

DATA_DICTIONARY = {
    "scene_id": {"Description": "Fan-transcript scene index the gap falls in"},
    "onset": {"Description": "Gap time: end of the turn before the gap, relative to episode start. Mapping to fMRI run time / TRs is the consumer's responsibility.", "Units": "s"},
    "block_distance": {"Description": "Cosine distance between mean-pooled w-turn blocks on either side of the gap; continuous topic-shift regressor. Higher = larger semantic shift.", "Units": "arbitrary (0-1 for normalized embeddings)"},
    "depth": {"Description": "TextTiling depth (rise above neighboring valleys) at local maxima of block_distance; NaN at non-maxima."},
    "is_peak": {"Description": "Gap accepted as a boundary by the topic-shift detector at the recorded params. NOTE: the detector is a documented negative result (docs/scene_segmentation_evaluation.md, Prototype #2); is_peak is an audit trail, not a validated boundary.", "Levels": {"true": "accepted", "false": "not accepted"}},
    "w": {"Description": "Block half-width in turns used to compute block_distance and is_peak."},
    "tau_depth": {"Description": "Depth threshold for is_peak."},
    "min_spacing": {"Description": "Minimum seconds between accepted peaks (greedy, deepest-first).", "Units": "s"},
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


def _episode_trace(episode, sentences_in, encoder, cache_dir, *, w, tau_depth, min_spacing):
    spath = _sentences_path(sentences_in, episode)
    if not spath.exists():
        return None
    sents = pd.read_csv(spath, sep="\t")
    if "scene_id" not in sents.columns:
        raise ValueError(f"{spath}: missing 'scene_id' column")
    by_scene = ts.turns_by_scene(sents)
    flat_texts, index = [], {}
    for sid, turns in by_scene.items():
        index[sid] = (len(flat_texts), len(flat_texts) + len(turns))
        flat_texts.extend(t.text for t in turns)
    all_vecs = ts.embed_texts_cached(episode, flat_texts, encoder, cache_dir)
    vecs_by_scene = {sid: all_vecs[lo:hi] for sid, (lo, hi) in index.items()}
    return ts.episode_topic_trace(by_scene, vecs_by_scene, w=w, tau_depth=tau_depth, min_spacing=min_spacing)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="ALL", help="ALL | sN | sN-sM | comma-list")
    ap.add_argument("--scenes-in", default=str(DEFAULT_SCENES_IN), help="root used only to resolve episode specs")
    ap.add_argument("--sentences-in", default=str(DEFAULT_SENTENCES_IN))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--w", type=int, default=1)
    ap.add_argument("--tau-depth", type=float, default=0.5)
    ap.add_argument("--min-spacing", type=float, default=30.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    episodes = expand_episode_spec(args.episodes, Path(args.scenes_in))
    encoder = ts.minilm_encoder()

    # schema docs (once): data dictionary at out_dir, dataset_description at the annotation root
    write_data_dictionary(out_dir / "topic_trace.json", DATA_DICTIONARY)
    write_dataset_description(
        out_dir.parent / "dataset_description.json",
        name="charnet Friends stimulus annotations",
        version=_git_version(),
        source_datasets=[{"Description": "Courtois NeuroMod Friends fMRI stimulus episodes"}],
    )

    print(f"Exporting topic trace for {len(episodes)} eps → {out_dir}")
    n_written = n_skipped = 0
    for ep in episodes:
        df = _episode_trace(ep, Path(args.sentences_in), encoder, Path(args.cache_dir),
                            w=args.w, tau_depth=args.tau_depth, min_spacing=args.min_spacing)
        if df is None:
            n_skipped += 1
            continue
        season = int(ep[1:3])
        ep_dir = out_dir / f"s{season}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(ep_dir / f"friends_{ep}_topic_trace.tsv", sep="\t", index=False)
        n_written += 1
        print(f"  {ep}: {len(df)} gaps")
    print(f"\nWrote {n_written} episodes ({n_skipped} missing sentence tables)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the integration smoke test**

```python
# tests/test_export_topic_trace.py
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
```

- [ ] **Step 3: Run the smoke tests**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_export_topic_trace.py -q`
Expected: 2 passed.

- [ ] **Step 4: Real one-episode run (writes the actual product + JSON docs)**

Run: `cd /orcd/home/002/yibei/te-charnet && python scripts/export_topic_trace.py --episodes s01e01a --out-dir /tmp/tt_smoke --cache-dir /tmp/tt_cache`
Expected: prints `s01e01a: N gaps`; `/tmp/tt_smoke/s1/friends_s01e01a_topic_trace.tsv` exists with the 8 documented columns; `/tmp/tt_smoke/topic_trace.json` and `/tmp/dataset_description.json` (i.e. `out_dir.parent`) exist and parse as JSON. (First run loads MiniLM, ~1 min; reports BLOCKED only if the model download fails.)

- [ ] **Step 5: Full suite + lint**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest -q && ruff check scripts/export_topic_trace.py src/charnet/bids_meta.py src/charnet/topic_shift.py`
Expected: all tests pass; ruff clean on these files.

- [ ] **Step 6: Commit**

```bash
git add scripts/export_topic_trace.py tests/test_export_topic_trace.py
git commit -m "Add export_topic_trace.py: timestamped trace TSV + BIDS-inspired schema docs"
```

---

## Self-review

- **Spec coverage:** data product schema (Task 2 + Task 4 DATA_DICTIONARY) ✓; `onset` naming ✓; constant param columns ✓; `is_peak` = accepted boundaries via shared `_accepted_peak_indices` (Task 1) ✓; depth NaN off local maxima (Task 2) ✓; scene `<2w+1` skipped (Task 2) ✓; shared data dictionary + dataset_description (Task 3 + Task 4) ✓; reuse of embedding cache, no re-encode (Task 4 `_episode_trace`) ✓; refactor behavior-preserving (Task 1 pinning test) ✓; testing plan (all tasks) ✓.
- **Placeholder scan:** none — `<ep>`/`<git-describe>` are runtime-filled; `_git_version()` implements the latter.
- **Type consistency:** `TRACE_COLUMNS` order matches the test assertions and the export TSV; `_accepted_peak_indices(trace, turns, *, tau_depth, min_spacing)` signature is identical in Task 1's definition, its use in `propose_topic_boundaries`, and `_scene_trace_rows`; `episode_topic_trace(turns_by_scene, vecs_by_scene, *, w, tau_depth, min_spacing)` matches between Task 2 and Task 4's `_episode_trace` call.
- **Lane:** export only produces te-charnet outputs; no brain-repo changes. ✓
```
