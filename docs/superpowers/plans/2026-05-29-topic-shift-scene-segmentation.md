# Topic-Shift Scene Subdivision (#2) + char×topic Hybrid — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a sentence-embedding topic-shift scene-boundary detector and a char×topic AND-logic hybrid, then run one calibrated 292-ep sweep that tests both against manual annotations.

**Architecture:** Refactor the existing char-presence augmenter into reusable modules (`char_presence.py` proposer + generic `scene_subdivide.py` plumbing), add a `topic_shift.py` module (ct-turn grouping → MiniLM embeddings, cached → TextTiling-style block depth-score detection), and a unified `augment_scenes.py` CLI with `char`/`topic`/`hybrid` modes. Thresholds are calibrated on s1–s2 and the headline is reported on held-out s3–s6 via the unchanged `evaluate_scene_segmentation.py`.

**Tech Stack:** Python ≥3.10, pandas, numpy, `sentence-transformers` (new, `all-MiniLM-L6-v2`, CPU), pytest.

**Spec:** `docs/superpowers/specs/2026-05-29-topic-shift-scene-segmentation-design.md` (local, untracked).

**Branch:** `003-topic-shift-segmentation` (already created).

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `src/charnet/char_presence.py` | char-presence candidate proposer (`propose_sub_boundaries` + helpers), lifted verbatim from the script | Create |
| `src/charnet/scene_subdivide.py` | generic plumbing: read `scene_summary.tsv`, call a per-scene `propose(scene)→times`, rewrite/renumber rows, write augmented TSV | Create |
| `src/charnet/topic_shift.py` | ct-turn grouping, MiniLM encoder + on-disk cache, block depth-score detector, `propose_topic_boundaries` | Create |
| `scripts/augment_scenes.py` | unified CLI: `--mode {char,topic,hybrid}`, episode-spec expansion, wires proposers into `scene_subdivide` | Create |
| `scripts/augment_scenes_char_presence.py` | reduced to a thin shim importing the moved proposer (keeps prototype-#1 invocation working) | Modify |
| `scripts/calibrate_topic_shift.py` | grid-search W×τ_depth×M on s1–s2, report grid + best params | Create |
| `tests/test_char_presence.py` | characterization tests for the moved proposer | Create |
| `tests/test_scene_subdivide.py` | row-rewrite/renumber/suffix behaviour | Create |
| `tests/test_topic_shift.py` | turn grouping, depth detection, cache, proposer, hybrid intersection, episode-spec expansion | Create |
| `pyproject.toml` | add `sentence-transformers` dependency | Modify |

**Frozen (do not modify):** `scripts/evaluate_scene_segmentation.py`, `src/charnet/visual_presence.py`.

---

## Task 1: Add dependency and episode-spec expansion helper

**Files:**
- Modify: `pyproject.toml`
- Create: `src/charnet/scene_subdivide.py`
- Test: `tests/test_scene_subdivide.py`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add `"sentence-transformers>=2.2"` to the `dependencies` list (the list currently containing `"numpy>=1.24"`).

- [ ] **Step 2: Install it**

Run: `uv pip install 'sentence-transformers>=2.2'`
Expected: resolves and installs (pulls torch CPU + transformers). May take a few minutes.

- [ ] **Step 3: Write the failing test for episode-spec expansion**

```python
# tests/test_scene_subdivide.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from charnet.scene_subdivide import expand_episode_spec


def _touch_scene_files(root: Path, episodes: list[str]) -> None:
    for ep in episodes:
        season = int(ep[1:3])
        d = root / f"s{season}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"friends_{ep}_scene_summary.tsv").write_text("scene_id\tstart\tend\n")


def test_expand_episode_spec_all(tmp_path):
    _touch_scene_files(tmp_path, ["s01e01a", "s02e03b", "s03e01a"])
    assert expand_episode_spec("ALL", tmp_path) == ["s01e01a", "s02e03b", "s03e01a"]


def test_expand_episode_spec_single_season(tmp_path):
    _touch_scene_files(tmp_path, ["s01e01a", "s02e03b", "s03e01a"])
    assert expand_episode_spec("s2", tmp_path) == ["s02e03b"]


def test_expand_episode_spec_season_range(tmp_path):
    _touch_scene_files(tmp_path, ["s01e01a", "s02e03b", "s03e01a", "s06e02a"])
    assert expand_episode_spec("s3-s6", tmp_path) == ["s03e01a", "s06e02a"]


def test_expand_episode_spec_explicit_list(tmp_path):
    _touch_scene_files(tmp_path, ["s01e01a", "s02e03b"])
    assert expand_episode_spec("s01e01a,s02e03b", tmp_path) == ["s01e01a", "s02e03b"]
```

- [ ] **Step 4: Run it to confirm failure**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_scene_subdivide.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'charnet.scene_subdivide'`.

- [ ] **Step 5: Implement the helper (module skeleton + expansion)**

```python
# src/charnet/scene_subdivide.py
"""Generic scene-subdivision plumbing shared by all augmenters.

Reads a fan-transcript ``scene_summary.tsv``, asks a per-scene ``propose``
callback for interior sub-boundary times, and rewrites the table with the
new boundaries (renumbered scene ids, inherited descriptions tagged with an
augmentation suffix).
"""
from __future__ import annotations

import csv
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SCENE_GLOB = "friends_*_scene_summary.tsv"
_EP_RE = re.compile(r"friends_(s\d{2}e\d{2}[a-z])_scene_summary\.tsv")


def _all_episodes(scenes_in_dir: Path) -> list[str]:
    eps = []
    for p in scenes_in_dir.rglob(SCENE_GLOB):
        m = _EP_RE.match(p.name)
        if m:
            eps.append(m.group(1))
    return sorted(eps)


def expand_episode_spec(spec: str, scenes_in_dir: Path) -> list[str]:
    """Expand an episode spec into a sorted episode-id list.

    Accepts: ``ALL``; a single season ``s3``; a season range ``s3-s6``
    (inclusive); or a comma-separated explicit list ``s01e01a,s02e03b``.
    Season specs filter the episodes actually present under *scenes_in_dir*.
    """
    spec = spec.strip()
    if spec == "ALL":
        return _all_episodes(scenes_in_dir)
    range_m = re.fullmatch(r"s(\d+)-s(\d+)", spec)
    if range_m:
        lo, hi = int(range_m.group(1)), int(range_m.group(2))
        return [e for e in _all_episodes(scenes_in_dir) if lo <= int(e[1:3]) <= hi]
    single_m = re.fullmatch(r"s(\d+)", spec)
    if single_m:
        n = int(single_m.group(1))
        return [e for e in _all_episodes(scenes_in_dir) if int(e[1:3]) == n]
    return [e.strip() for e in spec.split(",") if e.strip()]
```

- [ ] **Step 6: Run the tests to confirm pass**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_scene_subdivide.py -q`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/charnet/scene_subdivide.py tests/test_scene_subdivide.py
git commit -m "Add sentence-transformers dep + episode-spec expansion helper"
```

---

## Task 2: Generic scene-subdivision plumbing

**Files:**
- Modify: `src/charnet/scene_subdivide.py`
- Test: `tests/test_scene_subdivide.py`

This lifts the row-rewrite logic from `augment_scenes_char_presence.py:augment_episode` into a reusable `subdivide_episode`, parameterised by a `propose` callback and an `aug_tag`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_scene_subdivide.py
import pandas as pd
from charnet.scene_subdivide import Scene, subdivide_episode


def _write_scene_table(root: Path, episode: str, rows: list[dict]) -> None:
    season = int(episode[1:3])
    d = root / f"s{season}"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        d / f"friends_{episode}_scene_summary.tsv", sep="\t", index=False
    )


def test_subdivide_splits_and_renumbers(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _write_scene_table(in_dir, "s01e01a", [
        {"scene_id": 1, "scene_desc": "Central Perk", "start": 0.0, "end": 100.0, "shot_ids": "1|2"},
        {"scene_id": 2, "scene_desc": "Monica's", "start": 100.0, "end": 150.0, "shot_ids": "3"},
    ])

    # propose one interior boundary at t=50 inside the first scene only
    def propose(scene: Scene) -> list[float]:
        return [50.0] if scene.scene_id == 1 else []

    stats = subdivide_episode("s01e01a", in_dir, out_dir, propose, aug_tag="topic_aug")

    out = pd.read_csv(out_dir / "s1" / "friends_s01e01a_scene_summary.tsv", sep="\t")
    assert list(out["scene_id"]) == [1, 2, 3]          # renumbered contiguously
    assert list(out["start"]) == [0.0, 50.0, 100.0]
    assert list(out["end"]) == [50.0, 100.0, 150.0]
    # first sub-scene inherits desc + shot_ids; the new sub-scene is tagged, shot_ids cleared
    assert out.loc[0, "scene_desc"] == "Central Perk"
    assert out.loc[1, "scene_desc"] == "Central Perk [topic_aug 1]"
    assert str(out.loc[1, "shot_ids"]) in ("", "nan")
    assert stats["n_new_boundaries"] == 1
    assert stats["n_output_scenes"] == 3
```

- [ ] **Step 2: Run it to confirm failure**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_scene_subdivide.py::test_subdivide_splits_and_renumbers -q`
Expected: FAIL — cannot import `Scene` / `subdivide_episode`.

- [ ] **Step 3: Implement `Scene` + `subdivide_episode`**

```python
# add to src/charnet/scene_subdivide.py

@dataclass(frozen=True)
class Scene:
    scene_id: int
    scene_desc: str
    start: float
    end: float
    shot_ids: str


ProposeFn = Callable[[Scene], list[float]]


def subdivide_episode(
    episode: str,
    scenes_in_dir: Path,
    scenes_out_dir: Path,
    propose: ProposeFn,
    *,
    aug_tag: str,
) -> dict:
    """Rewrite one episode's scene table, inserting proposed sub-boundaries.

    *propose* receives each input :class:`Scene` and returns a list of strictly
    interior boundary times. Output rows are renumbered 1..N; sub-scenes after
    the first inherit ``scene_desc`` with a ``[<aug_tag> k]`` suffix and an
    empty ``shot_ids``.
    """
    season = int(episode[1:3])
    in_path = scenes_in_dir / f"s{season}" / f"friends_{episode}_scene_summary.tsv"
    out_dir = scenes_out_dir / f"s{season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"friends_{episode}_scene_summary.tsv"

    df = pd.read_csv(in_path, sep="\t")
    new_rows: list[dict] = []
    n_new = 0
    next_id = 1
    for _, row in df.iterrows():
        scene = Scene(
            scene_id=int(row["scene_id"]),
            scene_desc=str(row.get("scene_desc", "") or ""),
            start=float(row["start"]),
            end=float(row["end"]),
            shot_ids=str(row.get("shot_ids", "") or ""),
        )
        subs = [b for b in sorted(propose(scene)) if scene.start < b < scene.end]
        if not subs:
            new_rows.append({
                "scene_id": next_id, "scene_desc": scene.scene_desc,
                "start": f"{scene.start:.2f}", "end": f"{scene.end:.2f}",
                "shot_ids": scene.shot_ids,
            })
            next_id += 1
            continue
        n_new += len(subs)
        bounds = [scene.start] + subs + [scene.end]
        for k in range(len(bounds) - 1):
            desc = scene.scene_desc if k == 0 else f"{scene.scene_desc} [{aug_tag} {k}]"
            new_rows.append({
                "scene_id": next_id, "scene_desc": desc,
                "start": f"{bounds[k]:.2f}", "end": f"{bounds[k + 1]:.2f}",
                "shot_ids": scene.shot_ids if k == 0 else "",
            })
            next_id += 1

    out_df = pd.DataFrame(new_rows, columns=["scene_id", "scene_desc", "start", "end", "shot_ids"])
    out_df.to_csv(out_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    return {
        "episode": episode,
        "n_input_scenes": len(df),
        "n_output_scenes": len(out_df),
        "n_new_boundaries": n_new,
    }
```

- [ ] **Step 4: Run the tests**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_scene_subdivide.py -q`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/charnet/scene_subdivide.py tests/test_scene_subdivide.py
git commit -m "Add generic subdivide_episode plumbing"
```

---

## Task 3: Move the char-presence proposer into src/charnet

**Files:**
- Create: `src/charnet/char_presence.py`
- Modify: `scripts/augment_scenes_char_presence.py`
- Test: `tests/test_char_presence.py`

Lift the proposer verbatim so the hybrid can import it; reduce the script to a shim.

- [ ] **Step 1: Create `char_presence.py` by moving the pure functions**

Move these from `scripts/augment_scenes_char_presence.py` into `src/charnet/char_presence.py` **unchanged**: the constants `TILE_SECS`, `PRESENCE_FRAC`, `JACCARD_THRESH`, `MIN_SPACING_SECS`, `PERSISTENCE_TILES`, `SHOT_SNAP_WINDOW`, `MIN_SCENE_LENGTH`, `DEFAULT_SHOTS_DIR`, and the functions `jaccard_distance`, `load_shot_transitions`, `snap_to_shot`, `tile_active_set`, `propose_sub_boundaries`.

```python
# src/charnet/char_presence.py
"""Character-presence sub-boundary proposer (prototype #1).

Lifted from scripts/augment_scenes_char_presence.py so the hybrid augmenter
can import propose_sub_boundaries. Logic is unchanged; see
docs/scene_segmentation_evaluation.md "Prototype #1 results".
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_SHOTS_DIR = REPO / "data/friends_annotations/annotation_results/TSVpyscene"

TILE_SECS = 5.0
PRESENCE_FRAC = 0.20
JACCARD_THRESH = 0.5
MIN_SPACING_SECS = 15.0
PERSISTENCE_TILES = 2
SHOT_SNAP_WINDOW = 3.0
MIN_SCENE_LENGTH = 0.0

# (paste jaccard_distance, load_shot_transitions, snap_to_shot,
#  tile_active_set, propose_sub_boundaries here, verbatim from the script)
```

(Paste the five functions exactly as they appear at `scripts/augment_scenes_char_presence.py:52-170`. Do not alter their bodies.)

- [ ] **Step 2: Write a characterization test pinning the proposer's behaviour**

```python
# tests/test_char_presence.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from charnet.char_presence import jaccard_distance, propose_sub_boundaries


def test_jaccard_distance_basics():
    assert jaccard_distance(set(), set()) == 0.0
    assert jaccard_distance({"a"}, {"a"}) == 0.0
    assert jaccard_distance({"a"}, {"b"}) == 1.0
    assert jaccard_distance({"a", "b"}, {"a"}) == 0.5


def test_propose_fires_on_persistent_char_change():
    # 0-60s: chars present = {ross}; 60-120s: {monica}. One grid col each.
    chars = ["ross", "monica"]
    rows = [[1, 0]] * 60 + [[0, 1]] * 60
    subs = propose_sub_boundaries(
        0.0, 120.0, chars, rows,
        tile_secs=5.0, presence_frac=0.2, jaccard_thresh=0.5,
        min_spacing=15.0, persistence_tiles=2, shot_times=None,
        shot_snap_window=3.0, shot_snap_required=False, min_scene_length=0.0,
    )
    assert subs == [60.0]


def test_propose_ignores_transient_flicker():
    chars = ["ross", "monica"]
    rows = [[1, 0]] * 55 + [[0, 1]] * 5 + [[1, 0]] * 60  # 5s blip
    subs = propose_sub_boundaries(
        0.0, 120.0, chars, rows,
        tile_secs=5.0, presence_frac=0.2, jaccard_thresh=0.5,
        min_spacing=15.0, persistence_tiles=2, shot_times=None,
        shot_snap_window=3.0, shot_snap_required=False, min_scene_length=0.0,
    )
    assert subs == []
```

- [ ] **Step 3: Run the tests to confirm pass**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_char_presence.py -q`
Expected: 3 passed. (If `test_propose_ignores_transient_flicker` fails, the paste altered behaviour — re-copy verbatim.)

- [ ] **Step 4: Reduce the script to a shim**

Replace the moved definitions in `scripts/augment_scenes_char_presence.py` with an import, keeping its CLI (`augment_episode`, `main`) intact:

```python
# near the top of scripts/augment_scenes_char_presence.py, after sys.path insert
from charnet.char_presence import (  # noqa: E402
    DEFAULT_SHOTS_DIR, JACCARD_THRESH, MIN_SCENE_LENGTH, MIN_SPACING_SECS,
    PERSISTENCE_TILES, PRESENCE_FRAC, SHOT_SNAP_WINDOW, TILE_SECS,
    jaccard_distance, load_shot_transitions, propose_sub_boundaries, snap_to_shot,
    tile_active_set,
)
```

Delete the now-duplicated constant and function definitions from the script. Leave `augment_episode`/`main` as-is.

- [ ] **Step 5: Verify the script still runs on one episode**

Run: `cd /orcd/home/002/yibei/te-charnet && python scripts/augment_scenes_char_presence.py --episodes s01e01a --scenes-out /tmp/char_shim_check`
Expected: prints `s01e01a: N → M (+k)` with no import error; `/tmp/char_shim_check/s1/friends_s01e01a_scene_summary.tsv` exists.

- [ ] **Step 6: Commit**

```bash
git add src/charnet/char_presence.py scripts/augment_scenes_char_presence.py tests/test_char_presence.py
git commit -m "Move char-presence proposer into charnet.char_presence; script becomes a shim"
```

---

## Task 4: Topic-shift — ct-turn grouping

**Files:**
- Create: `src/charnet/topic_shift.py`
- Test: `tests/test_topic_shift.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic_shift.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from charnet.topic_shift import Turn, build_text, group_turns_for_scene


def test_build_text_prefers_ct_falls_back_to_utterance():
    assert build_text("clean line", "asr noise") == "clean line"
    assert build_text("", "asr fallback") == "asr fallback"
    assert build_text(np.nan, "asr fallback") == "asr fallback"   # NaN, not "nan"
    assert build_text("   ", "asr fallback") == "asr fallback"


def test_group_turns_dedups_consecutive_identical_ct():
    df = pd.DataFrame([
        {"utterance_ct": "Hi there folks", "utterance": "hi", "start": 1.0, "end": 2.0},
        {"utterance_ct": "Hi there folks", "utterance": "there", "start": 2.0, "end": 3.5},
        {"utterance_ct": "Different turn", "utterance": "diff", "start": 3.5, "end": 5.0},
    ])
    turns = group_turns_for_scene(df)
    assert turns == [
        Turn(text="Hi there folks", start=1.0, end=3.5),
        Turn(text="Different turn", start=3.5, end=5.0),
    ]


def test_group_turns_blank_ct_uses_utterance_and_does_not_merge():
    df = pd.DataFrame([
        {"utterance_ct": "", "utterance": "first asr", "start": 1.0, "end": 2.0},
        {"utterance_ct": "", "utterance": "second asr", "start": 2.0, "end": 3.0},
    ])
    turns = group_turns_for_scene(df)
    assert [t.text for t in turns] == ["first asr", "second asr"]
```

- [ ] **Step 2: Run it to confirm failure**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_topic_shift.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'charnet.topic_shift'`.

- [ ] **Step 3: Implement turn grouping**

```python
# src/charnet/topic_shift.py
"""Topic-shift sub-boundary proposer (improvement direction #2).

Pipeline per scene: group sentence rows into community-transcript "turns",
embed each turn (MiniLM, cached), score each inter-turn gap by the cosine
distance between mean-pooled blocks of W turns on either side, and propose
boundaries at local-maximum gaps whose TextTiling depth exceeds a threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Turn:
    text: str
    start: float
    end: float


def build_text(utterance_ct, utterance) -> str:
    """Best-available turn text: community transcript, else Speech2Text.

    NaN-safe per the repo gotcha — never stringify NaN to "nan".
    """
    for val in (utterance_ct, utterance):
        if val is None:
            continue
        if isinstance(val, float) and pd.isna(val):
            continue
        s = str(val).strip()
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
        ct = "" if (ct_raw is None or (isinstance(ct_raw, float) and pd.isna(ct_raw))) else str(ct_raw).strip()
        text = build_text(ct_raw, row.get("utterance"))
        start, end = float(row["start"]), float(row["end"])
        mergeable = ct != "" and ct == prev_ct_key
        if mergeable and turns:
            last = turns[-1]
            turns[-1] = Turn(text=last.text, start=last.start, end=max(last.end, end))
        else:
            turns.append(Turn(text=text, start=start, end=end))
        prev_ct_key = ct if ct != "" else None
    return turns
```

- [ ] **Step 4: Run the tests**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_topic_shift.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/charnet/topic_shift.py tests/test_topic_shift.py
git commit -m "Topic-shift: ct-turn grouping with NaN-safe text fallback"
```

---

## Task 5: Topic-shift — block depth-score detection

**Files:**
- Modify: `src/charnet/topic_shift.py`
- Test: `tests/test_topic_shift.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_topic_shift.py
from charnet.topic_shift import block_distance_trace, peak_depths, propose_topic_boundaries


def test_block_distance_trace_flags_the_topic_change():
    # 6 turns: first 3 ≈ vector A, last 3 ≈ vector B. W=2 → gap at index 2 (turn2|turn3) is largest.
    A = np.array([1.0, 0.0])
    B = np.array([0.0, 1.0])
    vecs = np.stack([A, A, A, B, B, B])
    trace = block_distance_trace(vecs, w=2)
    assert len(trace) == 5                       # n_turns - 1
    assert np.argmax(trace) == 2                 # boundary between the two halves
    assert trace[2] > 0.9                         # near-orthogonal blocks


def test_peak_depths_picks_local_maxima():
    trace = np.array([0.1, 0.2, 0.9, 0.2, 0.15, 0.8, 0.1])
    peaks = dict(peak_depths(trace))             # {gap_index: depth}
    assert 2 in peaks and 5 in peaks
    assert peaks[2] > peaks[5]                    # deeper peak first


def test_propose_topic_boundaries_returns_turn_end_time():
    A, B = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    vecs = np.stack([A, A, A, B, B, B])
    turns = [Turn("t", float(i), float(i) + 1.0) for i in range(6)]  # each 1s, contiguous
    subs = propose_topic_boundaries(
        turns, vecs, w=2, tau_depth=0.3, min_spacing=0.5,
    )
    # gap index 2 → boundary at turns[2].end == 3.0
    assert subs == [3.0]


def test_propose_topic_boundaries_too_few_turns():
    vecs = np.zeros((3, 2))
    turns = [Turn("t", float(i), float(i) + 1.0) for i in range(3)]
    assert propose_topic_boundaries(turns, vecs, w=2, tau_depth=0.3, min_spacing=0.5) == []
```

- [ ] **Step 2: Run it to confirm failure**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_topic_shift.py -q`
Expected: FAIL — cannot import `block_distance_trace`.

- [ ] **Step 3: Implement detection**

```python
# add to src/charnet/topic_shift.py

def _mean_block(vecs: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Mean of rows [lo, hi); caller guarantees lo < hi within bounds."""
    return vecs[lo:hi].mean(axis=0)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(1.0 - np.dot(a, b) / (na * nb))


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
    if len(turns) < 2 * w + 1 or len(vecs) != len(turns):
        return []
    trace = block_distance_trace(vecs, w)
    accepted_idx: list[int] = []
    for gap_i, depth in peak_depths(trace):
        if depth < tau_depth:
            continue
        t = turns[gap_i].end
        if any(abs(t - turns[j].end) < min_spacing for j in accepted_idx):
            continue
        accepted_idx.append(gap_i)
    return sorted(turns[i].end for i in accepted_idx)
```

- [ ] **Step 4: Run the tests**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_topic_shift.py -q`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/charnet/topic_shift.py tests/test_topic_shift.py
git commit -m "Topic-shift: block depth-score boundary detection"
```

---

## Task 6: Topic-shift — MiniLM encoder and on-disk cache

**Files:**
- Modify: `src/charnet/topic_shift.py`
- Test: `tests/test_topic_shift.py`

The cache must be testable without downloading a model, so encoding is injected as a callable. The real MiniLM encoder is a thin factory used only by the CLI.

- [ ] **Step 1: Write the failing test (cache round-trip with a fake encoder)**

```python
# append to tests/test_topic_shift.py
from charnet.topic_shift import embed_texts_cached


class _FakeEncoder:
    """Deterministic 2-d encoder: returns [len(text), n_vowels]. Counts calls."""
    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return np.array(
            [[float(len(t)), float(sum(c in "aeiou" for c in t.lower()))] for t in texts]
        )


def test_embed_texts_cached_round_trip_and_reuse(tmp_path):
    enc = _FakeEncoder()
    texts = ["hello world", "topic shift"]
    v1 = embed_texts_cached("s01e01a", texts, enc, tmp_path)
    assert v1.shape == (2, 2)
    assert enc.calls == 1
    # second call hits cache → no new encode
    v2 = embed_texts_cached("s01e01a", texts, enc, tmp_path)
    assert enc.calls == 1
    assert np.allclose(v1, v2)


def test_embed_texts_cached_invalidates_on_text_change(tmp_path):
    enc = _FakeEncoder()
    embed_texts_cached("s01e01a", ["a", "b"], enc, tmp_path)
    embed_texts_cached("s01e01a", ["a", "c"], enc, tmp_path)  # changed → re-encode
    assert enc.calls == 2
```

- [ ] **Step 2: Run it to confirm failure**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_topic_shift.py -q`
Expected: FAIL — cannot import `embed_texts_cached`.

- [ ] **Step 3: Implement cache + encoder factory**

```python
# add to src/charnet/topic_shift.py — extend the existing imports at the top with:
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

Encoder = Callable[[list[str]], np.ndarray]


def _texts_hash(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def embed_texts_cached(
    episode: str, texts: list[str], encoder: Encoder, cache_dir: Path
) -> np.ndarray:
    """Encode *texts*, caching by (episode, texts-hash) under *cache_dir*.

    Cache layout: ``<cache_dir>/<season>/<episode>.npz`` storing the vectors
    plus the text hash; a hash mismatch (texts changed) forces a re-encode.
    """
    cache_dir = Path(cache_dir)
    season = f"s{int(episode[1:3])}"
    path = cache_dir / season / f"{episode}.npz"
    key = _texts_hash(texts)
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if str(cached["key"]) == key and cached["vecs"].shape[0] == len(texts):
            return cached["vecs"]
    vecs = np.asarray(encoder(texts), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vecs=vecs, key=np.array(key))
    return vecs


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
```

(Note: remove the now-duplicated `from dataclasses import dataclass` / numpy / pandas imports only if they would conflict — keep a single import block at the top of the module.)

- [ ] **Step 4: Run the tests**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_topic_shift.py -q`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/charnet/topic_shift.py tests/test_topic_shift.py
git commit -m "Topic-shift: MiniLM encoder factory + on-disk embedding cache"
```

---

## Task 7: Hybrid intersection + per-episode topic plumbing

**Files:**
- Modify: `src/charnet/topic_shift.py`
- Test: `tests/test_topic_shift.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_topic_shift.py
from charnet.topic_shift import intersect_within, turns_by_scene


def test_intersect_within_keeps_topic_time_when_char_agrees():
    char_times = [10.0, 50.0, 90.0]
    topic_times = [12.0, 70.0]          # 12 is within 3s of 10; 70 has no char within 3s
    assert intersect_within(char_times, topic_times, eps=3.0) == [12.0]


def test_intersect_within_empty_side():
    assert intersect_within([], [12.0], eps=3.0) == []
    assert intersect_within([10.0], [], eps=3.0) == []


def test_turns_by_scene_groups_by_scene_id():
    df = pd.DataFrame([
        {"scene_id": 1, "utterance_ct": "a", "utterance": "a", "start": 0.0, "end": 1.0},
        {"scene_id": 1, "utterance_ct": "b", "utterance": "b", "start": 1.0, "end": 2.0},
        {"scene_id": 2, "utterance_ct": "c", "utterance": "c", "start": 2.0, "end": 3.0},
    ])
    by_scene = turns_by_scene(df)
    assert set(by_scene) == {1, 2}
    assert [t.text for t in by_scene[1]] == ["a", "b"]
    assert [t.text for t in by_scene[2]] == ["c"]
```

- [ ] **Step 2: Run it to confirm failure**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_topic_shift.py -q`
Expected: FAIL — cannot import `intersect_within`.

- [ ] **Step 3: Implement**

```python
# add to src/charnet/topic_shift.py

def intersect_within(char_times, topic_times, *, eps: float) -> list[float]:
    """Topic times that have a char time within *eps*; placed at the topic time."""
    out = [t for t in topic_times if any(abs(t - c) <= eps for c in char_times)]
    return sorted(out)


def turns_by_scene(sentences: pd.DataFrame) -> dict[int, list[Turn]]:
    """Group an episode's sentence table into per-scene turn sequences.

    Rows are ordered by ``start`` within each ``scene_id`` before grouping.
    """
    out: dict[int, list[Turn]] = {}
    for scene_id, grp in sentences.groupby("scene_id", sort=True):
        grp = grp.sort_values("start")
        out[int(scene_id)] = group_turns_for_scene(grp)
    return out
```

- [ ] **Step 4: Run the tests**

Run: `cd /orcd/home/002/yibei/te-charnet && python -m pytest tests/test_topic_shift.py -q`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/charnet/topic_shift.py tests/test_topic_shift.py
git commit -m "Topic-shift: hybrid intersection + per-scene turn grouping"
```

---

## Task 8: Unified augment_scenes.py CLI (char / topic / hybrid)

**Files:**
- Create: `scripts/augment_scenes.py`

This wires the proposers into `scene_subdivide.subdivide_episode` per mode. No new unit test (logic is covered in Tasks 2–7); validated by an integration smoke run.

- [ ] **Step 1: Implement the CLI**

```python
# scripts/augment_scenes.py
"""Augment fan-transcript scene boundaries — char / topic / hybrid modes.

  python scripts/augment_scenes.py --mode topic --episodes s3-s6 \
      --scenes-out output/annotations/scenes_topic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from charnet import char_presence as cp  # noqa: E402
from charnet import topic_shift as ts  # noqa: E402
from charnet.scene_subdivide import Scene, expand_episode_spec, subdivide_episode  # noqa: E402
from charnet.visual_presence import (  # noqa: E402
    DEFAULT_CHAR_TRACKER_DIR, char_tracker_csv_path, load_char_tracker_grid,
    resolve_char_tracker_dir,
)

DEFAULT_SCENES_IN = REPO / "output/annotations/scenes"
DEFAULT_SENTENCES_IN = REPO / "output/annotations/sentences"
DEFAULT_CACHE_DIR = REPO / "output/intermediate/sentence_embeddings"
HYBRID_EPS = 3.0


def _sentences_path(sentences_in: Path, episode: str) -> Path:
    season = int(episode[1:3])
    return sentences_in / f"s{season}" / f"friends_{episode}_sentence_speaker_table.tsv"


def _build_char_propose(episode, ct_dir, shots_dir, params):
    grid_path = char_tracker_csv_path(ct_dir, f"friends_{episode}")
    if grid_path is None:
        return None  # signal: no char data
    chars, rows = load_char_tracker_grid(grid_path)
    shot_times = cp.load_shot_transitions(episode, shots_dir)

    def propose(scene: Scene) -> list[float]:
        return cp.propose_sub_boundaries(
            scene.start, scene.end, chars, rows,
            tile_secs=cp.TILE_SECS, presence_frac=cp.PRESENCE_FRAC,
            jaccard_thresh=cp.JACCARD_THRESH, min_spacing=cp.MIN_SPACING_SECS,
            persistence_tiles=cp.PERSISTENCE_TILES,
            shot_times=shot_times if shot_times else None,
            shot_snap_window=cp.SHOT_SNAP_WINDOW, shot_snap_required=False,
            min_scene_length=cp.MIN_SCENE_LENGTH,
        )
    return propose


def _build_topic_propose(episode, sentences_in, encoder, cache_dir, params):
    spath = _sentences_path(sentences_in, episode)
    if not spath.exists():
        return None
    sents = pd.read_csv(spath, sep="\t")
    by_scene = ts.turns_by_scene(sents)
    # encode every turn text in the episode once (cached)
    flat_texts, index = [], {}
    for sid, turns in by_scene.items():
        index[sid] = (len(flat_texts), len(flat_texts) + len(turns))
        flat_texts.extend(t.text for t in turns)
    all_vecs = ts.embed_texts_cached(episode, flat_texts, encoder, cache_dir)

    def propose(scene: Scene) -> list[float]:
        turns = by_scene.get(scene.scene_id, [])
        lo, hi = index.get(scene.scene_id, (0, 0))
        vecs = all_vecs[lo:hi]
        return ts.propose_topic_boundaries(
            turns, vecs, w=params["w"], tau_depth=params["tau_depth"],
            min_spacing=params["min_spacing"],
        )
    return propose


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["char", "topic", "hybrid"])
    ap.add_argument("--episodes", default="ALL",
                    help="ALL | sN | sN-sM | comma-list (e.g. s01e01a,s02e03b)")
    ap.add_argument("--scenes-in", default=str(DEFAULT_SCENES_IN))
    ap.add_argument("--scenes-out", required=True)
    ap.add_argument("--sentences-in", default=str(DEFAULT_SENTENCES_IN))
    ap.add_argument("--char-tracker-dir", default=None)
    ap.add_argument("--shots-dir", default=str(cp.DEFAULT_SHOTS_DIR))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--w", type=int, default=2)
    ap.add_argument("--tau-depth", type=float, default=0.3)
    ap.add_argument("--min-spacing", type=float, default=20.0)
    ap.add_argument("--eps", type=float, default=HYBRID_EPS)
    args = ap.parse_args()

    scenes_in = Path(args.scenes_in)
    scenes_out = Path(args.scenes_out)
    sentences_in = Path(args.sentences_in)
    shots_dir = Path(args.shots_dir)
    ct_dir = resolve_char_tracker_dir(args.char_tracker_dir) or Path(DEFAULT_CHAR_TRACKER_DIR)
    params = {"w": args.w, "tau_depth": args.tau_depth, "min_spacing": args.min_spacing}
    aug_tag = {"char": "char_aug", "topic": "topic_aug", "hybrid": "hybrid_aug"}[args.mode]

    episodes = expand_episode_spec(args.episodes, scenes_in)
    encoder = ts.minilm_encoder() if args.mode in ("topic", "hybrid") else None

    print(f"Augmenting {len(episodes)} eps | mode={args.mode} → {scenes_out}")
    totals = {"in": 0, "out": 0, "new": 0, "skipped": 0}
    for ep in episodes:
        char_propose = _build_char_propose(ep, ct_dir, shots_dir, params) if args.mode in ("char", "hybrid") else None
        topic_propose = _build_topic_propose(ep, sentences_in, encoder, Path(args.cache_dir), params) if args.mode in ("topic", "hybrid") else None

        if args.mode == "char":
            propose = char_propose
        elif args.mode == "topic":
            propose = topic_propose
        else:  # hybrid
            if char_propose is None or topic_propose is None:
                propose = None
            else:
                def propose(scene: Scene, _c=char_propose, _t=topic_propose) -> list[float]:
                    return ts.intersect_within(_c(scene), _t(scene), eps=args.eps)

        if propose is None:
            # missing inputs → copy through with no boundaries
            propose = lambda scene: []  # noqa: E731
            totals["skipped"] += 1

        r = subdivide_episode(ep, scenes_in, scenes_out, propose, aug_tag=aug_tag)
        totals["in"] += r["n_input_scenes"]
        totals["out"] += r["n_output_scenes"]
        totals["new"] += r["n_new_boundaries"]
        print(f"  {ep}: {r['n_input_scenes']:>3} → {r['n_output_scenes']:>3} (+{r['n_new_boundaries']})")

    print(f"\nTotals: {totals['in']} → {totals['out']} (+{totals['new']}); "
          f"{totals['skipped']} eps missing inputs")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Integration smoke test — topic mode on one episode**

Run: `cd /orcd/home/002/yibei/te-charnet && python scripts/augment_scenes.py --mode topic --episodes s01e01a --scenes-out /tmp/topic_smoke --cache-dir /tmp/emb_cache`
Expected: prints `s01e01a: N → M (+k)` (first run downloads MiniLM, ~1 min); `/tmp/topic_smoke/s1/friends_s01e01a_scene_summary.tsv` exists and `M >= N`. A second run is fast (embedding cache hit).

- [ ] **Step 3: Integration smoke test — hybrid mode on one episode**

Run: `cd /orcd/home/002/yibei/te-charnet && python scripts/augment_scenes.py --mode hybrid --episodes s01e01a --scenes-out /tmp/hybrid_smoke --cache-dir /tmp/emb_cache`
Expected: runs without error; `+k` for hybrid is ≤ the topic-mode `+k` (intersection is a subset).

- [ ] **Step 4: Run the full test suite + lint**

Run: `cd /orcd/home/002/yibei/te-charnet/src && python -m pytest && ruff check ..`
Expected: all tests pass; ruff clean (fix any lint in the new files).

- [ ] **Step 5: Commit**

```bash
git add scripts/augment_scenes.py
git commit -m "Unified augment_scenes.py CLI: char / topic / hybrid modes"
```

---

## Task 9: Calibrate topic-shift on s1–s2

**Files:**
- Create: `scripts/calibrate_topic_shift.py`

Grid-search W × τ_depth × M on the calibration split, scoring each combo with the existing evaluator, and report the grid + the best combo by **segment F1@5s**. char params and ε stay frozen.

- [ ] **Step 1: Implement the calibration driver**

```python
# scripts/calibrate_topic_shift.py
"""Grid-search topic-shift params on the s1-s2 calibration split.

For each (W, tau_depth, min_spacing) combo: augment s1-s2 in topic mode to a
temp tree, run evaluate_scene_segmentation.py against it, parse aggregate.json,
and record segment F1@5s. Prints a sorted grid and the best combo.
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

W_GRID = [1, 2, 3]
TAU_GRID = [0.2, 0.3, 0.4, 0.5]
SPACING_GRID = [15.0, 20.0, 30.0]
CALIB_SPEC = "s1-s2"


def _seg_f1_at_5s(agg_path: Path) -> float:
    agg = json.loads(agg_path.read_text())
    # aggregate.json stores by-unit means; read Segment F1@5s.
    # Structure: agg["by_unit"]["segment"]["f1@5s"] (confirm against the file).
    return float(agg["by_unit"]["segment"]["f1@5s"])


def main() -> None:
    results = []
    for w, tau, spacing in itertools.product(W_GRID, TAU_GRID, SPACING_GRID):
        with tempfile.TemporaryDirectory() as tmp:
            scenes_out = Path(tmp) / "scenes"
            eval_out = Path(tmp) / "eval"
            subprocess.run([
                sys.executable, str(REPO / "scripts/augment_scenes.py"),
                "--mode", "topic", "--episodes", CALIB_SPEC,
                "--scenes-out", str(scenes_out),
                "--w", str(w), "--tau-depth", str(tau), "--min-spacing", str(spacing),
            ], check=True)
            subprocess.run([
                sys.executable, str(REPO / "scripts/evaluate_scene_segmentation.py"),
                "--episodes", CALIB_SPEC,
                "--ours-dir", str(scenes_out), "--out-dir", str(eval_out),
            ], check=True)
            f1 = _seg_f1_at_5s(eval_out / "aggregate.json")
            results.append((f1, w, tau, spacing))
            print(f"W={w} tau={tau} spacing={spacing} -> seg_F1@5s={f1:.4f}")

    results.sort(reverse=True)
    print("\nTop 5 combos (segment F1@5s):")
    for f1, w, tau, spacing in results[:5]:
        print(f"  {f1:.4f}  W={w} tau_depth={tau} min_spacing={spacing}")
    best = results[0]
    print(f"\nBEST: W={best[1]} tau_depth={best[2]} min_spacing={best[3]} (F1@5s={best[0]:.4f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm the aggregate.json key path before running the grid**

Run: `cd /orcd/home/002/yibei/te-charnet && python scripts/evaluate_scene_segmentation.py --episodes s01e01a --ours-dir output/annotations/scenes --out-dir /tmp/agg_probe && python -c "import json; print(list(json.load(open('/tmp/agg_probe/aggregate.json')).keys()))"`
Expected: prints the top-level keys. Inspect `/tmp/agg_probe/aggregate.json` and adjust `_seg_f1_at_5s` in `calibrate_topic_shift.py` to the actual key path for segment F1@5s (the placeholder `agg["by_unit"]["segment"]["f1@5s"]` must match the real structure — fix it now if it differs).

- [ ] **Step 3: Run the calibration grid (caches embeddings on first combo)**

Run: `cd /orcd/home/002/yibei/te-charnet && python scripts/calibrate_topic_shift.py`
Expected: 36 lines (3×4×3) then a Top-5 and a BEST line. First combo encodes s1–s2 turns (~1–2 min); later combos reuse the cache. **Record the BEST W, τ_depth, M** for Task 10.

- [ ] **Step 4: Commit**

```bash
git add scripts/calibrate_topic_shift.py
git commit -m "Calibration driver: grid-search topic-shift params on s1-s2"
```

---

## Task 10: Final sweep on s3–s6 + write up Prototype #2

**Files:**
- Modify: `docs/scene_segmentation_evaluation.md`

Run all four configs on the held-out test split with the calibrated params and record the result. `<W*>`, `<TAU*>`, `<M*>` are the BEST values from Task 9.

- [ ] **Step 1: Generate the four config trees on s3–s6**

```bash
cd /orcd/home/002/yibei/te-charnet
# baseline = the existing scenes tree, no augmentation (eval reads it directly)
python scripts/augment_scenes.py --mode char  --episodes s3-s6 --scenes-out output/annotations/scenes_char
python scripts/augment_scenes.py --mode topic --episodes s3-s6 --scenes-out output/annotations/scenes_topic \
    --w <W*> --tau-depth <TAU*> --min-spacing <M*>
python scripts/augment_scenes.py --mode hybrid --episodes s3-s6 --scenes-out output/annotations/scenes_hybrid \
    --w <W*> --tau-depth <TAU*> --min-spacing <M*>
```
Expected: each prints per-episode `+k` lines and a totals line over the 196 s3–s6 half-episodes.

- [ ] **Step 2: Score all four configs on s3–s6**

```bash
cd /orcd/home/002/yibei/te-charnet
python scripts/evaluate_scene_segmentation.py --episodes s3-s6 --ours-dir output/annotations/scenes        --out-dir output/evaluation/scene_seg_test_baseline
python scripts/evaluate_scene_segmentation.py --episodes s3-s6 --ours-dir output/annotations/scenes_char   --out-dir output/evaluation/scene_seg_test_char
python scripts/evaluate_scene_segmentation.py --episodes s3-s6 --ours-dir output/annotations/scenes_topic  --out-dir output/evaluation/scene_seg_test_topic
python scripts/evaluate_scene_segmentation.py --episodes s3-s6 --ours-dir output/annotations/scenes_hybrid --out-dir output/evaluation/scene_seg_test_hybrid
```
Expected: four `aggregate.json` + `boundary_diagnostics.tsv` written.

- [ ] **Step 3: Collect the comparison numbers**

Run: `cd /orcd/home/002/yibei/te-charnet && for c in baseline char topic hybrid; do echo "== $c =="; python -c "import json; a=json.load(open(f'output/evaluation/scene_seg_test_$c/aggregate.json')); print(json.dumps(a.get('by_unit', a), indent=2))"; done`
Expected: prints each config's by-unit metrics. Read **segment F1@5s, P@5s, R@5s, scene F1@5s** for all four, and the `charact_entry`/`charact_leave`/`goal_change` rows from each `boundary_diagnostics.tsv` aggregate (the `diagnostics` block in `aggregate.json`).

- [ ] **Step 4: Write the "Prototype #2 results" section**

Append a `## Prototype #2 results: topic-shift subdivision + char×topic hybrid` section to `docs/scene_segmentation_evaluation.md`, mirroring the Prototype #1 section's honest style. Include:
- Implementation summary (ct-turn unit, MiniLM, block depth-score, ε=3 s hybrid).
- Calibrated params (W*, τ_depth*, M*) and the s1–s2 / s3–s6 split note.
- A results table: baseline / char / topic / hybrid × (segment F1@5s, P@5s, R@5s, scene F1@5s, charact_entry, charact_leave, goal_change).
- Interpretation answering the two spec questions: (1) is `topic` an independent net-positive signal? (2) does `hybrid` recover the precision `char` lost? Quote the precision-of-new-boundaries figure if computable.
- Honest assessment + implication for the ranking, as with Prototype #1.

Verify every number against the JSON/TSV it came from (use the results-verification discipline — quote source files).

- [ ] **Step 5: Commit the write-up and the four output trees**

```bash
cd /orcd/home/002/yibei/te-charnet
git add docs/scene_segmentation_evaluation.md
git add output/annotations/scenes_char output/annotations/scenes_topic output/annotations/scenes_hybrid
git add output/evaluation/scene_seg_test_baseline output/evaluation/scene_seg_test_char output/evaluation/scene_seg_test_topic output/evaluation/scene_seg_test_hybrid
git status   # confirm no embedding cache or stray files staged (cache lives in output/intermediate/)
git commit -m "Prototype #2: topic-shift + char×topic hybrid — results vs manual (s3-s6)"
```

(If `output/**` gitignore rules block these, mirror the negation pattern already used for the prototype-#1 trees — recall the `output/**` directory-negation gotcha; verify with `git add -n` before committing.)

---

## Self-review

- **Spec coverage:** base unit (Task 4) ✓; embedding + cache (Task 6) ✓; block depth-score detection (Task 5) ✓; hybrid ε=3 s (Tasks 7–8) ✓; four-config matrix (Tasks 8, 10) ✓; calibration season split (Tasks 9–10) ✓; architecture refactor (Tasks 2–3) ✓; testing plan (Tasks 2,3,4,5,6,7) ✓; write-up as Prototype #2 (Task 10) ✓.
- **Frozen-params guarantee:** char proposer params and ε are hard-coded constants in `char_presence.py` / passed explicitly; only W, τ_depth, M are swept (Task 9). ✓
- **No-selection-bias guarantee:** calibration runs only on `s1-s2` (Task 9); headline scored only on `s3-s6` (Task 10). ✓
- **Known follow-up at execution time:** confirm the real `aggregate.json` key path for segment F1@5s (Task 9 Step 2) before trusting the grid — the `_seg_f1_at_5s` accessor is the one place a structural assumption must be checked against the live file.
- **Output-tracking caveat:** Task 10 Step 5 may hit the `output/**` gitignore directory-negation gotcha; the step flags the `git add -n` verification.
```
