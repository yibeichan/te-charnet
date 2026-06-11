# Network-metric Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the character-interaction network's per-scene structural metrics and per-character centrality as two timestamped TSVs with BIDS-inspired sidecars, recomputed from stage-2 `temporal_network.json`.

**Architecture:** A small, fully-tested `src/charnet/network_export.py` owns stable schemas, column order, empty-frame behavior, and measure validation, wrapping the existing `metrics.scene_metrics` / `metrics.centrality_timeseries`. A thin `scripts/export_network_metrics.py` does argument parsing, episode discovery, per-episode `temporal_network.json` resolution (with the `friends_<ep>` vs bare-`<ep>` fix), and writes TSVs + sidecars via `charnet.bids_meta`.

**Tech Stack:** Python ≥3.10, pandas, networkx (transitively via `metrics`), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-08-network-metric-export-design.md`

---

## File Structure

- **Create** `src/charnet/network_export.py` — column constants, two DataFrame builders, two data-dictionary dicts. Pure; no I/O.
- **Create** `scripts/export_network_metrics.py` — CLI: parse args, discover episodes, resolve network paths, write TSVs + sidecars. I/O only; no metric logic.
- **Create** `tests/test_network_export.py` — unit tests for the builders.
- **Create** `tests/test_export_network_metrics.py` — script-level end-to-end tests.

Reused as-is (do not modify): `src/charnet/metrics.py` (`scene_metrics`, `centrality_timeseries`, `SUPPORTED_CENTRALITY_MEASURES`), `src/charnet/io.py` (`load_temporal_network`), `src/charnet/models.py` (`SceneGraph`, `EdgeData`), `src/charnet/bids_meta.py`, `src/charnet/transcript_align.py` (`normalize_episode_key`), `src/charnet/scene_subdivide.py` (`expand_episode_spec`).

---

## Task 1: Scene-network builder + column constants

**Files:**
- Create: `src/charnet/network_export.py`
- Test: `tests/test_network_export.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_network_export.py
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from charnet.models import EdgeData, SceneGraph  # noqa: E402
from charnet import network_export as nx_exp  # noqa: E402


def _scene(scene_id, start, end):
    return SceneGraph(
        scene_id=scene_id, start=start, end=end,
        nodes=["A", "B", "C"],
        edges=[
            EdgeData(source="A", target="B", weight=2.0, adjacency=1.0, proximity=0.0),
            EdgeData(source="B", target="C", weight=1.0, adjacency=1.0, proximity=0.0),
        ],
    )


def test_scene_network_trace_columns_and_rows():
    df = nx_exp.scene_network_trace([_scene(1, 0.0, 10.0), _scene(2, 10.0, 20.0)])
    assert list(df.columns) == nx_exp.SCENE_NETWORK_COLUMNS
    assert len(df) == 2
    row = df.iloc[0]
    assert row["scene_id"] == 1
    assert row["start"] == 0.0 and row["end"] == 10.0
    assert row["n_nodes"] == 3 and row["n_edges"] == 2
    assert row["n_components"] == 1


def test_scene_network_trace_empty_keeps_columns():
    df = nx_exp.scene_network_trace([])
    assert list(df.columns) == nx_exp.SCENE_NETWORK_COLUMNS
    assert len(df) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest ../tests/test_network_export.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'charnet.network_export'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/charnet/network_export.py
"""Builders for the network-metric brain export (per-scene + per-character).

Pure assembly over charnet.metrics — no metric logic, no I/O. Owns the stable
column schemas, empty-frame behavior, and measure validation that the export
script and its consumers rely on.
"""
from __future__ import annotations

import pandas as pd

from charnet import metrics
from charnet.models import SceneGraph

SCENE_NETWORK_COLUMNS = [
    "scene_id",
    "start",
    "end",
    "duration",
    "n_nodes",
    "n_edges",
    "density",
    "n_components",
    "n_interaction_edges",
    "interaction_density",
    "interaction_entropy",
]


def scene_network_trace(scene_graphs: list[SceneGraph]) -> pd.DataFrame:
    """One row per scene of structural network metrics, timestamped.

    Wraps ``metrics.scene_metrics``. ``start``/``end`` are stage-2
    network-coverage windows (speaker-bearing rows), not necessarily full
    01a scene spans. Returns an empty frame carrying SCENE_NETWORK_COLUMNS
    when ``scene_graphs`` is empty.
    """
    rows = [metrics.scene_metrics(sg) for sg in scene_graphs]
    df = pd.DataFrame(rows, columns=SCENE_NETWORK_COLUMNS)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src && python -m pytest ../tests/test_network_export.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/charnet/network_export.py tests/test_network_export.py
git commit -m "Add scene_network_trace: per-scene structural metric builder"
```

---

## Task 2: Character-centrality builder + measure validation

**Files:**
- Modify: `src/charnet/network_export.py`
- Test: `tests/test_network_export.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_network_export.py`)

```python
import pytest  # noqa: E402


def test_character_centrality_trace_columns_and_rows():
    df = nx_exp.character_centrality_trace([_scene(1, 0.0, 10.0)], measures=["degree"])
    assert list(df.columns) == ["scene_id", "start", "end", "character", "degree"]
    assert sorted(df["character"]) == ["A", "B", "C"]
    assert (df["scene_id"] == 1).all()


def test_character_centrality_trace_multi_measure_order():
    df = nx_exp.character_centrality_trace([_scene(1, 0.0, 10.0)],
                                           measures=["betweenness", "degree"])
    assert list(df.columns) == ["scene_id", "start", "end", "character",
                                "betweenness", "degree"]


def test_character_centrality_trace_empty_keeps_columns():
    df = nx_exp.character_centrality_trace([], measures=["degree", "eigenvector"])
    assert list(df.columns) == ["scene_id", "start", "end", "character",
                                "degree", "eigenvector"]
    assert len(df) == 0


def test_character_centrality_trace_rejects_unknown_measure():
    with pytest.raises(ValueError, match="unknown"):
        nx_exp.character_centrality_trace([_scene(1, 0.0, 10.0)], measures=["bogus"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest ../tests/test_network_export.py -v`
Expected: FAIL — `AttributeError: module 'charnet.network_export' has no attribute 'character_centrality_trace'`.

- [ ] **Step 3: Write minimal implementation** (append to `src/charnet/network_export.py`)

```python
CHARACTER_CENTRALITY_BASE_COLUMNS = ["scene_id", "start", "end", "character"]


def character_centrality_trace(
    scene_graphs: list[SceneGraph],
    measures: list[str],
) -> pd.DataFrame:
    """Per-scene x character centrality, timestamped, stable column order.

    Wraps ``metrics.centrality_timeseries``. Validates ``measures`` against
    ``metrics.SUPPORTED_CENTRALITY_MEASURES`` and raises ValueError on an
    unknown measure (``compute_centralities`` only logs). Returns an empty
    frame carrying the declared columns when there are no rows.
    """
    unknown = [m for m in measures if m not in metrics.SUPPORTED_CENTRALITY_MEASURES]
    if unknown:
        raise ValueError(
            f"unknown centrality measure(s): {', '.join(unknown)}. "
            f"Supported: {', '.join(metrics.SUPPORTED_CENTRALITY_MEASURES)}"
        )
    columns = CHARACTER_CENTRALITY_BASE_COLUMNS + list(measures)
    df = metrics.centrality_timeseries(scene_graphs, measures=measures)
    if df.empty:
        return pd.DataFrame(columns=columns)
    return df.reindex(columns=columns)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src && python -m pytest ../tests/test_network_export.py -v`
Expected: PASS (all six tests).

- [ ] **Step 5: Commit**

```bash
git add src/charnet/network_export.py tests/test_network_export.py
git commit -m "Add character_centrality_trace with measure validation + stable schema"
```

---

## Task 3: Data-dictionary sidecar definitions

**Files:**
- Modify: `src/charnet/network_export.py`
- Test: `tests/test_network_export.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_network_export.py`)

```python
def test_data_dictionaries_cover_columns():
    # every non-id column is described; coverage guards silent schema drift
    for col in nx_exp.SCENE_NETWORK_COLUMNS:
        assert col in nx_exp.SCENE_NETWORK_DD
    for col in nx_exp.CHARACTER_CENTRALITY_BASE_COLUMNS:
        assert col in nx_exp.CHARACTER_CENTRALITY_DD
    # start/end documented as network-coverage windows, not full scene spans
    assert "coverage" in nx_exp.SCENE_NETWORK_DD["start"]["Description"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest ../tests/test_network_export.py::test_data_dictionaries_cover_columns -v`
Expected: FAIL — `AttributeError: ... has no attribute 'SCENE_NETWORK_DD'`.

- [ ] **Step 3: Write minimal implementation** (append to `src/charnet/network_export.py`)

```python
_COVERAGE_NOTE = (
    "Stage-2 network-coverage window (from speaker-bearing rows), not "
    "necessarily the full 01a scene span. Mapping to fMRI run time / TRs is "
    "the consumer's responsibility."
)

SCENE_NETWORK_DD = {
    "scene_id": {"Description": "Scene index within the episode."},
    "start": {"Description": f"Scene start. {_COVERAGE_NOTE}", "Units": "s"},
    "end": {"Description": f"Scene end. {_COVERAGE_NOTE}", "Units": "s"},
    "duration": {"Description": "end - start.", "Units": "s"},
    "n_nodes": {"Description": "Characters present in the scene interaction graph."},
    "n_edges": {"Description": "Edges (character pairs) in the scene graph."},
    "density": {"Description": "networkx graph density of the scene graph (0-1)."},
    "n_components": {"Description": "Connected components in the scene graph."},
    "n_interaction_edges": {"Description": "Edges with nonzero adjacency or proximity (true interaction, not pure co-presence)."},
    "interaction_density": {"Description": "n_interaction_edges / possible edges (0-1)."},
    "interaction_entropy": {"Description": "Shannon entropy (bits) of the scene's edge-weight distribution; higher = speech/interaction more evenly spread.", "Units": "bits"},
}

CHARACTER_CENTRALITY_DD = {
    "scene_id": {"Description": "Scene index within the episode."},
    "start": {"Description": f"Scene start. {_COVERAGE_NOTE}", "Units": "s"},
    "end": {"Description": f"Scene end. {_COVERAGE_NOTE}", "Units": "s"},
    "character": {"Description": "Character (node) the centrality row describes."},
    "degree": {"Description": "Weighted degree centrality (node strength share, 0-1, sums to 1 across nodes in a scene)."},
    "degree_unweighted": {"Description": "Unweighted networkx degree centrality (singleton scene = 0)."},
    "degree_weighted": {"Description": "Alias of degree (weighted strength share)."},
    "strength": {"Description": "Alias of degree (weighted strength share)."},
    "betweenness": {"Description": "Betweenness centrality on inverse-weight distances (stronger ties = shorter paths)."},
    "eigenvector": {"Description": "Eigenvector centrality (weighted); falls back to degree centrality on non-convergence."},
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src && python -m pytest ../tests/test_network_export.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add src/charnet/network_export.py tests/test_network_export.py
git commit -m "Add data-dictionary sidecar definitions for network export"
```

---

## Task 4: Episode → network-path resolution (the friends_<ep> fix)

**Files:**
- Create: `scripts/export_network_metrics.py`
- Test: `tests/test_export_network_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export_network_metrics.py
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import export_network_metrics as E  # noqa: E402
from charnet.io import save_temporal_network  # noqa: E402
from charnet.models import EdgeData, SceneGraph  # noqa: E402


def _write_network(network_root: Path, dirname: str, ep_scene_id=1):
    sg = SceneGraph(
        scene_id=ep_scene_id, start=0.0, end=10.0,
        nodes=["A", "B", "C"],
        edges=[
            EdgeData(source="A", target="B", weight=2.0, adjacency=1.0, proximity=0.0),
            EdgeData(source="B", target="C", weight=1.0, adjacency=1.0, proximity=0.0),
        ],
    )
    path = network_root / dirname / "temporal_network.json"
    save_temporal_network([sg], path)
    return path


def test_resolve_network_path_prefers_friends_prefix(tmp_path):
    root = tmp_path / "02_build_network"
    _write_network(root, "friends_s01e01a")
    resolved = E.resolve_network_path(root, "s01e01a")
    assert resolved == root / "friends_s01e01a" / "temporal_network.json"


def test_resolve_network_path_falls_back_to_bare(tmp_path):
    root = tmp_path / "02_build_network"
    _write_network(root, "s01e01a")
    resolved = E.resolve_network_path(root, "s01e01a")
    assert resolved == root / "s01e01a" / "temporal_network.json"


def test_resolve_network_path_missing_returns_none(tmp_path):
    root = tmp_path / "02_build_network"
    assert E.resolve_network_path(root, "s09e99z") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest ../tests/test_export_network_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'export_network_metrics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/export_network_metrics.py
"""Export per-scene network metrics + per-character centrality as timestamped TSVs.

  python scripts/export_network_metrics.py --episodes s3-s6 \
      --network-root "$SCRATCH_DIR/output/02_build_network" \
      --out-dir output/annotations/network_metrics
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from charnet import network_export as nx_exp  # noqa: E402
from charnet.bids_meta import write_data_dictionary, write_dataset_description  # noqa: E402
from charnet.io import load_temporal_network  # noqa: E402
from charnet.scene_subdivide import expand_episode_spec  # noqa: E402
from charnet.transcript_align import normalize_episode_key  # noqa: E402

SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")
DEFAULT_SCENES_IN = REPO / "output/annotations/scenes"
DEFAULT_NETWORK_ROOT = Path(SCRATCH_DIR) / "output" / "02_build_network"
DEFAULT_OUT_DIR = REPO / "output/annotations/network_metrics"


def resolve_network_path(network_root: Path, episode: str) -> Path | None:
    """Locate an episode's temporal_network.json.

    Stage-2 dirs from run_pipeline.py use the friends_-prefixed key
    (normalize_episode_key), while expand_episode_spec yields bare IDs.
    Probe the normalized name first, then the bare ID, and return whichever
    exists (else None).
    """
    candidates = [normalize_episode_key(episode), episode]
    seen = []
    for name in candidates:
        if name in seen:
            continue
        seen.append(name)
        path = network_root / name / "temporal_network.json"
        if path.exists():
            return path
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src && python -m pytest ../tests/test_export_network_metrics.py -v`
Expected: PASS (three resolution tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/export_network_metrics.py tests/test_export_network_metrics.py
git commit -m "Add export_network_metrics path resolution (friends_<ep> stage-2 dir fix)"
```

---

## Task 5: Script `main()` — write TSVs + sidecars, missing-input behavior

**Files:**
- Modify: `scripts/export_network_metrics.py`
- Test: `tests/test_export_network_metrics.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_export_network_metrics.py`)

```python
def _make_scenes_dir(tmp_path):
    # so expand_episode_spec can resolve the episode id
    scenes = tmp_path / "scenes" / "s1"
    scenes.mkdir(parents=True)
    (scenes / "friends_s01e01a_scene_summary.tsv").write_text("scene_id\tstart\tend\n")
    return tmp_path / "scenes"


def _run_main(monkeypatch, tmp_path, episodes, network_root, out_dir):
    scenes_in = _make_scenes_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "export_network_metrics.py", "--episodes", episodes,
        "--scenes-in", str(scenes_in),
        "--network-root", str(network_root),
        "--out-dir", str(out_dir),
        "--measures", "degree,betweenness",
    ])
    E.main()


def test_main_writes_tsvs_and_sidecars(tmp_path, monkeypatch):
    root = tmp_path / "02_build_network"
    _write_network(root, "friends_s01e01a")
    out_dir = tmp_path / "network_metrics"
    _run_main(monkeypatch, tmp_path, "s01e01a", root, out_dir)

    scene_tsv = out_dir / "s1" / "friends_s01e01a_scene_network.tsv"
    char_tsv = out_dir / "s1" / "friends_s01e01a_character_centrality.tsv"
    assert scene_tsv.exists() and char_tsv.exists()

    sdf = pd.read_csv(scene_tsv, sep="\t")
    assert list(sdf.columns) == nx_exp.SCENE_NETWORK_COLUMNS
    cdf = pd.read_csv(char_tsv, sep="\t")
    assert list(cdf.columns) == ["scene_id", "start", "end", "character",
                                 "degree", "betweenness"]

    assert (out_dir / "scene_network.json").exists()
    assert (out_dir / "character_centrality.json").exists()
    desc = json.loads((out_dir.parent / "dataset_description.json").read_text())
    assert desc["DatasetType"] == "derivative"


def test_main_explicit_missing_episode_errors(tmp_path, monkeypatch):
    root = tmp_path / "02_build_network"  # empty: no episode dirs
    out_dir = tmp_path / "network_metrics"
    # episode resolvable by expand_episode_spec but no network dir present
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, tmp_path, "s01e01a", root, out_dir)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest ../tests/test_export_network_metrics.py -v`
Expected: FAIL — `AttributeError: module 'export_network_metrics' has no attribute 'main'`.

- [ ] **Step 3: Write minimal implementation** (append to `scripts/export_network_metrics.py`)

```python
def _git_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "describe", "--tags", "--always", "--dirty"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _is_explicit_list(spec: str) -> bool:
    """True when --episodes names specific episodes (vs ALL / season / range)."""
    spec = spec.strip()
    if spec == "ALL":
        return False
    return re.fullmatch(r"s\d+(-s\d+)?", spec) is None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="ALL", help="ALL | sN | sN-sM | comma-list")
    ap.add_argument("--scenes-in", default=str(DEFAULT_SCENES_IN),
                    help="root used only to resolve episode specs")
    ap.add_argument("--network-root", default=str(DEFAULT_NETWORK_ROOT),
                    help="root holding <ep>/temporal_network.json (stage-2 output)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--measures", default="degree,betweenness,eigenvector")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    network_root = Path(args.network_root)
    measures = [m.strip().lower() for m in args.measures.split(",") if m.strip()]
    episodes = expand_episode_spec(args.episodes, Path(args.scenes_in))
    explicit = _is_explicit_list(args.episodes)

    # write schema sidecars first so the dir is self-describing on partial runs
    write_data_dictionary(out_dir / "scene_network.json", nx_exp.SCENE_NETWORK_DD)
    write_data_dictionary(out_dir / "character_centrality.json", nx_exp.CHARACTER_CENTRALITY_DD)
    write_dataset_description(
        out_dir.parent / "dataset_description.json",
        name="charnet Friends stimulus annotations",
        version=_git_version(),
        source_datasets=[{"Description": "Courtois NeuroMod Friends fMRI stimulus episodes"}],
    )

    print(f"Exporting network metrics for {len(episodes)} eps → {out_dir}")
    n_written = n_skipped = 0
    missing = []
    for ep in episodes:
        npath = resolve_network_path(network_root, ep)
        if npath is None:
            missing.append(ep)
            n_skipped += 1
            continue
        scene_graphs = load_temporal_network(npath)
        scene_df = nx_exp.scene_network_trace(scene_graphs)
        char_df = nx_exp.character_centrality_trace(scene_graphs, measures=measures)
        season = int(ep[1:3])
        ep_dir = out_dir / f"s{season}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        scene_df.to_csv(ep_dir / f"friends_{ep}_scene_network.tsv", sep="\t", index=False)
        char_df.to_csv(ep_dir / f"friends_{ep}_character_centrality.tsv", sep="\t", index=False)
        n_written += 1
        print(f"  {ep}: {len(scene_df)} scenes, {len(char_df)} character-rows")

    print(f"\nWrote {n_written} episodes ({n_skipped} missing network dirs)")

    if explicit and missing:
        sys.exit(f"error: no temporal_network.json for explicitly-named episode(s): "
                 f"{', '.join(missing)} (checked under {network_root})")
    if n_written == 0:
        sys.exit(f"error: 0 episodes written — check --network-root ({network_root}) "
                 f"and SCRATCH_DIR")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src && python -m pytest ../tests/test_export_network_metrics.py -v`
Expected: PASS (all five tests — three resolution + two main).

- [ ] **Step 5: Commit**

```bash
git add scripts/export_network_metrics.py tests/test_export_network_metrics.py
git commit -m "Add export_network_metrics main(): TSVs + sidecars + missing-input guards"
```

---

## Task 6: Full suite + lint, then docs

**Files:**
- Modify: `docs/data_products_catalog.md`

- [ ] **Step 1: Run the full test suite**

Run: `cd src && python -m pytest`
Expected: PASS — all prior tests (113) plus the new network-export tests.

- [ ] **Step 2: Lint**

Run: `cd src && ruff check .`
Expected: clean (no findings).

- [ ] **Step 3: Update the data-products catalog**

In `docs/data_products_catalog.md`, update the "Gaps & planned exports" section: move item 2 (Network-metric features) from *planned* to *shipped*, naming the new product path `output/annotations/network_metrics/` and the `export_network_metrics.py` producer, mirroring how the topic-shift trace entry reads now. Add a row to the product-inventory table for the two new TSVs with `Timestamped? = yes` and `Brain-status = exported`.

- [ ] **Step 4: Commit**

```bash
git add docs/data_products_catalog.md
git commit -m "Mark network-metric export shipped in data-products catalog"
```

---

## Self-Review

**Spec coverage:**
- Two products, one dir → Tasks 1, 2, 5 (two TSVs under `network_metrics/`). ✓
- Recompute from `temporal_network.json` → Task 5 (`load_temporal_network`). ✓
- `network_export.py` tiny, schemas/validation/empty-frames → Tasks 1-3. ✓
- `n_components` included → Task 1 (in `SCENE_NETWORK_COLUMNS` + asserted). ✓
- Empty-frame stable schemas → Tasks 1, 2 (explicit tests). ✓
- Measure validation raises → Task 2. ✓
- Episode-ID `friends_<ep>` fix + probe-both → Task 4. ✓
- `--network-root` default, measures default → Task 5. ✓
- Missing-input: skip-with-count for ALL/season, error for explicit, exit-nonzero on zero → Task 5. ✓
- Sidecars: two product-level + `dataset_description.json` at parent, written first, git provenance → Task 5. ✓
- Coverage-window caveat in data dictionary → Task 3 (asserted). ✓
- Output path layout → Task 5. ✓
- Tests (unit + script) → Tasks 1-5. ✓
- Docs update → Task 6. ✓

**Placeholder scan:** none — every code step shows complete code; the docs step (Task 6 Step 3) is descriptive prose for an existing-file edit, not code.

**Type consistency:** `scene_network_trace(scene_graphs)`, `character_centrality_trace(scene_graphs, measures)`, `resolve_network_path(network_root, episode)`, `SCENE_NETWORK_COLUMNS`, `CHARACTER_CENTRALITY_BASE_COLUMNS`, `SCENE_NETWORK_DD`, `CHARACTER_CENTRALITY_DD` — names used identically across builder definitions, sidecar writes, and tests.
