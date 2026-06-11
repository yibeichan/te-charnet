# Network-metric export (brain feature #2) — design

**Date:** 2026-06-08
**Status:** approved (design); pending implementation plan
**Feature dir:** brain-feature exports (follows brain feature #1, topic-shift trace)

## Goal

Persist the character-interaction network's time-resolved metrics as a
documented, timestamped data product for the downstream `brain-states-friends`
`08a` pipeline (TR-aligned feature arrays). This fills the **"social structure /
network metrics over time"** gap in `docs/data_products_catalog.md`, where today
only `n_scene_speakers` / `n_main_in_scene` reach the brain analysis.

The metrics already exist in `src/charnet/metrics.py` (`scene_metrics`,
`centrality_timeseries`) and are written by stage 3 (`03_analyze.py`) as
`centrality_timeseries.csv` + `metrics.json` under a per-run pipeline dir. This
feature does **not** invent new metrics — it produces a *brain-ready,
self-describing export* of them following the brain feature #1 contract
(timestamped TSV + BIDS-inspired JSON sidecars).

## Locked decisions

1. **Two products, one dir.** Emit both a per-scene structural summary TSV and a
   per-character centrality TSV under one product dir, each with its own
   data-dictionary sidecar. They are one conceptual derivative at two grains.
2. **Recompute from stage-2 output.** Read `temporal_network.json` (stage 2) and
   recompute via `scene_metrics()` + `centrality_timeseries()` inside the export,
   rather than reformatting stage-3's `centrality_timeseries.csv`. Self-contained,
   mirrors topic-trace's compute-from-source pattern, depends only on stage 2.
   Verified self-contained: `load_temporal_network()` reconstructs `SceneGraph`
   objects with `scene_id/start/end/nodes/edges`; both metric functions need only
   the scene graphs and requested measures — no sentence tables.

## Architecture

Mirror the brain feature #1 split: pure tested logic in `src/charnet/`, thin I/O
script in `scripts/`, tests against the src module.

### `src/charnet/network_export.py` (new, tested)

Tiny. Its value is **stable schemas, column order, empty-frame behavior, and
measure validation** — not metric logic (which stays in `metrics.py`).

- `SCENE_NETWORK_COLUMNS` / `CHARACTER_CENTRALITY_BASE_COLUMNS` — declared column
  orders.
- `scene_network_trace(scene_graphs) -> pd.DataFrame` — one row per scene with
  columns:
  `scene_id, start, end, duration, n_nodes, n_edges, density, n_components,
  n_interaction_edges, interaction_density, interaction_entropy`.
  Wraps `scene_metrics()` per scene. Returns an empty DataFrame **with these
  columns** when `scene_graphs` is empty.
- `character_centrality_trace(scene_graphs, measures) -> pd.DataFrame` — wraps
  `centrality_timeseries()`. Columns: `scene_id, start, end, character, <measures…>`
  in a stable order. Returns an empty DataFrame **with the declared columns** when
  there are no rows. Validates `measures` against
  `metrics.SUPPORTED_CENTRALITY_MEASURES` and **raises** on an unknown measure
  (`compute_centralities()` only logs a warning, so validation must live here).
- Two `DATA_DICTIONARY` dicts (one per TSV) for the sidecars.

### `scripts/export_network_metrics.py` (new, thin)

Arg parsing, episode discovery, per-episode load + write. No metric logic.

- `--episodes` spec: `ALL | sN | sN-sM | comma-list`, discovered via
  `expand_episode_spec` over the in-checkout scenes dir (returns bare IDs like
  `s06e01b`).
- `--network-root` default `SCRATCH_DIR/output/02_build_network`.
- `--measures` default `degree,betweenness,eigenvector`.
- `--out-dir` default `output/annotations/network_metrics`.

## Critical correctness points (from design review)

1. **Episode-ID normalization (the bug we are fixing).** `expand_episode_spec`
   yields bare IDs (`s06e01b`), but `run_pipeline.py` names stage-2 dirs with the
   `friends_`-prefixed key (`normalize_episode_key` →
   `02_build_network/friends_s06e01b/`). Resolving `<network-root>/s06e01b/...`
   would silently miss **every** real pipeline run. The export resolves the
   network dir via the normalized key and **probes both** `friends_<ep>` and bare
   `<ep>` for robustness, using whichever exists.
   - Evidence: `scripts/run_pipeline.py:138,173`, `src/charnet/transcript_align.py:87-91`,
     `src/charnet/scene_subdivide.py:135`.
2. **`start`/`end` are network-coverage windows.** Stage-2 scene spans come from
   speaker-bearing rows, not the full 01a scene summary. The data dictionary must
   document `start`/`end` as network-coverage windows, not necessarily full scene
   spans. (`src/charnet/network.py:54,73`.)
3. **`n_components` included.** `scene_metrics()` already emits it; it is part of
   the structural-fields set, so it ships in the scene TSV.
4. **Stable empty-frame schemas.** `centrality_timeseries()` returns a column-less
   `pd.DataFrame()` when empty (`metrics.py:256`); both builders must instead
   return empty frames carrying the declared columns so TSV headers and tests stay
   stable.

## I/O contract (mirrors brain feature #1)

- Per-episode TSVs:
  `output/annotations/network_metrics/sN/friends_<ep>_scene_network.tsv`
  `output/annotations/network_metrics/sN/friends_<ep>_character_centrality.tsv`
- Product-level data-dictionary sidecars (unambiguous names):
  `output/annotations/network_metrics/scene_network.json`
  `output/annotations/network_metrics/character_centrality.json`
- `output/annotations/dataset_description.json` at the product-dir parent,
  written via `bids_meta.write_dataset_description` with git provenance from the
  same `git describe --tags --always --dirty` helper as `export_topic_trace.py`.
- Sidecars written **first**, so the output dir is self-describing even on a
  partial or zero-episode run.

## Missing-input behavior

Network inputs live outside the checkout (under `SCRATCH_DIR`) and may be absent
wholesale, so silent-skip can mask a misconfigured `SCRATCH_DIR`:

- `ALL` / season specs → skip-with-count (absent episodes are legitimate).
- Explicit episode list → **error** if a named episode's network dir is absent.
- In all modes → **exit nonzero if zero episodes were written**, so a bad
  `--network-root` / `SCRATCH_DIR` cannot masquerade as a successful empty export.

## Testing

Unit tests against `network_export.py`:

- `scene_network_trace` column set/order incl. `n_components`; empty-input returns
  declared-column empty frame.
- `character_centrality_trace` stable column order; empty-input returns
  declared-column empty frame; unknown measure raises.
- A small synthetic `SceneGraph` list → expected row counts and timestamp
  passthrough.

Script-level test (mirroring `test_export_topic_trace.py`): a tiny temp
`temporal_network.json` under both `friends_<ep>/` and bare `<ep>/` layouts →
asserts the export resolves either, writes both TSVs + both sidecars +
`dataset_description.json`, and exits nonzero on zero-episode runs.

## Out of scope

- TR-alignment and correspondence statistics (remain the `brain-states-friends`
  `08a`/`08b` responsibility).
- Pivoting per-character centrality into fixed character columns (consumer's job).
- Brain feature #3 (dialogue embeddings), specced separately.
