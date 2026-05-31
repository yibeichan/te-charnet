# Design — Topic-Shift Trace Export (brain feature #1)

**Date:** 2026-05-29
**Status:** Design — awaiting implementation plan
**Catalog:** `docs/data_products_catalog.md` (this is the first of three planned
feature exports: topic-trace → network metrics → embeddings)
**Downstream consumer:** `brain-states-friends` `08a`/`08b` content↔brain-state
correspondence (separate repo; not modified here).

## Goal

Persist the continuous topic-shift signal — already computed inside the
(negative-result) boundary detector and then discarded — as a documented,
timestamped annotation product the brain pipeline can TR-align. The trustworthy
deliverable is the **continuous `block_distance` regressor** ("how much is the
dialogue topic shifting right now"); the depth/peak fields ride along as an
audit trail of where the rejected detector would have fired.

This export is also the **pattern-setter**: it establishes the timestamped-TSV
+ BIDS-inspired data-dictionary convention the network-metric and embedding
exports will reuse.

## Scope / lane

In-scope (te-charnet): *produce* the timestamped trace + its schema docs.
Out-of-scope: TR-alignment, HRF modeling, and the correspondence statistics —
those remain `brain-states-friends` `08a`/`08b`. We emit episode-relative
seconds; the brain side owns mapping stimulus time → fMRI run time / TRs.

## Data product

`output/annotations/topic_shift/sN/friends_<ep>_topic_trace.tsv` — one row per
inter-turn gap, within each fan-transcript scene, in time order.

| Column | Type | Meaning |
|---|---|---|
| `scene_id` | int | Fan-transcript scene the gap falls in |
| `onset` | float (s) | Gap time = `end` of the turn before the gap, **episode-relative seconds** |
| `block_distance` | float | Cosine distance between mean-pooled `w`-turn blocks either side of the gap — the continuous topic-shift regressor |
| `depth` | float | TextTiling depth at local maxima of `block_distance`; `NaN` at non-maxima |
| `is_peak` | bool | Gap accepted as a boundary by `propose_topic_boundaries` at the recorded params (audit trail; the detector was a negative result) |
| `w` | int | Block half-width in turns (constant per file) |
| `tau_depth` | float | Depth threshold used for `is_peak` (constant per file) |
| `min_spacing` | float (s) | Minimum spacing between accepted peaks (constant per file) |

**Naming decision:** the timestamp column is `onset` (BIDS events convention,
`Units: s`), with the data dictionary stating explicitly that it is
episode-relative. (Open for veto at review: keep `time` instead.)

**Params-as-columns decision:** `w`/`tau_depth`/`min_spacing` are repeated as
constant columns so each TSV is self-describing about the config that produced
it (survives file copying). Mildly un-BIDS (BIDS would put derivation params in
a sidecar). (Open for veto at review: move them to `dataset_description.json`'s
`GeneratedBy.Parameters` instead.)

Scenes with `< 2*w + 1` turns produce no rows (the detector can't form blocks
there) — consistent with `propose_topic_boundaries`' guard.

## Schema documentation (BIDS-inspired, not full-BIDS)

te-charnet outputs are **stimulus-level** annotations keyed by episode, not
subject-level neural data, so the BIDS `sub-/ses-/datatype/` machinery is a
forced fit and is **not** adopted. We adopt the two BIDS conventions that do
apply to a derivative annotation dataset:

1. **Shared column data dictionary** —
   `output/annotations/topic_shift/topic_trace.json`. One file for all episodes
   (columns are identical), each column documented with `Description`,
   `Units`, and `Levels` where applicable. Content:

   ```json
   {
     "scene_id":       {"Description": "Fan-transcript scene index the gap falls in"},
     "onset":          {"Description": "Gap time: end of the turn before the gap, relative to episode start. Mapping to fMRI run time / TRs is the consumer's responsibility.", "Units": "s"},
     "block_distance": {"Description": "Cosine distance between mean-pooled w-turn blocks on either side of the gap; continuous topic-shift regressor. Higher = larger semantic shift.", "Units": "arbitrary (0-1 for normalized embeddings)"},
     "depth":          {"Description": "TextTiling depth (rise above neighboring valleys) at local maxima of block_distance; NaN at non-maxima."},
     "is_peak":        {"Description": "Gap accepted as a boundary by the topic-shift detector at the recorded params. NOTE: the detector is a documented negative result (see docs/scene_segmentation_evaluation.md, Prototype #2); is_peak is an audit trail, not a validated boundary.", "Levels": {"true": "accepted", "false": "not accepted"}},
     "w":              {"Description": "Block half-width in turns used to compute block_distance and is_peak."},
     "tau_depth":      {"Description": "Depth threshold for is_peak."},
     "min_spacing":    {"Description": "Minimum seconds between accepted peaks (greedy, deepest-first).", "Units": "s"}
   }
   ```

2. **Dataset metadata** — `output/annotations/dataset_description.json`,
   describing the whole charnet annotation tree as a derivative dataset (so it
   forward-covers scenes/sentences when they get sidecars later):

   ```json
   {
     "Name": "charnet Friends stimulus annotations",
     "BIDSVersion": "1.9.0",
     "DatasetType": "derivative",
     "GeneratedBy": [{"Name": "charnet", "Version": "<git-describe>"}],
     "SourceDatasets": [{"Description": "Courtois NeuroMod Friends fMRI stimulus episodes"}]
   }
   ```
   `<git-describe>` is filled at write time from the repo's current commit.

These files are written/refreshed by the export script (idempotent — overwrite
if content changed). A `.bidsignore`-style concern does not apply since we are
not claiming full BIDS validity.

## Architecture

Almost entirely reuse of the `charnet.topic_shift` module built this session.

```
src/charnet/topic_shift.py   (extend)
  episode_topic_trace(turns_by_scene, vecs_by_scene, *, w, tau_depth, min_spacing)
      -> pd.DataFrame[scene_id, onset, block_distance, depth, is_peak, w, tau_depth, min_spacing]
  _accepted_peak_indices(trace, turns, *, w, tau_depth, min_spacing) -> set[int]
      # shared greedy logic; propose_topic_boundaries refactored to call it
      # so is_peak is exactly consistent with the detector (no float-equality match)

src/charnet/bids_meta.py     (new, small)
  write_data_dictionary(path, columns: dict)        # idempotent JSON write
  write_dataset_description(path, *, name, version, ...)

scripts/export_topic_trace.py  (new CLI)
  --episodes ALL|sN|sN-sM|list   --sentences-in   --out-dir
  --cache-dir   --w (default 1)  --tau-depth (default 0.5)  --min-spacing (default 30)
  per episode: load sentences -> turns_by_scene -> embed_texts_cached (REUSES cache,
  no re-encode) -> episode_topic_trace -> write TSV
  once: write topic_trace.json data dictionary + dataset_description.json
```

`episode_topic_trace` per gap `i`: `onset = turns[i].end`,
`block_distance = block_distance_trace(vecs, w)[i]`,
`depth = peak_depths(trace)` value if `i` is a local max else `NaN`,
`is_peak = i in _accepted_peak_indices(...)`.

**Refactor note (justified):** `propose_topic_boundaries` currently returns
times via inline greedy logic. Extracting `_accepted_peak_indices` lets both it
and `episode_topic_trace` share one implementation, so `is_peak` matches the
detector exactly without fragile float-time matching. `propose_topic_boundaries`
behavior is unchanged (re-verified by its existing tests).

**Defaults:** `w=1` (rawest continuous signal — downstream HRF convolution
smooths), `tau_depth=0.5`, `min_spacing=30` (hybrid-calibrated values). All
exposed as CLI flags; the continuous `block_distance` only depends on `w`.

## Testing (TDD)

- `_accepted_peak_indices` returns the gap indices whose `turns[i].end` equals
  `propose_topic_boundaries(...)` output, on the Task-5 fixture
  (`vecs=[A,A,B,B,C,C]`, `w=1`) — pins the refactor (no behavior change).
- `episode_topic_trace`: synthetic `turns_by_scene` + `vecs_by_scene` →
  one row per gap; `onset == turns[i].end`; `block_distance` matches
  `block_distance_trace`; `depth` non-NaN only at local maxima; `is_peak`
  matches `_accepted_peak_indices`; a scene with `< 2*w+1` turns contributes no
  rows; scene_id preserved.
- `bids_meta.write_data_dictionary` / `write_dataset_description`: round-trip
  the JSON, assert required keys (idempotent overwrite).
- Export smoke: run `export_topic_trace.py` on one episode against a tiny fake
  sentence TSV with a fake encoder (no model download); assert the TSV parses,
  has the documented columns, and the two JSON docs exist and validate.

## Out of scope (YAGNI)

- Network-metric and embedding exports (their own specs next).
- Retrofitting scenes/sentences with sidecars (later, or never).
- Full BIDS subject/session layout, bids-validator compliance.
- Choosing the "right" `w` empirically — the boundary calibration was
  degenerate for a continuous regressor; `w` is left a documented knob.

## Success criteria

A reproducible `export_topic_trace.py` run produces, for each requested
episode, a `friends_<ep>_topic_trace.tsv` with the documented columns, plus the
shared `topic_trace.json` and `dataset_description.json`; the continuous
`block_distance` column is a usable episode-relative regressor; `is_peak`
reproduces the detector exactly; all new code is tested and ruff-clean.
