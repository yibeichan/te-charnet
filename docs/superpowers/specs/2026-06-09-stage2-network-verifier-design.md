# Stage-2 network verifier — design

**Date:** 2026-06-09
**Status:** approved (pre-implementation)
**Author:** Yibei (with Claude)

## Problem

Stage-2 (`output/02_build_network/`) is intentionally **not** tracked in git — it
is a regenerable intermediate (8.1 MB, 682 JSON files), rebuildable from the
**tracked** speaker tables in ~19 s/season. The committed deliverable is the
network-metric export under `output/annotations/network_metrics/`.

`scripts/verify_network_export.py` already independently confirms one hop of the
chain: it recomputes every export metric from the stage-2 `temporal_network.json`
and matches the committed export TSVs. But it **treats stage-2 as ground truth** —
nothing verifies that stage-2 itself faithfully constructs the graphs from its
input. That construction step (`charnet.network.build_temporal_network_from_aligned_rows`)
is the "hard to verify" piece: it is a deterministic transform with no external
ground-truth graph to compare against — the graph *is* the operationalization of
the interaction model.

This verifier closes the remaining hop so the whole pipeline is independently
checked end-to-end:

```
tracked speaker table  ──(stage 2)──▶  temporal_network.json  ──(export)──▶  network_metrics TSVs
        └──────────── this verifier ────────────┘   └──── verify_network_export.py ────┘
```

Once both verifiers pass, leaving stage-2 untracked is genuinely safe: every
committed number is reproducible and checked from a tracked input.

## What "accuracy" means here (scope)

Two distinct notions; this verifier covers **both** of the checkable ones:

1. **Faithful reconstruction** — does the committed graph match an *independent*
   re-derivation from the tracked speaker table? (core check)
2. **Structural invariants** — does the committed graph satisfy properties it must
   hold regardless of input (weight formula, copresence domain, adjacency bounds,
   edge/node consistency, scene-boundary sanity, correct aggregation)? (defense-in-depth)

**Out of scope (not script-checkable):** whether the construction *rules* are the
right model — e.g. whether "adjacency = consecutive distinct-speaker turns" is a
valid interaction proxy, or whether the `1.0 / 0.5 / 0.25` weights are well chosen.
That is a Methods/construct-validity question, argued in prose, not asserted by a
verifier.

## Independence rule

The script reimplements the construction math in its own code. It imports
**nothing** from `charnet.network` (not `build_temporal_network_from_aligned_rows`,
not `aggregate_episode_graph`) and reads the TSV with pandas directly rather than
through `charnet.io.load_corrected_speaker_rows`. This mirrors the export
verifier's avoidance of `charnet.metrics`. Agreement is then real evidence the
committed graphs are correct, not a re-run of the same code.

The documented construction parameters (`weight_adjacency=1.0`,
`weight_proximity=0.5`, `weight_copresence=0.25`, `proximity_window=3`) are the
script's defaults and exposed as CLI flags, so a non-default stage-2 run can still
be checked against matching parameters.

## Stage-2 construction being verified (reference)

Per scene, from rows grouped by `scene_id` (rows with empty `start`, `scene_id`,
or `speaker` dropped; `start`/`end` coerced to float, `scene_id` to int):

- `turns` = scene rows sorted by `start`; `speakers_seq` = the speaker per turn.
- `nodes` = sorted unique speakers.
- `scene_start` = first turn's `start`; `scene_end` = max turn `end`.
- **adjacency**: for each consecutive turn pair `(i-1, i)` with distinct non-empty
  speakers, increment the unordered-pair count by 1.
- **proximity**: for each ordered pair `(i, j)` with `0 < j-i ≤ window` and
  distinct non-empty speakers, add `1 / (j-i)` to the unordered-pair score.
- **copresence**: `1.0` for every unordered pair of distinct co-present speakers.
- **edge weight**: `w = 1.0·adj + 0.5·prox + 0.25·cop`; an edge is emitted only
  if `w > 0`. With the **default** positive `weight_copresence=0.25`, every
  co-present distinct pair has `cop = 1` (`w ≥ 0.25`) so every such pair yields an
  edge. This no longer holds under non-default flags (`weight_copresence=0` →
  pairs with no adjacency/proximity drop; negative weights change the `w > 0`
  filter), so the verifier must apply the actual configured weights, not assume
  the all-pairs shortcut.

**Where rounding happens (verified against source):** `models.py`'s `to_dict`
does **no** rounding — it passes stored values through. Scene `weight` and
`proximity` are rounded to 4 dp in `network.py` at `EdgeData` construction (from
**raw** proximity); `adjacency` is stored as a raw integer-valued **float**
(`1.0`, `2.0`, …) and `copresence` as `0.0`/`1.0`. `02_build_network.py` rounds
the episode aggregate fields to 4 dp at JSON-assembly time.

Episode aggregate (`episode_network.json`, built by `aggregate_episode_graph` then
serialized in `02_build_network.py`): union of all scene nodes; each of the four
edge attributes (`weight`, `adjacency`, `proximity`, `copresence`) **summed across
scenes** — summing the **already-rounded** scene `weight`/`proximity` and the raw
`adjacency`/`copresence` — then **all four rounded to 4 dp** on write; `n_scenes`
= number of scene graphs; `start`/`end` = min scene start / max scene end.
**Empty episode** (zero retained scenes): `temporal_network.json` is `[]`;
`episode_network.json` has `start=0.0`, `end=0.0`, `n_scenes=0`, empty
nodes/edges.

**Turn ordering:** stage-2 builds `grouped[scene_id]` by appending rows in
input order, then `sorted(..., key=start)` — Python's **stable** sort. For tied
`start` values the original TSV row order decides `speakers_seq`, hence adjacency,
proximity, and edges. The verifier MUST preserve TSV row order and stable-sort by
`start` **only** — sorting on any secondary key (`end`, `sentence_id`, speaker)
would diverge.

**Duplicate `scene_id` rows are not an error** — they are intentionally grouped
into one scene graph; output scene IDs are the sorted unique set.

## Architecture

Single standalone CLI script `scripts/verify_stage2_network.py`, structured like
`verify_network_export.py`:

- `_reconstruct_scene(turns, params) -> dict` — own adjacency/proximity/copresence/
  weight derivation; returns `{nodes, start, end, edges{pair: attrs}}`.
- `_reconstruct_episode(scene_dicts) -> dict` — own aggregation of reconstructed
  scenes into the expected episode graph.
- `check_episode(ep, table_path, temporal_json, episode_json, params, tol) -> list[str]`
  — reconstruct, then compare + assert invariants; returns mismatch strings.
- `resolve_*` helpers to map an episode key to its committed stage-2 files
  (probe `friends_<ep>/` then bare `<ep>/`, mirroring the export verifier).
- `main()` — discover, loop, summarize, exit code.

## Data flow

1. **Discover** episodes from the tracked tables:
   `output/annotations/sentences/*/*_sentence_speaker_table.tsv`.
2. **Read** each table with pandas using **`keep_default_na=False`** (or
   `dtype=str` + explicit empty checks). *(BLOCKER if missed:* default
   `pd.read_csv` turns empty cells into `NaN`; `str(np.nan).strip() == "nan"` is
   non-empty and `float(np.nan)` succeeds, so blank fields would survive as bogus
   `"nan"` speakers or `NaN` start/end instead of being dropped — the exact
   pandas-NaN-cast gotcha.) Mirror stage-2's **two-phase** row filter precisely:
   (a) `load_corrected_speaker_rows` drops rows with empty `start` (scene-marker
   rows) and maps the chosen speaker column into `speaker`; (b) `network.py` then
   drops rows with empty `scene_id` or `speaker`, and rows where
   `scene_id`/`start`/`end` fail numeric coercion (`int(float(scene_id))`,
   `float(start)`, `float(end)`). Combined effective filter: keep a row iff
   `start`, `scene_id`, `speaker` are all non-empty **and** numerically coercible.
3. **Reconstruct** per-scene graphs and the episode aggregate in independent code,
   reproducing stage-2's 4-decimal rounding.
4. **Compare** to committed `temporal_network.json` (per-scene) and
   `episode_network.json` (aggregate).
5. **Assert invariants** on the committed graphs.
6. **Exit** 0 iff everything matches and holds.

## Comparison details

- **Per-scene `temporal_network.json`:**
  - Scene-id set matches (missing or extra scene fails).
  - Per scene: `start`, `end`, `nodes` (set equality) match.
  - Per scene: **edge set matches both ways** — a missing edge and an extra edge
    both fail (same teeth as the export verifier's row-set completeness check).
  - Per edge: `weight`, `adjacency`, `proximity`, `copresence` match within `--tol`.
- **Episode `episode_network.json`:**
  - Node set, `n_scenes`, `start`, `end` match. Empty episode: expect `[]` /
    `start=end=0.0` / `n_scenes=0`.
  - Edge set matches both ways; per-edge attributes match within `--tol`. The
    expected aggregate must sum the **already-rounded** scene `weight`/`proximity`
    (and raw `adjacency`/`copresence`) across scenes, then round all four to 4 dp —
    matching `aggregate_episode_graph` + the write-time rounding. Aggregating raw
    per-scene values and rounding once would diverge.
- Reconstruction reproduces stage-2's rounding (weight/proximity → 4 dp) so the
  default `--tol 1e-6` yields exact agreement, not slack absorption.

## Structural invariants (asserted on committed graphs)

Per scene:
- `weight ≈ 1.0·adjacency + 0.5·proximity + 0.25·copresence`, comparing the stored
  (4-dp) `weight` against the **un-rounded** formula evaluated from the stored
  (4-dp) `proximity`, with slack that accounts for *two* independent 4-dp
  roundings: (a) `weight` itself is `round(raw, 4)` → ≤ `5e-5`; (b) the formula
  here uses stored proximity while stage-2 used raw proximity → ≤
  `weight_proximity × 5e-5`. So slack = **`5e-5 · (1 + weight_proximity) + tol`**
  (≈ `7.6e-5` at defaults), and the recomputed formula is **not** re-rounded
  (re-rounding can push the two 4-dp values a full `1e-4` apart). The
  reconstruction check above is exact because it derives raw proximity itself.
- `copresence ∈ {0.0, 1.0}` — and `== 1.0` on every committed edge under the
  default positive `weight_copresence`, since edges then only exist between
  co-present speakers.
- `adjacency` is a non-negative **integer-valued float** (check `x == int(x)`,
  not JSON `int` type) and `≤ len(turns) - 1`.
- every edge endpoint is in the scene's `nodes`; no self-loops; no duplicate
  undirected edge.
- `nodes` equals the set of (non-empty) speakers seen in the scene's rows.
- `start == min turn start`, `end == max turn end`, `start ≤ end`.
- scene `[start, end]` lies within `[episode_start, episode_end]`.

Episode:
- `nodes` equals the union of all scene nodes.
- each edge attribute equals the summed per-scene values (within tol).
- `n_scenes` equals the number of scene graphs.

## CLI

```
python scripts/verify_stage2_network.py \
    --tables-root output/annotations/sentences \
    --network-root output/02_build_network \
    [--tol 1e-6] \
    [--weight-adjacency 1.0] [--weight-proximity 0.5] \
    [--weight-copresence 0.25] [--proximity-window 3]
```

Output: episodes found / checked / skipped, then a capped mismatch list (first 50,
"... and N more"). Skips (not failures) when a table has no corresponding
committed stage-2 files — e.g. stage-2 not regenerated on a fresh clone — and
names them.

Exit codes (mirroring the export verifier):
- **0** — every reconstruction value, every aggregate, and every invariant holds.
- **1** — at least one mismatch or invariant violation.
- **2** — nothing could be checked (no tables found, or all skipped).

## Testing

`tests/test_verify_stage2_network.py`. Fixtures are **hand-built** — the table TSV
and the expected `temporal_network.json` / `episode_network.json` are authored as
literal data, **not** generated by importing `charnet.network`/`charnet.io`, so
the test does not validate the verifier against the very code it is meant to be
independent of.

- **Clean fixture → exit 0.** A tiny hand-built table + matching committed JSON
  reconstructs and passes all invariants. Include a **one-speaker scene** (one
  node, zero edges) so node-only scene handling is exercised — the dropped-edge
  control does not cover it.
- **Rounded-aggregation fixture → exit 0.** A scene whose proximity is a fraction
  that rounds at the **scene** level (e.g. `1/3`), spanning ≥2 scenes that share
  an edge, so the episode aggregate must sum already-rounded scene values. Proves
  the verifier matches `aggregate_episode_graph`'s rounded-then-summed-then-rounded
  behavior rather than raw-summed-once.
- **Negative control: perturbed weight → exit 1.** Mutate one committed `weight`.
- **Negative control: dropped edge → set-level failure.** Remove one edge from
  `temporal_network.json`.
- **Negative control: corrupted aggregate → exit 1.** Alter one
  `episode_network.json` summed attribute.
- **Real-episode sanity (not counted as core coverage).** Reconstruction matches
  at least one committed real episode; skipped when stage-2 is absent, so it is a
  bonus check, not the regression guarantee — the hand-built fixtures are.

`ruff check .` clean; full `pytest` (currently 132 tests) stays green.

## Preconditions & non-goals

- Requires stage-2 on disk. On a fresh clone, regenerate first:
  `python scripts/run_pipeline.py --season sN --skip-stages 1a,1b,3,4`
  (~19 s/season), then run this verifier. Same precondition as the export
  verifier; absence is a *skip*, not a failure.
- Does **not** judge the construction model's validity (see scope). Only that the
  committed graphs faithfully implement the documented construction and satisfy
  required structural properties.

## Catalog & docs

Update `docs/data_products_catalog.md`'s stage-2 / network-metric provenance note
to mention the new verifier alongside `verify_network_export.py`, so the two-hop
verification chain is documented in one place.
