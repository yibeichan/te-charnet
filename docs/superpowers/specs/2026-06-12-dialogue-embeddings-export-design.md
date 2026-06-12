# Dialogue-embeddings export (brain feature #3) — design

**Date:** 2026-06-12
**Status:** approved (pre-implementation)
**Author:** Yibei (with Claude; design reviewed by Codex, REQUEST CHANGES round folded in)

## Problem

The brain-states-friends analysis needs per-dialogue-turn sentence embeddings
aligned to episode time. The embeddings already exist as a pipeline
intermediate — `output/intermediate/sentence_embeddings/sN/<ep>.npz`, written by
`embed_texts_cached` (`src/charnet/topic_shift.py`) for the topic-shift trace —
but the NPZ stores only the vector matrix and a cache key: **no timestamps, no
text, no scene structure**. `docs/data_products_catalog.md` flags exactly this
gap ("no turn timestamps stored in npz; needs turn timestamps to be a
product").

The turn timing is recoverable deterministically: `turns_by_scene`
(`topic_shift.py`) rebuilds the same turn sequence — with `start`/`end` — from
the **tracked** sentence speaker tables
(`output/annotations/sentences/sN/friends_<ep>_sentence_speaker_table.tsv`,
341 episodes, all 7 seasons). This feature turns intermediate + timing into a
proper exported data product, following the pattern of the two prior brain
exports (topic-shift trace, network metrics).

## Decisions (settled with user)

- **Format:** per-episode NPZ + TSV pair; TSV row `i` describes NPZ `vecs` row `i`.
- **Coverage:** all 7 seasons / 341 episodes — s7 embeddings are generated as
  part of this feature (the cache currently stops at s6).
- **Git:** TSVs and data dictionary **tracked**; NPZs **untracked** (~160 MB
  total), deterministically regenerable from tracked tables + pinned encoder.
- **TSV contents:** timing + IDs only — no text column (text lives in the
  tracked sentence tables and is rebuilt by `build_text`).
- **Single code path:** the exporter obtains vectors via `embed_texts_cached`
  against the existing cache dir, so cache maintenance and export share one
  contract.

## Product layout

Per episode, under `output/annotations/dialogue_embeddings/sN/`:

- `friends_<ep>_dialogue_turns.tsv` — one row per turn (**tracked**):
  - `turn_id` — 0-based, episode-wide, equals the row index into the NPZ `vecs`
  - `scene_id` — within-episode scene ID (int)
  - `start`, `end` — seconds, episode-relative, from the merged turn span
  - `n_sentences` — number of sentence-table rows merged into the turn
- `friends_<ep>_dialogue_embeddings.npz` — **untracked**:
  - `vecs` — `(n_turns, 384)` float32, `all-MiniLM-L6-v2`
  - `key` — SHA256 of model_id + texts (same value as the cache contract)

Sidecars (tracked): `dialogue_turns.json` data dictionary at the product root
(pattern of `scene_network.json`); the generic BIDS-style
`dataset_description.json` written via `charnet.bids_meta` (it records
provenance fields only — product membership is documented in
`docs/data_products_catalog.md`, **not** in `dataset_description.json`).

## Turn construction contract

The turn sequence — and therefore `turn_id`, the embedding row order, and the
cache key — is defined as:

1. Scenes in ascending `scene_id` order (`groupby(..., sort=True)`).
2. Within a scene, rows **stable-sorted** by `start` (`kind="mergesort"`); for
   tied `start` values the original TSV row order decides. (Codex P2: the
   current plain `sort_values("start")` is an unstable quicksort — the export
   makes the stable sort the explicit contract, applied in `turns_by_scene`
   itself so trace and export share it.)
3. Rows merged into turns by `group_turns_for_scene` semantics: consecutive
   rows sharing a non-blank `utterance_ct` merge; blank-`utterance_ct` rows
   never merge; turn span is `[first start, max end]`; text via `build_text`
   (ct preferred, speech-to-text fallback, NaN-safe).
4. `turn_id` numbers the concatenated per-scene turn lists 0..n-1.

**`n_sentences` API gap (Codex P1):** `Turn` stores only text/start/end and
`group_turns_for_scene` discards the merge count. Fix: add a counts-aware
variant in `topic_shift` — `group_turns_with_counts(scene_rows) ->
list[tuple[Turn, int]]` — with `group_turns_for_scene` delegating to it.
Existing trace behavior and tests unchanged.

## Export script

`scripts/export_dialogue_embeddings.py`, structured like
`export_topic_trace.py`:

- CLI: `--episodes ALL | sN | sN-sM | comma-list` (house convention), plus
  `--scenes-in`, `--sentences-in`, `--out-dir`, `--cache-dir` with the same
  defaults as the trace exporter. Episode IDs are bare `sXXeYYa` throughout
  (resolved via `expand_episode_spec`); the `friends_` prefix appears only in
  output filenames — never passed to `embed_texts_cached`, whose season
  routing parses `episode[1:3]`.
- Per episode: load sentence table → build turns + counts (contract above) →
  build texts → `embed_texts_cached(ep, texts, encoder, cache_dir)` → write
  TSV and NPZ (atomic: write to temp file in the destination dir, then rename).
- The product NPZ stores the exact `vecs` returned and the recomputed key.

**Cache reality (Codex P1):** the on-disk s1–s6 caches predate folding
`model_id` into the hash (`topic_shift.py` documents the one-time mismatch;
spot-checked: `s01e01a.npz` carries the pre-model_id key). So the first full
export run **re-encodes all 341 episodes**, refreshing the cache with current
keys; subsequent runs are cache hits. The exporter logs per-run counts:
`cache hits / re-encoded / total`, so a surprise mass re-encode is visible.
Runtime budget: full CPU encode with MiniLM — acceptable (no GPU dependency).

Empty-text turns (both transcript columns blank) are embedded as-is — the
cache contract hashes whatever text list is produced, and excluding rows would
break the turn_id ↔ NPZ row alignment. They remain auditable via the TSV.

## Verifier

`scripts/verify_dialogue_embeddings.py`, independent-reconstruction style
(pattern of `verify_network_export.py`): re-derives turns, counts, and texts
from the tracked sentence tables using its **own** implementation of the turn
construction contract (no imports from `charnet.topic_shift`), then checks per
episode:

1. **TSV correctness** — row count, `turn_id` sequence, `scene_id`, `start`,
   `end`, `n_sentences` all match the reconstruction exactly.
2. **Row accounting** — every sentence-table row that carries a usable
   `scene_id` lands in exactly one turn, and the per-turn `n_sentences` sum
   equals the retained row count. Rows with missing/NaN `scene_id` are counted
   and reported (pandas `groupby` drops NaN groups silently — make the drop
   explicit, never silent).
3. **Key check** — recompute SHA256(model_id + rebuilt texts) with its own
   hash implementation and require the product NPZ `key` to equal it.
4. **Vector binding (Codex P1)** — the key alone does not bind `vecs` (a
   permuted or substituted matrix would still carry a valid key). The verifier
   independently resolves the cache path for the episode, requires the cache
   NPZ's `key` to equal the recomputed hash, and requires the product `vecs`
   to be `np.array_equal` to the cache `vecs`. Chain: rebuilt texts → key →
   cache vecs → product vecs.
5. **Sanity** — `vecs` dtype float32, shape `(n_turns, 384)`, all values
   finite; `start ≤ end` per turn; turn spans within scene bounds.

Exit codes mirror the prior verifiers: 0 all checked episodes pass, 1 any
mismatch, 2 nothing checkable. Episodes whose product NPZ is absent (fresh
clone, untracked product) are **skips**, named in the summary — TSV-only
checks (1–2) still run for them. If the product NPZ exists but the cache NPZ
is absent or carries a different key, vector binding cannot be established and
that is a **failure**, not a skip (the product cannot be vouched for);
regenerate via the export script first.

**Residual limit (documented, accepted):** without re-running the encoder the
verifier proves vectors are *the cached vectors for exactly these texts*, not
that the encoder computed them correctly. An optional `--re-embed N` flag
re-encodes N randomly chosen episodes end-to-end and compares within
tolerance, for occasional deep checks.

## Testing

`tests/test_export_dialogue_embeddings.py` +
`tests/test_verify_dialogue_embeddings.py`, fixture style of the prior export
tests (synthetic sentence tables, fake deterministic encoder — no model
download in tests):

- Turn grouping → TSV rows: merge runs, blank-ct isolation, `n_sentences`,
  tie-on-`start` ordering stability.
- `turn_id` ↔ NPZ row alignment, including a multi-scene episode.
- Cache behavior: hit, stale-key re-encode (pre-model_id key fixture), miss.
- Empty-text turn embedded and counted.
- Verifier positive path → exit 0; negative controls: perturbed `vecs` row
  (must fail via vector binding), edited TSV timing, wrong `n_sentences`,
  permuted `vecs` with valid key (must fail), missing NPZ → skip not fail.
- CLI episode-spec expansion (`ALL`, season, range, list).

`ruff check .` clean; full suite (currently 155 tests) stays green.

## Non-goals

- No speaker column in the TSV (available in the sentence tables; add later if
  the brain side needs it — would be a column addition, not a redesign).
- No alternative encoders/dimensions; model is pinned to `all-MiniLM-L6-v2`.
- Does not judge embedding quality — only that the shipped product faithfully
  represents (turns × pinned encoder) over the tracked tables.

## Catalog & docs

`docs/data_products_catalog.md`: move dialogue embeddings from
"available-not-yet-exported" to a product entry (layout, columns, tracked/
untracked split, regeneration command, verifier). Regeneration one-liner:
`python scripts/export_dialogue_embeddings.py --episodes ALL`.
