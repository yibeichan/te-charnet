# Design — Topic-Shift Scene Subdivision (#2) + char×topic Hybrid

**Date:** 2026-05-29
**Status:** Design — awaiting implementation plan
**Companion:** `docs/scene_segmentation_evaluation.md` (Findings 1–2, Improvement
directions #1/#2, Prototype #1 results)

## Goal

Build the topic-shift sub-boundary detector (improvement direction #2) and,
in the same body of work, the **char × topic AND-logic hybrid** that
prototype #1's "honest assessment" identified as its most promising
continuation. One full 292-episode sweep then answers two questions at once:

1. **Standalone:** Does a sentence-embedding topic-shift signal recover any
   net F1 on the `goal_change` bucket without the precision collapse that
   sank prototype #1?
2. **Hybrid:** Does requiring char-set change **and** topic shift to agree
   buy back the precision prototype #1 lost (only 23% of its proposed
   boundaries were real)?

A flat standalone result is still informative — it tells us whether the
signal exists for the hybrid. This is a research prototype; a documented
negative result is an acceptable outcome, as with prototype #1.

## Background (from the eval doc)

- `goal_change` is the lowest-detected gold segment-boundary type
  (match@5s = 0.307, 417 single-reason boundaries). Fan transcripts almost
  never mark topic shifts inside a continuous-location scene.
- Prototype #1 (char-presence, tile+threshold) moved the targeted buckets
  the right way (charact_leave +4.4 pp, goal_change +5 pp) but net segment
  F1 was flat (−0.1 pp): precision fell 5 pp while recall rose only 2.2 pp.
  **Lesson: a noisy boundary proposer hurts precision faster than recall
  helps.**
- The 10-ep panel over-stated prototype #1's F1 by >1 pp vs the full sweep
  (selection bias). Thresholds must not be tuned on the same episodes the
  headline is read from.

## Design decisions (locked in brainstorming)

### 1. Base unit — community-transcript *turn*

We embed at the **ct-turn** granularity, not the raw Speech2Text row.

- A "turn" = a maximal run of consecutive sentence rows (within one scene)
  that share the same `utterance_ct` value. This reconstructs the spoken
  turn that S2T split across multiple rows.
- **Rationale:** embedding `utterance_ct` per row would make the
  adjacent-distance trace track *how many rows a ct line spans* (an S2T
  segmentation artifact) rather than topic. De-duplicating to turns removes
  that confound and yields richer text per embedding.
- **Text fallback:** for rows where `utterance_ct` is blank/NaN, fall back
  to `utterance`. Emptiness test follows the repo gotcha — `isna()` OR
  `fillna("").str.strip() == ""`, never `astype(str)` (which turns NaN into
  the non-empty string "nan").
- **Turn timestamp:** `start` = min row `start`, `end` = max row `end` over
  the turn's constituent rows. A proposed boundary placed "at a turn gap"
  uses the `end` of the turn before the gap — guaranteed to be a sentence
  end, where the evaluator and downstream consumers expect boundaries.

### 2. Embedding

- Model: `sentence-transformers` `all-MiniLM-L6-v2` (doc's suggested
  default; standard for short text; no reason to go heavier on dialogue
  this short).
- Device: CPU. ~86k sentences collapse to fewer turns; a one-time forward
  pass is minutes.
- **Cache:** embeddings written to
  `output/intermediate/sentence_embeddings/{season}/{episode}.npz` keyed by
  episode, so threshold sweeps never re-encode. Cache stores the turn text
  hash alongside vectors so a text change invalidates the entry.
- New dependency `sentence-transformers` added via `uv`.

### 3. Detection — TextTiling-style block depth score

Within each fan-transcript scene independently (the sliding window never
crosses an existing scene boundary):

1. Build the ordered turn sequence for the scene; embed each turn (mean of
   its row text is not needed — one turn = one text = one vector).
2. At each inter-turn gap `i` (between turn `i` and turn `i+1`), form a
   **left block** = mean-pooled embeddings of the `W` turns ending at `i`,
   and a **right block** = mean-pooled embeddings of the `W` turns starting
   at `i+1`. Gap score `g_i` = cosine **distance** between the two blocks
   (higher = bigger topic shift).
3. Identify **local maxima** of the `g` trace. For each peak, compute a
   **depth score** = `(g_peak − left_valley) + (g_peak − right_valley)`,
   where the valleys are the nearest lower scores on each side (standard
   TextTiling depth). Absolute peak height alone is not used — depth is
   robust to per-scene baseline drift.
4. Accept peaks with `depth ≥ τ_depth`, subject to min-spacing `M` from the
   scene endpoints and from each other (greedy, highest-depth first).
5. Emit the accepted boundary **times** (turn-`i` `end`).

Scenes with fewer than `2W + 2` turns cannot form full blocks → no
boundaries (these short scenes don't need subdivision). At scene edges,
blocks are truncated to available turns rather than padded.

**Fallback if absolute `τ_depth` proves brittle across scenes:** switch to
the standard TextTiling adaptive cutoff (`mean(depths) + c·std(depths)`).
Noted as a contingency; absolute threshold is the starting design because
the calibration split guards against overfitting it.

### 4. Combination — char × topic AND within ε = 3 s

- The char-presence proposer (prototype #1) emits candidate **times** per
  scene at its frozen published parameters: τ_jaccard = 0.7,
  persistence = 2 tiles, min-spacing = 30 s, **no** shot-snap (shot-snap was
  a no-op in prototype #1). These parameters are **not** re-tuned here.
- The topic-shift proposer emits candidate times per scene (Section 3).
- A **hybrid** boundary fires only where a char candidate and a topic
  candidate fall within **ε = 3 s** of each other. The boundary is placed at
  the **topic-shift time** (a sentence end; avoids landing mid-utterance on
  a shot cut).
- No OR config: firing on *either* signal is strictly more boundaries =
  more of the precision problem #1 already demonstrated.

### 5. Config matrix — one full 292-ep sweep

| Config | Boundaries added | Purpose |
|---|---|---|
| `baseline` | none | apples-to-apples reference (re-run) |
| `char` | char-presence only | reproduce prototype #1 as sanity check |
| `topic` | topic-shift only | does an independent signal exist? |
| `hybrid` | char ∧ topic within ε | does agreement buy back precision? |

Each config produces an augmented `scene_summary.tsv` tree and is scored by
the existing `scripts/evaluate_scene_segmentation.py`. Headline metric:
**segment F1@5s**; diagnostic rows: `charact_entry`, `charact_leave`,
`goal_change` from `boundary_diagnostics.tsv`; plus scene-unit F1@5s and
precision to watch for the prototype-#1 precision collapse.

### 6. Calibration — season split (no selection bias)

- **Tune** topic-shift parameters (`W`, `τ_depth`, `M`) on the
  **calibration split: s1–s2** (96 half-episodes) via a coarse grid.
- **Report** all four configs' headline metrics on the **disjoint test
  split: s3–s6** (196 half-episodes). The reported lift is on episodes the
  thresholds never saw.
- char-presence parameters and ε are **frozen**, not tuned — only the three
  topic-shift parameters are fit.
- Full-292 numbers may be tabulated in an appendix for continuity with the
  prototype-#1 table, but the **headline comparison is test-split only**.

## Architecture

The repo currently has one augmenter, `scripts/augment_scenes_char_presence.py`,
whose `augment_episode` mixes two concerns: (a) the char-presence candidate
proposer, and (b) the generic "rewrite a `scene_summary.tsv` given per-scene
sub-boundary times" plumbing. The hybrid needs to *reuse* the char proposer,
so a modest refactor is justified (and only what serves this goal):

```
src/charnet/
  scene_subdivide.py   NEW  generic plumbing: read scene_summary.tsv,
                            call a propose_fn(scene)->times per scene,
                            rewrite rows (lifted verbatim from the existing
                            augment_episode body), write augmented TSV.
  char_presence.py     NEW  propose_sub_boundaries(...) moved here from the
                            script (logic unchanged; importable by hybrid).
  topic_shift.py       NEW  turn grouping, embedding + cache, block depth-
                            score proposer: propose_topic_boundaries(scene)->times.

scripts/
  augment_scenes.py    NEW  unified CLI: --mode {char,topic,hybrid},
                            --scenes-in/--scenes-out, --calib/--test split
                            helpers, parameter flags. Thin: wires a proposer
                            into scene_subdivide.
  augment_scenes_char_presence.py   KEPT as a thin shim importing the moved
                            proposer, so prototype #1's documented invocation
                            still runs. (Or delete after confirming nothing
                            references it — decide at plan time.)
  evaluate_scene_segmentation.py    UNCHANGED — consumes the augmented trees.
```

`hybrid` mode in `augment_scenes.py` runs both proposers per scene and
intersects their times within ε before handing the surviving times to
`scene_subdivide`.

### Data flow

```
sentences/{s}/...sentence_speaker_table.tsv ─┐
                                             ├─ topic_shift: group turns →
                                             │   embed (cached) → block depth
                                             │   → candidate times
scenes/{s}/...scene_summary.tsv ─────────────┤
char-tracker stage-05 grid ──────────────────┴─ char_presence: tile Jaccard
                                                 → candidate times

  (mode=topic)  topic times ─────────────────┐
  (mode=char)   char times ──────────────────┤→ scene_subdivide → augmented
  (mode=hybrid) char ∧ topic within ε ───────┘   scenes_{mode}/  → evaluator
```

## Testing (TDD)

Unit tests, written before implementation, with hand-built fixtures:

- **Turn grouping:** consecutive identical `utterance_ct` collapse to one
  turn; blank-ct rows fall back to `utterance`; NaN handled per the repo
  gotcha (not "nan"); turn `start`/`end` = min/max of constituent rows.
- **Depth-score detection:** synthetic gap trace with known peaks/valleys →
  expected local maxima and depth values; min-spacing greedily drops the
  lower-depth neighbour; scenes with < 2W+2 turns yield no boundary.
- **AND-combination:** synthetic char/topic candidate lists → only pairs
  within ε survive; the surviving time equals the topic-side time; empty on
  either side yields no hybrid boundary.
- **Embedding cache:** round-trip write/read; text-hash mismatch invalidates
  and re-encodes.
- **scene_subdivide plumbing:** given a stub propose_fn, output row count,
  scene_id renumbering, and `[topic_aug N]` / inherited `scene_desc`
  suffixes match expectation (mirrors the char path's behaviour).

Integration smoke test: run `augment_scenes.py --mode topic` on one
calibration episode end-to-end and assert the augmented TSV parses and has
≥ the input scene count.

## Out of scope (YAGNI)

- Re-tuning char-presence parameters (frozen from prototype #1).
- Multi-task reason tagging (direction #4), spurious-marker filter
  (direction #3), supervised model (deferred) — unchanged from the eval doc.
- Touching `scene_desc` *content* quality — the eval never scores it.
- GPU inference, alternative/larger embedding models — only if MiniLM proves
  inadequate, revisited then.

## Success criteria

Not "F1 must go up" — this is a prototype. Success = a clean, reproducible
read on both questions:

1. The four configs are scored on the held-out s3–s6 split with the
   topic-shift parameters fit only on s1–s2.
2. We can state, with the bucket diagnostics, whether `topic` is an
   independent net-positive signal and whether `hybrid` recovers the
   precision `char` lost.
3. The result (positive or negative) is written up in
   `docs/scene_segmentation_evaluation.md` as a "Prototype #2" section, in
   the same honest style as Prototype #1.
```
