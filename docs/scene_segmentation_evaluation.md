# Scene Segmentation Evaluation vs Manual Annotations

Companion document for the per-episode scene summaries in
`output/annotations/scenes/sN/*_scene_summary.tsv`. Quantifies how well our
scene boundaries align with a human rater's hand-labelled scene/segment
structure, and traces residual disagreement to specific boundary-type signals.

**Mechanism note.** Our scenes are not LLM-generated. They come from
parsing community-written fan transcripts at `[Scene: ...]`, `(...)`, and
unrecognised-line markers (see `parse_community_transcript` in
`src/charnet/transcript_align.py`). Each Speech2Text sentence inherits a
scene id from monotonic alignment to a community-transcript dialogue, and
scene `start`/`end` are the min/max sentence timestamps per scene id. So
"our segmentation" is, operationally, the *fan transcript-writer's* scene
markers projected onto Speech2Text timestamps — not the output of any model
or clustering algorithm.

- **Pipeline commit (predictions):** `e245441` on `main` (2026-05-23)
- **Evaluator:** `scripts/evaluate_scene_segmentation.py`
- **Eval coverage:** Seasons 1–6, 292 half-episodes (s7 has no manual labels)
- **Outputs:** `output/evaluation/scene_segmentation/` (per-episode + aggregate + diagnostics)
- **Gold source:** `data/friends_annotations/annotation_results/manual_segmentation/`

## Data sources and units

The manual annotation defines two nested narrative units:

- **Scene** — "a section of an episode that takes place in one location in
  continuous time." Coarse; built from one or more segments.
- **Segment** — "a cohesive unit of story within an episode," delineated by
  any of seven boundary types (`ONbond_location`, `_charact_entry`,
  `_charact_leave`, `_time_jump`, `_goal_change`, `_music_transit`,
  `_theme_song`). Fine; a location-stable scene can contain multiple segments.

Our pipeline (`output/annotations/scenes/`) produces a single unit per row,
roughly named with location plus characters present. Each predicted scene
carries `start`/`end` in seconds and a list of contributing shot IDs.

The evaluator compares against BOTH units, since the right unit of comparison
is not a priori obvious.

## Methodology

For each episode:

1. **Window restriction.** Compare only on
   `t ∈ [max(gold_start, pred_start), min(gold_end, pred_end)]` so the
   theme-song / pre-roll our pipeline omits doesn't count as missed
   boundaries.
2. **Boundary detection.** Internal boundaries = ends of all but the last
   unit in the window. Match gold boundaries to predicted boundaries with a
   greedy nearest-first 1-to-1 pass within tolerance ε. Report
   precision / recall / F1 at ε ∈ {2, 5, 10} s.
3. **Per-unit IoU.** For each gold unit, find best-overlapping predicted
   scene, record IoU. Report mean / median / fraction ≥ 0.5.
4. **Boundary-type diagnostic.** For each gold *segment*-boundary in window,
   tag it with the ONbond_* reasons set True on the following segment, and
   record whether any predicted boundary fell within ±5 s. Tally match-rate
   by reason (single-reason boundaries and any-reason marginals).

Reproduce with `python scripts/evaluate_scene_segmentation.py`. Single-episode
mode: `--episodes s01e01a`.

## Headline results

**Means across 292 half-episodes:**

| Unit | F1@2s | F1@5s | F1@10s | P@5s | R@5s | mean IoU | IoU ≥ 0.5 |
|---|---|---|---|---|---|---|---|
| Scene | 0.150 | 0.402 | 0.497 | 0.365 | 0.492 | 0.462 | 48% |
| Segment | 0.217 | **0.498** | 0.596 | **0.617** | 0.441 | 0.442 | 44% |

Segment-unit F1 is higher overall, and precision is markedly higher there
(0.62 vs 0.37), meaning predicted boundaries usually correspond to a real
segment boundary even when they don't align with a scene boundary. Recall is
similar across units (0.44–0.49), bounded by our scene cardinality (below).

**Stability across seasons.** Segment F1@5s ranges 0.47–0.53; scene F1@5s
ranges 0.36–0.45 — small season-to-season variation despite the s1-4 vs s5-6
shift in theme-song handling.

## Finding 1 — Cardinality mismatch between fan transcripts and manual

| Quantity | Mean ± SD |
|---|---|
| Predicted scenes / ep | 16.0 ± 5.7 |
| Gold scenes / ep | 11.0 ± 2.9 |
| Gold segments / ep | 21.3 ± 4.9 |

| Correlation | r |
|---|---|
| pred vs gold_scenes | −0.026 |
| pred vs gold_segments | 0.201 |

Predicted cardinality lies between gold scenes and gold segments and is
nearly uncorrelated with either. This is the fan transcript-writer's choice
of what to mark with `[...]` brackets — primarily location changes, music
interludes, and time-lapse cards, with occasional character-entry notes and
essentially no goal/topic markers (see Finding 2). The transcript-writer is
not following manual's scene/segment definition; they are following their own
dramaturgical convention.

This re-frames the improvement question: not "calibrate a model's cluster
count" but "augment the transcript-writer's coverage with the boundary types
they don't mark."

## Finding 2 — Boundary-type performance

Match@5s for the 5,881 gold segment-boundaries in window, tagged by ONbond
reasons. **Single-reason** boundaries (carry exactly one reason):

| Boundary type | n | Match@5s |
|---|---|---|
| theme_song | 144 | 0.653 |
| location | 155 | 0.626 |
| music_transit | 970 | 0.623 |
| time_jump | 333 | 0.501 |
| (none) | 15 | 0.467 |
| charact_entry | 699 | 0.386 |
| charact_leave | 434 | 0.364 |
| goal_change | 417 | 0.307 |

Clean two-tier split:

- **Exogenous cues (~62%):** location swaps, music cues, theme song are
  signaled by shot / scenery change and naturally surface in our
  shot-clustering input. Our pipeline catches them.
- **Story-internal cues (~31–39%):** character entry / exit and topic /
  goal change require tracking *who is in the room* and *what they are
  talking about*. Our pipeline has no first-class signal for either, and
  recall reflects that.

The any-reason marginals (any boundary that carries the reason among others)
show the same ordering with smaller gaps, suggesting boundaries with
multiple reasons tend to be detected if at least one reason is exogenous.

## Improvement directions (ranked)

Each direction targets a specific failure mode from Findings 1–2. Directions
are ranked by expected-F1-lift × inverse-cost × diagnosability. #1 and #2
are independent and can be developed in parallel.

### 1. Character-presence subdivision  [highest priority]

**Evidence.** Finding 2: `charact_entry` and `charact_leave` are the worst
exogenously-attributable buckets (match@5s 0.39 / 0.36) and together account
for **1,133 single-reason boundaries** (19% of gold).

**Mechanism.** char-tracker stage-05 already provides per-second
per-character timestamps and is already consumed by the speaker-fill cascade.
Compute Jaccard distance between the character set in consecutive shots (or
fixed-width windows). When the set changes substantially inside an existing
fan-transcript scene, propose a candidate sub-boundary at the nearest
sentence end. Threshold-tunable.

**Expected lift.** Bounded above by [bucket size] × [target match rate
delta]. Pushing the two character buckets from ~37% to ~60% (the exogenous
tier) recovers ~260 boundaries → roughly **+4 to +5 pp absolute F1@5s** on
the segment unit, before counting cross-category boundaries that also carry
a character reason among others.

**Cost.** Low. char-tracker outputs are already loaded; new code is
~50 lines in `transcript_align.py` plus threshold validation against a
held-out episode set.

**Risk.** Over-firing on background-character drift. Mitigations: require
the changed character to be a known main character; require persistence
over a minimum number of shots.

**Diagnosability.** Direct: rerun the evaluator and read the bucket rows
for `charact_entry` / `charact_leave` in `boundary_diagnostics.tsv`.

**Reversibility.** Trivial — gate behind a flag.

### 2. Topic-shift subdivision  [secondary]

**Evidence.** Finding 2: `goal_change` is the lowest-detected category
(0.31, 417 single-reason boundaries). Fan transcripts essentially never mark
goal/topic shifts within a continuous-location scene.

**Mechanism.** Sentence embeddings + sliding-window cosine distance over each
fan-transcript scene's dialogue. Propose a sub-boundary where distance
exceeds threshold for a minimum window length.

**Expected lift.** Smaller bucket; pushing `goal_change` to 0.50 adds ~80
TPs → ~+1.4 pp F1. Modest but addresses the most semantically meaningful
boundary type for downstream character-network / social-cognition consumers.

**Cost.** Medium. Pick an embedding model (sentence-transformers MiniLM is
a sensible default), one-time forward pass over 86k sentences, threshold
calibration. New dependency.

**Risk.** Conversational dialogue is short and lexically sparse; embeddings
may not reliably separate story beats from in-beat reply variation. Pilot
first.

### 3. Spurious-marker filter  [investigative, lower priority]

**Evidence.** Predicted cardinality (16) > gold scene cardinality (11). Fan
transcripts mark stage-direction parentheticals and short interstitials that
manual annotators don't count.

**Mechanism.** Drop predicted scenes that (a) are shorter than N seconds,
or (b) match known interstitial patterns (`[Time Lapse]`, parenthesised stage
directions without a location). Tune N against scene-unit precision.

**Expected lift.** Precision-side; bounded — only ~5-scene gap to close at
the scene unit. Could *hurt* segment-unit metrics by removing fine-grained
boundaries that happen to align with manual segments.

**Cost.** Low logic, careful calibration needed.

### 4. Multi-task reason tagging  [orthogonal]

**Evidence.** Finding 2: different reasons have wildly different
detectabilities and likely matter differently to downstream consumers.

**Mechanism.** Tag each predicted boundary with its likely reason set
(location / character / topic / time) using char-tracker, shot transitions,
and embedding signals. Doesn't change segmentation itself.

**Expected lift.** Zero on these F1 metrics; enables consumers to weight
boundaries by application (e.g., a conversational-turn network probably
cares more about character/goal boundaries than music transitions).

**Cost.** Medium; mostly bookkeeping once #1 + #2 are built.

### Deferred: supervised model trained on manual annotations

The 292 manually-labelled episodes could train a boundary classifier over
shot / sentence / character-interval features. Defer until #1 + #2 set a
deterministic-signal ceiling — otherwise we won't know whether a learned
model is adding genuine value or just memorising the available signals.
Cross-validation on s1–s6 has to be designed carefully to avoid episode-arc
leakage.

### Suggested next step

Build #1 on a small held-out set first: pick ~10 episodes spanning seasons,
implement char-set Jaccard subdivision behind a flag, rerun the evaluator
in two-config mode (with / without the flag), and read the bucket diagnostics.
Promote to full pipeline only if the `charact_entry` / `charact_leave` rows
move in the right direction without hurting precision.

## What this evaluation does NOT measure

- **Content quality of scene descriptions.** Our pipeline produces a
  free-text `scene_desc` per scene — none of the metrics above touch it.
- **Single-half-episode bias.** Manual labels are at the half-episode level;
  full-episode narrative arcs aren't reconstructed.
- **Within-scene shot ordering.** We compare boundaries and intervals only.
- **The s7 special.** No manual labels for s7 (incl. the 4-part finale).

## Files

```
output/evaluation/scene_segmentation/
├── per_episode.tsv             584 rows (one per episode × unit)
├── per_episode_counts.tsv      292 rows (one per episode; scene / segment counts)
├── boundary_diagnostics.tsv    5,881 rows (one per gold segment-boundary in window)
└── aggregate.json              means, by-season, by-reason
```
