# Scene Segmentation Evaluation vs Manual Annotations

Companion document for the per-episode scene summaries in
`output/annotations/scenes/sN/*_scene_summary.tsv`. Quantifies how well our
LLM-clustered scenes align with a human rater's hand-labelled scene/segment
structure, and traces residual disagreement to specific boundary-type signals.

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

## Finding 1 — Fixed-cardinality output

| Quantity | Mean ± SD |
|---|---|
| Predicted scenes / ep | 16.0 ± 5.7 |
| Gold scenes / ep | 11.0 ± 2.9 |
| Gold segments / ep | 21.3 ± 4.9 |

| Correlation | r |
|---|---|
| pred vs gold_scenes | −0.026 |
| pred vs gold_segments | 0.201 |

Our pipeline produces a roughly constant number of scenes regardless of how
many actual narrative units the episode contains. The near-zero correlation
with gold-scene count and the only-weak correlation with gold-segment count
indicate the segmenter is not responding to actual narrative density. Recall
is consequently bounded — on a 33-segment episode, 16 predictions cannot
cover all the boundaries.

This is almost certainly an LLM artefact: the cluster-budget is implicit in
the prompt / model, not derived from the input's information content.

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

## Improvement directions

Each direction below is grounded in a specific failure mode above.

### Calibrate output cardinality

**Evidence:** Finding 1 — corr(pred, gold) = 0.02 / 0.20; std 5.7 with no
correlation to episode complexity.

**Options:**
- Prompt-side: instruct the LLM to produce more scenes when the input has
  more shots / sentences / character changes; provide cardinality guidance
  scaled to input length.
- Post-hoc subdivision: split predicted scenes whose internal character set
  or topic embedding variance exceeds a learned threshold.
- Hierarchical generation: ask the LLM to first produce coarse
  location-stable blocks, then subdivide each into story beats.

### Add character-presence signal

**Evidence:** Finding 2 — charact_entry / charact_leave match at 38% / 36%.

We already consume char-tracker stage-05 (per-second per-character
timestamps) for speaker disambiguation. The same signal can yield a
character-set-change feature: when `set(chars at t)` differs from
`set(chars at t-ε)`, propose a candidate boundary. Cheap, deterministic,
directly targets the weakest single-reason category.

### Add topic-shift signal

**Evidence:** Finding 2 — goal_change is the worst category (31%).

Compute sentence-level embeddings (already produced upstream for the
speaker-fill cascade?) and detect topic shifts via embedding distance over a
sliding window. Combine with character-set change to disambiguate
"goal_change with no other cue" boundaries.

### Two-stage segmentation

**Evidence:** Findings 1 + 2 — exogenous cues are reliable; story-internal
ones need more signal.

Stage A: detect location / shot / music boundaries with high precision
(could be deterministic from char-tracker + shot table). Stage B: inside
each location-stable block, subdivide via character-set and topic-shift
signals. Lets each stage use its appropriate signal and makes errors
attributable.

### Multi-task / multi-output framing

**Evidence:** Finding 2 — different boundary types have wildly different
detectabilities and require different signals.

Rather than collapse all signals into a single "is this a boundary?" output,
predict each boundary type separately (location / character / topic / time
jump). This matches the manual annotation schema and lets downstream
consumers weight reasons by application (e.g., a network of conversational
turns probably cares more about character / goal boundaries than location).

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
