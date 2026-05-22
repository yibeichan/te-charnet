# Speaker Identification: Method and Results Summary

Companion document for the per-season speaker tables in
`output/annotations/sentences/sN/*_sentence_speaker_table.tsv`. Describes how
each `speaker` label was assigned, what confidence to attach to it, and what
the per-method distribution looks like across all seven seasons of Friends.

- **Pipeline commit:** `8b9c365` on `main` (2026-05-22)
- **Scope:** Seasons 1–7, 341 episodes, 86,370 sentences
- **Visual source:** char-tracker stage-05 per-character per-second timestamps

## Where the data lives

All 341 per-episode speaker tables ship inside this repo at:

```
output/annotations/sentences/
├── s1/   48 files — friends_s01eNN[a|b]_sentence_speaker_table.tsv
├── s2/   48 files
├── s3/   50 files
├── s4/   48 files
├── s5/   48 files
├── s6/   50 files
└── s7/   49 files
```

Each filename has the form `friends_sNNeNN[a|b]_sentence_speaker_table.tsv`,
where `a` is the first half of a Friends episode and `b` is the second half
(this is the show's broadcast/Speech2Text segmentation, not an internal split).
Each file is one tab-separated UTF-8 table with a header row — see the
[Output schema](#output-schema--what-the-reviewer-sees) section at the bottom
of this document for every column's meaning.

**To load one file:**

```python
# Python
import pandas as pd
df = pd.read_csv(
    "output/annotations/sentences/s1/friends_s01e01a_sentence_speaker_table.tsv",
    sep="\t",
)
```

```r
# R
df <- read.delim(
  "output/annotations/sentences/s1/friends_s01e01a_sentence_speaker_table.tsv"
)
```

In Excel / Numbers: File → Open, choose "Text (Tab delimited)".

**To load all of a season** (Python):

```python
from pathlib import Path
import pandas as pd
season_dir = Path("output/annotations/sentences/s1")
df = pd.concat(
    [pd.read_csv(p, sep="\t").assign(episode=p.stem.replace("_sentence_speaker_table",""))
     for p in sorted(season_dir.glob("*.tsv"))],
    ignore_index=True,
)
```

## What the pipeline produces

For every sentence in every episode, the pipeline writes:

1. A `speaker` label (or leaves it empty and flags the row for human review).
2. A `speaker_method` tag recording which rule produced the label.
3. A `speaker_confidence` value in `{high, medium, low}` reflecting how
   trustworthy the assignment is in isolation.
4. An `annotation_confidence` value in `{high, medium, low}` that combines
   `speaker_confidence` with a *visual audit* — does char-tracker actually
   show the labeled speaker on screen during the sentence's time window? If
   not, the row is downgraded and `annotation_review_reason` is set
   (`speaker_offscreen`, `speaker_no_face`, or `low_speaker_confidence`).

## How each `speaker` label is assigned

The cascade runs in priority order. Most sentences (~83%) are resolved at the
very top — they carry a direct community-transcript word-level label or pass
a strong scene-context bridge — and never enter the inference rules. The
inference rules below only fire for the ~17% of sentences without a clean
upstream label.

### Stage 01a — `infer_speaker` (`src/charnet/speaker_fill.py:282-368`)

| Rule | Trigger | Method tag | Conf. |
|---|---|---|---|
| Non-dialogue rule | Utterance is a song lyric or stage direction. | `non_dialogue_rule` | high |
| Scene bridge | Previous and next speakers within the same scene are the same person. | `scene_context_bridge` | high |
| Name address | Utterance addresses prev or next speaker by name (signaling the other one is speaking). | `name_address_rule` | medium |
| **Visual disambiguation (primary)** | Two text-candidates {prev, next} are different; exactly one of them is visibly on screen per char-tracker. | `visual_disambiguation` | medium |
| Short turn alternation | ≤2 words between two known speakers — default to alternation. | `short_turn_alternation` | medium |
| Scene context continuation | Utterance starts with a continuation phrase ("and", "so") — favor previous speaker. | `scene_context_inference` | medium |
| Scene boundary fallback | Only one of {prev, next} is known. | `scene_context_inference` | medium |
| Two-sided ambiguous | Both neighbors known but different and no other signal. | `scene_context_inference` | medium (long line) / low (short line) |
| **Visual disambiguation (secondary)** | No text-context signal available, but exactly one scene-wide candidate is visibly on screen. | `visual_disambiguation` | medium |
| Scene majority | Most common known speaker in the scene. | `scene_majority_fallback` | low |
| Unresolved | None of the above fit. Speaker left empty; `review_flag=True` with reason `unresolved_no_reliable_inference \| missing_speaker`. | `unresolved` | low |

### Stage 01b — global QA re-resolution (`src/charnet/speaker_fill.py:736-772`)

After the entire season has been processed once, a global pass re-examines
rows that were flagged for review and upgrades any whose evidence is now
unambiguous:

| Method tag | Triggered when |
|---|---|
| `qa_review_resolved_bridge` | A previously-ambiguous row now has the same speaker on both sides within the same scene. |
| `qa_review_resolved_ct_support` | A previously-ambiguous row matches the community-transcript label. |
| `qa_review_resolved_short_bridge` | A short utterance now lies between two same-speaker turns. |
| `qa_speaker_ct_override` | Stage 01a label conflicted with a clear community-transcript label; CT wins. |
| `qa_scene_anomaly_override` | Single-row anomaly within an otherwise consistent scene; overridden by local pattern. |

## Results across all 7 seasons

**Where each speaker label came from (n=86,370 sentences):**

| Method | Count | % |
|---|---:|---:|
| `community_transcript_match` (CT word-level label) | 57,954 | 67.1 |
| `qa_review_resolved_ct_support` (01b QA, CT-confirmed) | 11,307 | 13.1 |
| `qa_review_resolved_bridge` (01b QA, same-speaker bridge) | 5,474 | 6.3 |
| **`visual_disambiguation` (new char-tracker rule)** | **3,938** | **4.6** |
| `scene_context_bridge` | 3,379 | 3.9 |
| `scene_context_inference` | 2,506 | 2.9 |
| `short_turn_alternation` | 1,542 | 1.8 |
| `name_address_rule` | 182 | 0.2 |
| `unresolved` (speaker NaN, review-flagged) | 85 | 0.1 |
| `non_dialogue_rule` | 3 | 0.0 |

**Resolution status:**

- **99.90% (86,285 / 86,370)** sentences received a speaker label.
- **0.10% (85 sentences)** were intentionally left unresolved with
  `review_flag=True` for human triage. These are not silent failures — they
  carry `review_reason="unresolved_no_reliable_inference | missing_speaker"`.

**Per-row `speaker_confidence` distribution:**

| Confidence | Count | % |
|---|---:|---:|
| high | 78,117 | 90.4 |
| medium | 7,490 | 8.7 |
| low | 763 | 0.9 |

**Per-row `annotation_confidence` distribution** (combines `speaker_confidence`
with the visual audit — see [`manual_review_speaker_offscreen.md`](manual_review_speaker_offscreen.md)
for the audit protocol):

| Confidence | Count | % |
|---|---:|---:|
| high | 62,095 | 71.9 |
| medium | 16,666 | 19.3 |
| low | 7,609 | 8.8 |

The drop from `speaker_confidence=high` (90.4%) to `annotation_confidence=high`
(71.9%) reflects sentences whose textual assignment was confident but whose
labeled speaker is not visible on screen — these are flagged for visual audit
without changing the underlying speaker label.

## Per-season breakdown

| Season | Episodes | Sentences | matched_to_ct | infer_speaker filled | Unresolved | high% | medium% | low% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s1 | 48 | 12,616 | 10,174 | 2,434 | 8 | 62.4 | 21.7 | 15.9 |
| s2 | 48 | 11,831 | 9,464 | 2,349 | 18 | 71.4 | 20.4 | 8.1 |
| s3 | 50 | 12,896 | 10,495 | 2,386 | 15 | 70.3 | 21.0 | 8.7 |
| s4 | 48 | 11,836 | 9,916 | 1,908 | 12 | 72.3 | 20.3 | 7.4 |
| s5 | 48 | 12,303 | 10,414 | 1,883 | 6 | 74.1 | 18.2 | 7.8 |
| s6 | 50 | 12,763 | 10,850 | 1,911 | 2 | 76.3 | 17.2 | 6.5 |
| s7 | 49 | 12,125 | 10,501 | 1,600 | 24 | 76.6 | 16.4 | 7.1 |
| **all** | **341** | **86,370** | **71,814** | **14,471** | **85** | **71.9** | **19.3** | **8.8** |

(`high/medium/low` columns are `annotation_confidence`. `matched_to_ct + infer_speaker filled + unresolved` ≠ total exactly because a small number of community-transcript-matched rows also flow through QA paths and are re-tagged.)

## Visual disambiguation: the new rule's contribution

US3 (feature 002) added the char-tracker visual-presence signal as a
conservative tiebreaker inside `infer_speaker`. Across all 7 seasons:

- **3,938 sentences (4.6%) had their speaker resolved via the visual rule** —
  cases where text context alone left two candidates and exactly one was
  visibly on screen.
- The rule **never introduces a brand-new speaker** beyond the text-derived
  candidate set. A silent on-screen extra cannot be promoted to speaker.
- Visual data also drives the post-assignment audit: 6,900 sentences (8.0%)
  carry the `speaker_offscreen` flag, indicating the labeled speaker is not
  visible during the time window. Whether each flag is a true positive is
  the subject of the manual-review protocol in
  [`manual_review_speaker_offscreen.md`](manual_review_speaker_offscreen.md)
  (SC-002: ≥90% expected to be correctly flagged).

## Output schema — what the reviewer sees

The TSVs at `output/annotations/sentences/sN/friends_sNNeNN[ab]_sentence_speaker_table.tsv`
have the following columns:

| Column | Meaning |
|---|---|
| `scene_id`, `sentence_id` | Scene and sentence index. |
| `start`, `end` | Sentence timing in seconds. |
| `utterance` | The dialogue text (from Speech2Text). |
| `speaker` | Final assigned speaker. May be empty for the 85 unresolved rows. |
| `speaker_confidence` | high/medium/low — confidence in the *text-based* assignment. |
| `speaker_method` | Which rule produced the label (see cascade above). |
| `alignment_score` | Score from the underlying CT alignment, when applicable. |
| `row_type` | `dialogue` / `song` / `direction`. |
| `filled_from_missing` | True if `infer_speaker` ran on this row (i.e., row was empty after CT match). |
| `matched_to_ct` | True if a community-transcript word-level label was available. |
| `scene_speaker_set` | All speakers in this scene, in order of first appearance. |
| `prev_speaker_scene`, `next_speaker_scene` | Adjacent speakers within the scene. |
| `review_flag`, `review_reason` | Set by 01a if the row needs human attention. |
| `utterance_ct`, `speaker_ct` | Underlying community-transcript text and label. |
| `speaker_original` | Pre-pipeline value (typically empty — the input has no labels). |
| `visual_presence` | `present` / `partial` / `absent` / `unknown` — any character visible during the window? |
| `visual_presence_source` | Always `char_tracker_stage05` in this run. |
| `visual_presence_chars` | Comma-separated list of characters char-tracker saw on screen. |
| `speaker_visual_presence` | `present` / `partial` / `absent` — was the *labeled* speaker on screen? |
| `speaker_visual_ratio` | Fraction of integer seconds in `[floor(start), ceil(end))` with labeled speaker visible. |
| `visual_presence_note` | Edge-case notes (e.g., sentence outside visual data range). |
| `annotation_confidence` | Final confidence after both text and visual signals. |
| `annotation_review_reason` | Why this row was downgraded (`speaker_offscreen`, `speaker_no_face`, `low_speaker_confidence`, `visual_presence_unknown`, `missing_speaker`). |

## Suggested reviewer focus

If the reviewer's time is constrained:

1. **Random sample from `annotation_confidence=low`** (7,609 rows). These are
   the pipeline's least-confident assignments and likely include genuine
   labeling errors.
2. **`annotation_review_reason=speaker_offscreen`** (6,900 rows). These have a
   confident text label but no visual confirmation. The blind-review protocol
   in [`manual_review_speaker_offscreen.md`](manual_review_speaker_offscreen.md)
   already samples 20 per season for this purpose.
3. **`speaker_method=unresolved`** (85 rows). Empty speaker, flagged. These
   genuinely need human input.
4. **`speaker_method=scene_majority_fallback`** (visible in `speaker_method`
   value counts; was 0 in this run but kept in the cascade as a last resort
   for future data).

## Provenance and reproducibility

- Code: `src/charnet/speaker_fill.py` (cascade), `src/charnet/transcript_align.py` (01a driver), `src/charnet/visual_presence.py` (char-tracker integration).
- Regenerate per-season outputs:
  ```bash
  for s in s1 s2 s3 s4 s5 s6 s7; do
    .venv/bin/python scripts/run_pipeline.py --season $s --skip-stages 2,3,4 -v \
      > logs/smoke_$s.log 2>&1
  done
  ```
- Smoke run: 2026-05-22, ~4 minutes for all 7 seasons end-to-end on the HPC head node.
- Stage-05 visual data: `/orcd/scratch/bcs/002/yibei/friends-char-track/output/05_character_timestamps/` (1,023 files, 100% s1–s7 episode coverage).
