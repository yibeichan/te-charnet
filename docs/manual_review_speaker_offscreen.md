# Manual Review Protocol: `speaker_offscreen` Flag (SC-002)

Validates the `speaker_offscreen` audit flag produced by feature 002
(annotation imperfection handling, merged in PR #1, commit `5e9d812`).

## Why this exists

The 01a stage uses char-tracker stage-05 per-character per-second visibility to
flag any sentence where the labeled speaker is not visible during
`[start, end]`. These rows are marked `annotation_review_reason = speaker_offscreen`
and downgraded in `annotation_confidence`. The spec target (SC-002) is that
**≥90% of these flagged rows must be correctly downgraded** — i.e., the
labeled speaker really is off-screen during the clip.

A full s1–s7 smoke run (2026-05-22) produced **6,900 `speaker_offscreen`
flags across 86,370 sentences** (8.0% overall; per-season range 6.3–15.0%).
The 15.0% in s1 is consistent with the original feature-002 dev numbers.

## Files

All artefacts live under `output/manual_review/` (gitignored — the per-run
files do **not** travel with the repo). Regenerate any time with:

```bash
.venv/bin/python scripts/sample_offscreen_review.py
```

Default sample: 20 rows × 7 seasons = 140 rows. Seed `20260522 + season_number`,
so the same seed reproduces the same sample exactly.

Two files are emitted per run, stamped `_YYYYMMDD`:

| File | Purpose |
|---|---|
| `speaker_offscreen_sample_YYYYMMDD.tsv` | Blind review file (reviewer fills) |
| `speaker_offscreen_sample_YYYYMMDD_KEY.tsv` | Sealed answer key — do **not** open while reviewing |

Blind columns:

- `review_id` (`R001`–`R140`)
- `episode`, `sentence_id`
- `start_seconds`, `end_seconds`, `start_hms`, `end_hms`, `duration_seconds`
- `visible_during_clip` (blank — reviewer fills)
- `audible_speaker` (blank — reviewer fills)
- `notes` (blank — reviewer fills)

Key columns (joined back after review):

- `labeled_speaker` — what the pipeline assigned as the speaker
- `system_visual_presence`, `system_speaker_visual_presence`, `system_speaker_visual_ratio`
- `system_visual_presence_chars` — who char-tracker says is on screen
- `system_annotation_confidence`, `system_review_reason`

## Reviewer protocol

For each row in the blind TSV:

1. Open the episode video for `episode` (e.g. `friends_s02e01a`) in a player
   that lets you scrub to a precise timecode (iina, vlc, quicktime).
2. Watch the window `start_hms` → `end_hms`.
3. Fill **`visible_during_clip`** with the comma-separated lowercase first
   names of any characters whose face is on-screen at any point during the
   window. Use `none` if no character's face is visible.
   - Use canonical first names: `ross`, `rachel`, `monica`, `chandler`,
     `joey`, `phoebe`. For guests, use whatever name is recognizable
     (consistency matters; the scorer is case-insensitive and tolerates
     commas, semicolons, slashes, pipes as separators).
4. Fill **`audible_speaker`** with whose voice you hear during the window
   (often but not always the same set).
5. Use `notes` for ambiguity (e.g., "back of head only", "fast cut between
   two speakers", "off-screen but reflection visible in mirror").

**Do not open the KEY file while reviewing.** Verdict comparison happens
post-hoc.

## Scoring

After all 140 rows are filled, score with:

```bash
.venv/bin/python scripts/score_offscreen_review.py \
  --review-tsv output/manual_review/speaker_offscreen_sample_YYYYMMDD.tsv \
  --key-tsv    output/manual_review/speaker_offscreen_sample_YYYYMMDD_KEY.tsv
```

Verdict logic per row:

| `visible_during_clip` | Verdict |
|---|---|
| empty | `unreviewed` |
| `none` / `nobody` / etc. | `confirmed` (no one visible ⇒ labeled speaker is off-screen) |
| contains `labeled_speaker` (case-insensitive token match) | `disconfirmed` (false positive — labeled speaker IS visible) |
| any other non-empty value | `confirmed` |

Confirmation rate = `confirmed / (confirmed + disconfirmed)`. SC-002 passes if
≥0.90 (override with `--threshold`). The scorer prints an overall + per-season
table and writes `*.scored.tsv` joining blind + key + verdict so individual
false positives can be inspected.

## Regenerating from scratch

If the per-season sentence TSVs in `output/annotations/sentences/` are
missing or stale, regenerate them first via the smoke script:

```bash
for s in s1 s2 s3 s4 s5 s6 s7; do
  .venv/bin/python scripts/run_pipeline.py --season $s --skip-stages 2,3,4 -v \
    > logs/smoke_$s.log 2>&1
done
```

(This is the same loop used for the 2026-05-22 smoke run; takes about 4 minutes
end-to-end on the HPC head node.)

Then regenerate the sample + key with `sample_offscreen_review.py`. Same seed
produces an identical sample.

## Provenance

- Spec: `specs/002-annotation-imperfection-handling/spec.md` (SC-002,
  in-tree but gitignored — see PR #1 description for the criterion text).
- Implementation: `src/charnet/visual_presence.py`, `src/charnet/transcript_align.py`
  (the `_visual_tiebreaker` and `assess_annotation_confidence` paths).
- Smoke + sample run: 2026-05-22, on `main` at commit `6de9c3d`.
