# te-charnet

Time-Evolved Character Interaction Network from transcript + community transcript + shot boundaries.

## What Changed (Current Pipeline)

The current stage-0/stage-1 flow is:

1. `00_preprocess.py`  
   Normalizes corrected transcript + shots + community transcript into structured intermediate files.
2. `01_segment_scenes.py`  
   Aligns timed utterances to community transcript, inserts `scene_desc`, assigns `shot_id`, and derives `scene_id` from `scene_desc` boundaries.

The stage-1 table is now the main scene segmentation output.

## Environment

Use the project micromamba env:

```bash
micromamba run -n charnet python --version
```

If needed:

```bash
micromamba env create -f environment.yaml
```

## Input Layout

Default structured paths (under `data/friends_annotations/annotation_results/`):

- `Speech2Text/s{season}/friends_sXXeYY{part}_model-AA_desc-wSpeaker_transcript.json`
- `TSVpyscene/s{season}/friends_sXXeYY{part}_pyscene.tsv`
- `community_based/s{season}/friends_sXXeYY_ufs.txt` (episode-level, no `a/b/c` suffix)

## Stage 0: Preprocess

### Single episode (recommended)

```bash
micromamba run -n charnet python scripts/00_preprocess.py --episode friends_s06e01a
```

Shorthand works too:

```bash
micromamba run -n charnet python scripts/00_preprocess.py --episode s06e01a
```

### Whole season

```bash
micromamba run -n charnet python scripts/00_preprocess.py --season s6
```

### Outputs (`output/00_preprocess/<episode>/`)

- `words.json`
- `utterances.json`
- `sentences.json`
- `shots.json` (if available)
- `community_events.json` (if available)
- `community_dialogues.json` (if available)

## Stage 1: Segment/Align

```bash
micromamba run -n charnet python scripts/01_segment_scenes.py --episode friends_s06e01a
```

Main outputs (`output/01_segment_scenes/<episode>/`):

- `aligned_rows.tsv`
- `aligned_rows.json`

Columns in aligned table:

- `start`, `end`, `shot_id`, `scene_id`, `speaker`, `utterance`, `speaker_ct`, `utterance_ct`, `scene_desc`

`scene_id` is derived from `scene_desc` boundaries (community-based scene markers), and propagated to dialogue rows.

Legacy output:

- `scenes.json` is still produced by default for compatibility with stage 2+.
- Disable with `--no-save-legacy-scenes` if you only need alignment table outputs.

## End-to-End (single episode)

```bash
micromamba run -n charnet python scripts/run_pipeline.py \
  --transcript data/friends_annotations/annotation_results/Speech2Text/s6/friends_s06e01a_model-AA_desc-wSpeaker_transcript.json \
  --shots data/friends_annotations/annotation_results/TSVpyscene/s6/friends_s06e01a_pyscene.tsv \
  --episode friends_s06e01a
```
