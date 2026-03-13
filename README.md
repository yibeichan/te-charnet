# Time-Evolved Character Interaction Network (te-charnet)

Time-Evolved Character Interaction Network from transcript + community transcript + shot boundaries.

## Cloning the repo

```bash
git clone --recursive git@github.com:yibeichan/te-charnet.git
cd te-charnet
```

## Environment

This project supports both **micromamba** and **uv** for environment management. Pick whichever you prefer — all examples below show both.

### Option A: micromamba

```bash
micromamba env create -f environment.yaml
micromamba run -n charnet python --version
```

### Option B: uv

This repo is configured as a non-package `uv` project. `uv sync` installs the dependencies but does not install `charnet` as an editable/local package. The scripts import from `src/` directly.

```bash
uv sync
uv run python --version
```

For tests and other dev-only tools, include the `dev` extra:

```bash
uv run --extra dev pytest -q
```

### Running scripts

Throughout this README, examples use `micromamba run -n charnet python`. If you use uv, replace that prefix with `uv run python`:

```bash
# micromamba
micromamba run -n charnet python scripts/run_pipeline.py --help

# uv
uv run python scripts/run_pipeline.py --help
```

## Input Layout

Default structured paths (under `data/friends_annotations/annotation_results/`):

- `Speech2Text/s{season}/friends_sXXeYY{part}_model-AA_desc-wUtter_transcript.json`
- `TSVpyscene/s{season}/friends_sXXeYY{part}_pyscene.tsv`
- `community_based/s{season}/friends_sXXeYY_ufs.txt` (full-episode, no `a/b/c` suffix)

Note: ASR data is per half-episode part (`a`/`b`/...), while community transcripts cover the full episode. The pipeline handles this automatically.

## Stage 1a: Extract Annotations

Aligns ASR (Speech2Text) sentences to community-transcript dialogues using monotonic fuzzy alignment to extract speaker labels and scene segmentation.

```bash
# Single episode
micromamba run -n charnet python scripts/01a_extract_annotations.py --episode friends_s01e01a

# Whole season
micromamba run -n charnet python scripts/01a_extract_annotations.py --season s1

# Scene summary only (skip sentence table)
micromamba run -n charnet python scripts/01a_extract_annotations.py --episode friends_s01e01a --scene-summary-only
```

Main outputs (`output/map_speaker/s{season}/`):

- `friends_sXXeYY{part}_sentence_speaker_table.tsv` — sentence-level speaker annotations
- `friends_sXXeYY{part}_scene_summary.tsv` — per-scene start/end times and shot IDs

Sentence table columns: `scene_id`, `sentence_id`, `start`, `end`, `utterance`, `speaker`, `utterance_ct`, `speaker_ct`

Scene summary columns: `scene_id`, `scene_desc`, `start`, `end`, `shot_ids`

## Stage 1b: Fill Missing Speakers

Fills the missing speaker rows from Stage 1a using cascading inference rules (community-transcript matching, same-speaker bridging, name-address detection, turn alternation, scene context) followed by a cross-season global QA pass.

```bash
# All seasons
micromamba run -n charnet python scripts/01b_fill_speakers.py

# Single season
micromamba run -n charnet python scripts/01b_fill_speakers.py --season s1

# Skip global QA
micromamba run -n charnet python scripts/01b_fill_speakers.py --skip-qa
```

Main outputs:

- `output/map_speaker_enhanced/s{season}/` — enhanced TSVs with filled speakers and metadata columns (`speaker_confidence`, `speaker_method`, `alignment_score`, `row_type`, `filled_from_missing`, `review_flag`, etc.)
- `output/map_speaker_enhanced/s{season}_review/` — subset of rows flagged for manual review
- `output/map_speaker_final/global_qa_work/final_cleaned/` — post-QA cleaned TSVs
- `output/map_speaker_final/global_qa_work/reports/` — QA summary and change reports

The filling pipeline is defined in `src/charnet/speaker_fill.py` with tunable score thresholds in `src/charnet/pipeline_config.yaml`.

## Stage 2: Build Network

```bash
micromamba run -n charnet python scripts/02_build_network.py --episode friends_s06e01a
```

Main outputs (`output/02_build_network/<episode>/`):

- `temporal_network.json` (scene-level graphs)
- `episode_network.json` (aggregate half-episode graph)

## Stage 3: Analyze

```bash
micromamba run -n charnet python scripts/03_analyze.py --episode friends_s06e01a
```

Main outputs (`output/03_analyze/<episode>/`):

- `metrics.json`
- `centrality_timeseries.csv` (if non-empty)
- `edge_birth_death.csv` (if non-empty)

## Stage 4: Visualize

```bash
micromamba run -n charnet python scripts/04_visualize.py --episode friends_s06e01a
```

Main outputs (`output/04_visualize/<episode>/figures/`):

- `scene_networks/` (network plot per scene)
- `episode/` (aggregate network plots + exported graph files)
- `scene_segments/` (scene timeline and durations)
- `metrics/` (centrality and other metrics plots)

## End-to-End Pipeline

### Single episode (recommended)

```bash
micromamba run -n charnet python scripts/run_pipeline.py --episode friends_s06e01a
```

Shorthand also works:

```bash
micromamba run -n charnet python scripts/run_pipeline.py --episode s06e01a
```

### Whole season

```bash
micromamba run -n charnet python scripts/run_pipeline.py --season s6
```

You can skip stages with:

```bash
--skip-stages 1a,1b,3,4
```

### Optional single-episode overrides

For unusual layouts, you can still override inferred inputs:

- `--transcript <path>`
- `--shots <path>`
- `--community-transcript <path>`
- `--speaker-map <path>`
