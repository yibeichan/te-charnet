# Time-Evolved Character Interaction Network (te-charnet)

Time-Evolved Character Interaction Network from transcript + community transcript + shot boundaries.

## Cloning the repo

```bash
git clone --recursive git@github.com:yibeichan/te-charnet.git
cd te-charnet
```


## Environment

Create the environment and verify Python version:

```bash
micromamba env create -f environment.yaml
micromamba run -n charnet python --version
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

`aligned_rows.tsv` / `aligned_rows.json` are the canonical stage-1 outputs consumed downstream.

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
--skip-stages 3,4
```

### Optional single-episode overrides

For unusual layouts, you can still override inferred inputs:

- `--transcript <path>`
- `--shots <path>`
- `--community-transcript <path>`
- `--speaker-map <path>`
