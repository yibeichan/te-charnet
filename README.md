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

## Output Layout

```
output/
  annotations/
    sentences/s{N}/        # canonical speaker-annotated sentence tables
    scenes/s{N}/           # scene summaries (timing, shot boundaries)
    intermediate/
      01a_raw/s{N}/        # raw alignment before speaker fill
      01b_enhanced/s{N}/   # enhanced with fill metadata columns
      01b_review/s{N}/     # rows flagged for manual review
      qa_reports/          # per-season summaries + global QA reports
  02_build_network/{ep}/   # temporal + episode network graphs
  03_analyze/{ep}/         # metrics, centrality, edge stats
  04_visualize/{ep}/       # figures (gitignored)
```

Downstream consumers (network building, scene clustering, brain-state mapping) read from `annotations/sentences/` and `annotations/scenes/`. Intermediate outputs are for debugging and auditing only.

## Pipeline

### Stage 1a: Extract Annotations

Aligns ASR sentences to community-transcript dialogues via monotonic fuzzy matching. Produces raw sentence tables and scene summaries.

```bash
python scripts/01a_extract_annotations.py --episode friends_s01e01a
python scripts/01a_extract_annotations.py --season s1
python scripts/01a_extract_annotations.py --episode friends_s01e01a --scene-summary-only
```

Sentence table columns: `scene_id`, `sentence_id`, `start`, `end`, `utterance`, `speaker`, `utterance_ct`, `speaker_ct`

Scene summary columns: `scene_id`, `scene_desc`, `start`, `end`, `shot_ids`

### Stage 1b: Fill Missing Speakers

Fills missing speakers using cascading rules (CT matching, same-speaker bridging, name-address, turn alternation, scene context) + cross-season global QA.

```bash
python scripts/01b_fill_speakers.py                # all seasons
python scripts/01b_fill_speakers.py --season s1     # single season
python scripts/01b_fill_speakers.py --skip-qa       # skip global QA
```

Config: `src/charnet/pipeline_config.yaml`

### Stage 2: Build Network

```bash
python scripts/02_build_network.py --episode friends_s06e01a
```

Outputs: `temporal_network.json` (per-scene graphs), `episode_network.json` (aggregate graph).

### Stage 3: Analyze

```bash
python scripts/03_analyze.py --episode friends_s06e01a
```

Outputs: `metrics.json`, `centrality_timeseries.csv`, `edge_birth_death.csv`

### Stage 4: Visualize

```bash
python scripts/04_visualize.py --episode friends_s06e01a
```

Outputs: `scene_networks/`, `episode/`, `scene_segments/`, `metrics/` (all under `figures/`)

### End-to-End

```bash
python scripts/run_pipeline.py --episode friends_s06e01a   # or --season s6
python scripts/run_pipeline.py --episode s06e01a           # shorthand works
python scripts/run_pipeline.py --season s6 --skip-stages 1a,1b,3,4
```
