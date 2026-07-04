# Time-Evolved Character Interaction Network (te-charnet)

Builds time-evolving character-interaction networks for the *Friends* TV
series from three inputs — ASR transcripts, fan ("community") transcripts,
and shot boundaries — and exports timestamped annotation products (topic-shift
trace, network metrics, dialogue embeddings) for downstream brain-imaging
analysis of the Courtois NeuroMod Friends dataset.

Two things live here:

1. **The pipeline** (stages 1a–4): align transcripts → fill speakers → build
   per-scene interaction graphs → analyze → visualize.
2. **The annotation exports** (brain features #1–#3): versioned, verified,
   BIDS-inspired TSV/NPZ products under `output/annotations/`, consumed by the
   brain-states pipeline.

See [docs/data_products_catalog.md](docs/data_products_catalog.md) for the
full catalog of data products and their provenance.

## Cloning the repo

```bash
git clone --recursive git@github.com:yibeichan/te-charnet.git
cd te-charnet
```

## Environment

This project supports both **micromamba** and **uv** for environment
management. Pick whichever you prefer.

### Option A: micromamba

```bash
micromamba env create -f environment.yaml
micromamba run -n charnet python --version
```

### Option B: uv

This repo is configured as a non-package `uv` project. `uv sync` installs the
dependencies but does not install `charnet` as an editable/local package. The
scripts import from `src/` directly.

```bash
uv sync
uv run python --version
```

### Running scripts

Examples below use a bare `python`. Prefix with your environment runner:

```bash
micromamba run -n charnet python scripts/run_pipeline.py --help
# or
uv run python scripts/run_pipeline.py --help
```

## Input layout

Default structured paths (under `data/friends_annotations/annotation_results/`):

- `Speech2Text/s{season}/friends_sXXeYY{part}_model-AA_desc-wUtter_transcript.json`
- `TSVpyscene/s{season}/friends_sXXeYY{part}_pyscene.tsv`
- `community_based/s{season}/friends_sXXeYY_ufs.txt` (full-episode, no `a/b/c` suffix)

Note: ASR data is per half-episode part (`a`/`b`/...), while community
transcripts cover the full episode. The pipeline handles this automatically.

## Output layout

```
output/
  annotations/
    sentences/s{N}/            # canonical speaker-annotated sentence tables (tracked)
    scenes/s{N}/               # scene summaries: timing + shot boundaries (tracked)
    topic_shift/s{N}/          # per-gap topic-shift trace TSVs (tracked)
    network_metrics/s{N}/      # per-scene network + per-character centrality TSVs (tracked)
    dialogue_embeddings/s{N}/  # per-turn timing TSVs (tracked) + embedding NPZs (untracked)
    dataset_description.json   # shared sidecar; each export also has its own data dictionary JSON
    intermediate/              # alignment QA intermediates (01a_raw, 01b_enhanced, 01b_review, qa_reports)
  intermediate/
    sentence_embeddings/s{N}/  # shared MiniLM sentence-embedding cache (untracked, regenerable)
  02_build_network/{ep}/       # temporal + episode network graphs (untracked, regenerable)
  03_analyze/{ep}/             # metrics, centrality, edge stats (untracked)
  04_visualize/{ep}/           # figures (untracked)
  evaluation/                  # scene-segmentation evaluation results (tracked)
```

The tracking rule: **text products and their sidecars are tracked** so
downstream consumers get a versioned, validated copy; **binaries and
intermediates are untracked** because they are regenerable from tracked inputs
and bound to them by the verifiers below.

## Annotation exports (brain features)

Three timestamped products, each with a BIDS-inspired JSON data dictionary and
an **independent verifier** that recomputes the load-bearing values from the
tracked sentence tables without importing pipeline code:

| Product | What it is | Export | Verify |
|---|---|---|---|
| `topic_shift` | Per-turn-gap continuous topic-shift regressor (`onset`, `block_distance`; `depth`/`is_peak` are an audit trail of a documented negative result) | `export_topic_trace.py` | `verify_topic_trace.py` |
| `network_metrics` | Per-scene social-structure summary + per-character centrality (degree, betweenness, eigenvector) | `export_network_metrics.py` | `verify_network_export.py` |
| `dialogue_embeddings` | Per-turn timing TSV + MiniLM (384-d) embedding NPZ per episode | `export_dialogue_embeddings.py` | `verify_dialogue_embeddings.py` |

```bash
# regenerate everything (also refreshes the embedding cache when needed)
python scripts/export_topic_trace.py --episodes ALL
python scripts/export_network_metrics.py --episodes ALL
python scripts/export_dialogue_embeddings.py --episodes ALL

# verify everything (each exits non-zero on any mismatch)
python scripts/verify_topic_trace.py
python scripts/verify_network_export.py
python scripts/verify_dialogue_embeddings.py
python scripts/verify_stage2_network.py   # cross-checks the stage-2 graphs themselves
```

`--episodes` accepts `ALL`, a season (`s3`), a season range (`s3-s6`), or an
explicit comma-list (`s01e01a,s01e01b`). Explicitly named episodes that cannot
be produced are an error; season/`ALL` specs skip them.

## Pipeline

### Stage 1a: Extract annotations

Aligns ASR sentences to community-transcript dialogues via monotonic fuzzy
matching. Produces raw sentence tables and scene summaries.

```bash
python scripts/01a_extract_annotations.py --episode friends_s01e01a
python scripts/01a_extract_annotations.py --season s1
python scripts/01a_extract_annotations.py --episode friends_s01e01a --scene-summary-only
```

Raw sentence-table columns: `scene_id`, `sentence_id`, `start`, `end`,
`utterance`, `speaker`, `utterance_ct`, `speaker_ct`.

Scene-summary columns: `scene_id`, `scene_desc`, `start`, `end`, `shot_ids`.

### Stage 1b: Fill missing speakers

Fills missing speakers using cascading rules (CT matching, same-speaker
bridging, name-address, turn alternation, scene context) + cross-season global
QA. The final tables under `output/annotations/sentences/` carry 29 columns
(fill provenance, visual presence, scene speaker sets, …).

```bash
python scripts/01b_fill_speakers.py                # all seasons
python scripts/01b_fill_speakers.py --season s1    # single season
python scripts/01b_fill_speakers.py --skip-qa      # skip global QA
```

Config: `src/charnet/pipeline_config.yaml`

### Stage 2: Build network

```bash
python scripts/02_build_network.py --episode friends_s06e01a
```

Outputs: `temporal_network.json` (per-scene graphs), `episode_network.json`
(aggregate graph).

### Stage 3: Analyze

```bash
python scripts/03_analyze.py --episode friends_s06e01a
```

Outputs: `metrics.json`, `centrality_timeseries.csv`, `edge_birth_death.csv`

### Stage 4: Visualize

```bash
python scripts/04_visualize.py --episode friends_s06e01a
```

Outputs: `scene_networks/`, `episode/`, `scene_segments/`, `metrics/` (all
under `figures/`)

### End-to-end

```bash
python scripts/run_pipeline.py --episode friends_s06e01a   # or --season s6
python scripts/run_pipeline.py --episode s06e01a           # shorthand works
python scripts/run_pipeline.py --season s6 --skip-stages 1a,1b,3,4
```

## Tests

```bash
micromamba run -n charnet pytest -q          # or: uv run --extra dev pytest -q
micromamba run -n charnet ruff check .
```

## Further reading

- [docs/data_products_catalog.md](docs/data_products_catalog.md) — every data
  product, its producer, schema, and brain-relevance status
- [docs/scene_segmentation_evaluation.md](docs/scene_segmentation_evaluation.md) —
  scene-boundary evaluation + the documented negative results for the
  subdivision prototypes
- [docs/speaker_identification_summary.md](docs/speaker_identification_summary.md) —
  how speaker labels were derived and QA'd
