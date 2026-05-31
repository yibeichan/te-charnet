# charnet Data Products & Brain-Analysis Readiness

`charnet` turns Friends episodes into time-resolved narrative annotations: who
is on screen, who is speaking, what is said, the scene/segment structure, and
the time-evolving character-interaction network. This catalog inventories every
data product the pipeline emits and flags how each maps onto downstream brain
analysis.

**Downstream consumer.** The `brain-states-friends` project fits sticky
HDP-HMM brain states on the Courtois NeuroMod Friends fMRI (6 subjects watching
all of Friends in-scanner) and, in its `08`-series, tests **content ↔
brain-state correspondence** — does a given brain state carry information about
the narrative content the subject is processing? Its `08a` step already
converts `charnet` annotations into TR-aligned feature arrays; `08b` runs the
per-state AUC / circular-shift-permutation / HRF-lag correspondence tests. This
catalog is the map of what `charnet` can feed that pipeline.

## Product inventory

| Product (path under `output/`) | Producer | Unit | Key fields | Timestamped? | Brain-status |
|---|---|---|---|---|---|
| `annotations/scenes/sN/friends_<ep>_scene_summary.tsv` | `01a_extract_annotations.py` | per-scene | `scene_id, scene_desc, start, end, shot_ids` | yes (`start`/`end` s) | **consumed-by-08a** (scene boundaries) |
| `annotations/sentences/sN/friends_<ep>_sentence_speaker_table.tsv` | `01a` + `01b_fill_speakers.py` | per-sentence | `scene_id, start, end, speaker, utterance, utterance_ct, visual_presence, visual_presence_chars, scene_speaker_set, …` (29 cols) | yes (`start`/`end` s) | **consumed-by-08a** (dialogue + character + presence tiers) |
| `<network_dir>/temporal_network.json` | `02_build_network.py` | per-scene graph | per scene: `scene_id, start, end, nodes, edges` (weighted speaker adjacency / proximity) | yes (per-scene `start`/`end`) | **available-not-yet-exported** |
| `<network_dir>/episode_network.json` | `02_build_network.py` | per-episode graph | aggregated weighted interaction graph | episode-level only | available-not-yet-exported |
| `<analysis_dir>/centrality_timeseries.csv` | `03_analyze.py` (`metrics.centrality_timeseries`) | per-scene × character | `scene_id, start, end, character, degree, betweenness, eigenvector` | yes (`start`/`end` s) | **available-not-yet-exported** (highest-value social product) |
| `<viz_dir>/*.gexf`, plots | `04_visualize.py` | episode graph | Gephi export, figures | — | not brain-relevant (presentation) |
| `intermediate/sentence_embeddings/sN/<ep>.npz` | `charnet.topic_shift` (this session) | per-turn | `vecs` (MiniLM `all-MiniLM-L6-v2`), `key` (cache hash) | **no** (timestamps not stored in npz) | **available-not-yet-exported** (needs turn timestamps to be a product) |
| topic-shift trace (block cosine-distance over turns) | `charnet.topic_shift.block_distance_trace` | per-turn-gap | continuous semantic-change score | derivable (turn `end`), **not persisted** | **planned** |

`<network_dir>` / `<analysis_dir>` / `<viz_dir>` are the per-run output
directories configured by `run_pipeline.py`; they are not in the default
checkout's `output/` tree.

## Brain-analysis mapping

Grouping the products by the kind of feature they provide, and how each lines
up with the existing `08a` feature tiers:

| Feature family | charnet source | In `08a` today? | Gap this fills |
|---|---|---|---|
| **Event / boundary** | scene `start`/`end` (discrete boundaries) | yes — `scene_boundary` | discrete only; no *continuous* narrative-change signal |
| **Dialogue semantics** | `utterance` / `utterance_ct` text → turn embeddings | **no** — `08a` table marks `utterance` "Not yet used" | the entire semantic content of dialogue is currently absent from the brain analysis |
| **Social structure** | `centrality_timeseries.csv`, `temporal_network.json` | partial — only `n_scene_speakers`, `n_main_in_scene` | per-character centrality / interaction-graph structure unused |
| **Identity / presence** | `speaker`, `visual_presence`, character columns | yes — character speaking tier | (well covered) |

The three open gaps map directly onto the recent topic-shift work and its
by-products:

1. **Continuous topic-shift trace** → a parametric "how much is the topic
   shifting right now" regressor, richer than the binary `scene_boundary`.
2. **Per-turn dialogue embeddings** → dialogue *semantics* as content features
   (or raw vectors for RSA / encoding models against neural patterns).
3. **Network metrics over time** → social-structure regressors (centrality,
   interaction density) beyond a speaker count.

## Gaps & planned exports

To become brain-analysis-ready, three products need a documented, timestamped
export. Each is scoped as its own feature-export spec (lightest first):

1. **Topic-shift trace** *(planned — first export)* — persist
   `block_distance_trace` per episode as a timestamped continuous signal
   (turn-gap time → semantic-distance score). Currently computed during
   detection and discarded. Spec: `docs/brain_features/` (forthcoming).
2. **Network-metric features** *(available, needs export)* —
   `centrality_timeseries.csv` is already timestamped per scene and per
   character; the export task is mainly to document it as a brain product and,
   if needed, reshape per-character columns into a fixed feature layout.
3. **Dialogue embeddings** *(available, needs export — heaviest)* — the
   `.npz` cache holds per-turn MiniLM vectors but **no turn timestamps or
   text**, so it cannot stand alone as a feature product. The export task is to
   emit per-turn `(start, end, embedding)` records (turn timing is recoverable
   via `topic_shift.turns_by_scene` over the sentence table).

All three exports stay in `charnet`'s lane: they *produce* timestamped
annotation outputs. TR-alignment and the correspondence statistics remain the
`brain-states-friends` `08a`/`08b` pipeline's responsibility.
