# charnet Data Products & Brain-Analysis Readiness

`charnet` turns Friends episodes into time-resolved narrative annotations: who
is on screen, who is speaking, what is said, the scene/segment structure, and
the time-evolving character-interaction network. This catalog inventories every
data product the pipeline emits and flags how each maps onto downstream brain
analysis.

**Downstream consumers.** These products are designed for brain-analysis
pipelines that test **content ↔ brain-state correspondence** on the Courtois
NeuroMod Friends fMRI (6 subjects watching all of Friends in-scanner) — does a
given brain state carry information about the narrative content the subject is
processing? The reference design is the `brain-states-friends` project's
`08`-series (`08a` converts `charnet` annotations into TR-aligned feature
arrays; `08b` runs the per-state AUC / circular-shift-permutation / HRF-lag
correspondence tests); consumption of the three feature exports below is
planned as its own dedicated research project. Either way, this repo stays the
source of record: consumers pull the tracked TSVs (and regenerate/read the
embedding NPZs) from here.

## Product inventory

| Product (path under `output/`) | Producer | Unit | Key fields | Timestamped? | Brain-status |
|---|---|---|---|---|---|
| `annotations/scenes/sN/friends_<ep>_scene_summary.tsv` | `01a_extract_annotations.py` | per-scene | `scene_id, scene_desc, start, end, shot_ids` | yes (`start`/`end` s) | **consumer-ready** (scene boundaries; already TR-aligned by `brain-states-friends` `08a`) |
| `annotations/sentences/sN/friends_<ep>_sentence_speaker_table.tsv` | `01a` + `01b_fill_speakers.py` | per-sentence | `scene_id, start, end, speaker, utterance, utterance_ct, visual_presence, visual_presence_chars, scene_speaker_set, …` (29 cols) | yes (`start`/`end` s) | **consumer-ready** (dialogue + character + presence tiers; already TR-aligned by `brain-states-friends` `08a`) |
| `<network_dir>/temporal_network.json` | `02_build_network.py` | per-scene graph | per scene: `scene_id, start, end, nodes, edges` (weighted speaker adjacency / proximity) | yes (per-scene `start`/`end`) | **available-not-yet-exported** |
| `<network_dir>/episode_network.json` | `02_build_network.py` | per-episode graph | aggregated weighted interaction graph | episode-level only | available-not-yet-exported |
| `<analysis_dir>/centrality_timeseries.csv` | `03_analyze.py` (`metrics.centrality_timeseries`) | per-scene × character | `scene_id, start, end, character, degree, betweenness, eigenvector` | yes (`start`/`end` s) | **available-not-yet-exported** (highest-value social product) |
| `annotations/network_metrics/sN/friends_<ep>_scene_network.tsv` | `export_network_metrics.py` (`network_export.scene_network_trace`) | per-scene | `scene_id, start, end, duration, n_nodes, n_edges, density, n_components, n_interaction_edges, interaction_density, interaction_entropy` | yes (`start`/`end` s) | **exported** (per-scene social-structure regressors) |
| `annotations/network_metrics/sN/friends_<ep>_character_centrality.tsv` | `export_network_metrics.py` (`network_export.character_centrality_trace`) | per-scene × character | `scene_id, start, end, character, <measures…>` (default `degree`, `betweenness`, `eigenvector`) | yes (`start`/`end` s) | **exported** (per-character centrality regressors) |
| `<viz_dir>/*.gexf`, plots | `04_visualize.py` | episode graph | Gephi export, figures | — | not brain-relevant (presentation) |
| `intermediate/sentence_embeddings/sN/<ep>.npz` | `charnet.topic_shift` (this session) | per-turn | `vecs` (MiniLM `all-MiniLM-L6-v2`), `key` (cache hash) | **no** (timestamps not stored in npz) | **exported** → `annotations/dialogue_embeddings/` (feature #3) |
| `annotations/dialogue_embeddings/sN/friends_<ep>_dialogue_turns.tsv` | `export_dialogue_embeddings.py` | per-turn | `turn_id, scene_id, start, end, n_sentences` (row i indexes the companion NPZ's `vecs` row i) | yes (`start`/`end` s) | **exported** (tracked; turn timing for the embedding matrix) |
| `annotations/dialogue_embeddings/sN/friends_<ep>_dialogue_embeddings.npz` | `export_dialogue_embeddings.py` | per-turn | `vecs` (n_turns × 384 float32, MiniLM `all-MiniLM-L6-v2`), `key` (SHA256 of model_id + texts) | via companion TSV | **exported** (untracked; regenerable, verified by `verify_dialogue_embeddings.py`) |
| `annotations/topic_shift/sN/friends_<ep>_topic_trace.tsv` | `export_topic_trace.py` (`charnet.topic_shift.episode_topic_trace`) | per-turn-gap | `scene_id, onset, block_distance, depth, is_peak, w, tau_depth, min_spacing` | yes (`onset` s) | **exported** (tracked; continuous topic-shift regressor, verified by `verify_topic_trace.py`; `depth`/`is_peak` are the negative-result audit trail) |

`<network_dir>` / `<analysis_dir>` / `<viz_dir>` are the per-run output
directories configured by `run_pipeline.py`; they are not in the default
checkout's `output/` tree.

## Brain-analysis mapping

Grouping the products by the kind of feature they provide, and how each lines
up with the `brain-states-friends` `08a` feature tiers (the reference for what
TR-aligned consumption looks like):

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

## Feature exports

Three products needed a documented, timestamped export to become
brain-analysis-ready; all three are now shipped. Each was scoped as its own
feature-export spec (lightest first):

1. **Topic-shift trace** *(shipped)* — `scripts/export_topic_trace.py` persists
   the per-episode `block_distance_trace` as a timestamped continuous signal
   (turn-gap time → semantic-distance score) under
   `output/annotations/topic_shift/`, with a BIDS-inspired `topic_trace.json`
   sidecar. The detector's `is_peak`/`depth` columns are an audit trail over a
   documented negative result, not a validated segmentation.
2. **Network-metric features** *(shipped)* — `scripts/export_network_metrics.py`
   reads the stage-2 `temporal_network.json` and writes two timestamped TSVs to
   `output/annotations/network_metrics/`: a per-scene structural summary
   (`friends_<ep>_scene_network.tsv`: density, component count, interaction
   entropy, etc.) and a per-scene × character centrality table
   (`friends_<ep>_character_centrality.tsv`). BIDS-inspired sidecars
   (`scene_network.json`, `character_centrality.json`,
   `dataset_description.json`) document each column. `start`/`end` are
   stage-2 network-coverage windows (not raw scene boundaries).
   The full 341-episode (seasons 1–7) export is **committed** under
   `output/annotations/network_metrics/`; every value and the per-scene/character
   row set were verified against an independent recomputation off the stage-2
   networks (all 341 match within `1e-6`) via `scripts/verify_network_export.py`.
   The stage-2 inputs (`output/02_build_network/`) are **not** tracked — they are
   regenerable intermediates. To re-run the verifier on a fresh clone, first
   rebuild stage 2 (`python scripts/run_pipeline.py --season sN
   --skip-stages 1a,1b,3,4`, ~19 s/season), then
   `python scripts/verify_network_export.py`.

   The stage-2 networks themselves are independently re-checked by
   `scripts/verify_stage2_network.py`, which reconstructs each
   `temporal_network.json` and `episode_network.json` from scratch off the
   tracked `*_sentence_speaker_table.tsv` (no `charnet.network` import) and
   asserts structural invariants (all 341 match within `1e-6`). Together the two
   verifiers cover the full chain — tracked speaker table → stage-2 graph →
   network-metric export — so the untracked stage-2 intermediate is safe: every
   committed number is reproducible and checked from a tracked input. (Same
   precondition: rebuild stage 2 first on a fresh clone.)
3. **Dialogue embeddings** *(shipped)* — `scripts/export_dialogue_embeddings.py`
   writes a per-episode pair under `output/annotations/dialogue_embeddings/sN/`:
   `friends_<ep>_dialogue_turns.tsv` (**tracked**; `turn_id, scene_id, start,
   end, n_sentences` — row i indexes NPZ row i) and
   `friends_<ep>_dialogue_embeddings.npz` (**untracked**, ~160 MB total;
   `vecs` (n_turns × 384) float32 MiniLM `all-MiniLM-L6-v2`, `key` =
   SHA256(model_id + texts), same contract as the embedding cache).
   Regenerate: `python scripts/export_dialogue_embeddings.py --episodes ALL`.
   Verify: `python scripts/verify_dialogue_embeddings.py` — independent turn
   reconstruction off the tracked sentence tables (no `charnet` import), key
   check, and cache↔product vector binding; exit 0 on a full pass.
   BIDS-inspired `dialogue_turns.json` data dictionary; spec at
   `docs/superpowers/specs/2026-06-12-dialogue-embeddings-export-design.md`.

All three exports stay in `charnet`'s lane: they *produce* timestamped
annotation outputs. TR-alignment and the correspondence statistics remain the
consuming project's responsibility (the `brain-states-friends` `08a`/`08b`
pipeline is the reference design).
