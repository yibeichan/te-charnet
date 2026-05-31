# scripts/export_topic_trace.py
"""Export the continuous topic-shift trace per episode as a timestamped TSV.

  python scripts/export_topic_trace.py --episodes s3-s6 \
      --out-dir output/annotations/topic_shift
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from charnet import topic_shift as ts  # noqa: E402
from charnet.bids_meta import write_data_dictionary, write_dataset_description  # noqa: E402
from charnet.scene_subdivide import expand_episode_spec  # noqa: E402

DEFAULT_SCENES_IN = REPO / "output/annotations/scenes"
DEFAULT_SENTENCES_IN = REPO / "output/annotations/sentences"
DEFAULT_OUT_DIR = REPO / "output/annotations/topic_shift"
DEFAULT_CACHE_DIR = REPO / "output/intermediate/sentence_embeddings"

DATA_DICTIONARY = {
    "scene_id": {"Description": "Fan-transcript scene index the gap falls in"},
    "onset": {"Description": "Gap time: end of the turn before the gap, relative to episode start. Mapping to fMRI run time / TRs is the consumer's responsibility.", "Units": "s"},
    "block_distance": {"Description": "Cosine distance between mean-pooled w-turn blocks on either side of the gap; continuous topic-shift regressor. Higher = larger semantic shift.", "Units": "arbitrary (0-1 for normalized embeddings)"},
    "depth": {"Description": "TextTiling depth (rise above neighboring valleys) at local maxima of block_distance; NaN at non-maxima."},
    "is_peak": {"Description": "Gap accepted as a boundary by the topic-shift detector at the recorded params. NOTE: the detector is a documented negative result (docs/scene_segmentation_evaluation.md, Prototype #2); is_peak is an audit trail, not a validated boundary.", "Levels": {"true": "accepted", "false": "not accepted"}},
    "w": {"Description": "Block half-width in turns used to compute block_distance and is_peak."},
    "tau_depth": {"Description": "Depth threshold for is_peak."},
    "min_spacing": {"Description": "Minimum seconds between accepted peaks (greedy, deepest-first).", "Units": "s"},
}


def _git_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "describe", "--tags", "--always", "--dirty"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _sentences_path(sentences_in: Path, episode: str) -> Path:
    season = int(episode[1:3])
    return sentences_in / f"s{season}" / f"friends_{episode}_sentence_speaker_table.tsv"


def _episode_trace(episode, sentences_in, encoder, cache_dir, *, w, tau_depth, min_spacing):
    spath = _sentences_path(sentences_in, episode)
    if not spath.exists():
        return None
    sents = pd.read_csv(spath, sep="\t")
    if "scene_id" not in sents.columns:
        raise ValueError(f"{spath}: missing 'scene_id' column")
    by_scene = ts.turns_by_scene(sents)
    flat_texts, index = [], {}
    for sid, turns in by_scene.items():
        index[sid] = (len(flat_texts), len(flat_texts) + len(turns))
        flat_texts.extend(t.text for t in turns)
    all_vecs = ts.embed_texts_cached(episode, flat_texts, encoder, cache_dir)
    vecs_by_scene = {sid: all_vecs[lo:hi] for sid, (lo, hi) in index.items()}
    return ts.episode_topic_trace(by_scene, vecs_by_scene, w=w, tau_depth=tau_depth, min_spacing=min_spacing)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="ALL", help="ALL | sN | sN-sM | comma-list")
    ap.add_argument("--scenes-in", default=str(DEFAULT_SCENES_IN), help="root used only to resolve episode specs")
    ap.add_argument("--sentences-in", default=str(DEFAULT_SENTENCES_IN))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--w", type=int, default=1)
    ap.add_argument("--tau-depth", type=float, default=0.5)
    ap.add_argument("--min-spacing", type=float, default=30.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    episodes = expand_episode_spec(args.episodes, Path(args.scenes_in))
    encoder = ts.minilm_encoder()

    write_data_dictionary(out_dir / "topic_trace.json", DATA_DICTIONARY)
    write_dataset_description(
        out_dir.parent / "dataset_description.json",
        name="charnet Friends stimulus annotations",
        version=_git_version(),
        source_datasets=[{"Description": "Courtois NeuroMod Friends fMRI stimulus episodes"}],
    )

    print(f"Exporting topic trace for {len(episodes)} eps → {out_dir}")
    n_written = n_skipped = 0
    for ep in episodes:
        df = _episode_trace(ep, Path(args.sentences_in), encoder, Path(args.cache_dir),
                            w=args.w, tau_depth=args.tau_depth, min_spacing=args.min_spacing)
        if df is None:
            n_skipped += 1
            continue
        season = int(ep[1:3])
        ep_dir = out_dir / f"s{season}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(ep_dir / f"friends_{ep}_topic_trace.tsv", sep="\t", index=False)
        n_written += 1
        print(f"  {ep}: {len(df)} gaps")
    print(f"\nWrote {n_written} episodes ({n_skipped} missing sentence tables)")


if __name__ == "__main__":
    main()
