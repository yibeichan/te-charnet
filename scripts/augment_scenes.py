"""Augment fan-transcript scene boundaries — char / topic / hybrid modes.

  python scripts/augment_scenes.py --mode topic --episodes s3-s6 \
      --scenes-out output/annotations/scenes_topic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from charnet import char_presence as cp  # noqa: E402
from charnet import topic_shift as ts  # noqa: E402
from charnet.scene_subdivide import Scene, expand_episode_spec, subdivide_episode  # noqa: E402
from charnet.visual_presence import (  # noqa: E402
    DEFAULT_CHAR_TRACKER_DIR, char_tracker_csv_path, load_char_tracker_grid,
    resolve_char_tracker_dir,
)

DEFAULT_SCENES_IN = REPO / "output/annotations/scenes"
DEFAULT_SENTENCES_IN = REPO / "output/annotations/sentences"
DEFAULT_CACHE_DIR = REPO / "output/intermediate/sentence_embeddings"
HYBRID_EPS = 3.0  # seconds: max gap for a char and topic boundary to count as agreeing


def _sentences_path(sentences_in: Path, episode: str) -> Path:
    season = int(episode[1:3])
    return sentences_in / f"s{season}" / f"friends_{episode}_sentence_speaker_table.tsv"


def _build_char_propose(episode, ct_dir, char_params):
    grid_path = char_tracker_csv_path(ct_dir, f"friends_{episode}")
    if grid_path is None:
        return None  # signal: no char data
    chars, rows = load_char_tracker_grid(grid_path)

    def propose(scene: Scene) -> list[float]:
        return cp.propose_sub_boundaries(
            scene.start, scene.end, chars, rows,
            tile_secs=cp.TILE_SECS, presence_frac=char_params["presence_frac"],
            jaccard_thresh=char_params["jaccard"], min_spacing=char_params["min_spacing"],
            persistence_tiles=char_params["persistence"],
            shot_times=None, shot_snap_window=cp.SHOT_SNAP_WINDOW,
            shot_snap_required=False, min_scene_length=cp.MIN_SCENE_LENGTH,
        )
    return propose


def _build_topic_propose(episode, sentences_in, encoder, cache_dir, params):
    spath = _sentences_path(sentences_in, episode)
    if not spath.exists():
        return None
    sents = pd.read_csv(spath, sep="\t")
    if "scene_id" not in sents.columns:
        raise ValueError(f"{spath}: missing 'scene_id' column")
    by_scene = ts.turns_by_scene(sents)
    # encode every turn text in the episode once (cached)
    flat_texts, index = [], {}
    for sid, turns in by_scene.items():
        index[sid] = (len(flat_texts), len(flat_texts) + len(turns))
        flat_texts.extend(t.text for t in turns)
    all_vecs = ts.embed_texts_cached(episode, flat_texts, encoder, cache_dir)

    def propose(scene: Scene) -> list[float]:
        turns = by_scene.get(scene.scene_id, [])
        lo, hi = index.get(scene.scene_id, (0, 0))
        vecs = all_vecs[lo:hi]
        return ts.propose_topic_boundaries(
            turns, vecs, w=params["w"], tau_depth=params["tau_depth"],
            min_spacing=params["min_spacing"],
        )
    return propose


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["char", "topic", "hybrid"])
    ap.add_argument("--episodes", default="ALL",
                    help="ALL | sN | sN-sM | comma-list (e.g. s01e01a,s02e03b)")
    ap.add_argument("--scenes-in", default=str(DEFAULT_SCENES_IN))
    ap.add_argument("--scenes-out", required=True)
    ap.add_argument("--sentences-in", default=str(DEFAULT_SENTENCES_IN))
    ap.add_argument("--char-tracker-dir", default=None)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--w", type=int, default=2)
    ap.add_argument("--tau-depth", type=float, default=0.3)
    ap.add_argument("--min-spacing", type=float, default=20.0)
    ap.add_argument("--eps", type=float, default=HYBRID_EPS)
    ap.add_argument("--char-jaccard", type=float, default=0.7)
    ap.add_argument("--char-min-spacing", type=float, default=30.0)
    ap.add_argument("--char-persistence", type=int, default=2)
    ap.add_argument("--char-presence-frac", type=float, default=0.20)
    args = ap.parse_args()

    scenes_in = Path(args.scenes_in)
    scenes_out = Path(args.scenes_out)
    sentences_in = Path(args.sentences_in)
    ct_dir = resolve_char_tracker_dir(args.char_tracker_dir) or Path(DEFAULT_CHAR_TRACKER_DIR)
    params = {"w": args.w, "tau_depth": args.tau_depth, "min_spacing": args.min_spacing}
    char_params = {
        "jaccard": args.char_jaccard,
        "min_spacing": args.char_min_spacing,
        "persistence": args.char_persistence,
        "presence_frac": args.char_presence_frac,
    }
    aug_tag = {"char": "char_aug", "topic": "topic_aug", "hybrid": "hybrid_aug"}[args.mode]

    episodes = expand_episode_spec(args.episodes, scenes_in)
    encoder = ts.minilm_encoder() if args.mode in ("topic", "hybrid") else None

    print(f"Augmenting {len(episodes)} eps | mode={args.mode} → {scenes_out}")
    totals = {"in": 0, "out": 0, "new": 0, "skipped": 0}
    for ep in episodes:
        char_propose = _build_char_propose(ep, ct_dir, char_params) if args.mode in ("char", "hybrid") else None
        topic_propose = _build_topic_propose(ep, sentences_in, encoder, Path(args.cache_dir), params) if args.mode in ("topic", "hybrid") else None

        if args.mode == "char":
            propose = char_propose
        elif args.mode == "topic":
            propose = topic_propose
        else:  # hybrid
            if char_propose is None or topic_propose is None:
                propose = None
            else:
                def propose(scene: Scene, _c=char_propose, _t=topic_propose) -> list[float]:
                    return ts.intersect_within(_c(scene), _t(scene), eps=args.eps)

        # propose is None when a builder found no input data (no char grid or
        # no sentence table) → pass the scene through unchanged.
        if propose is None:
            propose = lambda scene: []  # noqa: E731
            totals["skipped"] += 1

        r = subdivide_episode(ep, scenes_in, scenes_out, propose, aug_tag=aug_tag)
        totals["in"] += r["n_input_scenes"]
        totals["out"] += r["n_output_scenes"]
        totals["new"] += r["n_new_boundaries"]
        print(f"  {ep}: {r['n_input_scenes']:>3} → {r['n_output_scenes']:>3} (+{r['n_new_boundaries']})")

    print(f"\nTotals: {totals['in']} → {totals['out']} (+{totals['new']}); "
          f"{totals['skipped']} eps missing inputs")


if __name__ == "__main__":
    main()
