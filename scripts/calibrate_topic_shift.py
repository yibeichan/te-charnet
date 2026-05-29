"""Grid-search topic-shift params on the s1-s2 calibration split.

For each (W, tau_depth, min_spacing) combo: augment s1-s2 in topic mode to a
temp tree, run evaluate_scene_segmentation.py against it, parse aggregate.json,
and record segment F1@5s. Prints a sorted grid and the best combo.

The embedding cache is shared (default output/intermediate/sentence_embeddings),
so turns are encoded once on the first combo and reused thereafter.

Episode-spec note
-----------------
``evaluate_scene_segmentation.py`` only accepts "ALL" or a comma-separated
explicit episode list — it does NOT understand season-range specs like "s1-s2".
``augment_scenes.py`` does understand "s1-s2" (via expand_episode_spec), but we
pass the explicit comma-list to both scripts for consistency.
The expansion is done once at module load via
``charnet.scene_subdivide.expand_episode_spec`` against the annotations/scenes
directory.
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO / "src"))
from charnet.scene_subdivide import expand_episode_spec  # noqa: E402

W_GRID = [1, 2, 3]
TAU_GRID = [0.2, 0.3, 0.4, 0.5]
SPACING_GRID = [15.0, 20.0, 30.0]

_ANNOTATIONS_SCENES = REPO / "output" / "annotations" / "scenes"

# Expand once; both subprocesses get the same explicit episode CSV.
EPISODES: list[str] = expand_episode_spec("s1-s2", _ANNOTATIONS_SCENES)
EP_CSV: str = ",".join(EPISODES)


def _seg_f1_at_5s(agg_path: Path) -> float:
    agg = json.loads(agg_path.read_text())
    return float(agg["segment"]["F1@5s_mean"])


def _run_combo(w: int, tau: float, spacing: float) -> dict:
    # Use a temp dir under output/ so evaluate_scene_segmentation.py's
    # relative_to(REPO) display logic doesn't crash on /tmp paths.
    _calib_tmp = REPO / "output" / "_calib_tmp"
    _calib_tmp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=_calib_tmp) as tmp:
        scenes_out = Path(tmp) / "scenes"
        eval_out = Path(tmp) / "eval"
        subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/augment_scenes.py"),
                "--mode",
                "topic",
                "--episodes",
                EP_CSV,
                "--scenes-out",
                str(scenes_out),
                "--w",
                str(w),
                "--tau-depth",
                str(tau),
                "--min-spacing",
                str(spacing),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/evaluate_scene_segmentation.py"),
                "--episodes",
                EP_CSV,
                "--ours-dir",
                str(scenes_out),
                "--out-dir",
                str(eval_out),
            ],
            check=True,
        )
        return {
            "seg_f1": _seg_f1_at_5s(eval_out / "aggregate.json"),
        }


def main() -> None:
    results = []
    for w, tau, spacing in itertools.product(W_GRID, TAU_GRID, SPACING_GRID):
        m = _run_combo(w, tau, spacing)
        results.append((m["seg_f1"], w, tau, spacing))
        print(
            f"W={w} tau={tau} spacing={spacing} -> seg_F1@5s={m['seg_f1']:.4f}",
            flush=True,
        )

    results.sort(reverse=True)
    print("\nTop 5 combos (segment F1@5s):")
    for f1, w, tau, spacing in results[:5]:
        print(f"  {f1:.4f}  W={w} tau_depth={tau} min_spacing={spacing}")
    best = results[0]
    print(
        f"\nBEST: W={best[1]} tau_depth={best[2]} min_spacing={best[3]}"
        f" (seg_F1@5s={best[0]:.4f})"
    )


if __name__ == "__main__":
    main()
