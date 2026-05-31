"""Grid-search topic-shift params on the s1-s2 calibration split.

For each (W, tau_depth, min_spacing) combo: augment s1-s2 in the chosen mode
(topic or hybrid) to a temp tree, run evaluate_scene_segmentation.py against
it, parse aggregate.json, and record segment F1@5s, P@5s, R@5s plus the number
of new boundaries added. Prints a baseline row, a sorted grid, and the best combo.

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

Usage
-----
    python scripts/calibrate_topic_shift.py --mode topic
    python scripts/calibrate_topic_shift.py --mode hybrid
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
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
# Temp parent under output/ (not /tmp) so evaluate_scene_segmentation.py's
# relative_to(REPO) display logic doesn't crash; cleaned up at end of main().
_CALIB_TMP = REPO / "output" / "_calib_tmp"

# Expand once; both subprocesses get the same explicit episode CSV.
EPISODES: list[str] = expand_episode_spec("s1-s2", _ANNOTATIONS_SCENES)
EP_CSV: str = ",".join(EPISODES)


def _parse_agg(agg_path: Path) -> dict:
    agg = json.loads(agg_path.read_text())
    return {
        "seg_f1": float(agg["segment"]["F1@5s_mean"]),
        "seg_p": float(agg["segment"]["P@5s_mean"]),
        "seg_r": float(agg["segment"]["R@5s_mean"]),
    }


def _baseline_metrics() -> dict:
    """Evaluate the un-augmented annotations/scenes tree (no augment step)."""
    _CALIB_TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=_CALIB_TMP) as tmp:
        eval_out = Path(tmp) / "eval"
        subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/evaluate_scene_segmentation.py"),
                "--episodes",
                EP_CSV,
                "--ours-dir",
                str(_ANNOTATIONS_SCENES),
                "--out-dir",
                str(eval_out),
            ],
            check=True,
        )
        return _parse_agg(eval_out / "aggregate.json")


def _run_combo(w: int, tau: float, spacing: float, mode: str = "topic") -> dict:
    """Augment in *mode*, evaluate, return metrics dict with n_new."""
    _CALIB_TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=_CALIB_TMP) as tmp:
        scenes_out = Path(tmp) / "scenes"
        eval_out = Path(tmp) / "eval"

        aug_result = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/augment_scenes.py"),
                "--mode",
                mode,
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
            capture_output=True,
            text=True,
        )

        # Parse total new boundaries from the "Totals: A → B (+N); ..." line.
        n_new = 0
        for line in aug_result.stdout.splitlines():
            if line.startswith("Totals:"):
                matches = re.findall(r"\(\+(\d+)\)", line)
                if matches:
                    n_new = int(matches[0])
                break

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
        metrics = _parse_agg(eval_out / "aggregate.json")
        metrics["n_new"] = n_new
        return metrics


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calibrate topic-shift (or hybrid) params on the s1-s2 split."
    )
    ap.add_argument(
        "--mode",
        choices=["topic", "hybrid"],
        default="topic",
        help="Augmentation mode passed to augment_scenes.py (default: topic).",
    )
    args = ap.parse_args()
    mode = args.mode

    # --- Baseline (no augmentation) ---
    print("Computing baseline (no augmentation) …", flush=True)
    base = _baseline_metrics()
    print(
        f"BASELINE (no aug): segment F1@5s={base['seg_f1']:.4f}"
        f"  P@5s={base['seg_p']:.4f}  R@5s={base['seg_r']:.4f}\n",
        flush=True,
    )

    # --- Grid search ---
    results = []
    for w, tau, spacing in itertools.product(W_GRID, TAU_GRID, SPACING_GRID):
        m = _run_combo(w, tau, spacing, mode=mode)
        results.append((m["seg_f1"], m["seg_p"], m["seg_r"], m["n_new"], w, tau, spacing))
        print(
            f"W={w} tau={tau} spacing={spacing} ->"
            f" F1@5s={m['seg_f1']:.4f} P={m['seg_p']:.4f} R={m['seg_r']:.4f}"
            f" (+{m['n_new']} boundaries)",
            flush=True,
        )

    results.sort(reverse=True)
    print("\nTop 5 combos (sorted by segment F1@5s):")
    for f1, p, r, n_new, w, tau, spacing in results[:5]:
        print(
            f"  F1={f1:.4f} P={p:.4f} R={r:.4f} +{n_new:>4}b"
            f"  W={w} tau_depth={tau} min_spacing={spacing}"
        )
    best = results[0]
    print(
        f"\nBEST: W={best[4]} tau_depth={best[5]} min_spacing={best[6]}"
        f"  F1@5s={best[0]:.4f} P={best[1]:.4f} R={best[2]:.4f}"
        f" (+{best[3]} boundaries)"
    )

    # Drop the now-empty temp parent so it doesn't linger in the worktree.
    shutil.rmtree(_CALIB_TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
