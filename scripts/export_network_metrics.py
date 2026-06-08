# scripts/export_network_metrics.py
"""Export per-scene network metrics + per-character centrality as timestamped TSVs.

  python scripts/export_network_metrics.py --episodes s3-s6 \
      --network-root "$SCRATCH_DIR/output/02_build_network" \
      --out-dir output/annotations/network_metrics
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from charnet import network_export as nx_exp  # noqa: E402
from charnet.bids_meta import write_data_dictionary, write_dataset_description  # noqa: E402
from charnet.io import load_temporal_network  # noqa: E402
from charnet.scene_subdivide import expand_episode_spec  # noqa: E402
from charnet.transcript_align import normalize_episode_key  # noqa: E402

SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")
DEFAULT_SCENES_IN = REPO / "output/annotations/scenes"
DEFAULT_NETWORK_ROOT = Path(SCRATCH_DIR) / "output" / "02_build_network"
DEFAULT_OUT_DIR = REPO / "output/annotations/network_metrics"


def resolve_network_path(network_root: Path, episode: str) -> Path | None:
    """Locate an episode's temporal_network.json.

    Stage-2 dirs from run_pipeline.py use the friends_-prefixed key
    (normalize_episode_key), while expand_episode_spec yields bare IDs.
    Probe the normalized name first, then the bare ID, and return whichever
    exists (else None).
    """
    candidates = [normalize_episode_key(episode), episode]
    seen = []
    for name in candidates:
        if name in seen:
            continue
        seen.append(name)
        path = network_root / name / "temporal_network.json"
        if path.exists():
            return path
    return None


def _git_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "describe", "--tags", "--always", "--dirty"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _is_explicit_list(spec: str) -> bool:
    """True when --episodes names specific episodes (vs ALL / season / range)."""
    spec = spec.strip()
    if spec == "ALL":
        return False
    return re.fullmatch(r"s\d+(-s\d+)?", spec) is None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="ALL", help="ALL | sN | sN-sM | comma-list")
    ap.add_argument("--scenes-in", default=str(DEFAULT_SCENES_IN),
                    help="root used only to resolve episode specs")
    ap.add_argument("--network-root", default=str(DEFAULT_NETWORK_ROOT),
                    help="root holding <ep>/temporal_network.json (stage-2 output)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--measures", default="degree,betweenness,eigenvector")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    network_root = Path(args.network_root)
    measures = [m.strip().lower() for m in args.measures.split(",") if m.strip()]
    nx_exp.validate_measures(measures)  # fail fast on a bad --measures, before any I/O
    scenes_in = Path(args.scenes_in)
    episodes = expand_episode_spec(args.episodes, scenes_in)
    explicit = _is_explicit_list(args.episodes)

    # write schema sidecars first so the dir is self-describing on partial runs
    write_data_dictionary(out_dir / "scene_network.json", nx_exp.SCENE_NETWORK_DD)
    write_data_dictionary(out_dir / "character_centrality.json", nx_exp.CHARACTER_CENTRALITY_DD)
    write_dataset_description(
        out_dir.parent / "dataset_description.json",
        name="charnet Friends stimulus annotations",
        version=_git_version(),
        source_datasets=[{"Description": "Courtois NeuroMod Friends fMRI stimulus episodes"}],
    )

    print(f"Exporting network metrics for {len(episodes)} eps → {out_dir}")
    n_written = n_skipped = 0
    missing = []
    for ep in episodes:
        npath = resolve_network_path(network_root, ep)
        if npath is None:
            missing.append(ep)
            n_skipped += 1
            continue
        scene_graphs = load_temporal_network(npath)
        scene_df = nx_exp.scene_network_trace(scene_graphs)
        char_df = nx_exp.character_centrality_trace(scene_graphs, measures=measures)
        season = int(ep[1:3])
        ep_dir = out_dir / f"s{season}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        scene_df.to_csv(ep_dir / f"friends_{ep}_scene_network.tsv", sep="\t", index=False)
        char_df.to_csv(ep_dir / f"friends_{ep}_character_centrality.tsv", sep="\t", index=False)
        n_written += 1
        print(f"  {ep}: {len(scene_df)} scenes, {len(char_df)} character-rows")

    print(f"\nWrote {n_written} episodes ({n_skipped} missing network dirs)")

    if explicit and missing:
        sys.exit(f"error: no temporal_network.json for explicitly-named episode(s): "
                 f"{', '.join(missing)} (checked under {network_root})")
    if n_written == 0:
        sys.exit(f"error: 0 episodes written — check --scenes-in ({scenes_in}) has "
                 f"episode files, or --network-root ({network_root}) / SCRATCH_DIR")


if __name__ == "__main__":
    main()
