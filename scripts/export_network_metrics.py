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
