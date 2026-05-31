# src/charnet/bids_meta.py
"""BIDS-inspired sidecar writers for charnet annotation products.

Not full BIDS (stimulus-level, not subject-level) — only the data-dictionary
and dataset_description conventions that apply to a derivative annotation set.
"""
from __future__ import annotations

import json
from pathlib import Path

BIDS_VERSION = "1.9.0"


def write_data_dictionary(path: Path, columns: dict[str, dict]) -> None:
    """Write a column data dictionary (`{col: {Description, Units, Levels}}`).

    Idempotent: overwrites *path* with the given mapping.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(columns, indent=2) + "\n")


def write_dataset_description(
    path: Path,
    *,
    name: str,
    version: str,
    source_datasets: list[dict] | None = None,
) -> None:
    """Write a BIDS-style derivative `dataset_description.json`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = {
        "Name": name,
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "charnet", "Version": version}],
        "SourceDatasets": source_datasets or [],
    }
    path.write_text(json.dumps(desc, indent=2) + "\n")
