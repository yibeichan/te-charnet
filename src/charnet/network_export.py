# src/charnet/network_export.py
"""Builders for the network-metric brain export (per-scene + per-character).

Pure assembly over charnet.metrics — no metric logic, no I/O. Owns the stable
column schemas, empty-frame behavior, and measure validation that the export
script and its consumers rely on.
"""
from __future__ import annotations

import pandas as pd

from charnet import metrics
from charnet.models import SceneGraph

SCENE_NETWORK_COLUMNS = [
    "scene_id",
    "start",
    "end",
    "duration",
    "n_nodes",
    "n_edges",
    "density",
    "n_components",
    "n_interaction_edges",
    "interaction_density",
    "interaction_entropy",
]


def scene_network_trace(scene_graphs: list[SceneGraph]) -> pd.DataFrame:
    """One row per scene of structural network metrics, timestamped.

    Wraps ``metrics.scene_metrics``. ``start``/``end`` are stage-2
    network-coverage windows (speaker-bearing rows), not necessarily full
    01a scene spans. Returns an empty frame carrying SCENE_NETWORK_COLUMNS
    when ``scene_graphs`` is empty.
    """
    rows = [metrics.scene_metrics(sg) for sg in scene_graphs]
    if not rows:
        return pd.DataFrame(columns=SCENE_NETWORK_COLUMNS)
    df = pd.DataFrame(rows)
    missing = set(SCENE_NETWORK_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(SCENE_NETWORK_COLUMNS)
    if missing or extra:
        raise ValueError(
            f"scene_metrics schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return df[SCENE_NETWORK_COLUMNS]


CHARACTER_CENTRALITY_BASE_COLUMNS = ["scene_id", "start", "end", "character"]


def character_centrality_trace(
    scene_graphs: list[SceneGraph],
    measures: list[str],
) -> pd.DataFrame:
    """Per-scene x character centrality, timestamped, stable column order.

    Wraps ``metrics.centrality_timeseries``. Validates ``measures`` against
    ``metrics.SUPPORTED_CENTRALITY_MEASURES`` and raises ValueError on an
    unknown measure (``compute_centralities`` only logs). Returns an empty
    frame carrying the declared columns when there are no rows.
    """
    unknown = [m for m in measures if m not in metrics.SUPPORTED_CENTRALITY_MEASURES]
    if unknown:
        raise ValueError(
            f"unknown centrality measure(s): {', '.join(unknown)}. "
            f"Supported: {', '.join(metrics.SUPPORTED_CENTRALITY_MEASURES)}"
        )
    columns = CHARACTER_CENTRALITY_BASE_COLUMNS + list(measures)
    df = metrics.centrality_timeseries(scene_graphs, measures=measures)
    if df.empty:
        return pd.DataFrame(columns=columns)
    return df.reindex(columns=columns)


_COVERAGE_NOTE = (
    "Stage-2 network-coverage window (from speaker-bearing rows), not "
    "necessarily the full 01a scene span. Mapping to fMRI run time / TRs is "
    "the consumer's responsibility."
)

SCENE_NETWORK_DD = {
    "scene_id": {"Description": "Scene index within the episode."},
    "start": {"Description": f"Scene start. {_COVERAGE_NOTE}", "Units": "s"},
    "end": {"Description": f"Scene end. {_COVERAGE_NOTE}", "Units": "s"},
    "duration": {"Description": "end - start.", "Units": "s"},
    "n_nodes": {"Description": "Characters present in the scene interaction graph."},
    "n_edges": {"Description": "Edges (character pairs) in the scene graph."},
    "density": {"Description": "networkx graph density of the scene graph (0-1)."},
    "n_components": {"Description": "Connected components in the scene graph."},
    "n_interaction_edges": {"Description": "Edges with nonzero adjacency or proximity (true interaction, not pure co-presence)."},
    "interaction_density": {"Description": "n_interaction_edges / possible edges (0-1)."},
    "interaction_entropy": {"Description": "Shannon entropy (bits) of the scene's edge-weight distribution; higher = speech/interaction more evenly spread.", "Units": "bits"},
}

CHARACTER_CENTRALITY_DD = {
    "scene_id": {"Description": "Scene index within the episode."},
    "start": {"Description": f"Scene start. {_COVERAGE_NOTE}", "Units": "s"},
    "end": {"Description": f"Scene end. {_COVERAGE_NOTE}", "Units": "s"},
    "character": {"Description": "Character (node) the centrality row describes."},
    "degree": {"Description": "Weighted degree centrality (node strength share, 0-1, sums to 1 across nodes in a scene)."},
    "degree_unweighted": {"Description": "Unweighted networkx degree centrality (singleton scene = 0)."},
    "degree_weighted": {"Description": "Alias of degree (weighted strength share)."},
    "strength": {"Description": "Alias of degree (weighted strength share)."},
    "betweenness": {"Description": "Betweenness centrality on inverse-weight distances (stronger ties = shorter paths)."},
    "eigenvector": {"Description": "Eigenvector centrality (weighted); falls back to degree centrality on non-convergence."},
}
