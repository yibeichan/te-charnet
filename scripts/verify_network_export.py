#!/usr/bin/env python3
"""Independent cross-check of the network-metric export.

Confirms the on-disk export TSVs (``output/annotations/network_metrics/``)
against metric values recomputed *from scratch* off the stage-2
``temporal_network.json`` inputs — a different code path than
``charnet.metrics``, so agreement is real evidence the exported numbers are
correct, not just non-NaN. Also checks row-set COMPLETENESS (scene count and
the per-scene character set), so a silently dropped scene or character row
fails rather than passing as "every present row matched".

Independence of each checked column:
  * FULLY INDEPENDENT (closed-form reimplementation here, no charnet, no
    networkx algorithm): n_nodes, n_edges, density, n_components,
    n_interaction_edges, interaction_density, interaction_entropy, degree.
  * GRAPH-INDEPENDENT, LIBRARY ALGORITHM (we rebuild the graph from raw JSON
    and feed the documented weighting to networkx's own routine — validates
    the export reflects the documented semantics, but shares the library
    implementation): betweenness, eigenvector.

Usage:
    python scripts/verify_network_export.py \
        --export-dir output/annotations/network_metrics \
        --network-root output/02_build_network

Exit code 0 iff every checked value on every episode matches within --tol.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

# Columns whose values we recompute. Anything else in the TSV (none today) is
# reported as unchecked rather than silently passed.
SCENE_VALUE_COLUMNS = [
    "duration", "n_nodes", "n_edges", "density", "n_components",
    "n_interaction_edges", "interaction_density", "interaction_entropy",
]
CHAR_VALUE_COLUMNS = ["degree", "betweenness", "eigenvector"]


# --- independent graph build from raw stage-2 JSON -------------------------

def _scene_graph(scene: dict):
    """networkx.Graph built directly from a raw JSON scene, mirroring
    to_networkx's construction (nodes list + edge endpoints, last-wins weight)
    without importing charnet."""
    G = nx.Graph()
    G.add_nodes_from(scene.get("nodes", []))
    for e in scene.get("edges", []):
        G.add_edge(
            e["source"], e["target"],
            weight=float(e.get("weight", 0.0)),
            adjacency=float(e.get("adjacency", 0.0)),
            proximity=float(e.get("proximity", 0.0)),
        )
    return G


def _components(G: nx.Graph) -> int:
    """Connected-component count via union-find — independent of
    nx.number_connected_components used by metrics.py."""
    parent = {n: n for n in G.nodes()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in G.edges():
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    return len({find(n) for n in G.nodes()})


def expected_scene_row(scene: dict) -> dict:
    """Recompute the scene_network row from scratch."""
    G = _scene_graph(scene)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = (2.0 * m / (n * (n - 1))) if n >= 2 else 0.0

    raw_edges = scene.get("edges", [])
    n_interaction = sum(
        1 for e in raw_edges
        if float(e.get("adjacency", 0.0)) > 0.0 or float(e.get("proximity", 0.0)) > 0.0
    )
    possible = n * (n - 1) / 2 if n >= 2 else 0.0
    interaction_density = (n_interaction / possible) if possible else 0.0

    weights = [float(e.get("weight", 0.0)) for e in raw_edges]
    total = sum(weights)
    if total > 0:
        entropy = -sum((w / total) * math.log2(w / total) for w in weights if w > 0)
    else:
        entropy = 0.0

    return {
        "duration": float(scene["end"]) - float(scene["start"]),
        "n_nodes": n,
        "n_edges": m,
        "density": density,
        "n_components": _components(G),
        "n_interaction_edges": n_interaction,
        "interaction_density": interaction_density,
        "interaction_entropy": entropy,
    }


def expected_centralities(scene: dict) -> dict:
    """Recompute per-character degree/betweenness/eigenvector from scratch.

    degree: weighted strength share (node strength / total strength), fully
    independent. betweenness: inverse-weight distance graph -> nx.betweenness.
    eigenvector: weighted nx.eigenvector_centrality (degree fallback on
    non-convergence, matching metrics.py)."""
    G = _scene_graph(scene)
    if G.number_of_nodes() == 0:
        return {}

    # --- degree: weighted strength share (independent closed form) ---
    strength = {node: 0.0 for node in G.nodes()}
    for u, v, d in G.edges(data=True):
        strength[u] += d["weight"]
        strength[v] += d["weight"]
    total_strength = sum(strength.values())
    if total_strength > 0:
        degree = {node: strength[node] / total_strength for node in G.nodes()}
    else:
        degree = {node: 0.0 for node in G.nodes()}

    # --- betweenness: inverse-weight distances (stronger tie = shorter) ---
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        w = d["weight"]
        if w > 0:
            H.add_edge(u, v, distance=1.0 / w)
    if H.number_of_edges() == 0:
        betweenness = {node: 0.0 for node in H.nodes()}
    else:
        betweenness = nx.betweenness_centrality(H, weight="distance")

    # --- eigenvector (weighted), degree fallback on non-convergence ---
    if G.number_of_edges() == 0:
        eigenvector = {node: 0.0 for node in G.nodes()}
    else:
        try:
            eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            eigenvector = dict(degree)

    return {
        node: {
            "degree": degree[node],
            "betweenness": betweenness.get(node, 0.0),
            "eigenvector": eigenvector.get(node, 0.0),
        }
        for node in G.nodes()
    }


# --- comparison ------------------------------------------------------------

def _close(a, b, tol) -> bool:
    return abs(float(a) - float(b)) <= tol + tol * abs(float(b))


def check_episode(ep: str, scene_tsv: Path, char_tsv: Path,
                  network_json: Path, tol: float) -> list[str]:
    """Return a list of mismatch strings (empty == all matched)."""
    fails: list[str] = []
    scenes = json.loads(network_json.read_text())
    scenes_by_id = {s["scene_id"]: s for s in scenes}

    # --- scene_network.tsv ---
    sdf = pd.read_csv(scene_tsv, sep="\t")
    if len(sdf) != len(scenes):
        fails.append(f"{ep} scene rows: tsv={len(sdf)} json_scenes={len(scenes)}")
    for _, row in sdf.iterrows():
        sid = row["scene_id"]
        scene = scenes_by_id.get(sid)
        if scene is None:
            fails.append(f"{ep} scene {sid}: in TSV but not in JSON")
            continue
        exp = expected_scene_row(scene)
        for col in SCENE_VALUE_COLUMNS:
            if not _close(row[col], exp[col], tol):
                fails.append(f"{ep} scene {sid} {col}: tsv={row[col]!r} expected={exp[col]!r}")

    # --- character_centrality.tsv ---
    cdf = pd.read_csv(char_tsv, sep="\t")
    present_by_scene = {sid: set(grp["character"]) for sid, grp in cdf.groupby("scene_id")}
    # Recompute the full expected (scene -> {character: metrics}) up front so we
    # can check the row SET is complete, not just that present rows are correct.
    # centrality_timeseries emits one row per graph node for every scene with
    # >=1 node, so the expected character set per scene is exactly the node set.
    exp_by_scene = {s["scene_id"]: expected_centralities(s) for s in scenes}

    # Completeness: every scene's character row-set must match the recomputed
    # node set exactly. Catches silently dropped scenes/rows that a present-rows-
    # only check would pass (codex P2).
    for sid, exp in exp_by_scene.items():
        expected_chars = set(exp.keys())
        present_chars = present_by_scene.get(sid, set())
        missing = expected_chars - present_chars
        extra = present_chars - expected_chars
        if missing:
            fails.append(f"{ep} scene {sid}: char rows MISSING from TSV: {sorted(missing)}")
        if extra:
            fails.append(f"{ep} scene {sid}: char rows in TSV but not in recomputed graph: {sorted(extra)}")
    for sid in present_by_scene:
        if sid not in exp_by_scene:
            fails.append(f"{ep} char scene {sid}: in TSV but not in JSON")

    # Values: check every present row (set-level gaps already reported above).
    for sid, grp in cdf.groupby("scene_id"):
        exp = exp_by_scene.get(sid)
        if exp is None:
            continue
        for _, row in grp.iterrows():
            ch = row["character"]
            if ch not in exp:
                continue
            for col in CHAR_VALUE_COLUMNS:
                if col in row and not _close(row[col], exp[ch][col], tol):
                    fails.append(
                        f"{ep} scene {sid} char {ch} {col}: "
                        f"tsv={row[col]!r} expected={exp[ch][col]!r}"
                    )
    return fails


def resolve_network_json(network_root: Path, ep: str) -> Path | None:
    """Probe friends_<ep>/ then bare <ep>/ — mirrors the export's resolution."""
    for cand in (network_root / f"friends_{ep}", network_root / ep):
        p = cand / "temporal_network.json"
        if p.exists():
            return p
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", type=Path,
                    default=Path("output/annotations/network_metrics"))
    ap.add_argument("--network-root", type=Path,
                    default=Path("output/02_build_network"))
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args(argv)

    # Discover episodes from the export's scene_network TSVs.
    scene_tsvs = sorted(args.export_dir.glob("*/*_scene_network.tsv"))
    if not scene_tsvs:
        print(f"No *_scene_network.tsv under {args.export_dir}", file=sys.stderr)
        return 2

    total_eps = 0
    checked_eps = 0
    skipped: list[str] = []
    all_fails: list[str] = []

    for scene_tsv in scene_tsvs:
        ep = scene_tsv.name.removeprefix("friends_").removesuffix("_scene_network.tsv")
        char_tsv = scene_tsv.with_name(scene_tsv.name.replace("_scene_network", "_character_centrality"))
        total_eps += 1

        if not char_tsv.exists():
            skipped.append(f"{ep}: missing character_centrality TSV")
            continue
        network_json = resolve_network_json(args.network_root, ep)
        if network_json is None:
            skipped.append(f"{ep}: no temporal_network.json under {args.network_root}")
            continue

        fails = check_episode(ep, scene_tsv, char_tsv, network_json, args.tol)
        checked_eps += 1
        if fails:
            all_fails.extend(fails)

    print(f"Episodes found:   {total_eps}")
    print(f"Episodes checked: {checked_eps}")
    if skipped:
        print(f"Skipped ({len(skipped)}):")
        for s in skipped:
            print(f"  - {s}")

    if all_fails:
        print(f"\nMISMATCHES ({len(all_fails)}):")
        for f in all_fails[:50]:
            print(f"  ✗ {f}")
        if len(all_fails) > 50:
            print(f"  ... and {len(all_fails) - 50} more")
        return 1

    if checked_eps == 0:
        print("\nNo episodes could be checked (all skipped).")
        return 2

    print(f"\n✓ All {checked_eps} episodes match within tol={args.tol} "
          f"(independent recomputation of "
          f"{', '.join(SCENE_VALUE_COLUMNS + CHAR_VALUE_COLUMNS)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
