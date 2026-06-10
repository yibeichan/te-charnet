#!/usr/bin/env python3
"""Independent cross-check of the stage-2 networks.

Reconstructs each episode's per-scene and aggregate interaction graphs *from
scratch* off the tracked speaker tables (a different code path than
``charnet.network`` — nothing is imported from it) and confirms the committed
``output/02_build_network/<ep>/temporal_network.json`` and
``episode_network.json`` match, plus a layer of structural invariants. Together
with ``verify_network_export.py`` this independently verifies the whole chain:
tracked speaker table -> stage-2 graph -> network-metric export.

Usage:
    python scripts/verify_stage2_network.py \
        --tables-root output/annotations/sentences \
        --network-root output/02_build_network

Exit 0 iff every reconstruction value, every aggregate, and every invariant
holds; 1 on any mismatch/violation; 2 if nothing could be checked.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

DEFAULTS = {
    "weight_adjacency": 1.0,
    "weight_proximity": 0.5,
    "weight_copresence": 0.25,
    "proximity_window": 3,
}

# Half-ulp of 4-decimal rounding; used to size the weight-formula invariant slack.
HALF_ULP_4DP = 5e-5


def _round4(x: float) -> float:
    return round(float(x), 4)


def read_table_rows(path: Path) -> list[dict]:
    """Read a speaker TSV, mirroring stage-2's combined two-phase row filter.

    Replicates the combined production filter:
    load_corrected_speaker_rows (empty start) +
    build_temporal_network_from_aligned_rows (empty scene_id/speaker,
    coercion failures).

    keep_default_na=False keeps blank cells as "" (default pd.read_csv would
    make them NaN, which str()/float() would silently let through).
    Preserves TSV row order (matters for stable sort on tied start times).
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"scene_id", "start", "end", "speaker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required column(s): {sorted(missing)}")
    rows: list[dict] = []
    for _, r in df.iterrows():
        scene_raw = str(r.get("scene_id", "")).strip()
        start_raw = str(r.get("start", "")).strip()
        end_raw = str(r.get("end", "")).strip()
        speaker = str(r.get("speaker", "")).strip()
        if not start_raw:                      # phase (a): load_corrected_speaker_rows drops empty-start (scene-marker) rows
            continue
        if not scene_raw or not speaker:       # phase (b): build_temporal_network_from_aligned_rows drops empty scene_id/speaker
            continue
        try:
            scene_id = int(float(scene_raw))   # mirrors network.py's int(float(...)) coercion
            start = float(start_raw)
            end = float(end_raw)
        except (TypeError, ValueError):        # phase (b): network.py skips rows whose scene_id/start/end fail coercion
            continue
        rows.append({"scene_id": scene_id, "start": start, "end": end, "speaker": speaker})
    return rows


def _adjacency(speakers_seq: list[str]) -> dict[tuple[str, str], float]:
    counts: dict[tuple[str, str], float] = {}
    for i in range(1, len(speakers_seq)):
        a, b = speakers_seq[i - 1], speakers_seq[i]
        if a != b and a and b:
            key = tuple(sorted([a, b]))
            counts[key] = counts.get(key, 0.0) + 1.0
    return counts


def _proximity(speakers_seq: list[str], window: int) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for i, a in enumerate(speakers_seq):
        if not a:
            continue
        for j in range(i + 1, min(i + window + 1, len(speakers_seq))):
            b = speakers_seq[j]
            if not b or b == a:
                continue
            key = tuple(sorted([a, b]))
            scores[key] = scores.get(key, 0.0) + 1.0 / (j - i)
    return scores


def reconstruct_scenes(rows: list[dict], params: dict) -> list[dict]:
    """Rebuild per-scene graphs independently of charnet.network.

    `rows` must be in TSV order so the stable sort-by-start reproduces stage-2's
    tie handling. Returns scene dicts with edges keyed by sorted (a, b) pair.
    """
    grouped: dict[int, list[tuple[float, float, str]]] = {}
    for row in rows:
        grouped.setdefault(row["scene_id"], []).append(
            (row["start"], row["end"], row["speaker"])
        )

    wa, wp, wc = params["weight_adjacency"], params["weight_proximity"], params["weight_copresence"]
    window = params["proximity_window"]

    scenes: list[dict] = []
    for scene_id in sorted(grouped):
        turns = sorted(grouped[scene_id], key=lambda t: t[0])  # stable, start only
        if not turns:
            continue
        speakers_seq = [t[2] for t in turns]
        unique = sorted(set(speakers_seq))
        adj = _adjacency(speakers_seq)
        prox = _proximity(speakers_seq, window)

        pairs = set(itertools.combinations(unique, 2)) | set(adj) | set(prox)
        edges: dict[tuple[str, str], dict] = {}
        for pair in sorted(pairs):
            a, b = pair
            adj_v = adj.get(pair, 0.0)
            prox_v = prox.get(pair, 0.0)
            cop_v = 1.0 if (a in unique and b in unique) else 0.0
            w = wa * adj_v + wp * prox_v + wc * cop_v
            if w > 0:
                edges[pair] = {
                    "weight": _round4(w),
                    "adjacency": adj_v,
                    "proximity": _round4(prox_v),
                    "copresence": cop_v,
                }
        scenes.append({
            "scene_id": scene_id,
            "start": turns[0][0],
            "end": max(t[1] for t in turns),
            "nodes": unique,
            "edges": edges,
            "n_turns": len(turns),   # carried for the adjacency-bound invariant (Task 4)
        })
    return scenes


def aggregate_episode(scenes: list[dict]) -> dict:
    """Re-aggregate reconstructed scenes into the expected episode graph.

    Sums the already-rounded scene weight/proximity and raw adjacency/copresence
    (matching aggregate_episode_graph, which sums EdgeData.weight — already
    rounded), then rounds all four fields to 4 dp (matching 02_build_network's
    write-time rounding). Summing raw values and rounding once would diverge.
    """
    nodes: set[str] = set()
    agg: dict[tuple[str, str], dict] = {}
    for s in scenes:
        nodes.update(s["nodes"])
        for pair, e in s["edges"].items():
            tot = agg.setdefault(
                pair, {"weight": 0.0, "adjacency": 0.0, "proximity": 0.0, "copresence": 0.0}
            )
            for k in ("weight", "adjacency", "proximity", "copresence"):
                tot[k] += e[k]
    edges = {
        pair: {k: _round4(val) for k, val in attrs.items()} for pair, attrs in agg.items()
    }
    if scenes:
        start = min(s["start"] for s in scenes)
        end = max(s["end"] for s in scenes)
    else:
        start, end = 0.0, 0.0
    return {"start": start, "end": end, "n_scenes": len(scenes), "nodes": sorted(nodes), "edges": edges}


def _close(a, b, tol: float) -> bool:
    # Mixed abs+rel tolerance; the relative term uses abs(b), so pass the
    # reference (reconstruction) value as b. Sufficient for weights ~0-10 / counts.
    return abs(float(a) - float(b)) <= tol + tol * abs(float(b))


def _edges_by_pair(edge_list: list[dict]) -> dict[tuple[str, str], dict]:
    return {tuple(sorted((e["source"], e["target"]))): e for e in edge_list}


_VALUE_COLS = ("weight", "adjacency", "proximity", "copresence")


def _compare_edges(label: str, ep: str, sid, recon_edges: dict, committed_edges: dict,
                   tol: float, fails: list[str]) -> None:
    missing = set(recon_edges) - set(committed_edges)
    extra = set(committed_edges) - set(recon_edges)
    for pair in sorted(missing):
        fails.append(f"{ep} {label} {sid}: edge MISSING from committed JSON: {pair}")
    for pair in sorted(extra):
        fails.append(f"{ep} {label} {sid}: edge in committed JSON but not reconstructed: {pair}")
    for pair in sorted(set(recon_edges) & set(committed_edges)):
        for col in _VALUE_COLS:
            if not _close(committed_edges[pair][col], recon_edges[pair][col], tol):
                fails.append(
                    f"{ep} {label} {sid} {pair} {col}: "
                    f"committed={committed_edges[pair][col]!r} expected={recon_edges[pair][col]!r}"
                )


def _check_invariants(ep: str, committed_scenes: list[dict], recon_by_id: dict,
                      params: dict, tol: float, fails: list[str]) -> None:
    wa, wp, wc = params["weight_adjacency"], params["weight_proximity"], params["weight_copresence"]
    weight_slack = HALF_ULP_4DP * (1.0 + wp) + tol  # weight's own rounding + proximity propagation
    if committed_scenes:
        epi_start = min(s["start"] for s in committed_scenes)
        epi_end = max(s["end"] for s in committed_scenes)
    else:
        epi_start, epi_end = 0.0, 0.0
    for cs in committed_scenes:
        sid = cs["scene_id"]
        node_set = set(cs.get("nodes", []))
        seen_pairs: set[tuple[str, str]] = set()
        for e in cs.get("edges", []):
            a, b = e["source"], e["target"]
            pair = tuple(sorted((a, b)))
            if a == b:
                fails.append(f"{ep} scene {sid}: self-loop on {a}")
            if pair in seen_pairs:
                fails.append(f"{ep} scene {sid}: duplicate edge {pair}")
            seen_pairs.add(pair)
            if a not in node_set or b not in node_set:
                fails.append(f"{ep} scene {sid}: edge {pair} endpoint not in nodes")
            if e["copresence"] not in (0.0, 1.0):
                fails.append(f"{ep} scene {sid} {pair}: copresence {e['copresence']} not in {{0,1}}")
            adj = e["adjacency"]
            if adj < 0 or adj % 1 != 0:
                fails.append(f"{ep} scene {sid} {pair}: adjacency {adj} not a non-negative integer")
            n_turns = recon_by_id.get(sid, {}).get("n_turns")
            if n_turns is not None and adj > n_turns - 1:
                fails.append(f"{ep} scene {sid} {pair}: adjacency {adj} > n_turns-1 ({n_turns - 1})")
            formula = wa * adj + wp * e["proximity"] + wc * e["copresence"]
            if abs(float(e["weight"]) - formula) > weight_slack:
                fails.append(
                    f"{ep} scene {sid} {pair}: weight {e['weight']} != formula {formula:.6f} "
                    f"(slack {weight_slack:.2e})"
                )
        if cs["start"] > cs["end"]:
            fails.append(f"{ep} scene {sid}: start {cs['start']} > end {cs['end']}")
        if cs["start"] < epi_start - tol or cs["end"] > epi_end + tol:
            fails.append(f"{ep} scene {sid}: bounds outside episode span")


def check_episode(ep: str, table_path: Path, temporal_json: Path, episode_json: Path,
                  params: dict, tol: float) -> list[str]:
    fails: list[str] = []
    rows = read_table_rows(table_path)
    recon = reconstruct_scenes(rows, params)
    recon_by_id = {s["scene_id"]: s for s in recon}
    expected_epi = aggregate_episode(recon)

    committed_scenes = json.loads(temporal_json.read_text())
    committed_by_id = {s["scene_id"]: s for s in committed_scenes}

    # --- temporal_network.json ---
    exp_ids, got_ids = set(recon_by_id), set(committed_by_id)
    for sid in sorted(exp_ids - got_ids):
        fails.append(f"{ep} scene {sid}: reconstructed but absent from committed JSON")
    for sid in sorted(got_ids - exp_ids):
        fails.append(f"{ep} scene {sid}: in committed JSON but not reconstructed")
    for sid in sorted(exp_ids & got_ids):
        rs, cs = recon_by_id[sid], committed_by_id[sid]
        if not _close(rs["start"], cs["start"], tol):
            fails.append(f"{ep} scene {sid} start: committed={cs['start']!r} expected={rs['start']!r}")
        if not _close(rs["end"], cs["end"], tol):
            fails.append(f"{ep} scene {sid} end: committed={cs['end']!r} expected={rs['end']!r}")
        if set(rs["nodes"]) != set(cs.get("nodes", [])):
            fails.append(f"{ep} scene {sid} nodes mismatch")
        _compare_edges("scene", ep, sid, rs["edges"], _edges_by_pair(cs.get("edges", [])), tol, fails)

    # --- episode_network.json ---
    cepi = json.loads(episode_json.read_text())
    if set(expected_epi["nodes"]) != set(cepi.get("nodes", [])):
        fails.append(f"{ep} episode: node set mismatch")
    if expected_epi["n_scenes"] != cepi.get("n_scenes"):
        fails.append(f"{ep} episode: n_scenes committed={cepi.get('n_scenes')} expected={expected_epi['n_scenes']}")
    for key in ("start", "end"):
        if not _close(expected_epi[key], cepi.get(key, 0.0), tol):
            fails.append(f"{ep} episode {key}: committed={cepi.get(key)!r} expected={expected_epi[key]!r}")
    _compare_edges("episode", ep, "(agg)", expected_epi["edges"], _edges_by_pair(cepi.get("edges", [])), tol, fails)

    # --- structural invariants on committed graphs ---
    _check_invariants(ep, committed_scenes, recon_by_id, params, tol, fails)
    return fails


def _episode_key(table_path: Path) -> str:
    name = table_path.name.removesuffix("_sentence_speaker_table.tsv")
    return name.removeprefix("friends_")


def resolve_network_files(network_root: Path, ep: str) -> tuple[Path, Path] | None:
    """Probe friends_<ep>/ then bare <ep>/ for both stage-2 JSON files."""
    for cand in (network_root / f"friends_{ep}", network_root / ep):
        temporal = cand / "temporal_network.json"
        episode = cand / "episode_network.json"
        if temporal.exists() and episode.exists():
            return temporal, episode
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tables-root", type=Path, default=Path("output/annotations/sentences"))
    ap.add_argument("--network-root", type=Path, default=Path("output/02_build_network"))
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--weight-adjacency", type=float, default=DEFAULTS["weight_adjacency"])
    ap.add_argument("--weight-proximity", type=float, default=DEFAULTS["weight_proximity"])
    ap.add_argument("--weight-copresence", type=float, default=DEFAULTS["weight_copresence"])
    ap.add_argument("--proximity-window", type=int, default=DEFAULTS["proximity_window"])
    args = ap.parse_args(argv)

    params = {
        "weight_adjacency": args.weight_adjacency,
        "weight_proximity": args.weight_proximity,
        "weight_copresence": args.weight_copresence,
        "proximity_window": args.proximity_window,
    }

    tables = sorted(args.tables_root.glob("*/*_sentence_speaker_table.tsv"))
    if not tables:
        print(f"No *_sentence_speaker_table.tsv under {args.tables_root}", file=sys.stderr)
        return 2

    total = checked = 0
    skipped: list[str] = []
    all_fails: list[str] = []
    for table in tables:
        ep = _episode_key(table)
        total += 1
        resolved = resolve_network_files(args.network_root, ep)
        if resolved is None:
            skipped.append(f"{ep}: stage-2 JSON not found under {args.network_root} (regenerate stage 2)")
            continue
        temporal, episode = resolved
        fails = check_episode(ep, table, temporal, episode, params, args.tol)
        checked += 1
        all_fails.extend(fails)

    print(f"Episodes found:   {total}")
    print(f"Episodes checked: {checked}")
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
    if checked == 0:
        print("\nNo episodes could be checked (all skipped).")
        return 2
    print(f"\n✓ All {checked} episodes match within tol={args.tol} "
          f"(independent reconstruction + structural invariants).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
