#!/usr/bin/env python3
"""Prototype: align timed transcript with community transcript (speaker + scene) and export TSV."""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - optional fallback
    fuzz = None


@dataclass
class TimedSentence:
    idx: int
    start: float
    end: float
    speaker: str
    utterance: str
    norm: str


@dataclass
class CommunityEvent:
    idx: int
    kind: str  # "scene" | "dialogue"
    scene_desc: str = ""
    speaker_ct: str = ""
    utterance_ct: str = ""
    norm: str = ""
    dialogue_idx: int = -1


def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def normalize_for_match(text: str) -> str:
    text = _normalize_unicode(text).lower().strip()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_speaker(name: str) -> str:
    if not name:
        return ""
    text = _normalize_unicode(name).lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if fuzz is not None:
        ratio = fuzz.ratio(a, b) / 100.0
        token_ratio = fuzz.token_set_ratio(a, b) / 100.0
        ratio = 0.7 * ratio + 0.3 * token_ratio
    else:
        ratio = SequenceMatcher(None, a, b).ratio()
    ta = a.split()
    tb = b.split()
    if not ta or not tb:
        return ratio
    sa = set(ta)
    sb = set(tb)
    overlap = len(sa & sb) / max(1, min(len(sa), len(sb)))
    return 0.65 * ratio + 0.35 * overlap


def parse_community_transcript(path: Path) -> tuple[list[CommunityEvent], list[CommunityEvent]]:
    events: list[CommunityEvent] = []
    dialogues: list[CommunityEvent] = []
    dialogue_idx = 0

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    for line_idx, raw in enumerate(raw_lines):
        line = raw.strip()
        if not line:
            continue

        # Bracketed scene marker.
        if line.startswith("[") and line.endswith("]"):
            desc = line[1:-1].strip()
            desc = re.sub(r"^\s*scene\s*:\s*", "", desc, flags=re.IGNORECASE)
            events.append(CommunityEvent(idx=line_idx, kind="scene", scene_desc=desc))
            continue

        # Parenthetical narrative marker.
        if line.startswith("(") and line.endswith(")"):
            desc = line[1:-1].strip()
            events.append(CommunityEvent(idx=line_idx, kind="scene", scene_desc=desc))
            continue

        if line.lower() in {"opening credits", "closing credits"}:
            events.append(CommunityEvent(idx=line_idx, kind="scene", scene_desc=line))
            continue

        # Dialogue line.
        match = re.match(r"^([^:]{1,80}):\s*(.+)$", line)
        if match:
            speaker = match.group(1).strip()
            utterance = match.group(2).strip()
            ev = CommunityEvent(
                idx=line_idx,
                kind="dialogue",
                speaker_ct=speaker,
                utterance_ct=utterance,
                norm=normalize_for_match(utterance),
                dialogue_idx=dialogue_idx,
            )
            events.append(ev)
            dialogues.append(ev)
            dialogue_idx += 1
            continue

        # Unstructured text: keep as scene marker.
        events.append(CommunityEvent(idx=line_idx, kind="scene", scene_desc=line))

    return events, dialogues


def load_timed_sentences(path: Path) -> list[TimedSentence]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_sentences = data.get("sentences", [])
    sentences: list[TimedSentence] = []
    for idx, row in enumerate(raw_sentences):
        text = str(row.get("text", "")).strip()
        speaker = str(row.get("speaker", "")).strip()
        if not text:
            continue
        start = float(row.get("start", 0.0))
        end = float(row.get("end", start))
        sentences.append(
            TimedSentence(
                idx=idx,
                start=start,
                end=end,
                speaker=speaker,
                utterance=text,
                norm=normalize_for_match(text),
            )
        )
    return sentences


def align_monotonic(
    timed: list[TimedSentence],
    community: list[CommunityEvent],
    skip_timed_penalty: float = 0.6,
    skip_comm_penalty: float = 0.2,
) -> tuple[dict[int, int], dict[tuple[int, int], float]]:
    """Return mapping timed_idx -> community_dialogue_idx and pair similarity scores."""
    n = len(timed)
    m = len(community)
    neg_inf = -10**12

    dp = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]  # 1=diag, 2=up, 3=left

    dp[0][0] = 0.0
    for j in range(1, m + 1):
        dp[0][j] = 0.0  # free leading skips on community side (localize episode-half window)
        bt[0][j] = 3
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - skip_timed_penalty
        bt[i][0] = 2

    sim_cache: dict[tuple[int, int], float] = {}
    for i in range(1, n + 1):
        ts = timed[i - 1]
        ts_spk = normalize_speaker(ts.speaker)
        for j in range(1, m + 1):
            ct = community[j - 1]
            ct_spk = normalize_speaker(ct.speaker_ct)

            sim = text_similarity(ts.norm, ct.norm)
            sim_cache[(i - 1, j - 1)] = sim

            # Match score centered near zero so weak textual matches tend to be skipped.
            diag_score = (sim - 0.45) * 2.2
            if ts_spk and ct_spk:
                if ts_spk == ct_spk:
                    diag_score += 0.35
                else:
                    diag_score -= 0.05

            diag = dp[i - 1][j - 1] + diag_score
            up = dp[i - 1][j] - skip_timed_penalty
            left = dp[i][j - 1] - skip_comm_penalty

            if diag >= up and diag >= left:
                dp[i][j] = diag
                bt[i][j] = 1
            elif up >= left:
                dp[i][j] = up
                bt[i][j] = 2
            else:
                dp[i][j] = left
                bt[i][j] = 3

    end_j = max(range(m + 1), key=lambda j: dp[n][j])
    i, j = n, end_j
    mapping: dict[int, int] = {}
    while i > 0 and j > 0:
        move = bt[i][j]
        if move == 1:
            mapping[timed[i - 1].idx] = community[j - 1].dialogue_idx
            i -= 1
            j -= 1
        elif move == 2:
            i -= 1
        else:
            j -= 1

    return mapping, sim_cache


def scene_by_dialogue(events: list[CommunityEvent]) -> dict[int, str]:
    current = ""
    mapping: dict[int, str] = {}
    for ev in events:
        if ev.kind == "scene":
            current = ev.scene_desc
        else:
            mapping[ev.dialogue_idx] = current
    return mapping


def write_output_tsv(
    out_path: Path,
    timed: list[TimedSentence],
    community_dialogues: list[CommunityEvent],
    events: list[CommunityEvent],
    mapping: dict[int, int],
    sim_cache: dict[tuple[int, int], float],
    min_similarity: float,
) -> tuple[int, int]:
    by_dialogue = {d.dialogue_idx: d for d in community_dialogues}
    scene_map = scene_by_dialogue(events)

    fieldnames = [
        "start",
        "end",
        "speaker",
        "utterance",
        "speaker_ct",
        "utterance_ct",
        "scene_desc",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    last_scene = None
    matched = 0
    for row in timed:
        ct_idx = mapping.get(row.idx)
        ct = by_dialogue.get(ct_idx) if ct_idx is not None else None

        keep_match = False
        if ct is not None:
            sim = sim_cache.get((row.idx, ct_idx), 0.0)
            spk_match = normalize_speaker(row.speaker) == normalize_speaker(ct.speaker_ct)
            keep_match = sim >= min_similarity or (sim >= (min_similarity - 0.08) and spk_match)

        if keep_match and ct is not None:
            scene = scene_map.get(ct.dialogue_idx, "")
            if scene and scene != last_scene:
                rows.append(
                    {
                        "start": "",
                        "end": "",
                        "speaker": "",
                        "utterance": "",
                        "speaker_ct": "",
                        "utterance_ct": "",
                        "scene_desc": scene,
                    }
                )
                last_scene = scene

            rows.append(
                {
                    "start": f"{row.start:.2f}",
                    "end": f"{row.end:.2f}",
                    "speaker": row.speaker,
                    "utterance": row.utterance,
                    "speaker_ct": ct.speaker_ct,
                    "utterance_ct": ct.utterance_ct,
                    "scene_desc": "",
                }
            )
            matched += 1
        else:
            rows.append(
                {
                    "start": f"{row.start:.2f}",
                    "end": f"{row.end:.2f}",
                    "speaker": row.speaker,
                    "utterance": row.utterance,
                    "speaker_ct": "",
                    "utterance_ct": "",
                    "scene_desc": "",
                }
            )

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    return matched, len(rows)


def default_output_path(timed_json: Path) -> Path:
    name = timed_json.stem.replace("_model-AA_desc-wSpeaker_transcript", "")
    return timed_json.parent / f"{name}_community_aligned.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype alignment of timed transcript with community speaker/scene transcript."
    )
    parser.add_argument("--timed-json", type=Path, required=True, help="Path to Speech2Text JSON file.")
    parser.add_argument(
        "--community-txt",
        type=Path,
        required=True,
        help="Path to community transcript text file (episode-level).",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=None,
        help="Output TSV path. Defaults beside timed JSON.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.52,
        help="Minimum text similarity to accept a mapped pair (default: 0.52).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timed_json = args.timed_json
    community_txt = args.community_txt
    output_tsv = args.output_tsv or default_output_path(timed_json)

    if not timed_json.exists():
        raise FileNotFoundError(f"timed json not found: {timed_json}")
    if not community_txt.exists():
        raise FileNotFoundError(f"community txt not found: {community_txt}")

    timed = load_timed_sentences(timed_json)
    events, community_dialogues = parse_community_transcript(community_txt)
    mapping, sim_cache = align_monotonic(timed, community_dialogues)
    matched, nrows = write_output_tsv(
        out_path=output_tsv,
        timed=timed,
        community_dialogues=community_dialogues,
        events=events,
        mapping=mapping,
        sim_cache=sim_cache,
        min_similarity=args.min_similarity,
    )

    print(f"timed_sentences={len(timed)}")
    print(f"community_dialogues={len(community_dialogues)}")
    print(f"accepted_matches={matched}")
    print(f"output_rows={nrows}")
    print(f"output_tsv={output_tsv}")


if __name__ == "__main__":
    main()
