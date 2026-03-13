"""Transcript alignment: map community-transcript speakers to ASR sentence/word entries.

Moved from data/friends_annotations/src/map_speaker_code/map_speaker_from_community.py
into the charnet package for proper reuse across the pipeline.
"""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None


TRANSCRIPT_SUFFIXES = [
    "_model-AA_desc-wUtter_transcript",
    "_desc-wUtter_transcript",
    "_model-AA_desc-wSpeaker_transcript",
    "_desc-wSpeaker_transcript",
    "_transcript",
]


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

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
    return re.sub(r"\s+", " ", text).strip()


def text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if fuzz is not None:
        ratio = fuzz.ratio(a, b) / 100.0
        token_ratio = fuzz.token_set_ratio(a, b) / 100.0
        base = 0.7 * ratio + 0.3 * token_ratio
    else:
        base = SequenceMatcher(None, a, b).ratio()

    ta = a.split()
    tb = b.split()
    if not ta or not tb:
        return base
    overlap = len(set(ta) & set(tb)) / max(1, min(len(set(ta)), len(set(tb))))
    return 0.65 * base + 0.35 * overlap


# ---------------------------------------------------------------------------
# Episode / season helpers
# ---------------------------------------------------------------------------

def normalize_episode_key(value: str) -> str:
    key = value.strip()
    if re.fullmatch(r"s\d{2}e\d{2}[a-z]*", key, flags=re.IGNORECASE):
        key = f"friends_{key}"
    return key.lower()


def normalize_season_id(value: str) -> str:
    token = value.strip().lower()
    token = re.sub(r"^friends_", "", token)
    match = re.fullmatch(r"s?0*(\d+)", token)
    if not match:
        raise ValueError(f"Invalid season id: {value}")
    return f"s{int(match.group(1))}"


def episode_to_season_dir(episode: str) -> str:
    match = re.search(r"s0*(\d+)e\d{2}", episode, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not infer season from episode key: {episode}")
    return f"s{int(match.group(1))}"


def episode_to_community_root(episode: str) -> str:
    match = re.match(r"^(friends_s\d{2}e\d{2})([a-z]+)?$", episode, flags=re.IGNORECASE)
    return match.group(1).lower() if match else episode


def episode_from_transcript_filename(path: Path) -> str:
    stem = path.stem
    for suffix in TRANSCRIPT_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


# ---------------------------------------------------------------------------
# Community transcript parsing
# ---------------------------------------------------------------------------

def parse_community_transcript(path: Path) -> tuple[list[dict], dict[int, str]]:
    """Parse community transcript into dialogues and a scene_id → scene_desc mapping.

    Returns:
        dialogues: list of dialogue dicts with keys:
            dialogue_idx, scene_idx, speaker_ct, utterance_ct, norm
        scene_descs: dict mapping scene_id (int) → scene description text
    """
    dialogues: list[dict] = []
    scene_descs: dict[int, str] = {}
    current_scene = 0

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue

        # Scene markers: [Scene: ...] or [...]
        if re.match(r"^\[.*\]$", line):
            current_scene += 1
            desc = line[1:-1].strip()
            desc = re.sub(r"^\s*scene\s*:\s*", "", desc, flags=re.IGNORECASE)
            scene_descs[current_scene] = desc
            continue

        # Parenthesised scene descriptions
        if line.startswith("(") and line.endswith(")"):
            current_scene += 1
            scene_descs[current_scene] = line[1:-1].strip()
            continue

        # Credits lines
        if line.lower() in {"opening credits", "closing credits"}:
            current_scene += 1
            scene_descs[current_scene] = line
            continue

        # Dialogue: SPEAKER: utterance
        match = re.match(r"^([^:]{1,80}):\s*(.+)$", line)
        if match:
            # Ensure we have at least scene 1 if no marker appeared yet
            if current_scene == 0:
                current_scene = 1
                scene_descs[current_scene] = ""
            speaker = match.group(1).strip()
            utterance = match.group(2).strip()
            dialogues.append(
                {
                    "dialogue_idx": len(dialogues),
                    "scene_idx": current_scene,
                    "speaker_ct": speaker,
                    "utterance_ct": utterance,
                    "norm": normalize_for_match(utterance),
                }
            )
            continue

        # Unrecognised lines treated as scene markers
        current_scene += 1
        scene_descs[current_scene] = line

    return dialogues, scene_descs


# ---------------------------------------------------------------------------
# Monotonic alignment (DP)
# ---------------------------------------------------------------------------

def align_monotonic(
    sentences: list[dict],
    community_dialogues: list[dict],
    skip_sentence_penalty: float = 0.6,
    skip_community_penalty: float = 0.2,
) -> tuple[dict[int, int], dict[tuple[int, int], float]]:
    """Return sentence index -> community dialogue index mapping and similarity cache."""
    n = len(sentences)
    m = len(community_dialogues)
    neg_inf = -10**12

    dp = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]  # 1=diag, 2=up, 3=left

    dp[0][0] = 0.0
    for j in range(1, m + 1):
        dp[0][j] = 0.0
        bt[0][j] = 3
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - skip_sentence_penalty
        bt[i][0] = 2

    sim_cache: dict[tuple[int, int], float] = {}
    for i in range(1, n + 1):
        sent_norm = normalize_for_match(str(sentences[i - 1].get("text", "")))
        for j in range(1, m + 1):
            comm_idx = int(community_dialogues[j - 1]["dialogue_idx"])
            sim = text_similarity(sent_norm, str(community_dialogues[j - 1]["norm"]))
            sim_cache[(i - 1, comm_idx)] = sim

            diag_score = (sim - 0.45) * 2.2
            diag = dp[i - 1][j - 1] + diag_score
            up = dp[i - 1][j] - skip_sentence_penalty
            left = dp[i][j - 1] - skip_community_penalty

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
            mapping[i - 1] = int(community_dialogues[j - 1]["dialogue_idx"])
            i -= 1
            j -= 1
        elif move == 2:
            i -= 1
        else:
            j -= 1
    return mapping, sim_cache


# ---------------------------------------------------------------------------
# Multi-stage speaker filling
# ---------------------------------------------------------------------------

def _best_span_to_remove(source_text: str, remove_text: str, min_remove_similarity: float = 0.55) -> str:
    """Return normalized source_text with best fuzzy-matching span removed."""
    src_norm = normalize_for_match(source_text)
    rem_norm = normalize_for_match(remove_text)
    if not src_norm:
        return ""
    if not rem_norm:
        return src_norm

    src_tokens = src_norm.split()
    rem_tokens = rem_norm.split()
    if not src_tokens:
        return ""

    target_len = max(1, len(rem_tokens))
    lo = max(1, target_len - 3)
    hi = min(len(src_tokens), target_len + 3)

    best_sim = -1.0
    best_span: tuple[int, int] | None = None
    for span_len in range(lo, hi + 1):
        for start in range(0, len(src_tokens) - span_len + 1):
            end = start + span_len
            span_text = " ".join(src_tokens[start:end])
            sim = text_similarity(span_text, rem_norm)
            if sim > best_sim:
                best_sim = sim
                best_span = (start, end)

    if best_span is None or best_sim < min_remove_similarity:
        return src_norm
    start, end = best_span
    return " ".join(src_tokens[:start] + src_tokens[end:]).strip()


def _sentence_center(sent: dict) -> float:
    try:
        start = float(sent.get("start", 0.0))
        end = float(sent.get("end", start))
    except (TypeError, ValueError):
        return 0.0
    return (start + end) / 2.0


def _split_utterance_chunks(text: str) -> list[tuple[str, str]]:
    """Split a long community utterance into sentence-like chunks."""
    cleaned = _normalize_unicode(text)
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
    cleaned = cleaned.replace("...", ". ")

    chunks: list[tuple[str, str]] = []
    for part in re.split(r"[.!?;]+", cleaned):
        raw = re.sub(r"\s+", " ", part).strip(" ,:-")
        norm = normalize_for_match(raw)
        if norm:
            chunks.append((raw, norm))

    if chunks:
        return chunks

    fallback_raw = re.sub(r"\s+", " ", cleaned).strip()
    fallback_norm = normalize_for_match(fallback_raw)
    if not fallback_norm:
        return []
    return [(fallback_raw, fallback_norm)]


def _best_chunk_index(target_norm: str, chunk_norms: list[str]) -> tuple[int, float, float]:
    """Return best and second-best chunk similarity for ambiguity checks."""
    scores = [text_similarity(target_norm, chunk_norm) for chunk_norm in chunk_norms]
    best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
    best_score = scores[best_idx]
    second_best_score = sorted(scores, reverse=True)[1] if len(scores) > 1 else -1.0
    return best_idx, best_score, second_best_score


def _set_anchor_chunk_fill(
    row: dict,
    anchor: dict,
    anchor_idx: int,
    chunk_text: str,
    similarity: float,
) -> None:
    row["speaker_ct"] = anchor.get("speaker_ct")
    row["speaker_mapped"] = anchor.get("speaker_ct")
    row["utterance_ct"] = chunk_text
    row["comm_dialogue_idx"] = anchor.get("comm_dialogue_idx")
    if row.get("scene_id") is None and anchor.get("scene_id") is not None:
        row["scene_id"] = anchor.get("scene_id")
    row["match_similarity"] = round(similarity, 4)
    row["filled_from_anchor_idx"] = anchor_idx
    row["fill_method"] = "anchor_chunk"


def fill_unmatched_from_anchor_chunks(
    sentences: list[dict],
    anchor_min_similarity: float = 0.6,
    expand_min_similarity: float = 0.72,
    ambiguity_margin: float = 0.05,
    short_sent_token_threshold: int = 4,
    short_sent_min_similarity: float = 0.40,
) -> int:
    """Fill empty rows by expanding from mapped anchor rows using chunked utterance_ct."""
    anchors: list[tuple[float, int]] = []
    for i, sent in enumerate(sentences):
        if not sent.get("speaker_ct"):
            continue
        if not str(sent.get("utterance_ct", "")).strip():
            continue
        try:
            sim = float(sent.get("match_similarity", 0.0))
        except (TypeError, ValueError):
            sim = 0.0
        anchors.append((sim, i))

    anchors.sort(reverse=True)
    n_filled = 0

    for _, anchor_idx in anchors:
        anchor = sentences[anchor_idx]
        if not anchor.get("speaker_ct"):
            continue
        anchor_norm = normalize_for_match(str(anchor.get("text", "")))
        if not anchor_norm:
            continue

        chunks = _split_utterance_chunks(str(anchor.get("utterance_ct", "")))
        if len(chunks) < 2:
            continue
        chunk_norms = [norm for _, norm in chunks]

        best_idx, best_score, second_best = _best_chunk_index(anchor_norm, chunk_norms)
        if best_score < anchor_min_similarity:
            continue
        if second_best >= 0 and (best_score - second_best) < ambiguity_margin:
            continue

        row_idx = anchor_idx - 1
        chunk_idx = best_idx - 1
        while row_idx >= 0 and chunk_idx >= 0:
            row = sentences[row_idx]
            if row.get("speaker_ct"):
                break
            row_norm = normalize_for_match(str(row.get("text", "")))
            if not row_norm:
                break
            effective_min = (
                short_sent_min_similarity
                if len(row_norm.split()) < short_sent_token_threshold
                else expand_min_similarity
            )
            sim = text_similarity(row_norm, chunk_norms[chunk_idx])
            if sim < effective_min:
                break
            _set_anchor_chunk_fill(
                row=row,
                anchor=anchor,
                anchor_idx=anchor_idx,
                chunk_text=chunks[chunk_idx][0],
                similarity=sim,
            )
            n_filled += 1
            row_idx -= 1
            chunk_idx -= 1

        row_idx = anchor_idx + 1
        chunk_idx = best_idx + 1
        while row_idx < len(sentences) and chunk_idx < len(chunks):
            row = sentences[row_idx]
            if row.get("speaker_ct"):
                break
            row_norm = normalize_for_match(str(row.get("text", "")))
            if not row_norm:
                break
            effective_min = (
                short_sent_min_similarity
                if len(row_norm.split()) < short_sent_token_threshold
                else expand_min_similarity
            )
            sim = text_similarity(row_norm, chunk_norms[chunk_idx])
            if sim < effective_min:
                break
            _set_anchor_chunk_fill(
                row=row,
                anchor=anchor,
                anchor_idx=anchor_idx,
                chunk_text=chunks[chunk_idx][0],
                similarity=sim,
            )
            n_filled += 1
            row_idx += 1
            chunk_idx += 1

    return n_filled


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _join_chunk_slice(chunks: list[tuple[str, str]], start: int, end: int) -> str:
    return " ".join(raw for raw, _ in chunks[start:end]).strip()


def fill_unmatched_scene_directional_residual(
    sentences: list[dict],
    fill_min_similarity: float = 0.72,
    context_min_similarity: float = 0.65,
    ambiguity_margin: float = 0.05,
    max_rounds: int = 8,
) -> int:
    """Fill empty rows using scene-local directional residual matching."""
    n_filled = 0
    if len(sentences) < 3:
        return 0

    for _ in range(max_rounds):
        changed = False
        for i in range(1, len(sentences) - 1):
            cur = sentences[i]
            if cur.get("speaker_ct"):
                continue

            prev = sentences[i - 1]
            nxt = sentences[i + 1]
            scene_id = cur.get("scene_id")
            if scene_id is None:
                continue
            if prev.get("scene_id") != scene_id or nxt.get("scene_id") != scene_id:
                continue
            if not prev.get("speaker_ct") or not nxt.get("speaker_ct"):
                continue

            d_prev = abs(_safe_float(cur.get("start")) - _safe_float(prev.get("end")))
            d_next = abs(_safe_float(nxt.get("start")) - _safe_float(cur.get("end")))
            donor_idx = i - 1 if d_prev <= d_next else i + 1
            donor = sentences[donor_idx]
            donor_text = str(donor.get("utterance_ct", "")).strip()
            donor_anchor_norm = normalize_for_match(str(donor.get("text", "")))
            cur_norm = normalize_for_match(str(cur.get("text", "")))
            if not donor_text or not donor_anchor_norm or not cur_norm:
                continue

            chunks = _split_utterance_chunks(donor_text)
            if len(chunks) < 3:
                continue
            chunk_norms = [norm for _, norm in chunks]

            donor_chunk_idx, donor_best, donor_second = _best_chunk_index(
                donor_anchor_norm, chunk_norms
            )
            if donor_best < fill_min_similarity:
                continue
            if donor_second >= 0 and (donor_best - donor_second) < ambiguity_margin:
                continue

            if donor_idx == i + 1:
                candidate_idx = donor_chunk_idx - 1
                context_row = prev
                context_chunk_idx = donor_chunk_idx - 2
            else:
                candidate_idx = donor_chunk_idx + 1
                context_row = nxt
                context_chunk_idx = donor_chunk_idx + 2

            if candidate_idx < 0 or candidate_idx >= len(chunks):
                continue

            candidate_raw, candidate_norm = chunks[candidate_idx]
            candidate_sim = text_similarity(cur_norm, candidate_norm)
            if candidate_sim < fill_min_similarity:
                continue

            cur_best_idx, cur_best, cur_second = _best_chunk_index(cur_norm, chunk_norms)
            if cur_best_idx != candidate_idx:
                continue
            if cur_second >= 0 and (cur_best - cur_second) < ambiguity_margin:
                continue

            if context_chunk_idx < 0 or context_chunk_idx >= len(chunks):
                continue
            context_norm = normalize_for_match(str(context_row.get("text", "")))
            if not context_norm:
                continue
            context_sim = text_similarity(context_norm, chunk_norms[context_chunk_idx])
            if context_sim < context_min_similarity:
                continue

            cur["speaker_ct"] = donor.get("speaker_ct")
            cur["speaker_mapped"] = donor.get("speaker_ct")
            cur["utterance_ct"] = candidate_raw
            cur["comm_dialogue_idx"] = donor.get("comm_dialogue_idx")
            cur["match_similarity"] = round(candidate_sim, 4)
            cur["filled_from_neighbor_idx"] = donor_idx
            cur["fill_method"] = "scene_directional_residual"

            if donor_idx == i + 1:
                donor["utterance_ct"] = _join_chunk_slice(chunks, donor_chunk_idx, len(chunks))
            else:
                donor["utterance_ct"] = _join_chunk_slice(chunks, 0, donor_chunk_idx + 1)

            n_filled += 1
            changed = True

        if not changed:
            break

    return n_filled


def fill_unmatched_from_neighbors(
    sentences: list[dict],
    min_similarity: float = 0.7,
    min_remove_similarity: float = 0.55,
    max_rounds: int = 8,
) -> int:
    """Iteratively fill empty speaker_ct using closest mapped neighbor rows."""
    n_filled = 0
    for _ in range(max_rounds):
        changed = False
        for i, cur in enumerate(sentences):
            if cur.get("speaker_ct"):
                continue
            cur_norm = normalize_for_match(str(cur.get("text", "")))
            if not cur_norm:
                continue

            candidates: list[tuple[float, int]] = []
            for j in (i - 1, i + 1):
                if j < 0 or j >= len(sentences):
                    continue
                neigh = sentences[j]
                if not neigh.get("speaker_ct") or not str(neigh.get("utterance_ct", "")).strip():
                    continue
                dist = abs(_sentence_center(cur) - _sentence_center(neigh))
                candidates.append((dist, j))
            candidates.sort(key=lambda x: x[0])

            for _, j in candidates:
                neigh = sentences[j]
                residual = _best_span_to_remove(
                    source_text=str(neigh.get("utterance_ct", "")),
                    remove_text=str(neigh.get("text", "")),
                    min_remove_similarity=min_remove_similarity,
                )
                if not residual:
                    continue
                sim = text_similarity(cur_norm, normalize_for_match(residual))
                if sim < min_similarity:
                    continue
                cur["speaker_ct"] = neigh.get("speaker_ct")
                cur["speaker_mapped"] = neigh.get("speaker_ct")
                cur["utterance_ct"] = residual
                cur["comm_dialogue_idx"] = neigh.get("comm_dialogue_idx")
                cur["match_similarity"] = round(sim, 4)
                cur["filled_from_neighbor_idx"] = j
                cur["fill_method"] = "neighbor_residual"
                n_filled += 1
                changed = True
                break
        if not changed:
            break
    return n_filled


def fill_unmatched_by_dialogue_index(sentences: list[dict]) -> int:
    """Fill empty rows whose comm_dialogue_idx unambiguously maps to one speaker."""
    idx_to_speakers: dict[int, set[str]] = {}
    for sent in sentences:
        speaker = sent.get("speaker_ct")
        idx = sent.get("comm_dialogue_idx")
        if speaker and idx is not None:
            idx_to_speakers.setdefault(idx, set()).add(speaker)

    n_filled = 0
    for sent in sentences:
        if sent.get("speaker_ct"):
            continue
        idx = sent.get("comm_dialogue_idx")
        if idx is None:
            continue
        speakers = idx_to_speakers.get(idx, set())
        if len(speakers) != 1:
            continue
        speaker = next(iter(speakers))
        sent["speaker_ct"] = speaker
        sent["speaker_mapped"] = speaker
        sent["fill_method"] = "comm_dialogue_propagation"
        n_filled += 1
    return n_filled


def fill_scene_ids(sentences: list[dict]) -> None:
    """Forward-fill scene_id from matched sentences to unmatched ones."""
    last_scene = 1
    for sent in sentences:
        if sent.get("scene_id") is not None:
            last_scene = sent["scene_id"]
        else:
            sent["scene_id"] = last_scene


# ---------------------------------------------------------------------------
# Word-to-sentence mapping
# ---------------------------------------------------------------------------

def map_words_to_sentence_indices(words: list[dict], sentences: list[dict]) -> list[int | None]:
    """Assign each top-level word to a sentence index by time overlap."""
    if not words or not sentences:
        return [None] * len(words)

    out: list[int | None] = [None] * len(words)
    sent_i = 0
    eps = 1e-3

    for i, w in enumerate(words):
        try:
            w_start = float(w.get("start"))
            w_end = float(w.get("end", w_start))
        except (TypeError, ValueError):
            continue

        while sent_i + 1 < len(sentences):
            s_end = float(sentences[sent_i].get("end", sentences[sent_i].get("start", 0.0)))
            if w_start > s_end + eps:
                sent_i += 1
            else:
                break

        candidates = [sent_i]
        if sent_i > 0:
            candidates.append(sent_i - 1)
        if sent_i + 1 < len(sentences):
            candidates.append(sent_i + 1)

        best_idx = None
        best_overlap = -1.0
        for idx in candidates:
            s = sentences[idx]
            s_start = float(s.get("start", 0.0))
            s_end = float(s.get("end", s_start))
            overlap = min(w_end, s_end) - max(w_start, s_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
            if s_start - eps <= w_start <= s_end + eps:
                best_idx = idx
                break
        out[i] = best_idx
    return out


# ---------------------------------------------------------------------------
# Episode path inference and discovery
# ---------------------------------------------------------------------------

def infer_episode_paths(
    episode: str,
    annotation_root: Path,
    wutter_override: Path | None = None,
    community_override: Path | None = None,
) -> tuple[Path, Path]:
    """Return (wutter_json_path, community_txt_path) for an episode."""
    season_dir = episode_to_season_dir(episode)
    speech_dir = annotation_root / "Speech2Text" / season_dir
    community_dir = annotation_root / "community_based" / season_dir

    wutter_path = wutter_override
    if wutter_path is None:
        candidates = [
            speech_dir / f"{episode}_model-AA_desc-wUtter_transcript.json",
            speech_dir / f"{episode}_desc-wUtter_transcript.json",
            speech_dir / f"{episode}_model-AA_desc-wSpeaker_transcript.json",
            speech_dir / f"{episode}_desc-wSpeaker_transcript.json",
        ]
        wutter_path = next((p for p in candidates if p.exists()), None)
        if wutter_path is None:
            glob_matches = sorted(speech_dir.glob(f"{episode}*_transcript.json"))
            if glob_matches:
                wutter_path = glob_matches[0]
    if wutter_path is None or not wutter_path.exists():
        raise FileNotFoundError(f"Transcript JSON not found for {episode} in {speech_dir}")

    community_path = community_override
    if community_path is None:
        community_root = episode_to_community_root(episode)
        community_path = community_dir / f"{community_root}_ufs.txt"
    if not community_path.exists():
        raise FileNotFoundError(f"Community transcript not found for {episode}: {community_path}")

    return wutter_path, community_path


def discover_episodes_in_season(season_dir: str, annotation_root: Path) -> list[str]:
    """Discover episode keys from Speech2Text transcript files in a season directory."""
    speech_dir = annotation_root / "Speech2Text" / season_dir
    if not speech_dir.exists():
        raise FileNotFoundError(f"Speech2Text season directory does not exist: {speech_dir}")

    episodes = {
        normalize_episode_key(episode_from_transcript_filename(path))
        for path in speech_dir.glob("friends_s*e*_*transcript.json")
    }
    if not episodes:
        raise FileNotFoundError(f"No transcripts found in {speech_dir}")
    return sorted(episodes)


# ---------------------------------------------------------------------------
# Core episode processing
# ---------------------------------------------------------------------------

def map_speakers_for_episode(
    wutter_json: Path,
    community_txt: Path,
    min_similarity: float,
    neighbor_min_similarity: float,
    anchor_min_similarity: float,
    anchor_expand_min_similarity: float,
    scene_iter_fill_similarity: float,
    scene_iter_context_similarity: float,
    scene_iter_ambiguity_margin: float,
    scene_iter_max_rounds: int,
    overwrite_speaker: bool,
    length_ratio_cap: float = 5.0,
    anchor_short_sent_tokens: int = 4,
    anchor_short_sent_similarity: float = 0.40,
) -> tuple[dict, dict, dict[int, str]]:
    """Align ASR transcript to community transcript and map speakers.

    Returns:
        mapped: enriched transcript dict (sentences + words with speaker_ct, scene_id, etc.)
        stats: alignment statistics dict
        scene_descs: scene_id → scene description mapping from community transcript
    """
    raw = json.loads(wutter_json.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict JSON at {wutter_json}")

    sentences = raw.get("sentences")
    words = raw.get("words")
    if not isinstance(sentences, list) or not isinstance(words, list):
        raise ValueError(f"{wutter_json} must contain list fields 'sentences' and 'words'")

    community_dialogues, scene_descs = parse_community_transcript(community_txt)
    mapping, sim_cache = align_monotonic(sentences, community_dialogues)
    community_by_idx = {int(d["dialogue_idx"]): d for d in community_dialogues}

    result = deepcopy(raw)
    result_sentences = result["sentences"]
    matched_sentences = 0
    for i, sent in enumerate(result_sentences):
        comm_idx = mapping.get(i)
        speaker_ct = None
        utterance_ct = None
        sim = 0.0
        comm = None
        if comm_idx is not None:
            sim = sim_cache.get((i, comm_idx), 0.0)
            comm = community_by_idx.get(comm_idx)
            if comm is not None and sim >= min_similarity:
                local_tokens = len(normalize_for_match(str(sent.get("text", ""))).split())
                comm_tokens = len(normalize_for_match(str(comm.get("utterance_ct", ""))).split())
                length_ok = comm_tokens == 0 or (local_tokens / comm_tokens) <= length_ratio_cap
                if length_ok:
                    speaker_ct = comm["speaker_ct"]
                    utterance_ct = comm["utterance_ct"]
                    matched_sentences += 1

        scene_id = comm["scene_idx"] if (comm is not None and sim >= min_similarity) else None
        sent["scene_id"] = scene_id
        sent["speaker_ct"] = speaker_ct
        sent["utterance_ct"] = utterance_ct
        sent["match_similarity"] = round(sim, 4)
        sent["comm_dialogue_idx"] = comm_idx
        sent["speaker_mapped"] = speaker_ct
        if overwrite_speaker and speaker_ct:
            sent["speaker"] = speaker_ct

    anchor_chunk_filled_sentences = fill_unmatched_from_anchor_chunks(
        result_sentences,
        anchor_min_similarity=anchor_min_similarity,
        expand_min_similarity=anchor_expand_min_similarity,
        short_sent_token_threshold=anchor_short_sent_tokens,
        short_sent_min_similarity=anchor_short_sent_similarity,
    )
    scene_directional_filled_sentences = fill_unmatched_scene_directional_residual(
        result_sentences,
        fill_min_similarity=scene_iter_fill_similarity,
        context_min_similarity=scene_iter_context_similarity,
        ambiguity_margin=scene_iter_ambiguity_margin,
        max_rounds=scene_iter_max_rounds,
    )
    fallback_neighbor_filled_sentences = fill_unmatched_from_neighbors(
        result_sentences,
        min_similarity=neighbor_min_similarity,
    )
    comm_dialogue_propagation_filled_sentences = fill_unmatched_by_dialogue_index(
        result_sentences,
    )
    fill_scene_ids(result_sentences)

    word_to_sent = map_words_to_sentence_indices(result["words"], result_sentences)
    matched_words = 0
    for i, w in enumerate(result["words"]):
        sent_idx = word_to_sent[i]
        speaker_ct = None
        if sent_idx is not None and 0 <= sent_idx < len(result_sentences):
            speaker_ct = result_sentences[sent_idx].get("speaker_ct")
        w["sentence_idx"] = sent_idx
        w["speaker_ct"] = speaker_ct
        w["speaker_mapped"] = speaker_ct
        if overwrite_speaker and speaker_ct:
            w["speaker"] = speaker_ct
        if speaker_ct:
            matched_words += 1

    total_filled_sentences = (
        matched_sentences
        + anchor_chunk_filled_sentences
        + scene_directional_filled_sentences
        + fallback_neighbor_filled_sentences
        + comm_dialogue_propagation_filled_sentences
    )

    stats = {
        "method": "monotonic_fuzzy_sentence_to_community",
        "min_similarity": min_similarity,
        "length_ratio_cap": length_ratio_cap,
        "anchor_min_similarity": anchor_min_similarity,
        "anchor_expand_min_similarity": anchor_expand_min_similarity,
        "anchor_short_sent_tokens": anchor_short_sent_tokens,
        "anchor_short_sent_similarity": anchor_short_sent_similarity,
        "scene_iter_fill_similarity": scene_iter_fill_similarity,
        "scene_iter_context_similarity": scene_iter_context_similarity,
        "scene_iter_ambiguity_margin": scene_iter_ambiguity_margin,
        "scene_iter_max_rounds": scene_iter_max_rounds,
        "neighbor_min_similarity": neighbor_min_similarity,
        "n_sentences": len(result_sentences),
        "n_words": len(result["words"]),
        "n_community_dialogues": len(community_dialogues),
        "matched_sentences": matched_sentences,
        "anchor_chunk_filled_sentences": anchor_chunk_filled_sentences,
        "scene_directional_filled_sentences": scene_directional_filled_sentences,
        "fallback_neighbor_filled_sentences": fallback_neighbor_filled_sentences,
        "comm_dialogue_propagation_filled_sentences": comm_dialogue_propagation_filled_sentences,
        "total_filled_sentences": total_filled_sentences,
        "unfilled_sentences": len(result_sentences) - total_filled_sentences,
        "matched_words": matched_words,
    }
    result["speaker_mapping"] = stats
    return result, stats, scene_descs


# ---------------------------------------------------------------------------
# TSV output
# ---------------------------------------------------------------------------

def write_sentence_table_tsv(mapped: dict, out_path: Path) -> None:
    """Write per-sentence alignment TSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scene_id", "sentence_id", "start", "end", "utterance", "speaker", "utterance_ct", "speaker_ct"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for sentence_id, sent in enumerate(mapped.get("sentences", []), start=1):
            speaker_ct = sent.get("speaker_ct", "") or ""
            speaker = speaker_ct or (sent.get("speaker", "") or "")
            writer.writerow(
                {
                    "scene_id": sent.get("scene_id", ""),
                    "sentence_id": sentence_id,
                    "start": sent.get("start", ""),
                    "end": sent.get("end", ""),
                    "utterance": sent.get("text", ""),
                    "speaker": speaker,
                    "utterance_ct": sent.get("utterance_ct", "") or "",
                    "speaker_ct": speaker_ct,
                }
            )


def write_scene_summary_tsv(
    mapped: dict,
    scene_descs: dict[int, str],
    shots: list | None,
    out_path: Path,
) -> None:
    """Write per-scene summary TSV with scene_id, scene_desc, start, end, shot_ids."""
    from charnet.community_align import find_shot_id_at_time

    # Aggregate per-scene: start, end, shot_ids
    scene_starts: dict[int, float] = {}
    scene_ends: dict[int, float] = {}
    scene_shot_ids: dict[int, set[int]] = {}

    for sent in mapped.get("sentences", []):
        scene_id = sent.get("scene_id")
        if scene_id is None:
            continue
        try:
            start = float(sent.get("start", 0.0))
            end = float(sent.get("end", start))
        except (TypeError, ValueError):
            continue

        if scene_id not in scene_starts or start < scene_starts[scene_id]:
            scene_starts[scene_id] = start
        if scene_id not in scene_ends or end > scene_ends[scene_id]:
            scene_ends[scene_id] = end

        if shots:
            shot_id = find_shot_id_at_time(shots, start)
            if shot_id is not None:
                scene_shot_ids.setdefault(scene_id, set()).add(shot_id)
            shot_id_end = find_shot_id_at_time(shots, end)
            if shot_id_end is not None:
                scene_shot_ids.setdefault(scene_id, set()).add(shot_id_end)

    # Only include scenes that have at least one sentence with timestamps
    # (the community transcript may cover the full episode while wutter covers a half)
    all_scene_ids = sorted(scene_starts.keys())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scene_id", "scene_desc", "start", "end", "shot_ids"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for sid in all_scene_ids:
            shot_ids_str = ",".join(str(s) for s in sorted(scene_shot_ids.get(sid, set())))
            writer.writerow(
                {
                    "scene_id": sid,
                    "scene_desc": scene_descs.get(sid, ""),
                    "start": f"{scene_starts[sid]:.2f}" if sid in scene_starts else "",
                    "end": f"{scene_ends[sid]:.2f}" if sid in scene_ends else "",
                    "shot_ids": shot_ids_str,
                }
            )


def process_episode(
    episode: str,
    annotation_root: Path,
    output_dir: Path,
    min_similarity: float,
    neighbor_min_similarity: float,
    anchor_min_similarity: float,
    anchor_expand_min_similarity: float,
    scene_iter_fill_similarity: float,
    scene_iter_context_similarity: float,
    scene_iter_ambiguity_margin: float,
    scene_iter_max_rounds: int,
    overwrite_speaker: bool,
    length_ratio_cap: float = 5.0,
    anchor_short_sent_tokens: int = 4,
    anchor_short_sent_similarity: float = 0.40,
    wutter_override: Path | None = None,
    community_override: Path | None = None,
    shots: list | None = None,
    scene_summary_only: bool = False,
) -> tuple[Path, dict]:
    """Process one episode: align, fill speakers, write TSVs.

    If scene_summary_only is True, only the scene summary TSV is written.

    Returns (output_tsv_path, stats_dict).
    """
    wutter_json, community_txt = infer_episode_paths(
        episode=episode,
        annotation_root=annotation_root,
        wutter_override=wutter_override,
        community_override=community_override,
    )
    mapped, stats, scene_descs = map_speakers_for_episode(
        wutter_json=wutter_json,
        community_txt=community_txt,
        min_similarity=min_similarity,
        neighbor_min_similarity=neighbor_min_similarity,
        anchor_min_similarity=anchor_min_similarity,
        anchor_expand_min_similarity=anchor_expand_min_similarity,
        scene_iter_fill_similarity=scene_iter_fill_similarity,
        scene_iter_context_similarity=scene_iter_context_similarity,
        scene_iter_ambiguity_margin=scene_iter_ambiguity_margin,
        scene_iter_max_rounds=scene_iter_max_rounds,
        overwrite_speaker=overwrite_speaker,
        length_ratio_cap=length_ratio_cap,
        anchor_short_sent_tokens=anchor_short_sent_tokens,
        anchor_short_sent_similarity=anchor_short_sent_similarity,
    )

    season_dir = episode_to_season_dir(episode)
    out_season_dir = output_dir / season_dir
    out_season_dir.mkdir(parents=True, exist_ok=True)

    if not scene_summary_only:
        out_tsv = out_season_dir / f"{episode}_sentence_speaker_table.tsv"
        write_sentence_table_tsv(mapped, out_tsv)

    # Scene summary TSV
    scene_summary_tsv = out_season_dir / f"{episode}_scene_summary.tsv"
    write_scene_summary_tsv(mapped, scene_descs, shots, scene_summary_tsv)

    primary_out = scene_summary_tsv if scene_summary_only else out_season_dir / f"{episode}_sentence_speaker_table.tsv"
    return primary_out, stats
