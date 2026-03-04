"""Scene segmentation: triangulate shots + transcript into meaningful scenes."""
from __future__ import annotations

import logging
from typing import Any, Optional

from charnet.models import Utterance, Shot, Scene

logger = logging.getLogger(__name__)


def get_speakers_in_window(utterances: list[Utterance], t_start: float, t_end: float) -> set[str]:
    """Return set of speakers with any overlap with [t_start, t_end]."""
    speakers = set()
    for u in utterances:
        if u.speaker and u.end > t_start and u.start < t_end:
            speakers.add(u.speaker)
    return speakers


def jaccard(a: set, b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0  # both empty = same (silence)
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def segment_with_shots(
    utterances: list[Utterance],
    shots: list[Shot],
    jaccard_threshold: float = 0.3,
    max_shot_gap: float = 2.0,
    min_scene_duration: float = 10.0,
    max_scene_duration: float = 300.0,
    silence_lookahead: int = 3,
) -> list[Scene]:
    """Triangulate shots and transcript utterances into scenes.

    Algorithm:
      For each consecutive pair of shots, compute the Jaccard similarity of
      their active speaker sets. Merge into the same scene if similarity is
      above threshold AND the gap is small; otherwise start a new scene.
    """
    if not shots:
        logger.warning("No shots provided — falling back to transcript-only segmentation")
        return segment_transcript_only(utterances)

    # Group shots into scenes
    scene_shot_groups: list[list[Shot]] = []
    current_group: list[Shot] = [shots[0]]

    for i in range(1, len(shots)):
        prev = shots[i - 1]
        curr = shots[i]
        gap = curr.start - prev.end

        speakers_prev = get_speakers_in_window(utterances, prev.start, prev.end)
        speakers_curr = get_speakers_in_window(utterances, curr.start, curr.end)

        # Silence lookahead: if prev shot is silent, look ahead for speaker context
        if not speakers_prev:
            for look in range(1, min(silence_lookahead + 1, len(shots) - i)):
                look_shot = shots[i - 1 + look]
                speakers_prev = get_speakers_in_window(utterances, look_shot.start, look_shot.end)
                if speakers_prev:
                    break

        sim = jaccard(speakers_prev, speakers_curr)
        merge = sim >= jaccard_threshold and gap < max_shot_gap

        logger.debug(
            "Shot %d→%d: gap=%.2fs jaccard=%.2f merge=%s",
            prev.shot_id, curr.shot_id, gap, sim, merge,
        )

        if merge:
            current_group.append(curr)
        else:
            scene_shot_groups.append(current_group)
            current_group = [curr]

    scene_shot_groups.append(current_group)

    # Build Scene objects from shot groups
    raw_scenes: list[Scene] = []
    for scene_id, group in enumerate(scene_shot_groups):
        s_start = group[0].start
        s_end = group[-1].end
        utts_in = [u for u in utterances if u.end > s_start and u.start < s_end]
        speakers = sorted(set(u.speaker for u in utts_in if u.speaker))
        raw_scenes.append(Scene(
            scene_id=scene_id,
            start=s_start,
            end=s_end,
            speakers=speakers,
            n_shots=len(group),
            n_utterances=len(utts_in),
            utterance_indices=[u.index for u in utts_in],
        ))

    # Post-processing
    scenes = _merge_short_scenes(raw_scenes, utterances, min_scene_duration)
    scenes = _split_long_scenes(scenes, utterances, max_scene_duration)

    # Re-number scene_ids
    for i, sc in enumerate(scenes):
        sc.scene_id = i

    logger.info("Segmented into %d scenes (from %d shots)", len(scenes), len(shots))
    return scenes


def segment_transcript_only(
    utterances: list[Utterance],
    silence_gap: float = 5.0,
    speaker_window: int = 10,
    speaker_jaccard_break: float = 0.2,
    min_scene_duration: float = 10.0,
    max_scene_duration: float = 300.0,
) -> list[Scene]:
    """Fallback segmentation using only transcript data.

    Uses silence gaps and sliding-window speaker set Jaccard drop to find
    scene boundaries.
    """
    if not utterances:
        return []

    break_points: list[int] = [0]

    for i in range(1, len(utterances)):
        gap = utterances[i].start - utterances[i - 1].end

        # Silence gap break
        if gap >= silence_gap:
            break_points.append(i)
            continue

        # Sliding-window speaker Jaccard break
        half = speaker_window // 2
        window_a = {u.speaker for u in utterances[max(0, i - half):i] if u.speaker}
        window_b = {u.speaker for u in utterances[i:min(len(utterances), i + half)] if u.speaker}
        if window_a and window_b and jaccard(window_a, window_b) < speaker_jaccard_break:
            break_points.append(i)

    break_points.append(len(utterances))

    raw_scenes: list[Scene] = []
    for scene_id, (bp_start, bp_end) in enumerate(zip(break_points[:-1], break_points[1:])):
        group = utterances[bp_start:bp_end]
        if not group:
            continue
        speakers = sorted(set(u.speaker for u in group if u.speaker))
        raw_scenes.append(Scene(
            scene_id=scene_id,
            start=group[0].start,
            end=group[-1].end,
            speakers=speakers,
            n_shots=0,
            n_utterances=len(group),
            utterance_indices=[u.index for u in group],
        ))

    scenes = _merge_short_scenes(raw_scenes, utterances, min_scene_duration)
    scenes = _split_long_scenes(scenes, utterances, max_scene_duration)

    for i, sc in enumerate(scenes):
        sc.scene_id = i

    logger.info("Transcript-only segmentation: %d scenes", len(scenes))
    return scenes


def _merge_short_scenes(
    scenes: list[Scene], utterances: list[Utterance], min_duration: float
) -> list[Scene]:
    """Merge scenes shorter than min_duration into their neighbor."""
    if not scenes:
        return scenes

    merged = [scenes[0]]
    for sc in scenes[1:]:
        prev = merged[-1]
        if sc.duration < min_duration or prev.duration < min_duration:
            # Absorb sc into prev
            prev.end = sc.end
            prev.n_shots += sc.n_shots
            prev.utterance_indices.extend(sc.utterance_indices)
            prev.n_utterances = len(prev.utterance_indices)
            utts = [utterances[i] for i in prev.utterance_indices if i < len(utterances)]
            prev.speakers = sorted(set(u.speaker for u in utts if u.speaker))
        else:
            merged.append(sc)

    return merged


def _split_long_scenes(
    scenes: list[Scene], utterances: list[Utterance], max_duration: float
) -> list[Scene]:
    """Split scenes longer than max_duration at the point of lowest speaker overlap."""
    result: list[Scene] = []
    for sc in scenes:
        if sc.duration <= max_duration:
            result.append(sc)
            continue

        # Find split point: midpoint within the scene
        sc_utts = [u for u in utterances if u.end > sc.start and u.start < sc.end]
        if len(sc_utts) < 4:
            result.append(sc)
            continue

        mid = len(sc_utts) // 2
        split_time = sc_utts[mid].start

        first_half = [u for u in sc_utts if u.start < split_time]
        second_half = [u for u in sc_utts if u.start >= split_time]

        if not first_half or not second_half:
            result.append(sc)
            continue

        sc1 = Scene(
            scene_id=sc.scene_id,
            start=sc.start,
            end=split_time,
            speakers=sorted(set(u.speaker for u in first_half if u.speaker)),
            n_shots=sc.n_shots // 2,
            n_utterances=len(first_half),
            utterance_indices=[u.index for u in first_half],
        )
        sc2 = Scene(
            scene_id=sc.scene_id + 1,
            start=split_time,
            end=sc.end,
            speakers=sorted(set(u.speaker for u in second_half if u.speaker)),
            n_shots=sc.n_shots - sc.n_shots // 2,
            n_utterances=len(second_half),
            utterance_indices=[u.index for u in second_half],
        )
        result.extend([sc1, sc2])
        logger.info("Split long scene (%.1fs) at %.2fs", sc.duration, split_time)

    return result
