"""Tests for charnet.scene_segment — shot-transcript triangulation."""
from __future__ import annotations

import pytest

from charnet.models import Utterance, Shot, Scene
from charnet.scene_segment import (
    segment_with_shots, segment_transcript_only, jaccard, get_speakers_in_window,
)


class TestJaccard:
    def test_identical_sets(self):
        assert jaccard({"A", "B"}, {"A", "B"}) == 1.0

    def test_disjoint_sets(self):
        assert jaccard({"A"}, {"B"}) == 0.0

    def test_partial_overlap(self):
        assert jaccard({"A", "B"}, {"B", "C"}) == pytest.approx(1 / 3)

    def test_both_empty(self):
        assert jaccard(set(), set()) == 1.0


class TestGetSpeakersInWindow:
    def test_returns_overlapping_speakers(self, sample_utterances):
        speakers = get_speakers_in_window(sample_utterances, 0.0, 5.0)
        assert "Monica" in speakers
        assert "Ross" in speakers

    def test_empty_window(self, sample_utterances):
        speakers = get_speakers_in_window(sample_utterances, 100.0, 200.0)
        assert speakers == set()


class TestSegmentWithShots:
    def test_produces_scenes(self, sample_utterances, sample_shots):
        scenes = segment_with_shots(sample_utterances, sample_shots,
                                    jaccard_threshold=0.3, max_shot_gap=5.0,
                                    min_scene_duration=0.0, max_scene_duration=1000.0)
        assert len(scenes) >= 1
        for sc in scenes:
            assert isinstance(sc, Scene)
            assert sc.end > sc.start

    def test_scene_ids_sequential(self, sample_utterances, sample_shots):
        scenes = segment_with_shots(sample_utterances, sample_shots,
                                    min_scene_duration=0.0, max_scene_duration=1000.0)
        ids = [sc.scene_id for sc in scenes]
        assert ids == list(range(len(scenes)))

    def test_no_shots_falls_back(self, sample_utterances):
        scenes = segment_with_shots(sample_utterances, shots=[],
                                    min_scene_duration=0.0, max_scene_duration=1000.0)
        assert len(scenes) >= 1

    def test_speakers_assigned(self, sample_utterances, sample_shots):
        scenes = segment_with_shots(sample_utterances, sample_shots,
                                    min_scene_duration=0.0, max_scene_duration=1000.0)
        all_speakers = set()
        for sc in scenes:
            all_speakers.update(sc.speakers)
        assert "Monica" in all_speakers
        assert "Ross" in all_speakers


class TestSegmentTranscriptOnly:
    def test_produces_scenes(self, sample_utterances):
        scenes = segment_transcript_only(sample_utterances,
                                         min_scene_duration=0.0, max_scene_duration=1000.0)
        assert len(scenes) >= 1

    def test_scene_covers_all_utterances(self, sample_utterances):
        scenes = segment_transcript_only(sample_utterances,
                                          min_scene_duration=0.0, max_scene_duration=1000.0)
        all_indices = set()
        for sc in scenes:
            all_indices.update(sc.utterance_indices)
        expected = {u.index for u in sample_utterances}
        assert all_indices == expected

    def test_empty_input(self):
        scenes = segment_transcript_only([])
        assert scenes == []

    def test_large_gap_splits(self):
        utts = [
            Utterance("A", 0.0, 1.0, "first", 0),
            Utterance("B", 100.0, 101.0, "second", 1),  # big gap
        ]
        scenes = segment_transcript_only(utts, silence_gap=5.0,
                                          min_scene_duration=0.0, max_scene_duration=1000.0)
        assert len(scenes) == 2
