"""Tests for charnet.io — loading, parsing, field auto-detection."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from charnet.io import (
    load_transcript, load_shots, load_speaker_map,
    load_word_transcript, load_pyscene_tsv,
    estimate_missing_end_times, save_utterances, load_utterances,
    save_shots, load_shots_json, load_sentence_transcript,
    infer_community_transcript_path, save_records, load_records,
)
from charnet.models import Utterance, Shot


class TestLoadTranscript:
    def test_standard_fields(self, transcript_json_file):
        utts = load_transcript(transcript_json_file)
        assert len(utts) == 5
        assert utts[0].speaker == "Monica"
        assert utts[0].start == 0.0
        assert utts[0].end == 3.5

    def test_variant_fields(self, transcript_json_variant_fields):
        utts = load_transcript(transcript_json_variant_fields)
        assert len(utts) == 5
        assert utts[0].speaker == "Monica"
        assert utts[0].start == 0.0

    def test_speaker_map_applied(self, tmp_path):
        data = [{"speaker_label": "SPEAKER_01", "start_time": 0.0, "end_time": 1.0, "content": "hi"}]
        path = tmp_path / "t.json"
        path.write_text(json.dumps(data))
        spk_map = {"SPEAKER_01": "Monica"}
        utts = load_transcript(path, speaker_map=spk_map)
        assert utts[0].speaker == "Monica"

    def test_sorted_by_start(self, tmp_path):
        data = [
            {"speaker": "B", "start": 5.0, "end": 6.0, "text": "later"},
            {"speaker": "A", "start": 1.0, "end": 2.0, "text": "first"},
        ]
        path = tmp_path / "t.json"
        path.write_text(json.dumps(data))
        utts = load_transcript(path)
        assert utts[0].start < utts[1].start

    def test_missing_end_handled(self, tmp_path):
        data = [
            {"speaker": "A", "start": 0.0, "text": "no end"},
            {"speaker": "B", "start": 2.0, "end": 3.0, "text": "has end"},
        ]
        path = tmp_path / "t.json"
        path.write_text(json.dumps(data))
        utts = load_transcript(path)
        assert len(utts) == 2  # should not crash


class TestEstimateMissingEndTimes:
    def test_estimates_from_next(self):
        utts = [
            Utterance("A", 0.0, 0.0, "first", 0),
            Utterance("B", 2.0, 3.0, "second", 1),
        ]
        result = estimate_missing_end_times(utts)
        assert result[0].end == 2.0

    def test_last_utterance_gets_min_duration(self):
        utts = [Utterance("A", 5.0, 5.0, "last", 0)]
        result = estimate_missing_end_times(utts, min_duration=0.5)
        assert result[0].end == 5.5


class TestLoadShots:
    def test_loads_shots(self, shots_csv_file):
        shots = load_shots(shots_csv_file)
        assert len(shots) == 3
        assert shots[0].shot_id == 1
        assert shots[0].start == 0.0
        assert shots[0].end == 8.0

    def test_sorted_by_start(self, shots_csv_file):
        shots = load_shots(shots_csv_file)
        starts = [s.start for s in shots]
        assert starts == sorted(starts)


class TestLoadWordTranscript:
    def test_basic_grouping(self, tmp_path):
        """Words from the same speaker with small gaps are grouped into one utterance."""
        data = {
            "words": [
                {"word": "Hey,", "start": 0.0, "end": 0.3, "speaker": "monica"},
                {"word": "Ross!", "start": 0.35, "end": 0.7, "speaker": "monica"},
            ]
        }
        path = tmp_path / "w.json"
        path.write_text(json.dumps(data))
        utts = load_word_transcript(path)
        assert len(utts) == 1
        assert utts[0].speaker == "Monica"
        assert utts[0].start == 0.0
        assert utts[0].end == 0.7
        assert "Hey," in utts[0].text

    def test_speaker_change_splits(self, tmp_path):
        """Speaker change creates a new utterance."""
        data = {
            "words": [
                {"word": "Hi", "start": 0.0, "end": 0.3, "speaker": "monica"},
                {"word": "Hello", "start": 0.4, "end": 0.7, "speaker": "chandler"},
            ]
        }
        path = tmp_path / "w.json"
        path.write_text(json.dumps(data))
        utts = load_word_transcript(path)
        assert len(utts) == 2
        assert utts[0].speaker == "Monica"
        assert utts[1].speaker == "Chandler"

    def test_gap_threshold_splits(self, tmp_path):
        """Gap >= word_gap_threshold splits utterance even with same speaker."""
        data = {
            "words": [
                {"word": "Okay.", "start": 0.0, "end": 0.3, "speaker": "ross"},
                {"word": "Sure.", "start": 1.0, "end": 1.3, "speaker": "ross"},
            ]
        }
        path = tmp_path / "w.json"
        path.write_text(json.dumps(data))
        utts = load_word_transcript(path, word_gap_threshold=0.5)
        assert len(utts) == 2

    def test_gap_below_threshold_merges(self, tmp_path):
        """Gap < word_gap_threshold keeps same-speaker words together."""
        data = {
            "words": [
                {"word": "Okay.", "start": 0.0, "end": 0.3, "speaker": "ross"},
                {"word": "Sure.", "start": 0.6, "end": 0.9, "speaker": "ross"},
            ]
        }
        path = tmp_path / "w.json"
        path.write_text(json.dumps(data))
        utts = load_word_transcript(path, word_gap_threshold=0.5)
        assert len(utts) == 1

    def test_speaker_normalization_title_case(self, tmp_path):
        """Speaker names are title-cased."""
        data = {
            "words": [
                {"word": "Hi", "start": 0.0, "end": 0.2, "speaker": "joey tribbiani"},
            ]
        }
        path = tmp_path / "w.json"
        path.write_text(json.dumps(data))
        utts = load_word_transcript(path)
        assert utts[0].speaker == "Joey Tribbiani"

    def test_speaker_map_overrides_normalization(self, tmp_path):
        """speaker_map overrides the title-cased speaker name."""
        data = {
            "words": [
                {"word": "Hi", "start": 0.0, "end": 0.2, "speaker": "SPEAKER_01"},
            ]
        }
        path = tmp_path / "w.json"
        path.write_text(json.dumps(data))
        utts = load_word_transcript(path, speaker_map={"Speaker_01": "Monica"})
        # After title() "SPEAKER_01" → "Speaker_01"; map applies
        assert utts[0].speaker == "Monica"

    def test_empty_words_returns_empty(self, tmp_path):
        data = {"words": []}
        path = tmp_path / "w.json"
        path.write_text(json.dumps(data))
        utts = load_word_transcript(path)
        assert utts == []


class TestLoadPysceneTsv:
    def test_basic_load(self, tmp_path):
        """Basic TSV with onset/duration columns loads correctly."""
        tsv_content = "onset\tduration\tonset_frame\n0.0\t5.0\t0\n5.0\t3.5\t125\n8.5\t4.0\t212\n"
        path = tmp_path / "shots.tsv"
        path.write_text(tsv_content)
        shots = load_pyscene_tsv(path)
        assert len(shots) == 3
        assert shots[0].shot_id == 1
        assert shots[0].start == 0.0
        assert shots[0].end == 5.0
        assert shots[1].start == 5.0
        assert shots[1].end == 8.5
        assert shots[2].start == 8.5
        assert shots[2].end == 12.5

    def test_sorted_by_start(self, tmp_path):
        """Results are sorted by start time."""
        tsv_content = "onset\tduration\tonset_frame\n10.0\t2.0\t250\n0.0\t5.0\t0\n5.0\t5.0\t125\n"
        path = tmp_path / "shots.tsv"
        path.write_text(tsv_content)
        shots = load_pyscene_tsv(path)
        starts = [s.start for s in shots]
        assert starts == sorted(starts)

    def test_sequential_shot_ids(self, tmp_path):
        """Shot IDs are assigned sequentially (1-based) regardless of TSV order."""
        tsv_content = "onset\tduration\tonset_frame\n0.0\t2.0\t0\n2.0\t3.0\t50\n"
        path = tmp_path / "shots.tsv"
        path.write_text(tsv_content)
        shots = load_pyscene_tsv(path)
        assert [s.shot_id for s in shots] == [1, 2]


class TestRoundtrip:
    def test_utterances_roundtrip(self, tmp_path, sample_utterances):
        path = tmp_path / "utterances.json"
        save_utterances(sample_utterances, path)
        loaded = load_utterances(path)
        assert len(loaded) == len(sample_utterances)
        assert loaded[0].speaker == sample_utterances[0].speaker

    def test_shots_roundtrip(self, tmp_path, sample_shots):
        path = tmp_path / "shots.json"
        save_shots(sample_shots, path)
        loaded = load_shots_json(path)
        assert len(loaded) == len(sample_shots)
        assert loaded[0].start == sample_shots[0].start

    def test_records_roundtrip(self, tmp_path):
        path = tmp_path / "records.json"
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        save_records(rows, path)
        loaded = load_records(path)
        assert loaded == rows


class TestLoadSentenceTranscript:
    def test_loads_sentences_dict_shape(self, tmp_path):
        data = {
            "sentences": [
                {"speaker": "ross", "start": 1.0, "end": 2.0, "text": "Hi."},
                {"speaker": "rachel", "start": 2.5, "end": 3.0, "text": "Hello."},
            ]
        }
        path = tmp_path / "sentences.json"
        path.write_text(json.dumps(data))
        utts = load_sentence_transcript(path)
        assert len(utts) == 2
        assert utts[0].speaker == "ross"
        assert utts[0].text == "Hi."


class TestInferCommunityTranscriptPath:
    def test_infers_from_speechtotext_layout(self, tmp_path):
        transcript = tmp_path / "annotation_results" / "Speech2Text" / "s6" / "friends_s06e01a.json"
        transcript.parent.mkdir(parents=True)
        transcript.write_text("{}")

        community = (
            tmp_path
            / "annotation_results"
            / "community_based"
            / "s6"
            / "friends_s06e01_ufs.txt"
        )
        community.parent.mkdir(parents=True)
        community.write_text("Ross: hi")

        inferred = infer_community_transcript_path(transcript, "friends_s06e01a")
        assert inferred == community
