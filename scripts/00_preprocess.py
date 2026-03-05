#!/usr/bin/env python3
"""Stage 0: preprocess transcript + shots + community transcript artifacts."""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

import click

# Allow running as script without installing package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from charnet.community_align import parse_community_transcript
from charnet.io import (
    estimate_missing_end_times,
    load_sentence_transcript,
    load_shots,
    load_speaker_map,
    load_transcript,
    load_word_transcript,
    save_records,
    save_shots,
    save_utterances,
)

logger = logging.getLogger(__name__)
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")
REPO_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "friends_annotations" / "annotation_results"


def normalize_episode_key(value: str) -> str:
    key = value.strip()
    if re.fullmatch(r"s\d{2}e\d{2}[a-z]*", key, flags=re.IGNORECASE):
        key = f"friends_{key}"
    return key.lower()


def normalize_season_id(value: str) -> str:
    token = value.strip().lower()
    match = re.fullmatch(r"s?0*(\d+)", token)
    if not match:
        raise click.ClickException(f"Invalid season id: {value}. Use values like 's6' or '6'.")
    return f"s{int(match.group(1))}"


def episode_from_transcript_filename(path: Path) -> str:
    episode = path.stem
    for suffix in [
        "_model-AA_desc-wSpeaker_transcript",
        "_desc-wSpeaker_transcript",
        "_model-AA_desc-wUtter_transcript",
        "_desc-wUtter_transcript",
        "_transcript",
    ]:
        if episode.endswith(suffix):
            return episode[: -len(suffix)]
    return episode


def episode_to_season_dir(episode: str) -> str:
    match = re.search(r"s0*(\d+)e\d{2}", episode, flags=re.IGNORECASE)
    if not match:
        raise click.ClickException(f"Could not infer season from episode key: {episode}")
    return f"s{int(match.group(1))}"


def episode_to_root_for_community(episode: str) -> str:
    match = re.match(r"^(friends_s\d{2}e\d{2})([a-z]+)?$", episode, flags=re.IGNORECASE)
    if not match:
        return episode
    return match.group(1).lower()


def infer_episode_paths(
    episode: str,
    data_root: Path,
    transcript_override: Path | None,
    shots_override: Path | None,
    community_override: Path | None,
) -> tuple[Path, Path | None, Path | None]:
    season_dir = episode_to_season_dir(episode)
    speech_dir = data_root / "Speech2Text" / season_dir
    shots_dir = data_root / "TSVpyscene" / season_dir
    community_dir = data_root / "community_based" / season_dir

    transcript_path = transcript_override
    if transcript_path is None:
        candidates = [
            speech_dir / f"{episode}_model-AA_desc-wSpeaker_transcript.json",
            speech_dir / f"{episode}_desc-wSpeaker_transcript.json",
            speech_dir / f"{episode}_transcript.json",
        ]
        transcript_path = next((p for p in candidates if p.exists()), None)
        if transcript_path is None:
            glob_matches = sorted(speech_dir.glob(f"{episode}*_wSpeaker_transcript.json"))
            if glob_matches:
                transcript_path = glob_matches[0]
    if transcript_path is None or not transcript_path.exists():
        raise click.ClickException(f"Transcript not found for {episode} under {speech_dir}")

    shots_path = shots_override
    if shots_path is None:
        candidate = shots_dir / f"{episode}_pyscene.tsv"
        shots_path = candidate if candidate.exists() else None

    community_path = community_override
    if community_path is None:
        root_episode = episode_to_root_for_community(episode)
        candidate = community_dir / f"{root_episode}_ufs.txt"
        community_path = candidate if candidate.exists() else None

    return transcript_path, shots_path, community_path


def discover_season_episodes(season_dir: str, data_root: Path) -> list[str]:
    transcript_dir = data_root / "Speech2Text" / season_dir
    if not transcript_dir.exists():
        raise click.ClickException(f"Transcript directory not found: {transcript_dir}")

    episodes = {
        normalize_episode_key(episode_from_transcript_filename(path))
        for path in transcript_dir.glob("friends_s*e*_model-AA_desc-wSpeaker_transcript.json")
    }
    if not episodes:
        raise click.ClickException(f"No transcripts found in {transcript_dir}")
    return sorted(episodes)


def process_one_episode(
    episode: str,
    transcript_path: Path,
    shots_path: Path | None,
    community_path: Path | None,
    output_dir: Path,
    spk_map: dict[str, str] | None,
    min_utterance_duration: float,
    estimate_end_times: bool,
    word_gap_threshold: float,
    inspect_only: bool,
) -> tuple[int, int]:
    logger.info("Episode: %s", episode)
    logger.info("Loading transcript: %s", transcript_path)

    word_utterances = load_transcript(transcript_path, speaker_map=spk_map)
    grouped_utterances = load_word_transcript(
        transcript_path,
        word_gap_threshold=word_gap_threshold,
        speaker_map=spk_map,
    )
    sentence_utterances = load_sentence_transcript(transcript_path, speaker_map=spk_map)
    logger.info(
        "Loaded transcript as %d word-level + %d grouped + %d sentence utterances",
        len(word_utterances),
        len(grouped_utterances),
        len(sentence_utterances),
    )

    if estimate_end_times:
        word_utterances = estimate_missing_end_times(word_utterances, min_duration=min_utterance_duration)
        grouped_utterances = estimate_missing_end_times(
            grouped_utterances, min_duration=min_utterance_duration
        )
        sentence_utterances = estimate_missing_end_times(
            sentence_utterances, min_duration=min_utterance_duration
        )

    if min_utterance_duration > 0:
        word_utterances = [u for u in word_utterances if (u.end - u.start) >= min_utterance_duration]
        grouped_utterances = [u for u in grouped_utterances if (u.end - u.start) >= min_utterance_duration]
        sentence_utterances = [
            u for u in sentence_utterances if (u.end - u.start) >= min_utterance_duration
        ]

    shot_list = []
    if shots_path and shots_path.exists():
        logger.info("Loading shots: %s", shots_path)
        shot_list = load_shots(shots_path)
        logger.info("Loaded %d shots", len(shot_list))
    elif shots_path:
        logger.warning("Shots path does not exist: %s", shots_path)
    else:
        logger.warning("No shots file inferred for %s", episode)

    community_events: list[dict] = []
    community_dialogues: list[dict] = []
    if community_path and community_path.exists():
        logger.info("Loading community transcript: %s", community_path)
        community_events, community_dialogues = parse_community_transcript(community_path)
        logger.info(
            "Parsed community transcript into %d events (%d dialogue rows)",
            len(community_events),
            len(community_dialogues),
        )
    else:
        logger.warning("No community transcript found for %s", episode)

    speakers = sorted({u.speaker.strip() for u in grouped_utterances if u.speaker and u.speaker.strip()})
    logger.info("Speakers (%d): %s", len(speakers), ", ".join(speakers))

    if inspect_only:
        click.echo(f"Episode:                  {episode}")
        click.echo(f"Transcript:               {transcript_path}")
        click.echo(f"Shots:                    {shots_path or 'N/A'}")
        click.echo(f"Community:                {community_path or 'N/A'}")
        click.echo(f"Utterances (word-level):  {len(word_utterances)}")
        click.echo(f"Utterances (grouped):     {len(grouped_utterances)}")
        click.echo(f"Utterances (sentence):    {len(sentence_utterances)}")
        click.echo(f"Community dialogue rows:  {len(community_dialogues)}")
        return len(sentence_utterances), len(community_dialogues)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_utterances(word_utterances, output_dir / "words.json")
    save_utterances(grouped_utterances, output_dir / "utterances.json")
    save_utterances(sentence_utterances, output_dir / "sentences.json")
    if shot_list:
        save_shots(shot_list, output_dir / "shots.json")
    if community_events:
        save_records(community_events, output_dir / "community_events.json", label="community events")
        save_records(community_dialogues, output_dir / "community_dialogues.json", label="community dialogues")

    logger.info("Preprocessing complete. Output: %s", output_dir)
    return len(sentence_utterances), len(community_dialogues)


@click.command()
@click.option("--episode", "-e", default=None,
              help="Single episode key (e.g. friends_s06e01a or s06e01a).")
@click.option("--season", default=None,
              help="Season id (e.g. s6 or 6). Processes all episode parts in that season.")
@click.option("--data-root", type=click.Path(exists=True), default=str(DEFAULT_DATA_ROOT),
              show_default=True,
              help="Root folder containing Speech2Text, TSVpyscene, and community_based.")
@click.option("--transcript", "-t", type=click.Path(exists=True), default=None,
              help="Transcript override for single-episode mode.")
@click.option("--shots", "-s", type=click.Path(exists=True), default=None,
              help="Shots override for single-episode mode.")
@click.option("--speaker-map", "-m", type=click.Path(exists=True), default=None,
              help="Path to speaker map JSON file (optional).")
@click.option("--community-transcript", "-u", type=click.Path(exists=True), default=None,
              help="Community transcript override for single-episode mode.")
@click.option("--output-dir", "-o", default=None,
              help="Output dir for episode mode, or base output dir for season mode.")
@click.option("--min-utterance-duration", default=0.0, show_default=True,
              help="Discard utterances shorter than this many seconds.")
@click.option("--estimate-end-times/--no-estimate-end-times", default=True, show_default=True,
              help="Estimate missing end times from next utterance start.")
@click.option("--word-gap-threshold", default=0.5, show_default=True, type=float,
              help="Max gap (seconds) between same-speaker words before starting a new grouped utterance.")
@click.option("--inspect-only", is_flag=True, default=False,
              help="Only inspect inputs and print summary; do not write output.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Enable debug logging.")
def main(
    episode,
    season,
    data_root,
    transcript,
    shots,
    speaker_map,
    community_transcript,
    output_dir,
    min_utterance_duration,
    estimate_end_times,
    word_gap_threshold,
    inspect_only,
    verbose,
):
    """Preprocess one episode or a whole season using structured path conventions."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    if not episode and not season and not transcript:
        raise click.ClickException("Provide one of --episode, --season, or --transcript.")
    if episode and season:
        raise click.ClickException("Use either --episode or --season, not both.")
    if season and (transcript or shots or community_transcript):
        raise click.ClickException(
            "--transcript/--shots/--community-transcript overrides are only supported in --episode mode."
        )

    data_root_path = Path(data_root)

    spk_map = None
    if speaker_map:
        spk_map = load_speaker_map(Path(speaker_map))
        logger.info("Loaded speaker map with %d entries", len(spk_map))

    output_base = Path(output_dir) if output_dir else Path(SCRATCH_DIR) / "output" / "00_preprocess"

    if season:
        season_dir = normalize_season_id(season)
        episodes = discover_season_episodes(season_dir, data_root_path)
    else:
        if episode:
            episode_key = normalize_episode_key(episode)
        else:
            episode_key = normalize_episode_key(episode_from_transcript_filename(Path(transcript)))
        episodes = [episode_key]

    n_ok = 0
    for ep in episodes:
        transcript_override = Path(transcript) if transcript and len(episodes) == 1 else None
        shots_override = Path(shots) if shots and len(episodes) == 1 else None
        community_override = Path(community_transcript) if community_transcript and len(episodes) == 1 else None

        transcript_path, shots_path, community_path = infer_episode_paths(
            episode=ep,
            data_root=data_root_path,
            transcript_override=transcript_override,
            shots_override=shots_override,
            community_override=community_override,
        )

        episode_out = output_base if (output_dir and len(episodes) == 1) else output_base / ep
        process_one_episode(
            episode=ep,
            transcript_path=transcript_path,
            shots_path=shots_path,
            community_path=community_path,
            output_dir=episode_out,
            spk_map=spk_map,
            min_utterance_duration=min_utterance_duration,
            estimate_end_times=estimate_end_times,
            word_gap_threshold=word_gap_threshold,
            inspect_only=inspect_only,
        )
        if not inspect_only:
            click.echo(f"Output written to: {episode_out}")
        n_ok += 1

    click.echo(f"Processed {n_ok} episode(s).")


if __name__ == "__main__":
    main()
