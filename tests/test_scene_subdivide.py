import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from charnet.scene_subdivide import Scene, expand_episode_spec, subdivide_episode


def _touch_scene_files(root: Path, episodes: list[str]) -> None:
    for ep in episodes:
        season = int(ep[1:3])
        d = root / f"s{season}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"friends_{ep}_scene_summary.tsv").write_text("scene_id\tstart\tend\n")


def _write_scene_table(root: Path, episode: str, rows: list[dict]) -> None:
    season = int(episode[1:3])
    d = root / f"s{season}"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        d / f"friends_{episode}_scene_summary.tsv", sep="\t", index=False
    )


def test_expand_episode_spec_all(tmp_path):
    _touch_scene_files(tmp_path, ["s01e01a", "s02e03b", "s03e01a"])
    assert expand_episode_spec("ALL", tmp_path) == ["s01e01a", "s02e03b", "s03e01a"]


def test_expand_episode_spec_single_season(tmp_path):
    _touch_scene_files(tmp_path, ["s01e01a", "s02e03b", "s03e01a"])
    assert expand_episode_spec("s2", tmp_path) == ["s02e03b"]


def test_expand_episode_spec_season_range(tmp_path):
    _touch_scene_files(tmp_path, ["s01e01a", "s02e03b", "s03e01a", "s06e02a"])
    assert expand_episode_spec("s3-s6", tmp_path) == ["s03e01a", "s06e02a"]


def test_expand_episode_spec_explicit_list(tmp_path):
    _touch_scene_files(tmp_path, ["s01e01a", "s02e03b"])
    assert expand_episode_spec("s01e01a,s02e03b", tmp_path) == ["s01e01a", "s02e03b"]


def test_expand_episode_spec_rejects_malformed_id(tmp_path):
    with pytest.raises(ValueError):
        expand_episode_spec("s01e01a,garbage", tmp_path)


def test_subdivide_splits_and_renumbers(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _write_scene_table(in_dir, "s01e01a", [
        {"scene_id": 1, "scene_desc": "Central Perk", "start": 0.0, "end": 100.0, "shot_ids": "1|2"},
        {"scene_id": 2, "scene_desc": "Monica's", "start": 100.0, "end": 150.0, "shot_ids": "3"},
    ])

    # propose one interior boundary at t=50 inside the first scene only
    def propose(scene: Scene) -> list[float]:
        return [50.0] if scene.scene_id == 1 else []

    stats = subdivide_episode("s01e01a", in_dir, out_dir, propose, aug_tag="topic_aug")

    out = pd.read_csv(out_dir / "s1" / "friends_s01e01a_scene_summary.tsv", sep="\t")
    assert list(out["scene_id"]) == [1, 2, 3]          # renumbered contiguously
    assert list(out["start"]) == [0.0, 50.0, 100.0]
    assert list(out["end"]) == [50.0, 100.0, 150.0]
    # first sub-scene inherits desc + shot_ids; the new sub-scene is tagged, shot_ids cleared
    assert out.loc[0, "scene_desc"] == "Central Perk"
    assert out.loc[1, "scene_desc"] == "Central Perk [topic_aug 1]"
    assert str(out.loc[1, "shot_ids"]) in ("", "nan")
    assert stats["n_new_boundaries"] == 1
    assert stats["n_output_scenes"] == 3


def test_subdivide_dedupes_and_drops_out_of_range(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _write_scene_table(in_dir, "s01e01a", [
        {"scene_id": 1, "scene_desc": "Perk", "start": 0.0, "end": 100.0, "shot_ids": "1"},
    ])

    def propose(scene):
        return [50.0, 50.0, 200.0, 0.0]  # duplicate, out-of-range high, equals start

    stats = subdivide_episode("s01e01a", in_dir, out_dir, propose, aug_tag="topic_aug")
    out = pd.read_csv(out_dir / "s1" / "friends_s01e01a_scene_summary.tsv", sep="\t")
    assert list(out["start"]) == [0.0, 50.0]   # one cut at 50, no zero-length rows
    assert list(out["end"]) == [50.0, 100.0]
    assert stats["n_new_boundaries"] == 1


def test_subdivide_pass_through_when_no_subs(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _write_scene_table(in_dir, "s01e01a", [
        {"scene_id": 1, "scene_desc": "Perk", "start": 0.0, "end": 50.0, "shot_ids": "1|2"},
        {"scene_id": 2, "scene_desc": "Monica's", "start": 50.0, "end": 90.0, "shot_ids": "3"},
    ])
    stats = subdivide_episode("s01e01a", in_dir, out_dir, lambda s: [], aug_tag="topic_aug")
    out = pd.read_csv(out_dir / "s1" / "friends_s01e01a_scene_summary.tsv", sep="\t")
    assert list(out["scene_id"]) == [1, 2]
    assert list(out["scene_desc"]) == ["Perk", "Monica's"]
    assert str(out.loc[0, "shot_ids"]) == "1|2"
    assert stats["n_new_boundaries"] == 0
    assert stats["n_output_scenes"] == 2
