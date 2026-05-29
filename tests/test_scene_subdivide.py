import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from charnet.scene_subdivide import expand_episode_spec


def _touch_scene_files(root: Path, episodes: list[str]) -> None:
    for ep in episodes:
        season = int(ep[1:3])
        d = root / f"s{season}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"friends_{ep}_scene_summary.tsv").write_text("scene_id\tstart\tend\n")


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
