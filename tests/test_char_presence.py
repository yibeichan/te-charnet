import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from charnet.char_presence import jaccard_distance, propose_sub_boundaries


def test_jaccard_distance_basics():
    assert jaccard_distance(set(), set()) == 0.0
    assert jaccard_distance({"a"}, {"a"}) == 0.0
    assert jaccard_distance({"a"}, {"b"}) == 1.0
    assert jaccard_distance({"a", "b"}, {"a"}) == 0.5


def test_propose_fires_on_persistent_char_change():
    # 0-60s: chars present = {ross}; 60-120s: {monica}. One grid col each.
    chars = ["ross", "monica"]
    rows = [[1, 0]] * 60 + [[0, 1]] * 60
    subs = propose_sub_boundaries(
        0.0, 120.0, chars, rows,
        tile_secs=5.0, presence_frac=0.2, jaccard_thresh=0.5,
        min_spacing=15.0, persistence_tiles=2, shot_times=None,
        shot_snap_window=3.0, shot_snap_required=False, min_scene_length=0.0,
    )
    assert subs == [60.0]


def test_propose_ignores_transient_flicker():
    chars = ["ross", "monica"]
    # 5s monica blip near the end (tiles 22-23), then only 1 tile of ross —
    # the new set never holds for persistence_tiles=2 in either direction so no
    # boundary is accepted.
    rows = [[1, 0]] * 110 + [[0, 1]] * 5 + [[1, 0]] * 5
    subs = propose_sub_boundaries(
        0.0, 120.0, chars, rows,
        tile_secs=5.0, presence_frac=0.2, jaccard_thresh=0.5,
        min_spacing=15.0, persistence_tiles=2, shot_times=None,
        shot_snap_window=3.0, shot_snap_required=False, min_scene_length=0.0,
    )
    assert subs == []
