import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from charnet.bids_meta import write_data_dictionary, write_dataset_description


def test_write_data_dictionary_round_trip(tmp_path):
    cols = {"onset": {"Description": "gap time", "Units": "s"}}
    p = tmp_path / "topic_trace.json"
    write_data_dictionary(p, cols)
    loaded = json.loads(p.read_text())
    assert loaded["onset"]["Units"] == "s"
    # idempotent overwrite with new content
    write_data_dictionary(p, {"onset": {"Description": "changed", "Units": "s"}})
    assert json.loads(p.read_text())["onset"]["Description"] == "changed"


def test_write_data_dictionary_creates_parent_dirs(tmp_path):
    p = tmp_path / "nested" / "deeper" / "topic_trace.json"
    write_data_dictionary(p, {"x": {"Description": "y"}})
    assert p.exists()


def test_write_dataset_description_required_keys(tmp_path):
    p = tmp_path / "dataset_description.json"
    write_dataset_description(p, name="charnet annotations", version="abc1234",
                              source_datasets=[{"Description": "NeuroMod Friends"}])
    d = json.loads(p.read_text())
    assert d["Name"] == "charnet annotations"
    assert d["DatasetType"] == "derivative"
    assert d["BIDSVersion"]
    assert d["GeneratedBy"][0]["Name"] == "charnet"
    assert d["GeneratedBy"][0]["Version"] == "abc1234"
    assert d["SourceDatasets"][0]["Description"] == "NeuroMod Friends"


def test_write_dataset_description_defaults_source_datasets(tmp_path):
    p = tmp_path / "dataset_description.json"
    write_dataset_description(p, name="x", version="v")
    assert json.loads(p.read_text())["SourceDatasets"] == []
