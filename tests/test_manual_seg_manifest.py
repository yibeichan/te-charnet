"""Tests for the manual-segmentation manifest: writing it and verifying against it.

Uses temp files and a throwaway local git repo only (no network).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import evaluate_scene_segmentation as EV  # noqa: E402
import write_manual_seg_manifest as W  # noqa: E402

TSV = "segment\tscene\tonset\toffset\n1\t1\t0\t10\n2\t1\t10\t20\n"
EPISODES = ["s01e01a", "s01e01b"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def manual_tree(tmp_path, monkeypatch):
    """Manual dir with two TSVs + manual_seg.json, and a manifest matching it."""
    manual = tmp_path / "manual_segmentation"
    (manual / "s1").mkdir(parents=True)
    contents = {"manual_seg.json": "{}\n"}
    for ep in EPISODES:
        contents[f"s1/friends_{ep}_manualseg.tsv"] = TSV
    for rel, text in contents.items():
        (manual / rel).write_text(text)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "source_repo": W.DEFAULT_SOURCE_REPO,
        "commit": "0" * 40,
        "path_prefix": W.PATH_PREFIX,
        "generated": "2026-01-01",
        "n_files": len(contents),
        "files": {rel: _sha(text) for rel, text in contents.items()},
    }))
    monkeypatch.setattr(EV, "MANUAL_DIR", manual)
    return manual, manifest


def test_verify_match(manual_tree):
    manual, manifest = manual_tree
    assert EV.verify_manual_files(EPISODES, manifest, manual) == []


def test_verify_mismatch_names_file(manual_tree):
    manual, manifest = manual_tree
    (manual / "s1/friends_s01e01b_manualseg.tsv").write_text(TSV + "3\t2\t20\t30\n")
    problems = EV.verify_manual_files(EPISODES, manifest, manual)
    assert problems == ["sha256 mismatch: s1/friends_s01e01b_manualseg.tsv"]


def test_verify_missing_file(manual_tree):
    manual, manifest = manual_tree
    (manual / "s1/friends_s01e01a_manualseg.tsv").unlink()
    problems = EV.verify_manual_files(["s01e01b"], manifest, manual)
    assert problems == ["missing: s1/friends_s01e01a_manualseg.tsv"]


def test_verify_unlisted_episode(manual_tree):
    manual, manifest = manual_tree
    (manual / "s1/friends_s01e02a_manualseg.tsv").write_text(TSV)
    problems = EV.verify_manual_files(EPISODES + ["s01e02a"], manifest, manual)
    assert problems == ["not in manifest: s1/friends_s01e02a_manualseg.tsv"]


def test_verify_missing_manifest(manual_tree, tmp_path):
    manual, _ = manual_tree
    problems = EV.verify_manual_files(EPISODES, tmp_path / "nope.json", manual)
    assert len(problems) == 1 and problems[0].startswith("manifest not found")


def test_main_exits_on_mismatch(manual_tree, tmp_path, monkeypatch, capsys):
    manual, manifest = manual_tree
    (manual / "s1/friends_s01e01a_manualseg.tsv").write_text("tampered\n")
    monkeypatch.setattr(sys, "argv", ["evaluate_scene_segmentation.py",
                                      "--episodes", "s01e01a",
                                      "--out-dir", str(tmp_path / "out")])
    real_verify = EV.verify_manual_files
    monkeypatch.setattr(EV, "verify_manual_files",
                        lambda eps: real_verify(eps, manifest, manual))
    with pytest.raises(SystemExit) as exc:
        EV.main()
    assert exc.value.code == 1
    assert "sha256 mismatch: s1/friends_s01e01a_manualseg.tsv" in capsys.readouterr().err


# --- manifest writer -------------------------------------------------------

def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def clone(tmp_path):
    repo = tmp_path / "friends_annotations"
    seg = repo / W.PATH_PREFIX / "s1"
    seg.mkdir(parents=True)
    (seg / "friends_s01e01a_manualseg.tsv").write_text(TSV)
    (repo / W.PATH_PREFIX / "manual_seg.json").write_text("{}\n")
    (repo / "README.md").write_text("other\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.org",
         "commit", "-q", "-m", "init")
    return repo


def test_writer_hashes_tracked_files(clone):
    head = _git(clone, "rev-parse", "HEAD")
    m = W.build_manifest(clone, head[:7], generated="2026-01-01")
    assert m["commit"] == head
    assert m["n_files"] == 2
    assert m["files"] == {
        "manual_seg.json": _sha("{}\n"),
        "s1/friends_s01e01a_manualseg.tsv": _sha(TSV),
    }


def test_writer_output_verifies(clone, tmp_path, monkeypatch):
    out = tmp_path / "manifest.json"
    assert W.main(["--commit", "HEAD", "--clone", str(clone), "--out", str(out)]) == 0
    manual = clone / W.PATH_PREFIX
    monkeypatch.setattr(EV, "MANUAL_DIR", manual)
    assert EV.verify_manual_files(["s01e01a"], out, manual) == []


def test_writer_refuses_wrong_head(clone):
    first = _git(clone, "rev-parse", "HEAD")
    (clone / "README.md").write_text("changed\n")
    _git(clone, "-c", "user.name=t", "-c", "user.email=t@example.org",
         "commit", "-q", "-am", "second")
    with pytest.raises(W.ManifestError, match="not the requested commit"):
        W.build_manifest(clone, first)


@pytest.mark.parametrize("change", ["modify", "untracked"])
def test_writer_refuses_dirty_tree(clone, change):
    seg = clone / W.PATH_PREFIX / "s1"
    if change == "modify":
        (seg / "friends_s01e01a_manualseg.tsv").write_text("edited\n")
    else:
        (seg / "friends_s01e01b_manualseg.tsv").write_text(TSV)
    with pytest.raises(W.ManifestError, match="local changes"):
        W.build_manifest(clone, "HEAD")


def test_writer_ignores_changes_outside_prefix(clone):
    (clone / "README.md").write_text("edited\n")
    assert W.build_manifest(clone, "HEAD")["n_files"] == 2


# --- library callers and operator messages ----------------------------------

@pytest.fixture
def lib_tree(manual_tree, monkeypatch):
    """manual_tree, with the module-level manifest pointed at it (as a library caller sees it)."""
    manual, manifest = manual_tree
    monkeypatch.setattr(EV, "MANIFEST_PATH", manifest)
    EV._manifest_files.cache_clear()
    yield manual, manifest
    EV._manifest_files.cache_clear()


def test_load_manual_raw_checks_hash_for_library_callers(lib_tree):
    manual, _ = lib_tree
    assert len(EV.load_manual_raw("s01e01a")) == 2
    (manual / "s1/friends_s01e01b_manualseg.tsv").write_text(TSV + "3\t2\t20\t30\n")
    with pytest.raises(EV.GoldMismatchError, match="sha256 mismatch: s1/friends_s01e01b_manualseg.tsv"):
        EV.load_manual_raw("s01e01b")


def test_load_manual_raw_rejects_unlisted_file(lib_tree):
    manual, _ = lib_tree
    (manual / "s1/friends_s01e02a_manualseg.tsv").write_text(TSV)
    with pytest.raises(EV.GoldMismatchError, match="not in manifest"):
        EV.load_manual_raw("s01e02a")


def test_main_message_names_repo_and_commit_when_folder_missing(manual_tree, tmp_path, monkeypatch, capsys):
    manual, manifest = manual_tree
    monkeypatch.setattr(EV, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(EV, "MANUAL_DIR", tmp_path / "absent")
    monkeypatch.setattr(sys, "argv", ["evaluate_scene_segmentation.py", "--episodes", "s01e01a",
                                      "--out-dir", str(tmp_path / "out")])
    real_verify = EV.verify_manual_files
    monkeypatch.setattr(EV, "verify_manual_files",
                        lambda eps: real_verify(eps, manifest, tmp_path / "absent"))
    with pytest.raises(SystemExit):
        EV.main()
    err = capsys.readouterr().err
    assert "manual segmentation folder not found" in err
    assert W.DEFAULT_SOURCE_REPO in err and "0" * 40 in err
    assert len(err.splitlines()) < 12             # no per-file list when the folder is absent


def test_main_message_caps_long_lists(manual_tree, tmp_path, monkeypatch, capsys):
    manual, manifest = manual_tree
    data = json.loads(manifest.read_text())
    data["files"].update({f"s2/friends_s02e{k:02d}a_manualseg.tsv": "0" * 64 for k in range(1, 40)})
    manifest.write_text(json.dumps(data))
    monkeypatch.setattr(EV, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(sys, "argv", ["evaluate_scene_segmentation.py", "--episodes", "s01e01a",
                                      "--out-dir", str(tmp_path / "out")])
    real_verify = EV.verify_manual_files
    monkeypatch.setattr(EV, "verify_manual_files", lambda eps: real_verify(eps, manifest, manual))
    with pytest.raises(SystemExit):
        EV.main()
    err = capsys.readouterr().err
    assert "(39 problem(s))" in err and "... and 19 more" in err


def test_writer_manifest_mode_follows_umask(clone, tmp_path):
    import os
    import stat
    out = tmp_path / "m.json"
    old = os.umask(0o002)
    try:
        assert W.main(["--commit", "HEAD", "--clone", str(clone), "--out", str(out)]) == 0
    finally:
        os.umask(old)
    assert stat.S_IMODE(out.stat().st_mode) == 0o664


def test_writer_refuses_ignored_file(clone):
    (clone / ".gitignore").write_text("*.bak\n")
    _git(clone, "add", ".gitignore")
    _git(clone, "-c", "user.name=t", "-c", "user.email=t@example.org", "commit", "-q", "-m", "ignore")
    (clone / W.PATH_PREFIX / "s1" / "friends_s01e01a_manualseg.tsv.bak").write_text("old\n")
    with pytest.raises(W.ManifestError, match="local changes"):
        W.build_manifest(clone, "HEAD")
