#!/usr/bin/env python
# scripts/write_manual_seg_manifest.py
"""Write the pinned manifest of the manual scene/segment annotations.

The gold labels for scripts/evaluate_scene_segmentation.py live in a local,
gitignored clone of courtois-neuromod/friends_annotations under
data/friends_annotations/. This script records which commit of that repo the
evaluation uses, plus a sha256 for every tracked file under
annotation_results/manual_segmentation/, in data/manifests/friends_annotations.json.
The evaluator checks the clone against this manifest before scoring.

Refuses to write (exit 1) when:
  - the clone's HEAD is not the commit passed with --commit, or
  - the clone has modified, staged, untracked or ignored files under the
    manual_segmentation path.

Usage:
    python scripts/write_manual_seg_manifest.py --commit 3e5f108
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CLONE = REPO / "data/friends_annotations"
DEFAULT_OUT = REPO / "data/manifests/friends_annotations.json"
DEFAULT_SOURCE_REPO = "https://github.com/courtois-neuromod/friends_annotations.git"
PATH_PREFIX = "annotation_results/manual_segmentation"


class ManifestError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(clone: Path, *args: str) -> str:
    res = subprocess.run(["git", "-C", str(clone), *args],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise ManifestError(f"git {' '.join(args)} failed in {clone}: {res.stderr.strip()}")
    return res.stdout


def build_manifest(clone: Path, commit: str, source_repo: str = DEFAULT_SOURCE_REPO,
                   path_prefix: str = PATH_PREFIX, generated: str | None = None) -> dict:
    """Hash the files under ``path_prefix`` in ``clone`` after checking its state."""
    head = _git(clone, "rev-parse", "HEAD").strip()
    want = _git(clone, "rev-parse", "--verify", f"{commit}^{{commit}}").strip()
    if head != want:
        raise ManifestError(
            f"clone HEAD is {head}, not the requested commit {want}; "
            f"check out {commit} in {clone} first"
        )
    dirty = _git(clone, "status", "--porcelain", "--untracked-files=all",
                 "--ignored=matching", "--", path_prefix).strip()
    if dirty:
        raise ManifestError(
            f"clone has local changes under {path_prefix}:\n{dirty}"
        )
    tracked = [p for p in _git(clone, "ls-files", "-z", "--", path_prefix).split("\0") if p]
    if not tracked:
        raise ManifestError(f"no tracked files under {path_prefix} at {head}")

    prefix = path_prefix.rstrip("/") + "/"
    files = {p[len(prefix):]: sha256_file(clone / p) for p in sorted(tracked)}
    return {
        "source_repo": source_repo,
        "commit": head,
        "path_prefix": path_prefix,
        "generated": generated or dt.date.today().isoformat(),
        "n_files": len(files),
        "files": files,
    }


def write_manifest(manifest: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", required=True,
                    help="Commit of the annotations repo to record; the clone's HEAD must be it.")
    ap.add_argument("--clone", default=str(DEFAULT_CLONE),
                    help="Local clone of friends_annotations (default: data/friends_annotations).")
    ap.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO,
                    help="Source repo URL to record (default: the underscore repo).")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="Manifest path (default: data/manifests/friends_annotations.json).")
    args = ap.parse_args(argv)

    try:
        manifest = build_manifest(Path(args.clone), args.commit, source_repo=args.source_repo)
    except ManifestError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    out = Path(args.out)
    write_manifest(manifest, out)
    print(f"Wrote {out} ({manifest['n_files']} files at {manifest['commit']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
