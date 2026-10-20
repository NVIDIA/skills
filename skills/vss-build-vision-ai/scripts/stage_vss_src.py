#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Stage the local checkout's harness inputs into a harness build context.
#
# NemoClaw's `onboard --from <Dockerfile>` fixes the Docker build context to
# the Dockerfile's parent directory, so .openclaw/ and .hermes/ cannot COPY
# from the repo root; their Dockerfiles fetch one pinned commit (VSS_REF)
# instead. That stays the shipping path — reproducible, provenance-carrying —
# but a local skill or CLI edit cannot reach the image until it is pushed and
# the pin moves. This tool is the dev-loop override: it snapshots the working
# tree's harness inputs (STAGE_ROOTS below) into <harness>/.vss-src/, which
# the Dockerfiles use INSTEAD of the pinned fetch when present. `--clean`
# removes the snapshots, returning builds to the pin.
#
# The snapshot is the working tree as git sees it: tracked files plus
# untracked-but-not-ignored ones (`git ls-files --cached --others
# --exclude-standard`), so .git, .venv, __pycache__ and friends never ride
# along — matching NemoClaw's own build-context name filter, which drops them.
#
# A STAGED marker records provenance (HEAD sha, dirty flag, timestamp) plus a
# PEP 440 `version=` the Dockerfiles export as SETUPTOOLS_SCM_PRETEND_VERSION:
# the vss CLI's hatch-vcs versioning reads git metadata, and a snapshot has
# none. The version mimics services/agent's no-guess-dev scheme (release tag,
# or tag.post1.devN plus +g<sha>[.dirty] when off a release) — nothing parses
# it, but `vss --version` should tell the truth about what it is.

from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess
import sys
from pathlib import Path

# Everything the harness Dockerfiles consume from the VSS checkout. A unit
# test (test_stage_vss_src.py) parses the Dockerfiles and fails when they
# start consuming a path outside these roots. libs/vss is the CLI library
# workspace (nvidia-vss-cli + nvidia-vss-core), the images' whole install
# surface — the agent service under services/agent is deliberately not staged.
STAGE_ROOTS = ("skills", ".openclaw/workspace", "libs/vss")
STAGE_DIR_NAME = ".vss-src"
MARKER_NAME = "STAGED"
DEFAULT_HARNESS_DIRS = (".openclaw", ".hermes")


def git(repo_root: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True, text=True, check=True,
    )
    return r.stdout.strip()


def snapshot_files(repo_root: Path) -> list[Path]:
    """The staged roots as git sees them: tracked + untracked-unignored,
    minus files deleted from the working tree (still in the index)."""
    out = git(repo_root, "ls-files", "-z", "--cached", "--others",
              "--exclude-standard", "--", *STAGE_ROOTS)
    rels = sorted({Path(p) for p in out.split("\0") if p})
    return [p for p in rels if (repo_root / p).is_file()]


def pretend_version(repo_root: Path, sha: str, dirty: bool) -> str:
    try:
        tag = git(repo_root, "describe", "--tags", "--match", "v[0-9]*", "--abbrev=0")
    except subprocess.CalledProcessError:
        tag = ""  # no release tag reachable (fresh or shallow clone)
    base = tag.lstrip("v") or "0.0.0"
    anchor = tag or ""
    distance = int(git(repo_root, "rev-list", "--count", f"{anchor}..HEAD" if anchor else "HEAD"))
    if distance == 0 and not dirty:
        return base
    return f"{base}.post1.dev{distance}+g{sha[:10]}" + (".dirty" if dirty else "")


def provenance(repo_root: Path) -> dict[str, str]:
    sha = git(repo_root, "rev-parse", "HEAD")
    # Dirty relative to what is staged: an edit elsewhere in the repo does not
    # change image content, so it does not taint the snapshot.
    dirty = bool(git(repo_root, "status", "--porcelain", "--", *STAGE_ROOTS))
    return {
        "sha": sha,
        "dirty": "true" if dirty else "false",
        "version": pretend_version(repo_root, sha, dirty),
        "staged_at": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def resolve_harness_dir(repo_root: Path, name: str) -> Path:
    d = Path(name)
    if not d.is_absolute():
        d = repo_root / d
    if not (d / "Dockerfile").is_file():
        raise FileNotFoundError(f"{d} has no Dockerfile — not a harness build context")
    return d


def stage(repo_root: Path, harness_dir: Path, files: list[Path], marker: dict[str, str]) -> int:
    dest = harness_dir / STAGE_DIR_NAME
    if dest.exists():
        shutil.rmtree(dest)
    for rel in files:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / rel, target)
    (dest / MARKER_NAME).write_text(
        "".join(f"{k}={v}\n" for k, v in marker.items()), encoding="utf-8"
    )
    return len(files)


def clean(harness_dir: Path) -> bool:
    dest = harness_dir / STAGE_DIR_NAME
    if dest.exists():
        shutil.rmtree(dest)
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Snapshot the working tree's harness inputs into <harness>/.vss-src/, "
                    "overriding the Dockerfiles' pinned VSS_REF fetch.")
    ap.add_argument("harness_dirs", nargs="*", metavar="HARNESS_DIR",
                    help=f"harness build-context dirs (default: {' '.join(DEFAULT_HARNESS_DIRS)})")
    ap.add_argument("--repo-root", type=Path,
                    default=Path(__file__).resolve().parents[3],
                    help="VSS checkout to snapshot (default: this script's repo)")
    ap.add_argument("--clean", action="store_true",
                    help="remove the staged snapshots instead of creating them")
    a = ap.parse_args(argv)
    repo_root = a.repo_root.resolve()
    try:
        git(repo_root, "rev-parse", "--git-dir")
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"[vss] {repo_root} is not a git checkout: {exc}", file=sys.stderr)
        return 1
    try:
        harness_dirs = [resolve_harness_dir(repo_root, n)
                        for n in (a.harness_dirs or DEFAULT_HARNESS_DIRS)]
        if a.clean:
            for d in harness_dirs:
                removed = clean(d)
                print(f"[vss] {d / STAGE_DIR_NAME}: {'removed' if removed else 'not present'}")
            return 0
        files = snapshot_files(repo_root)
        if not files:
            print(f"[vss] nothing to stage under {', '.join(STAGE_ROOTS)}", file=sys.stderr)
            return 1
        marker = provenance(repo_root)
        for d in harness_dirs:
            n = stage(repo_root, d, files, marker)
            print(f"[vss] staged {n} files into {d / STAGE_DIR_NAME} "
                  f"(sha {marker['sha'][:10]}, dirty={marker['dirty']}, "
                  f"version {marker['version']})")
        print("[vss] next build of these contexts uses the snapshot instead of the "
              "pinned VSS_REF; run with --clean to go back to the pin")
        return 0
    except (subprocess.CalledProcessError, OSError, FileNotFoundError) as exc:
        print(f"[vss] staging failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
