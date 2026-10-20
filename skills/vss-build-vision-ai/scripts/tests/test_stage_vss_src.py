# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior contract for stage_vss_src.py, the dev-loop pre-staging tool.

Pins what the harness Dockerfiles rely on: the snapshot is git's view of the
working tree (tracked + untracked-unignored, no ignored artifacts), the STAGED
marker carries usable provenance (sha, dirty, a PEP 440 pretend version for
the CLI's hatch-vcs install), staging is exact-set, and — the drift guard —
the real Dockerfiles never consume a /opt/vss-src path outside STAGE_ROOTS.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stage_vss_src  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]

# PEP 440: release, optional .postN.devM, optional local segment.
PEP440 = re.compile(r"^\d+(\.\d+)*(\.post\d+\.dev\d+)?(\+[a-z0-9]+(\.[a-z0-9]+)*)?$")


def git(root, *args):
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True, capture_output=True,
    )


def read_marker(harness_dir):
    text = (harness_dir / ".vss-src" / "STAGED").read_text()
    return dict(line.split("=", 1) for line in text.splitlines())


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "skills" / "vss-x").mkdir(parents=True)
    (root / "skills" / "vss-x" / "SKILL.md").write_text('---\nvss-requires: "always"\n---\n')
    (root / ".openclaw" / "workspace" / "_nemoclaw").mkdir(parents=True)
    (root / ".openclaw" / "workspace" / "AGENTS.md").write_text("# agents\n")
    (root / ".openclaw" / "workspace" / "_nemoclaw" / "ENV.md").write_text("# env\n")
    (root / "services" / "agent").mkdir(parents=True)
    (root / "services" / "agent" / "pyproject.toml").write_text('[project]\nname = "nvidia-vss"\n')
    (root / "libs" / "vss" / "core").mkdir(parents=True)
    (root / "libs" / "vss" / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["core"]\n')
    (root / "libs" / "vss" / "core" / "pyproject.toml").write_text('[project]\nname = "nvidia-vss-core"\n')
    (root / "libs" / "other").mkdir(parents=True)
    (root / "libs" / "other" / "keep.txt").write_text("not a staged root\n")
    (root / ".openclaw" / "Dockerfile").write_text("FROM scratch\n")
    (root / ".hermes").mkdir()
    (root / ".hermes" / "Dockerfile").write_text("FROM scratch\n")
    (root / ".gitignore").write_text("__pycache__/\n*.secret\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    git(root, "tag", "v1.2.3")
    return root


def run(root, *argv):
    return stage_vss_src.main(["--repo-root", str(root), *argv])


# --- what gets staged ---------------------------------------------------------

def test_stages_all_roots_into_both_default_harness_dirs(repo):
    assert run(repo) == 0
    for harness in (".openclaw", ".hermes"):
        dest = repo / harness / ".vss-src"
        assert (dest / "skills" / "vss-x" / "SKILL.md").is_file()
        assert (dest / ".openclaw" / "workspace" / "AGENTS.md").is_file()
        assert (dest / ".openclaw" / "workspace" / "_nemoclaw" / "ENV.md").is_file()
        # libs/vss (nvidia-vss-cli + nvidia-vss-core) is the images' whole
        # install surface; the agent service package is deliberately not staged.
        assert (dest / "libs" / "vss" / "core" / "pyproject.toml").is_file()
        assert not (dest / "services").exists()
        assert not (dest / "libs" / "other").exists()
        assert (dest / "STAGED").is_file()


def test_untracked_included_ignored_and_git_excluded(repo):
    (repo / "skills" / "vss-new").mkdir()
    (repo / "skills" / "vss-new" / "SKILL.md").write_text("untracked, not ignored\n")
    pyc = repo / "skills" / "vss-x" / "__pycache__"
    pyc.mkdir()
    (pyc / "x.pyc").write_text("ignored\n")
    (repo / "skills" / "creds.secret").write_text("ignored\n")
    assert run(repo, ".openclaw") == 0
    dest = repo / ".openclaw" / ".vss-src"
    assert (dest / "skills" / "vss-new" / "SKILL.md").is_file()
    assert not (dest / "skills" / "vss-x" / "__pycache__").exists()
    assert not (dest / "skills" / "creds.secret").exists()
    assert not (dest / ".git").exists()


def test_file_deleted_from_worktree_is_not_staged(repo):
    (repo / ".openclaw" / "workspace" / "_nemoclaw" / "ENV.md").unlink()
    assert run(repo, ".openclaw") == 0
    dest = repo / ".openclaw" / ".vss-src"
    assert not (dest / ".openclaw" / "workspace" / "_nemoclaw" / "ENV.md").exists()
    assert (dest / ".openclaw" / "workspace" / "AGENTS.md").is_file()


def test_restage_removes_stale_content(repo):
    stale = repo / ".openclaw" / ".vss-src" / "skills" / "vss-gone"
    stale.mkdir(parents=True)
    (stale / "SKILL.md").write_text("from an older snapshot\n")
    assert run(repo, ".openclaw") == 0
    assert not stale.exists()


def test_clean_removes_snapshots(repo):
    assert run(repo) == 0
    assert run(repo, "--clean") == 0
    assert not (repo / ".openclaw" / ".vss-src").exists()
    assert not (repo / ".hermes" / ".vss-src").exists()


def test_harness_dir_without_dockerfile_is_refused(repo):
    (repo / "not-a-harness").mkdir()
    assert run(repo, "not-a-harness") == 1


# --- the STAGED marker --------------------------------------------------------

def test_marker_provenance_clean_tree(repo):
    assert run(repo, ".openclaw") == 0
    m = read_marker(repo / ".openclaw")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert m["sha"] == head
    assert m["dirty"] == "false"
    assert m["version"] == "1.2.3"  # exactly on the release tag, clean
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", m["staged_at"])


def test_marker_dirty_tree_gets_local_version(repo):
    (repo / "skills" / "vss-x" / "SKILL.md").write_text("edited\n")
    assert run(repo, ".openclaw") == 0
    m = read_marker(repo / ".openclaw")
    assert m["dirty"] == "true"
    assert re.fullmatch(r"1\.2\.3\.post1\.dev0\+g[0-9a-f]{10}\.dirty", m["version"])
    assert PEP440.fullmatch(m["version"])


def test_marker_dirty_only_counts_staged_roots(repo):
    (repo / "unrelated.txt").write_text("elsewhere in the repo\n")
    git(repo, "add", "unrelated.txt")
    assert run(repo, ".openclaw") == 0
    assert read_marker(repo / ".openclaw")["dirty"] == "false"


def test_version_without_release_tag_is_pep440(repo):
    git(repo, "tag", "-d", "v1.2.3")
    assert run(repo, ".openclaw") == 0
    m = read_marker(repo / ".openclaw")
    assert re.fullmatch(r"0\.0\.0\.post1\.dev1\+g[0-9a-f]{10}", m["version"])
    assert PEP440.fullmatch(m["version"])


# --- CLI errors ---------------------------------------------------------------

def test_not_a_git_checkout_exits_one(tmp_path):
    assert stage_vss_src.main(["--repo-root", str(tmp_path)]) == 1


# --- drift guard against the real Dockerfiles ---------------------------------

def test_dockerfiles_consume_only_staged_paths():
    """Every /opt/vss-src/<path> a harness Dockerfile touches must be covered
    by STAGE_ROOTS (or be the STAGED marker / the COPY-anchor Dockerfile) —
    otherwise a pre-staged build silently diverges from a pinned one."""
    allowed_files = {"STAGED", "Dockerfile"}
    refs = set()
    for df in (REPO_ROOT / ".openclaw" / "Dockerfile", REPO_ROOT / ".hermes" / "Dockerfile"):
        for m in re.finditer(r"/opt/vss-src/([^\s\"'();]+)", df.read_text()):
            refs.add(m.group(1).rstrip("/"))
    assert refs, "Dockerfiles no longer reference /opt/vss-src — update this test"
    for ref in refs:
        covered = ref in allowed_files or any(
            ref == root or ref.startswith(root + "/") for root in stage_vss_src.STAGE_ROOTS
        )
        assert covered, (
            f"Dockerfile consumes /opt/vss-src/{ref}, which stage_vss_src.py does not "
            f"stage (STAGE_ROOTS={stage_vss_src.STAGE_ROOTS}) — a pre-staged build "
            "would miss it"
        )
