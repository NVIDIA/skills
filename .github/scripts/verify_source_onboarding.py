#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Verify source-repo skills before an onboarding PR is merged.

An onboarding PR adds ``components.d/<slug>.yml`` pointing at skill
directories in a product repo. Nothing on the PR looks at those directories:
``verify_content_integrity.py`` checks *this* repository's tree, which is
sound, while the content being onboarded lives somewhere else entirely. The
PR therefore goes green on evidence that has nothing to do with what is
being merged.

The cost lands after the merge. The sync fetches the source directory,
recomputes each signed file's digest, and refuses the skill on any mismatch:
an existing skill reverts to its last good version, and a *new* skill is
dropped outright, because there is no earlier version to fall back to. The PR
is closed by then, so the failure has nowhere to surface. The catalog looks
correct and the skill is simply absent.

That happened twice on 2026-08-31. ``paidf-augmentation`` (#501) merged green
and was dropped by the next sync; ``paidf-orchestration`` (#507) carried the
same defect in all four declared skills and was caught only because someone
verified the signatures by hand during review. In every case the offending
file was ``BENCHMARK.md``, regenerated after the signing run so the signature
covered an earlier copy — an easy sequencing mistake that is invisible until
it is expensive.

This script follows the pointer. For each skill directory a PR declares it
fetches the source at the component's ``ref`` and checks that:

  * every file listed in ``skill.oms.sig`` is present and still hashes to its
    signed digest;
  * ``SKILL.md``, ``skill.oms.sig``, ``skill-card.md`` and an eval dataset
    exist, applying the same filename rules the hourly sync applies;
  * ``BENCHMARK.md`` reports an overall verdict of PASS backed by real
    measurements.

Blocking scope: only paths the PR *adds*. Around 30 catalog skills already
carry signature drift (tracked in #216 / #357); failing a PR for drift its
author neither caused nor can fix would make the gate an obstacle rather than
a signal. Pre-existing paths are still checked, and reported, but never fail
the build.

Reads the source repository read-only over an anonymous clone. No signing
credentials, and nothing here writes to the source repo.

Exit code 0 = no blocking problems; 1 = a newly-added skill is not fit to
merge.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aggregate_benchmarks as ab  # noqa: E402
import verify_content_integrity as vci  # noqa: E402

COMPONENTS_DIR = Path("components.d")
SIG_NAME = vci.SIG_NAME

# Required alongside the signature. The card name is exactly skill-card.md:
# the hourly sync drops a skill that lacks that file (sync-skills.yml), so
# accepting SKILLCARD.yaml or card.yaml here would pass a skill at onboarding
# that the next sync silently removes. Both spellings do appear in the catalog
# (SKILLCARD.yaml 47, card.yaml 18) but always *alongside* skill-card.md,
# which all 350 published skills carry — no skill relies on an alternate.
REQUIRED_FILES = ("SKILL.md", SIG_NAME, "BENCHMARK.md", "skill-card.md")

RE_REPO = re.compile(r"^repo:\s*(\S+)\s*$")
RE_REF = re.compile(r"^ref:\s*(\S+)\s*$")
RE_PATH = re.compile(r"^-?\s*path:\s*(\S+)\s*$")


@dataclass(frozen=True)
class Finding:
    """One problem with one skill directory.

    ``blocking`` records whether this path was added by the PR under review.
    Only blocking findings affect the exit code; the rest are reported so a
    reviewer can see them without the author being held responsible.
    """

    path: str
    problem: str
    blocking: bool


# --------------------------------------------------------------------------
# Component files
# --------------------------------------------------------------------------

def parse_component(text: str) -> dict:
    """Read repo, ref and declared skill paths from a component file.

    Deliberately line-based rather than PyYAML: every other script in this
    directory parses these files the same way, and CI carries no third-party
    Python dependencies. The schema is flat and fixed (components.d/README.md),
    so a real parser buys nothing here.

    Trailing slashes are stripped so a path compares equal however it was
    written — ``skills/write-dag/`` and ``skills/write-dag`` are the same
    directory, and treating them as different would report a path as newly
    added when a PR only reformatted it.
    """
    repo = None
    ref = None
    paths: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if repo is None:
            m = RE_REPO.match(stripped)
            if m:
                repo = m.group(1)
                continue
        if ref is None:
            m = RE_REF.match(stripped)
            if m:
                ref = m.group(1)
                continue
        m = RE_PATH.match(stripped)
        if m:
            paths.append(m.group(1).rstrip("/"))
    # An absent ref means the component tracks the default branch. Several
    # components (AIQ, VSS) track develop instead; verifying those against
    # main compares a tree the sync will never fetch.
    return {"repo": repo, "ref": ref or "main", "paths": paths}


def added_paths(base_text: str | None, head_text: str) -> set[str]:
    """Skill paths this PR introduces.

    ``base_text`` is None when the component file itself is new, in which case
    every path in it is new — the common onboarding case, and the one where a
    bad signature causes the skill to be dropped rather than reverted.
    """
    head = set(parse_component(head_text)["paths"])
    if base_text is None:
        return head
    return head - set(parse_component(base_text)["paths"])


# --------------------------------------------------------------------------
# Per-skill verification (pure — operates on a directory on disk)
# --------------------------------------------------------------------------

def check_signature(skill_dir: Path) -> list[str]:
    """Content-vs-signature check, reusing the catalog's integrity logic.

    ``vci.verify_skill`` recomputes each signed file's sha256 and compares it
    to the digest in the DSSE payload. It takes a directory and does not care
    whose repository that directory came from, so pointing it at a clone of
    the source is the entire check — a second implementation here would be a
    copy that could drift from the one the sync effectively enforces.

    One behaviour is deliberately overridden: verify_skill treats a missing
    signature as out of scope and returns no problems, because a separate gate
    covers presence for already-published skills. On an onboarding PR there is
    no such gate upstream, and silently passing an unsigned skill is exactly
    the outcome this script exists to prevent.
    """
    if not (skill_dir / SIG_NAME).is_file():
        return [f"{SIG_NAME}: MISSING — the skill is not signed"]
    # verify_skill prefixes each problem with the directory it was given,
    # which here is a temporary clone path. Strip it so the reader sees the
    # file to re-sign rather than /tmp/source-onboarding-xyz/skills/....
    prefix = f"{skill_dir}/"
    return [p[len(prefix):] if p.startswith(prefix) else p
            for p in vci.verify_skill(skill_dir)]


def has_eval_dataset(skill_dir: Path) -> bool:
    """Mirror the eval-dataset rule the hourly sync applies.

    The sync accepts an ``evals.json`` at any depth, or any ``*.json`` under
    ``evals/`` or ``eval/``, which is what lets multi-profile datasets through
    (rag-blueprint, rag-eval and rag-perf ship ``eval/*.json`` rather than the
    canonical single file). Requiring ``evals/evals.json`` exactly would fail
    an onboarding PR the sync would carry happily.
    """
    if any(skill_dir.rglob("evals.json")):
        return True
    return any(
        (skill_dir / sub).is_dir() and any((skill_dir / sub).glob("*.json"))
        for sub in ("evals", "eval")
    )


def check_required_files(skill_dir: Path) -> list[str]:
    problems = []
    for rel in REQUIRED_FILES:
        if not (skill_dir / rel).is_file():
            problems.append(f"{rel}: MISSING — required for onboarding")
    if not has_eval_dataset(skill_dir):
        problems.append(
            "eval dataset: MISSING — expected evals/evals.json, or any "
            "*.json under evals/ or eval/"
        )
    return problems


def check_benchmark(skill_dir: Path) -> list[str]:
    """Require a PASS verdict backed by actual measurements.

    ``ab.parse_benchmark`` handles the v1, v2 and v3 report layouts, which
    state the verdict in different places; matching the string by hand would
    read v2's methodology bullet ("Overall verdict: PASS only when every
    configured dimension...") as a passing verdict on every v2 report.

    A verdict alone is not enough. A report can carry PASS in its header while
    its results table is an unfilled template, which is a claim with no
    evidence under it — the same has_results:false condition that left
    rag-blueprint and rag-eval reporting a verdict no run ever produced.
    """
    bm = skill_dir / "BENCHMARK.md"
    if not bm.is_file():
        return []  # already reported by check_required_files

    try:
        entry = ab.parse_benchmark(bm)
    except Exception as exc:
        return [f"BENCHMARK.md: could not be parsed: {exc}"]

    problems = []
    verdict = entry.get("verdict")
    if verdict != "PASS":
        problems.append(
            f"BENCHMARK.md: overall verdict is {verdict or 'absent'}, expected PASS"
        )
    if not entry.get("results"):
        problems.append(
            "BENCHMARK.md: reports no results — the verdict is not backed by "
            "any measurements"
        )
    return problems


def check_skill(skill_dir: Path) -> list[str]:
    """All problems with one skill directory (empty == fit to merge)."""
    if not skill_dir.is_dir():
        return ["directory does not exist in the source repository at this ref"]
    return (
        check_required_files(skill_dir)
        + check_signature(skill_dir)
        + check_benchmark(skill_dir)
    )


# --------------------------------------------------------------------------
# Fetching the source
# --------------------------------------------------------------------------

def fetch_source(repo: str, ref: str, paths: list[str], dest: Path) -> Path | None:
    """Sparse-clone the declared skill directories at the component's ref.

    Mirrors the sync's own fetch (shallow, blobless, sparse) so this check
    sees the same tree the sync will. Anonymous over HTTPS: the source repos
    are public, and the check must not need credentials to run on a fork PR.

    Returns None when the clone fails — an unreachable repo or a ref that does
    not exist is itself a finding, not a reason to crash.
    """
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             "-b", ref, f"https://github.com/{repo}.git", str(dest)],
            capture_output=True, text=True, check=True, timeout=300,
        )
        if paths:
            subprocess.run(
                ["git", "sparse-checkout", "set", *paths],
                cwd=dest, capture_output=True, text=True, check=True, timeout=120,
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return dest


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def changed_component_files(base_sha: str, head_sha: str) -> list[str]:
    diff = subprocess.run(
        ["git", "diff", "--name-only", f"{base_sha}...{head_sha}"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return sorted(
        p for p in diff
        if p.startswith(f"{COMPONENTS_DIR}/") and p.endswith(".yml")
    )


def file_at(sha: str, path: str) -> str | None:
    """A file's contents at a commit, or None when it did not exist there."""
    result = subprocess.run(
        ["git", "show", f"{sha}:{path}"], capture_output=True, text=True,
    )
    return result.stdout if result.returncode == 0 else None


def exit_code(findings: list[Finding]) -> int:
    return 1 if any(f.blocking for f in findings) else 0


def main() -> int:
    base_sha = os.environ.get("BASE_SHA")
    head_sha = os.environ.get("HEAD_SHA")
    if not base_sha or not head_sha:
        print("BASE_SHA and HEAD_SHA are required; nothing to check.")
        return 0

    component_files = changed_component_files(base_sha, head_sha)
    print(f"Source-onboarding check — {len(component_files)} changed component file(s)")
    if not component_files:
        print("No components.d changes in scope; nothing to verify.")
        return 0

    findings: list[Finding] = []
    workdir = Path(tempfile.mkdtemp(prefix="source-onboarding-"))
    try:
        for rel in component_files:
            head_text = file_at(head_sha, rel)
            if head_text is None:
                # Deleted by this PR — there is no source left to verify.
                print(f"\n── {rel} (removed) ──")
                continue
            component = parse_component(head_text)
            repo, ref = component["repo"], component["ref"]
            new = added_paths(file_at(base_sha, rel), head_text)

            print(f"\n── {rel} → {repo}@{ref} ──")
            if not repo:
                findings.append(Finding(rel, "no repo: declared", blocking=True))
                print("  FAIL  no repo: declared")
                continue

            clone = fetch_source(
                repo, ref, component["paths"],
                workdir / rel.replace("/", "-"),
            )
            if clone is None:
                findings.append(Finding(
                    rel, f"could not clone {repo} at ref {ref}", blocking=True,
                ))
                print(f"  FAIL  could not clone {repo} at ref {ref}")
                continue

            for path in component["paths"]:
                blocking = path in new
                problems = check_skill(clone / path)
                label = "new" if blocking else "existing"
                if not problems:
                    print(f"  ok    {path} ({label})")
                    continue
                print(f"  {'FAIL' if blocking else 'warn'}  {path} ({label})")
                for p in problems:
                    print(f"          {p}")
                    findings.append(Finding(path, p, blocking))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    blocking = [f for f in findings if f.blocking]
    advisory = [f for f in findings if not f.blocking]

    if advisory:
        print(f"\n{len(advisory)} problem(s) on skills this PR did not add. "
              "Reported for visibility; not blocking this merge. "
              "Catalog-wide drift is tracked in #216 and #357.")

    if not blocking:
        print("\nOK — every skill this PR adds is fit to merge.")
        return 0

    print(f"\nFAILED — {len(blocking)} problem(s) on skills this PR adds:")
    for f in blocking:
        print(f"  - {f.path}: {f.problem}")

    # Only explain re-signing when something actually drifted. Printing it
    # under a missing directory or an absent evals file sends the author to
    # re-run signing over a problem signing does not fix.
    if any("MISMATCH" in f.problem for f in blocking):
        print(
            "\nA MISMATCH means the file changed after the signing run, so the "
            "signature covers an earlier copy. This is most often BENCHMARK.md, "
            "which is easy to regenerate after signing. The sync applies the "
            "same check and will drop the skill rather than publish it, so "
            "fixing it here is the difference between a red check and a skill "
            "that silently never appears.\n"
            "\nTo fix: re-run signing in the source repository so the signature "
            "covers the current content, and do not modify skill files "
            "afterward."
        )
    print("\nSee CONTRIBUTING.md for the onboarding requirements, or ask a CODEOWNER.")
    return exit_code(findings)


if __name__ == "__main__":
    sys.exit(main())
