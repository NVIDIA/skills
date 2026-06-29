#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Verify that each skill's files still match its signed manifest.

A ``skill.oms.sig`` is a Sigstore bundle wrapping a DSSE / in-toto statement
whose ``predicate.resources`` lists every signed file together with its
sha256 digest. Signature *presence* and *authenticity* tell you a genuine
signature exists; they do NOT tell you the files still match what was signed.

A skill can therefore carry a perfectly genuine signature while its content
has drifted — e.g. a team signs commit 1, makes more commits, and the sync
picks up a later content with the stale signature. This script closes that
gap: for each signed file it recomputes the sha256 on disk and compares it to
the signed digest, failing on any mismatch or missing file.

Scope:
  * pull_request  -> only the skills changed in the PR (fast; catches drift
                     being introduced).
  * schedule / workflow_dispatch / anything else -> the whole catalog
                     (safety net; catches pre-existing drift already on main).

Exit code 0 = all checked skills match their signatures; 1 = drift found.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SKILLS_DIR = Path("skills")
SIG_NAME = "skill.oms.sig"


def changed_skill_dirs(base_sha: str, head_sha: str) -> list[Path]:
    """Skill directories touched between the PR merge-base and its head."""
    diff = subprocess.run(
        ["git", "diff", "--name-only", f"{base_sha}...{head_sha}"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    dirs: set[Path] = set()
    for path in diff:
        parts = path.split("/")
        if len(parts) >= 2 and parts[0] == "skills":
            dirs.add(SKILLS_DIR / parts[1])
    return sorted(d for d in dirs if d.is_dir())


def all_skill_dirs() -> list[Path]:
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(d for d in SKILLS_DIR.iterdir() if d.is_dir())


def signed_resources(sig_path: Path) -> list[dict]:
    """Decode the DSSE/in-toto payload and return its resource list."""
    bundle = json.loads(sig_path.read_text())
    payload = base64.b64decode(bundle["dsseEnvelope"]["payload"])
    statement = json.loads(payload)
    return statement["predicate"]["resources"]


def verify_skill(skill_dir: Path) -> list[str]:
    """Return a list of human-readable problems (empty == content matches)."""
    sig = skill_dir / SIG_NAME
    if not sig.exists():
        # Out of scope for this check (a separate gate enforces signature
        # presence). Surface it as a note, but don't fail the integrity check.
        return []
    try:
        resources = signed_resources(sig)
    except Exception as exc:  # malformed bundle is itself a real problem
        return [f"{skill_dir}/{SIG_NAME}: could not parse signature bundle: {exc}"]

    problems: list[str] = []
    for res in resources:
        name = res["name"]
        want = res["digest"]
        target = skill_dir / name
        if not target.is_file():
            problems.append(f"{skill_dir}/{name}: MISSING — listed in signature, absent on disk")
            continue
        got = hashlib.sha256(target.read_bytes()).hexdigest()
        if got != want:
            problems.append(
                f"{skill_dir}/{name}: HASH MISMATCH — content does not match signature"
            )
    return problems


def main() -> int:
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    if event == "pull_request":
        base_sha = os.environ["BASE_SHA"]
        head_sha = os.environ["HEAD_SHA"]
        targets = changed_skill_dirs(base_sha, head_sha)
        scope = f"PR — {len(targets)} changed skill dir(s)"
    else:
        targets = all_skill_dirs()
        scope = f"full catalog — {len(targets)} skill dir(s)"

    print(f"Content-integrity check ({scope})")
    if not targets:
        print("No skill directories in scope; nothing to verify.")
        return 0

    problems: list[str] = []
    unsigned: list[str] = []
    checked = 0
    for skill_dir in targets:
        if not (skill_dir / SIG_NAME).exists():
            if (skill_dir / "SKILL.md").exists():
                unsigned.append(skill_dir.name)
            continue
        checked += 1
        skill_problems = verify_skill(skill_dir)
        if skill_problems:
            problems.extend(skill_problems)
            print(f"  FAIL  {skill_dir.name}")
            for p in skill_problems:
                print(f"          {p}")
        else:
            print(f"  ok    {skill_dir.name}")

    print(f"\nChecked {checked} signed skill(s).")
    if unsigned:
        print(f"Note: {len(unsigned)} changed skill(s) have no {SIG_NAME} "
              f"(reported, not failed here): {', '.join(sorted(unsigned))}")

    if problems:
        print(f"\nFAILED — {len(problems)} content-integrity problem(s):")
        for p in problems:
            print(f"  - {p}")
        print(
            "\nThese files no longer match what their signature signed. The skill "
            "owner must re-run the signing pipeline so content and skill.oms.sig "
            "are consistent, then re-sync. See CONTRIBUTING.md."
        )
        return 1

    print("OK — all checked skills match their signatures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
