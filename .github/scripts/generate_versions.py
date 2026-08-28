#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate versions.json — per-skill content provenance for catalog consumers.

Consumers that mirror skills have no way to learn a mirrored skill changed
short of walking every directory and diffing it. This emits one small file
they can fetch and compare against the digests they last mirrored.

The digest is derived from each skill's ``skill.oms.sig`` rather than from the
upstream commit the sync observed. That matters: when content changes without
a signature refresh, the sync reverts the skill and the old content stays
published. Keying on the signed manifest describes what *is* published, so a
reverted update never shows up as an update.

``content_digest`` is a SHA-256 over the skill's signed file list — each
``name\\0digest`` pair, sorted by name, joined by newlines. Sorting makes it
independent of manifest ordering; the NUL separator keeps a name containing
the separator from colliding with a different name/digest split.

Modes:
    --write  (default)  Regenerate and write versions.json.
    --check             Regenerate in memory; fail if the checked-in file is
                        stale. Used in PR CI.

Git history is required for the commit fields, so run with a full checkout
(``fetch-depth: 0``), the same as verify_content_integrity.py.
"""

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "skills"
OUTPUT = REPO_ROOT / "versions.json"
SCHEMA = Path(__file__).resolve().parent / "versions.schema.json"
SCHEMA_URL = "https://developer.nvidia.com/schemas/versions.schema.json"


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


def content_digest(resources: list[dict]) -> str:
    """Stable digest over the signed file list, independent of manifest order."""
    pairs = sorted((r["name"], r["digest"]) for r in resources)
    blob = "\n".join(f"{name}\0{digest}" for name, digest in pairs)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def last_commit(path: Path) -> tuple[str, str]:
    """(sha, YYYY-MM-DD) of the most recent commit touching ``path``."""
    rel = path.relative_to(REPO_ROOT).as_posix()
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "log", "-1", "--format=%H %cs", "--", rel],
        capture_output=True, text=True, check=True,
    )
    line = result.stdout.strip()
    if not line:
        raise SystemExit(
            f"No git history for {rel}. Run with a full checkout (fetch-depth: 0)."
        )
    sha, date = line.split(" ", 1)
    return sha, date


def build() -> dict:
    skills = []
    for skill_dir in all_skill_dirs():
        sig = skill_dir / "skill.oms.sig"
        if not sig.is_file():
            raise SystemExit(
                f"{skill_dir.name} has no skill.oms.sig — cannot derive a content "
                f"digest. Every published skill must carry a signature."
            )
        sha, date = last_commit(skill_dir)
        skills.append({
            "name": skill_dir.name,
            "path": skill_dir.relative_to(REPO_ROOT).as_posix(),
            "content_digest": content_digest(signed_resources(sig)),
            "last_commit": sha,
            "last_modified": date,
        })
    return {"$schema": SCHEMA_URL, "skills": skills}


def validate(doc: dict) -> None:
    """Fail loudly rather than publishing a malformed contract."""
    jsonschema.validate(doc, json.loads(SCHEMA.read_text()))


def removed_skills(new: dict, old: dict) -> list[str]:
    """Skills present in the checked-in file but missing from a regeneration."""
    return sorted({s["name"] for s in old.get("skills", [])}
                  - {s["name"] for s in new["skills"]})


def serialize(doc: dict) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Fail if the checked-in versions.json is stale.")
    parser.add_argument("--allow-removals", action="store_true",
                        help="Permit skills to disappear from the output.")
    args = parser.parse_args()

    doc = build()
    validate(doc)

    # A generated file that stays schema-valid with one fewer entry is how a
    # skill silently vanished from metadata.json on 2026-08-03. Removals are
    # legitimate when a components.d entry is dropped, so this is a speed bump
    # rather than a wall — but it must be deliberate.
    if OUTPUT.is_file():
        gone = removed_skills(doc, json.loads(OUTPUT.read_text()))
        if gone and not args.allow_removals:
            print(f"Refusing to write versions.json: {len(gone)} skill(s) "
                  f"disappeared from the output.", file=sys.stderr)
            for name in gone:
                print(f"  - {name}", file=sys.stderr)
            print("\nIf these were deregistered in components.d, re-run with "
                  "--allow-removals.", file=sys.stderr)
            return 1

    rendered = serialize(doc)

    if args.check:
        if not OUTPUT.is_file():
            print("versions.json is missing. Regenerate it with: "
                  "python3 .github/scripts/generate_versions.py", file=sys.stderr)
            return 1
        if OUTPUT.read_text() != rendered:
            print("versions.json is out of date with skills/*/skill.oms.sig.",
                  file=sys.stderr)
            print("Regenerate it with: python3 .github/scripts/generate_versions.py",
                  file=sys.stderr)
            return 1
        print(f"versions.json is current ({len(doc['skills'])} skills).")
        return 0

    OUTPUT.write_text(rendered)
    print(f"Wrote versions.json ({len(doc['skills'])} skills).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
