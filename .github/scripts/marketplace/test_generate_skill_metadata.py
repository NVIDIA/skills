#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Induced-failure tests for generate-skill-metadata.py.

The generator's failure handling is the part that cannot be exercised by a
normal run: on a healthy catalog every skill builds, so the recovery and
reporting paths never execute and a regression in them is invisible until it
drops a skill from the published catalog in production.

These tests induce enrichment failures deterministically with a stub inference
API. Nothing here reaches the network, and nothing is written to the repo --
the generator is driven in --check mode and its emitted documents are captured
in memory.

Properties under test:
  1. An optional amendment that fails on an already-complete entry keeps the
     entry with its existing values, and is NOT reported as an omission.
  2. A successful amendment is applied.
  3. A skill with no prior entry that cannot be built is omitted AND warned
     about -- a genuinely unbuildable new skill must fail the run.
  4. A published skill that cannot be rebuilt is recovered byte-identically
     from the checked-in metadata.json rather than silently delisted.

Run:
    python3 .github/scripts/marketplace/test_generate_skill_metadata.py
Exits 0 when every assertion holds, 1 otherwise.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

GENERATOR = Path(__file__).resolve().parent / "generate-skill-metadata.py"


def load_generator():
    # The generator's filename is not a valid module name, so it is loaded by
    # path rather than imported.
    spec = importlib.util.spec_from_file_location("skill_metadata_generator", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Results:
    def __init__(self) -> None:
        self.items: list[tuple[str, bool, str]] = []

    def check(self, name: str, condition, detail: str = "") -> None:
        self.items.append((name, bool(condition), detail))

    def report(self) -> int:
        ok = True
        for name, passed, detail in self.items:
            ok &= passed
            suffix = f"  {detail}" if detail else ""
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}{suffix}")
        print("\nALL PASS" if ok else "\nFAILURES PRESENT")
        return 0 if ok else 1


def pick_reference_entry(gen, baseline: dict, validator: Draft202012Validator) -> dict:
    """Return a published entry that is complete under the current schema.

    Chosen from real catalog data rather than hand-written so the tests stay
    honest about what the generator actually consumes. Selecting dynamically
    keeps them from breaking when a specific skill is renamed or retired.
    """
    for entry in baseline.get("skills", []):
        metadata = entry.get("metadata") or {}
        if any(field not in metadata for field in gen.MVP_FIELDS):
            continue
        try:
            validator.validate({"skills": [entry]})
        except Exception:
            continue
        return entry
    raise SystemExit(
        "FATAL: no complete, schema-valid entry found in metadata.json; "
        "cannot construct test fixtures."
    )


def main() -> int:
    gen = load_generator()
    schema = gen.load_json(gen.SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    taxonomy = gen.taxonomy_from_schema(schema)
    baseline = gen.load_json(gen.METADATA_PATH)

    prior = pick_reference_entry(gen, baseline, validator)
    # Name and description must match the baseline entry, otherwise the skill
    # reads as materially changed and takes the missing-fields path instead of
    # the amendment path these tests are aimed at.
    skill = gen.Skill(
        path=prior["path"],
        name=prior["name"],
        description=prior["description"],
        frontmatter={},
    )
    r = Results()

    def failing_api(*_args, **_kwargs):
        raise gen.EnrichmentError("HTTP 403")

    def stub_api(_skill, _component, missing=None, taxonomy=None, current_values=None):
        return {f: prior["metadata"][f] for f in (missing or prior["metadata"])}

    # --- 1. amendment fails on a complete entry -------------------------
    warnings: list[str] = []
    notices: list[str] = []
    entry = gen.build_skill_entry(
        skill, {}, baseline, {}, validator, taxonomy,
        failing_api, warnings, notices, is_materially_changed=True,
    )
    r.check("amendment fails -> entry kept", entry is not None)
    r.check("amendment fails -> not reported as omitted", warnings == [], f"warnings={warnings}")
    r.check("amendment fails -> recorded as retained", len(notices) == 1, f"notices={len(notices)}")
    r.check(
        "amendment fails -> existing values preserved",
        entry and entry["metadata"] == prior["metadata"],
    )

    # --- 2. amendment succeeds ------------------------------------------
    warnings, notices = [], []
    entry = gen.build_skill_entry(
        skill, {}, baseline, {}, validator, taxonomy,
        stub_api, warnings, notices, is_materially_changed=True,
    )
    r.check("amendment succeeds -> no warnings or notices", warnings == [] and notices == [])
    r.check(
        "amendment succeeds -> entry intact",
        entry and entry["metadata"] == prior["metadata"],
    )

    # --- 3. new skill, no prior entry, cannot be built -------------------
    new_skill = gen.Skill(
        path="skills/zzz-not-a-real-skill",
        name="zzz-not-a-real-skill",
        description="fixture for the unbuildable-new-skill path",
        frontmatter={},
    )
    warnings, notices = [], []
    entry = gen.build_skill_entry(
        new_skill, {}, baseline, {}, validator, taxonomy,
        failing_api, warnings, notices,
    )
    r.check("unbuildable new skill -> omitted", entry is None)
    r.check("unbuildable new skill -> warned", len(warnings) == 1, f"warnings={warnings}")

    # --- 4. published skill unbuildable, end to end through main() -------
    victim = prior["path"]
    original_build = gen.build_skill_entry
    original_client = gen.build_ai_client
    original_dumps = gen.dumps_canonical
    emitted: list[dict] = []

    def drop_victim(sk, *args, **kwargs):
        return None if sk.path == victim else original_build(sk, *args, **kwargs)

    def stub_client(allow_ai=True):
        # Every other skill enriches deterministically, so any skill that is
        # legitimately new in the working tree cannot confound the victim.
        return stub_api

    def capture(obj):
        emitted.append(obj)
        return original_dumps(obj)

    gen.build_skill_entry = drop_victim
    gen.build_ai_client = stub_client
    gen.dumps_canonical = capture
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            gen.main(["--check"])
    finally:
        gen.build_skill_entry = original_build
        gen.build_ai_client = original_client
        gen.dumps_canonical = original_dumps

    log = stderr.getvalue()
    emitted_entries = {e["path"]: e for e in emitted[0]["skills"]} if emitted else {}
    r.check(
        "unbuildable published skill -> recovered byte-identically",
        emitted_entries.get(victim) == prior,
        "" if emitted_entries.get(victim) == prior else f"present={victim in emitted_entries}",
    )
    r.check("unbuildable published skill -> not silently delisted", victim in emitted_entries)
    r.check(
        "unbuildable published skill -> recovery surfaced",
        "kept the entry from the previous metadata.json" in log,
    )
    # Drift from unrelated skills is not this test's concern; the victim
    # specifically must not be a source of it.
    r.check(
        "unbuildable published skill -> victim causes no drift",
        victim not in log.split("DRIFT DETECTED")[-1] if "DRIFT DETECTED" in log else True,
    )

    return r.report()


if __name__ == "__main__":
    sys.exit(main())
