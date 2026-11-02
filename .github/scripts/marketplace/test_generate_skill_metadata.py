#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure and product-declaration tests for generate-skill-metadata.py.

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
  5. A run that fails validation reports the per-skill enrichment reason, in
     the log and in the structured warnings file CI reads, alongside the
     summary line naming the skills that produced no entry.
  6. Product declarations are validated and take precedence over carried
     metadata and AI amendments, including when an enum spelling changes.

Run:
    python3 .github/scripts/marketplace/test_generate_skill_metadata.py
Exits 0 when every assertion holds, 1 otherwise.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
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


def check_product_declarations(gen, validator, taxonomy, baseline, prior, skill, r):
    """Exercise declared products without inference calls or repository writes."""
    import tempfile
    from unittest.mock import patch

    products = taxonomy["product.primary"]["values"]
    product = next(v for v in products if v != prior["metadata"]["product.primary"])
    r.check("no component -> no derived product",
            gen.derive_product_from_component(None, products) is None)
    r.check("legacy exact-name match is preserved",
            gen.derive_product_from_component({"component_name": product}, products) == product)
    r.check("legacy non-match still falls through",
            gen.derive_product_from_component({"component_name": "not-a-product"}, products) is None)
    r.check("explicit product wins over component name",
            gen.derive_product_from_component(
                {"component_name": products[0], "product_primary": product}, products) == product)

    def no_api(*args, **kwargs):
        raise AssertionError("complete entry must not call AI")

    component = {"product_primary": product, "components_file": "fixture.yml"}
    expected = dict(prior["metadata"], **{"product.primary": product})
    entry = gen.build_skill_entry(skill, {skill.path: component}, baseline, {},
                                 validator, taxonomy, no_api, [])
    r.check("declared product overrides carried metadata", entry["metadata"] == expected)
    legacy = json.loads(json.dumps(prior))
    legacy["metadata"]["product.primary"] = "removed-product-spelling"
    original_legacy = json.loads(json.dumps(legacy))
    migrated = gen.build_skill_entry(
        skill, {skill.path: component}, {"skills": [legacy]}, {},
        validator, taxonomy, no_api, [],
    )
    r.check("enum rename preserves unrelated metadata without AI", migrated["metadata"] == expected)
    r.check("enum migration does not mutate the baseline", legacy == original_legacy)
    requested = []

    def amend_api(_skill, _component, fields, _taxonomy, current_values=None):
        requested.extend(fields)
        # Include a conflicting product to verify only requested fields apply.
        return dict(prior["metadata"])

    entry = gen.build_skill_entry(skill, {skill.path: component}, baseline, {},
                                 validator, taxonomy, amend_api, [], is_materially_changed=True)
    r.check("AI amendment excludes the authoritative product", "product.primary" not in requested)
    r.check("AI cannot replace the declared product", entry["metadata"] == expected)
    requested.clear()
    entry = gen.build_skill_entry(skill, {skill.path: {"component_name": product}}, baseline,
                                 {}, validator, taxonomy, amend_api, [], is_materially_changed=True)
    r.check("AI cannot replace legacy exact-name mapping", entry["metadata"]["product.primary"] == product)

    def fill_api(_skill, _component, fields, _taxonomy):
        requested.extend(fields)
        return {f: prior["metadata"][f] for f in fields}

    requested.clear()
    entry = gen.build_skill_entry(skill, {skill.path: component}, None, {},
                                 validator, taxonomy, fill_api, [])
    r.check("new skill fill preserves the declared product", entry["metadata"] == expected)
    r.check("new skill fill requests only other fields",
            set(requested) == set(gen.MVP_FIELDS) - {"product.primary"})

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        registry = directory / "fixture.yml"
        data = {
            "name": "Display name", "repo": "NVIDIA/fixture", "product_primary": product,
            "skills": [
                {"path": ".agents/skills/one/", "catalog_dir": "one"},
                {"path": "skills/models/two/", "catalog_dir": "two"},
                {"path": "skills/three/"},
            ],
        }

        def write_registry(obj):
            registry.write_text(gen.yaml.safe_dump(obj))

        with patch.object(gen, "COMPONENTS_DIR", directory):
            write_registry(data)
            mapping = gen.load_components(products)
            r.check("declared product applies across source layouts",
                    set(mapping) == {"skills/one", "skills/two", "skills/three"}
                    and all(v["product_primary"] == product for v in mapping.values()))
            del data["product_primary"]
            write_registry(data)
            mapping = gen.load_components(products)
            r.check("legacy registry adds no product declaration",
                    all("product_primary" not in v for v in mapping.values()))

            for value in ("unknown-value", "", None, True, [product]):
                write_registry({"name": "Unused registry", "product_primary": value, "skills": []})
                try:
                    gen.load_components(products)
                except gen.GeneratorError as exc:
                    valid_error = "fixture.yml" in str(exc) and "product_primary" in str(exc)
                else:
                    valid_error = False
                r.check(f"invalid product_primary {value!r} fails even without skills", valid_error)

            # main must fail cleanly before inference, rendering, or writes.
            out, err = io.StringIO(), io.StringIO()
            with patch.object(gen, "build_ai_client", no_api), \
                 patch.object(gen, "write_if_changed", no_api), \
                 contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = gen.main(["--no-ai"])
            r.check("invalid product fails main before output writes",
                    rc == 1 and "product_primary" in err.getvalue())


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

    # --- 5. exit code: --no-ai tolerates unenriched new skills ----------
    # PR CI runs --no-ai with no inference key, so a newly synced skill always
    # arrives unenriched there. That must not fail the PR; the same condition
    # with AI available must fail. A synthetic skill is injected into discovery
    # so the outcome does not depend on whether the working tree happens to
    # contain an unenriched skill today.
    #
    # --check returns 1 for drift as well as for warnings, and a PR that edits
    # skill content legitimately drifts the checked-in metadata. Asserting on
    # the bare exit code therefore failed on unrelated PRs. Capture the output
    # and judge the warning path on its own.
    original_discover = gen.discover_skills
    ghost = gen.Skill(
        path="skills/zzz-unenriched-fixture",
        name="zzz-unenriched-fixture",
        description="fixture with no metadata and no prior entry",
        frontmatter={},
    )

    def discover_with_ghost(exclusions):
        found, excluded = original_discover(exclusions)
        return list(found) + [ghost], excluded

    original_diff = gen.diff_text

    def run_for(argv, client):
        gen.discover_skills = discover_with_ghost
        gen.build_ai_client = lambda allow_ai=True: client
        # Suppress drift reporting for the duration of the run. --check returns
        # 1 for drift as well as for warnings, and any PR that edits skill
        # content drifts the checked-in metadata legitimately. Neutralising the
        # comparison makes the exit code attributable to the warning gate alone,
        # which is what this case is about. Drift itself is covered by case 4.
        gen.diff_text = lambda *a, **k: ""
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                rc = gen.main(argv)
        finally:
            gen.discover_skills = original_discover
            gen.build_ai_client = original_client
            gen.diff_text = original_diff
        return rc, out.getvalue() + err.getvalue()

    rc_no_ai, log_no_ai = run_for(["--check", "--no-ai"], None)
    rc_with_ai, _ = run_for(["--check"], failing_api)

    r.check("unenriched new skill under --no-ai -> reported as a warning",
            "PARTIAL SUCCESS" in log_no_ai)
    r.check("unenriched new skill under --no-ai -> does not fail PR CI", rc_no_ai == 0,
            f"rc={rc_no_ai}")
    r.check("unenrichable new skill with AI available -> fails the run", rc_with_ai == 1,
            f"rc={rc_with_ai}")

    # --- 6. the failing run reports WHY, not just which skill ------------
    # The case above asserts only the exit code, and case 3 asserts only that
    # the warning is collected -- neither checks that the reason is emitted.
    # That gap let the failing path return before the reporting block, so the
    # tracking issue carried the summary line ("no entry could be built for
    # these skills") with no indication of which field failed enrichment.
    reason = "AI returned UNRESOLVED for `discovery.activity_tags`"

    def unresolved_api(*_args, **_kwargs):
        raise gen.EnrichmentError(reason)

    warnings_file = Path("/tmp/skill-warnings.json")
    if warnings_file.exists():
        warnings_file.unlink()
    _, log_unresolved = run_for(["--check"], unresolved_api)

    r.check("failing run -> per-skill reason reaches the log", reason in log_unresolved)
    r.check(
        "failing run -> summary line still present",
        "no entry could be built" in log_unresolved,
    )
    r.check("failing run -> warnings file written for the issue body", warnings_file.exists())
    r.check(
        "failing run -> warnings file carries the reason",
        warnings_file.exists()
        and any(
            reason in w
            for w in json.loads(warnings_file.read_text()).get("stale_skills", [])
        ),
    )

    check_product_declarations(gen, validator, taxonomy, baseline, prior, skill, r)

    return r.report()


if __name__ == "__main__":
    sys.exit(main())
