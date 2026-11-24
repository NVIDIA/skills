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


def _expected_metadata(gen):
    """Return the metadata.json a healthy run produces, as written text.

    Tests that drive main() in --check mode need a baseline that already
    agrees with the current tree, or the run fails on drift before it reaches
    the behaviour under test. Reading the checked-in file supplied one only
    while that file happened to carry a `plugins` block, which made the tests
    pass or fail on repo state. Generating it keeps them self-contained.

    The skills come from the repo, because --no-ai can only carry a skill
    forward from a baseline that already holds it. The `plugins` block does
    not: it is seeded empty and rebuilt by the run.
    """
    import tempfile
    from unittest.mock import patch

    catalog = json.loads(gen.METADATA_PATH.read_text(encoding="utf-8"))
    captured = {}

    def capture(path, content):
        captured[path] = content
        return False

    with tempfile.TemporaryDirectory() as tmp:
        seed = Path(tmp) / "metadata.json"
        seed.write_text(
            gen.dumps_canonical(dict(catalog, plugins=[])), encoding="utf-8"
        )
        buf = io.StringIO()
        with patch.object(gen, "METADATA_PATH", seed), \
             patch.object(gen, "write_if_changed", capture), \
             contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = gen.main(["--no-ai"])
        if rc != 0 or seed not in captured:
            raise AssertionError(
                "could not build a baseline for the --check tests: "
                f"rc={rc}; {buf.getvalue()[-300:]}"
            )
        return captured[seed]


def check_plugin_never_blocks_main(gen, r):
    """Drive main() with a hostile plugins.d and prove output still happens.

    This is the executable form of the invariant the rest of the plugin code
    only asserts in prose: a plugin problem degrades that plugin and never the
    catalog. It is tested through main() rather than through the helpers
    because the ways it was broken before -- an entry that passed the
    vocabulary check and failed whole-document validation, and a TypeError out
    of a sort -- were invisible at the helper level and only showed up as a
    non-zero exit and an unwritten file.

    main() is driven in --check mode, so nothing in the repository is written.
    A non-zero return here means the catalog would not have been regenerated.
    """
    import shutil
    import tempfile
    from unittest.mock import patch

    real_plugins_d = gen.PLUGINS_D_DIR

    # Each fixture must reach the code path it is named for, so every one
    # carries a resolvable include and a complete taxonomy unless the defect
    # under test is precisely a missing one. An earlier version of this test
    # used fixtures that exited before the path they claimed to exercise.
    real_skill = "skills/nvidia-skill-finder/"
    full_taxonomy = (
        "  classification.category.primary: developer_tools\n"
        "  product.primary:\n    - Learning\n"
        "  audience: developer\n"
    )
    hostile = {
        "empty product.primary":
            f"name: p\ndescription: d\ninclude_skills:\n  - {real_skill}\n"
            f"taxonomy:\n  product.primary: []\n",
        "empty audience":
            f"name: p\ndescription: d\ninclude_skills:\n  - {real_skill}\n"
            f"taxonomy:\n  audience: ''\n",
        "mixed-type taxonomy keys":
            f"name: p\ndescription: d\ninclude_skills:\n  - {real_skill}\n"
            f"taxonomy:\n  ? ~\n  : x\n  typo: y\n",
        "surrogate in description":
            f"name: p\ndescription: \"\\uD800\"\ninclude_skills:\n  - {real_skill}\n",
        "YAML value raising a builtin": "name: p\nrelease_date: 2026-02-30\n",
        "include_skills as a string": "name: p\ndescription: d\ninclude_skills: skills/nope/\n",
        "unparseable YAML": "name: p\n  bad indent:\n",
        "non-mapping root": "- a\n- b\n",
        "unresolvable include":
            "name: p\ndescription: d\ninclude_skills:\n  - skills/does-not-exist/\n",
        "bad kebab-case name": "name: Not_Kebab\ndescription: d\n",
        "directory-form include":
            f"name: p\ndescription: d\ninclude_skills:\n  - skills/\ntaxonomy:\n{full_taxonomy}",
    }

    # --check fails on any drift, so it needs a baseline that already matches a
    # healthy run. Taking the checked-in metadata.json made every case below
    # depend on repo state: on a tree whose metadata.json has no `plugins`
    # block yet -- main before this change -- all of them failed on drift
    # rather than on the defect they name. Build the baseline here instead.
    expected = _expected_metadata(gen)

    for label, body in hostile.items():
        with tempfile.TemporaryDirectory() as tmp:
            # The real plugins.d is copied in first so the genuine plugins
            # still build. None of the hostile inputs below produce an entry,
            # so a healthy run stays byte-identical to the baseline above.
            directory = Path(tmp) / "plugins.d"
            shutil.copytree(real_plugins_d, directory)
            (directory / "p.yml").write_text(body, encoding="utf-8")
            meta = Path(tmp) / "metadata.json"
            meta.write_text(expected, encoding="utf-8")
            buf = io.StringIO()
            try:
                with patch.object(gen, "PLUGINS_D_DIR", directory), \
                     patch.object(gen, "METADATA_PATH", meta), \
                     contextlib.redirect_stdout(buf), \
                     contextlib.redirect_stderr(buf):
                    rc = gen.main(["--check", "--no-ai"])
                raised = None
            except Exception as exc:  # noqa: BLE001
                rc, raised = None, exc
        log = buf.getvalue()
        # Three conditions, because the first two alone passed when
        # build_plugin_entries was stubbed out to return nothing: recovery
        # restored the genuine plugin and the run looked healthy. The third
        # requires evidence that the hostile file was actually seen.
        ok = raised is None and rc == 0 and ("p.yml" in log or "p`" in log)
        detail = ""
        if not ok:
            detail = (
                f"raised={raised!r}" if raised
                else f"rc={rc}; hostile file mentioned={'p.yml' in log}; {log[-300:]}"
            )
        r.check(f"main() survives: {label}", ok, detail)

    # The guard above must fail if the plugin pass stops doing its job. This
    # asserts the test itself has teeth, which the previous version did not.
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "plugins.d"
        shutil.copytree(real_plugins_d, directory)
        (directory / "p.yml").write_text("- not a mapping\n", encoding="utf-8")
        meta = Path(tmp) / "metadata.json"
        meta.write_text(expected, encoding="utf-8")
        buf = io.StringIO()
        with patch.object(gen, "PLUGINS_D_DIR", directory), \
             patch.object(gen, "METADATA_PATH", meta), \
             patch.object(gen, "build_plugin_entries", lambda *a, **k: []), \
             contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            gen.main(["--check", "--no-ai"])
        r.check("the hostile-input guard detects a disabled plugin pass",
                "p.yml" not in buf.getvalue())


def check_mirrored_plugins_through_main(gen, r):
    """Drive main() against a mirrored plugin that does not exist in the repo.

    There is no mirrored plugin in NVIDIA/skills yet -- sync-plugins.yml has
    not landed -- so every scenario here is built as a fixture: a components.d
    registry with a `plugins:` entry, and a plugins/<name>/ tree holding a
    plugin.json and copies of catalog skills.

    The copies are deliberately given directory names that do NOT match the
    catalog, because that is the real shape: `catalog_dir` renames a skill on
    the way into the catalog, so a product's own directory name and the
    catalog's differ for 43 of the 484 skills today. A resolver that matched
    on directory name would pass a test built from well-behaved names and fail
    in production, so the fixture never offers it that option.
    """
    import shutil
    import tempfile
    from unittest.mock import patch

    catalog = json.loads(gen.METADATA_PATH.read_text(encoding="utf-8"))
    # Two real catalog skills that share a domain, so the roll-up has a
    # dominant value to find rather than a tie to break.
    pool = [s for s in catalog["skills"]
            if s["metadata"].get("classification.category.primary")]
    by_domain = {}
    for s in pool:
        by_domain.setdefault(
            s["metadata"]["classification.category.primary"], []).append(s)
    sample = max(by_domain.values(), key=len)[:2]
    if len(sample) < 2:
        r.check("mirrored: two catalog skills available to bundle", False)
        return

    def build(tmp, *, dir_names, registry_skills, plugin_json=None,
              catalog_dir="nvidia-tao", extra_registry=""):
        """Lay out a fixture components.d + plugins/ tree."""
        comp = Path(tmp) / "components.d"
        comp.mkdir()
        skills_yaml = "".join(
            f"  - path: {p}\n    catalog_dir: {c}\n"
            for p, c in registry_skills
        )
        comp.joinpath("tao.yml").write_text(
            "name: TAO Toolkit\n"
            "repo: NVIDIA-TAO/tao-skill-bank\n"
            "description: d\n"
            + ("skills:\n" + skills_yaml if skills_yaml else "")
            + "plugins:\n"
            + f"  - path: plugins/{catalog_dir}/\n"
            + f"    catalog_dir: {catalog_dir}\n"
            + extra_registry,
            encoding="utf-8",
        )
        plugins = Path(tmp) / "plugins"
        pdir = plugins / catalog_dir
        (pdir / "skills").mkdir(parents=True)
        manifest = {"name": catalog_dir, "version": "1.0.0",
                    "description": "Mirrored TAO plugin.",
                    "skills": ["./skills/"]}
        if plugin_json is not None:
            manifest = plugin_json
        if manifest is not False:
            pdir.joinpath("plugin.json").write_text(
                json.dumps(manifest), encoding="utf-8")
        for dirname, skill in zip(dir_names, sample):
            d = pdir / "skills" / dirname
            d.mkdir()
            d.joinpath("SKILL.md").write_text(
                f"---\nname: {skill['name']}\n"
                f"description: {skill['description']}\n---\n\nBody.\n",
                encoding="utf-8",
            )
        return comp, plugins

    def run(tmp, comp, plugins, baseline=None):
        meta = Path(tmp) / "metadata.json"
        meta.write_text(
            gen.dumps_canonical(baseline if baseline is not None
                                else dict(catalog, plugins=[])),
            encoding="utf-8")
        captured = {}

        def capture(path, content):
            if path == meta:
                captured["out"] = json.loads(content)
            return False

        buf = io.StringIO()
        with patch.object(gen, "COMPONENTS_DIR", comp), \
             patch.object(gen, "PLUGINS_DIR", plugins), \
             patch.object(gen, "METADATA_PATH", meta), \
             patch.object(gen, "write_if_changed", capture), \
             contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = gen.main(["--no-ai"])
        return rc, captured.get("out"), buf.getvalue()

    def plugins_of(doc):
        return {p["name"]: p for p in (doc or {}).get("plugins", [])}

    # --- The main case: bundled dirs named as the PRODUCT names them. -------
    # The catalog publishes these two skills under different directory names,
    # so only SKILL.md-name matching can resolve them.
    product_dirs = ["bundled-one", "bundled-two"]
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = build(tmp, dir_names=product_dirs, registry_skills=[])
        rc, out, log = run(tmp, comp, plugins)
        got = plugins_of(out)
        entry = got.get("nvidia-tao")
        r.check("mirrored: a mirrored plugin gets an entry",
                rc == 0 and entry is not None, f"rc={rc}; {list(got)}")
        r.check("mirrored: bundled skills resolve by SKILL.md name, not folder",
                entry is not None
                and entry["skills"] == sorted(s["path"] for s in sample),
                f"{entry and entry['skills']}")
        r.check("mirrored: taxonomy is rolled up from them",
                entry is not None
                and entry["metadata"]["classification.category.primary"]
                == sample[0]["metadata"]["classification.category.primary"],
                f"{entry and entry['metadata']}")
        r.check("mirrored: source points at the registry, not plugins.d",
                entry is not None
                and entry["source"] == "components.d/tao.yml",
                f"{entry and entry.get('source')}")
        r.check("mirrored: description comes from plugin.json",
                entry is not None
                and entry["description"] == "Mirrored TAO plugin.",
                f"{entry and entry.get('description')}")

    # --- The second key: SKILL.md unreadable, registry path resolves it. ----
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = build(
            tmp, dir_names=product_dirs,
            registry_skills=[(f"skills/{d}", s["path"].split("/")[-1])
                             for d, s in zip(product_dirs, sample)])
        # Blank the frontmatter name so key 1 cannot fire.
        for d in product_dirs:
            p = plugins / "nvidia-tao" / "skills" / d / "SKILL.md"
            p.write_text("---\ndescription: d\n---\n\nBody.\n", encoding="utf-8")
        rc, out, log = run(tmp, comp, plugins)
        entry = plugins_of(out).get("nvidia-tao")
        r.check("mirrored: falls back to the registry source path",
                entry is not None
                and entry["skills"] == sorted(s["path"] for s in sample),
                f"rc={rc}; {entry and entry.get('skills')}")

    # --- Precedence: plugins.d wins, the registry entry is ignored. ---------
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = build(tmp, dir_names=product_dirs, registry_skills=[],
                              catalog_dir="nvidia-skills")
        rc, out, log = run(tmp, comp, plugins)
        entry = plugins_of(out).get("nvidia-skills")
        r.check("mirrored: a plugins.d plugin is never claimed as mirrored",
                entry is not None and entry["source"].startswith("plugins.d/"),
                f"rc={rc}; {entry and entry.get('source')}")

    # --- Registered but not mirrored yet: warned, not an entry, not fatal. --
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = build(tmp, dir_names=product_dirs, registry_skills=[])
        shutil.rmtree(plugins / "nvidia-tao")
        rc, out, log = run(tmp, comp, plugins)
        r.check("mirrored: registered but unmirrored is reported, not fatal",
                rc == 0 and "nvidia-tao" not in plugins_of(out)
                and "not in the repository yet" in log, f"rc={rc}")

    # --- Recovery: a registered plugin that fails to build is KEPT. ---------
    # This is Moshe's point. Without the registry entry counting as declared,
    # a mirrored plugin whose skills stop resolving for one run is delisted.
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = build(tmp, dir_names=product_dirs, registry_skills=[])
        rc, out, _ = run(tmp, comp, plugins)
        published = dict(catalog, plugins=(out or {}).get("plugins", []))
        # Now break it: remove the manifest so it cannot be rebuilt.
        (plugins / "nvidia-tao" / "plugin.json").unlink()
        rc, out, log = run(tmp, comp, plugins, baseline=published)
        r.check("mirrored: a registered plugin that fails to build is KEPT",
                rc == 0 and "nvidia-tao" in plugins_of(out) and "KEPT." in log,
                f"rc={rc}; {list(plugins_of(out))}")

    # --- Deregistered: removed from the registry, correctly delisted. -------
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = build(tmp, dir_names=product_dirs, registry_skills=[])
        rc, out, _ = run(tmp, comp, plugins)
        published = dict(catalog, plugins=(out or {}).get("plugins", []))
        comp.joinpath("tao.yml").write_text(
            "name: TAO Toolkit\nrepo: r\ndescription: d\n", encoding="utf-8")
        rc, out, log = run(tmp, comp, plugins, baseline=published)
        r.check("mirrored: a deregistered plugin is delisted",
                rc == 0 and "nvidia-tao" not in plugins_of(out),
                f"rc={rc}; {list(plugins_of(out))}")

    # --- Hostile registries must never stop the catalog. --------------------
    hostile = {
        "catalog_dir escaping the tree": "plugins:\n  - catalog_dir: ../evil\n",
        "catalog_dir not kebab-case": "plugins:\n  - catalog_dir: Not_Kebab\n",
        "catalog_dir missing": "plugins:\n  - path: plugins/x/\n",
        "plugins: not a list": "plugins: nope\n",
        "plugins entry not a mapping": "plugins:\n  - just-a-string\n",
    }
    for label, body in hostile.items():
        with tempfile.TemporaryDirectory() as tmp:
            comp, plugins = build(tmp, dir_names=product_dirs,
                                  registry_skills=[])
            comp.joinpath("hostile.yml").write_text(body, encoding="utf-8")
            rc, out, log = run(tmp, comp, plugins)
            r.check(f"mirrored: catalog survives {label}",
                    rc == 0 and out is not None
                    and len(out.get("skills", [])) == len(catalog["skills"]),
                    f"rc={rc}; {log[-200:]}")

    # An unparseable components.d file is deliberately NOT in the list above.
    # The skills pass reads the same directory and treats a malformed registry
    # as fatal by existing design -- "invalid declarations fail before any
    # output is written" -- and it runs first, so the generator stops there.
    # Asserted so nobody later reads the plugin pass's own guard as a promise
    # that a broken registry degrades gracefully. It does not, and changing
    # that is a decision about the skills pass, not this one.
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = build(tmp, dir_names=product_dirs, registry_skills=[])
        comp.joinpath("hostile.yml").write_text(
            "name: x\n  bad indent:\n", encoding="utf-8")
        try:
            rc, out, log = run(tmp, comp, plugins)
            raised = None
        except Exception as exc:  # noqa: BLE001
            raised = exc
        r.check("mirrored: an unparseable registry still fails loudly",
                raised is not None, f"raised={raised!r}")

    # Teeth: the guard above must notice if the mirrored pass stops working.
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = build(tmp, dir_names=product_dirs, registry_skills=[])
        with patch.object(gen, "build_mirrored_plugin_entries",
                          lambda *a, **k: []):
            rc, out, log = run(tmp, comp, plugins)
        r.check("mirrored: the fixture detects a disabled mirrored pass",
                "nvidia-tao" not in plugins_of(out), f"{list(plugins_of(out))}")


def check_mirrored_plugin_edge_cases(gen, r):
    """Cases a review found the first version of the mirrored pass got wrong.

    Each one is written so that reverting its fix makes it fail. Several
    correspond to mutations that survived the first version of this suite.
    """
    import tempfile
    from unittest.mock import patch

    catalog = json.loads(gen.METADATA_PATH.read_text(encoding="utf-8"))
    by_domain = {}
    for s in catalog["skills"]:
        d = s["metadata"].get("classification.category.primary")
        if d:
            by_domain.setdefault(d, []).append(s)
    # Two skills from DIFFERENT domains, chosen so the alphabetically first
    # domain is not the one that would win a rigged plurality. Used by the
    # de-duplication case below.
    domains = sorted(d for d, v in by_domain.items() if len(v) >= 2)
    if len(domains) < 2:
        r.check("edge: two domains available", False)
        return
    dom_lo, dom_hi = domains[0], domains[-1]
    skill_hi = by_domain[dom_hi][0]
    skill_lo = by_domain[dom_lo][0]

    def md(skill):
        return (f"---\nname: {skill['name']}\n"
                f"description: {skill['description']}\n---\n\nBody.\n")

    def layout(tmp, registries, trees):
        """registries: {filename: text}. trees: {plugin: {dirname: md|None}}."""
        comp = Path(tmp) / "components.d"
        comp.mkdir(exist_ok=True)
        for fname, text in registries.items():
            comp.joinpath(fname).write_text(text, encoding="utf-8")
        plugins = Path(tmp) / "plugins"
        plugins.mkdir(exist_ok=True)
        for pname, dirs in trees.items():
            pdir = plugins / pname
            (pdir / "skills").mkdir(parents=True, exist_ok=True)
            pdir.joinpath("plugin.json").write_text(
                json.dumps({"name": pname, "version": "1.0.0",
                            "description": f"{pname} plugin."}),
                encoding="utf-8")
            for dirname, text in dirs.items():
                d = pdir / "skills" / dirname
                d.mkdir(exist_ok=True)
                if text is not None:
                    d.joinpath("SKILL.md").write_text(text, encoding="utf-8")
        return comp, plugins

    def run(tmp, comp, plugins, baseline=None, plugins_d=None):
        meta = Path(tmp) / "metadata.json"
        meta.write_text(
            gen.dumps_canonical(baseline if baseline is not None
                                else dict(catalog, plugins=[])),
            encoding="utf-8")
        captured = {}

        def capture(path, content):
            if path == meta:
                captured["out"] = json.loads(content)
            return False

        patches = [
            patch.object(gen, "COMPONENTS_DIR", comp),
            patch.object(gen, "PLUGINS_DIR", plugins),
            patch.object(gen, "METADATA_PATH", meta),
            patch.object(gen, "write_if_changed", capture),
        ]
        if plugins_d is not None:
            patches.append(patch.object(gen, "PLUGINS_D_DIR", plugins_d))
        buf = io.StringIO()
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with contextlib.redirect_stdout(buf), \
                 contextlib.redirect_stderr(buf):
                rc = gen.main(["--no-ai"])
        return rc, captured.get("out"), buf.getvalue()

    def names_of(doc):
        return [p["name"] for p in (doc or {}).get("plugins", [])]

    def reg(entries, skills=""):
        return ("name: R\nrepo: r\ndescription: d\n"
                + (f"skills:\n{skills}" if skills else "")
                + "plugins:\n"
                + "".join(f"  - path: plugins/{e}/\n    catalog_dir: {e}\n"
                          for e in entries))

    # --- F1a: removing ONE plugin from a shared registry delists it. -------
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = layout(
            tmp, {"r.yml": reg(["keep-me", "retire-me"])},
            {"keep-me": {"a": md(skill_hi)}, "retire-me": {"a": md(skill_hi)}})
        rc, out, _ = run(tmp, comp, plugins)
        published = dict(catalog, plugins=(out or {}).get("plugins", []))
        r.check("edge: both plugins publish first",
                sorted(names_of(out)) == ["keep-me", "nvidia-skills",
                                          "retire-me"], f"{names_of(out)}")
        comp.joinpath("r.yml").write_text(reg(["keep-me"]), encoding="utf-8")
        rc, out, log = run(tmp, comp, plugins, baseline=published)
        r.check("edge: a plugin removed from a shared registry is delisted",
                rc == 0 and "retire-me" not in names_of(out)
                and "keep-me" in names_of(out), f"rc={rc}; {names_of(out)}")

    # --- F1b: an unreadable `plugins:` keeps what it published. ------------
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = layout(tmp, {"r.yml": reg(["keep-me"])},
                               {"keep-me": {"a": md(skill_hi)}})
        rc, out, _ = run(tmp, comp, plugins)
        published = dict(catalog, plugins=(out or {}).get("plugins", []))
        comp.joinpath("r.yml").write_text(
            "name: R\nrepo: r\ndescription: d\nplugins: nope\n",
            encoding="utf-8")
        rc, out, log = run(tmp, comp, plugins, baseline=published)
        r.check("edge: an unreadable `plugins:` keeps its published plugins",
                rc == 0 and "keep-me" in names_of(out) and "KEPT." in log,
                f"rc={rc}; {names_of(out)}")

    # --- F2a: one plugin declared twice emits once, with a warning. --------
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = layout(tmp, {"r.yml": reg(["dup", "dup"])},
                               {"dup": {"a": md(skill_hi)}})
        rc, out, log = run(tmp, comp, plugins)
        r.check("edge: a plugin declared twice in one registry emits once",
                rc == 0 and names_of(out).count("dup") == 1
                and "already declared" in log, f"rc={rc}; {names_of(out)}")

    # --- M1: two registries claiming one name. -----------------------------
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = layout(
            tmp, {"a.yml": reg(["shared"]), "b.yml": reg(["shared"])},
            {"shared": {"a": md(skill_hi)}})
        rc, out, log = run(tmp, comp, plugins)
        r.check("edge: two registries cannot both claim one plugin",
                rc == 0 and names_of(out).count("shared") == 1
                and "already declared" in log, f"rc={rc}; {names_of(out)}")

    # --- F2b: precedence keyed on the DECLARED name, not the filename. -----
    # plugins.d/bundle.yml declaring `name: owned` owns plugins/owned/ even
    # though no plugins.d/owned.yml exists. A filename check misses it.
    with tempfile.TemporaryDirectory() as tmp:
        pd = Path(tmp) / "plugins.d"
        pd.mkdir()
        pd.joinpath("bundle.yml").write_text(
            "name: owned\ndescription: Built by the catalog.\n"
            f"include_skills:\n  - {skill_hi['path']}/\n", encoding="utf-8")
        comp, plugins = layout(tmp, {"r.yml": reg(["owned"])},
                               {"owned": {"a": md(skill_lo)}})
        rc, out, log = run(tmp, comp, plugins, plugins_d=pd)
        entries = [p for p in (out or {}).get("plugins", [])
                   if p["name"] == "owned"]
        r.check("edge: a plugins.d name owns its dir whatever the filename",
                rc == 0 and len(entries) == 1
                and entries[0]["source"] == "plugins.d/bundle.yml",
                f"rc={rc}; {[e['source'] for e in entries]}")

    # --- F3: the two keys resolving to different skills is reported. -------
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = layout(
            tmp,
            {"r.yml": reg(["conflict"],
                          skills=f"  - path: skills/a/\n"
                                 f"    catalog_dir: "
                                 f"{skill_lo['path'].split('/')[-1]}\n")},
            {"conflict": {"a": md(skill_hi)}})
        rc, out, log = run(tmp, comp, plugins)
        entry = next((p for p in (out or {}).get("plugins", [])
                      if p["name"] == "conflict"), None)
        r.check("edge: conflicting skill identities are reported",
                rc == 0 and "identify DIFFERENT catalog skills" in log,
                f"rc={rc}")
        r.check("edge: the SKILL.md name wins the conflict",
                entry is not None and entry["skills"] == [skill_hi["path"]],
                f"{entry and entry.get('skills')}")

    # --- F4: a long catalog_dir must not raise out of the generator. -------
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = layout(tmp, {"r.yml": reg(["ok"])},
                               {"ok": {"a": md(skill_hi)}})
        # Longer than the 255-byte filename limit. 252 would not do: the
        # precedence check is a set lookup now, so the only remaining
        # filesystem call on this name is plugins/<name>, which needs to
        # exceed the limit on its own to raise.
        comp.joinpath("long.yml").write_text(
            "name: R\nrepo: r\ndescription: d\nplugins:\n"
            f"  - catalog_dir: {'a' * 300}\n", encoding="utf-8")
        try:
            rc, out, log = run(tmp, comp, plugins)
            raised = None
        except Exception as exc:  # noqa: BLE001
            rc, out, log, raised = None, None, "", exc
        r.check("edge: an over-long catalog_dir does not stop the catalog",
                raised is None and rc == 0 and "ok" in names_of(out),
                f"raised={raised!r}; rc={rc}")

    # --- M7: an invalid name is refused even when its tree exists. ---------
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = layout(tmp, {"r.yml": reg(["ok"])},
                               {"ok": {"a": md(skill_hi)},
                                "Not_Kebab": {"a": md(skill_hi)}})
        comp.joinpath("bad.yml").write_text(
            "name: R\nrepo: r\ndescription: d\nplugins:\n"
            "  - catalog_dir: Not_Kebab\n", encoding="utf-8")
        rc, out, log = run(tmp, comp, plugins)
        r.check("edge: an invalid name is refused even with a tree present",
                rc == 0 and "Not_Kebab" not in names_of(out)
                and "not lowercase kebab-case" in log,
                f"rc={rc}; {names_of(out)}")

    # --- M2: a description that cannot be encoded as UTF-8. ----------------
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = layout(tmp, {"r.yml": reg(["bad-desc", "ok"])},
                               {"bad-desc": {"a": md(skill_hi)},
                                "ok": {"a": md(skill_hi)}})
        (plugins / "bad-desc" / "plugin.json").write_text(
            '{"name":"bad-desc","description":"\\ud800"}', encoding="utf-8")
        rc, out, log = run(tmp, comp, plugins)
        r.check("edge: an unencodable description skips only that plugin",
                rc == 0 and "bad-desc" not in names_of(out)
                and "ok" in names_of(out), f"rc={rc}; {names_of(out)}")

    # --- M3: a malformed plugin.json must not propagate. -------------------
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = layout(tmp, {"r.yml": reg(["broken", "ok"])},
                               {"broken": {"a": md(skill_hi)},
                                "ok": {"a": md(skill_hi)}})
        (plugins / "broken" / "plugin.json").write_text(
            "{not json at all", encoding="utf-8")
        rc, out, log = run(tmp, comp, plugins)
        r.check("edge: a malformed plugin.json skips only that plugin",
                rc == 0 and "broken" not in names_of(out)
                and "ok" in names_of(out), f"rc={rc}; {names_of(out)}")

    # --- M5: a failed mirrored plugin must not stop the ones after it. -----
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = layout(tmp, {"r.yml": reg(["aaa-fails", "zzz-ok"])},
                               {"aaa-fails": {"a": md(skill_hi)},
                                "zzz-ok": {"a": md(skill_hi)}})
        (plugins / "aaa-fails" / "plugin.json").unlink()
        rc, out, log = run(tmp, comp, plugins)
        r.check("edge: a failed mirrored plugin does not stop later ones",
                rc == 0 and "zzz-ok" in names_of(out)
                and "aaa-fails" not in names_of(out),
                f"rc={rc}; {names_of(out)}")

    # --- M5b: an unmirrored plugin must not stop the ones after it. --------
    # Separate from the case above: this one leaves the loop through the
    # "registered but not mirrored" branch, not through a failed assembly.
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = layout(tmp, {"r.yml": reg(["aaa-absent", "zzz-ok"])},
                               {"zzz-ok": {"a": md(skill_hi)}})
        rc, out, log = run(tmp, comp, plugins)
        r.check("edge: an unmirrored plugin does not stop later ones",
                rc == 0 and "zzz-ok" in names_of(out)
                and "aaa-absent" not in names_of(out),
                f"rc={rc}; {names_of(out)}")

    # --- M5c: a plugin that fails AT ASSEMBLY must not stop later ones. ----
    # Distinct from both cases above. The bad-description and malformed-
    # manifest plugins are skipped earlier in the loop and never reach
    # assemble_plugin_entry, so neither covers its failure return. A plugin
    # whose bundled skills resolve to nothing does reach it and is refused
    # there.
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = layout(
            tmp, {"r.yml": reg(["aaa-unresolvable", "zzz-ok"])},
            {"aaa-unresolvable": {"x": md({"name": "no-such-skill-at-all",
                                           "description": "d"})},
             "zzz-ok": {"a": md(skill_hi)}})
        rc, out, log = run(tmp, comp, plugins)
        r.check("edge: assembly failure does not stop later mirrored plugins",
                rc == 0 and "zzz-ok" in names_of(out)
                and "aaa-unresolvable" not in names_of(out),
                f"rc={rc}; {names_of(out)}")

    # --- F1b (direct): the declared set on unreadable declarations. --------
    # Driven against load_mirrored_plugin_specs rather than main(), because
    # an unparseable registry file never reaches this function in a real run:
    # the skills pass reads the same directory first and fails hard on it.
    # Without a direct test that branch has no coverage at all.
    with tempfile.TemporaryDirectory() as tmp:
        comp = Path(tmp) / "components.d"
        comp.mkdir()
        cases = {
            "unparseable.yml": "name: x\n  bad indent:\n",
            "scalar.yml": "name: R\nplugins: nope\n",
            "notmapping.yml": "name: R\nplugins:\n  - just-a-string\n",
            "nocatalogdir.yml": "name: R\nplugins:\n  - path: plugins/x/\n",
        }
        for fname, text in cases.items():
            comp.joinpath(fname).write_text(text, encoding="utf-8")
        warns, decl = [], set()
        with patch.object(gen, "COMPONENTS_DIR", comp):
            gen.load_mirrored_plugin_specs(warns, set(), decl)
        missing = [f"components.d/{f}" for f in cases
                   if f"components.d/{f}" not in decl]
        r.check("edge: every unreadable declaration declares its file",
                not missing, f"not declared: {missing}")
        r.check("edge: an unreadable declaration declares no plugin name",
                all("/" in d for d in decl), f"{sorted(decl)}")

    # --- F1c (direct): a refused name is deliberately NOT declared. --------
    # The sync refuses it, so no such plugin is published and recovery must
    # not keep an entry describing one.
    with tempfile.TemporaryDirectory() as tmp:
        comp = Path(tmp) / "components.d"
        comp.mkdir()
        comp.joinpath("r.yml").write_text(
            "name: R\nplugins:\n  - catalog_dir: Not_Kebab\n", encoding="utf-8")
        warns, decl = [], set()
        with patch.object(gen, "COMPONENTS_DIR", comp):
            gen.load_mirrored_plugin_specs(warns, set(), decl)
        r.check("edge: a refused name is not declared for recovery",
                decl == set(), f"{sorted(decl)}")

    # --- M6: de-duplication before the roll-up changes the answer. ---------
    # Two directories resolve to the SAME skill. Without de-duplication that
    # skill votes twice and its domain wins the plurality; with it, the two
    # domains tie and the alphabetically first wins instead.
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = layout(tmp, {"r.yml": reg(["votes"])},
                               {"votes": {"one": md(skill_hi),
                                          "two": md(skill_hi),
                                          "three": md(skill_lo)}})
        rc, out, log = run(tmp, comp, plugins)
        entry = next((p for p in (out or {}).get("plugins", [])
                      if p["name"] == "votes"), None)
        r.check("edge: a skill bundled twice votes once in the roll-up",
                entry is not None
                and entry["metadata"]["classification.category.primary"]
                == dom_lo,
                f"got {entry and entry['metadata'].get('classification.category.primary')}; "
                f"expected {dom_lo}")

    # --- F5: the no-skills message is one line and offers a real remedy. ---
    with tempfile.TemporaryDirectory() as tmp:
        comp, plugins = layout(tmp, {"r.yml": reg(["empty"])},
                               {"empty": {"nothing-here": md(
                                   {"name": "no-such-skill-anywhere",
                                    "description": "d"})}})
        rc, out, log = run(tmp, comp, plugins)
        line = next((l for l in log.splitlines()
                     if "bundles no catalog skill" in l), "")
        r.check("edge: the no-skills warning is a single line",
                bool(line) and "plugins/empty/skills/" in line, line[:120])
        r.check("edge: the no-skills warning offers no taxonomy: remedy",
                bool(line) and "taxonomy:" not in line, line[:120])


def check_recovery_through_main(gen, r):
    """Prove through main() that a published plugin survives a failed rebuild.

    This is the launch-day failure mode. A plugin's taxonomy is derived from
    skills it does not own, so a skill pulled for compliance, renamed upstream,
    or added unenriched under --no-ai can take a compliant plugin down with it.
    `nvidia-skills` bundles a single skill, so one absent skill would otherwise
    remove the flagship plugin from build.nvidia.com.

    It is driven through main() on purpose. Mutation testing showed the helper
    tests could not see this: they hand recover_plugin_entries a declared set
    directly, so breaking the wiring between discovery and recovery in main()
    left every assertion passing.

    Nothing is written -- main() runs in --check mode, and the baseline it
    reads is a temporary copy.
    """
    import shutil
    import tempfile
    from unittest.mock import patch

    real_plugins_d = gen.PLUGINS_D_DIR
    # The catalog's skills come from the repo because --no-ai can only carry a
    # skill forward from a baseline that already holds it. The `plugins` block
    # deliberately does not: this suite has to pass on a tree that has no such
    # block yet, which is main before this change. A healthy run below supplies
    # it instead, so the test depends on the generator, not on repo state.
    catalog = json.loads(gen.METADATA_PATH.read_text(encoding="utf-8"))
    seed = dict(catalog, plugins=[])

    def run(mutate_plugins_d, baseline_doc):
        """Run main() against a temp plugins.d and a temp baseline copy."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "plugins.d"
            shutil.copytree(real_plugins_d, directory)
            mutate_plugins_d(directory)
            meta = Path(tmp) / "metadata.json"
            meta.write_text(gen.dumps_canonical(baseline_doc), encoding="utf-8")
            captured = {}

            def capture(path, content):
                if path == meta:
                    captured["out"] = json.loads(content)
                return False

            buf = io.StringIO()
            with patch.object(gen, "PLUGINS_D_DIR", directory), \
                 patch.object(gen, "METADATA_PATH", meta), \
                 patch.object(gen, "write_if_changed", capture), \
                 contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = gen.main(["--no-ai"])
            return rc, captured.get("out"), buf.getvalue()

    def names(doc):
        return [p["name"] for p in (doc or {}).get("plugins", [])]

    # A healthy run proves the happy path and supplies the baseline that the
    # recovery scenarios below fall back on.
    rc, generated, log = run(lambda d: None, seed)
    r.check("recovery through main(): healthy run emits a plugin",
            rc == 0 and bool(names(generated)), f"rc={rc}; {names(generated)}")
    if not names(generated):
        return

    target = names(generated)[0]
    baseline = dict(catalog, plugins=generated["plugins"])

    # The plugin's file becomes unparseable. build-plugins.py leaves the built
    # tree in place, so the catalog must keep the entry too.
    def break_file(d):
        for f in d.glob("*.yml"):
            if f.name.startswith("_"):
                continue
            if f.stem == target or target in f.read_text(encoding="utf-8"):
                f.write_text("name: x\n  bad indent:\n", encoding="utf-8")

    rc, out, log = run(break_file, baseline)
    r.check("recovery through main(): a broken file keeps the published plugin",
            rc == 0 and target in names(out), f"rc={rc}; {names(out)}")
    r.check("recovery through main(): the retention is reported",
            "KEPT." in log)

    # Deliberate deletion must still delist.
    def delete_file(d):
        for f in d.glob("*.yml"):
            if not f.name.startswith("_"):
                f.unlink()

    rc, out, log = run(delete_file, baseline)
    r.check("recovery through main(): a deleted plugin is delisted",
            rc == 0 and target not in names(out), f"rc={rc}; {names(out)}")

    # Nothing to fall back on: a broken file with no baseline entry stays out.
    empty_baseline = dict(baseline, plugins=[])
    rc, out, log = run(break_file, empty_baseline)
    r.check("recovery through main(): an unpublished plugin is not invented",
            rc == 0 and not names(out), f"rc={rc}; {names(out)}")

    # The case the real catalog cannot exercise: every plugin file today is
    # named after its plugin, so matching on the filename stem and matching on
    # the declared source path give the same answer. Rename the file so they
    # diverge, then break it. Only source matching can save the entry, because
    # a broken file's `name` cannot be read and its stem is now wrong.
    renamed_baseline = json.loads(json.dumps(baseline))
    for plugin in renamed_baseline["plugins"]:
        if plugin["name"] == target:
            plugin["source"] = "plugins.d/bundle.yml"

    def rename_then_break(d):
        for f in d.glob("*.yml"):
            if f.name.startswith("_"):
                continue
            if f.stem == target or target in f.read_text(encoding="utf-8"):
                f.rename(d / "bundle.yml")
        (d / "bundle.yml").write_text("name: x\n  bad indent:\n", encoding="utf-8")

    rc, out, log = run(rename_then_break, renamed_baseline)
    r.check("recovery through main(): survives when filename and name differ",
            rc == 0 and target in names(out), f"rc={rc}; {names(out)}")


def check_plugin_discovery(gen, taxonomy, validator, r):
    """Drive load_plugin_specs against a real directory.

    The roll-up tests patch this function out, so without these the whole I/O
    half of the plugin pass -- the parts that read maintainer-authored YAML and
    decide what is a plugin -- would never execute.
    """
    import tempfile
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "_defaults.yml").write_text("version: '1.0.0'\n", encoding="utf-8")
        (directory / "good.yml").write_text(
            "name: good\ndescription: A good plugin.\ninclude_skills:\n  - skills/a/\n",
            encoding="utf-8",
        )
        (directory / "broken.yml").write_text("name: broken\n  bad indent:\n", encoding="utf-8")
        (directory / "notamap.yml").write_text("- just\n- a\n- list\n", encoding="utf-8")
        (directory / "Shouty.yml").write_text(
            "name: Shouty\ndescription: d\n", encoding="utf-8"
        )
        (directory / "noname.yml").write_text("description: d\n", encoding="utf-8")

        warnings: list[str] = []
        with patch.object(gen, "PLUGINS_D_DIR", directory), \
             patch.object(gen, "REPO_ROOT", directory.parent):
            specs = gen.load_plugin_specs(warnings)

        names = [s["name"] for s in specs]
        r.check("underscore files are not plugins", "_defaults" not in names)
        r.check("a valid plugin file is discovered", names == ["good"],
                "" if names == ["good"] else str(names))
        joined = "\n".join(warnings)
        r.check("malformed YAML warns instead of raising", "could not be parsed" in joined)
        r.check("a non-mapping file warns", "expected a YAML mapping" in joined)
        r.check("a non-kebab-case name is rejected", "not lowercase kebab-case" in joined)
        r.check("a missing name warns", "missing or non-string `name`" in joined)

    # _defaults.yml is merged by build-plugins.py before it reads any field,
    # so a description or include_skills declared there is inherited. Reading
    # only the per-plugin file would describe a different plugin.
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "_defaults.yml").write_text(
            "description: Inherited description.\n", encoding="utf-8"
        )
        (directory / "inherits.yml").write_text("name: inherits\n", encoding="utf-8")
        warnings = []
        with patch.object(gen, "PLUGINS_D_DIR", directory), \
             patch.object(gen, "REPO_ROOT", directory.parent):
            specs_out = gen.load_plugin_specs(warnings)
        r.check("fields are inherited from _defaults.yml",
                len(specs_out) == 1
                and specs_out[0].get("description") == "Inherited description.")

        (directory / "inherits.yml").write_text(
            "name: inherits\ndescription: Own description.\n", encoding="utf-8"
        )
        warnings = []
        with patch.object(gen, "PLUGINS_D_DIR", directory), \
             patch.object(gen, "REPO_ROOT", directory.parent):
            specs_out = gen.load_plugin_specs(warnings)
        r.check("a per-plugin field wins over the default",
                specs_out[0].get("description") == "Own description.")

    # Discovery is the single place plugin identity is resolved, and it fills
    # the declared set recovery uses. The two disagreeing is what previously
    # delisted published plugins and resurrected deleted ones.
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        # name differs from filename, and comes via _defaults.yml
        (directory / "_defaults.yml").write_text("name: inherited\n", encoding="utf-8")
        (directory / "bundle.yml").write_text("description: d\n", encoding="utf-8")
        declared: set[str] = set()
        warnings = []
        with patch.object(gen, "PLUGINS_D_DIR", directory), \
             patch.object(gen, "REPO_ROOT", directory.parent):
            specs_out = gen.load_plugin_specs(warnings, declared)
        r.check("declared identity follows `name`, including from defaults",
                declared == {"inherited"}, str(declared))
        r.check("the declared name is the one the entry is built under",
                len(specs_out) == 1 and specs_out[0]["name"] == "inherited")

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "broken.yml").write_text("name: x\n  bad:\n", encoding="utf-8")
        (directory / "baddate.yml").write_text("release_date: 2026-02-30\n", encoding="utf-8")
        (directory / "shouty.yml").write_text("name: NOT-VALID\n", encoding="utf-8")
        declared = set()
        warnings = []
        with patch.object(gen, "PLUGINS_D_DIR", directory), \
             patch.object(gen, "REPO_ROOT", directory.parent):
            gen.load_plugin_specs(warnings, declared)
        # The name cannot be read from a broken file, and need not equal the
        # filename, so the SOURCE PATH is declared and recovery matches on it.
        r.check("an unparseable file declares its source path, not its stem",
                any(d.endswith("broken.yml") for d in declared)
                and "broken" not in declared, str(declared))
        r.check("a YAML value that raises a builtin is caught, not propagated",
                any(d.endswith("baddate.yml") for d in declared)
                and any("could not be parsed" in w for w in warnings),
                str(warnings))
        r.check("a name discovery rejects is NOT declared, so it cannot be recovered",
                "NOT-VALID" not in declared and "shouty" not in declared, str(declared))

    # Recovery: a published plugin must not vanish because a skill it bundles
    # went missing. nvidia-skills bundles one skill, so without this a single
    # excluded skill would remove the flagship plugin from the launch page.
    prior_plugin = {
        "name": "fixture",
        "path": "plugins/fixture",
        "source": "plugins.d/fixture.yml",
        "description": "A fixture plugin.",
        "skills": ["skills/a"],
        "metadata": {
            "classification.category.primary":
                taxonomy["classification.category.primary"]["values"][0],
            "product.primary": [taxonomy["product.primary"]["values"][0]],
            "audience": taxonomy["audience"]["values"][0],
        },
    }
    baseline_with_plugin = {"skills": [], "plugins": [prior_plugin]}

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "fixture.yml").write_text("name: fixture\n", encoding="utf-8")
        warnings = []
        with patch.object(gen, "PLUGINS_D_DIR", directory):
            kept = gen.recover_plugin_entries([], baseline_with_plugin, validator, warnings, {'fixture'})
        r.check("a plugin that fails to rebuild is kept from the baseline",
                len(kept) == 1 and kept[0] == prior_plugin)
        r.check("the recovery is warned about, not silent",
                any("KEPT." in w for w in warnings))

        # A plugin that rebuilt normally must not be duplicated.
        warnings = []
        with patch.object(gen, "PLUGINS_D_DIR", directory):
            rebuilt = gen.recover_plugin_entries(
                [dict(prior_plugin, description="fresh")],
                baseline_with_plugin, validator, warnings, {"fixture"},
            )
        r.check("a plugin that rebuilt is not recovered over",
                len(rebuilt) == 1 and rebuilt[0]["description"] == "fresh" and not warnings)

    # Deliberate deletion must still delist: no plugins.d file, no recovery.
    with tempfile.TemporaryDirectory() as tmp:
        warnings = []
        with patch.object(gen, "PLUGINS_D_DIR", Path(tmp)):
            # Deleted means not declared, so the declared set is empty.
            gone = gen.recover_plugin_entries(
                [], baseline_with_plugin, validator, warnings, set()
            )
        r.check("a deleted plugin is not resurrected", gone == [] and not warnings)

    # The baseline is a file on disk: hand-editable, possibly older than this
    # code. A malformed plugins block there must not stop the skills path.
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "fixture.yml").write_text("name: fixture\n", encoding="utf-8")
        for label, bad_baseline in (
            ("a non-string plugin name", {"plugins": [{"name": ["fixture"]}]}),
            ("a null plugins block", {"plugins": None}),
            ("a non-list plugins block", {"plugins": {"fixture": {}}}),
            ("a non-dict plugin entry", {"plugins": ["fixture"]}),
        ):
            warnings = []
            try:
                with patch.object(gen, "PLUGINS_D_DIR", directory):
                    gen.recover_plugin_entries([], bad_baseline, validator, warnings, {'fixture'})
                raised = None
            except Exception as exc:  # noqa: BLE001
                raised = exc
            r.check(f"{label} in the baseline does not raise", raised is None,
                    f"raised={raised!r}" if raised is not None else "")

    # The emitted block has to satisfy the published schema. Validating a
    # built document here is what catches a field the roll-up leaves out.
    skills = [{
        "path": "skills/a",
        "name": "a",
        "description": "d",
        "metadata": {
            "product.primary": taxonomy["product.primary"]["values"][0],
            "classification.category.primary":
                taxonomy["classification.category.primary"]["values"][0],
            "catalog.subdomain": "agentic-ai",
            "audience": taxonomy["audience"]["values"][0],
            "discovery.activity_tags": "get_started",
        },
    }]
    spec = {
        "name": "fixture", "__source": "plugins.d/fixture.yml",
        "description": "A fixture plugin.", "include_skills": ["skills/a/"],
    }
    warnings = []
    with patch.object(gen, "load_plugin_specs", lambda _w, _d=None: [spec]):
        built = gen.build_plugin_entries(skills, taxonomy, warnings, validator)
    errors: list[str] = []
    gen.validate_against_schema(
        {"skills": skills, "plugins": built}, validator, "metadata.json", errors
    )
    r.check("emitted plugins block validates against the schema",
            len(built) == 1 and not errors,
            "; ".join(errors))


def check_plugin_rollup(gen, taxonomy, validator, r):
    """Exercise the plugin taxonomy roll-up without touching plugins.d/.

    The launch plugins other than nvidia-skills do not exist yet, and they
    must not be faked in plugins.d/ to test this: build-plugins.py globs that
    directory and upserts whatever it finds into three marketplace registries,
    so a fixture file there would publish a plugin. The shapes those plugins
    will have are reproduced here instead, as synthetic entries.
    """

    def pick(field, preferred, index):
        values = taxonomy[field]["values"]
        return preferred if preferred in values else values[index]

    # Real catalog values where the vocabulary still has them, positional
    # fallbacks otherwise, so an enum rename cannot silently void these tests.
    ai = pick("classification.category.primary", "ai_and_machine_learning", 0)
    infra = pick("classification.category.primary", "infrastructure", 1)
    devtools = pick("classification.category.primary", "developer_tools", 2)
    tao = pick("product.primary", "TAO Toolkit", 0)
    doca = pick("product.primary", "DOCA", 1)
    bn_a = pick("product.primary", "BioNeMo NIMs", 2)
    bn_b = pick("product.primary", "BioNeMo Open Models", 3)
    bn_c = pick("product.primary", "BioNeMo Libraries", 4)
    dev = pick("audience", "developer", 0)
    mle = pick("audience", "ml_engineer", 1)
    sec = pick("audience", "security_operations", 2)

    def entry(path, domain, product, audience):
        return {
            "path": path,
            "metadata": {
                "classification.category.primary": domain,
                "product.primary": product,
                "audience": ",".join(audience),
            },
        }

    def bundle(domains, product=None, audiences=None):
        out = []
        for i, domain in enumerate(domains):
            out.append(
                entry(
                    f"skills/fixture-{i}",
                    domain,
                    product or tao,
                    audiences[i] if audiences else [dev],
                )
            )
        return out

    # NVFLARE shape: every skill in one domain.
    pure = gen.derive_plugin_taxonomy(bundle([ai] * 9))
    r.check("single-domain bundle keeps that domain",
            pure["classification.category.primary"] == ai)
    r.check("single-product bundle yields one library",
            pure["product.primary"] == [tao])

    # TAO shape: 88% in one domain, the rest spread.
    tao_like = gen.derive_plugin_taxonomy(bundle([ai] * 72 + [infra] * 10))
    r.check("dominant domain wins over a minority",
            tao_like["classification.category.primary"] == ai)

    # DOCA shape: 60% purity, the weakest real case.
    doca_like = gen.derive_plugin_taxonomy(bundle([infra] * 32 + [ai] * 21, product=doca))
    r.check("low-purity bundle still resolves to its plurality",
            doca_like["classification.category.primary"] == infra)

    # A tie must not depend on input order, or the output stops being
    # byte-stable across runs.
    forward = gen.derive_plugin_taxonomy(bundle([ai, infra]))
    reverse = gen.derive_plugin_taxonomy(bundle([infra, ai]))
    r.check("tied domains break alphabetically", forward == reverse)
    r.check("tie-break picks the alphabetically first domain",
            forward["classification.category.primary"] == sorted([ai, infra])[0])

    # BioNeMo shape: one product registered under several names.
    multi = gen.derive_plugin_taxonomy(
        [entry(f"skills/bn-{i}", ai, p, [dev])
         for i, p in enumerate([bn_a, bn_b, bn_c, bn_a])]
    )
    r.check("multi-product bundle lists every library, sorted",
            multi["product.primary"] == sorted({bn_a, bn_b, bn_c}))

    # Audience: held by half or more is kept, below half is dropped.
    threshold = gen.derive_plugin_taxonomy(
        bundle([ai] * 4, audiences=[[dev, mle], [dev, mle], [dev], [dev]])
    )
    r.check("audience kept at exactly half", mle in threshold["audience"].split(","))
    sparse = gen.derive_plugin_taxonomy(
        bundle([ai] * 5, audiences=[[dev, sec], [dev], [dev], [dev], [dev]])
    )
    r.check("audience below half is dropped",
            sec not in sparse["audience"].split(","))

    # The field is never empty: with nothing above the line, the most common
    # value is kept rather than emitting a value the schema pattern rejects.
    fallback = gen.derive_plugin_taxonomy(
        bundle([ai] * 3, audiences=[[dev], [mle], [sec]])
    )
    r.check("audience never empties out", fallback["audience"] != "")

    # Every derived document has to satisfy the vocabulary it was built from.
    for label, built in (("pure", pure), ("tao", tao_like), ("multi", multi)):
        errors: list[str] = []
        gen.validate_plugin_taxonomy(built, taxonomy, label, errors)
        r.check(f"derived taxonomy is in-vocabulary ({label})", not errors, "; ".join(errors))

    # Entry assembly, with plugins.d stubbed out.
    from unittest.mock import patch

    def specs(entries):
        return patch.object(gen, "load_plugin_specs", lambda _w, _d=None: entries)

    skills = bundle([ai] * 2)
    spec = {
        "name": "fixture", "__source": "plugins.d/fixture.yml",
        "description": "A fixture plugin.",
        # Trailing slashes are how plugins.d writes these; metadata.json keys
        # them without one.
        "include_skills": [s["path"] + "/" for s in skills],
    }

    errors = []
    with specs([spec]):
        built = gen.build_plugin_entries(skills, taxonomy, errors, validator)
    r.check("plugin entry builds from include_skills", len(built) == 1 and not errors,
            "; ".join(errors))
    r.check("trailing slashes resolve",
            len(built) == 1 and built[0]["skills"] == sorted(s["path"] for s in skills))
    r.check("plugin path points at the built tree",
            len(built) == 1 and built[0]["path"] == "plugins/fixture")

    # A path that does not resolve is normal operational drift -- a skill that
    # is new and unenriched under --no-ai, excluded for compliance, or renamed
    # upstream. It must degrade the plugin, never the catalog, and the plugin
    # still builds from whatever did resolve.
    errors = []
    partial_includes = [s["path"] + "/" for s in skills] + ["skills/does-not-exist/"]
    with specs([dict(spec, include_skills=partial_includes)]):
        survived = gen.build_plugin_entries(skills, taxonomy, errors, validator)
    r.check("an unresolvable path still builds the plugin", len(survived) == 1)
    r.check("an unresolvable path drops only that skill",
            survived and survived[0]["skills"] == sorted(s["path"] for s in skills))
    r.check("an unresolvable path is warned about",
            any("have no entry in metadata.json" in e for e in errors))
    r.check("the warning does not tell the maintainer to delete a valid path",
            not any("or remove it" in e for e in errors))

    errors = []
    with specs([dict(spec, include_skills=[])]):
        orphan = gen.build_plugin_entries(skills, taxonomy, errors, validator)
    r.check("plugin with no catalog skill is reported, not guessed",
            not orphan and any("bundles no catalog skill" in e for e in errors))

    # With nothing to roll up, a partial declaration cannot be completed. It
    # must say so, not emit an entry missing a field and blame a `None`.
    errors = []
    with specs([dict(spec, include_skills=[], taxonomy={"product.primary": [tao]})]):
        partial = gen.build_plugin_entries(skills, taxonomy, errors, validator)
    r.check("a partial declaration on a skill-less plugin is refused", not partial)
    r.check("the refusal names the missing fields",
            any("Missing: classification.category.primary" in e for e in errors))
    r.check("the refusal does not blame a None the maintainer never wrote",
            not any("`None`" in e for e in errors))

    # A dotted-name typo previously produced a clean entry with rolled-up
    # values and no warning at all.
    errors = []
    with specs([dict(spec, taxonomy={"classification.category": devtools})]):
        typo_out = gen.build_plugin_entries(skills, taxonomy, errors, validator)
    r.check("an unknown taxonomy field is rejected, not ignored",
            not typo_out and any("unknown `taxonomy:` field" in e for e in errors))

    # Hostile or careless YAML shapes must warn, never raise.
    for label, bad in (
        ("include_skills as a bare string", dict(spec, include_skills="skills/a/")),
        ("taxonomy as a list", dict(spec, taxonomy=["nope"])),
        ("nested list under product.primary",
         dict(spec, taxonomy={"product.primary": [[tao]]})),
        ("non-string domain", dict(spec, taxonomy={"classification.category.primary": 7})),
        ("missing description", dict(spec, description=None)),
    ):
        errors = []
        try:
            with specs([bad]):
                out = gen.build_plugin_entries(skills, taxonomy, errors, validator)
            raised = None
        except Exception as exc:  # noqa: BLE001 - the point is that nothing escapes
            raised, out = exc, None
        r.check(f"{label} warns instead of raising",
                raised is None and not out and bool(errors),
                f"raised={raised!r}" if raised is not None else "")

    errors = []
    override = {"classification.category.primary": devtools, "audience": sec}
    with specs([dict(spec, taxonomy=override)]):
        overridden = gen.build_plugin_entries(skills, taxonomy, errors, validator)
    r.check("declared taxonomy overrides the roll-up",
            len(overridden) == 1
            and overridden[0]["metadata"]["classification.category.primary"] == devtools
            and overridden[0]["metadata"]["audience"] == sec,
            "; ".join(errors))
    r.check("undeclared fields still roll up under an override",
            len(overridden) == 1 and overridden[0]["metadata"]["product.primary"] == [tao])

    errors = []
    with specs([dict(spec, taxonomy={"classification.category.primary": "not-a-domain"})]):
        rejected = gen.build_plugin_entries(skills, taxonomy, errors, validator)
    r.check("an out-of-vocabulary override is rejected",
            not rejected and any("not a value in metadata.schema.json" in e for e in errors))

    # Error text is the only documentation most maintainers will read, so the
    # guidance in it is held to the same standard as the behaviour.
    errors = []
    with specs([dict(spec, include_skills=[])]):
        gen.build_plugin_entries(skills, taxonomy, errors, validator)
    orphan_msg = "\n".join(errors)
    r.check("skill-less plugin error shows a pasteable block",
            "taxonomy:" in orphan_msg and "classification.category.primary:" in orphan_msg)
    r.check("skill-less plugin error lists the domain vocabulary",
            all(v in orphan_msg for v in taxonomy["classification.category.primary"]["values"]))
    r.check("skill-less plugin error names the file to edit",
            orphan_msg.count("plugins.d/fixture.yml") >= 2)
    r.check("skill-less plugin error offers the other remedy",
            "include_skills" in orphan_msg)

    errors = []
    typo = {
        "classification.category.primary": devtools,
        "product.primary": [tao[:-1] + "x"],
        "audience": dev[:-1] + "x",
    }
    gen.validate_plugin_taxonomy(typo, taxonomy, "plugins.d/fixture.yml", errors)
    typo_msg = "\n".join(errors)
    r.check("a mistyped product suggests the near match", f"Did you mean: {tao}" in typo_msg)
    r.check("a mistyped audience suggests the near match", f"Did you mean: {dev}" in typo_msg)
    # Inputs that satisfy the vocabulary check but violate the schema. Each of
    # these used to build a clean entry and then fail whole-document
    # validation, which writes no output at all and blocks every skill.
    for label, bad in (
        ("empty product.primary", dict(spec, taxonomy={"product.primary": []})),
        ("empty audience", dict(spec, taxonomy={"audience": ""})),
        ("trailing comma in audience", dict(spec, taxonomy={"audience": dev + ","})),
    ):
        errors = []
        with specs([bad]):
            out = gen.build_plugin_entries(skills, taxonomy, errors, validator)
        r.check(f"{label} is caught before whole-document validation",
                not out and bool(errors))

    # Duplicate paths violate uniqueItems on `skills`.
    errors = []
    dupes = [skills[0]["path"] + "/", skills[0]["path"] + "/"]
    with specs([dict(spec, include_skills=dupes)]):
        deduped = gen.build_plugin_entries(skills, taxonomy, errors, validator)
    r.check("duplicate include_skills are de-duplicated, not emitted twice",
            len(deduped) == 1 and deduped[0]["skills"] == [skills[0]["path"]])

    # A YAML mapping may have non-string keys.
    errors = []
    try:
        with specs([dict(spec, taxonomy={None: "x", "typo": "y"})]):
            out = gen.build_plugin_entries(skills, taxonomy, errors, validator)
        raised = None
    except Exception as exc:  # noqa: BLE001
        raised, out = exc, None
    r.check("a non-string taxonomy key warns instead of raising",
            raised is None and not out and bool(errors),
            f"raised={raised!r}" if raised is not None else "")

    # One skill must not vote twice: the skill schema permits a repeated
    # audience token, and counting tokens let a single skill carry a value
    # over the halfway line on its own.
    repeated = gen.derive_plugin_taxonomy([
        entry("skills/r0", ai, tao, [dev, dev]),
        entry("skills/r1", ai, tao, [mle]),
        entry("skills/r2", ai, tao, [sec]),
    ])
    r.check("a repeated audience token does not outvote other skills",
            sorted(repeated["audience"].split(",")) == sorted([dev, mle, sec]),
            repeated["audience"])

    # Directory-form includes are refused, not approximated. build-plugins.py
    # expands them to immediate children holding a SKILL.md; guessing at that
    # produced two separate divergences. The catalog is flat today so nothing
    # uses the form, and a loud refusal beats a silent wrong answer.
    errors = []
    with specs([dict(spec, include_skills=["skills/"])]):
        refused = gen.build_plugin_entries(skills, taxonomy, errors, validator)
    r.check("a directory-form include is refused, not guessed at",
            not refused and any("directory-form" in e for e in errors))
    r.check("the refusal says the published plugin would disagree",
            any("would disagree" in e for e in errors))

    # Normalisation matches build-plugins.py: it resolves "./", and it does
    # NOT strip surrounding whitespace.
    errors = []
    with specs([dict(spec, include_skills=["./" + skills[0]["path"] + "/"])]):
        dotted = gen.build_plugin_entries(skills, taxonomy, errors, validator)
    r.check("a './' prefixed include resolves, as it does in the builder",
            len(dotted) == 1 and dotted[0]["skills"] == [skills[0]["path"]])

    errors = []
    with specs([dict(spec, include_skills=[" " + skills[0]["path"] + "/ "])]):
        spaced = gen.build_plugin_entries(skills, taxonomy, errors, validator)
    r.check("a whitespace-padded include does NOT resolve, as in the builder",
            not spaced or spaced[0]["skills"] != [skills[0]["path"]])

    errors = []
    gen.validate_plugin_taxonomy(
        {"classification.category.primary": "nope",
         "product.primary": [tao], "audience": dev},
        taxonomy, "plugins.d/fixture.yml", errors,
    )
    domain_msg = "\n".join(errors)
    r.check("an invalid domain lists every valid domain",
            all(v in domain_msg
                for v in taxonomy["classification.category.primary"]["values"]))


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
    check_plugin_rollup(gen, taxonomy, validator, r)
    check_plugin_discovery(gen, taxonomy, validator, r)
    check_plugin_never_blocks_main(gen, r)
    check_recovery_through_main(gen, r)
    check_mirrored_plugins_through_main(gen, r)
    check_mirrored_plugin_edge_cases(gen, r)

    return r.report()


if __name__ == "__main__":
    sys.exit(main())
