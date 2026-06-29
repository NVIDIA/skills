#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Per-plugin SemVer enforcement and auto-bump.

For every catalog plugin defined in plugins.d/<name>.yml, compare the
current head payload against a base ref and decide:

  - No payload change     -> keep version.
  - Content-only change   -> auto-bump z.
  - Structural change     -> auto-bump y.
  - Builder already bumped version -> validate and accept.

Major (x) bumps are always builder-only — they encode a breaking-change
claim that automation cannot infer from a diff. Everything else is
mechanical: the script computes the magnitude from the diff shape and
writes it back. Reviewers see the proposed bump in the PR diff alongside
the change that triggered it.

Validation of builder-set versions enforces only mechanical safety:

  - Monotonic: new version must be strictly greater than base.
  - No pre-release tags: ^\\d+\\.\\d+\\.\\d+$ only.
  - No oversized major skip: 1.x -> 5.x rejected as a likely typo.

Magnitude-vs-change-shape is intentionally NOT validated; if the builder
explicitly set a version, automation respects it. The reviewer sees the
diff and pushes back if the magnitude is wrong.

Payload hashing is deterministic and excludes the YAML `version` field
itself plus generic noise (`__pycache__`, `.DS_Store`, `.pyc`, etc.).
See `_hash_plugin_tree` and `_excluded` for the exact rules.

Usage:
  version-plugins.py                Plan only; prints findings, no writes.
  version-plugins.py --apply        Apply auto-bumps in place (rewrites
                                    plugins.d/<name>.yml comment-preserved).
  version-plugins.py --base REF     Override base ref (default: auto).
  version-plugins.py --only NAME    Restrict to one plugin.
  version-plugins.py --check        Exit non-zero if any plugin needs a
                                    builder action (no writes).

Exit codes:
  0   Nothing to do, or all bumps applied successfully.
  1   Internal error (bad YAML, git failure, etc.).
  2   Validation findings exist that require a builder decision
      (e.g. invalid version edit on the PR).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGINS_D = REPO_ROOT / "plugins.d"
PLUGINS_D_DEFAULTS = PLUGINS_D / "_defaults.yml"
PLUGINS_DIR = REPO_ROOT / "plugins"
BUILD_SCRIPT = REPO_ROOT / ".github" / "scripts" / "build-plugins.py"

# Files/dirs excluded from the payload hash. Kept simple and explicit
# (substring + suffix checks) rather than fnmatch-style globs because
# the rule set is small and stable. Matched against the path relative
# to plugins/<name>/.
EXCLUDED_DIR_SEGMENTS = frozenset({"__pycache__", ".idea", ".vscode"})
EXCLUDED_FILENAMES = frozenset({".DS_Store"})
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".swp")

# Spec fields that count as structural (user-visible discovery surface).
# A change to any of these forces a builder-owned x or y bump.
#
# Asset *paths* (logo/composer_icon/screenshots) are structural; asset
# *bytes* are content-only and handled separately when walking files.
STRUCTURAL_SPEC_FIELDS = (
    "capabilities",
    "default_prompts",
)
STRUCTURAL_ASSET_FIELDS = (
    "logo",
    "composer_icon",
    "screenshots",
)

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


# ---------------------------- semver helpers ---------------------------------


@dataclass(frozen=True, order=True)
class SemVer:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, raw: str) -> "SemVer":
        """Parse a strict MAJOR.MINOR.PATCH string. No pre-release tags."""
        m = SEMVER_RE.match(raw.strip())
        if not m:
            raise ValueError(
                f"invalid version {raw!r}: must match MAJOR.MINOR.PATCH "
                "(no pre-release tags, no leading zeros expected)"
            )
        return cls(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def bump_patch(self) -> "SemVer":
        return SemVer(self.major, self.minor, self.patch + 1)

    def bumped_part(self, other: "SemVer") -> str | None:
        """Return 'major' | 'minor' | 'patch' | None describing which part
        of `other` is incremented relative to `self`. Returns None when
        the versions are equal or `other` is not strictly greater."""
        if other <= self:
            return None
        if other.major > self.major:
            return "major"
        if other.minor > self.minor:
            return "minor"
        if other.patch > self.patch:
            return "patch"
        return None


# ---------------------------- yaml helpers -----------------------------------


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a YAML mapping at top level")
    return data


def read_yaml_at_ref(ref: str, rel_path: str) -> dict[str, Any] | None:
    """Read a YAML file from a git ref via `git show`. Returns None if the
    file is missing at that ref (e.g. plugin yaml didn't exist yet)."""
    try:
        out = subprocess.run(
            ["git", "show", f"{ref}:{rel_path}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None
    data = yaml.safe_load(out)
    if not isinstance(data, dict):
        return None
    return data


def write_version_to_yaml(path: Path, new_version: str) -> None:
    """Rewrite the `version:` field in a plugin yaml, preserving comments,
    blank lines, key order, and existing long-string formatting. Uses
    ruamel.yaml because pyyaml round-trips drop both comments and order.

    If the file doesn't yet declare `version:` (inheriting from defaults),
    insert it as the second key (right after `name:`) so it stays near the
    top of the file. The new version scalar is always double-quoted to
    match the convention in `plugins.d/_defaults.yml`."""
    try:
        from ruamel.yaml import YAML
        from ruamel.yaml.scalarstring import DoubleQuotedScalarString
    except ImportError as e:  # pragma: no cover - install handled by wrapper
        raise SystemExit(
            "ruamel.yaml is required for in-place version writes. "
            "Install with `pip install ruamel.yaml` and re-run."
        ) from e

    ryaml = YAML()
    ryaml.preserve_quotes = True
    ryaml.indent(mapping=2, sequence=4, offset=2)
    # Default width is 80, which reflows long inline strings (description,
    # default_prompts, etc.). Set to effectively-infinite to leave them alone.
    ryaml.width = 10_000

    with path.open() as f:
        data = ryaml.load(f)
    if data is None:
        raise SystemExit(f"{path}: empty or unreadable")

    quoted = DoubleQuotedScalarString(new_version)
    if "version" in data:
        data["version"] = quoted
    else:
        # Insert right after `name:` if present, otherwise at position 0.
        keys = list(data.keys())
        insert_at = (keys.index("name") + 1) if "name" in keys else 0
        data.insert(insert_at, "version", quoted)

    with path.open("w") as f:
        ryaml.dump(data, f)


# ---------------------------- payload model ----------------------------------


def _excluded(rel: Path) -> bool:
    if any(part in EXCLUDED_DIR_SEGMENTS for part in rel.parts[:-1]):
        return True
    name = rel.name
    if name in EXCLUDED_FILENAMES:
        return True
    if name.endswith(EXCLUDED_SUFFIXES):
        return True
    return False


def _spec_to_structural_tuple(spec: dict[str, Any]) -> tuple[Any, ...]:
    """Project a plugin spec to the tuple of values that drive structural
    classification. Order is fixed so equality is comparable across runs."""
    parts: list[Any] = []
    for field_name in STRUCTURAL_SPEC_FIELDS:
        value = spec.get(field_name)
        if isinstance(value, list):
            parts.append((field_name, tuple(value)))
        else:
            parts.append((field_name, value))
    for field_name in STRUCTURAL_ASSET_FIELDS:
        value = spec.get(field_name)
        if isinstance(value, list):
            parts.append((field_name, tuple(value)))
        else:
            parts.append((field_name, value))
    return tuple(parts)


def _resolved_skill_set(spec: dict[str, Any], source_root: Path) -> tuple[str, ...]:
    """Return the sorted tuple of skill basenames the plugin would ship,
    resolved against `source_root` (so we can call this for both head and
    a worktree at the base ref)."""
    entries = spec.get("include_skills") or []
    names: set[str] = set()
    for entry in entries:
        rel = entry.rstrip("/")
        src = (source_root / rel).resolve()
        if not src.is_dir():
            continue
        if (src / "SKILL.md").is_file():
            names.add(src.name)
            continue
        for child in src.iterdir():
            if child.is_dir() and (child / "SKILL.md").is_file():
                names.add(child.name)
    return tuple(sorted(names))


def _hash_plugin_tree(plugin_dir: Path) -> dict[str, str]:
    """Walk plugins/<name>/ and return a dict of {relative_path: sha256_hex}.
    Skips excluded paths. Symlinks are followed (skill_files: symlink
    materializes the curated subset as relative symlinks pointing back at
    skills/; we want to hash the target bytes, not the link text).

    Special handling for plugin.json files: drop the `version` field
    before hashing so that bumping `version` alone doesn't change the
    payload hash."""
    out: dict[str, str] = {}
    if not plugin_dir.is_dir():
        return out
    for root, dirs, files in os.walk(plugin_dir, followlinks=True):
        # Skip excluded directories early so we don't descend.
        dirs.sort()
        for fname in sorted(files):
            abs_path = Path(root) / fname
            rel = abs_path.relative_to(plugin_dir)
            if _excluded(rel):
                continue
            if fname == "plugin.json":
                try:
                    with abs_path.open() as f:
                        data = json.load(f)
                except (OSError, json.JSONDecodeError):
                    # Hash bytes as a fallback so corrupt files still
                    # produce a deterministic hash and surface in diffs.
                    out[rel.as_posix()] = _sha256_file(abs_path)
                    continue
                data.pop("version", None)
                blob = json.dumps(
                    data, sort_keys=True, ensure_ascii=False
                ).encode("utf-8")
                out[rel.as_posix()] = hashlib.sha256(blob).hexdigest()
            else:
                out[rel.as_posix()] = _sha256_file(abs_path)
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _payload_fingerprint(file_hashes: dict[str, str]) -> str:
    """Combine the per-file hashes into one stable fingerprint string.
    Two trees with identical {path -> hash} maps produce the same value."""
    h = hashlib.sha256()
    for rel in sorted(file_hashes):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(file_hashes[rel].encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------- git helpers ------------------------------------


def git(*args: str, cwd: Path | None = None) -> str:
    """Run a git command and return stdout. Raises on non-zero exit."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def resolve_base_ref(explicit: str | None) -> str:
    """Pick the right base ref per the PRD:
      - --base CLI flag wins.
      - GITHUB_BASE_REF (set on pull_request events) -> origin/<that>.
      - On `main` (post-merge push or local main) -> HEAD~1.
      - Anywhere else -> fail with a clear message.
    """
    if explicit:
        return explicit
    pr_base = os.environ.get("GITHUB_BASE_REF")
    if pr_base:
        return f"origin/{pr_base}"
    try:
        current = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    except subprocess.CalledProcessError as e:  # pragma: no cover - rare
        raise SystemExit(f"git rev-parse failed: {e.stderr}") from e
    if current == "main":
        return "HEAD~1"
    raise SystemExit(
        "could not auto-resolve base ref. "
        "Pass --base <ref> explicitly (e.g. --base origin/main)."
    )


@contextlib.contextmanager
def git_worktree_at(ref: str) -> Iterator[Path]:
    """Materialize a worktree at `ref` in a temp directory. Yields the
    worktree path. Cleans up on exit (even if the caller raises)."""
    tmp = Path(tempfile.mkdtemp(prefix="version-plugins-worktree-"))
    try:
        git("worktree", "add", "--detach", str(tmp), ref)
        yield tmp
    finally:
        # Best effort: prune the worktree even if rm fails for some reason.
        try:
            git("worktree", "remove", "--force", str(tmp))
        except subprocess.CalledProcessError:  # pragma: no cover
            shutil.rmtree(tmp, ignore_errors=True)
            try:
                git("worktree", "prune")
            except subprocess.CalledProcessError:
                pass


def run_build(repo_root: Path, only: str | None = None) -> None:
    """Invoke build-plugins.py inside `repo_root` (i.e. either the live
    repo or a base-ref worktree). Required so the head and base both
    materialize plugins/<name>/ consistently before hashing."""
    cmd = [sys.executable, str(repo_root / ".github" / "scripts" / "build-plugins.py")]
    if only:
        cmd += ["--only", only]
    subprocess.run(cmd, cwd=repo_root, check=True, capture_output=True)


# ---------------------------- classification ---------------------------------


class ChangeKind:
    NONE = "none"
    CONTENT = "content"
    STRUCTURAL = "structural"


@dataclass
class PluginAnalysis:
    name: str
    yaml_path: Path
    base_version: SemVer
    head_version: SemVer
    builder_changed_version: bool
    change_kind: str
    structural_reasons: list[str] = field(default_factory=list)


def classify_change(
    base_spec: dict[str, Any] | None,
    head_spec: dict[str, Any],
    base_skill_names: tuple[str, ...],
    head_skill_names: tuple[str, ...],
    base_file_hashes: dict[str, str],
    head_file_hashes: dict[str, str],
) -> tuple[str, list[str]]:
    """Return (ChangeKind, [reasons]). Reasons explain why the change is
    structural; they get surfaced to builders so the failure message is
    actionable."""
    reasons: list[str] = []

    if base_spec is None:
        # Plugin yaml is brand new at head. Treat as structural so the
        # builder explicitly stamps an initial version.
        reasons.append("plugin yaml is newly introduced")
        return ChangeKind.STRUCTURAL, reasons

    if base_skill_names != head_skill_names:
        added = sorted(set(head_skill_names) - set(base_skill_names))
        removed = sorted(set(base_skill_names) - set(head_skill_names))
        if added:
            reasons.append(f"skills added: {', '.join(added)}")
        if removed:
            reasons.append(f"skills removed: {', '.join(removed)}")

    base_structural = _spec_to_structural_tuple(base_spec)
    head_structural = _spec_to_structural_tuple(head_spec)
    if base_structural != head_structural:
        changed_fields: list[str] = []
        for (bk, bv), (_, hv) in zip(base_structural, head_structural):
            if bv != hv:
                changed_fields.append(bk)
        if changed_fields:
            reasons.append(
                "structural spec fields changed: " + ", ".join(changed_fields)
            )

    if reasons:
        return ChangeKind.STRUCTURAL, reasons

    base_fp = _payload_fingerprint(base_file_hashes)
    head_fp = _payload_fingerprint(head_file_hashes)
    if base_fp == head_fp:
        return ChangeKind.NONE, []
    return ChangeKind.CONTENT, []


# ---------------------------- validation -------------------------------------


def validate_builder_version(base: SemVer, head: SemVer) -> list[str]:
    """Apply validation rules to a builder-set version change.
    Returns a list of human-readable findings; empty list means OK.

    Magnitude-matches-change is intentionally NOT checked here — if the
    builder explicitly set a version, we respect their choice. The
    reviewer sees the proposed version delta in the PR diff and can
    push back if it's wrong. We only catch mechanical mistakes:
    non-monotonic edits and accidental large major jumps.
    """
    findings: list[str] = []

    if head <= base:
        findings.append(
            f"version did not increase: base {base} -> head {head} "
            "(must be strictly greater)"
        )
        return findings

    bumped = base.bumped_part(head)
    if bumped == "major" and head.major - base.major > 1:
        findings.append(
            f"major version jumped by more than 1 ({base} -> {head}); "
            "if intentional, add an explicit MAJOR_BUMP signal in the PR"
        )
    return findings


# ---------------------------- analysis loop ----------------------------------


def _list_catalog_plugin_yamls() -> list[Path]:
    """Return the catalog plugin yaml files (skips includes starting `_`)."""
    if not PLUGINS_D.is_dir():
        return []
    return sorted(
        p for p in PLUGINS_D.glob("*.yml") if not p.name.startswith("_")
    )


def _merged_spec(
    defaults: dict[str, Any], plugin_yaml: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(plugin_yaml)
    return merged


def analyze_plugin(
    plugin_yaml_path: Path,
    base_ref: str,
    base_worktree: Path,
) -> PluginAnalysis | None:
    """Compute the analysis for a single catalog plugin. Returns None if
    we can't analyze it (e.g. yaml unreadable, missing name)."""
    rel = plugin_yaml_path.relative_to(REPO_ROOT)

    head_defaults = read_yaml(PLUGINS_D_DEFAULTS) if PLUGINS_D_DEFAULTS.is_file() else {}
    head_plugin_yaml = read_yaml(plugin_yaml_path)
    head_spec = _merged_spec(head_defaults, head_plugin_yaml)

    name = head_spec.get("name")
    if not name:
        print(f"  ! {rel}: missing 'name', skipping", file=sys.stderr)
        return None

    try:
        head_version = SemVer.parse(str(head_spec.get("version", "")))
    except ValueError as e:
        print(f"  ! {rel}: head version invalid: {e}", file=sys.stderr)
        return None

    base_defaults_rel = PLUGINS_D_DEFAULTS.relative_to(REPO_ROOT).as_posix()
    base_defaults = read_yaml_at_ref(base_ref, base_defaults_rel) or {}
    base_plugin_yaml = read_yaml_at_ref(base_ref, rel.as_posix())

    if base_plugin_yaml is None:
        base_spec = None
        base_version = head_version  # placeholder; classify_change handles None spec
        builder_changed_version = True  # everything is new
    else:
        base_spec = _merged_spec(base_defaults, base_plugin_yaml)
        try:
            base_version = SemVer.parse(str(base_spec.get("version", "")))
        except ValueError as e:
            print(f"  ! {rel}: base version invalid: {e}", file=sys.stderr)
            return None
        # "Builder changed version" means the *effective* version moved,
        # which we detect by comparing the raw yaml fields (not the
        # merged spec) plus the defaults fallback.
        head_raw_version = head_plugin_yaml.get("version")
        base_raw_version = base_plugin_yaml.get("version")
        if head_raw_version != base_raw_version:
            builder_changed_version = True
        elif head_raw_version is None:
            # Both sides inherit from defaults; did the default move?
            builder_changed_version = (
                head_defaults.get("version") != base_defaults.get("version")
            )
        else:
            builder_changed_version = False

    head_plugin_dir = PLUGINS_DIR / name
    head_file_hashes = _hash_plugin_tree(head_plugin_dir)
    head_skill_names = _resolved_skill_set(head_spec, REPO_ROOT)

    base_file_hashes: dict[str, str] = {}
    base_skill_names: tuple[str, ...] = ()
    if base_spec is not None:
        base_plugin_dir = base_worktree / "plugins" / name
        base_file_hashes = _hash_plugin_tree(base_plugin_dir)
        base_skill_names = _resolved_skill_set(base_spec, base_worktree)

    change_kind, reasons = classify_change(
        base_spec,
        head_spec,
        base_skill_names,
        head_skill_names,
        base_file_hashes,
        head_file_hashes,
    )

    return PluginAnalysis(
        name=name,
        yaml_path=plugin_yaml_path,
        base_version=base_version,
        head_version=head_version,
        builder_changed_version=builder_changed_version,
        change_kind=change_kind,
        structural_reasons=reasons,
    )


# ---------------------------- reporting & action -----------------------------


@dataclass
class Plan:
    findings: list[str] = field(default_factory=list)  # builder-action-needed
    bumps: list[tuple[PluginAnalysis, SemVer]] = field(default_factory=list)
    no_ops: list[PluginAnalysis] = field(default_factory=list)


def decide(analysis: PluginAnalysis) -> tuple[str, str | None]:
    """Return (verdict, payload) for one plugin.

    verdict ∈ {'noop', 'bump', 'accept', 'fail'}
      noop    -> nothing changed, nothing to do.
      bump    -> auto-bump (z for content, y for structural). Payload
                 is the new SemVer string.
      accept  -> builder already changed version; validated OK.
      fail    -> validation failure; payload is the message.
    """
    if analysis.builder_changed_version:
        findings = validate_builder_version(
            analysis.base_version, analysis.head_version
        )
        if findings:
            return "fail", "; ".join(findings)
        return "accept", None

    if analysis.change_kind == ChangeKind.NONE:
        return "noop", None

    if analysis.change_kind == ChangeKind.STRUCTURAL:
        new = SemVer(
            analysis.head_version.major,
            analysis.head_version.minor + 1,
            0,
        )
        return "bump", str(new)

    return "bump", str(analysis.head_version.bump_patch())


def build_plan(analyses: list[PluginAnalysis]) -> Plan:
    plan = Plan()
    for a in analyses:
        verdict, payload = decide(a)
        if verdict == "noop":
            plan.no_ops.append(a)
        elif verdict == "bump":
            assert payload is not None
            plan.bumps.append((a, SemVer.parse(payload)))
        elif verdict == "accept":
            plan.no_ops.append(a)
        elif verdict == "fail":
            assert payload is not None
            plan.findings.append(f"{a.name}: {payload}")
    return plan


def print_plan(plan: Plan, apply_mode: bool) -> None:
    if plan.no_ops:
        print(f"── no-op ({len(plan.no_ops)}) ──")
        for a in plan.no_ops:
            note = (
                "(builder-set version validated)"
                if a.builder_changed_version
                else "(no payload change)"
            )
            print(f"  · {a.name} {a.head_version} {note}")
    if plan.bumps:
        verb = "applying" if apply_mode else "would apply"
        print(f"── auto-bumps ({len(plan.bumps)}) — {verb} ──")
        for a, new in plan.bumps:
            if a.change_kind == ChangeKind.STRUCTURAL:
                reason = "structural change: " + "; ".join(a.structural_reasons)
            else:
                reason = "content-only change"
            print(f"  > {a.name}: {a.head_version} -> {new}  ({reason})")
    if plan.findings:
        print(f"── findings ({len(plan.findings)}) — builder action needed ──")
        for f in plan.findings:
            print(f"  ! {f}")


def apply_bumps(plan: Plan) -> None:
    """Rewrite each affected plugin yaml with its new version."""
    for analysis, new_version in plan.bumps:
        write_version_to_yaml(analysis.yaml_path, str(new_version))
        print(f"  wrote version: {new_version} -> {analysis.yaml_path.relative_to(REPO_ROOT)}")


# ---------------------------- main -------------------------------------------


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", help="Base ref to compare against (default: auto).")
    p.add_argument("--only", help="Restrict to a single plugin name.")
    p.add_argument(
        "--apply",
        action="store_true",
        help="Apply auto-bumps to plugins.d/<name>.yml. Default is plan-only.",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any plugin needs a builder decision. No writes.",
    )
    args = p.parse_args(argv)

    yamls = _list_catalog_plugin_yamls()
    if args.only:
        yamls = [y for y in yamls if y.stem == args.only]
        if not yamls:
            print(f"error: no catalog plugin named {args.only!r}", file=sys.stderr)
            return 1
    if not yamls:
        print("no catalog plugins to analyze.")
        return 0

    base_ref = resolve_base_ref(args.base)
    print(f"base ref: {base_ref}")

    analyses: list[PluginAnalysis] = []
    with git_worktree_at(base_ref) as worktree:
        # Build the base worktree once so plugins/<name>/ exists there
        # with the right materialization mode (copy vs symlink). Skip
        # gracefully if the base ref predates build-plugins.py.
        if (worktree / ".github" / "scripts" / "build-plugins.py").is_file():
            try:
                run_build(worktree, only=args.only)
            except subprocess.CalledProcessError as e:
                print(
                    f"warning: base build failed; base payloads may be empty. "
                    f"stderr:\n{e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}",
                    file=sys.stderr,
                )

        for ymlfile in yamls:
            a = analyze_plugin(ymlfile, base_ref, worktree)
            if a is not None:
                analyses.append(a)

    plan = build_plan(analyses)
    print_plan(plan, apply_mode=args.apply)

    if args.apply and plan.bumps:
        print("── applying bumps ──")
        apply_bumps(plan)
        # Re-run the canonical build so generated plugin.json files
        # reflect the bumped versions. This keeps `build-plugins.py
        # --check` happy in the same CI run.
        print("── rebuilding to refresh plugin.json files ──")
        run_build(REPO_ROOT, only=args.only)

    if plan.findings:
        return 2
    if args.check and plan.bumps and not args.apply:
        # In --check mode, an unbumped auto-eligible change is also a
        # builder-action item (or, equivalently, a missing CI bot push).
        print(
            "error: --check found auto-bumps that have not been applied. "
            "Re-run with --apply.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
