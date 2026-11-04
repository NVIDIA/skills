#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve pinned upstream skills into a runtime manifest.

Adapted from isaac-mission-control-showcase's parent-owned resolver.
Resolution is read-only except for manifest/env output. Only explicit --prepare
may fetch managed checkouts. No host or simulator operations are performed.
Exit codes: 0 ready, 3 blocked, 2 configuration or preparation failure.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess  # nosec B404
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
DEPENDENCIES = SKILL_DIR / "upstream-versions.lock.json"
sys.path.insert(0, str(SKILL_DIR / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


HOME_ENV = "HUMANOID_CLOUD_CONTROL_HOME"
ROOT_ENV = "HUMANOID_CLOUD_CONTROL_UPSTREAM_ROOT"
MANIFEST_ENV = "HUMANOID_CLOUD_CONTROL_DEPS_MANIFEST"
MANIFEST_SCHEMA = "humanoid-cloud-control/deps-manifest@1"
MANIFEST_NAME = "humanoid-deps.json"

EXIT_BLOCKED = 3
EXIT_CONFIG = 2


def atomic_write(path: Path, text: str) -> None:
    """Write via a temporary sibling plus os.replace so no reader sees a
    partial manifest, and a crash leaves the previous file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def humanoid_home() -> Path:
    return Path(os.environ.get(HOME_ENV) or Path.home() / ".humanoid-cloud-control")


def default_root() -> Path:
    return Path(os.environ.get(ROOT_ENV) or humanoid_home() / "upstreams")


def default_manifest_path() -> Path:
    explicit = os.environ.get(MANIFEST_ENV)
    if explicit:
        return Path(explicit).expanduser()
    return humanoid_home() / "state" / MANIFEST_NAME


def _git(args, cwd):
    git = shutil.which("git")
    if not git:
        raise FileNotFoundError("git is required for dependency verification")
    return subprocess.run(  # nosec B603
        [git, *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=300
    ).stdout.strip()


def head_commit(path: Path):
    try:
        return _git(["rev-parse", "HEAD"], path)
    except (subprocess.SubprocessError, OSError):
        return None


def worktree_dirty(path: Path):
    """True when tracked files differ from HEAD, so runtime code != the commit."""
    try:
        return bool(_git(["status", "--porcelain", "--untracked-files=no"], path))
    except (subprocess.SubprocessError, OSError):
        return None


def resolve_upstream_root(name, spec, root):
    """Return (path, source) for an upstream checkout without searching."""
    override = os.environ.get(spec["root_env"])
    if override:
        return Path(override).expanduser().resolve(), f"env:{spec['root_env']}"
    return root / spec["checkout"], "normalized-root"


def resolve(dependencies, root, *, strict_commit=True):
    """Resolve every dependency. Read-only; never mutates anything."""
    upstreams, skills, blockers = {}, {}, []

    for name, spec in dependencies["upstreams"].items():
        path, source = resolve_upstream_root(name, spec, root)
        commit = head_commit(path) if path.is_dir() else None
        dirty = worktree_dirty(path) if path.is_dir() else None
        entry = {
            "url": spec["url"],
            "expected_ref": spec["ref"],
            "expected_commit": spec["commit"],
            "root": str(path),
            "source": source,
            "present": path.is_dir(),
            "commit": commit,
            "dirty": dirty,
        }
        if not path.is_dir():
            entry["status"] = "missing"
        elif commit is None:
            entry["status"] = "not-a-git-checkout"
        elif strict_commit and commit != spec["commit"]:
            entry["status"] = "commit-mismatch"
        elif dirty and strict_commit:
            entry["status"] = "dirty-worktree"
        else:
            entry["status"] = "ready"
            entry["pinned"] = bool(
                commit == spec["commit"] and not dirty
            )
        upstreams[name] = entry

    for name, spec in dependencies["skills"].items():
        up = spec["upstream"]
        up_entry = upstreams[up]
        up_spec = dependencies["upstreams"][up]
        override = os.environ.get(spec["path_env"])
        if override:
            skill_dir = Path(override).expanduser().resolve()
            source = f"env:{spec['path_env']}"
        else:
            skill_dir = Path(up_entry["root"]) / up_spec["skills_path"] / name
            source = up_entry["source"]

        entrypoints, missing = {}, []
        for key, rel in spec["entrypoints"].items():
            target = skill_dir / rel
            entrypoints[key] = str(target)
            if not target.is_file():
                missing.append(rel)

        resources, missing_resources = {}, []
        for rel in spec.get("resources", []):
            target = skill_dir / rel
            resources[rel] = str(target)
            if not target.exists():
                missing_resources.append(rel)

        has_skill_md = (skill_dir / "SKILL.md").is_file()
        entry = {
            "upstream": up,
            "skill_dir": str(skill_dir),
            "source": source,
            "present": skill_dir.is_dir(),
            "skill_md": str(skill_dir / "SKILL.md") if has_skill_md else None,
            "entrypoints": entrypoints,
            "resources": resources,
            "adapter": spec.get("adapter"),
            "runner_env": spec.get("runner_env", {}),
            "agent_only": bool(spec.get("agent_only")),
        }
        # A per-skill override is trusted only as far as it can be verified. It
        # is never assumed to sit at the pinned commit.
        if override:
            override_commit = head_commit(skill_dir)
            override_dirty = worktree_dirty(skill_dir)
            pinned = bool(
                override_commit == up_spec["commit"] and not override_dirty
            )
            entry["commit"] = override_commit
            entry["dirty"] = override_dirty
            entry["pinned"] = pinned
            if not pinned:
                entry["unpinned_reason"] = (
                    "per-skill override is not at the pinned commit"
                    if override_commit != up_spec["commit"]
                    else "per-skill override has local modifications"
                )
        else:
            entry["pinned"] = bool(up_entry.get("pinned"))
            if not entry["pinned"]:
                entry["unpinned_reason"] = f"upstream {up} is {up_entry['status']}"

        if not skill_dir.is_dir():
            entry["status"] = "missing"
            blockers.append(
                f"{name}: not found at {skill_dir}. It is owned by {up_spec['url']} "
                f"at ref {up_spec['ref']}. Set {spec['path_env']}=<skill dir>, or "
                f"set {up_spec['root_env']}=<checkout>, or run: "
                f"doctor.py dependencies --prepare"
            )
        else:
            problems = []
            if missing:
                problems.append(
                    f"{name}: missing entrypoint(s) {', '.join(missing)} under "
                    f"{skill_dir}. Expected from {up_spec['url']} at "
                    f"{up_spec['ref']}. Override with {spec['path_env']}=<skill dir> "
                    f"or re-run --prepare"
                )
            if missing_resources:
                problems.append(
                    f"{name}: missing declared resource(s) "
                    f"{', '.join(missing_resources)} under {skill_dir}. "
                    f"Expected from {up_spec['url']} at {up_spec['ref']}."
                )
            if not has_skill_md:
                problems.append(
                    f"{name}: {skill_dir} has no SKILL.md, so its own contract "
                    f"cannot be read. This does not look like the upstream skill. "
                    f"Point {spec['path_env']} at the real skill directory."
                )
            if problems:
                entry["status"] = "incomplete"
                blockers.extend(problems)
            elif override is None and up_entry["status"] != "ready":
                entry["status"] = "upstream-" + up_entry["status"]
                if up_entry["status"] == "dirty-worktree":
                    blockers.append(
                        f"{name}: upstream {up} at {up_entry['root']} has local "
                        f"modifications, so runtime code differs from "
                        f"{up_spec['commit'][:12]}. Commit, stash, or re-run "
                        f"--prepare; or accept it with --allow-commit-drift."
                    )
                elif up_entry["status"] == "commit-mismatch":
                    blockers.append(
                        f"{name}: upstream {up} requires {up_spec['ref']} "
                        f"({up_spec['commit'][:12]}), found "
                        f"{(up_entry['commit'] or '?')[:12]} at {up_entry['root']}"
                    )
                else:
                    blockers.append(
                        f"{name}: upstream {up} is {up_entry['status']} at "
                        f"{up_entry['root']}, so its commit cannot be checked "
                        f"against {up_spec['commit'][:12]}. Expected "
                        f"{up_spec['url']} at {up_spec['ref']}. Set "
                        f"{spec['path_env']}=<skill dir>, set "
                        f"{up_spec['root_env']}=<checkout>, or run: "
                        f"{Path(__file__).name} dependencies --prepare"
                    )
            elif strict_commit and not entry["pinned"]:
                entry["status"] = "unpinned"
                blockers.append(f"{name}: {entry['unpinned_reason']}")
            else:
                entry["status"] = "ready"
        skills[name] = entry

    return upstreams, skills, blockers


def build_manifest(dependencies, root, upstreams, skills, blockers, drift_allowed=False):
    return {
        "schema": MANIFEST_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "upstream_root": str(root),
        "ready": not blockers,
        "pinned": all(u.get("pinned") for u in upstreams.values()),
        "drift_allowed": drift_allowed,
        "blockers": blockers,
        "upstreams": upstreams,
        "skills": skills,
    }


def env_lines(manifest):
    """Shell assignments the runner and references source. Paths only."""
    lines = [
        f"export {MANIFEST_ENV}="
        f"{shlex.quote(str(manifest.get('manifest_path', '')))}",
        f"export {ROOT_ENV}={shlex.quote(manifest['upstream_root'])}",
    ]
    for name, entry in sorted(manifest["upstreams"].items()):
        var = name.upper() + "_UPSTREAM_ROOT"
        lines.append(f"export {var}={shlex.quote(entry['root'])}")
    for name, entry in sorted(manifest["skills"].items()):
        var = name.upper().replace("-", "_") + "_SKILL_DIR"
        lines.append(f"export {var}={shlex.quote(entry['skill_dir'])}")
        for key, path in sorted(entry["entrypoints"].items()):
            runner = (entry.get("runner_env") or {}).get(key)
            if runner:
                lines.append(f"export {runner}={shlex.quote(path)}")
    return "\n".join(lines) + "\n"


def prepare(dependencies, root, upstreams, only=None):
    """Create or update managed checkouts. Never touches an overridden path."""
    changed = []
    for name, spec in dependencies["upstreams"].items():
        if only and name not in only:
            continue
        entry = upstreams[name]
        if entry["source"].startswith("env:"):
            print(f"  skip {name}: supplied through {entry['source']}, not managed here")
            continue
        path = Path(entry["root"])
        if path.exists() and any(path.iterdir()) and not (path / ".git").is_dir():
            raise ValueError(f"refusing nonempty unmanaged checkout directory: {path}")
        if (path / ".git").is_dir():
            if _git(["remote", "get-url", "origin"], path) != spec["url"]:
                raise ValueError(f"unexpected origin in managed checkout: {path}")
            if _git(["status", "--porcelain"], path):
                raise ValueError(f"refusing to update dirty managed checkout: {path}")
        path.mkdir(parents=True, exist_ok=True)
        if not (path / ".git").is_dir():
            print(f"  init {name} at {path}")
            _git(["init", "--quiet", "."], path)
            _git(["remote", "add", "origin", spec["url"]], path)
        try:
            _git(["fetch", "--quiet", "--depth", "1", "origin", spec["commit"]], path)
        except subprocess.CalledProcessError:
            _git(["fetch", "--quiet", "origin", spec["ref"]], path)
        _git(["checkout", "--quiet", spec["commit"]], path)
        print(f"  ready {name} at {path} ({spec['commit'][:12]})")
        changed.append(name)
    return changed




def run_dependencies(args):
    """Resolve the upstream skills and write the manifest the runner consumes."""
    if not DEPENDENCIES.is_file():
        print(f"error: {DEPENDENCIES} not found", file=sys.stderr)
        return EXIT_CONFIG
    dependencies = json.loads(DEPENDENCIES.read_text())
    root = Path(args.upstream_root).expanduser().resolve() if args.upstream_root else default_root().expanduser().resolve()

    upstreams, skills, blockers = resolve(
        dependencies, root, strict_commit=not args.allow_commit_drift
    )

    if args.prepare:
        if args.check_only:
            print("error: --prepare and --check-only are mutually exclusive", file=sys.stderr)
            return EXIT_CONFIG
        print(f"Preparing managed checkouts under {root}")
        prepare(dependencies, root, upstreams)
        upstreams, skills, blockers = resolve(
            dependencies, root, strict_commit=not args.allow_commit_drift
        )

    manifest = build_manifest(
        dependencies, root, upstreams, skills, blockers, args.allow_commit_drift
    )

    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else default_manifest_path().resolve()
    manifest["manifest_path"] = str(manifest_path)

    if not args.check_only:
        atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if args.env_file:
            atomic_write(Path(args.env_file).expanduser(), env_lines(manifest))

    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"upstream root: {root}")
        for name, entry in sorted(skills.items()):
            print(f"  {entry['status']:<22} {name:<26} {entry['skill_dir']}")
        if not args.check_only:
            print(f"manifest: {manifest_path}")
        if blockers:
            print("\nBlockers:", file=sys.stderr)
            for line in blockers:
                print(f"  - {line}", file=sys.stderr)

    return EXIT_BLOCKED if blockers else 0

def build_parser():
    """Build the dependency preflight CLI."""
    parser = argparse.ArgumentParser(
        prog="doctor.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    phases = parser.add_subparsers(dest="phase", required=True, metavar="PHASE")

    deps = phases.add_parser(
        "dependencies",
        aliases=["deps"],
        help="resolve the upstream skills and write the dependency manifest",
        description=(
            "Resolve the upstream skills this humanoid workflow orchestrates and write "
            "the manifest the runner and the adapters consume. Exits 3 when a "
            "dependency is missing, unpinned, or incomplete."
        ),
    )
    deps.add_argument("--manifest", help="Where to write the dependency manifest.")
    deps.add_argument("--env-file", help="Also write sourceable shell assignments here.")
    deps.add_argument("--upstream-root", help="Normalized checkout root override.")
    deps.add_argument("--prepare", action="store_true",
                      help="Create or update managed checkouts. Not read-only.")
    deps.add_argument("--check-only", action="store_true",
                      help="Resolve and report; write nothing.")
    deps.add_argument(
        "--allow-commit-drift",
        action="store_true",
        help=(
            "Accept a checkout whose commit differs from the pin or whose "
            "worktree is dirty. The manifest is marked unpinned and "
            "drift_allowed, and the runtime always refuses it."
        ),
    )
    deps.add_argument("--json", action="store_true", help="Print the manifest to stdout.")
    deps.set_defaults(run=run_dependencies)

    return parser


def main(argv=None):
    """Resolve dependencies without launching any demo services."""
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"dependency preparation failed: {exc}", file=sys.stderr)
        return EXIT_CONFIG


if __name__ == "__main__":
    raise SystemExit(main())
