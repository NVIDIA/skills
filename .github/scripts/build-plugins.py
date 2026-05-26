#!/usr/bin/env python3
"""
Build plugin packages from plugins.d/ and per-plugin .skills-manifest.yml files.

Inputs (hand-maintained):
  plugins.d/_defaults.yml
      Shared default fields applied to every catalog plugin (author, license,
      homepage, brand color, policy URLs, etc.). Per-plugin yaml fields
      override these. Filenames starting with `_` are treated as includes and
      are never built into a plugin themselves.

  plugins.d/<name>.yml
      Catalog plugin spec. Drives full regeneration of plugins/<name>/
      (skills/ tree, .claude-plugin/plugin.json, .codex-plugin/plugin.json)
      and its entry in both marketplace.json files.

  plugins/<name>/.skills-manifest.yml
      Per-plugin skill manifest for hand-curated plugins. Only the skills/
      tree is rebuilt; plugin.jsons, assets, README, and the marketplace
      entry stay hand-edited.

  plugins/<name>/assets/<file>      Hand-maintained logo/image assets.
  plugins/<name>/README.md          Optional plugin-specific notes.

Generated outputs (committed):
  plugins/<name>/skills/<skill>/    Real dir or symlink (see below)
  plugins/<name>/.claude-plugin/plugin.json   (catalog plugins only)
  plugins/<name>/.codex-plugin/plugin.json    (catalog plugins only)
  .claude-plugin/marketplace.json   (catalog plugin entries; others preserved)
  .agents/plugins/marketplace.json  (catalog plugin entries; others preserved)

How the skills/ tree gets populated is selected by `skill_files:` in
plugins.d/<name>.yml (defaults to `copy` via plugins.d/_defaults.yml):

  copy     rsync each curated skill into plugins/<name>/skills/<skill>/.
           Real files. Required for `codex plugin add` (Codex 0.132
           silently drops symlinks during install).
  symlink  Relative symlink plugins/<name>/skills/<skill> →
           ../../../skills/<Product>/<skill>. No duplicated SKILL.md;
           the plugin tree stays in sync with the canonical catalog
           automatically. Works for `claude plugin install` and
           `npx skills add`. NOT compatible with Codex local install.

Usage:
  build-plugins.py             Build everything.
  build-plugins.py --check     Build, then fail if the working tree changed
                               (CI drift guard).
  build-plugins.py --only NAME Restrict to a single plugin name.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGINS_D = REPO_ROOT / "plugins.d"
PLUGINS_D_DEFAULTS = PLUGINS_D / "_defaults.yml"
PLUGINS_DIR = REPO_ROOT / "plugins"
SKILLS_DIR = REPO_ROOT / "skills"
CLAUDE_MARKETPLACE = REPO_ROOT / ".claude-plugin" / "marketplace.json"
AGENTS_MARKETPLACE = REPO_ROOT / ".agents" / "plugins" / "marketplace.json"

# Default policy block written for every plugin entry in
# .agents/plugins/marketplace.json. Mirrors the existing convention.
AGENTS_PLUGIN_POLICY = {
    "installation": "AVAILABLE",
    "authentication": "ON_INSTALL",
}


# ---------------------------- helpers ----------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        die(f"{path}: expected a YAML mapping at top level")
    return data


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def make_relative_symlink(target: Path, link: Path) -> None:
    """Create a symlink at `link` pointing at `target`, using a path relative
    to `link`'s parent directory so the link survives repo relocation."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    rel = os.path.relpath(target, link.parent)
    os.symlink(rel, link, target_is_directory=True)


def expand_skill_paths(entries: list[str]) -> list[tuple[str, Path]]:
    """
    Expand each entry into a list of (skill_basename, absolute_source_path).

    - "skills/cuopt/cuopt-install/"   → [("cuopt-install", REPO/skills/cuopt/cuopt-install)]
    - "skills/cuopt/"                 → all immediate children of skills/cuopt that contain SKILL.md
    """
    out: list[tuple[str, Path]] = []
    for entry in entries:
        rel = entry.rstrip("/")
        src = (REPO_ROOT / rel).resolve()
        if not src.exists():
            die(f"include_skills entry not found: {entry}")
        if not src.is_dir():
            die(f"include_skills entry is not a directory: {entry}")
        skill_md = src / "SKILL.md"
        if skill_md.is_file():
            out.append((src.name, src))
        else:
            children = sorted(p for p in src.iterdir() if p.is_dir())
            found = 0
            for child in children:
                if (child / "SKILL.md").is_file():
                    out.append((child.name, child))
                    found += 1
            if found == 0:
                die(f"no SKILL.md found under {entry}")
    seen: dict[str, Path] = {}
    for name, src in out:
        if name in seen and seen[name] != src:
            die(f"duplicate skill name '{name}': {seen[name]} vs {src}")
        seen[name] = src
    return out


VALID_MATERIALIZATIONS = ("copy", "symlink")


def rsync_dir(src: Path, dst: Path) -> None:
    """rsync -a --delete src/ dst/ ; src must exist as a directory."""
    if not src.is_dir():
        die(f"source is not a directory: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-a", "--delete", f"{src}/", f"{dst}/"],
        check=True,
    )


def materialize_skills(
    plugin_name: str,
    entries: list[str],
    mode: str = "copy",
) -> list[str]:
    """Replace plugins/<name>/skills/ with the curated subset. Returns skill names.

    `mode` selects materialization strategy:
      - "copy"    rsync each skill into plugins/<name>/skills/<skill>/.
                  Real files. Larger repo. Required for Codex local-install
                  (codex plugin add silently drops symlinks).
      - "symlink" relative symlinks under plugins/<name>/skills/<skill> →
                  ../../../skills/<Product>/<skill>. No duplicate SKILL.md.
                  Works for Claude install and `npx skills add`. Currently
                  NOT supported by Codex local marketplace install.
    """
    if mode not in VALID_MATERIALIZATIONS:
        die(
            f"plugin {plugin_name!r}: invalid skill_files={mode!r} "
            f"(allowed: {', '.join(VALID_MATERIALIZATIONS)})"
        )
    plugin_dir = PLUGINS_DIR / plugin_name
    target = plugin_dir / "skills"
    if target.is_symlink() or target.exists():
        if target.is_symlink():
            target.unlink()
        else:
            shutil.rmtree(target)
    target.mkdir(parents=True)

    pairs = expand_skill_paths(entries)
    names: list[str] = []
    for name, src in pairs:
        dst = target / name
        if mode == "symlink":
            make_relative_symlink(src, dst)
        else:  # copy
            rsync_dir(src, dst)
        names.append(name)
    if mode == "symlink":
        word = "symlink" if len(names) == 1 else "symlinks"
    else:
        word = "copy" if len(names) == 1 else "copies"
    log(f"  ✓ skills/  ({len(names)} {word})")
    return names


# ---------------------------- catalog plugin ---------------------------------


def render_claude_plugin_json(spec: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": spec["name"],
        "version": str(spec.get("version", "1.0.0")),
        "description": spec["description"],
    }
    if "display_name" in spec:
        out["displayName"] = spec["display_name"]
    if "author" in spec:
        out["author"] = spec["author"]
    if "homepage" in spec:
        out["homepage"] = spec["homepage"]
    if "repository" in spec:
        out["repository"] = spec["repository"]
    if "license" in spec:
        out["license"] = spec["license"]
    if "keywords" in spec:
        out["keywords"] = list(spec["keywords"])
    # Skills tree is always inside the plugin (Codex rejects ".." paths,
    # Anthropic spec requires "./" prefixes). Container path; Claude scans
    # immediate children for SKILL.md.
    out["skills"] = ["./skills/"]
    return out


def render_codex_plugin_json(spec: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": spec["name"],
        "version": str(spec.get("version", "1.0.0")),
        "description": spec["description"],
    }
    if "author" in spec:
        out["author"] = spec["author"]
    if "homepage" in spec:
        out["homepage"] = spec["homepage"]
    if "repository" in spec:
        out["repository"] = spec["repository"]
    if "license" in spec:
        out["license"] = spec["license"]
    if "keywords" in spec:
        out["keywords"] = list(spec["keywords"])

    # Skills tree is always inside the plugin (Codex rejects ".." paths).
    out["skills"] = "./skills/"

    interface: dict[str, Any] = {}
    if "display_name" in spec:
        interface["displayName"] = spec["display_name"]
    if "short_description" in spec:
        interface["shortDescription"] = spec["short_description"]
    if "long_description" in spec:
        interface["longDescription"] = spec["long_description"]
    if "author" in spec and "name" in spec["author"]:
        interface["developerName"] = spec["author"]["name"]
    if "category" in spec:
        interface["category"] = spec["category"]
    if "capabilities" in spec:
        interface["capabilities"] = list(spec["capabilities"])
    for src_key, dst_key in (
        ("website_url", "websiteURL"),
        ("privacy_policy_url", "privacyPolicyURL"),
        ("terms_of_service_url", "termsOfServiceURL"),
    ):
        value = spec.get(src_key)
        if value is None or value == "":
            continue
        interface[dst_key] = value
    if "logo" in spec:
        interface["logo"] = spec["logo"]
    if "composer_icon" in spec:
        interface["composerIcon"] = spec["composer_icon"]
    if "screenshots" in spec:
        interface["screenshots"] = list(spec["screenshots"])
    if "brand_color" in spec:
        interface["brandColor"] = spec["brand_color"]
    if "default_prompts" in spec:
        interface["defaultPrompt"] = list(spec["default_prompts"])
    if interface:
        out["interface"] = interface
    return out


def build_catalog_plugin(spec: dict[str, Any]) -> str:
    name = spec["name"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        die(f"plugin name must be lowercase kebab-case: {name!r}")
    plugin_dir = PLUGINS_DIR / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    log(f"── catalog plugin: {name} ──")

    mode = spec.get("skill_files", "copy")
    materialize_skills(name, spec.get("include_skills", []), mode=mode)

    write_json(plugin_dir / ".claude-plugin" / "plugin.json", render_claude_plugin_json(spec))
    log("  ✓ .claude-plugin/plugin.json")
    write_json(plugin_dir / ".codex-plugin" / "plugin.json", render_codex_plugin_json(spec))
    log("  ✓ .codex-plugin/plugin.json")

    asset_fields = ["logo", "composer_icon"]
    asset_paths: list[str] = [spec[f] for f in asset_fields if isinstance(spec.get(f), str)]
    asset_paths += [s for s in (spec.get("screenshots") or []) if isinstance(s, str)]
    for asset in asset_paths:
        if not asset.startswith("./"):
            continue
        rel = asset[2:]
        path = plugin_dir / rel
        if not path.is_file():
            log(f"  ! warning: asset declared but missing on disk: {path.relative_to(REPO_ROOT)}")

    return name


# ---------------------------- curated plugin ---------------------------------


def build_curated_plugin(plugin_dir: Path) -> str:
    """
    A curated plugin is one with plugins/<name>/.skills-manifest.yml but
    no plugins.d/<name>.yml. We rebuild only its skills/ tree (mode
    selected by `skill_files:` in the manifest, default `copy`) and
    trust everything else to be hand-maintained.
    """
    manifest_path = plugin_dir / ".skills-manifest.yml"
    spec = read_yaml(manifest_path)
    name = plugin_dir.name
    log(f"── curated plugin: {name} ──")
    skills = spec.get("skills") or []
    if not skills:
        die(f"{manifest_path}: 'skills' list is empty")
    mode = spec.get("skill_files", "copy")
    materialize_skills(name, skills, mode=mode)
    return name


# ---------------------------- marketplaces -----------------------------------


def is_marketplace_enabled(spec: dict[str, Any], marketplace_key: str) -> bool:
    flags = spec.get("marketplace_enabled") or {}
    return bool(flags.get(marketplace_key, True))


def upsert_claude_marketplace(
    catalog_specs: dict[str, dict[str, Any]],
    curated_names: set[str],
) -> None:
    if not CLAUDE_MARKETPLACE.is_file():
        die(f"missing {CLAUDE_MARKETPLACE.relative_to(REPO_ROOT)}")
    data = read_json(CLAUDE_MARKETPLACE)
    existing = data.get("plugins", [])
    managed = set(catalog_specs.keys())

    new_plugins: list[dict[str, Any]] = []
    for entry in existing:
        name = entry.get("name")
        if name in managed:
            continue
        if name in curated_names:
            new_plugins.append(entry)
    for name in sorted(catalog_specs):
        spec = catalog_specs[name]
        if not is_marketplace_enabled(spec, "claude"):
            continue
        new_plugins.append({
            "name": name,
            "source": f"./plugins/{name}",
            "description": spec["description"],
        })
    new_plugins.sort(key=lambda p: p.get("name", ""))
    data["plugins"] = new_plugins
    write_json(CLAUDE_MARKETPLACE, data)
    log(f"  ✓ {CLAUDE_MARKETPLACE.relative_to(REPO_ROOT)} ({len(new_plugins)} plugin(s))")


def upsert_agents_marketplace(
    catalog_specs: dict[str, dict[str, Any]],
    curated_names: set[str],
) -> None:
    if not AGENTS_MARKETPLACE.is_file():
        die(f"missing {AGENTS_MARKETPLACE.relative_to(REPO_ROOT)}")
    data = read_json(AGENTS_MARKETPLACE)
    existing = data.get("plugins", [])
    managed = set(catalog_specs.keys())

    new_plugins: list[dict[str, Any]] = []
    for entry in existing:
        name = entry.get("name")
        if name in managed:
            continue
        if name in curated_names:
            new_plugins.append(entry)
    for name in sorted(catalog_specs):
        spec = catalog_specs[name]
        if not is_marketplace_enabled(spec, "codex"):
            continue
        new_plugins.append({
            "name": name,
            "source": {"source": "local", "path": f"./plugins/{name}"},
            "policy": dict(AGENTS_PLUGIN_POLICY),
            "category": spec.get("category", "Developer Tools"),
        })
    new_plugins.sort(key=lambda p: p.get("name", ""))
    data["plugins"] = new_plugins
    write_json(AGENTS_MARKETPLACE, data)
    log(f"  ✓ {AGENTS_MARKETPLACE.relative_to(REPO_ROOT)} ({len(new_plugins)} plugin(s))")


# ---------------------------- main -------------------------------------------


def merge_with_defaults(defaults: dict[str, Any], plugin: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(plugin)
    return merged


def discover() -> tuple[dict[str, dict[str, Any]], list[Path]]:
    defaults: dict[str, Any] = {}
    if PLUGINS_D_DEFAULTS.is_file():
        defaults = read_yaml(PLUGINS_D_DEFAULTS)

    catalog: dict[str, dict[str, Any]] = {}
    if PLUGINS_D.is_dir():
        for ymlfile in sorted(PLUGINS_D.glob("*.yml")):
            if ymlfile.name.startswith("_"):
                continue
            plugin_spec = read_yaml(ymlfile)
            spec = merge_with_defaults(defaults, plugin_spec)
            name = spec.get("name")
            if not name:
                die(f"{ymlfile}: missing 'name'")
            if name in catalog:
                die(f"duplicate plugin name '{name}' across plugins.d/")
            spec["__source"] = str(ymlfile.relative_to(REPO_ROOT))
            catalog[name] = spec

    curated: list[Path] = []
    if PLUGINS_DIR.is_dir():
        for plugin_dir in sorted(PLUGINS_DIR.iterdir()):
            if not plugin_dir.is_dir():
                continue
            if plugin_dir.name in catalog:
                continue
            if (plugin_dir / ".skills-manifest.yml").is_file():
                curated.append(plugin_dir)
    return catalog, curated


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Build NVIDIA skills plugins.")
    p.add_argument("--only", help="Only build the named plugin.")
    p.add_argument(
        "--check",
        action="store_true",
        help="After building, fail if the working tree changed (CI drift guard).",
    )
    args = p.parse_args(argv)

    catalog, curated = discover()
    if args.only:
        catalog = {k: v for k, v in catalog.items() if k == args.only}
        curated = [d for d in curated if d.name == args.only]
        if not catalog and not curated:
            die(f"no plugin named '{args.only}' found in plugins.d/ or plugins/")

    log(f"Found {len(catalog)} catalog plugin(s) and {len(curated)} curated plugin(s).")

    for name in sorted(catalog):
        build_catalog_plugin(catalog[name])
    for plugin_dir in curated:
        build_curated_plugin(plugin_dir)

    log("── marketplaces ──")
    curated_names = {d.name for d in curated}
    upsert_claude_marketplace(catalog, curated_names)
    upsert_agents_marketplace(catalog, curated_names)

    if args.check:
        log("── drift check ──")
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "plugins/",
                ".claude-plugin/marketplace.json",
                ".agents/plugins/marketplace.json",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(result.stdout, file=sys.stderr)
            die(
                "plugin tree drifted from sources; "
                "run .github/scripts/build-plugins.sh and commit the result"
            )
        log("  ✓ no drift")

    log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
