#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Select which shipped VSS skills are active, from what the vss CLI knows about
# the deployment. Harness-neutral port of the OpenClaw plugin's sync.ts (which
# it replaces): the OpenClaw plugin, the Hermes image, the harness Dockerfiles,
# and host-side tooling all call this one implementation, so the selection
# contract cannot drift between harnesses.
#
# "Active" means "present in --active-dir": OpenClaw reads a plugin's skills
# from its static manifest path (skills-active/), Hermes scans its writable
# skill root ($HERMES_HOME/skills/vss) natively, and both are just directories
# this script fills. Everything shipped stays in --skills-dir; this only
# decides what the agent sees.
#
# Each shipped skill declares what it needs in its own SKILL.md frontmatter,
# `vss-requires`, as the vss CLI names it:
#   a vss command group  -> active when `vss configure check` lists the group
#                           as available (the CLI joins what the deployment
#                           exposes with what each group needs)
#   alerts               -> active when the Alert Bridge answers at
#                           <base_url>/alert-bridge/health (the ingress route
#                           strips the prefix; the service has no root route,
#                           only /health, /ready, /metrics) or <host>:9080/health
#   always               -> active on every deployment
# Several may be listed, space-separated; all must hold. `a|b` holds when any
# of its alternatives does (e.g. `summarize|vlm`).
# No recorded deployment (vss configure never ran) -> everything active, so
# the agent can still configure. --all forces that.
#
# Exit codes: 0 = at least one skill active; 3 = none active; 1 = error.

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

PROBE_TIMEOUT_SEC = 5
CLI_TIMEOUT_SEC = 60

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def _default_run(cmd: list[str], timeout: int) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


@dataclass
class SkillSpec:
    name: str
    needs: list[str]


@dataclass
class Selection:
    active: list[str]
    inactive: dict[str, str]
    reason: str


@dataclass
class Availability:
    configured: bool
    available: set = field(default_factory=set)
    base_url: str = ""
    error: str = ""


def skill_requires(skill_md: Path) -> list[str]:
    """`vss-requires` from a SKILL.md frontmatter; empty when undeclared."""
    text = skill_md.read_text(encoding="utf-8")
    fm = re.match(r"^---\r?\n([\s\S]*?)\r?\n---", text)
    if not fm:
        return []
    m = re.search(r"^\s*vss-requires:\s*[\"']?([^\"'\n]*)[\"']?\s*$", fm.group(1), re.M)
    if not m:
        return []
    return [n for n in m.group(1).strip().split() if n and n != "always"]


def read_skill_specs(skills_dir: Path) -> list[SkillSpec]:
    specs = []
    for entry in sorted(skills_dir.iterdir()) if skills_dir.is_dir() else []:
        skill_md = entry / "SKILL.md"
        if skill_md.is_file():
            specs.append(SkillSpec(entry.name, skill_requires(skill_md)))
    return specs


def command_availability(vss_bin: str, run: Runner = _default_run) -> Availability:
    """`vss configure check`: which command groups the recorded deployment can serve."""
    try:
        r = run([vss_bin, "configure", "check"], timeout=CLI_TIMEOUT_SEC)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Availability(configured=False, error=f"{vss_bin}: {exc}")
    out = f"{r.stdout or ''}\n{r.stderr or ''}"
    # The CLI's exact wording for "vss configure never ran".
    if re.search(r"no deployment configured", out, re.I):
        return Availability(configured=False)
    if not re.search(r"^commands:", out, re.M):
        first = next((line.strip() for line in out.split("\n") if line.strip()), f"exit {r.returncode}")
        return Availability(configured=False, error=first)
    available = set()
    in_commands = False
    for line in out.split("\n"):
        if re.match(r"^commands:", line):
            in_commands = True
            continue
        if not in_commands:
            continue
        m = re.match(r"^\s+(\S+)\s+(available|unavailable)\b", line)
        if m and m.group(2) == "available":
            available.add(m.group(1))
    base = re.search(r"against (\S+)", out)
    return Availability(configured=True, available=available, base_url=base.group(1) if base else "")


def http_answers(url: str, run: Runner = _default_run) -> bool:
    """curl is in every runtime; any HTTP status but 404 means "something is
    routed there", the vss CLI's own presence rule for ingress probes."""
    try:
        r = run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             "--max-time", str(PROBE_TIMEOUT_SEC), url],
            timeout=PROBE_TIMEOUT_SEC + 5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    try:
        code = int((r.stdout or "").strip())
    except ValueError:
        return False
    return code > 0 and code != 404


def alerts_available(base_url: str, run: Runner = _default_run) -> bool:
    if not base_url:
        return False
    if http_answers(f"{base_url.rstrip('/')}/alert-bridge/health", run):
        return True
    try:
        u = urlsplit(base_url)
        if not u.scheme or not u.hostname:
            return False
        return http_answers(f"{u.scheme}://{u.hostname}:9080/health", run)
    except ValueError:
        return False


def select(specs: list[SkillSpec], *, all_skills: bool = False, vss_bin: str = "vss",
           run: Runner = _default_run) -> Selection:
    if all_skills:
        return Selection([s.name for s in specs], {}, "all shipped skills (forced)")
    availability = command_availability(vss_bin, run)
    if not availability.configured:
        # Not knowing the deployment is not a reason to hide skills: every one
        # of them starts with `vss configure check` and can recover from there.
        why = (f"vss configure check failed ({availability.error})"
               if availability.error else "no deployment recorded by `vss configure`")
        return Selection([s.name for s in specs], {}, f"{why}; all shipped skills active")
    alerts = (alerts_available(availability.base_url, run)
              if any("alerts" in n.split("|") for s in specs for n in s.needs) else False)
    active: list[str] = []
    inactive: dict[str, str] = {}
    for s in specs:
        missing = [n for n in s.needs
                   if not any(alerts if a == "alerts" else a in availability.available
                              for a in n.split("|"))]
        if not missing:
            active.append(s.name)
        else:
            inactive[s.name] = "; ".join(
                "alert-bridge not reachable" if n == "alerts"
                else f"vss command group '{n}' unavailable" if "|" not in n
                else f"none of vss command groups '{n}' available"
                for n in missing
            )
    listed = ", ".join(sorted(availability.available)) or "none"
    return Selection(active, inactive, f"deployment {availability.base_url}: commands available = {listed}")


def apply_active(skills_dir: Path, active_dir: Path, active: list[str]) -> None:
    """Make --active-dir hold exactly `active` (real copies: OpenClaw skips
    symlinked skill dirs, and Hermes state tooling expects real trees)."""
    active_dir.mkdir(parents=True, exist_ok=True)
    want = set(active)
    for present in active_dir.iterdir():
        if present.name not in want:
            shutil.rmtree(present, ignore_errors=False) if present.is_dir() else present.unlink()
    for name in active:
        src = skills_dir / name
        if not src.exists():
            raise FileNotFoundError(f"{src} is missing")
        dst = active_dir / name
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        shutil.copytree(src, dst)


def sync(skills_dir: Path, active_dir: Path, *, all_skills: bool = False,
         vss_bin: str = "vss", run: Runner = _default_run,
         log: Callable[[str], None] = print) -> Selection:
    specs = read_skill_specs(skills_dir)
    sel = select(specs, all_skills=all_skills, vss_bin=vss_bin, run=run)
    apply_active(skills_dir, active_dir, sel.active)
    off = ", ".join(f"{k} ({v})" for k, v in sel.inactive.items())
    log(f"[vss] skills active: {', '.join(sel.active) or 'none'}"
        f"{f'; inactive: {off}' if off else ''} — {sel.reason}")
    return sel


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sync active VSS skills from deployment capabilities.")
    ap.add_argument("--skills-dir", type=Path, help="shipped skill set (read-only)")
    ap.add_argument("--active-dir", type=Path, help="directory to hold exactly the active skills")
    ap.add_argument("--plugin-dir", type=Path,
                    help="legacy OpenClaw layout: skills-dir=<dir>/skills, active-dir=<dir>/skills-active")
    ap.add_argument("--all", action="store_true", help="activate every shipped skill (no probing)")
    ap.add_argument("--vss", default="vss", help="vss CLI binary (default: vss)")
    a = ap.parse_args(argv)
    skills_dir, active_dir = a.skills_dir, a.active_dir
    if a.plugin_dir:
        skills_dir = skills_dir or a.plugin_dir / "skills"
        active_dir = active_dir or a.plugin_dir / "skills-active"
    if not skills_dir or not active_dir:
        ap.error("need --skills-dir and --active-dir (or --plugin-dir)")
    try:
        sel = sync(skills_dir, active_dir, all_skills=a.all, vss_bin=a.vss)
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"[vss] sync failed: {exc}", file=sys.stderr)
        return 1
    return 0 if sel.active else 3


if __name__ == "__main__":
    sys.exit(main())
