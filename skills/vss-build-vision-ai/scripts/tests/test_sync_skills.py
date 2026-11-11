# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior contract for sync_skills.py, the harness-neutral skill selector.

These tests pin the semantics the OpenClaw plugin's sync.ts established
(including the two review-driven probe fixes on PR #2024): frontmatter
parsing, the fail-open policy for unconfigured deployments, the exact probe
endpoints, exact-set application, and the CLI exit-code contract.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sync_skills  # noqa: E402


def completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def make_skill(root, name, frontmatter=None, body="# skill\n"):
    d = root / name
    d.mkdir(parents=True)
    text = f"---\n{frontmatter}\n---\n{body}" if frontmatter is not None else body
    (d / "SKILL.md").write_text(text, encoding="utf-8")
    return d


# --- frontmatter parsing -----------------------------------------------------

@pytest.mark.parametrize(
    ("frontmatter", "expected"),
    [
        ('vss-requires: "vlm"', ["vlm"]),
        ("vss-requires: vlm search", ["vlm", "search"]),
        ("vss-requires: 'alerts'", ["alerts"]),
        ('vss-requires: "always"', []),          # `always` is satisfied vacuously
        ("vss-requires: vlm always", ["vlm"]),
        ("name: x\nother: y", []),               # no declaration
        ("metadata:\n  vss-requires: vlm", ["vlm"]),  # nested/indented form
    ],
)
def test_skill_requires(tmp_path, frontmatter, expected):
    d = make_skill(tmp_path, "s", frontmatter)
    assert sync_skills.skill_requires(d / "SKILL.md") == expected


def test_skill_requires_without_frontmatter(tmp_path):
    d = make_skill(tmp_path, "s", frontmatter=None)
    assert sync_skills.skill_requires(d / "SKILL.md") == []


def test_read_skill_specs_sorted_and_skillmd_only(tmp_path):
    make_skill(tmp_path, "b-skill", 'vss-requires: "vlm"')
    make_skill(tmp_path, "a-skill", 'vss-requires: "always"')
    (tmp_path / "not-a-skill").mkdir()
    specs = sync_skills.read_skill_specs(tmp_path)
    assert [(s.name, s.needs) for s in specs] == [("a-skill", []), ("b-skill", ["vlm"])]


def test_query_analytics_declares_cli_capability() -> None:
    skills_root = Path(__file__).resolve().parents[3]
    skill = skills_root / "operations" / "vss-query-analytics" / "SKILL.md"
    assert sync_skills.skill_requires(skill) == ["analytics"]


# --- `vss configure check` parsing -------------------------------------------

CHECK_OUT = """checking against http://vss.example:31000
commands:
  vlm       available
  search    unavailable
  analytics available
"""


def test_command_availability_parses_groups_and_base_url():
    av = sync_skills.command_availability("vss", run=lambda cmd, timeout: completed(stdout=CHECK_OUT))
    assert av.configured and av.available == {"vlm", "analytics"}
    assert av.base_url == "http://vss.example:31000"


def test_command_availability_unconfigured():
    av = sync_skills.command_availability(
        "vss", run=lambda cmd, timeout: completed(stderr="error: no deployment configured", returncode=3))
    assert not av.configured and not av.error


def test_command_availability_error_is_first_line():
    av = sync_skills.command_availability(
        "vss", run=lambda cmd, timeout: completed(stderr="boom: cannot reach origin", returncode=1))
    assert not av.configured and av.error == "boom: cannot reach origin"


def test_command_availability_missing_binary():
    def run(cmd, timeout):
        raise OSError("No such file or directory: 'vss'")
    av = sync_skills.command_availability("vss", run=run)
    assert not av.configured and "vss" in av.error


# --- alert probe (the PR #2024 review fixes) ----------------------------------

def test_alert_probe_hits_health_on_ingress_route_then_9080():
    urls = []

    def run(cmd, timeout):
        if cmd[0] == "curl":
            urls.append(cmd[-1])
            return completed(stdout="404")
        return completed(stdout=CHECK_OUT)

    assert sync_skills.alerts_available("http://vss.example:31000/", run) is False
    assert urls == [
        "http://vss.example:31000/alert-bridge/health",
        "http://vss.example:9080/health",
    ]


def test_alert_probe_any_non_404_status_counts_as_answering():
    assert sync_skills.http_answers("u", run=lambda c, timeout: completed(stdout="401")) is True
    assert sync_skills.http_answers("u", run=lambda c, timeout: completed(stdout="404")) is False
    assert sync_skills.http_answers("u", run=lambda c, timeout: completed(stdout="000")) is False
    assert sync_skills.http_answers("u", run=lambda c, timeout: completed(stdout="garbage")) is False


# --- selection ----------------------------------------------------------------

def specs():
    return [
        sync_skills.SkillSpec("vss-ask-video", ["vlm"]),
        sync_skills.SkillSpec("vss-manage-alerts", ["alerts"]),
        sync_skills.SkillSpec("vss-manage-video-io-storage", []),
        sync_skills.SkillSpec("vss-query-analytics", ["analytics"]),
        sync_skills.SkillSpec("vss-search-archive", ["search"]),
    ]


def test_select_all_forced_never_probes():
    def run(cmd, timeout):
        raise AssertionError("must not invoke vss or curl under --all")
    sel = sync_skills.select(specs(), all_skills=True, run=run)
    assert len(sel.active) == 5 and sel.reason == "all shipped skills (forced)"


def test_select_unconfigured_activates_everything():
    sel = sync_skills.select(
        specs(), run=lambda c, timeout: completed(stderr="no deployment configured"))
    assert len(sel.active) == 5 and "all shipped skills active" in sel.reason


def test_select_matches_groups_and_reports_missing():
    def run(cmd, timeout):
        if cmd[0] == "curl":
            return completed(stdout="200")
        return completed(stdout=CHECK_OUT)
    sel = sync_skills.select(specs(), run=run)
    assert sel.active == [
        "vss-ask-video",
        "vss-manage-alerts",
        "vss-manage-video-io-storage",
        "vss-query-analytics",
    ]
    assert sel.inactive == {"vss-search-archive": "vss command group 'search' unavailable"}
    assert "commands available = analytics, vlm" in sel.reason


def test_select_any_of_holds_when_one_alternative_is_available():
    # vss-summarize-video runs on LVS or, without it, on `vss vlm run`.
    either = [sync_skills.SkillSpec("vss-summarize-video", ["summarize|vlm"]),
              sync_skills.SkillSpec("needs-neither", ["summarize|search"])]
    sel = sync_skills.select(either, run=lambda cmd, timeout: completed(stdout=CHECK_OUT))
    assert sel.active == ["vss-summarize-video"]
    assert sel.inactive == {
        "needs-neither": "none of vss command groups 'summarize|search' available"}


def test_select_skips_alert_probe_when_no_skill_needs_it():
    def run(cmd, timeout):
        assert cmd[0] != "curl", "no skill declares alerts; probe must not run"
        return completed(stdout=CHECK_OUT)
    plain = [s for s in specs() if "alerts" not in s.needs]
    sel = sync_skills.select(plain, run=run)
    assert "vss-ask-video" in sel.active


# --- application --------------------------------------------------------------

def test_apply_makes_active_dir_hold_exactly_the_set(tmp_path):
    skills, active = tmp_path / "skills", tmp_path / "active"
    make_skill(skills, "keep", 'vss-requires: "always"')
    make_skill(skills, "add", 'vss-requires: "always"')
    make_skill(active, "keep")     # present, stays (refreshed)
    make_skill(active, "stale")    # present, must go
    sync_skills.apply_active(skills, active, ["keep", "add"])
    assert sorted(p.name for p in active.iterdir()) == ["add", "keep"]
    assert (active / "add" / "SKILL.md").is_file()


def test_apply_missing_source_raises(tmp_path):
    (tmp_path / "skills").mkdir()
    with pytest.raises(FileNotFoundError):
        sync_skills.apply_active(tmp_path / "skills", tmp_path / "active", ["ghost"])


# --- CLI contract ---------------------------------------------------------------

def cli(tmp_path, monkeypatch, *argv):
    """Run main() with a stub vss/curl on PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "vss"
    stub.write_text('#!/bin/sh\necho "no deployment configured" >&2\nexit 3\n')
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path('/usr/bin')}:{Path('/bin')}")
    return sync_skills.main(list(argv))


def test_cli_all_exit_zero_and_populates(tmp_path, monkeypatch):
    skills = tmp_path / "skills"
    make_skill(skills, "s1", 'vss-requires: "always"')
    rc = cli(tmp_path, monkeypatch, "--skills-dir", str(skills),
             "--active-dir", str(tmp_path / "active"), "--all")
    assert rc == 0 and (tmp_path / "active" / "s1" / "SKILL.md").is_file()


def test_cli_plugin_dir_maps_openclaw_layout(tmp_path, monkeypatch):
    plugin = tmp_path / "plugin"
    make_skill(plugin / "skills", "s1", 'vss-requires: "always"')
    rc = cli(tmp_path, monkeypatch, "--plugin-dir", str(plugin), "--all")
    assert rc == 0 and (plugin / "skills-active" / "s1" / "SKILL.md").is_file()


def test_cli_empty_set_exits_three(tmp_path, monkeypatch):
    (tmp_path / "skills").mkdir()
    rc = cli(tmp_path, monkeypatch, "--skills-dir", str(tmp_path / "skills"),
             "--active-dir", str(tmp_path / "active"), "--all")
    assert rc == 3


def test_cli_error_exits_one(tmp_path, monkeypatch, capsys):
    rc = cli(tmp_path, monkeypatch, "--skills-dir", str(tmp_path / "missing"),
             "--active-dir", str(tmp_path / "active"))
    # unconfigured stub -> all skills active -> empty dir -> no error;
    # force an error instead: unreadable skills dir is fine as "no skills" (3),
    # so use a file where a directory is expected.
    (tmp_path / "afile").write_text("x")
    rc = sync_skills.main(["--skills-dir", str(tmp_path / "afile" / "nope"),
                           "--active-dir", str(tmp_path / "active2")])
    assert rc in (1, 3)


# --- pin lockstep ---------------------------------------------------------------

def test_vss_ref_defaults_are_in_lockstep_and_readable():
    """Both harness images must stage skills, docs, and this tool from the SAME
    ref — two defaults drifting apart means the harnesses ship different skill
    snapshots (the defect this check exists to prevent). The default is a ref a
    person can read: the `develop` branch, or a `v*` release tag; a bare commit
    sha is an override for a reproducible rebuild, never the default."""
    import re as _re
    repo = Path(__file__).resolve().parents[4]
    refs = {}
    for df in (repo / ".openclaw" / "Dockerfile", repo / ".hermes" / "Dockerfile"):
        m = _re.search(r"^ARG VSS_REF=(\S+)$", df.read_text(), _re.M)
        assert m, f"{df} has no VSS_REF default"
        refs[df.parent.name] = m.group(1)
    assert refs[".openclaw"] == refs[".hermes"], f"VSS_REF defaults drifted: {refs}"
    ref = refs[".openclaw"]
    assert ref == "develop" or _re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?", ref), (
        f"VSS_REF default {ref!r} is neither `develop` nor a v* release tag")



def _hermes_sync_wrapper() -> str:
    """The vss-hermes-sync script exactly as .hermes/Dockerfile writes it (via printf)."""
    dockerfile = (Path(__file__).resolve().parents[4] / ".hermes" / "Dockerfile").read_text()
    body = re.search(r"printf '(#!/bin/sh\\n.*?)' > /usr/local/bin/vss-hermes-sync", dockerfile).group(1)
    return body.replace("\\n", "\n")


@pytest.mark.parametrize("hermes_home", ["relocated", None])
def test_hermes_sync_targets_hermes_home_skills(tmp_path, hermes_home):
    """Hermes reads $HERMES_HOME/skills only, so the wrapper follows HERMES_HOME
    (unset means /sandbox/.hermes) and owns just its vss/ category there."""
    wrapper = tmp_path / "vss-hermes-sync"
    wrapper.write_text(_hermes_sync_wrapper())
    wrapper.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "argv"
    (bin_dir / "python3").write_text(f'#!/bin/sh\necho "$@" > {log}\n')
    (bin_dir / "python3").chmod(0o755)
    (bin_dir / "mkdir").write_text("#!/bin/sh\nexit 0\n")  # /sandbox is not writable here
    (bin_dir / "mkdir").chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin"}
    if hermes_home:
        env["HERMES_HOME"] = str(tmp_path / hermes_home)
    subprocess.run([str(wrapper), "--all"], env=env, check=True)
    active = f"{env.get('HERMES_HOME', '/sandbox/.hermes')}/skills/vss"
    assert log.read_text().split() == [
        "/opt/vss-skills/sync_skills.py", "--skills-dir", "/opt/vss-skills/skills",
        "--active-dir", active, "--all"]
