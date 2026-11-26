# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An inherited environment variable can neither redirect a request of the scripts nor pick a
program they run, a redirect is never followed, and --out writes only where no other user can
change the file (references/security.md).

Each variable test points one variable that earlier versions of the scripts read at a rogue
target, then shows that the requests still go to the built-in host, or that the program still
comes from PATH. No request leaves this machine: the `sent` fixture stops every urllib request
before it connects and records its URL, and the other tests use local servers."""

from __future__ import annotations

import ast
import errno
import json
import os
import re
import shlex
import stat
import struct
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cnd
import env_inspect
import locality
import ngc_api
import ns_common as ns
import pytest
import registry_labels as rl
from ns_testkit import (
    CLOSED,
    SCRIPTS,
    TESTS,
    fake_docker,
    fixture,
    run_cli,
    serve,
    set_harbor_base,
    set_ngc_base,
    set_tgc_base,
    set_turingcm_base,
    use_built_in_endpoints,
)

ROGUE = ["http://127.0.0.1:9", "http://169.254.169.254", "file:///etc"]
# An image on each China NIM distributor host, and the built-in Harbor API host it is read from
# (the tgcr-gz mirror has no API of its own).
HARBOR_IMAGES = {
    "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b:1.0": "tgcr.turing-agi.com",
    "resource.turingcm.com/qwen/qwen3.6-27b:1.0": "resource.turingcm.com",
    "io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0": "io.chancloud.com",
}


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The URL of every urllib request, each stopped before it connects (as unreachable), with
    the endpoints the scripts ship with in place of conftest.py's closed port."""
    use_built_in_endpoints(monkeypatch)
    urls: list[str] = []

    def stop(self: urllib.request.OpenerDirector, request: Any, *args: Any, **kwargs: Any) -> Any:
        urls.append(request.full_url if isinstance(request, urllib.request.Request) else request)
        raise urllib.error.URLError("stopped by the test")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", stop)
    return urls


def all_start_with(urls: list[str], prefix: str) -> bool:
    return bool(urls) and all(url.startswith(prefix) for url in urls)


# --- request targets --------------------------------------------------------------------------


@pytest.mark.parametrize("target", ROGUE)
def test_ngc_base_ignores_the_environment(
    monkeypatch: pytest.MonkeyPatch, sent: list[str], target: str
) -> None:
    monkeypatch.setenv("NIM_SELECT_NGC_BASE", target)
    assert ngc_api.base_url() == "https://api.ngc.nvidia.com/v2"
    with pytest.raises(ns.ScriptError) as info:
        ngc_api.get_json("/repos/nim/meta/llama-3.1-8b-instruct", attempts=1)
    assert info.value.code == "ngc_unreachable"
    assert sent == ["https://api.ngc.nvidia.com/v2/repos/nim/meta/llama-3.1-8b-instruct"]


@pytest.mark.parametrize("target", ROGUE)
def test_harbor_hosts_ignore_the_environment(
    monkeypatch: pytest.MonkeyPatch, sent: list[str], target: str
) -> None:
    monkeypatch.setenv("NIM_SELECT_HARBOR_BASE", target)
    for image, api_host in HARBOR_IMAGES.items():
        sent.clear()
        ref = rl.parse_image(image)
        assert ref is not None
        assert rl.harbor_base(ref.api_host) == f"https://{api_host}"
        assert rl.read_labels(ref).status == "unreachable"
        assert rl.read_repository(ref)[0] == "unreachable"
        assert rl.list_repositories(ref.host, ref.project)[0] == "unreachable"
        assert all_start_with(sent, f"https://{api_host}/api/v2.0/projects/"), sent
    # Lichan's catalog is its Harbor
    sent.clear()
    with pytest.raises(ns.ScriptError) as info:
        cnd.search("lichan", "qwen3.5-35b-a3b")
    assert info.value.code == "cnd_unreachable"
    assert all_start_with(sent, "https://io.chancloud.com/api/v2.0/projects/cnd/"), sent


@pytest.mark.parametrize("target", ROGUE)
def test_tgc_catalog_ignores_the_environment(
    monkeypatch: pytest.MonkeyPatch, sent: list[str], target: str
) -> None:
    monkeypatch.setenv("NIM_SELECT_TGC_BASE", target)
    with pytest.raises(ns.ScriptError) as info:
        cnd.search("tgcr", "qwen3.6-27b")
    assert info.value.code == "cnd_unreachable"
    assert all_start_with(sent, "https://tgc.turing-agi.com/tgc/api/resources?"), sent


@pytest.mark.parametrize("target", ROGUE)
def test_tgc_catalog_reads_of_an_image_ignore_the_environment(
    monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """The second TGC site, TgcReader.catalog_tags: the tags of a tgcr image no catalog name
    holds (GLM-5.1 is z-ai/glm5_1) are looked up by name, then in every catalog item."""
    use_built_in_endpoints(monkeypatch)
    monkeypatch.setenv("NIM_SELECT_TGC_BASE", target)
    urls: list[str] = []

    def catalog_json(self: cnd.TgcReader, url: str) -> Any:
        urls.append(url)
        return {"data": {"items": [], "total": 0}}

    monkeypatch.setattr(cnd.TgcReader, "catalog_json", catalog_json)
    ref = rl.parse_image("tgcr-gz.turing-agi.com/public/z-ai/glm5_1:1.0")
    assert ref is not None
    reader = cnd.reader("tgcr")
    assert isinstance(reader, cnd.TgcReader)
    tags, _ = reader.catalog_tags(ref, None, "1.0")
    assert tags == []
    assert urls == [
        "https://tgc.turing-agi.com/tgc/api/resources?page=1&size=100&search=glm5_1",
        "https://tgc.turing-agi.com/tgc/api/resources?page=1&size=100",
    ]


@pytest.mark.parametrize("target", ROGUE)
def test_turingcm_catalog_ignores_the_environment(
    monkeypatch: pytest.MonkeyPatch, sent: list[str], target: str
) -> None:
    monkeypatch.setenv("NIM_SELECT_TURINGCM_BASE", target)
    with pytest.raises(ns.ScriptError) as info:
        cnd.search("turingcm", "qwen3.6-27b")
    assert info.value.code == "cnd_unreachable"
    assert sent == ["https://nim.turingcm.com/server/resource/nimImages/simple-list"]


@pytest.mark.parametrize("target", ROGUE)
def test_the_locate_check_ignores_the_environment(
    monkeypatch: pytest.MonkeyPatch, sent: list[str], target: str
) -> None:
    monkeypatch.setenv("NIM_SELECT_LOCATE_URL", target)
    assert locality.locate({"TZ": "Europe/Berlin"})["google_reachable"] is False
    assert sent == ["https://www.google.com/generate_204"]


@pytest.mark.parametrize("target", ROGUE)
def test_the_country_lookup_ignores_the_environment(
    monkeypatch: pytest.MonkeyPatch, sent: list[str], target: str
) -> None:
    monkeypatch.setenv("NIM_SELECT_COUNTRY_URL", target)
    code, why = locality.lookup_country()
    assert code is None and why
    assert sent == ["https://www.cloudflare.com/cdn-cgi/trace"]


def test_a_redirect_is_never_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    """urllib's default opener follows a redirect to any host, https to http included. NGC, the
    Harbor hosts and the CND catalogs send none, so a redirect is reported as a bad answer, and
    the host it names gets no request."""
    with serve({}) as elsewhere:
        target = elsewhere.base.removesuffix("/v2") + "/stolen"
        with serve({"*": (302, target)}) as first:
            root = first.base.removesuffix("/v2")
            set_ngc_base(monkeypatch, first.base)
            for set_base in (set_harbor_base, set_tgc_base, set_turingcm_base):
                set_base(monkeypatch, root)
            with pytest.raises(ns.ScriptError) as ngc:
                ngc_api.get_json("/repos/nim/meta/llama-3.1-8b-instruct")
            assert (ngc.value.code, ngc.value.exit_code) == ("ngc_bad_response", ns.EXIT_EXTERNAL)
            assert "HTTP 302, a redirect to 127.0.0.1" in ngc.value.message
            image = rl.parse_image("io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0")
            assert image is not None
            labels = rl.read_labels(image)
            assert labels.status == "unreachable" and "redirect" in str(labels.problem)
            for source in ("tgcr", "turingcm", "lichan"):
                with pytest.raises(ns.ScriptError) as info:
                    cnd.search(source, "qwen3.6-27b")
                assert info.value.code == "cnd_unreachable"
                assert "redirect" in info.value.message, info.value.message
            assert len(first.requests) >= 5
    assert elsewhere.requests == []


def test_every_request_goes_through_open_url() -> None:
    """ns.open_url is the one opener, and it follows no redirect; urllib.request.urlopen, or
    another opener, would follow them."""
    openers = [
        (path.name, name)
        for path in sorted(SCRIPTS.glob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and (name := getattr(node.func, "attr", getattr(node.func, "id", "")))
        in ("urlopen", "build_opener", "OpenerDirector", "HTTPConnection", "HTTPSConnection")
    ]
    assert openers == [("ns_common.py", "build_opener")]


# --- programs and files -----------------------------------------------------------------------


def write_tool(folder: Path, name: str, body: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def fake_host_tools(bin_dir: Path) -> None:
    """nvidia-smi (8 H100 NVL), docker (a local daemon with the nvidia runtime) and kubectl (a
    kubeconfig with the context `c`)."""
    write_tool(bin_dir, "nvidia-smi", f"cat '{fixture('env', 'nvidia-smi-h100nvl-8.csv')}'")
    write_tool(bin_dir, "docker", fake_docker(fixture("env", "docker-info-h100nvl.json")))
    nodes = fixture("env", "kubectl-nodes-h100nvl.json")
    write_tool(
        bin_dir,
        "kubectl",
        f"case \"$*\" in *get-contexts*) echo c;; *nodes*) cat '{nodes}';; *) echo '{{}}';; esac",
    )


@pytest.mark.parametrize(
    ("tool", "variable"),
    [
        ("nvidia-smi", "NIM_SELECT_NVIDIA_SMI"),
        ("docker", "NIM_SELECT_DOCKER"),
        ("kubectl", "NIM_SELECT_KUBECTL"),
    ],
)
def test_a_tool_variable_cannot_pick_the_program(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool: str, variable: str
) -> None:
    """Only the named program on PATH is found, and inspect runs it, never the variable's."""
    marker = tmp_path / "rogue-ran"
    rogue = write_tool(tmp_path / "elsewhere", "rogue", f"touch '{marker}'; exit 99")
    monkeypatch.setenv(variable, str(rogue))
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    assert env_inspect.resolve_tool(tool) is None
    trusted = write_tool(tmp_path / "bin", tool, "exit 0")
    assert env_inspect.resolve_tool(tool) == str(trusted)

    # the same through the command line, started as SKILL.md starts it
    bin_dir = tmp_path / "bin"
    fake_host_tools(bin_dir)
    env = {
        variable: str(rogue),
        "PATH": str(bin_dir) + os.pathsep + os.defpath,
        "DOCKER_HOST": "",
        "HOME": str(tmp_path),  # inspect reads the free disk of ~/.cache/nim
    }
    target = ["--target", "k8s", "--context", "c", "--namespace", "n"] if tool == "kubectl" else []
    code, payload = run_cli("nimselect.py", "inspect", *target, env=env, direct=True)
    assert code == 0, payload
    assert payload["facts"]["groups"][0]["count"] == 8
    if tool == "docker":
        assert payload["facts"]["docker"]["nvidia_runtime"] is True
    assert not marker.exists()


@pytest.mark.parametrize(
    "entries", [["", "BIN"], ["BIN", ""], [".", "BIN"], ["project", "BIN"], ["./", "BIN"]]
)
def test_an_empty_or_relative_path_entry_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entries: list[str]
) -> None:
    """An empty PATH entry (`::`, a leading or trailing `:`) or a relative one names the current
    folder, often a project someone else wrote: a program there is never run."""
    project = tmp_path / "project"
    marker = tmp_path / "project-tool-ran"
    for tool in ("nvidia-smi", "docker", "kubectl"):
        write_tool(project, tool, f"touch '{marker}'; exit 99")
        write_tool(project / "project", tool, f"touch '{marker}'; exit 99")
    bin_dir = tmp_path / "bin"
    fake_host_tools(bin_dir)
    path = os.pathsep.join(str(bin_dir) if entry == "BIN" else entry for entry in entries)
    no_bin = os.pathsep.join("/nonexistent" if entry == "BIN" else entry for entry in entries)
    monkeypatch.chdir(project)
    for tool in ("nvidia-smi", "docker", "kubectl"):
        assert env_inspect.resolve_tool(tool, {"PATH": path}) == str(bin_dir / tool)
        assert env_inspect.resolve_tool(tool, {"PATH": no_bin}) is None

    # the fake tools need `cat` from the default PATH
    env = {"PATH": path + os.pathsep + os.defpath, "DOCKER_HOST": "", "HOME": str(tmp_path)}
    code, payload = run_cli("nimselect.py", "inspect", env=env, direct=True)
    assert code == 0, payload
    assert payload["facts"]["groups"][0]["name"] == "NVIDIA H100 NVL"
    assert not marker.exists()


def test_the_host_memory_file_ignores_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NIM_SELECT_MEMINFO used to pick the file the host memory (and, on a GPU that shares the
    host memory, its VRAM) is read from."""
    rogue = tmp_path / "meminfo"
    rogue.write_text("MemTotal:       999999999999 kB\n", encoding="utf-8")
    monkeypatch.setenv("NIM_SELECT_MEMINFO", str(rogue))
    monkeypatch.setenv("HOME", str(tmp_path))  # inspect_host reads the free disk of ~/.cache/nim
    read: list[Path] = []
    real = env_inspect.meminfo_total_kb

    def meminfo_total_kb(path: Path) -> int | None:
        read.append(path)
        return real(path)

    monkeypatch.setattr(env_inspect, "meminfo_total_kb", meminfo_total_kb)
    rows = fixture("env", "nvidia-smi-gb10.csv").read_text(encoding="utf-8")
    env = env_inspect.inspect_host(
        runner=lambda argv, timeout: (0, rows, ""),
        tools={"nvidia-smi": "nvidia-smi", "docker": None},
    )
    assert read == [Path("/proc/meminfo")]
    assert env["groups"][0]["unified_memory"] is True  # GB10: its VRAM is this host memory


# --- --out files ------------------------------------------------------------------------------


def test_out_replaces_a_planted_symlink_and_never_follows_it(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("precious\n", encoding="utf-8")
    folder = tmp_path / "out"
    folder.mkdir(mode=0o700)
    (folder / "env.json").symlink_to(victim)
    ns.write_json_file(folder / "env.json", {"a": 1})
    assert victim.read_text(encoding="utf-8") == "precious\n"
    assert not (folder / "env.json").is_symlink()
    assert json.loads((folder / "env.json").read_text(encoding="utf-8")) == {"a": 1}
    assert [path.name for path in folder.iterdir()] == ["env.json"]  # no temporary file left


def test_out_creates_a_private_folder(tmp_path: Path) -> None:
    ns.write_json_file(tmp_path / "new" / "nim-select" / "env.json", {"a": 1})
    assert (tmp_path / "new" / "nim-select").stat().st_mode & 0o777 == 0o700


def fresh_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A fresh home folder (HOME)."""
    home = tmp_path / "home"
    home.mkdir(mode=0o750)
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX owners only")
def test_out_creates_each_missing_folder_private_under_a_umask_of_002(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A umask of 002 is the default for an account with a group of its own. Path.mkdir with
    parents=True would then make a new ~/.cache 0775, which a host whose accounts come from SSSD
    refuses: there the group is not known to be private. Each folder --out creates is 0700, one
    level at a time; the home folder, which exists, is left as it is."""
    home = fresh_home(monkeypatch, tmp_path)
    home_mode = home.stat().st_mode
    umask = os.umask(0o002)
    try:
        ns.write_json_file(Path("~/.cache/nim-select/env.json"), {"a": 1})
    finally:
        os.umask(umask)
    out = home / ".cache" / "nim-select" / "env.json"
    assert json.loads(out.read_text(encoding="utf-8")) == {"a": 1}
    for folder in (home / ".cache", out.parent):
        assert folder.stat().st_mode & 0o777 == 0o700, folder
    assert home.stat().st_mode == home_mode


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX owners only")
def test_out_refuses_a_folder_other_users_can_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Another user could replace the file between inspect and select, or plant a symlink: a
    folder, or one above it, that others can write (without the sticky bit of /tmp) or that
    belongs to another user is refused. The user's private group may write: on a system with a
    group per user (umask 002), a project folder is group-writable."""
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    for path in (shared / "env.json", shared / "run1" / "env.json"):
        with pytest.raises(ns.ScriptError) as info:
            ns.write_json_file(path, {"a": 1})
        assert (info.value.code, info.value.exit_code) == ("out_not_private", ns.EXIT_INPUT)
        assert f"{shared.resolve()} can be written by other users" in info.value.message
        # the user's own folder: the fix names it and the chmod that takes that access away
        # (not as root: test_root_is_never_told_to_chmod_a_folder_a_group_shares)
        if os.getuid() != 0:
            assert f"chmod go-w {shlex.quote(str(shared.resolve()))}," in info.value.fix
        assert not path.exists()
    sticky = tmp_path / "sticky"
    sticky.mkdir()
    sticky.chmod(0o1777)
    ns.write_json_file(sticky / "env.json", {"a": 1})
    group = tmp_path / "group"
    group.mkdir()
    group.chmod(0o770)  # even when its group is the user's own private group
    with pytest.raises(ns.ScriptError) as grouped:
        ns.write_json_file(group / "env.json", {"a": 1})
    assert grouped.value.code == "out_not_private"
    owner = os.stat(tmp_path).st_uid
    if owner == 0:
        return  # a folder of root is trusted (root can change any file), so test as a user
    monkeypatch.setattr(os, "getuid", lambda: owner + 4242)
    with pytest.raises(ns.ScriptError) as other:
        ns.write_json_file(tmp_path / "mine" / "env.json", {"a": 1})
    assert other.value.code == "out_not_private" and "belongs to another user" in str(other.value)
    assert other.value.fix.startswith("Pass --out a file in a folder of your own")


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX owners only")
def test_inspect_out_in_a_shared_folder_is_refused(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    fake_host_tools(bin_dir)
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    victim = tmp_path / "victim.txt"
    victim.write_text("precious\n", encoding="utf-8")
    (shared / "env.json").symlink_to(victim)
    env = {
        "PATH": str(bin_dir) + os.pathsep + os.defpath,
        "DOCKER_HOST": "",
        "HOME": str(tmp_path),
    }
    out = str(shared / "env.json")
    code, payload = run_cli("nimselect.py", "inspect", "--out", out, env=env, direct=True)
    assert (code, payload["errors"][0]["code"]) == (2, "out_not_private"), payload
    assert victim.read_text(encoding="utf-8") == "precious\n"


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX owners only")
def test_out_refuses_a_folder_the_users_own_group_can_write(tmp_path: Path) -> None:
    """A folder the user's group can write is refused even when that group is the user's private
    group: whether another account is in a group cannot be verified on every host (directory
    services, initgroups), so no group is trusted. Without the group's write it is accepted."""
    folder = tmp_path / "group"
    folder.mkdir()
    folder.chmod(0o770)
    with pytest.raises(ns.ScriptError) as info:
        ns.write_json_file(folder / "env.json", {"a": 1})
    assert f"{folder.resolve()} can be written by other users" in info.value.message
    assert not (folder / "env.json").exists()
    folder.chmod(0o750)
    ns.write_json_file(folder / "env.json", {"a": 1})


@pytest.mark.skipif(
    not hasattr(os, "getuid") or os.getuid() == 0, reason="POSIX owners, as a user other than root"
)
def test_a_group_writable_folder_of_the_user_is_refused_with_the_chmod_that_fixes_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ~/.cache that another program made 0775 is refused, even when its group is the user's
    own private group. The fix names that folder and the chmod g-w that leaves it writable by the user
    alone, never ~/.cache/nim-select/, which is in it."""
    cache = fresh_home(monkeypatch, tmp_path) / ".cache"
    cache.mkdir()
    cache.chmod(0o775)
    with pytest.raises(ns.ScriptError) as info:
        ns.write_json_file(Path("~/.cache/nim-select/env.json"), {"a": 1})
    assert info.value.code == "out_not_private"
    assert f"{cache.resolve()} can be written by other users" in info.value.message
    assert f"chmod g-w {shlex.quote(str(cache.resolve()))}," in info.value.fix
    assert "or pass --out a file in a folder only the user can write" in info.value.fix
    assert "~/.cache/nim-select" not in info.value.fix
    assert not (cache / "nim-select" / "env.json").exists()


def posix_acl(*named: tuple[int, int, int]) -> bytes:
    """The system.posix_acl_access value Linux keeps for an ACL with the `named` entries (tag,
    permissions, id): rwx for the owner and the folder's group, a mask of rwx, none for others."""
    none = 0xFFFFFFFF  # the id of an entry that names no user or group
    base = [(0x01, 0o7, none), (0x04, 0o7, none), (0x10, 0o7, none), (0x20, 0, none)]
    entries = sorted([*base, *named], key=lambda entry: (entry[0], entry[2]))  # as Linux wants
    return struct.pack("<I", 2) + b"".join(struct.pack("<HHI", *entry) for entry in entries)


@pytest.mark.skipif(not hasattr(os, "setxattr"), reason="POSIX ACLs as Linux keeps them")
def test_out_refuses_a_folder_an_acl_lets_another_user_write(tmp_path: Path) -> None:
    """With a POSIX ACL the group bits show the ACL's mask, which bounds every user and group the
    ACL names: after setfacl -m u:bob:rwx shared, the mask has w and the folder is refused. After
    the chmod g-w of the fix the mask loses w, so bob can no longer write, and it is accepted."""
    shared = tmp_path.resolve() / "shared"
    shared.mkdir(mode=0o700)
    path = shared / "out" / "env.json"
    bob = (0x02, 0o7, os.getuid() + 4242)  # a named user (ACL_USER) with rwx
    try:
        os.setxattr(shared, "system.posix_acl_access", posix_acl(bob), follow_symlinks=False)
    except OSError as exc:
        if exc.errno not in (errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL, errno.EPERM):
            raise
        pytest.skip(f"this file system or user namespace takes no such ACL: {exc.strerror}")
    assert shared.stat().st_mode & 0o070 == 0o070  # the group bits: the mask
    with pytest.raises(ns.ScriptError) as info:
        ns.write_json_file(path, {"a": 1})
    assert info.value.code == "out_not_private"
    assert f"{shared} can be written by other users" in info.value.message
    assert not path.exists()
    shared.chmod(0o750)  # the fix: the mask loses w
    ns.write_json_file(path, {"a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX owners only")
def test_root_is_never_told_to_chmod_a_folder_a_group_shares(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Root owns the system folders, and a group shares some of them on purpose: /var/local is
    root:staff 2775 on Debian and Ubuntu. Run as root, --out /var/local/nim-select/env.json is
    refused, and the fix sends root to a folder of its own, never to chmod g-w /var/local."""
    local = tmp_path.resolve() / "local"
    local.mkdir()
    real_stat = Path.stat

    def stat_as_root(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        info = real_stat(self, follow_symlinks=follow_symlinks)
        if self != local and local not in self.parents:
            return info
        fields = list(info)  # root's, as /var/local and the folders root makes in it
        fields[stat.ST_UID] = 0
        if self == local:  # root:staff 2775
            fields[stat.ST_MODE], fields[stat.ST_GID] = stat.S_IFDIR | 0o2775, 50
        return os.stat_result(fields)

    monkeypatch.setattr(Path, "stat", stat_as_root)
    monkeypatch.setattr(os, "getuid", lambda: 0)
    monkeypatch.setattr(os, "getgid", lambda: 0)
    path = local / "nim-select" / "env.json"
    with pytest.raises(ns.ScriptError) as info:
        ns.write_json_file(path, {"a": 1})
    assert info.value.code == "out_not_private"
    assert f"{local} can be written by other users" in info.value.message
    assert info.value.fix.startswith("Pass --out a file in a folder of your own")
    assert "chmod" not in info.value.fix
    assert not path.exists()


def test_out_is_written_in_the_folder_that_was_checked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A symlink on the way to --out that changes right after the check (one another user can
    change, in a folder they can write) cannot move the file: it goes to the folder checked."""
    checked = tmp_path / "checked"
    checked.mkdir(mode=0o700)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(checked)
    check = ns.check_private

    def check_then_change_the_link(folder: Path) -> Path:
        result = check(folder)
        link.unlink()
        link.symlink_to(elsewhere)
        return result

    monkeypatch.setattr(ns, "check_private", check_then_change_the_link)
    ns.write_json_file(link / "env.json", {"a": 1})
    assert json.loads((checked / "env.json").read_text(encoding="utf-8")) == {"a": 1}
    assert list(elsewhere.iterdir()) == []


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX owners only")
def test_out_is_refused_when_a_folder_on_the_way_is_swapped_during_the_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A folder on the way that another user can rename (theirs, in /tmp) is refused. Swapped
    right after the path is resolved for a symlink to a private folder of the user, it is still
    refused: read through that symlink, it would pass, and swapped again after the check for a
    symlink to their own folder, it would take the file there. No folder is read through a
    symlink."""
    tmp_path = tmp_path.resolve()
    stage = tmp_path / "stage"  # theirs: other users can write in it, so the check refuses it
    stage.mkdir()
    stage.chmod(0o777)
    (stage / "out").mkdir(mode=0o700)
    private, theirs = tmp_path / "private", tmp_path / "theirs"
    for folder in (private, private / "out", theirs, theirs / "out"):
        folder.mkdir(mode=0o700)
    real_resolve = Path.resolve

    def resolve_then_swap(self: Path, strict: bool = False) -> Path:
        resolved = real_resolve(self, strict)
        if resolved == stage / "out" and not stage.is_symlink():
            stage.rename(tmp_path / "moved")
            stage.symlink_to(private)
        return resolved

    check = ns.check_private

    def check_then_point_at_theirs(folder: Path) -> Path:
        result = check(folder)
        stage.unlink()
        stage.symlink_to(theirs)
        return result

    monkeypatch.setattr(Path, "resolve", resolve_then_swap)
    monkeypatch.setattr(ns, "check_private", check_then_point_at_theirs)
    with pytest.raises(ns.ScriptError) as info:
        ns.write_json_file(stage / "out" / "env.json", {"a": 1})
    assert info.value.code == "out_not_private"
    assert f"{stage} is no longer a folder (it changed during the check)" in info.value.message
    assert list((theirs / "out").iterdir()) == list((private / "out").iterdir()) == []


def test_a_private_file_the_json_parser_refuses_is_a_problem_never_a_traceback(
    tmp_path: Path,
) -> None:
    """read_private_json answers (None, why) for JSON the parser refuses: nesting too deep, or an
    integer too long (Python's limit on the digits of an int, where it has one)."""
    refused = {"deep.json": "[" * 100_000}
    limit = sys.get_int_max_str_digits() if hasattr(sys, "get_int_max_str_digits") else 0
    if limit:
        refused["long.json"] = "1" * (limit + 1)
    for name, text in refused.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        path.chmod(0o600)
        assert ns.read_private_json(path) == (None, f"{path} is not valid JSON")


# --- what is left -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "delay"),
    [
        ("", 1.0),
        ("0", 0.0),
        ("2.5", 2.5),
        ("-1", 0.0),
        ("1e9", ngc_api.MAX_RETRY_DELAY),
        ("inf", 1.0),
        ("-inf", 1.0),
        ("1e400", 1.0),
        ("nan", 1.0),
        ("soon", 1.0),
    ],
)
def test_the_retry_delay_is_bounded(
    monkeypatch: pytest.MonkeyPatch, raw: str, delay: float
) -> None:
    """NIM_SELECT_RETRY_DELAY changes only how long a retry waits, and never for more than
    MAX_RETRY_DELAY seconds: a huge value used to hang a run, and inf to crash it with no JSON."""
    monkeypatch.setenv("NIM_SELECT_RETRY_DELAY", raw)
    assert ngc_api.retry_delay() == delay


def test_a_huge_retry_delay_neither_hangs_nor_crashes(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(ngc_api.time, "sleep", slept.append)
    set_ngc_base(monkeypatch, f"{CLOSED}/v2")
    for raw in ("inf", "1e400", "1e9"):
        monkeypatch.setenv("NIM_SELECT_RETRY_DELAY", raw)
        with pytest.raises(ns.ScriptError) as info:
            ngc_api.get_json("/repos/nim/meta/llama-3.1-8b-instruct", attempts=3)
        assert info.value.code == "ngc_unreachable"
    assert slept and max(slept) <= 2 * ngc_api.MAX_RETRY_DELAY


# The environment variables each script reads (references/security.md lists the same names).
# HOME is read through Path.home() and expanduser().
ENV_READS = {
    "env_inspect": {"PATH", "DOCKER_HOST", "HOME"},
    "locality": {"TZ", "LC_ALL", "LANG"},
    "ngc_api": {"NIM_SELECT_RETRY_DELAY"},
    "ns_common": {"HOME"},
}
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """The nodes of one scope, without those of the functions and classes inside it."""
    found: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        found.append(node)
        if not isinstance(node, _SCOPES):
            stack.extend(ast.iter_child_nodes(node))
    return found


def environment_reads(source: str, module: str) -> tuple[set[str], list[str]]:
    """The environment variables a script reads, and the reads this check cannot name.

    An environment mapping is `os.environ` (any `.environ` or `.environb`), a name `environ`
    (the scripts' parameter for one), or a name assigned from either in the same function or
    one around it. A read is a subscript, `.get`, `.pop`, `.setdefault`, `in`, or os.getenv; its
    key must be a literal, or a loop variable over literals. Any other use of the mapping as a
    whole (copy, items, dict(), **, env=) cannot be named, nor can a write. Path.home() and
    expanduser() read HOME."""
    reads: set[str] = set()
    problems: list[str] = []

    def visit(scope: ast.AST, inherited: set[str]) -> None:
        nodes = _own_nodes(scope)
        aliases = set(inherited)
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = scope.args
            every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
            aliases -= {arg.arg for arg in every}  # a parameter hides an outer name

        def is_env(node: ast.AST) -> bool:
            if isinstance(node, ast.Attribute):
                return node.attr in ("environ", "environb")
            return isinstance(node, ast.Name) and (node.id == "environ" or node.id in aliases)

        def holds_env(value: ast.AST) -> bool:
            """`value` is a mapping (`os.environ if environ is None else environ`), not a read."""
            if isinstance(value, ast.IfExp):
                return holds_env(value.body) or holds_env(value.orelse)
            if isinstance(value, ast.BoolOp):
                return any(holds_env(item) for item in value.values)
            return is_env(value)

        changed = True
        while changed:  # names assigned from an environment mapping, in this scope
            changed = False
            for node in nodes:
                if isinstance(node, ast.Assign):
                    targets, value = list(node.targets), node.value
                elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
                    targets, value = [node.target], node.value
                else:
                    continue
                if holds_env(value):
                    for target in targets:
                        if isinstance(target, ast.Name) and target.id not in aliases:
                            aliases.add(target.id)
                            changed = True
        loops = {
            node.target.id: [item.value for item in node.iter.elts]
            for node in nodes
            if isinstance(node, (ast.For, ast.comprehension))
            and isinstance(node.target, ast.Name)
            and isinstance(node.iter, (ast.Tuple, ast.List, ast.Set))
            and all(
                isinstance(item, ast.Constant) and isinstance(item.value, str)
                for item in node.iter.elts
            )
        }

        def read(key: ast.AST | None, node: ast.AST) -> None:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                reads.add(key.value)
            elif isinstance(key, ast.Name) and key.id in loops:
                reads.update(loops[key.id])
            else:
                problems.append(f"{module}.py:{getattr(node, 'lineno', '?')}: name not literal")

        for node in nodes:
            where = f"{module}.py:{getattr(node, 'lineno', '?')}"
            if isinstance(node, ast.Subscript) and is_env(node.value):
                read(node.slice, node)
            elif (
                isinstance(node, ast.Compare)
                and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
                and any(is_env(item) for item in node.comparators)
            ):
                read(node.left, node)
            elif isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if isinstance(func, ast.Attribute) and is_env(func.value):
                    if name in ("get", "pop", "setdefault"):
                        read(node.args[0] if node.args else None, node)
                    else:
                        problems.append(f"{where}: the environment as a whole (.{name})")
                elif name in ("getenv", "getenvb"):
                    read(node.args[0] if node.args else None, node)
                elif name in ("putenv", "unsetenv"):
                    problems.append(f"{where}: an environment write ({name})")
                elif name in ("home", "expanduser"):
                    reads.add("HOME")
                elif any(is_env(arg) for arg in node.args) and name in ("dict", "list", "sorted"):
                    problems.append(f"{where}: the environment as a whole ({name}())")
                if any(is_env(keyword.value) for keyword in node.keywords if keyword.arg is None):
                    problems.append(f"{where}: the environment as a whole (**)")
                if any(is_env(keyword.value) for keyword in node.keywords if keyword.arg == "env"):
                    problems.append(f"{where}: the environment passed whole to a program")
            elif isinstance(node, ast.Dict) and any(
                key is None and is_env(value) for key, value in zip(node.keys, node.values)
            ):
                problems.append(f"{where}: the environment as a whole (**)")
        for node in nodes:
            if isinstance(node, _SCOPES):
                visit(node, aliases)

    visit(ast.parse(source), set())
    return reads, problems


def documented_variables() -> set[str]:
    """The names in the first column of the table under 'What the scripts read from the
    environment' in references/security.md."""
    text = (TESTS.parent / "references" / "security.md").read_text(encoding="utf-8")
    section = text.split("## What the scripts read from the environment", 1)[1].split("\n## ")[0]
    rows = [line for line in section.splitlines() if line.startswith("| `")]
    return {name for row in rows for name in re.findall(r"`([A-Z][A-Z0-9_]*)`", row.split("|")[1])}


def test_the_scripts_read_only_the_listed_environment_variables() -> None:
    """Every environment read of every script is one the security notes list, by an AST check:
    an override under any name (os.environ.get("NGC_API_BASE"), os.environ["HARBOR_ROOT"], a
    name built at run time) fails here. A new variable needs a review against
    references/security.md first."""
    found: dict[str, set[str]] = {}
    problems: list[str] = []
    for path in sorted(SCRIPTS.glob("*.py")):
        reads, unnamed = environment_reads(path.read_text(encoding="utf-8"), path.stem)
        problems.extend(unnamed)
        if reads:
            found[path.stem] = reads
    assert problems == []
    assert found == ENV_READS
    assert documented_variables() == set().union(*ENV_READS.values())


@pytest.mark.parametrize(
    "line",
    [
        'base = os.environ.get("NGC_API_BASE", DEFAULT_BASE)',
        'base = os.environ["HARBOR_ROOT"]',
        'base = os.getenv("NIM_SELECT_" + "TGC_BASE")',
        'env = os.environ\nbase = env.get("TGC_ROOT")',
        'from os import environ\nbase = environ.get("X") if "X" in environ else None',
        "copied = dict(os.environ)",
        "subprocess.run(argv, env=os.environ)",
    ],
)
def test_the_environment_check_finds_a_hidden_override(line: str) -> None:
    """environment_reads sees what a scan for NIM_SELECT_ names and getenv missed."""
    source = "import os\n\n\ndef base_url():\n" + textwrap.indent(line, "    ") + "\n"
    reads, problems = environment_reads(source, "mutant")
    assert reads or problems, line


def test_the_environment_check_follows_a_name_into_an_inner_function() -> None:
    nested = "def outer():\n    env = os.environ\n\n    def inner():\n        return env.get('X')\n"
    assert environment_reads(nested, "mutant") == ({"X"}, [])
    hidden = (
        "def outer():\n    env = os.environ\n\n    def inner(env):\n        return env.get('X')\n"
    )
    assert environment_reads(hidden, "mutant") == (set(), [])  # a parameter hides the outer name
