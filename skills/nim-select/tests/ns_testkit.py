# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the test_*.py files in this folder (not a script).

The scripts send every request to a built-in endpoint and find nvidia-smi, docker and kubectl on
PATH; no environment variable changes either. Tests point them at local servers and files only
here: the set_* helpers patch the scripts' modules in the test process, and run_cli starts a
script through this file, which applies the same patches first (the runner at the end), or, with
direct=True, starts the script itself as SKILL.md does."""

from __future__ import annotations

import contextlib
import json
import os
import re
import runpy
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cnd
import env_inspect
import listing
import locality
import ngc_api
import ns_common as ns
import registry_labels as rl

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
FIXTURES = TESTS / "fixtures"
CONTRACT_KEYS = {"ok", "state", "next", "facts", "errors"}

# --- test-only endpoints ------------------------------------------------------------------------

CLOSED = "http://127.0.0.1:9"  # the discard port: nothing listens, so a request fails at once
# The built-in values, to restore them (live tests, and tests of the URLs themselves).
BUILT_IN: dict[tuple[Any, str], Any] = {
    (ngc_api, "DEFAULT_BASE"): ngc_api.DEFAULT_BASE,
    (rl, "harbor_base"): rl.harbor_base,
    (cnd, "TGC_BASE"): cnd.TGC_BASE,
    (cnd, "TURINGCM_BASE"): cnd.TURINGCM_BASE,
    (locality, "PROBE_URL"): locality.PROBE_URL,
    (locality, "COUNTRY_URL"): locality.COUNTRY_URL,
    (env_inspect, "MEMINFO"): env_inspect.MEMINFO,
    (listing, "DEADLINE"): listing.DEADLINE,
    (ns, "PRIVATE_FOLDER"): ns.PRIVATE_FOLDER,
}


def set_ngc_base(patches: pytest.MonkeyPatch, base: str) -> None:
    """Send the NGC API requests to `base` (it ends in /v2)."""
    patches.setattr(ngc_api, "DEFAULT_BASE", base)


def set_harbor_base(patches: pytest.MonkeyPatch, root: str) -> None:
    """Send the Harbor requests of every China NIM distributor host to `root`
    (scheme://host:port)."""
    patches.setattr(rl, "harbor_base", lambda _api_host: root)


def set_tgc_base(patches: pytest.MonkeyPatch, root: str) -> None:
    patches.setattr(cnd, "TGC_BASE", root)


def set_turingcm_base(patches: pytest.MonkeyPatch, root: str) -> None:
    patches.setattr(cnd, "TURINGCM_BASE", root)


def set_locate_url(patches: pytest.MonkeyPatch, url: str) -> None:
    """The URL the location check of `nimselect.py locate` reads instead of Google's."""
    patches.setattr(locality, "PROBE_URL", url)


def set_country_url(patches: pytest.MonkeyPatch, url: str) -> None:
    """The URL the country lookup of `nimselect.py country` reads instead of Cloudflare's."""
    patches.setattr(locality, "COUNTRY_URL", url)


def set_meminfo(patches: pytest.MonkeyPatch, path: str) -> None:
    """The file host inspection reads instead of /proc/meminfo."""
    patches.setattr(env_inspect, "MEMINFO", Path(path))


def set_list_deadline(patches: pytest.MonkeyPatch, seconds: str) -> None:
    """The seconds the list command runs for, instead of its 120 s."""
    patches.setattr(listing, "DEADLINE", float(seconds))


def set_private_folder(patches: pytest.MonkeyPatch, folder: str) -> None:
    """The folder of the list's cache instead of ~/.cache/nim-select (the runner needs the real
    HOME: its pytest is in the user's site folder)."""
    patches.setattr(ns, "PRIVATE_FOLDER", folder)


def keep_offline(patches: pytest.MonkeyPatch) -> None:
    """The China NIM distributor catalogs and Harbor hosts, the location check and the country
    lookup go to a closed port: conftest.py does this for every test, and the runner for every
    script it runs."""
    set_harbor_base(patches, CLOSED)
    set_tgc_base(patches, CLOSED)
    set_turingcm_base(patches, CLOSED)
    set_locate_url(patches, f"{CLOSED}/generate_204")
    set_country_url(patches, f"{CLOSED}/cdn-cgi/trace")


def use_built_in_endpoints(patches: pytest.MonkeyPatch) -> None:
    """Undo keep_offline and every set_* helper: the endpoints the scripts ship with."""
    for (module, name), value in BUILT_IN.items():
        patches.setattr(module, name, value)


# The `env` keys of run_cli that the runner applies; they never reach the script's environment.
TEST_HOOKS: dict[str, Callable[[pytest.MonkeyPatch, str], None]] = {
    "TEST_NGC_BASE": set_ngc_base,
    "TEST_HARBOR_BASE": set_harbor_base,
    "TEST_TGC_BASE": set_tgc_base,
    "TEST_TURINGCM_BASE": set_turingcm_base,
    "TEST_LOCATE_URL": set_locate_url,
    "TEST_COUNTRY_URL": set_country_url,
    "TEST_MEMINFO": set_meminfo,
    "TEST_LIST_DEADLINE": set_list_deadline,
    "TEST_PRIVATE_FOLDER": set_private_folder,
}


def fixture(*parts: str) -> Path:
    """Path of a file under tests/fixtures."""
    return FIXTURES.joinpath(*parts)


LOCAL_DOCKER = "unix:///var/run/docker.sock"


def fake_docker(info: Path, endpoint: str = LOCAL_DOCKER) -> str:
    """The /bin/sh body of a fake docker: `docker context inspect` names the daemon `endpoint`,
    and any other command (`docker info`) prints the file `info`."""
    return f"case \"$1\" in context) echo '{endpoint}';; *) cat '{info}';; esac"


def on_path(folder: Path) -> str:
    """A PATH that finds the fake tools in `folder` first (they need /bin for `cat`)."""
    return str(folder) + os.pathsep + os.defpath


def fixture_json(*parts: str) -> Any:
    return json.loads(fixture(*parts).read_text(encoding="utf-8"))


def assert_contract(payload: dict[str, Any]) -> None:
    """The object follows the output contract {ok, state, next, facts, errors}."""
    assert set(payload) == CONTRACT_KEYS
    assert isinstance(payload["ok"], bool)
    assert isinstance(payload["state"], str) and payload["state"]
    assert isinstance(payload["next"], str) and payload["next"]
    assert isinstance(payload["facts"], dict)
    for item in payload["errors"]:
        assert set(item) == {"code", "message", "fix"}
        assert re.fullmatch(r"[a-z][a-z0-9_]*", item["code"]), item["code"]
        assert item["message"] and item["fix"]


def run_cli(
    script: str,
    *args: str,
    env: Mapping[str, str] | None = None,
    stdin: str | None = None,
    direct: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Run a script as a subprocess; return (exit code, the one JSON object on stdout).

    `env` adds to the environment, except its TEST_HOOKS keys (TEST_NGC_BASE and the like): the
    runner at the end of this file applies those, so the script never sees them. `direct` starts
    the script itself, as SKILL.md does (`python3 -I SCRIPT`; -B too, as -I ignores
    PYTHONDONTWRITEBYTECODE), with no runner and so no hook and no keep_offline: for commands
    that send no request."""
    child_env = dict(os.environ)
    if env:
        child_env.update(env)
    hooks = {name: child_env.pop(name) for name in TEST_HOOKS if name in child_env}
    if direct:
        assert not hooks, f"a direct run applies no test hook: {sorted(hooks)}"
        argv = [sys.executable, "-I", "-B", str(SCRIPTS / script), *args]
    else:
        argv = [sys.executable, str(TESTS / "ns_testkit.py"), script, json.dumps(hooks), *args]
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        input=stdin,
        stdin=None if stdin is not None else subprocess.DEVNULL,
        env=child_env,
        timeout=120,
        check=False,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, f"expected one JSON line, got {proc.stdout!r}; stderr: {proc.stderr}"
    payload = json.loads(lines[0])
    assert_contract(payload)
    return proc.returncode, payload


Route = tuple[int, Any]
# An answer that closes the connection without a response, as a reset or a dropped proxy does.
CLOSE: Route = (0, None)


@dataclass
class FakeNgc:
    """A running canned-answer server: `base` ends in /v2; `requests` lists each path requested,
    `headers` the headers of each request, and `peak` the most requests open at once."""

    base: str
    requests: list[str] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    peak: int = 0


@contextlib.contextmanager
def serve(
    routes: Mapping[str, Route], delay: float = 0.0, slow: Mapping[str, float] | None = None
) -> Iterator[FakeNgc]:
    """Serve canned JSON answers on 127.0.0.1, for the set_* helpers and TEST_HOOKS.

    `routes` maps a request path with its query (e.g. "/v2/repos/nim/meta/x") or the path alone to
    (status, body). An exact path-and-query key wins over a path-only key, and that over the key
    "*", which answers any other path; anything else is a 404 with the NGC error envelope. A list
    body of several (status, body) pairs answers successive requests in order (for retry tests),
    repeating the last one. CLOSE sends no answer at all. A 3xx status is a redirect to the URL
    in its body. `delay` seconds pass before each answer (a slow registry), and `slow` adds
    seconds to the answers of some keys (path and query, or path), as routes are keyed.
    """

    requests: list[str] = []
    headers: list[dict[str, str]] = []
    calls: dict[str, int] = {}
    lock = threading.Lock()
    open_now = [0]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            with lock:
                open_now[0] += 1
                fake.peak = max(fake.peak, open_now[0])
            try:
                if delay:
                    time.sleep(delay)
                extra = (slow or {}).get(self.path, (slow or {}).get(self.path.split("?", 1)[0]))
                if extra:
                    time.sleep(extra)
                self.respond()
            finally:
                with lock:
                    open_now[0] -= 1

        def respond(self) -> None:
            requests.append(self.path)
            headers.append(dict(self.headers))
            path_only = self.path.split("?", 1)[0]
            key = self.path if self.path in routes else path_only
            answer: Any = (
                routes.get(key)
                or routes.get("*")
                or (404, {"requestStatus": {"statusCode": "NOT_FOUND", "requestId": "test-404"}})
            )
            if isinstance(answer, list):
                index = calls.get(key, 0)
                calls[key] = index + 1
                answer = answer[min(index, len(answer) - 1)]
            status, body = answer
            if status == CLOSE[0]:
                self.close_connection = True
                return
            if 300 <= status < 400:
                self.send_response(status)
                self.send_header("Location", body)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            data = (body if isinstance(body, str) else json.dumps(body)).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    fake = FakeNgc(
        base=f"http://127.0.0.1:{server.server_address[1]}/v2", requests=requests, headers=headers
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield fake
    finally:
        server.shutdown()
        server.server_close()


# --- nemotron-3.5-lightning-30b-a3b on NGC ------------------------------------------------------

LIGHTNING_ID = "nim/nvidia/nemotron-3.5-lightning-30b-a3b"
LIGHTNING_TAGS = ["2.1.6", "2.1.5", "2.1.4", "2.0.13", "latest", "2.0.9-variant"]
# the ids of its 1-GPU profiles in fixtures/ngc: bf16, nvfp4, and nvfp4 with LoRA
LIGHTNING_BF16 = "aa2e5b4037abde0c64186023f4ffcd5947d1a40654dcd7b735e66ee3fcda34bd"
LIGHTNING_NVFP4 = "920207996e4ec77b396e25f4e360ac80331a67dcae0fec428803f556afe1d47c"
LIGHTNING_NVFP4_LORA = "15766f03e594ee013d9328b909c1b4bbd432f7f7fcdb5503372b354e34d14662"


def lightning_routes(*published: str, catalog: Route | None = None) -> dict[str, Any]:
    """NGC's nemotron-3.5-lightning-30b-a3b: its record (Deployment Geography Global, pulled
    without a key), every tag for amd64 and arm64, a search that finds no hardware variant, and
    the metadata of the tags `published` (2.1.6 alone by default) as NGC published it on
    2026-11-18 (fixtures/ngc); the other tags answer 404. The search finds the model alone; given
    `catalog`, the list's catalog page (it holds the model), the search answers that, and the
    record carries no namespace, as the list's recorded ones (fixtures/list) do not."""
    record = {
        "name": "nemotron-3.5-lightning-30b-a3b",
        "namespace": "nim/nvidia",
        "latestTag": "latest",
        "tags": LIGHTNING_TAGS,
        "canGuestPull": True,
        "description": "## Deployment Geography:\nGlobal\n",
    }
    found = [{"resourceId": LIGHTNING_ID, "name": record["name"]}]
    search: Route = (200, {"results": [{"groupValue": "CONTAINER", "resources": found}]})
    if catalog is not None:
        search = catalog
        del record["namespace"]
    archs = [{"os": "linux", "architecture": arch} for arch in ("amd64", "arm64")]
    images = [{"tag": tag, "architectureVariants": archs} for tag in LIGHTNING_TAGS]
    routes: dict[str, Any] = {
        f"/v2/repos/{LIGHTNING_ID}": (200, record),
        f"/v2/repos/{LIGHTNING_ID}/images": (200, {"images": images}),
        "/v2/search/catalog/resources/CONTAINER": search,
    }
    for tag in published or ("2.1.6",):
        document = fixture_json("ngc", f"metadata-nemotron-3.5-lightning-30b-a3b-{tag}.json")
        routes[f"/v2/nim/metadata?resourceId={LIGHTNING_ID}&tag={tag}"] = (200, document)
    return routes


# --- GLM-5.1 on TGC -----------------------------------------------------------------------------

GLM51 = "tgcr-gz.turing-agi.com/public/z-ai/glm5_1:1.0"
# tgcr's Harbor: the artifacts of the repository z-ai/{name}
GLM_HARBOR = "/api/v2.0/projects/public/repositories/z-ai%252F{}/artifacts"


def glm_routes(*tags: str) -> dict[str, Any]:
    """TGC as it answers for GLM-5.1, whose name does not hold its image path (z-ai/glm5_1): a
    search for glm5_1 finds nothing, so every item is read (real answers, the list trimmed to two
    items); tgcr's Harbor needs a login (401) for z-ai/glm5_1 and its tags `tags` (1.0 by
    default), and for z-ai/glm9, a repository no item lists."""
    tgc = "/tgc/api/resources"
    login = (401, fixture_json("cnd", "tgcr-401.json"))
    glm5_1 = GLM_HARBOR.format("glm5_1")
    return {
        tgc: (200, fixture_json("cnd", "tgc-search-glm5_1.json")),
        f"{tgc}?page=1&size=100": (200, fixture_json("cnd", "tgc-list-glm.json")),
        f"{tgc}/100332": (200, fixture_json("cnd", "tgc-detail-glm-5.1.json")),
        f"{tgc}/100301": (200, fixture_json("cnd", "tgc-detail-glm-5.json")),
        glm5_1: login,
        **{f"{glm5_1}/{tag}": login for tag in tags or ("1.0",)},
        GLM_HARBOR.format("glm9"): login,
    }


# --- the runner of run_cli ----------------------------------------------------------------------


def run_script(script: str, hooks: Mapping[str, str], arguments: list[str]) -> None:
    """Run scripts/SCRIPT as __main__ in this process, offline (keep_offline) and with the
    endpoints and files of `hooks` (TEST_HOOKS). The patches reach the script because runpy runs
    it in this interpreter: its `import ngc_api` (and the others) returns the module patched here,
    from sys.modules."""
    with pytest.MonkeyPatch.context() as patches:
        keep_offline(patches)
        for name, value in hooks.items():
            TEST_HOOKS[name](patches, value)
        sys.argv = [str(SCRIPTS / script), *arguments]
        runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    run_script(sys.argv[1], json.loads(sys.argv[2]), sys.argv[3:])
