# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the test_*.py files in this folder (not a script)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nd_common as nd

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
GOLDEN = TESTS / "golden"
CONTRACT_KEYS = {"ok", "state", "next", "facts", "errors"}
# Fake credentials for the "refuse secret values" tests, built at run time so that no
# secret-looking literal sits in the repository (secret scanners flag those).
FAKE_NVAPI = "nvapi" + "-" + "q" * 30
FAKE_HF = "hf" + "_" + "y" * 34


OFFLINE_TOOL = "#!/usr/bin/env bash\necho 'offline test double' >&2\nexit 1\n"


@pytest.fixture
def runs_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the run store at a temporary folder; hide the real docker and skopeo.

    Image metadata lookups then fail (no network in tests); tests that need metadata put their
    own fake docker or skopeo first on PATH."""
    home = tmp_path / "nd"
    monkeypatch.setenv("NIM_DEPLOY_HOME", str(home))
    offline = tmp_path / "offline-bin"
    offline.mkdir()
    for name in ("docker", "skopeo"):
        (offline / name).write_text(OFFLINE_TOOL)
        (offline / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{offline}{os.pathsep}{os.environ['PATH']}")
    from runners import docker as docker_runner

    docker_runner.inspect_image.cache_clear()
    return home


def docs_entry(**change: Any) -> dict[str, Any]:
    """A fresh online docs check, as `nimdeploy.py docs-check` records it."""
    entry: dict[str, Any] = {
        "mode": "online",
        "urls": [
            "https://docs.nvidia.com/nim/large-language-models/latest/get-started/quickstart.html"
        ],
        "doc_version": "2.1.4",
        "note": "port=8000",
        "reason": "",
        "fetched_at": nd.utc_now().isoformat(),
    }
    return {**entry, **change}


def with_docs(state: dict[str, Any], **change: Any) -> dict[str, Any]:
    """Record a docs check in a run state and save it."""
    state["docs_check"] = docs_entry(**change)
    nd.save_state(state)
    return state


def docs_cli(run: str, env: dict[str, str]) -> None:
    code, _ = run_cli(
        "nimdeploy.py", "docs-check", run, "--url", "https://docs.nvidia.com/nim/x.html",
        "--doc-version", "2.1.4", "--note", "port=8000", env=env,
    )  # fmt: skip
    assert code == 0


def assert_contract(payload: dict[str, Any]) -> None:
    """The object matches write-skill's script-output.schema.json."""
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
    script: str, *args: str, env: dict[str, str] | None = None, stdin: str | None = None
) -> tuple[int, dict[str, Any]]:
    """Run a script as a subprocess; return (exit code, the one JSON object on stdout)."""
    # Test-only runners such as _fake_runner live here, so the script must be able to import them.
    child_env = dict(os.environ if env is None else env)
    paths = [str(TESTS), child_env.get("PYTHONPATH", "")]
    child_env["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        input=stdin,
        stdin=None if stdin is not None else subprocess.DEVNULL,
        env=child_env,
        timeout=60,
        check=False,
    )
    lines = proc.stdout.strip().splitlines()
    assert len(lines) == 1, proc.stdout + proc.stderr
    payload = json.loads(lines[0])
    assert_contract(payload)
    return proc.returncode, payload


def make_run(
    values: dict[str, Any],
    target: str = "nim-operator",
    state: str = "render",
    run_id: str = "run1",
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "run_id": run_id,
        "state": state,
        "target": target,
        "inputs": {key: {"value": value, "source": "user"} for key, value in values.items()},
        "artifacts": {},
        "attempts": {},
        "history": [],
        "approvals": [],
    }
    nd.save_state(record)
    return record


def fake_tool(folder: Path, name: str, body: str) -> None:
    """Write an executable shell script `name` into FOLDER (to put on PATH)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text("#!/usr/bin/env bash\n" + body + "\n")
    path.chmod(0o755)


DOCKER_VALUES: dict[str, Any] = {
    "image": "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8",
    "gpus": "all",
    "port": "8000",
    "container_port": "8000",
    "name": "nim-llama",
    "model": "meta/llama-3.1-8b-instruct",
}
K8S_VALUES: dict[str, Any] = {
    "image": "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8",
    "namespace": "nim-prod",
    "context": "prod-a",
    "name": "llama",
    "ngc_secret": "ngc-api-secret",
    "gpus": "1",
    "model": "meta/llama-3.1-8b-instruct",
}
