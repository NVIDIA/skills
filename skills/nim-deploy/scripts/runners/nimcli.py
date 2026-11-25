# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The nimcli runner: one `nim-run.sh` that starts a NIM with the nim CLI (`nim run`).

It implements the runner contract in runners/__init__.py. Facts about the nim CLI:
- `nim run MODEL -p PORT --gpus G -d --no-auto-start [--profile ID] [-e KEY=VALUE]` are clap flags;
  `--gpus` takes `all` or device ids such as `0,1`.
- nim reads NGC_API_KEY and HF_TOKEN itself (environment or config) and refuses -e NGC_API_KEY,
  so no secret is ever on the command line; `-e` carries only non-secret KEY=VALUE pairs.
- `nim run` force-removes any container named `nim-<leaf>` before it starts (leaf: the last path
  segment of the model, lowercased, tag and registry host dropped, nim/ cnd/ public/
  cnd-enterprise/ prefixes stripped), so validate refuses when that name belongs to something else.
- stdin comes from /dev/null so nim never waits on a prompt.
"""

from __future__ import annotations

import re
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nd_common as nd
from nd_common import RenderError, whole_number

SCRIPT = "nim-run.sh"
NAMESPACE_PREFIXES = ("cnd/", "public/", "nim/", "cnd-enterprise/")
# Environment that nim itself owns; it refuses these in -e.
MANAGED_ENV = {
    "NGC_API_KEY",
    "NIM_DISABLE_MODEL_DOWNLOAD",
    "NIM_MODEL_PROFILE",
    "NIM_NODE_MANAGER_PORT",
    "NIM_PRIMARY_NODE",
    "NIM_SERVER_PORT",
    "NIM_TRUST_REMOTE_CODE",
}
INPUT_HELP = """\
nim CLI run inputs (required: model port):
  model              model name or image reference, as `nim run` takes it
  port               host port (-p); the API is on container port 8000
  gpus               all (default) or device ids such as 0,1 (--gpus)
  model_profile      profile id (--profile); default: nim picks
  model_path         model-free images: -e NIM_MODEL_PATH=hf://org/model
  served_model_name  -e NIM_SERVED_MODEL_NAME=name (default: unset)
  env                extra non-secret variables, {NAME: value} (-e NAME=value)
nim reads NGC_API_KEY and HF_TOKEN from your environment or its own config.
"""
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SECRET_NAME = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
_SECRET_VALUE = re.compile(r"nvapi-|^hf_[A-Za-z0-9]{8,}")


def _deny(rule_id: str, pattern: str, reason: str) -> dict[str, str]:
    return {"id": rule_id, "pattern": pattern, "reason": reason}


def deny_rules() -> list[dict[str, str]]:
    stop_fix = (
        "Fix: never run the blanket command. List what runs with read-only 'nim list' or "
        "'docker ps', ask the user which one to stop, and act only on one this skill created, "
        "with its printed command."
    )
    return [
        _deny(
            "nim-diag-fix",
            r"^nim{F} diag( .*)? --fix(=\S*)?( |$)",
            "'nim diag --fix' runs sudo commands, including chmod 666 on /var/run/docker.sock, "
            "and restarts docker. Fix: run 'nim diag' without --fix and show the findings.",
        ),
        _deny(
            "nim-all",
            r"^nim{F} (stop|rm)( .*)? --all(=\S*)?( |$)",
            f"'--all' acts on every nim container. {stop_fix}",
        ),
        _deny(
            "nim-uninstall",
            r"^nim{F} uninstall( |$)",
            "'nim uninstall' removes nim and, with --purge, all models. Fix: do not run it. "
            "Say what it removes; to free GPUs, list what runs with 'nim list' and "
            "ask the user which one to stop.",
        ),
        _deny(
            "nim-config-key",
            r"^nim{F} config set (ngc-api-key|nvidia-api-key)( |$)",
            "This writes an API key in plain text to ~/.nim/config.toml. "
            "Fix: the user exports NGC_API_KEY.",
        ),
        _deny(
            "nim-registry-password",
            r"^nim{F} config registry login( .*)? (-p|--password)( |=|$)",
            "A password on the command line leaks into the shell history. "
            "Fix: the user logs in themselves.",
        ),
        _deny(
            "nim-raw-mutation",
            r"^nim{F} (run|pull|launch|update)( |$)",
            "Start a NIM only through the approved run script. Fix: run facts.next_command from "
            "'nimdeploy.py status RUN' after the user approved the run.",
        ),
    ]


def model_name(model: str) -> str:
    """nim's normalized model name."""
    value = model.strip()
    head, _, rest = value.partition("/")
    if rest and (head.lower() == "localhost" or "." in head or ":" in head):
        value = rest  # a full image reference: drop the registry host
    value = value.split("@", 1)[0]
    if value.rfind(":") > value.rfind("/"):
        value = value[: value.rfind(":")]
    value = value.strip().strip("/").lower()
    while prefix := next((p for p in NAMESPACE_PREFIXES if value.startswith(p)), None):
        value = value[len(prefix) :]
    return value


def model_leaf(model: str) -> str:
    """The `<leaf>` in the `nim-<leaf>` container name (default_container_name)."""
    return model_name(model).rsplit("/", 1)[-1]


def is_image_ref(model: str) -> bool:
    head, _, rest = model.partition("/")
    return bool(rest) and (head.lower() == "localhost" or "." in head or ":" in head)


def read_only(argv: list[str]) -> bool:
    """`nim list|info|diag|search` and the version and help flags never change anything."""
    if not argv or argv[0] != "nim" or "--fix" in argv:
        return False
    return argv[1:2] in (["list"], ["info"], ["diag"], ["search"]) or argv[1:] in (
        ["--version"],
        ["-V"],
        ["--help"],
        ["-h"],
    )


def required_inputs() -> tuple[str, ...]:
    return ("model", "port")


def container_name(inputs: dict[str, Any]) -> str:
    return f"nim-{model_leaf(str(inputs.get('model') or ''))}"


def ready_url(inputs: dict[str, Any]) -> str:
    return f"http://127.0.0.1:{inputs.get('port') or 8000}"


def apply_command(state: dict[str, Any]) -> list[str]:
    return ["bash", str(nd.run_dir(state["run_id"]) / SCRIPT)]


def rollback_command(state: dict[str, Any]) -> list[str]:
    return ["nim", "stop", str(nd.input_value(state, "model"))]


def guard_forms(approval: Any) -> list[list[str]]:
    """The approved script and `nim stop MODEL`: nothing else."""
    model = str(approval.record.get("model") or "")
    return [["bash", str(Path(approval.folder) / SCRIPT)], ["nim", "stop", model]]


def _gpus(value: Any) -> str:
    text = str(value if value not in (None, "") else "all").strip()
    if text == "all" or re.fullmatch(r"[0-9]+(,[0-9]+)*", text):
        return text
    raise RenderError(f"gpus must be all or device ids such as 0,1, got {value!r}")


def _env_pairs(inputs: dict[str, Any]) -> list[str]:
    extra = inputs.get("env") or {}
    if not isinstance(extra, dict):
        raise RenderError("env must be a mapping of NAME to value")
    pairs = [("NIM_MODEL_PATH", inputs["model_path"])] if inputs.get("model_path") else []
    if inputs.get("served_model_name"):
        pairs.append(("NIM_SERVED_MODEL_NAME", inputs["served_model_name"]))
    pairs += [(name, extra[name]) for name in sorted(extra)]
    result = []
    for name, value in pairs:
        text = str(value)
        if not _ENV_NAME.fullmatch(name) or any(ord(char) < 32 for char in text):
            raise RenderError(f"env {name!r} is not a valid variable")
        if name in MANAGED_ENV:
            raise RenderError(f"env {name} is set by nim itself; use the matching input instead")
        if _SECRET_NAME.search(name) or _SECRET_VALUE.search(text):
            raise RenderError(f"env {name} looks like a credential; nim reads credentials itself")
        result.append(f"{name}={text}")
    return result


def render_text(inputs: dict[str, Any]) -> str:
    model = str(inputs.get("model") or "")
    if not _MODEL.fullmatch(model) or not model_leaf(model):
        raise RenderError("model must be a model name or image reference without spaces")
    port = whole_number(inputs, "port")
    if port > 65535:
        raise RenderError("port must be between 1 and 65535")
    if nd.parse_extra_args(inputs.get("extra_args")):
        raise RenderError(
            "the nim CLI runner cannot pass engine arguments (extra_args); use the docker runner"
        )
    words = ["nim", "run", model, "-p", str(port), "--gpus", _gpus(inputs.get("gpus")), "-d"]
    words.append("--no-auto-start")
    if inputs.get("model_profile"):
        words += ["--profile", str(inputs["model_profile"])]
    for pair in _env_pairs(inputs):
        words += ["-e", pair]
    lines = [
        "#!/usr/bin/env bash",
        f"# Rendered by nim-deploy for run {inputs.get('run_id', '')}. Starts one NIM (nim CLI).",
        "# nim reads NGC_API_KEY and HF_TOKEN from your environment or config; no secret here.",
        "set -euo pipefail",
        f"{shlex.join(words)} </dev/null",
    ]
    return "\n".join(lines) + "\n"


def render(inputs: dict[str, Any], run_dir: Path) -> list[Path]:
    path = run_dir / SCRIPT
    nd.write_atomic(path, render_text(inputs), mode=0o700)
    return [path]


# --- preflight ------------------------------------------------------------------------------


def _port_free(port: Any) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # All interfaces on purpose: nim run -p PORT publishes on every address,
            # so a listener on any of them would conflict. The socket never accepts.
            sock.bind(("0.0.0.0", int(port)))  # nosec B104
        return True
    except OSError:
        return False


def collision(state: dict[str, Any]) -> dict[str, str] | None:
    """An error when `nim run` would force-remove a container this run does not own."""
    model = str(nd.input_value(state, "model") or "")
    name = container_name({"model": model})
    fields = '{{.Image}}|{{.Label "nim.managed"}}|{{.Label "nim.model.name"}}'
    argv = ["docker", "ps", "-a", "--filter", f"name=^/{name}$", "--format", fields]
    found = subprocess.run(
        argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False
    )
    line = found.stdout.strip().splitlines()[0] if found.stdout.strip() else ""
    if not line:
        return None
    image, managed, model_label = ([*line.split("|"), "", ""])[:3]
    fix = f"Stop. Ask the user whether {name} may be removed, or deploy under another model name."
    if managed != "true":
        message = (
            f"container {name} exists and is not managed by nim; `nim run` would force-remove it"
        )
        return nd.error("name_collision", message, fix)
    same = (
        image.lower() == model.lower() if is_image_ref(model) else model_label == model_name(model)
    )
    if not same:
        message = f"container {name} runs {image or model_label}; `nim run` would force-remove it"
        return nd.error("name_collision", message, fix)
    return None


def preflight(state: dict[str, Any], offline: bool) -> tuple[str, list[dict[str, str]]]:
    """nim on PATH, a free host port, and no foreign container under the name nim will use."""
    if offline:
        return "skipped", []
    errors: list[dict[str, str]] = []
    if not shutil.which("nim"):
        errors.append(
            nd.error(
                "nim_missing", "the nim CLI is not on PATH", "Ask the user to install the nim CLI"
            )
        )
    port = nd.input_value(state, "port")
    if port and not _port_free(port):
        fix = "Ask the user for a free port: nimdeploy.py set RUN port=NNNN, then re-render"
        errors.append(nd.error("port_in_use", f"host port {port} is busy", fix))
    if shutil.which("docker"):
        clash = collision(state)
        if clash:
            errors.append(clash)
    return ("failed" if errors else "ok"), errors
