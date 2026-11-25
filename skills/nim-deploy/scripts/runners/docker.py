# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The docker runner: one `docker-run.sh` that starts a NIM LLM container.

It implements the runner contract in runners/__init__.py.

The script has GPUs, /dev/shm, NGC_API_KEY passed by name only, a host model cache mounted at
/opt/nim/.cache and the OpenAI API on the container port. That port is not assumed: it comes from
the user, the image's own metadata, docs-supplied facts or, last, the documented NIM default (8000),
and the source is recorded as `port_source`. It is rendered into the run folder,
hashed at approval and run exactly as `bash <run folder>/docker-run.sh`.
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

from runners.imageinfo import inspect_image, resolve_container_port

DEFAULT_GPUS = "all"
CONTAINER_CACHE = "/opt/nim/.cache"
RESTART_POLICIES = ("no", "always", "unless-stopped", "on-failure")
SCRIPT = "docker-run.sh"
INPUT_HELP = """\
docker run inputs (required: image; a model-free image also needs model_path):
  image              NIM image, e.g. nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8
  name               container name (default: the image name without registry path, tag, digest)
  container_port     the port the container serves on (default: from the image metadata, else 8000)
  port               host port mapped to container_port (default: container_port)
  gpus               all (default) | a count (2) | device ids ("0,1")
  cache_dir          host model cache mounted at /opt/nim/.cache (default ~/.cache/nim)
  shm_size           /dev/shm size (default 16g)
  model_profile      NIM_MODEL_PROFILE (default: unset, NIM auto-selects)
  model_path         NIM_MODEL_PATH for a model-free NIM, e.g. hf://org/model (default: unset)
  served_model_name  NIM_SERVED_MODEL_NAME (default: unset)
  pass_hf_token      also pass HF_TOKEN by name from your environment (default false)
  env                extra non-secret environment, {NAME: value} (default: none)
  extra_args         engine arguments after the image, e.g. "--max-num-seqs 64" (default: none)
  run_as_host_user   add -u "$(id -u):0" so the cache stays writable (default true)
  restart            docker restart policy (default unless-stopped)
The NGC_API_KEY value is never read from the run or written to the script; export it first.
"""
_CONTAINER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SHM = re.compile(r"[1-9][0-9]*([kmg]b?)?", re.IGNORECASE)
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SECRET_NAME = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
_SECRET_VALUE = re.compile(r"nvapi-|^hf_[A-Za-z0-9]{8,}")
_HOME_SAFE = re.compile(r"[^\"$`\\\n]*")


__all__ = [
    "inspect_image",
    "resolve_container_port",
]  # image metadata lives in runners/imageinfo.py


def required_inputs() -> tuple[str, ...]:
    return ("image",)  # the model is inside a regular NIM image; name, port and gpus default


def is_model_free(image: Any) -> bool:
    return "model-free" in str(image or "").rsplit("/", 1)[-1].lower()


def missing_inputs(inputs: dict[str, Any]) -> list[str]:
    """A model-free image has no model inside: the user names the source (hf://org/model)."""
    missing = (
        ["model_path"]
        if is_model_free(inputs.get("image")) and not inputs.get("model_path")
        else []
    )
    several = "," in str(inputs.get("image_ports") or "") and not inputs.get("container_port")
    return missing + (["container_port"] if several else [])


def default_name(image: str) -> str:
    """The image name without registry path, tag and digest, as a valid container name."""
    leaf = image.split("@", 1)[0].rsplit("/", 1)[-1].split(":", 1)[0].lower()
    leaf = re.sub(r"[^a-z0-9_.-]+", "-", leaf).strip("-._")[:128]
    return leaf or "nim"


def default_inputs(
    inputs: dict[str, Any], previous: dict[str, Any] | None = None
) -> dict[str, dict[str, str]]:
    """Inputs the user did not give, as {key: {value, source, note}} (see nimdeploy.apply_defaults).

    `inputs` holds the user-given and docs-supplied values; `previous` the recorded entries, so an
    unchanged image is not looked up again."""
    found = {"gpus": {"value": DEFAULT_GPUS, "source": "default"}}
    image = str(inputs.get("image") or "")
    if image:
        found["name"] = {"value": default_name(image), "source": "default"}
    container = inputs.get("container_port")
    if image:
        recorded = (previous or {}).get("image_ports") or {}
        if recorded.get("for_image") == image:  # reuse the earlier lookup
            for key in ("container_port", "image_ports"):
                entry = (previous or {}).get(key)
                if entry and entry.get("source") not in (None, "user", "docs"):
                    found[key] = {
                        k: entry[k] for k in ("value", "source", "note", "for_image") if k in entry
                    }
        else:
            res = resolve_container_port(image)
            found["image_ports"] = {
                "value": ",".join(res["ports"]),
                "source": "image-metadata",
                "note": res["note"],
                "for_image": image,
            }
            if res["port"]:
                found["container_port"] = {
                    "value": res["port"],
                    "source": res["source"],
                    "note": res["note"],
                    "for_image": image,
                }
    if container:
        found.pop("container_port", None)
    resolved = container or found.get("container_port", {}).get("value")
    if resolved:
        found["port"] = {"value": str(resolved), "source": "default"}
    return found


def container_name(inputs: dict[str, Any]) -> str:
    return str(inputs.get("name") or "")


def deny_rules() -> list[dict[str, str]]:
    return []  # docker's own denies live in guard-rules.json


def ready_url(inputs: dict[str, Any]) -> str:
    return f"http://localhost:{inputs.get('port') or 8000}"


def _state_inputs(state: dict[str, Any]) -> dict[str, Any]:
    return nd.run_inputs(state)


def apply_command(state: dict[str, Any]) -> list[str]:
    return ["bash", str(nd.run_dir(state["run_id"]) / SCRIPT)]


def rollback_command(state: dict[str, Any]) -> list[str]:
    return ["docker", "rm", "-f", container_name(_state_inputs(state))]


def guard_forms(approval: Any) -> list[list[str]]:
    """The approved script, and stop or remove of the approved container."""
    name = str(approval.record.get("name") or "")
    return [
        ["bash", str(Path(approval.folder) / SCRIPT)],
        ["docker", "stop", name],
        ["docker", "rm", name],
        ["docker", "rm", "-f", name],
    ]


# --- render ---------------------------------------------------------------------------------


def _gpus(value: Any) -> str:
    if value == "all":
        return "all"
    if isinstance(value, list):
        devices = [str(item) for item in value]
    elif isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        if int(value) < 1:
            raise RenderError("gpus count must be at least 1")
        return str(int(value))
    elif isinstance(value, str):
        devices = value.removeprefix("device=").split(",")
    else:
        raise RenderError("gpus must be all, a count, or device ids")
    if not devices or not all(device.strip().isdigit() for device in devices):
        raise RenderError(f"gpus device ids must be numbers, got {value!r}")
    return shlex.quote('"device=' + ",".join(device.strip() for device in devices) + '"')


def _cache_expression(path: str) -> dict[str, str]:
    """Shell text for the cache dir; `~/` becomes "$HOME/..." so the shell expands it."""
    if path.startswith("~/"):
        rest = path[2:]
        if not _HOME_SAFE.fullmatch(rest):
            raise RenderError('cache_dir must not contain ", $, ` or \\')
        return {"dir": f'"$HOME/{rest}"', "volume": f'"$HOME/{rest}:{CONTAINER_CACHE}"'}
    if not path.startswith("/") or ":" in path or any(ord(char) < 32 for char in path):
        raise RenderError("cache_dir must be an absolute path (or start with ~/) without ':'")
    return {"dir": shlex.quote(path), "volume": shlex.quote(f"{path}:{CONTAINER_CACHE}")}


def _env_pair(variable: str, value: Any) -> str:
    text = str(value)
    if not _ENV_NAME.fullmatch(variable):
        raise RenderError(f"env name {variable!r} is not a valid variable name")
    if _SECRET_NAME.search(variable) or _SECRET_VALUE.search(text):
        raise RenderError(
            f"env {variable} looks like a credential; credentials are passed by name only"
        )
    if any(ord(char) < 32 for char in text):
        raise RenderError(f"env {variable} contains control characters")
    return shlex.quote(f"{variable}={text}")


def render_text(inputs: dict[str, Any]) -> str:
    image, name = inputs.get("image"), inputs.get("name")
    if not isinstance(image, str) or not nd.IMAGE_REFERENCE.fullmatch(image):
        raise RenderError("image must be a container image reference")
    if not isinstance(name, str) or not _CONTAINER_NAME.fullmatch(name):
        raise RenderError("name must be a docker container name: letters, digits, _ . - (max 128)")
    port, container_port = whole_number(inputs, "port"), whole_number(inputs, "container_port")
    if port > 65535 or container_port > 65535:
        raise RenderError("port and container_port must be between 1 and 65535")
    shm = str(inputs.get("shm_size", "16g"))
    if not _SHM.fullmatch(shm):
        raise RenderError("shm_size must look like 16g")
    restart = inputs.get("restart", "unless-stopped")
    if restart not in RESTART_POLICIES:
        raise RenderError(f"restart must be one of {', '.join(RESTART_POLICIES)}")
    cache = _cache_expression(str(inputs.get("cache_dir", "~/.cache/nim")))
    pass_hf_token = nd.flag_input(inputs, "pass_hf_token")
    env = ["NGC_API_KEY", *(["HF_TOKEN"] if pass_hf_token else [])]
    for key, variable in (
        ("model_profile", "NIM_MODEL_PROFILE"),
        ("model_path", "NIM_MODEL_PATH"),
        ("served_model_name", "NIM_SERVED_MODEL_NAME"),
    ):
        if inputs.get(key):
            env.append(_env_pair(variable, inputs[key]))
    extra_env = inputs.get("env") or {}
    if not isinstance(extra_env, dict):
        raise RenderError("env must be a mapping of NAME to value")
    env += [_env_pair(variable, extra_env[variable]) for variable in sorted(extra_env)]
    flags = [
        f"--name {shlex.quote(name)}",
        f"--label {shlex.quote('nim-deploy.run=' + str(inputs.get('run_id', '')))}",
        f"--gpus {_gpus(inputs.get('gpus'))}",
        f"--shm-size={shm}",
    ]
    if nd.flag_input(inputs, "run_as_host_user", True):
        flags.append('-u "$(id -u):0"')
    flags += [f"-e {item}" for item in env]
    flags += [f"-v {cache['volume']}", f"-p {port}:{container_port}", f"--restart {restart}"]
    flags.append(shlex.quote(image))
    flags += [shlex.quote(arg) for arg in nd.parse_extra_args(inputs.get("extra_args"))]
    lines = [
        "#!/usr/bin/env bash",
        f"# Rendered by nim-deploy for run {inputs.get('run_id', '')}. Starts one NIM container.",
        "# NGC_API_KEY is read from your environment; its value is never written to this file.",
        "set -euo pipefail",
        ': "${NGC_API_KEY:?export NGC_API_KEY before running this script}"',
    ]
    if pass_hf_token:
        lines.append(': "${HF_TOKEN:?export HF_TOKEN before running this script}"')
    lines += [f"mkdir -p {cache['dir']}", "docker run -d \\"]
    lines += [f"  {flag} \\" for flag in flags[:-1]]
    lines.append(f"  {flags[-1]}")
    return "\n".join(lines) + "\n"


def render(inputs: dict[str, Any], run_dir: Path) -> list[Path]:
    path = run_dir / SCRIPT
    nd.write_atomic(path, render_text(inputs), mode=0o700)
    return [path]


# --- preflight ------------------------------------------------------------------------------


def port_free(port: Any) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # All interfaces on purpose: docker run -p PORT:... publishes on every address,
            # so a listener on any of them would conflict. The socket never accepts.
            sock.bind(("0.0.0.0", int(port)))  # nosec B104
        return True
    except OSError:
        return False


def _tool(argv: list[str]) -> subprocess.CompletedProcess[str]:
    nd.log("validate: running " + " ".join(argv))
    return subprocess.run(
        argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120, check=False
    )


def preflight(state: dict[str, Any], offline: bool) -> tuple[str, list[dict[str, str]]]:
    """Free host port, docker reachable with an NVIDIA runtime, and no container of that name."""
    if offline:
        return "skipped", []
    errors: list[dict[str, str]] = []
    port, name = nd.input_value(state, "port"), nd.input_value(state, "name")
    if port and not port_free(port):
        fix = "Ask the user for a free port: nimdeploy.py set RUN port=NNNN, then re-render"
        errors.append(nd.error("port_in_use", f"host port {port} is busy", fix))
    if not shutil.which("docker"):
        fix = "Ask the user to install Docker"
        return "failed", [*errors, nd.error("docker_missing", "docker is not on PATH", fix)]
    info = _tool(["docker", "info", "--format", "{{json .}}"])
    if info.returncode != 0:
        fix = "Ask the user to start Docker or join the docker group"
        errors.append(
            nd.error("docker_missing", "\n".join(info.stderr.strip().splitlines()[-5:]), fix)
        )
    elif "nvidia" not in info.stdout.lower():
        fix = (
            "Ask the user to install the NVIDIA Container Toolkit and run: "
            "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
        )
        errors.append(nd.error("no_nvidia_runtime", "docker info shows no NVIDIA runtime", fix))
    found = _tool(["docker", "ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"])
    if found.stdout.strip():
        fix = f"Ask the user: pick another name (nimdeploy.py set RUN name=...) or remove {name}"
        errors.append(nd.error("name_in_use", f"a container named {name} already exists", fix))
    return ("failed" if errors else "ok"), errors
