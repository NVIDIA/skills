#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate TAO launch prerequisites before generating workflow artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

# "python3 -I <script>" leaves the folder of the script off the import path, so
# add it for the sibling modules. A run as a script writes no __pycache__ here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
if __name__ == "__main__":
    sys.dont_write_bytecode = True

try:
    import yaml
except ModuleNotFoundError:
    # One line and an exit code of its own instead of a traceback. "python3 -I" and
    # "python3 -s" leave out the user site, where pip installs for a user who is not
    # root, so the line names that case before it names the install command.
    _no_user_site = (
        " It runs without the user site (python3 -I or -s, or PYTHONNOUSERSITE), which hides "
        "a PyYAML that pip installed for the user: start the script without that option "
        "first. If PyYAML is still missing, install it with:"
    )
    print(f"error: this Python ({sys.executable}) cannot import PyYAML, which this helper "
          "script needs." + (_no_user_site if sys.flags.no_user_site else " Install it with:")
          + " python3 -m pip install pyyaml", file=sys.stderr)
    raise SystemExit(4)

from skill_layers import skill_dir  # noqa: E402
from tao_slurm_ssh import ssh_command  # noqa: E402


# The bank this script ships in, so it works from a clone or any plugin install
# without setup. $TAO_SKILL_BANK_PATH overrides it when the caller means another bank.
DEFAULT_SKILL_BANK = Path(
    os.environ.get("TAO_SKILL_BANK_PATH") or Path(__file__).resolve().parents[1]
)
REMOTE_SCHEMES = ("s3://", "azure://", "gs://", "http://", "https://")
DEFAULT_GPU_SMOKE_IMAGE = os.environ.get("TAO_GPU_SMOKE_IMAGE", "ubuntu:22.04")
DEFAULT_LOW_VRAM_THRESHOLD_GB = 50.0
KNOWN_IMAGE_SMS = {
    "cosmos-rl": ["sm_80", "sm_90", "sm_100", "sm_103", "sm_103a", "sm_120"],
    # TAO 7.x multi-architecture PyTorch and data-services images used by PAS.
    # Keeping this in the shared checker makes an unsupported target fail before
    # a workflow writes state, instead of silently skipping the gate.
    "tao-toolkit-pyt": [
        "sm_80",
        "sm_86",
        "sm_89",
        "sm_90",
        "sm_100",
        "sm_103",
        "sm_103a",
        "sm_120",
    ],
    "tao-toolkit-ds": [
        "sm_80",
        "sm_86",
        "sm_89",
        "sm_90",
        "sm_100",
        "sm_103",
        "sm_103a",
        "sm_120",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill-bank",
        type=Path,
        default=DEFAULT_SKILL_BANK,
        help="Path to the packaged TAO skill bank.",
    )
    parser.add_argument("--platform", required=True, help="TAO execution platform.")
    parser.add_argument(
        "--docker-host",
        help=(
            "Optional Docker daemon URL such as ssh://user@host. Sets "
            "DOCKER_HOST for the docker platform (remote GPU box) preflight."
        ),
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Dataset/spec path to verify. May be repeated.",
    )
    parser.add_argument(
        "--min-free-disk-gb",
        action="append",
        default=[],
        metavar="LABEL=GIB",
        help=(
            "Require at least GIB GiB free on the filesystem containing a "
            "previously supplied local --path LABEL=PATH. May be repeated."
        ),
    )
    parser.add_argument(
        "--allow-missing-path",
        action="append",
        default=[],
        metavar="LABEL",
        help=(
            "Allow a named local --path to be an approved creation target. "
            "Its nearest existing parent must be a writable, searchable directory, "
            "and no existing path component may be a symlink. May be repeated."
        ),
    )
    parser.add_argument(
        "--json-required-field",
        action="append",
        default=[],
        metavar="LABEL=FIELD[,FIELD...]",
        help=(
            "Require one or more top-level fields in sample records from a JSON "
            "annotation file identified by LABEL. May be repeated."
        ),
    )
    parser.add_argument(
        "--json-sample-limit",
        type=int,
        default=20,
        help="Number of JSON annotation records to sample for required fields.",
    )
    parser.add_argument(
        "--gpu-arch-allowlist",
        action="append",
        default=[],
        metavar="LABEL=SM[,SM...]",
        help=(
            "Require target GPU architectures to be supported by a model/image, "
            "for example cosmos_rl=sm_80,sm_90,sm_100,sm_103a,sm_120."
        ),
    )
    parser.add_argument(
        "--gpu-arch",
        action="append",
        default=[],
        metavar="SM",
        help=(
            "Known target GPU architecture such as sm_90, sm_103a, or 12.0. May be repeated. "
            "If omitted, local nvidia-smi is used when an allowlist is provided."
        ),
    )
    parser.add_argument(
        "--gpu-min-count",
        type=int,
        default=None,
        help="Require at least this many target GPUs before launch.",
    )
    parser.add_argument(
        "--gpu-min-memory-gb",
        type=float,
        default=None,
        help="Require each counted target GPU to have at least this much memory in GiB.",
    )
    parser.add_argument(
        "--gpu-min-total-memory-gb",
        type=float,
        default=None,
        help=(
            "Require the visible target GPUs to have at least this much cumulative "
            "memory in decimal GB. This does not impose a GPU count or per-GPU size."
        ),
    )
    parser.add_argument(
        "--target-gpu-count",
        type=int,
        default=None,
        help="Known target GPU count when nvidia-smi is not available on the launch host.",
    )
    parser.add_argument(
        "--target-gpu-memory-gb",
        action="append",
        type=float,
        default=[],
        help=(
            "Known target GPU memory in GiB. May be repeated once per GPU, or "
            "provided once with --target-gpu-count to apply to all target GPUs."
        ),
    )
    parser.add_argument(
        "--target-gpu-index",
        action="append",
        default=[],
        metavar="INDEX",
        help=(
            "Restrict local/remote Docker GPU validation and the smoke container "
            "to an explicitly allocated GPU index. May be repeated."
        ),
    )
    parser.add_argument(
        "--effective-batch-limit",
        action="append",
        default=[],
        metavar="LABEL=BATCH_SIZE,SHARD_COUNT",
        help=(
            "Require BATCH_SIZE <= JSON record_count / SHARD_COUNT for a "
            "previously supplied annotation path label. May be repeated."
        ),
    )
    parser.add_argument(
        "--skip-platform-access",
        action="store_true",
        help="Only validate environment variables and paths.",
    )
    parser.add_argument(
        "--install-missing-tools",
        action="store_true",
        help=(
            "Install small missing host/client tools needed for this preflight "
            "after user approval, currently awscli for s3:// path checks."
        ),
    )
    parser.add_argument(
        "--container-image",
        help=(
            "Selected TAO container image. Local Docker preflight uses this for "
            "GPU smoke checks and known image/GPU architecture compatibility."
        ),
    )
    parser.add_argument(
        "--gpu-smoke-image",
        default=DEFAULT_GPU_SMOKE_IMAGE,
        help=(
            "Fallback image for the local Docker GPU smoke test when "
            "--container-image is not supplied. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--pull-smoke-image",
        action="store_true",
        help=(
            "Allow local Docker preflight to pull the smoke image before running "
            "the GPU visibility check. Use only after user approval."
        ),
    )
    parser.add_argument(
        "--defer-container-probes",
        action="store_true",
        help=(
            "Defer checks that start containers. Use during a workflow's "
            "read-only, pre-approval discovery pass, then rerun without this "
            "flag after approval."
        ),
    )
    parser.add_argument(
        "--image-supported-sm",
        action="append",
        default=[],
        metavar="SM[,SM...]",
        help=(
            "Supported GPU architectures for the selected image, for example "
            "sm_80,sm_90,sm_100,sm_103a,sm_120. May be repeated. If omitted, known "
            "limits are inferred from --container-image when possible."
        ),
    )
    parser.add_argument(
        "--min-gpu-memory-gb",
        type=float,
        help=(
            "Fail local Docker preflight if any visible GPU has less than this "
            "much memory."
        ),
    )
    parser.add_argument(
        "--low-vram-threshold-gb",
        type=float,
        default=DEFAULT_LOW_VRAM_THRESHOLD_GB,
        help=(
            "Print a low-VRAM warning for local Docker GPUs below this memory "
            "threshold. Default: %(default)s"
        ),
    )
    return parser.parse_args()


# Docker is one daemon reachable three ways (local, DOCKER_HOST=ssh://, sudo);
# they share one self-describing skill.
PLATFORM_ALIASES = {"local-docker": "docker", "remote-docker": "docker"}


def resolve_platform(skill_bank: Path, requested: str) -> dict[str, Any]:
    """Load the platform's SELF-DESCRIBING record from its own skill —
    skills/tao-run-on-<name>/references/skill_info.yaml — with no central
    manifest. A credential-free platform may ship only a SKILL.md; then fall back to
    a minimal record and SAY SO loudly, so the empty credential gate is never silent.
    Only require-one-of credential groups and numeric wall-time caps genuinely need
    the structured record; a single required env var is checked fine from prose."""
    name = PLATFORM_ALIASES.get(requested.strip().lower(), requested.strip().lower())
    info = (
        skill_dir(skill_bank, "platform", f"tao-run-on-{name}")
        / "references" / "skill_info.yaml"
    )
    if info.is_file():
        record = yaml.safe_load(info.read_text(encoding="utf-8")) or {}
        record.setdefault("name", name)
        return record
    print(
        f"NOTE: no references/skill_info.yaml for platform '{name}' "
        f"(skills/tao-run-on-{name}/) — treating it as a self-describing, "
        f"credential-free platform. The structured credential/resource gate is empty; "
        f"rely on the skill's own Preflight prose. Add a skill_info.yaml only if this "
        f"platform needs a require-one-of credential group or a wall-time cap.",
        file=sys.stderr,
    )
    return {
        "name": name,
        "required_credentials": [],
        "credential_groups": [],
        "resource_defaults": {},
    }


def parse_paths(values: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for value in values:
        if "=" in value:
            label, path = value.split("=", 1)
        else:
            label, path = value, value
        parsed.append((label.strip() or path.strip(), path.strip()))
    return parsed


def parse_required_fields(values: list[str]) -> dict[str, list[str]]:
    fields_by_label: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(
                "--json-required-field must use LABEL=FIELD[,FIELD...] syntax"
            )
        label, fields = value.split("=", 1)
        label = label.strip()
        parsed_fields = [field.strip() for field in fields.split(",") if field.strip()]
        if not label or not parsed_fields:
            raise SystemExit(
                "--json-required-field must include a label and at least one field"
            )
        fields_by_label.setdefault(label, []).extend(parsed_fields)
    return fields_by_label


def normalize_gpu_arch(value: str) -> str:
    normalized = value.strip().lower().replace("compute_", "sm_")
    normalized = normalized.replace("-", "_")
    if not normalized:
        raise SystemExit("GPU architecture value must not be empty")
    if re.fullmatch(r"sm_?\d{2,3}[af]?", normalized):
        arch = normalized.split("_", 1)[-1] if "_" in normalized else normalized[2:]
        return "sm_" + arch
    if re.fullmatch(r"\d{2,3}[af]?", normalized):
        return "sm_" + normalized
    match = re.fullmatch(r"(\d+)\.(\d+)([af]?)", normalized)
    if match:
        major, minor, suffix = match.groups()
        return f"sm_{major}{minor}{suffix}"
    raise SystemExit(
        f"Unsupported GPU architecture format: {value}. Use sm_90, sm_103a, 90, or 9.0."
    )


def gpu_arch_is_supported(target: str, supported: set[str]) -> bool:
    """Match a compute capability to any target variant in the same SM family."""
    target_family = re.sub(r"[af]$", "", normalize_gpu_arch(target))
    return any(
        re.sub(r"[af]$", "", normalize_gpu_arch(candidate)) == target_family
        for candidate in supported
    )


def parse_gpu_arch_allowlists(values: list[str]) -> dict[str, set[str]]:
    allowlists: dict[str, set[str]] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit("--gpu-arch-allowlist must use LABEL=SM[,SM...] syntax")
        label, raw_arches = value.split("=", 1)
        label = label.strip()
        arches = {
            normalize_gpu_arch(arch)
            for arch in raw_arches.split(",")
            if arch.strip()
        }
        if not label or not arches:
            raise SystemExit(
                "--gpu-arch-allowlist must include a label and at least one SM value"
            )
        allowlists[label] = arches
    return allowlists


def parse_effective_batch_limits(values: list[str]) -> dict[str, list[tuple[int, int]]]:
    limits: dict[str, list[tuple[int, int]]] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(
                "--effective-batch-limit must use LABEL=BATCH_SIZE,SHARD_COUNT syntax"
            )
        label, raw_values = value.split("=", 1)
        parts = [part.strip() for part in raw_values.split(",") if part.strip()]
        if len(parts) != 2:
            raise SystemExit(
                "--effective-batch-limit must include exactly BATCH_SIZE,SHARD_COUNT"
            )
        try:
            batch_size = int(parts[0])
            shard_count = int(parts[1])
        except ValueError as exc:
            raise SystemExit(
                "--effective-batch-limit values must be integers"
            ) from exc
        if batch_size <= 0 or shard_count <= 0:
            raise SystemExit(
                "--effective-batch-limit values must be positive integers"
            )
        limits.setdefault(label.strip(), []).append((batch_size, shard_count))
    return limits


def parse_min_free_disk_gb(values: list[str]) -> dict[str, float]:
    requirements: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit("--min-free-disk-gb must use LABEL=GIB syntax")
        label, raw_gib = value.split("=", 1)
        label = label.strip()
        try:
            gib = float(raw_gib)
        except ValueError as exc:
            raise SystemExit("--min-free-disk-gb GIB must be numeric") from exc
        if not label or gib <= 0:
            raise SystemExit(
                "--min-free-disk-gb must include a label and a positive GIB value"
            )
        requirements[label] = gib
    return requirements


def _existing_disk_probe_path(path: Path) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise FileNotFoundError(path)
    return candidate


def check_local_creation_target(label: str, path: str) -> bool:
    """Validate a missing, approved local output root without creating it."""
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        print(f"Local creation target must be absolute: {label}={path}")
        return False
    requested = Path(os.path.abspath(requested))
    current = Path(requested.anchor)
    for part in requested.parts[1:]:
        current /= part
        if current.is_symlink():
            print(
                f"Local creation target has a symlink component: {label}={requested}"
            )
            return False
        if not current.exists():
            break
        if not current.is_dir() and current != requested:
            print(
                f"Local creation target has a non-directory parent: "
                f"{label}={requested}"
            )
            return False
    try:
        parent = _existing_disk_probe_path(requested)
    except OSError as exc:
        print(f"Local creation target cannot resolve a parent: {label}={requested}: {exc}")
        return False
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        print(
            "Local creation target parent is not writable/searchable: "
            f"{label}={requested}, parent={parent}"
        )
        return False
    print(
        "Local creation target OK: "
        f"{label}={requested} (missing; parent={parent} is writable/searchable)"
    )
    return True


def check_free_disk_space(
    paths: list[tuple[str, str]],
    requirements: dict[str, float],
    *,
    skip_access: bool,
) -> bool:
    if not requirements:
        return True
    if skip_access:
        print(
            "Free-disk requirements present; skipped filesystem capacity checks. "
            "Verify them on the target host before launch."
        )
        return True

    path_by_label = dict(paths)
    ok = True
    for label, required_gib in requirements.items():
        raw_path = path_by_label.get(label)
        if raw_path is None:
            print(f"Free-disk check failed: no --path found for label {label}")
            ok = False
            continue
        path = normalize_local_path(raw_path)
        if path is None:
            print(
                f"Free-disk check failed: {label}={raw_path} is not a local filesystem path"
            )
            ok = False
            continue
        try:
            probe = _existing_disk_probe_path(Path(path))
            free_gib = shutil.disk_usage(probe).free / 1024**3
        except OSError as exc:
            print(f"Free-disk check failed: {label}={path}: {exc}")
            ok = False
            continue
        if free_gib < required_gib:
            print(
                "Free-disk check failed: "
                f"{label}={path}, free={free_gib:.1f}GiB < required={required_gib:g}GiB"
            )
            ok = False
        else:
            print(
                "Free-disk OK: "
                f"{label}={path}, free={free_gib:.1f}GiB, required={required_gib:g}GiB"
            )
    return ok


def env_missing(platform: dict[str, Any]) -> list[str]:
    missing = []
    for item in platform.get("required_credentials", []):
        name = item.get("name")
        if item.get("source") == "env_var" and name and not os.environ.get(name):
            missing.append(name)
    for group in platform.get("credential_groups", []):
        choices = [name for name in group.get("require_one_of", []) if name]
        if choices and not any(os.environ.get(name) for name in choices):
            preferred = group.get("preferred")
            if preferred:
                missing.append(f"{preferred} ({group.get('description', 'required')})")
            else:
                missing.append("one of " + ", ".join(choices))
    return missing


def normalize_local_path(path: str) -> str | None:
    if path.startswith(REMOTE_SCHEMES):
        return None
    if path.startswith("file://"):
        return path[len("file://") :]
    if path.startswith("lustre:///"):
        return "/" + path[len("lustre:///") :].lstrip("/")
    return path


def run(
    command: list[str],
    timeout: int = 30,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "command timed out",
        )


def detect_local_gpu_arches() -> list[str]:
    ok, gpus = query_host_gpus(print_result=False)
    if not ok:
        return []
    return [gpu["sm"] for gpu in gpus if gpu.get("sm")]


def detect_local_gpu_memory_gb() -> list[float]:
    ok, gpus = query_host_gpus(print_result=False)
    if not ok:
        return []
    return [
        float(gpu["memory_mib"]) / 1024.0
        for gpu in gpus
        if gpu.get("memory_mib") is not None
    ]


def check_gpu_arch_allowlists(
    allowlists: dict[str, set[str]],
    provided_arches: list[str],
    skip_access: bool,
) -> bool:
    if not allowlists:
        return True

    if provided_arches:
        target_arches = [normalize_gpu_arch(arch) for arch in provided_arches]
    elif skip_access:
        labels = ", ".join(sorted(allowlists))
        print(
            "GPU architecture allowlist present but target GPU detection was skipped: "
            f"{labels}. Provide --gpu-arch sm_XX when the target architecture is known."
        )
        return True
    else:
        target_arches = detect_local_gpu_arches()

    if not target_arches:
        labels = ", ".join(sorted(allowlists))
        print(
            "GPU architecture check failed: could not detect target GPU architecture "
            f"for {labels}. Run on the target GPU host or provide --gpu-arch sm_XX."
        )
        return False

    ok = True
    for label, allowed in allowlists.items():
        label_ok = True
        for arch in target_arches:
            if not gpu_arch_is_supported(arch, allowed):
                print(
                    f"GPU architecture unsupported for {label}: target={arch}, "
                    f"allowed={','.join(sorted(allowed))}"
                )
                label_ok = False
                ok = False
        if label_ok:
            print(
                f"GPU architecture OK for {label}: "
                f"target={','.join(target_arches)}, allowed={','.join(sorted(allowed))}"
            )
    return ok


def check_gpu_resources(
    min_count: int | None,
    min_memory_gb: float | None,
    min_total_memory_gb: float | None,
    target_count: int | None,
    target_memory_gb: list[float],
    skip_access: bool,
) -> bool:
    if min_count is None and min_memory_gb is None and min_total_memory_gb is None:
        return True
    if min_count is not None and min_count <= 0:
        raise SystemExit("--gpu-min-count must be a positive integer")
    if min_memory_gb is not None and min_memory_gb <= 0:
        raise SystemExit("--gpu-min-memory-gb must be positive")
    if min_total_memory_gb is not None and min_total_memory_gb <= 0:
        raise SystemExit("--gpu-min-total-memory-gb must be positive")

    if target_memory_gb:
        memory_gb = list(target_memory_gb)
        if target_count and len(memory_gb) == 1:
            memory_gb = memory_gb * target_count
    elif target_count and not skip_access:
        detected_memory = detect_local_gpu_memory_gb()
        if len(detected_memory) < target_count:
            print(
                "GPU resource check failed: detected fewer GPUs than "
                f"--target-gpu-count={target_count}"
            )
            return False
        memory_gb = detected_memory[:target_count]
    elif target_count:
        if min_memory_gb is not None or min_total_memory_gb is not None:
            print(
                "GPU memory requirements cannot be verified from --target-gpu-count alone; "
                "provide --target-gpu-memory-gb or allow target access."
            )
            return False
        memory_gb = [1.0] * target_count
    elif skip_access:
        print(
            "GPU resource requirement present but target GPU detection was skipped. "
            "Provide --target-gpu-count and --target-gpu-memory-gb when the target "
            "hardware is known."
        )
        return True
    else:
        memory_gb = detect_local_gpu_memory_gb()

    if not memory_gb:
        print(
            "GPU resource check failed: could not detect target GPU memory/count. "
            "Run on the target GPU host or provide --target-gpu-count and "
            "--target-gpu-memory-gb."
        )
        return False

    if min_memory_gb is None:
        qualifying = len(memory_gb)
    else:
        qualifying = sum(1 for value in memory_gb if value >= min_memory_gb)

    required_count = min_count or 1
    if qualifying < required_count:
        detected = ",".join(f"{value:.1f}GiB" for value in memory_gb)
        print(
            "GPU resource check failed: "
            f"qualifying_gpus={qualifying} < required={required_count}; "
            f"min_memory_gb={min_memory_gb if min_memory_gb is not None else 'any'}; "
            f"detected={detected}"
        )
        return False

    # nvidia-smi reports MiB and detect_local_gpu_memory_gb converts that to GiB.
    # Convert the aggregate to decimal GB so nominal accelerator capacities can be
    # expressed without making assumptions about GPU count or per-device size.
    total_memory_gb = sum(memory_gb) * (1024**3 / 1_000_000_000)
    if min_total_memory_gb is not None and total_memory_gb < min_total_memory_gb:
        detected = ",".join(f"{value:.1f}GiB" for value in memory_gb)
        print(
            "GPU resource check failed: "
            f"total_memory_gb={total_memory_gb:.1f} < "
            f"required_total_memory_gb={min_total_memory_gb:g}; "
            f"detected={detected}"
        )
        return False

    detected = ",".join(f"{value:.1f}GiB" for value in memory_gb)
    print(
        "GPU resources OK: "
        f"qualifying_gpus={qualifying}, required={required_count}, "
        f"min_memory_gb={min_memory_gb if min_memory_gb is not None else 'any'}, "
        f"total_memory_gb={total_memory_gb:.1f}, "
        "required_total_memory_gb="
        f"{min_total_memory_gb if min_total_memory_gb is not None else 'any'}, "
        f"detected={detected}"
    )
    return True


def parse_target_gpu_indices(values: list[str]) -> list[str]:
    indices: list[str] = []
    for value in values:
        index = value.strip()
        if not index.isdigit():
            raise SystemExit("--target-gpu-index must be a nonnegative integer")
        if index not in indices:
            indices.append(index)
    return indices


def filter_target_gpus(
    gpus: list[dict[str, Any]], target_indices: list[str]
) -> tuple[bool, list[dict[str, Any]]]:
    if not target_indices:
        return True, gpus
    by_index = {str(gpu.get("index")): gpu for gpu in gpus}
    missing = [index for index in target_indices if index not in by_index]
    if missing:
        print(
            "Target GPU selection failed: missing index(es)=" + ",".join(missing)
        )
        return False, []
    selected = [by_index[index] for index in target_indices]
    print("Target GPU selection OK: indices=" + ",".join(target_indices))
    return True, selected


def docker_gpu_request(target_indices: list[str]) -> str:
    if not target_indices:
        return "all"
    # Docker parses --gpus with encoding/csv. Preserve literal quotes around a
    # comma-separated device selector even when subprocess bypasses a shell.
    return '"device=' + ",".join(target_indices) + '"'


def command_detail(result: subprocess.CompletedProcess[str]) -> str:
    detail = (result.stderr or result.stdout).strip().splitlines()
    return detail[-1] if detail else "exit " + str(result.returncode)


def docker_host_is_remote(docker_host: str | None) -> bool:
    if not docker_host:
        return False
    value = docker_host.strip()
    if not value:
        return False
    local_prefixes = ("unix://", "npipe://")
    if value.startswith(local_prefixes):
        return False
    return value not in {"/var/run/docker.sock"}


JSON_FIELD_CHECK_SCRIPT = r"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
fields = [field for field in sys.argv[2].split(",") if field]
limit = int(sys.argv[3])

with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)

if isinstance(payload, list):
    records = payload
elif isinstance(payload, dict):
    for key in ("annotations", "data", "samples", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            records = value
            break
    else:
        records = [payload]
else:
    raise SystemExit(f"unsupported JSON top-level type: {type(payload).__name__}")

if not records:
    raise SystemExit("annotation JSON has no records")

checked = records[: max(1, min(limit, len(records)))]
missing = []
for index, record in enumerate(checked):
    if not isinstance(record, dict):
        missing.append(f"record {index}: not an object")
        continue
    absent = [field for field in fields if field not in record]
    if absent:
        missing.append(f"record {index}: missing {','.join(absent)}")

if missing:
    raise SystemExit("; ".join(missing[:5]))

print(f"checked={len(checked)} fields={','.join(fields)}")
"""


def check_json_required_fields_local(
    label: str,
    path: str,
    fields: list[str],
    sample_limit: int,
) -> bool:
    command = [
        sys.executable,
        "-c",
        JSON_FIELD_CHECK_SCRIPT,
        path,
        ",".join(fields),
        str(sample_limit),
    ]
    result = run(command, timeout=30)
    if result.returncode == 0:
        print(f"JSON fields OK: {label}={path}: {result.stdout.strip()}")
        return True
    reason = (result.stderr or result.stdout).strip().splitlines()
    detail = reason[-1] if reason else "exit " + str(result.returncode)
    print(f"JSON fields missing or invalid: {label}={path}: {detail}")
    return False


def maybe_report_json_record_count(label: str, path: str) -> None:
    lowered_label = label.lower()
    lowered_name = Path(path).name.lower()
    if "annotation" not in lowered_label and "annotation" not in lowered_name:
        return
    try:
        count = json_record_count(path)
    except Exception:
        return
    if count:
        print(f"JSON record count: {label}={path}: records={count}")


def check_json_required_fields_remote(
    host: str,
    label: str,
    path: str,
    fields: list[str],
    sample_limit: int,
) -> bool:
    remote_command = " ".join(
        [
            "python3",
            "-c",
            shlex.quote(JSON_FIELD_CHECK_SCRIPT),
            shlex.quote(path),
            shlex.quote(",".join(fields)),
            str(sample_limit),
        ]
    )
    result = run(ssh_command(host, remote_command), timeout=45)
    if result.returncode == 0:
        print(f"Remote JSON fields OK: {label}={path}: {result.stdout.strip()}")
        return True
    reason = (result.stderr or result.stdout).strip().splitlines()
    detail = reason[-1] if reason else "exit " + str(result.returncode)
    print(f"Remote JSON fields missing or invalid: {label}={path}: {detail}")
    return False


def s3_paths(paths: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(label, path) for label, path in paths if path.startswith("s3://")]


def has_unverified_remote_mounts(
    platform_name: str,
    paths: list[tuple[str, str]],
    skip_access: bool,
) -> bool:
    if skip_access or platform_name in {"slurm", "docker", "local-docker", "remote-docker"}:
        return False

    failed = False
    for label, raw_path in paths:
        if raw_path.startswith(REMOTE_SCHEMES):
            continue
        print(
            f"{platform_name} path requires mounted-volume proof before launch: "
            f"{label}={raw_path}. Use s3://, or verify the mount/PVC/volume from "
            "the platform and rerun only after that manual proof exists."
        )
        failed = True
    return failed


def ensure_aws_cli(install_missing_tools: bool) -> str | None:
    aws = shutil.which("aws")
    if aws:
        return aws
    install_command = [sys.executable, "-m", "pip", "install", "awscli"]
    if not install_missing_tools:
        print(
            "aws CLI not found, so s3:// dataset paths cannot be verified. "
            "Approve remediation and rerun with --install-missing-tools, or "
            f"install manually with: {' '.join(install_command)}"
        )
        return None

    print("aws CLI not found; installing awscli with pip before S3 checks.")
    result = run(install_command, timeout=180)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        reason = detail[-1] if detail else "exit " + str(result.returncode)
        print(f"awscli install failed: {reason}")
        return None
    aws = shutil.which("aws")
    if not aws:
        print("awscli install completed, but aws is still not on PATH.")
        return None
    print(f"aws CLI installed: {aws}")
    return aws


def check_s3_storage(
    paths: list[tuple[str, str]],
    skip_access: bool,
    install_missing_tools: bool = False,
) -> bool:
    targets = s3_paths(paths)
    if not targets:
        return True

    missing = [key for key in ("ACCESS_KEY", "SECRET_KEY") if not os.environ.get(key)]
    if missing:
        print("Missing S3 requirement(s): " + ", ".join(missing))
        return False

    if skip_access:
        print("S3 credentials are present; skipped object-store access checks.")
        return True

    aws = ensure_aws_cli(install_missing_tools)
    if not aws:
        return False

    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = os.environ["ACCESS_KEY"]
    env["AWS_SECRET_ACCESS_KEY"] = os.environ["SECRET_KEY"]
    env.setdefault("AWS_DEFAULT_REGION", os.environ.get("CLOUD_REGION", "us-east-1"))

    ok = True
    for label, uri in targets:
        command = [aws]
        if os.environ.get("S3_ENDPOINT_URL"):
            command.extend(["--endpoint-url", os.environ["S3_ENDPOINT_URL"]])
        command.extend(["s3", "ls", uri])
        result = run(command, timeout=45, env=env)
        if result.returncode == 0:
            print(f"S3 path OK: {label}={uri}")
        else:
            detail = (result.stderr or result.stdout).strip().splitlines()
            reason = detail[-1] if detail else "exit " + str(result.returncode)
            print(f"S3 path missing or inaccessible: {label}={uri}: {reason}")
            ok = False
    return ok


def load_json_payload(path: str) -> Any:
    suffix = Path(path).suffix.lower()
    if path.startswith("s3://"):
        aws = shutil.which("aws")
        if not aws:
            raise RuntimeError(
                "aws CLI not found. After user approval, install it with: "
                "python -m pip install awscli"
            )
        missing = [key for key in ("ACCESS_KEY", "SECRET_KEY") if not os.environ.get(key)]
        if missing:
            raise RuntimeError("Missing S3 requirement(s): " + ", ".join(missing))
        env = os.environ.copy()
        env["AWS_ACCESS_KEY_ID"] = os.environ["ACCESS_KEY"]
        env["AWS_SECRET_ACCESS_KEY"] = os.environ["SECRET_KEY"]
        env.setdefault("AWS_DEFAULT_REGION", os.environ.get("CLOUD_REGION", "us-east-1"))
        command = [aws]
        if os.environ.get("S3_ENDPOINT_URL"):
            command.extend(["--endpoint-url", os.environ["S3_ENDPOINT_URL"]])
        command.extend(["s3", "cp", path, "-"])
        result = run(command, timeout=60, env=env)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            reason = detail[-1] if detail else "exit " + str(result.returncode)
            raise RuntimeError(f"S3 annotation download failed: {reason}")
        if suffix == ".jsonl":
            return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        return json.loads(result.stdout)

    local_path = normalize_local_path(path)
    if local_path is None:
        raise RuntimeError(f"Cannot count records for remote path: {path}")
    if Path(local_path).suffix.lower() == ".jsonl":
        with Path(local_path).open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with Path(local_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_record_count(path: str) -> int:
    payload = load_json_payload(path)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        for key in ("annotations", "data", "samples", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        else:
            records = [payload]
    else:
        raise RuntimeError(f"unsupported JSON top-level type: {type(payload).__name__}")
    if not records:
        raise RuntimeError("annotation JSON has no records")
    return len(records)


def check_effective_batch_limits(
    paths: list[tuple[str, str]],
    limits: dict[str, list[tuple[int, int]]],
    skip_access: bool,
) -> bool:
    if not limits:
        return True
    if skip_access:
        print(
            "Effective batch limits present; skipped annotation record-count checks."
        )
        return True

    path_by_label = dict(paths)
    ok = True
    count_cache: dict[str, int] = {}
    for label, label_limits in limits.items():
        path = path_by_label.get(label)
        if not path:
            print(f"Effective batch check failed: no --path found for label {label}")
            ok = False
            continue
        try:
            count = count_cache.setdefault(path, json_record_count(path))
        except Exception as exc:
            print(f"Effective batch check failed: {label}={path}: {exc}")
            ok = False
            continue
        for batch_size, shard_count in label_limits:
            max_batch = count / shard_count
            if batch_size > max_batch:
                print(
                    "Effective batch check failed: "
                    f"{label} records={count}, batch_size={batch_size}, "
                    f"shard_count={shard_count}, max_batch_per_replica={max_batch:g}"
                )
                ok = False
            else:
                print(
                    "Effective batch OK: "
                    f"{label} records={count}, batch_size={batch_size}, "
                    f"shard_count={shard_count}, max_batch_per_replica={max_batch:g}"
                )
    return ok


def parse_sm_list(values: list[str]) -> list[str]:
    sms: list[str] = []
    for value in values:
        for item in value.split(","):
            sm = item.strip()
            if sm:
                sms.append(normalize_gpu_arch(sm))
    return sms


def sm_from_compute_cap(value: str) -> str:
    compact = value.strip().replace(".", "")
    if not compact:
        return ""
    return "sm_" + compact


def known_supported_sms(image: str | None, explicit_sms: list[str]) -> list[str]:
    if explicit_sms:
        return explicit_sms
    if not image:
        return []
    lowered = image.lower()
    for token, sms in KNOWN_IMAGE_SMS.items():
        if token in lowered:
            return sms
    return []


def docker_image_exists(image: str) -> bool:
    result = run(["docker", "image", "inspect", image], timeout=20)
    return result.returncode == 0


def docker_runtimes() -> tuple[bool, set[str]]:
    result = run(["docker", "info", "--format", "{{json .Runtimes}}"], timeout=20)
    if result.returncode != 0:
        print(f"Docker runtime query failed: {command_detail(result)}")
        return False, set()
    try:
        payload = json.loads(result.stdout)
        runtimes = set(payload.keys()) if isinstance(payload, dict) else set()
    except json.JSONDecodeError:
        runtimes = set()
        if "nvidia" in result.stdout.lower():
            runtimes.add("nvidia")
    return True, runtimes


def parse_gpu_query_output(
    stdout: str, has_compute_cap: bool, *, has_uuid: bool = False
) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for row in csv.reader(stdout.splitlines()):
        minimum_columns = 5 if has_uuid else 4
        if len(row) < minimum_columns:
            continue
        offset = 1 if has_uuid else 0
        memory_mib = None
        try:
            memory_mib = float(row[3 + offset].strip())
        except ValueError:
            pass
        compute_cap_index = 4 + offset
        compute_cap = (
            row[compute_cap_index].strip()
            if has_compute_cap and len(row) > compute_cap_index
            else ""
        )
        gpus.append(
            {
                "index": row[0].strip(),
                "uuid": row[1].strip() if has_uuid else "",
                "name": row[1 + offset].strip(),
                "driver_version": row[2 + offset].strip(),
                "memory_mib": memory_mib,
                "sm": sm_from_compute_cap(compute_cap) if compute_cap else "",
            }
        )
    return gpus


def filter_cuda_visible_gpus(
    gpus: list[dict[str, Any]], raw_visibility: str | None
) -> tuple[bool, list[dict[str, Any]]]:
    """Apply the launcher process's CUDA visibility contract to host inventory.

    ``nvidia-smi`` intentionally reports physical inventory. CUDA workloads see
    only ``CUDA_VISIBLE_DEVICES`` and renumber the selected devices densely, so
    preflight must preserve both namespaces instead of treating inventory as
    availability.
    """
    physical_count = len(gpus)
    if raw_visibility is None:
        visible = [dict(gpu, visible_index=str(index)) for index, gpu in enumerate(gpus)]
        print(
            "CUDA-visible GPU comparison: "
            f"physical={physical_count}, visible={len(visible)}, "
            "source=unrestricted"
        )
        return True, visible

    raw = raw_visibility.strip()
    if not raw or raw == "-1":
        print(
            "CUDA-visible GPU comparison: "
            f"physical={physical_count}, visible=0, source=CUDA_VISIBLE_DEVICES"
        )
        return False, []

    tokens = [token.strip() for token in raw.split(",")]
    if any(not token for token in tokens):
        print("CUDA_VISIBLE_DEVICES is invalid: empty device token")
        return False, []

    by_index = {str(gpu.get("index")): gpu for gpu in gpus}
    selected: list[dict[str, Any]] = []
    selected_host_ids: set[str] = set()
    for token in tokens:
        match = by_index.get(token) if token.isdigit() else None
        if match is None and token.upper().startswith("GPU-"):
            candidates = [
                gpu
                for gpu in gpus
                if str(gpu.get("uuid", "")).upper().startswith(token.upper())
            ]
            if len(candidates) == 1:
                match = candidates[0]
        if match is None:
            print(
                "CUDA_VISIBLE_DEVICES cannot be resolved against host inventory: "
                f"token={token!r}"
            )
            return False, []
        host_id = str(match.get("index"))
        if host_id in selected_host_ids:
            print(
                "CUDA_VISIBLE_DEVICES selects the same host GPU more than once: "
                f"index={host_id}"
            )
            return False, []
        selected_host_ids.add(host_id)
        selected.append(dict(match, visible_index=str(len(selected))))

    print(
        "CUDA-visible GPU comparison: "
        f"physical={physical_count}, visible={len(selected)}, "
        "source=CUDA_VISIBLE_DEVICES"
    )
    return True, selected


def print_gpus(gpus: list[dict[str, Any]], prefix: str = "Host GPU OK") -> None:
    for gpu in gpus:
        memory = gpu["memory_mib"]
        memory_text = f"{memory / 1024:.1f}GB" if memory else "unknown"
        sm_text = gpu["sm"] or "unknown-sm"
        print(
            f"{prefix}: "
            f"index={gpu['index']} name={gpu['name']} "
            f"driver={gpu['driver_version']} memory={memory_text} arch={sm_text}"
        )


def query_host_gpus(*, print_result: bool = True) -> tuple[bool, list[dict[str, Any]]]:
    if not shutil.which("nvidia-smi"):
        print("nvidia-smi not found on host PATH")
        return False, []

    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    result = run(command, timeout=20)
    has_compute_cap = result.returncode == 0
    if not has_compute_cap:
        command = [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
        result = run(command, timeout=20)
    if result.returncode != 0:
        print(f"nvidia-smi GPU query failed: {command_detail(result)}")
        return False, []

    gpus = parse_gpu_query_output(result.stdout, has_compute_cap, has_uuid=True)
    if not gpus:
        print("nvidia-smi did not report any GPUs")
        return False, []
    visibility_ok, visible_gpus = filter_cuda_visible_gpus(
        gpus, os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if print_result and visibility_ok:
        print_gpus(visible_gpus)
    return visibility_ok, visible_gpus


def query_docker_gpus(
    image: str, pull_smoke_image: bool, target_gpu_indices: list[str]
) -> tuple[bool, list[dict[str, Any]]]:
    if not pull_smoke_image and not docker_image_exists(image):
        print(
            "Remote Docker GPU query image is not present on the Docker host: "
            f"{image}. Pull it after user approval or rerun preflight with "
            "--pull-smoke-image."
        )
        return False, []
    command = [
        "docker",
        "run",
        "--rm",
        "--runtime=nvidia",
        "--gpus",
        docker_gpu_request(target_gpu_indices),
        image,
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    result = run(command, timeout=60)
    has_compute_cap = result.returncode == 0
    if not has_compute_cap:
        command = [
            "docker",
            "run",
            "--rm",
            "--runtime=nvidia",
            "--gpus",
            docker_gpu_request(target_gpu_indices),
            image,
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
        result = run(command, timeout=60)
    if result.returncode != 0:
        print(f"Remote Docker GPU query failed: image={image}: {command_detail(result)}")
        return False, []
    gpus = parse_gpu_query_output(result.stdout, has_compute_cap)
    if not gpus:
        print("Remote Docker GPU query did not report any GPUs")
        return False, []
    print_gpus(gpus, prefix="Remote Docker GPU OK")
    return True, gpus


def check_docker_bind_path(label: str, path: str, image: str, pull_smoke_image: bool) -> bool:
    if not path.startswith("/"):
        print(f"Remote Docker path is not absolute for {label}: {path}")
        return False
    if not pull_smoke_image and not docker_image_exists(image):
        print(
            "Remote Docker path-check image is not present on the Docker host: "
            f"{image}. Pull it after user approval or rerun preflight with "
            "--pull-smoke-image."
        )
        return False
    command = [
        "docker",
        "run",
        "--rm",
        "--mount",
        f"type=bind,source={path},target=/tao_preflight_path,readonly",
        image,
        "test",
        "-e",
        "/tao_preflight_path",
    ]
    result = run(command, timeout=45)
    if result.returncode == 0:
        print(f"Remote Docker path OK: {label}={path}")
        return True
    print(f"Remote Docker path missing or inaccessible: {label}={path}: {command_detail(result)}")
    return False


def check_gpu_memory(
    gpus: list[dict[str, Any]],
    min_gpu_memory_gb: float | None,
    low_vram_threshold_gb: float,
) -> bool:
    ok = True
    for gpu in gpus:
        memory_mib = gpu.get("memory_mib")
        if memory_mib is None:
            print(f"GPU memory unknown: index={gpu.get('index')}")
            continue
        memory_gb = memory_mib / 1024
        if min_gpu_memory_gb is not None and memory_gb < min_gpu_memory_gb:
            print(
                "GPU memory below required minimum: "
                f"index={gpu['index']} memory={memory_gb:.1f}GB "
                f"< required={min_gpu_memory_gb:.1f}GB"
            )
            ok = False
        if memory_gb < low_vram_threshold_gb:
            print(
                "Low-VRAM GPU detected: "
                f"index={gpu['index']} memory={memory_gb:.1f}GB. "
                "Apply the selected model's low-VRAM profile before launch."
            )
    return ok


def check_image_architecture(
    gpus: list[dict[str, Any]],
    container_image: str | None,
    image_supported_sm: list[str],
) -> bool:
    supported = known_supported_sms(container_image, image_supported_sm)
    if not supported:
        if container_image:
            print(
                "Image architecture check skipped: no known supported SM list "
                f"for image={container_image}. Pass --image-supported-sm to enforce it."
            )
        return True

    supported_set = set(supported)
    ok = True
    for gpu in gpus:
        sm = gpu.get("sm") or ""
        if not sm:
            print(
                "Image architecture check failed: could not determine host GPU "
                f"architecture for index={gpu.get('index')}"
            )
            ok = False
            continue
        if not gpu_arch_is_supported(sm, supported_set):
            print(
                "Image architecture unsupported: "
                f"image={container_image or '<selected-image>'} "
                f"gpu_index={gpu['index']} host_arch={sm} "
                f"supported={','.join(supported)}"
            )
            ok = False
    if ok:
        print(
            "Image architecture OK: "
            f"host={','.join(sorted({gpu.get('sm') for gpu in gpus if gpu.get('sm')}))} "
            f"supported={','.join(supported)}"
        )
    return ok


def check_docker_gpu_smoke(
    container_image: str | None,
    gpu_smoke_image: str,
    pull_smoke_image: bool,
    target_gpu_indices: list[str],
) -> bool:
    image = container_image or gpu_smoke_image
    if not image:
        print("GPU smoke container check failed: no image provided")
        return False
    if not pull_smoke_image and not docker_image_exists(image):
        print(
            "GPU smoke container image is not present on the Docker host: "
            f"{image}. Pull it after user approval or rerun preflight with "
            "--pull-smoke-image."
        )
        return False

    command = [
        "docker",
        "run",
        "--rm",
        "--runtime=nvidia",
        "--gpus",
        docker_gpu_request(target_gpu_indices),
        image,
        "nvidia-smi",
        "-L",
    ]
    result = run(command, timeout=60)
    if result.returncode == 0 and "GPU" in result.stdout:
        first = result.stdout.strip().splitlines()[0]
        print(f"Docker GPU smoke OK: image={image}: {first}")
        return True
    print(f"Docker GPU smoke failed: image={image}: {command_detail(result)}")
    return False


def check_brev(platform: dict[str, Any], skip_access: bool) -> bool:
    missing = env_missing(platform)
    if missing:
        print("Missing Brev requirement(s): " + ", ".join(missing))
        return False
    if skip_access:
        print("Brev credentials are present; skipped CLI/API access check.")
        return True

    brev = shutil.which("brev")
    if not brev:
        print("brev CLI not found. Install from https://docs.nvidia.com/brev/.")
        return False

    token = os.environ.get("BREV_API_TOKEN")
    if token:
        login = run([brev, "login", "--token", token], timeout=45)
        if login.returncode != 0:
            detail = (login.stderr or login.stdout).strip().splitlines()
            reason = detail[-1] if detail else "exit " + str(login.returncode)
            print(f"Brev token login failed: {reason}")
            return False

    result = run([brev, "ls", "--json"], timeout=60)
    if result.returncode != 0 and token:
        # Headless `brev ls` occasionally hits an auth-EOF even after a
        # successful token login — the cached session desyncs. Force one
        # refresh and retry before declaring failure.
        run([brev, "login", "--token", token], timeout=45)
        result = run([brev, "ls", "--json"], timeout=60)
    if result.returncode == 0:
        print("Brev CLI/API OK")
        return True
    detail = (result.stderr or result.stdout).strip().splitlines()
    reason = detail[-1] if detail else "exit " + str(result.returncode)
    print(f"Brev CLI/API check failed: {reason}")
    return False


def kubectl_base() -> list[str]:
    command = ["kubectl"]
    if os.environ.get("KUBECONFIG"):
        command.extend(["--kubeconfig", os.environ["KUBECONFIG"]])
    if os.environ.get("TAO_K8S_CONTEXT"):
        command.extend(["--context", os.environ["TAO_K8S_CONTEXT"]])
    return command


def check_kubernetes(platform: dict[str, Any], skip_access: bool) -> bool:
    missing = env_missing(platform)
    if missing:
        print("Missing Kubernetes requirement(s): " + ", ".join(missing))
        return False
    if skip_access:
        print("Kubernetes environment is present; skipped cluster access check.")
        return True

    if not shutil.which("kubectl"):
        print("kubectl not found. Install kubectl or run from inside the cluster.")
        return False

    namespace = os.environ.get("TAO_K8S_NAMESPACE", "default")
    base = kubectl_base()
    auth = run(base + ["auth", "can-i", "create", "jobs", "-n", namespace], timeout=30)
    if auth.returncode != 0 or auth.stdout.strip().lower() != "yes":
        detail = (auth.stderr or auth.stdout).strip().splitlines()
        reason = detail[-1] if detail else "not allowed"
        print(f"Kubernetes job-create permission check failed: {reason}")
        return False

    nodes = run(base + ["get", "nodes", "-o", "json"], timeout=45)
    if nodes.returncode != 0:
        detail = (nodes.stderr or nodes.stdout).strip().splitlines()
        reason = detail[-1] if detail else "exit " + str(nodes.returncode)
        print(f"Kubernetes node/GPU check failed: {reason}")
        return False
    try:
        payload = json.loads(nodes.stdout)
        gpu_total = 0
        for node in payload.get("items", []):
            allocatable = node.get("status", {}).get("allocatable", {})
            value = allocatable.get("nvidia.com/gpu", "0")
            gpu_total += int(str(value))
    except Exception as exc:
        print(f"Kubernetes node/GPU check failed: could not parse nodes JSON: {exc}")
        return False
    if gpu_total <= 0:
        print("Kubernetes node/GPU check failed: no allocatable nvidia.com/gpu found")
        return False
    print(f"Kubernetes API OK: namespace={namespace}, allocatable_gpus={gpu_total}")
    return True


def check_slurm(
    platform: dict[str, Any],
    paths: list[tuple[str, str]],
    required_json_fields: dict[str, list[str]],
    json_sample_limit: int,
    skip_access: bool,
) -> bool:
    ok = True
    missing = env_missing(platform)
    if missing:
        print("Missing SLURM requirement(s): " + ", ".join(missing))
        if any("SSH_KEY_PATH" in item for item in missing):
            print(
                "Provide SSH_KEY_PATH=/path/to/private_key. To set up passwordless "
                "access: 1) ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519; "
                "2) ssh-copy-id -i ~/.ssh/id_ed25519.pub $SLURM_USER@<login-host>; "
                "3) ssh-keyscan -H <login-host> >> ~/.ssh/known_hosts; "
                "4) chmod 600 ~/.ssh/id_ed25519; 5) verify with "
                "ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 "
                "$SLURM_USER@<login-host> 'hostname'."
            )
        return False

    key_path = os.environ.get("SSH_KEY_PATH")
    if key_path and not Path(key_path).expanduser().exists():
        print(f"SSH_KEY_PATH does not exist: {key_path}")
        return False

    hosts = [host.strip() for host in os.environ["SLURM_HOSTNAME"].split(",") if host.strip()]
    if not hosts:
        print("SLURM_HOSTNAME did not contain any hosts")
        return False

    if not check_slurm_runtime(platform):
        return False

    working_host = ""
    if not skip_access:
        for host in hosts:
            try:
                socket.getaddrinfo(host, 22)
            except socket.gaierror as exc:
                print(f"Host did not resolve: {host} ({exc})")
                continue
            result = run(ssh_command(host, "echo TAO_SSH_OK"), timeout=25)
            if result.returncode == 0 and "TAO_SSH_OK" in result.stdout:
                working_host = host
                print(f"Passwordless SSH OK: {host}")
                break
            reason = (result.stderr or result.stdout).strip().splitlines()
            detail = reason[-1] if reason else "exit " + str(result.returncode)
            print(f"Passwordless SSH failed: {host}: {detail}")

        if not working_host:
            print(
                "SLURM preflight failed before artifact generation. Provide "
                "SSH_KEY_PATH=/path/to/private_key for a key accepted by at least "
                "one SLURM login host. To set up passwordless access: "
                "1) create a key if needed: ssh-keygen -t ed25519 -N '' -f "
                "~/.ssh/id_ed25519; 2) install it once: ssh-copy-id -i "
                "~/.ssh/id_ed25519.pub $SLURM_USER@<login-host>; 3) trust the "
                "host key: ssh-keyscan -H <login-host> >> ~/.ssh/known_hosts; "
                "4) lock permissions: chmod 600 ~/.ssh/id_ed25519; 5) verify: "
                "ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 "
                "$SLURM_USER@<login-host> 'hostname'. Then rerun with "
                "SSH_KEY_PATH set to the private key path."
            )
            return False
    else:
        working_host = hosts[0]

    if not skip_access:
        runtime_command = (
            "command -v sbatch >/dev/null && "
            "command -v srun >/dev/null && "
            "command -v sacct >/dev/null && "
            "command -v enroot >/dev/null && "
            "srun --help 2>&1 | grep -q -- --container-image"
        )
        result = run(ssh_command(working_host, runtime_command), timeout=30)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            reason = detail[-1] if detail else f"exit {result.returncode}"
            print(
                "Remote SLURM/Pyxis/Enroot preflight failed: "
                f"host={working_host}: {reason}"
            )
            return False
        print(f"Remote SLURM/Pyxis/Enroot tools OK: {working_host}")

    for label, raw_path in paths:
        path = normalize_local_path(raw_path)
        if path is None:
            continue
        if not path.startswith("/"):
            print(f"SLURM dataset path is not absolute for {label}: {raw_path}")
            ok = False
            continue
        if skip_access:
            continue
        result = run(ssh_command(working_host, f"test -e {shlex.quote(path)}"), timeout=25)
        if result.returncode == 0:
            print(f"Remote path OK: {label}={path}")
        else:
            print(f"Remote path missing or inaccessible: {label}={path}")
            ok = False
            continue
        fields = required_json_fields.get(label)
        if fields and not check_json_required_fields_remote(
            working_host,
            label,
            path,
            fields,
            json_sample_limit,
        ):
            ok = False

    return ok


def parse_float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        raise SystemExit(f"{name} must be a number of hours, got: {value}")


def check_slurm_runtime(platform: dict[str, Any]) -> bool:
    defaults = platform.get("resource_defaults", {})
    default_partition = defaults.get("partition")
    default_time = float(defaults.get("time_hours", 4))
    default_timeout = float(defaults.get("timeout_hours", max(default_time - 0.2, 0.1)))
    # A wall-time limit belongs to the user's partition, so none is assumed: it is
    # enforced only when SLURM_MAX_TIME_HOURS or the platform metadata names one.
    max_time = parse_float_env("SLURM_MAX_TIME_HOURS")
    if max_time is None and defaults.get("max_time_hours") is not None:
        max_time = float(defaults["max_time_hours"])

    requested_partition = os.environ.get("SLURM_PARTITION", default_partition)
    requested_time = parse_float_env("SLURM_TIME_HOURS")
    requested_timeout = parse_float_env("SLURM_TIMEOUT_HOURS")

    if requested_time is None:
        print(
            "SLURM runtime default: "
            f"partition={requested_partition}, SLURM_TIME_HOURS={default_time:g}, "
            f"SLURM_TIMEOUT_HOURS={default_timeout:g}"
        )
        requested_time = default_time
    else:
        print(
            "SLURM runtime requested: "
            f"partition={requested_partition}, SLURM_TIME_HOURS={requested_time:g}"
        )

    if requested_timeout is None:
        requested_timeout = min(default_timeout, max(requested_time - 0.1, 0.1))
    print(f"SLURM internal timeout: SLURM_TIMEOUT_HOURS={requested_timeout:g}")

    if max_time is not None and requested_time > max_time:
        print(
            "SLURM runtime exceeds the configured wall-time limit: "
            f"requested {requested_time:g}h > max {max_time:g}h. "
            f"Use SLURM_TIME_HOURS={max_time:g} or less, or raise "
            "SLURM_MAX_TIME_HOURS when the partition allows a longer wall time."
        )
        return False
    if requested_timeout >= requested_time:
        print(
            "SLURM_TIMEOUT_HOURS must be smaller than SLURM_TIME_HOURS "
            f"({requested_timeout:g} >= {requested_time:g})."
        )
        return False
    return True


def check_local_docker(
    paths: list[tuple[str, str]],
    required_json_fields: dict[str, list[str]],
    json_sample_limit: int,
    skip_access: bool,
    container_image: str | None,
    gpu_smoke_image: str,
    pull_smoke_image: bool,
    image_supported_sm: list[str],
    min_gpu_memory_gb: float | None,
    low_vram_threshold_gb: float,
    require_remote_docker: bool,
    target_gpu_indices: list[str],
    defer_container_probes: bool,
    allow_missing_paths: set[str] | None = None,
) -> bool:
    ok = True
    allow_missing_paths = allow_missing_paths or set()
    effective_target_gpu_indices = list(target_gpu_indices)
    docker_host = os.environ.get("DOCKER_HOST")
    remote_docker = docker_host_is_remote(docker_host)
    if require_remote_docker and not remote_docker:
        print(
            "Missing remote Docker requirement: set DOCKER_HOST to a remote "
            "daemon URL such as ssh://user@gpu-host."
        )
        return False
    if not skip_access:
        if not shutil.which("docker"):
            print("docker executable not found")
            ok = False
        else:
            result = run(["docker", "info"], timeout=20)
            if result.returncode == 0:
                print("Docker daemon OK")
            else:
                print(f"Docker daemon check failed: {command_detail(result)}")
                ok = False

            runtime_query_ok, runtimes = docker_runtimes()
            if runtime_query_ok and "nvidia" in runtimes:
                print("Docker NVIDIA runtime OK")
            elif runtime_query_ok:
                print(
                    "Docker NVIDIA runtime missing: install/configure "
                    "NVIDIA Container Toolkit before launch."
                )
                ok = False
            else:
                ok = False

            if remote_docker:
                print(
                    "Remote Docker daemon requested: "
                    f"DOCKER_HOST={os.environ.get('DOCKER_HOST')}"
                )
                if not defer_container_probes:
                    gpu_ok, gpus = query_docker_gpus(
                        gpu_smoke_image, pull_smoke_image, target_gpu_indices
                    )
                else:
                    gpu_ok, gpus = False, []
                    print(
                        "Remote Docker GPU/container checks deferred until approval; "
                        "rerun without --defer-container-probes after approval."
                    )
            else:
                gpu_ok, gpus = query_host_gpus()
                selection_ok, gpus = filter_target_gpus(gpus, target_gpu_indices)
                gpu_ok = gpu_ok and selection_ok
                if gpu_ok and not effective_target_gpu_indices:
                    effective_target_gpu_indices = [str(gpu["index"]) for gpu in gpus]
            if not (remote_docker and defer_container_probes):
                ok = gpu_ok and ok
            if gpu_ok:
                ok = (
                    check_gpu_memory(
                        gpus,
                        min_gpu_memory_gb,
                        low_vram_threshold_gb,
                    )
                    and ok
                )
                ok = (
                    check_image_architecture(
                        gpus,
                        container_image,
                        image_supported_sm,
                    )
                    and ok
                )
            if gpu_ok and not defer_container_probes:
                ok = (
                    check_docker_gpu_smoke(
                        container_image,
                        gpu_smoke_image,
                        pull_smoke_image,
                        effective_target_gpu_indices,
                    )
                    and ok
                )
            elif gpu_ok:
                print(
                    "Docker GPU smoke container deferred until approval; rerun with "
                    "the --defer-container-probes flag removed after approval."
                )
            else:
                if not (remote_docker and defer_container_probes):
                    print(
                        "Docker GPU smoke skipped: effective GPU allocation failed validation"
                    )

    for label, raw_path in paths:
        path = normalize_local_path(raw_path)
        if path is None:
            continue
        if remote_docker:
            if skip_access:
                print(f"Remote Docker path accepted without access check: {label}={path}")
                continue
            if defer_container_probes:
                print(
                    "Remote Docker bind-path check deferred until approval: "
                    f"{label}={path}"
                )
                continue
            if not check_docker_bind_path(label, path, gpu_smoke_image, pull_smoke_image):
                ok = False
                continue
            fields = required_json_fields.get(label)
            if fields:
                print(
                    "Remote Docker JSON field sampling skipped: "
                    f"{label}={path}. Validate required fields from a mounted "
                    "container image before launch if this model requires them."
                )
            continue
        if Path(path).exists():
            print(f"Local path OK: {label}={path}")
            maybe_report_json_record_count(label, path)
        elif label in allow_missing_paths:
            if label in required_json_fields:
                print(
                    f"Missing creation target cannot satisfy JSON input checks: {label}={path}"
                )
                ok = False
                continue
            ok = check_local_creation_target(label, path) and ok
            continue
        else:
            print(f"Local path missing: {label}={path}")
            ok = False
            continue
        fields = required_json_fields.get(label)
        if fields and not check_json_required_fields_local(
            label,
            path,
            fields,
            json_sample_limit,
        ):
            ok = False
    return ok


def check_env_only(platform: dict[str, Any], paths: list[tuple[str, str]]) -> bool:
    missing = env_missing(platform)
    if missing:
        print("Missing requirement(s): " + ", ".join(missing))
        return False
    for label, raw_path in paths:
        if raw_path.startswith(REMOTE_SCHEMES):
            print(f"Path accepted for remote platform: {label}={raw_path}")
        else:
            print(
                "Path provided for remote platform, verify it is mounted in the "
                f"job: {label}={raw_path}"
            )
    return True


def main() -> int:
    args = parse_args()
    if args.docker_host:
        os.environ["DOCKER_HOST"] = args.docker_host
    platform = resolve_platform(args.skill_bank, args.platform)
    paths = parse_paths(args.path)
    required_json_fields = parse_required_fields(args.json_required_field)
    gpu_arch_allowlists = parse_gpu_arch_allowlists(args.gpu_arch_allowlist)
    effective_batch_limits = parse_effective_batch_limits(args.effective_batch_limit)
    min_free_disk_gb = parse_min_free_disk_gb(args.min_free_disk_gb)
    allow_missing_paths = set(args.allow_missing_path)
    unknown_missing_labels = allow_missing_paths - {label for label, _ in paths}
    if unknown_missing_labels:
        raise SystemExit(
            "--allow-missing-path label(s) have no matching --path: "
            + ", ".join(sorted(unknown_missing_labels))
        )
    target_gpu_indices = parse_target_gpu_indices(args.target_gpu_index)
    name = platform["name"]
    remote_docker_requested = docker_host_is_remote(os.environ.get("DOCKER_HOST"))
    local_docker_paths = name in {"docker", "local-docker"} and not remote_docker_requested
    if allow_missing_paths and not local_docker_paths:
        raise SystemExit(
            "--allow-missing-path is supported only for local Docker paths; "
            "use the selected remote platform's native path contract"
        )

    if name == "slurm":
        platform_ok = check_slurm(
            platform,
            paths,
            required_json_fields,
            args.json_sample_limit,
            args.skip_platform_access,
        )
    elif name in {"docker", "local-docker", "remote-docker"}:
        platform_ok = check_local_docker(
            paths,
            required_json_fields,
            args.json_sample_limit,
            args.skip_platform_access,
            args.container_image,
            args.gpu_smoke_image,
            args.pull_smoke_image,
            parse_sm_list(args.image_supported_sm),
            args.min_gpu_memory_gb,
            args.low_vram_threshold_gb,
            name == "remote-docker" or remote_docker_requested,
            target_gpu_indices,
            args.defer_container_probes,
            allow_missing_paths,
        )
    elif name == "brev":
        platform_ok = check_brev(platform, args.skip_platform_access)
    elif name == "kubernetes":
        platform_ok = check_kubernetes(platform, args.skip_platform_access)
    else:
        platform_ok = check_env_only(platform, paths)

    storage_ok = check_s3_storage(
        paths,
        args.skip_platform_access,
        args.install_missing_tools,
    )
    mounts_ok = not has_unverified_remote_mounts(
        name,
        paths,
        args.skip_platform_access,
    )
    gpu_arch_ok = check_gpu_arch_allowlists(
        gpu_arch_allowlists,
        args.gpu_arch,
        args.skip_platform_access,
    )
    gpu_resources_ok = check_gpu_resources(
        args.gpu_min_count,
        args.gpu_min_memory_gb,
        args.gpu_min_total_memory_gb,
        args.target_gpu_count,
        args.target_gpu_memory_gb,
        args.skip_platform_access,
    )
    effective_batch_ok = check_effective_batch_limits(
        paths,
        effective_batch_limits,
        args.skip_platform_access,
    )
    free_disk_ok = check_free_disk_space(
        paths,
        min_free_disk_gb,
        skip_access=args.skip_platform_access,
    )
    ok = (
        platform_ok
        and storage_ok
        and mounts_ok
        and gpu_arch_ok
        and gpu_resources_ok
        and effective_batch_ok
        and free_disk_ok
    )

    if ok:
        print("TAO launch preflight passed")
        return 0
    print("TAO launch preflight failed")
    return 2


if __name__ == "__main__":
    sys.exit(main())
