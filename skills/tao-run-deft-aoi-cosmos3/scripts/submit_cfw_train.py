#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Compose or execute the Docker submit command for Framework CR3 Train.

The caller owns the unchanged platform four-verb flow: open the job-record,
pass its id here, mark it RUNNING, then use Docker status/logs/cancel.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in supported TAO host environments
    import tomli as tomllib

from render_cfw_sft import load_toml, validate_config as validate_rendered_config


IMAGE_KEY = "images.tao_toolkit.cosmos_framework"
CONTAINER_CONFIG = "/tao/config/train.toml"
CONTAINER_RESULTS = "/tao/results"
CONTAINER_ADAPTER = (
    "/workspace/.venv/lib/python3.13/site-packages/cosmos_framework/"
    "configs/base/reasoner/experiment/tao_cr3_aoi.py"
)


def _username() -> str:
    return os.environ.get("USER") or os.environ.get("LOGNAME") or "tao"


def validate_submission_config(config: dict[str, Any]) -> None:
    """Enforce the current renderer contract at the submit boundary."""
    validate_rendered_config(config)


def validate_existing_dcp_output(
    results_dir: pathlib.Path,
    *,
    resume_checkpoint: pathlib.Path | None,
    allow_existing_dcp: bool,
) -> None:
    """Refuse an implicit restart from a completed checkpoint tree."""
    if not results_dir.exists():
        return
    if not results_dir.is_dir():
        raise ValueError(f"--results-dir must identify a directory: {results_dir}")
    dcp_dirs = sorted(
        {
            metadata.parent.resolve()
            for metadata in results_dir.rglob(".metadata")
            if metadata.is_file()
        },
        key=str,
    )
    if not dcp_dirs or allow_existing_dcp:
        return
    if resume_checkpoint is not None and resume_checkpoint.resolve() in dcp_dirs:
        return
    found = ", ".join(str(path) for path in dcp_dirs)
    raise ValueError(
        "target DCP output directory already contains checkpoint .metadata at "
        f"{found}; refusing submission because Framework would resume from the "
        "capped checkpoint, run zero optimizer steps, and abort with zero valid "
        "training labels. Set checkpoint.load_path to that exact DCP or pass "
        "--allow-existing-dcp only for intentional reuse"
    )


def default_bank() -> pathlib.Path:
    configured = os.environ.get("TAO_SKILL_BANK_PATH")
    return pathlib.Path(configured).expanduser().resolve() if configured else pathlib.Path(__file__).resolve().parents[4]


def resolve_image(skill_bank: pathlib.Path) -> tuple[str, str]:
    resolver = skill_bank / "scripts/resolve_versions_key.py"
    completed = subprocess.run(
        [sys.executable, str(resolver), "--skill-bank", str(skill_bank), IMAGE_KEY],
        check=True,
        capture_output=True,
        text=True,
    )
    tagged = completed.stdout.strip()
    if not tagged:
        raise ValueError(f"resolver returned no value for {IMAGE_KEY}")
    inspected = subprocess.run(
        ["docker", "image", "inspect", tagged],
        check=True,
        capture_output=True,
        text=True,
    )
    image = json.loads(inspected.stdout)[0]
    digests = image.get("RepoDigests") or []
    repository = tagged.rsplit(":", 1)[0]
    immutable = next((value for value in digests if value.startswith(repository + "@")), None)
    if not immutable:
        raise ValueError(f"local image has no repository digest for resolver value {tagged}")
    return tagged, immutable


def _mount(host: pathlib.Path, container: str, *, readonly: bool) -> str:
    suffix = ",readonly" if readonly else ""
    return f"type=bind,src={host},dst={container}{suffix}"


def parse_extra_mount(value: str) -> tuple[pathlib.Path, str, bool]:
    parts = value.rsplit(":", 2)
    readonly = bool(parts and parts[-1] == "ro")
    if readonly:
        parts = parts[:-1]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("mount must be HOST:CONTAINER[:ro]")
    host = pathlib.Path(parts[0]).expanduser().resolve(strict=True)
    container = parts[1]
    if not container.startswith("/"):
        raise argparse.ArgumentTypeError("mount container path must be absolute")
    return host, container, readonly


def _under(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def annotation_identity_mounts(
    annotation_host: pathlib.Path, media_host: pathlib.Path
) -> list[tuple[pathlib.Path, str, bool]]:
    """Return least-privilege mounts for absolute paths in ShareGPT rows.

    Framework's interim adapter receives the host-authored JSON unchanged.
    Relative paths resolve through the regular compute-frame media mount;
    absolute paths must also exist at that exact path inside the container.
    """
    try:
        records = json.loads(annotation_host.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"training annotation must be UTF-8 JSON: {annotation_host}"
        ) from exc
    if not isinstance(records, list):
        raise ValueError("training annotation must contain one JSON array")

    media_host = media_host.resolve(strict=True)
    mounts: list[tuple[pathlib.Path, str, bool]] = [
        (media_host, str(media_host), True)
    ]
    destinations = {str(media_host)}
    for index, record in enumerate(records):
        images = record.get("images") if isinstance(record, dict) else None
        if not isinstance(images, list) or len(images) != 2 or not all(
            isinstance(image, str) and image for image in images
        ):
            raise ValueError(
                f"training annotation record[{index}].images must contain "
                "exactly two non-empty paths in [AOI, golden_reference] order"
            )
        for image in images:
            annotated = pathlib.Path(image).expanduser()
            if not annotated.is_absolute():
                continue
            source = annotated.resolve(strict=True)
            if _under(source, media_host):
                continue
            source = annotated.parent.resolve(strict=True)
            destination = os.path.normpath(str(annotated.parent))
            if destination in destinations:
                continue
            mounts.append((source, destination, True))
            destinations.add(destination)
    return mounts


def build_docker_argv(
    *,
    immutable_image: str,
    job_id: str,
    config_host: pathlib.Path,
    model_host: pathlib.Path,
    annotation_host: pathlib.Path,
    media_host: pathlib.Path,
    results_dir: pathlib.Path,
    adapter_host: pathlib.Path,
    config: dict[str, Any],
    gpus: str,
    identity_mounts: list[tuple[pathlib.Path, str, bool]],
    extra_mounts: list[tuple[pathlib.Path, str, bool]],
    resume_mount: tuple[pathlib.Path, str, bool] | None,
    offline: bool,
) -> list[str]:
    custom = config["custom"]
    model_container = config["model"]["backbone"]["model_name"]
    annotation_container = custom["annotation_path"]
    media_container = custom["media_root"]
    for label, value in (
        ("model", model_container),
        ("annotation", annotation_container),
        ("media root", media_container),
    ):
        if not isinstance(value, str) or not value.startswith("/"):
            raise ValueError(f"rendered {label} compute-frame path must be absolute")

    username = _username()
    command = [
        "docker", "run", "-d",
        "--name", job_id,
        "--label", f"tao-job={job_id}",
        "--gpus", gpus,
        "--shm-size=8g",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", f"USER={username}",
        "-e", f"LOGNAME={username}",
        "-e", "HOME=/tmp",
        "-v", "/etc/passwd:/etc/passwd:ro",
        "-v", "/etc/group:/etc/group:ro",
        "--mount", _mount(config_host, CONTAINER_CONFIG, readonly=True),
        "--mount", _mount(adapter_host, CONTAINER_ADAPTER, readonly=True),
        "--mount", _mount(model_host, model_container, readonly=True),
        "--mount", _mount(annotation_host, annotation_container, readonly=True),
        "--mount", _mount(results_dir, CONTAINER_RESULTS, readonly=False),
        "-w", f"{CONTAINER_RESULTS}/cwd",
        "-e", f"IMAGINAIRE_OUTPUT_ROOT={CONTAINER_RESULTS}",
        "-e", f"TAO_CR3_TRAIN_ANNOTATION={annotation_container}",
        "-e", f"TAO_CR3_MEDIA_ROOT={media_container}",
        "-e", f"TAO_CR3_SEED={custom['seed']}",
        "-e", f"TAO_JOB_ID={job_id}",
        "-e", f"TAO_RESULTS_ROOT={CONTAINER_RESULTS}",
        "-e", "TOKENIZERS_PARALLELISM=false",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "RANK=0",
        "-e", "WORLD_SIZE=1",
        "-e", "LOCAL_RANK=0",
        "-e", "LOCAL_WORLD_SIZE=1",
        "-e", "MASTER_ADDR=127.0.0.1",
        "-e", "MASTER_PORT=29500",
    ]
    if offline:
        command.extend(["-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1"])
    mounts = [*identity_mounts, *extra_mounts]
    if resume_mount is not None:
        mounts.append(resume_mount)
    for host, container, readonly in mounts:
        command.extend(["--mount", _mount(host, container, readonly=readonly)])
    command.extend(
        [
            "--entrypoint", "/workspace/.venv/bin/cosmos-framework-train",
            immutable_image,
            f"--sft-toml={CONTAINER_CONFIG}",
            "--",
            f"trainer.seed={custom['seed']}",
            f"model.config.parallelism.fsdp_master_dtype={custom['fsdp_master_dtype']}",
            f"model.config.policy.model_max_length={custom['model_max_length']}",
        ]
    )
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-bank", type=pathlib.Path, default=default_bank())
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--model-host", required=True, type=pathlib.Path)
    parser.add_argument("--annotation-host", required=True, type=pathlib.Path)
    parser.add_argument("--media-host", required=True, type=pathlib.Path)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--gpus", default="all")
    parser.add_argument("--mount", action="append", default=[], type=parse_extra_mount)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--allow-existing-dcp",
        action="store_true",
        help="Allow an existing checkpoint .metadata in --results-dir.",
    )
    parser.add_argument("--execute", action="store_true", help="Launch detached; default is a read-only plan")
    parser.add_argument("--output", type=pathlib.Path, help="Optional JSON plan path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_host = args.config.expanduser().resolve(strict=True)
        model_host = args.model_host.expanduser().resolve(strict=True)
        annotation_host = args.annotation_host.expanduser().resolve(strict=True)
        media_host = args.media_host.expanduser().resolve(strict=True)
        results_dir = args.results_dir.expanduser().resolve()
        adapter_host = pathlib.Path(__file__).with_name("cfw_cr3_aoi_adapter.py").resolve(strict=True)
        config = load_toml(config_host)
        validate_submission_config(config)
        resume_value = config.get("checkpoint", {}).get("load_path", "???")
        resume_mount = None
        if resume_value != "???":
            resume_host = pathlib.Path(resume_value).expanduser().resolve(strict=True)
            if not resume_host.is_dir():
                raise ValueError("checkpoint.load_path must identify a Framework DCP directory")
            resume_mount = (resume_host, str(resume_host), True)
        validate_existing_dcp_output(
            results_dir,
            resume_checkpoint=resume_mount[0] if resume_mount else None,
            allow_existing_dcp=args.allow_existing_dcp,
        )
        identity_mounts = annotation_identity_mounts(
            annotation_host, media_host
        )
        tagged, immutable = resolve_image(args.skill_bank.expanduser().resolve(strict=True))
        command = build_docker_argv(
            immutable_image=immutable,
            job_id=args.job_id,
            config_host=config_host,
            model_host=model_host,
            annotation_host=annotation_host,
            media_host=media_host,
            results_dir=results_dir,
            adapter_host=adapter_host,
            config=config,
            gpus=args.gpus,
            identity_mounts=identity_mounts,
            extra_mounts=args.mount,
            resume_mount=resume_mount,
            offline=args.offline,
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "train_backend": "cosmos-framework",
            "image_key": IMAGE_KEY,
            "resolved_image": tagged,
            "image_digest": immutable,
            "job_id": args.job_id,
            "config": str(config_host),
            "results_dir": str(results_dir),
            "identity_mounts": [
                _mount(host, container, readonly=readonly)
                for host, container, readonly in identity_mounts
            ],
            "resume_checkpoint": str(resume_mount[0]) if resume_mount else None,
            "allow_existing_dcp": args.allow_existing_dcp,
            "command_argv": command,
            "command": shlex.join(command),
            "executed": False,
        }
        if args.execute:
            results_dir.mkdir(parents=True, exist_ok=True)
            (results_dir / "cwd").mkdir(exist_ok=True)
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            payload["executed"] = True
            payload["backend_ref"] = completed.stdout.strip()
            manifest = results_dir / "train_submission.json"
            manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.output:
            args.output.expanduser().resolve().write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError, tomllib.TOMLDecodeError) as exc:
        print(f"submit_cfw_train: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
