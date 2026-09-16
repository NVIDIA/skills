#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Cosmos3-Nano Omni weights for the selected backend loader.

The caller supplies the selected backend runtime. The helper keeps the
converter implementation and Nano architecture mapping internal and records
source, runtime, and output checkpoint provenance.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_NANO_VLM_ARCHITECTURE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CONVERTER_ENTRYPOINT_MODULE = "cosmos_rl.model_preparation.vlm_safetensors"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_uri(value: str) -> bool:
    return "://" in value or ("/" in value and not Path(value).expanduser().exists())


def huggingface_model_id(value: str) -> str:
    """Translate user-facing Hugging Face URI schemes to a Hub repository ID."""
    if value.startswith("hf_model://"):
        return value.removeprefix("hf_model://").strip("/")
    if value.startswith("hf://"):
        return value.removeprefix("hf://").removeprefix("models/").strip("/")
    return value


def identity(value: str) -> dict[str, Any]:
    path = Path(value).expanduser()
    return {
        "original": value,
        "resolved": str(path.resolve()) if path.exists() else None,
        "kind": "uri" if is_uri(value) else "local",
    }


def validate(path: Path) -> dict[str, Any]:
    config_file = path / "config.json"
    if not config_file.is_file():
        raise ValueError(f"prepared checkpoint is missing config.json: {path}")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl":
        raise ValueError(
            f"prepared model_type must be qwen3_vl, found {config.get('model_type')!r}"
        )
    weights = sorted(path.glob("*.safetensors"))
    index = path / "model.safetensors.index.json"
    if not weights and not index.is_file():
        raise ValueError("prepared checkpoint has no safetensors weights/index")
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        missing = sorted(
            {name for name in weight_map.values() if not (path / name).is_file()}
        )
        if missing:
            raise ValueError(
                f"prepared checkpoint is missing indexed shards: {missing[:10]}"
            )
    required_processor = ("tokenizer_config.json", "tokenizer.json")
    missing_processor = [
        name for name in required_processor if not (path / name).is_file()
    ]
    if missing_processor:
        raise ValueError(
            f"prepared checkpoint is missing tokenizer files: {missing_processor}"
        )
    files = []
    for file in sorted(path.iterdir()):
        if file.is_file() and (file.suffix in {".json", ".safetensors", ".jinja"}):
            files.append(
                {"name": file.name, "size": file.stat().st_size, "sha256": sha256(file)}
            )
    return {
        "model_type": "qwen3_vl",
        "files": files,
        "fingerprint": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
    }


def docker_mount(value: str, container_root: str) -> tuple[list[str], str]:
    path = Path(value).expanduser()
    if not path.exists():
        return [], value
    resolved = path.resolve()
    container = f"{container_root}/{resolved.name}"
    return ["-v", f"{resolved}:{container}:ro"], container


def command(args: argparse.Namespace, output: Path, cache: Path) -> list[str]:
    source_input = huggingface_model_id(args.base_model_path_or_uri)
    architecture_input = huggingface_model_id(args.vlm_architecture_model_path_or_uri)
    source_mount, source = docker_mount(source_input, "/inputs/base")
    donor_mount, donor = docker_mount(architecture_input, "/inputs/architecture")
    script = r"""
set -Eeuo pipefail
source_value="$BASE_MODEL"
architecture_value="$ARCHITECTURE_MODEL"
if [[ "$BASE_MODEL_KIND" == "uri" ]]; then
  source_value="$(python - <<'PY'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(os.environ['BASE_MODEL'], revision=os.environ['BASE_MODEL_REVISION'], cache_dir='/cache/huggingface'))
PY
)"
fi
if [[ "$ARCHITECTURE_MODEL_KIND" == "uri" ]]; then
  architecture_value="$(python - <<'PY'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(os.environ['ARCHITECTURE_MODEL'], revision=os.environ['ARCHITECTURE_MODEL_REVISION'], cache_dir='/cache/huggingface'))
PY
)"
fi
python -m cosmos_rl.model_preparation.vlm_safetensors \
  --checkpoint-path "$source_value" --output-path "/output/$OUTPUT_NAME" \
  --vlm-model-name "$architecture_value"
"""
    runtime_user = os.environ.get("USER") or os.environ.get("LOGNAME") or getpass.getuser() or "tao"
    result = [
        # Run as the invoking user so the host-side validation pass can read
        # the prepared files; the selected backend image keeps its venv readable.
        "docker",
        "run",
        "--rm",
        "--shm-size=8g",
        "--entrypoint",
        "bash",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        f"USER={runtime_user}",
        "-e",
        f"LOGNAME={runtime_user}",
        "-e",
        "HOME=/cache/tao-home",
        "-e",
        "XDG_CACHE_HOME=/cache/tao-home/.cache",
        "-e",
        "TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor",
        "-e",
        f"BASE_MODEL={source}",
        "-e",
        f"BASE_MODEL_KIND={'uri' if is_uri(source_input) else 'local'}",
        "-e",
        f"BASE_MODEL_REVISION={args.base_model_revision}",
        "-e",
        f"ARCHITECTURE_MODEL={donor}",
        "-e",
        f"ARCHITECTURE_MODEL_KIND={'uri' if is_uri(architecture_input) else 'local'}",
        "-e",
        f"ARCHITECTURE_MODEL_REVISION={args.vlm_architecture_model_revision}",
        "-e",
        f"OUTPUT_NAME={output.name}",
        "-e",
        "HF_HOME=/cache/huggingface",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-v",
        f"{output.parent}:/output",
        "-v",
        f"{cache}:/cache",
        *source_mount,
        *donor_mount,
    ]
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name):
            result.extend(["-e", name])
    result.extend([args.runtime_image, "-lc", script])
    return result


def resolve_input(value: str, revision: str, cache: Path) -> str:
    """Resolve an immutable URI inside the selected preparation container."""
    if not is_uri(value):
        return str(Path(value).expanduser().resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(
        huggingface_model_id(value),
        revision=revision,
        cache_dir=str(cache / "huggingface"),
    )


def inside_container_command(
    args: argparse.Namespace,
    output: Path,
    cache: Path,
) -> list[str]:
    source = resolve_input(args.base_model_path_or_uri, args.base_model_revision, cache)
    architecture = resolve_input(
        args.vlm_architecture_model_path_or_uri,
        args.vlm_architecture_model_revision,
        cache,
    )
    return [
        sys.executable,
        "-m",
        CONVERTER_ENTRYPOINT_MODULE,
        "--checkpoint-path",
        source,
        "--output-path",
        str(output),
        "--vlm-model-name",
        architecture,
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path-or-uri", required=True)
    parser.add_argument("--base-model-revision", default="")
    parser.add_argument(
        "--vlm-architecture-model-path-or-uri",
        default=DEFAULT_NANO_VLM_ARCHITECTURE_MODEL,
    )
    parser.add_argument("--vlm-architecture-model-revision", default="")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--runtime-image-digest", required=True)
    parser.add_argument(
        "--inside-container",
        action="store_true",
        help="Run the packaged converter directly in the already selected backend runtime.",
    )
    parser.add_argument("--base-model-identity", default="")
    parser.add_argument("--vlm-architecture-model-identity", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for value, revision, label in (
        (args.base_model_path_or_uri, args.base_model_revision, "base model"),
        (
            args.vlm_architecture_model_path_or_uri,
            args.vlm_architecture_model_revision,
            "architecture model",
        ),
    ):
        if is_uri(value) and not revision:
            print(
                f"ERROR: immutable revision is required for {label} URI {value!r}",
                file=sys.stderr,
            )
            return 2
        if not is_uri(value) and not Path(value).expanduser().is_dir():
            print(
                f"ERROR: local {label} path is inaccessible: {value}", file=sys.stderr
            )
            return 2
    output = Path(args.output_path).expanduser()
    cache = Path(args.cache_dir).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    if output.exists():
        try:
            existing = validate(output)
        except (OSError, ValueError, json.JSONDecodeError):
            if not args.force:
                print(
                    "ERROR: output exists but is incomplete; use --force to replace this exact target",
                    file=sys.stderr,
                )
                return 2
            shutil.rmtree(output)
        else:
            metadata = output / "tao_conversion_provenance.json"
            if metadata.is_file() and not args.force:
                print(json.dumps({"status": "reused_verified", **existing}, indent=2))
                return 0
            if not args.force:
                print(
                    "ERROR: prepared model lacks conversion provenance; use --force to reproduce it",
                    file=sys.stderr,
                )
                return 2
            shutil.rmtree(output)
    run_command = (
        inside_container_command(args, output, cache)
        if args.inside_container
        else command(args, output, cache)
    )
    run = subprocess.run(run_command, check=False)
    if run.returncode:
        return run.returncode
    prepared = validate(output)
    provenance = {
        "schema_version": 1,
        "base_model": identity(args.base_model_identity or args.base_model_path_or_uri),
        "base_model_revision": args.base_model_revision or None,
        "architecture_model": identity(
            args.vlm_architecture_model_identity
            or args.vlm_architecture_model_path_or_uri
        ),
        "architecture_model_revision": args.vlm_architecture_model_revision or None,
        "preparation_runtime": {
            "image": args.runtime_image,
            "image_digest": args.runtime_image_digest,
        },
        "output": identity(str(output)),
        "prepared": prepared,
    }
    (output / "tao_conversion_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
