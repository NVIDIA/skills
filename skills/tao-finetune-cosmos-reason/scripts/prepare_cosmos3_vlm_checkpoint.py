#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Prepare Cosmos3-Nano weights for Cosmos-RL Qwen3-VL loaders.
The upstream Cosmos3-Nano checkpoint is a Cosmos3 Omni checkpoint. Cosmos-RL
images that load Qwen3-VL models need the documented VLM safetensors conversion
from NVIDIA/cosmos-framework.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_COSMOS_FRAMEWORK_REPO = "https://github.com/NVIDIA/cosmos-framework.git"
DEFAULT_CONVERSION_IMAGE = "nvcr.io/nvidia/pytorch:25.09-py3"
DEFAULT_VLM_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"


def expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(cmd[:2]) if cmd[:2] == ["docker", "run"] else " ".join(cmd)
    print(f"running: {printable}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def converted_checkpoint_status(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing output directory"
    if not path.is_dir():
        return False, "output path exists but is not a directory"

    config_path = path / "config.json"
    index_path = path / "model.safetensors.index.json"
    if not config_path.is_file():
        return False, "missing config.json"
    if not index_path.is_file():
        return False, "missing model.safetensors.index.json"

    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        return False, f"invalid config.json: {exc}"

    if config.get("model_type") != "qwen3_vl":
        return False, f"config model_type is {config.get('model_type')!r}, expected 'qwen3_vl'"
    if "Qwen3VLForConditionalGeneration" not in config.get("architectures", []):
        return False, "config architectures does not include Qwen3VLForConditionalGeneration"

    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError as exc:
        return False, f"invalid model.safetensors.index.json: {exc}"

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False, "model.safetensors.index.json has no weight_map"

    missing = sorted({rel for rel in weight_map.values() if not (path / rel).is_file()})
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        return False, f"missing referenced shard(s): {preview}{suffix}"

    tokenizer_files = ["tokenizer_config.json", "tokenizer.json"]
    missing_tokenizer = [name for name in tokenizer_files if not (path / name).is_file()]
    if missing_tokenizer:
        return False, f"missing tokenizer file(s): {', '.join(missing_tokenizer)}"

    return True, "complete qwen3_vl safetensors directory"


def ensure_cosmos_framework(path: Path, repo: str, no_clone: bool) -> None:
    converter = path / "cosmos_framework" / "scripts" / "convert_model_to_vlm_safetensors.py"
    if converter.is_file():
        return
    if path.exists():
        raise SystemExit(f"{path} exists but does not contain {converter.relative_to(path)}")
    if no_clone:
        raise SystemExit(f"{path} is missing and --no-clone was set")
    path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", repo, str(path)])


def docker_checkpoint_mount(checkpoint_path: str) -> tuple[list[str], str]:
    host_path = expand_path(checkpoint_path)
    if not host_path.exists():
        return [], checkpoint_path
    container_path = f"/checkpoint/{host_path.name}"
    return ["-v", f"{host_path}:{container_path}:ro"], container_path


def docker_env_args(args: argparse.Namespace) -> list[str]:
    env_args: list[str] = []
    secrets_env = expand_path(args.secrets_env) if args.secrets_env else None
    if secrets_env and secrets_env.is_file():
        env_args.extend(["--env-file", str(secrets_env)])

    # Pass through common token names by reference without printing values.
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name):
            env_args.extend(["-e", name])
    return env_args


def conversion_command() -> str:
    return r"""
set -euo pipefail
cd /workspace/cosmos-framework

if [ ! -x .venv/bin/python ]; then
  uv sync --all-extras --group=cu130
fi

source .venv/bin/activate
export LD_LIBRARY_PATH=

python - <<'PY' || uv sync --all-extras --group=cu130
import inspect
from torch.distributed.checkpoint.hf_storage import _HFStorageInfo
params = list(inspect.signature(_HFStorageInfo).parameters)
if params[:3] != ["relative_path", "shape", "dtype"]:
    raise SystemExit(1)
PY

source .venv/bin/activate
export LD_LIBRARY_PATH=
python -m cosmos_framework.scripts.convert_model_to_vlm_safetensors \
  --checkpoint-path "$CONVERSION_CHECKPOINT_PATH" \
  -o "$CONVERSION_OUTPUT_PATH" \
  --vlm-model-name "$CONVERSION_VLM_MODEL_NAME"

python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["CONVERSION_OUTPUT_PATH"])
config = json.loads((path / "config.json").read_text())
if config.get("model_type") != "qwen3_vl":
    raise SystemExit(f"converted config model_type={config.get('model_type')!r}, expected qwen3_vl")
print(f"converted_checkpoint={path}")
print("converted_model_type=qwen3_vl")
PY

chown -R "${HOST_UID}:${HOST_GID}" "$CONVERSION_OUTPUT_PATH" /cache || true
"""


def run_conversion(args: argparse.Namespace) -> None:
    cosmos_framework_path = expand_path(args.cosmos_framework_path)
    output_path = expand_path(args.output_path)
    cache_dir = expand_path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_cosmos_framework(cosmos_framework_path, args.cosmos_framework_repo, args.no_clone)

    ckpt_mount, container_checkpoint_path = docker_checkpoint_mount(args.checkpoint_path)
    container_output_path = f"/output/{output_path.name}"

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--ipc=host",
        *docker_env_args(args),
        "-e",
        "HOME=/cache/home",
        "-e",
        "XDG_CACHE_HOME=/cache/xdg",
        "-e",
        "UV_CACHE_DIR=/cache/uv",
        "-e",
        "HF_HOME=/cache/huggingface",
        "-e",
        "UV_LINK_MODE=copy",
        "-e",
        f"HOST_UID={os.getuid()}",
        "-e",
        f"HOST_GID={os.getgid()}",
        "-e",
        f"CONVERSION_CHECKPOINT_PATH={container_checkpoint_path}",
        "-e",
        f"CONVERSION_OUTPUT_PATH={container_output_path}",
        "-e",
        f"CONVERSION_VLM_MODEL_NAME={args.vlm_model_name}",
        "-v",
        f"{cosmos_framework_path}:/workspace/cosmos-framework",
        "-v",
        f"{output_path.parent}:/output",
        "-v",
        f"{cache_dir}:/cache",
        *ckpt_mount,
        "-w",
        "/workspace/cosmos-framework",
        "--entrypoint",
        "bash",
        args.conversion_image,
        "-lc",
        conversion_command(),
    ]
    run(docker_cmd)


def validate_with_image(output_path: Path, image: str) -> None:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "bash",
        "-v",
        f"{output_path}:/converted:ro",
        image,
        "-lc",
        (
            "python - <<'PY'\n"
            "from transformers import AutoConfig\n"
            "cfg = AutoConfig.from_pretrained('/converted')\n"
            "print(type(cfg).__name__, cfg.model_type)\n"
            "if cfg.model_type != 'qwen3_vl':\n"
            "    raise SystemExit(1)\n"
            "PY"
        ),
    ]
    run(cmd)


def write_metadata(output_path: Path, args: argparse.Namespace, status: str) -> None:
    metadata = {
        "status": status,
        "checkpoint_path": args.checkpoint_path,
        "output_path": str(output_path),
        "vlm_model_name": args.vlm_model_name,
        "cosmos_framework_path": str(expand_path(args.cosmos_framework_path)),
        "conversion_image": args.conversion_image,
    }
    (output_path / "tao_conversion_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Source Cosmos3 checkpoint path or model name, for example /models/Cosmos3-Nano.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Host output directory for the converted Qwen3-VL safetensors checkpoint.",
    )
    parser.add_argument("--vlm-model-name", default=DEFAULT_VLM_MODEL_NAME)
    parser.add_argument(
        "--cosmos-framework-path",
        default="~/cosmos-framework",
        help="Local NVIDIA/cosmos-framework checkout. Cloned if missing.",
    )
    parser.add_argument("--cosmos-framework-repo", default=DEFAULT_COSMOS_FRAMEWORK_REPO)
    parser.add_argument("--conversion-image", default=DEFAULT_CONVERSION_IMAGE)
    parser.add_argument(
        "--cache-dir",
        default="~/.cache/tao-cosmos3-conversion",
        help="Host cache for HuggingFace and uv artifacts used by the conversion container.",
    )
    parser.add_argument(
        "--secrets-env",
        default="~/.tao/secrets.env",
        help="Optional env file passed to docker; values are never printed.",
    )
    parser.add_argument(
        "--validate-with-image",
        default="",
        help="Optional Cosmos-RL image used to validate AutoConfig can load the converted directory.",
    )
    parser.add_argument("--force", action="store_true", help="Remove and recreate an incomplete/existing output path.")
    parser.add_argument("--no-clone", action="store_true", help="Fail instead of cloning cosmos-framework when missing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = expand_path(args.output_path)

    complete, reason = converted_checkpoint_status(output_path)
    if complete and not args.force:
        print(f"converted_checkpoint={output_path}", flush=True)
        print("status=skipped_existing", flush=True)
        print(f"reason={reason}", flush=True)
        if args.validate_with_image:
            validate_with_image(output_path, args.validate_with_image)
        return 0

    if output_path.exists():
        if not args.force:
            raise SystemExit(f"{output_path} exists but is not complete: {reason}. Use --force to recreate it.")
        shutil.rmtree(output_path)

    run_conversion(args)

    complete, reason = converted_checkpoint_status(output_path)
    if not complete:
        raise SystemExit(f"conversion did not produce a complete checkpoint: {reason}")

    write_metadata(output_path, args, "converted")
    print(f"converted_checkpoint={output_path}", flush=True)
    print("status=converted", flush=True)

    if args.validate_with_image:
        validate_with_image(output_path, args.validate_with_image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
