#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrap the upstream NV-Reason-PathWSI inference.py CLI without changing it."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

MODEL_ID = "nvidia/NV-Reason-PathWSI"
DEFAULT_PROMPT = "Describe the histopathological features visible in this image."
DEFAULT_MAX_TOKENS = 2048  # Upstream single-image CLI default.
DEFAULT_TIMEOUT = 900  # Bound unattended model loading and generation.
PROBE_TIMEOUT = 5  # Seconds for read-only git and nvidia-smi probes.
SYNTHETIC_SIZE = 64  # Tiny RGB geometry fixture, with no tissue or patient data.
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
# Packages the upstream single-image path imports; versions are recorded, never pinned.
RUNTIME_PACKAGES = ("torch", "torchvision", "transformers", "accelerate", "Pillow")
# Forward only runtime settings needed by the documented upstream CLI. In
# particular, unrelated API keys, CI credentials, and Python injection settings
# must not enter the model process.
CHILD_RUNTIME_ENV = (
    "PATH",
    "HOME",
    "SYSTEMROOT",
    "WINDIR",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)
CHILD_DOWNLOAD_ENV = (
    "HF_TOKEN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)
LIMITATIONS = [
    "Research and engineering use only; not for diagnosis, treatment, or clinical reporting.",
    "Accepts PNG/JPEG patches or tiles, not raw whole-slide files; slide tiling and aggregation are out of scope.",
    "Mock output tests wrapper plumbing only and contains no model inference.",
    "Upstream response text is not checked for medical correctness or hallucinations.",
    "The upstream CLI does not expose generated token counts or termination reasons; truncation is unknown.",
    "Upstream generation can continue past the model's first answer when the model package's EOS ids omit "
    "the chat end-of-turn token; response text is returned unchanged.",
    "Local model identity is caller supplied; this wrapper records provenance without authenticating weights.",
]
RESULT_FILES = (
    "result.json",
    "response.txt",
    "synthetic_input.png",
    "upstream.stdout.txt",
    "upstream.stderr.log",
)


class SkillError(Exception):
    """An actionable input, setup, or upstream failure."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def read_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def missing_packages(versions: dict) -> list[str]:
    return [name for name, version in versions.items() if version is None]


def pillow_modules():
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SkillError(
            "Pillow is required; install the tested environment from SKILL.md."
        ) from exc
    return Image, ImageDraw


def read_request(path: Path, prompt_override: str | None) -> dict:
    if not path.is_file():
        raise SkillError(f"Input file not found: {path}")
    request = {"image": str(path), "prompt": DEFAULT_PROMPT, "mock": False, "case_id": path.stem}
    if path.suffix.lower() == ".json":
        request = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise SkillError("Request JSON must be an object.")
        request = {"prompt": DEFAULT_PROMPT, "mock": False, "case_id": path.stem, **request}
        image = request.get("image")
        if not isinstance(image, str) or not image.strip():
            raise SkillError("Request JSON requires a nonempty image path.")
        if image != "generated://synthetic_rgb":
            local = Path(image).expanduser()
            request["image"] = str((path.parent / local).resolve())
    if not isinstance(request.get("mock"), bool):
        raise SkillError("Request mock must be a JSON boolean.")
    if not isinstance(request.get("case_id"), str) or not request["case_id"].strip():
        raise SkillError("Request case_id must be a nonempty string.")
    if prompt_override is not None:
        request["prompt"] = prompt_override
    if not isinstance(request["prompt"], str) or not request["prompt"].strip():
        raise SkillError("Prompt must be a nonempty string.")
    return request


def prepare_image(request: dict, out_dir: Path, mock: bool) -> tuple[Path, dict]:
    Image, ImageDraw = pillow_modules()
    source = request["image"]
    if source == "generated://synthetic_rgb":
        if not mock:
            raise SkillError(
                "The generated fixture is mock-only; provide a real PNG/JPEG for live inference."
            )
        path = out_dir / "synthetic_input.png"
        image = Image.new("RGB", (SYNTHETIC_SIZE, SYNTHETIC_SIZE), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 56, 56), fill="pink")
        draw.ellipse((24, 24, 40, 40), fill="purple")
        image.save(path)
    else:
        path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise SkillError(f"Image not found: {path}")
    if is_lfs_pointer(path):
        raise SkillError(
            "Image is a Git LFS pointer; fetch the upstream examples with git lfs pull."
        )
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise SkillError(
            "Only PNG/JPEG patches or tiles are supported; raw WSI, TIFF, and DICOM are not."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                image_format, width, height = image.format, image.width, image.height
    except (
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise SkillError(f"Invalid or unsafe image: {path} ({exc})") from exc
    if image_format not in {"PNG", "JPEG"}:
        raise SkillError("Decoded image must be PNG or JPEG.")
    return path, {
        "path": str(path),
        "source": source,
        "format": image_format.lower(),
        "width": width,
        "height": height,
        "sha256": sha256_file(path),
    }


def upstream_entrypoint(root: str | None) -> Path:
    if not root:
        raise SkillError(
            "Live inference requires --upstream-dir (or NV_REASON_PATHWSI_ROOT) with upstream inference.py."
        )
    script = Path(root).expanduser().resolve() / "inference.py"
    if not script.is_file():
        raise SkillError(f"Upstream entrypoint not found: {script}")
    return script


def git_output(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_commit(root: Path) -> str | None:
    # An upstream copy nested in another repository must not report that repository's HEAD.
    top = git_output(root, "rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() != root.resolve():
        return None
    return git_output(root, "rev-parse", "HEAD")


def child_environment(args, out_dir: Path) -> dict[str, str]:
    names = CHILD_RUNTIME_ENV + (CHILD_DOWNLOAD_ENV if args.allow_downloads else ())
    env = {name: value for name in names if (value := os.environ.get(name)) is not None}
    # Never change the parent environment or write model/compiler caches in home.
    cache = out_dir / "cache"
    env["HF_HOME"] = str(
        Path(args.hf_home).expanduser().resolve() if args.hf_home else cache / "huggingface"
    )
    env["HF_HUB_CACHE"] = str(Path(env["HF_HOME"]) / "hub")
    env["HF_XET_CACHE"] = str(cache / "xet")
    env["XDG_CACHE_HOME"] = str(cache / "xdg")
    env["TORCH_HOME"] = str(cache / "torch")
    env["TRITON_CACHE_DIR"] = str(cache / "triton")
    env["CUDA_CACHE_PATH"] = str(cache / "cuda")
    env["PYTORCH_KERNEL_CACHE_PATH"] = str(cache / "torch_kernels")
    for name in (
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "PYTORCH_KERNEL_CACHE_PATH",
    ):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if not args.allow_downloads:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def redact_logs(paths: list[Path], env: dict[str, str]) -> None:
    secrets = [
        env.get(name) for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for secret in filter(None, secrets):
            text = text.replace(secret, "<redacted>")
        path.write_text(text, encoding="utf-8")


def run_upstream(args, request: dict, image_path: Path, out_dir: Path) -> tuple[str, dict]:
    script = upstream_entrypoint(args.upstream_dir)
    model = args.model
    if Path(model).expanduser().is_dir():
        model = str(Path(model).expanduser().resolve())
    command = [
        sys.executable,
        str(script),
        "--model",
        model,
        "--image",
        str(image_path),
        "--prompt",
        request["prompt"],
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.think:
        command.append("--think")
    if args.device_map:
        command.extend(["--device-map", args.device_map])
    stdout_path, stderr_path = out_dir / "upstream.stdout.txt", out_dir / "upstream.stderr.log"
    env = child_environment(args, out_dir)
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            result = subprocess.run(
                command,
                cwd=script.parent,
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=args.timeout,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise SkillError(
            f"Upstream timed out after {args.timeout}s; inspect {stderr_path}. No mock fallback was used."
        ) from exc
    finally:
        redact_logs([stdout_path, stderr_path], env)
    if result.returncode != 0:
        raise SkillError(
            f"Upstream exited {result.returncode}; inspect {stderr_path}. Check weights, CUDA, and dependencies. No mock fallback was used."
        )
    response = stdout_path.read_text(encoding="utf-8").strip()
    if not response:
        raise SkillError(
            f"Upstream exited successfully but returned empty text; inspect {stderr_path}."
        )
    return response, {
        "command": command,
        "upstream_entrypoint_sha256": sha256_file(script),
        "upstream_commit": git_commit(script.parent),
        "process_returncode": result.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "offline": env.get("HF_HUB_OFFLINE") == "1" or env.get("TRANSFORMERS_OFFLINE") == "1",
    }


def gpu_is_visible() -> bool:
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return gpu.returncode == 0 and bool(gpu.stdout.strip())


def eos_report(root: Path) -> dict:
    """Compare the ids that stop upstream generate() with the tokenizer end-of-turn token."""
    # Transformers reads generation_config.json first and derives it from config.json otherwise.
    generation = read_json_object(root / "generation_config.json") or read_json_object(
        root / "config.json"
    )
    stop_ids = generation.get(
        "eos_token_id", (generation.get("text_config") or {}).get("eos_token_id")
    )
    stop_ids = (
        [stop_ids]
        if isinstance(stop_ids, int)
        else [i for i in stop_ids or [] if isinstance(i, int)]
    )
    eos_token = read_json_object(root / "tokenizer_config.json").get("eos_token")
    if isinstance(eos_token, dict):
        eos_token = eos_token.get("content")
    added = read_json_object(root / "tokenizer.json").get("added_tokens") or []
    eos_id = next(
        (t.get("id") for t in added if isinstance(t, dict) and t.get("content") == eos_token), None
    )
    return {
        "generation_eos_token_ids": stop_ids,
        "tokenizer_eos_token": eos_token,
        "tokenizer_eos_token_id": eos_id,
        "stops_at_tokenizer_eos": (
            eos_id in stop_ids if stop_ids and isinstance(eos_id, int) else None
        ),
    }


def local_model_report(model: str) -> dict | None:
    """File-level checks for a local snapshot; reads small JSON files and never loads weights."""
    root = Path(model).expanduser()
    if not root.is_dir():
        return None

    def has(*names: str) -> bool:
        return any((root / name).is_file() for name in names)

    tokenizer_config = read_json_object(root / "tokenizer_config.json")
    parts = {
        "config": has("config.json"),
        "tokenizer": bool(tokenizer_config) and has("tokenizer.json", "vocab.json"),
        "processor": has("processor_config.json", "preprocessor_config.json"),
        "chat_template": has("chat_template.jinja", "chat_template.json")
        or "chat_template" in tokenizer_config,
        "weights": any(not is_lfs_pointer(path) for path in root.glob("*.safetensors")),
    }
    return {"missing": [name for name, ok in parts.items() if not ok], **eos_report(root)}


def readiness(
    missing: list[str], entrypoint: str | None, gpu_visible: bool, snapshot: dict | None
) -> tuple[bool | None, str]:
    if missing:
        return False, (
            "Install the tested environment from SKILL.md and run the wrapper with that Python; "
            "missing: " + ", ".join(missing) + "."
        )
    if not entrypoint:
        return (
            False,
            "Provide --upstream-dir (or NV_REASON_PATHWSI_ROOT) containing upstream inference.py.",
        )
    if not gpu_visible:
        return (
            False,
            "No NVIDIA GPU is visible; live inference needs CUDA. Use --mock only for wiring checks.",
        )
    if snapshot is None:
        return None, (
            "Model ID selected; offline runs need --hf-home with a cached snapshot, "
            "--allow-downloads for an authorized download, or a local snapshot directory."
        )
    if snapshot["missing"]:
        return (
            False,
            "Local model snapshot is incomplete; missing: " + ", ".join(snapshot["missing"]) + ".",
        )
    return True, "Run the wrapper with --live on a PNG/JPEG tile and a fresh --out-dir."


def setup_report(args) -> dict:
    versions = package_versions()
    missing = missing_packages(versions)
    try:
        entrypoint = str(upstream_entrypoint(args.upstream_dir))
    except SkillError:
        entrypoint = None
    gpu_visible = gpu_is_visible()
    snapshot = local_model_report(args.model)
    warnings = []
    if snapshot and snapshot["stops_at_tokenizer_eos"] is False:
        warnings.append(
            f"Generation stops only at token ids {snapshot['generation_eos_token_ids']}, not the tokenizer "
            f"end-of-turn token {snapshot['tokenizer_eos_token']!r} ({snapshot['tokenizer_eos_token_id']}); "
            "upstream output can continue past the first answer, up to --max-new-tokens."
        )
    ready, next_step = readiness(missing, entrypoint, gpu_visible, snapshot)
    return {
        "skill": "nv_reason_pathwsi",
        "mode": "setup",
        "packages": versions,
        "missing_packages": missing,
        "upstream_entrypoint": entrypoint,
        "gpu_visible": gpu_visible,
        "model": args.model,
        "local_model_directory": snapshot is not None,
        "local_model": snapshot,
        "offline_by_default": True,
        "network_access_checked": False,
        "ready_for_live": ready,
        "warnings": warnings,
        "next_step": next_step,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_or_request", nargs="?", type=Path)
    parser.add_argument(
        "--out-dir", type=Path, help="Fresh directory for JSON, response, logs, and per-run caches."
    )
    parser.add_argument("--upstream-dir", default=os.environ.get("NV_REASON_PATHWSI_ROOT"))
    parser.add_argument(
        "--model",
        default=os.environ.get("NV_REASON_PATHWSI_MODEL", MODEL_ID),
        help="Model ID or a complete local model directory.",
    )
    parser.add_argument("--prompt")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--mock", action="store_true", help="Explicit wiring-only test; never loads model weights."
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Require real upstream execution; reject a mock fixture request.",
    )
    parser.add_argument(
        "--think", action="store_true", help="Pass upstream thinking-mode flag (default off)."
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--device-map", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--hf-home", help="Explicit existing Hugging Face cache; otherwise use out-dir/cache."
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow upstream HF asset downloads; inherited offline settings still apply.",
    )
    parser.add_argument(
        "--check-setup",
        action="store_true",
        help="Read-only local setup report; no downloads or model loading.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.max_new_tokens < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
            raise SkillError("--max-new-tokens and --timeout must be positive finite values.")
        if args.check_setup:
            print(json.dumps(setup_report(args), indent=2))
            return 0
        if args.image_or_request is None or args.out_dir is None:
            raise SkillError("Provide an image/request and --out-dir, or use --check-setup.")
        request = read_request(args.image_or_request.expanduser().resolve(), args.prompt)
        if args.live and request["mock"]:
            raise SkillError(
                "--live refuses a mock request; provide a real PNG/JPEG or a live request JSON."
            )
        mock = args.mock or request["mock"]
        out_dir = args.out_dir.expanduser().resolve()
        if any((out_dir / name).exists() for name in RESULT_FILES):
            raise SkillError(
                "Output artifacts already exist; choose a fresh --out-dir to preserve prior evidence."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path, image_info = prepare_image(request, out_dir, mock)
        started = time.perf_counter()
        extra = {
            "command": [],
            "upstream_entrypoint_sha256": None,
            "upstream_commit": None,
            "process_returncode": None,
            "stdout_path": None,
            "stderr_path": None,
            "offline": True,
        }
        response = "MOCK: wrapper wiring check completed; no model was loaded and no tissue was interpreted."
        if not mock:
            response, extra = run_upstream(args, request, image_path, out_dir)
        response_path = out_dir / "response.txt"
        response_path.write_text(response + "\n", encoding="utf-8")
        payload = {
            "skill": "nv_reason_pathwsi",
            "input": {
                "case_id": request["case_id"],
                "prompt": request["prompt"],
                "image": image_info,
            },
            "output": {
                "response_text": response,
                "text_chars": len(response),
                "response_path": str(response_path),
                "response_sha256": sha256_file(response_path),
            },
            "runtime": {
                "mode": "mock" if mock else "upstream_cli",
                "mock": mock,
                "model": args.model,
                "thinking": args.think,
                "max_new_tokens": args.max_new_tokens,
                "inference_seconds": round(time.perf_counter() - started, 6),
                "truncation_status": "unknown" if not mock else "not_applicable",
                **extra,
            },
            "environment": {"python": sys.version.split()[0], "packages": package_versions()},
            "limitations": LIMITATIONS,
        }
        serialized = json.dumps(payload, indent=2, sort_keys=True)
        (out_dir / "result.json").write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
        return 0
    except (SkillError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
