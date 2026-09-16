# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the agent scripts.

Kept intentionally small — only logic that appears (or would otherwise be
duplicated) in two or more `scripts/*.py` modules. Each script
maintains its own primary CLI + main flow.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import pickle
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


# Conventional pretrain vocab filename stems. Used by prepare_data.py +
# upgrade_to_hybrid.py + the README "Released models" bundling docs +
# the test helpers. Centralized here so a future rename only touches one
# spot.
PRETRAIN_VOCAB_STEMS = {
    "atom":   "pretrain_atom_vocab",
    "bond":   "pretrain_bond_vocab",
    "smiles": "pretrain_smiles_vocab",
}


def resolve_kermt_repo() -> Path:
    """Find the runtime checkout independently of the installed skill location.

    An explicit KERMT_REPO takes precedence. In a repository checkout, walking
    up from this helper or the working directory also supports local use.
    """
    explicit = os.environ.get("KERMT_REPO")
    if explicit:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        candidates = []
        for start in (Path(__file__).resolve().parent, Path.cwd()):
            candidates.extend((start, *start.parents))
    for candidate in candidates:
        if (candidate / "main.py").is_file() and (candidate / "kermt").is_dir():
            return candidate
    raise FileNotFoundError(
        "KERMT checkout not found. Set KERMT_REPO to the checkout containing "
        "main.py and kermt/; the installed skill directory is separate."
    )


def load_json(path: Path, *, name: str) -> dict[str, Any]:
    """Load a JSON file with consistent error messages.

    `name` is a human-readable label for the document (e.g. "prepare_data.json")
    so the error tells the user which schema we expected at that path.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found at {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} at {path} is not valid JSON: {exc}") from exc


def count_vocab_entries(vocab_path: Path) -> int:
    """Return the number of entries in a KERMT vocab file.

    Handles three layouts:
      - JSON with `{stoi: {token: idx}, ...}` (MolVocab.save_vocab default)
      - JSON as a raw `{token: idx}` dict (legacy / hand-edited)
      - Legacy MolVocab / SMILESVocab pickles, read as inert vocabulary state.

    The pickle reader accepts only the known vocabulary containers and their
    Counter/regex metadata. It cannot import arbitrary classes or run reducers
    supplied by the artifact, and it never falls back to an unrestricted loader.
    """
    if vocab_path.suffix == ".json":
        data = json.loads(vocab_path.read_text())
        if isinstance(data, dict) and "stoi" in data:
            return len(data["stoi"])
        if isinstance(data, dict):
            return len(data)
        raise ValueError(f"unsupported JSON vocab shape at {vocab_path}: {type(data).__name__}")

    with vocab_path.open("rb") as f:
        data = _VocabUnpickler(f).load()
    if isinstance(data, _VocabState) and isinstance(data.stoi, dict):
        return len(data.stoi)
    if isinstance(data, (dict, list, tuple)):
        return len(data)
    raise ValueError(f"could not count entries in {vocab_path}")


class _VocabState:
    """Data-only stand-in: counting tokens does not require tokenizer methods."""


class _VocabUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module in {"kermt.data.torchvocab", "grover.data.torchvocab"} and name in {
            "TorchVocab", "MolVocab", "SMILESVocab",
        }:
            return _VocabState
        if (module, name) == ("collections", "Counter"):
            return Counter
        if (module, name) == ("re", "_compile"):
            return re.compile
        raise pickle.UnpicklingError(f"unsupported vocabulary object: {module}.{name}")


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    """Read KERMT tensors and known metadata with PyTorch's restricted loader.

    Saved arguments use argparse.Namespace; finetuned checkpoints also contain
    numeric NumPy scaler arrays. Explicit globals cover those formats, including
    NumPy 1/2 module names, without accepting artifact-selected imports.
    """
    import numpy as np
    import torch

    multiarray = np._core.multiarray if hasattr(np, "_core") else np.core.multiarray
    allowed = [argparse.Namespace, np.ndarray, np.dtype]
    for module in ("numpy.core.multiarray", "numpy._core.multiarray"):
        allowed.extend([
            (multiarray._reconstruct, f"{module}._reconstruct"),
            (multiarray.scalar, f"{module}.scalar"),
        ])
    allowed.extend(type(np.dtype(name)) for name in (
        "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64",
    ))
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def runner_environment(repo: Path, *, wandb: bool = False) -> dict[str, str]:
    """Forward named runtime settings, keeping unrelated credentials out of jobs.

    W&B credentials/settings are included only for an explicitly enabled W&B
    run. Hugging Face authentication belongs to the separate download helper.
    """
    names = (
        "PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
        "LD_LIBRARY_PATH", "LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "PYTHONWARNINGS",
        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_LAUNCH_BLOCKING", "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
        "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "PYTORCH_NO_CUDA_MEMORY_CACHING",
        "TORCH_CPP_LOG_LEVEL", "TORCH_DISTRIBUTED_DEBUG", "NCCL_DEBUG", "NCCL_SOCKET_IFNAME",
        "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "GLOO_SOCKET_IFNAME",
        "MASTER_ADDR", "MASTER_PORT",
        "KERMT_REPO", "KERMT_REPO_COMMIT", "KERMT_REPO_DIRTY",
        "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
    )
    if wandb:
        names += (
            "WANDB_API_KEY", "WANDB_BASE_URL", "WANDB_MODE", "WANDB_DIR", "WANDB_ENTITY",
            "WANDB_PROJECT", "WANDB_RUN_ID", "WANDB_RESUME", "WANDB_CACHE_DIR",
            "WANDB_CONFIG_DIR", "WANDB_DATA_DIR", "WANDB_DISABLED",
        )
    env = {name: value for name in names if (value := os.environ.get(name)) is not None}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(repo), env.get("PYTHONPATH"))))
    return env


def validate_vocab_file(vocab_path: Path, *, kind: str) -> None:
    """Verify a user-provided vocab file is loadable BEFORE copying it into a
    run directory. Raises ValueError on failure with a clear, user-facing message.

    `kind` is one of {"atom", "bond", "smiles"} — used only in the error message
    so the user knows which file is wrong.
    """
    if not vocab_path.is_file():
        raise FileNotFoundError(f"{kind} vocab file not found: {vocab_path}")
    try:
        n = count_vocab_entries(vocab_path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"{kind} vocab file {vocab_path} is not loadable as a KERMT vocab "
            f"({type(exc).__name__}: {exc}). Expected a MolVocab JSON or pickle "
            f"(or a SMILESVocab pickle for the smiles vocab)."
        ) from exc
    if n <= 0:
        raise ValueError(f"{kind} vocab file {vocab_path} contains zero entries")


# ---------------------------------------------------------------------------
# Runner-shared helpers (run.json manifest fields)
# ---------------------------------------------------------------------------

def git_commit_with_env_override(repo: Path) -> tuple[str, bool]:
    """Returns (commit_sha, dirty_tree). Honors `KERMT_REPO_COMMIT` /
    `KERMT_REPO_DIRTY` env vars first — set by `scripts/kermt_container.sh`
    from the host before launching docker (necessary because `git -C /workspace`
    inside the container fails due to bind-mount ownership). Falls back to the
    in-container git probe when the env vars aren't set."""
    env_commit = os.environ.get("KERMT_REPO_COMMIT")
    if env_commit:
        env_dirty = os.environ.get("KERMT_REPO_DIRTY", "false").strip().lower() == "true"
        return env_commit, env_dirty
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
        return sha, bool(diff.stdout.strip())
    except Exception:
        return "unknown", False


def docker_image_digest(tag: str) -> str | None:
    """Return the docker image's content-addressable Id (sha256:…) for the given
    tag, or None if docker isn't available / the image isn't local."""
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", tag, "--format", "{{.Id}}"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def format_cmd_replay(argv: list[str], *, env: dict[str, str] | None = None) -> str:
    """Render a copy-pasteable env-prefix + command for the cmd_replay manifest
    field. `env` is the set of environment variables to prefix (typically
    {CUDA_VISIBLE_DEVICES, WORLD_SIZE})."""
    env = env or {}
    env_prefix = [f"{k}={shlex.quote(str(v))}" for k, v in env.items()]
    quoted = " ".join(shlex.quote(a) for a in argv)
    return " ".join(env_prefix + [quoted])


def resolve_single_gpu(override: str | None, *, workflow: str) -> int:
    """Returns a single GPU id (int). The finetune/inference/embed workflows are
    single-GPU only; `--gpus '0,1'` or multi-id CUDA_VISIBLE_DEVICES is rejected
    with a workflow-specific error. (The pretrain runner has its own multi-GPU
    `_detect_gpus` helper — see run_pretrain_local.py.)"""
    if override is None:
        env_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if env_visible:
            ids = [g for g in env_visible.split(",") if g]
            if len(ids) > 1:
                raise ValueError(
                    f"CUDA_VISIBLE_DEVICES='{env_visible}' selects multiple GPUs but "
                    f"the {workflow} workflow is single-GPU only. Restrict to one id."
                )
            return int(ids[0])
        return 0
    parts = [p.strip() for p in override.split(",") if p.strip()]
    if len(parts) != 1:
        raise ValueError(
            f"--gpus '{override}' selects {len(parts)} GPUs; the {workflow} workflow is single-GPU only."
        )
    return int(parts[0])


def assert_prepare_manifest_basics(manifest: dict[str, Any], expected_mode: str) -> None:
    """Standard pre-check for a prepare_data.json before a runner consumes it:
    verify `mode` matches and `ok` is True. Raises ValueError with a consistent
    error message on either mismatch.

    Each runner is responsible for its own required-outputs check after this
    (those vary per-mode — e.g. pretrain wants train_dir/val_dir/atom_vocab/
    bond_vocab; finetune has the split-method branch; inference/embed want
    clean_csv)."""
    if manifest.get("mode") != expected_mode:
        raise ValueError(
            f"prepare_data manifest is mode='{manifest.get('mode')}', expected '{expected_mode}'. "
            f"Run `prepare_data.py --mode {expected_mode}` to produce a valid manifest."
        )
    if not manifest.get("ok"):
        raise ValueError(
            f"prepare_data manifest reports ok=False: {manifest.get('errors')}"
        )


def merge_default_into_applied(
    applied: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    name: str,
    defaults_group: dict[str, Any],
) -> None:
    """Standard CLI-override / default-config merge for one hyperparameter.

    Mutates `applied` in place:
      - If the user passed `--<name>` on the CLI (so `getattr(args, name)` is
        not None), records `{"value": cli_val, "source": "user"}`.
      - Else if `name` is present in `defaults_group`, records
        `{"value": defaults_group[name], "source": "default-config"}`.
      - Else `applied[name]` is left absent — the runner's argv-builder skips
        the flag, and the downstream argparse default takes effect.

    `name` is the snake_case argparse dest (same form used as the dict key);
    argparse automatically converts CLI `--<name-with-hyphens>` to that dest,
    so `getattr(args, name, None)` is the correct CLI lookup."""
    cli_val = getattr(args, name, None)
    if cli_val is not None:
        applied[name] = {"value": cli_val, "source": "user"}
    elif name in defaults_group:
        applied[name] = {"value": defaults_group[name], "source": "default-config"}


def run_checkpoint_validator(ckpt: Path, *, mode: str, script_path: Path) -> dict[str, Any]:
    """Invoke `check_checkpoint.py --mode <mode> --ckpt <path>` as a subprocess
    and return the parsed JSON. Raises RuntimeError on non-JSON output (e.g. the
    validator crashed before printing). `script_path` is the absolute path to
    `scripts/check_checkpoint.py` — passed in so this helper has no
    dependency on the caller's layout."""
    r = subprocess.run(
        [sys.executable, str(script_path), "--mode", mode, "--ckpt", str(ckpt)],
        capture_output=True, text=True,
    )
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"check_checkpoint.py emitted non-JSON output (exit {r.returncode}). "
            f"stdout (first 200 chars): {r.stdout[:200]}\n"
            f"stderr (first 200 chars): {r.stderr[:200]}"
        ) from exc
