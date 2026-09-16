#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download a released KERMT model bundle from Hugging Face.

Runs INSIDE the kermt container (huggingface_hub is part of the image env).
Writes the released-model directory bundle — the checkpoint plus its vocab
files — into the directory mounted at `--out` (the skills mount the user's
chosen save location there via `kermt_container.sh --model-dir <host>`), then
emits a single JSON object to stdout that the calling skill parses.

The downloaded directory is exactly the repo's "released model bundle" layout
(see skills/README.md "Released models"): `<ckpt>.pt` + the three
`pretrain_*_vocab.*` files in one flat directory. The downstream skill then
feeds it through the existing `--ckpt <out>/<ckpt_name>` flow; for
continue-pretrain the bundled vocab files are auto-detected in the ckpt's
parent directory. No runner changes are needed.

Defaults (repo id, pinned revision, ckpt + vocab filenames) come from
`config/released_model.json` so the pin lives in one place; every value
is overridable on the CLI.

Idempotent: if the bundle is already complete in `--out` (ckpt + all vocab
files present), nothing is downloaded and `reused: true` is reported — so a
re-invocation never re-fetches the 282 MB checkpoint.

Authentication: the repo is public (no token needed). If `HF_TOKEN` is set in
the environment (forwarded into the container by `kermt_container.sh`),
huggingface_hub picks it up automatically — useful against shared-IP rate
limits or if the repo is ever gated.

Output (stdout)
---------------
{
  "ok": true | false,
  "repo_id": str,
  "revision": str,
  "out": str,                 // container path of the bundle dir (e.g. /model)
  "ckpt": str | null,         // container path of the checkpoint file
  "vocab_dir": str | null,    // == out (where the vocab files live)
  "ckpt_name": str,
  "vocab_files": [str, ...],
  "files_present": [str, ...],
  "ckpt_bytes": int | null,
  "reused": bool,             // true if the bundle already existed (no download)
  "license": str | null,
  "license_url": str | null,
  "errors": [str, ...]
}

Exit code: 0 on `ok: true`, 1 on `ok: false`.

CLI
---
    fetch_released_model.py [--out /model]
                            [--repo-id <id>] [--revision <sha|tag|branch>]
                            [--ckpt-name <name>] [--config <path>]
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

# Default config lives at config/released_model.json (one dir up from
# scripts/). Resolved relative to this file so the script is
# location-independent.
DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "config" / "released_model.json"
)


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"released-model config not found at {config_path}")
    return json.loads(config_path.read_text())


def fetch(
    *,
    out: Path,
    repo_id: str,
    revision: str,
    ckpt_name: str,
    vocab_files: list[str],
    license_name: str | None = None,
    license_url: str | None = None,
) -> dict[str, Any]:
    """Resolve-or-download the released bundle into `out`. Returns the manifest
    dict (never raises for the expected failure modes — they land in
    `errors[]` with `ok: false`)."""
    result: dict[str, Any] = {
        "ok": False,
        "repo_id": repo_id,
        "revision": revision,
        "out": str(out),
        "ckpt": None,
        "vocab_dir": None,
        "ckpt_name": ckpt_name,
        "vocab_files": list(vocab_files),
        "files_present": [],
        "ckpt_bytes": None,
        "reused": False,
        "license": license_name,
        "license_url": license_url,
        "errors": [],
    }

    required = [ckpt_name, *vocab_files]

    def _present() -> list[str]:
        return [name for name in required if (out / name).is_file()]

    # 1. Idempotent reuse — bundle already complete in `out`.
    if out.is_dir() and set(_present()) == set(required):
        result["reused"] = True
    else:
        # 2. Download. Import here so a stale image (missing huggingface_hub)
        #    surfaces a clean, actionable JSON error rather than a traceback.
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            result["errors"].append(
                "huggingface_hub is not available in the container image. The "
                "released-model download needs it; rebuild the image with "
                "`kermt-setup` (it now ships huggingface_hub) and retry."
            )
            return result

        out.mkdir(parents=True, exist_ok=True)
        try:
            # local_dir gives a flat copy (the bundle layout) rather than the
            # opaque blob/snapshot cache. HF_TOKEN, if set, is read by the lib.
            snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(out))
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(
                f"download failed for {repo_id}@{revision}: {type(exc).__name__}: {exc}"
            )
            return result

    # 3. Verify the bundle is complete regardless of download/reuse path.
    present = _present()
    result["files_present"] = present
    missing = [name for name in required if name not in present]
    if missing:
        result["errors"].append(
            f"bundle at {out} is missing expected file(s): {missing}. "
            f"Present: {present}."
        )
        return result

    ckpt_path = out / ckpt_name
    result["ckpt"] = str(ckpt_path)
    result["vocab_dir"] = str(out)
    try:
        result["ckpt_bytes"] = ckpt_path.stat().st_size
    except OSError:
        result["ckpt_bytes"] = None

    result["ok"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download a released KERMT model bundle from Hugging Face (runs in-container)."
    )
    parser.add_argument(
        "--out",
        default="/model",
        help="Directory to write the bundle into (default: /model, the --model-dir mount).",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to released_model.json (default: config/released_model.json).",
    )
    parser.add_argument(
        "--repo-id", default=None, help="Override the HF repo id from the config."
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Override the pinned revision (sha/tag/branch).",
    )
    parser.add_argument(
        "--ckpt-name",
        default=None,
        help="Override the checkpoint filename from the config.",
    )
    args = parser.parse_args(argv)

    try:
        cfg = _load_config(Path(args.config))
        repo_id = args.repo_id or cfg["repo_id"]
        revision = args.revision or cfg["revision"]
        ckpt_name = args.ckpt_name or cfg["ckpt_name"]
        vocab_files = list(cfg.get("vocab_files", []))
        result = fetch(
            out=Path(args.out),
            repo_id=repo_id,
            revision=revision,
            ckpt_name=ckpt_name,
            vocab_files=vocab_files,
            license_name=cfg.get("license"),
            license_url=cfg.get("license_url"),
        )
    except Exception as exc:  # noqa: BLE001
        print(traceback.format_exc(), file=sys.stderr)
        print(
            json.dumps(
                {
                    "ok": False,
                    "out": args.out,
                    "errors": [
                        f"unhandled exception in fetch_released_model: {type(exc).__name__}: {exc}"
                    ],
                },
                indent=2,
            )
        )
        return 1

    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
