#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resolve a local or Hugging Face model/dataset snapshot without false failures."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_COMMIT_SHA_RE = re.compile(r"[0-9a-fA-F]{40}")


def _load_huggingface_hub():
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    return snapshot_download, LocalEntryNotFoundError


def _resolve_hub_revision(identifier: str, repo_type: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    if repo_type == "model":
        revision = api.model_info(repo_id=identifier).sha
    elif repo_type == "dataset":
        revision = api.dataset_info(repo_id=identifier).sha
    else:
        raise ValueError("repo_type must be 'model' or 'dataset'")
    if not _COMMIT_SHA_RE.fullmatch(revision or ""):
        raise ValueError("Hub revision lookup did not return a 40-character commit SHA")
    return revision


def resolve_model_snapshot(
    identifier: str,
    *,
    source: str,
    source_root: Optional[str] = None,
    allow_download: bool = False,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    repo_type: str = "model",
) -> Dict[str, Any]:
    """Return structured resolution evidence; an expected cache miss is not an error."""
    if source not in {"hub", "local"}:
        raise ValueError("source must be 'local' or 'hub'")
    if repo_type not in {"model", "dataset"}:
        raise ValueError("repo_type must be 'model' or 'dataset'")

    if source == "local":
        if repo_type != "model":
            raise ValueError("repo_type only applies to source='hub'")
        if allow_download or revision is not None or cache_dir is not None:
            raise ValueError("Hub download, revision, and cache options cannot be used with source='local'")
        candidate = Path(identifier).expanduser()
        if candidate.is_absolute():
            if source_root is not None:
                raise ValueError("source_root can only be used with a relative local identifier")
        else:
            if source_root is None:
                raise ValueError("relative local identifiers require an absolute source_root")
            root = Path(source_root).expanduser()
            if not root.is_absolute():
                raise ValueError("source_root must be an absolute path")
            candidate = root / candidate
        if candidate.exists():
            return {
                "download_authorized": False,
                "identifier": identifier,
                "resolved_path": str(candidate.resolve()),
                "source": "local",
                "status": "available",
            }
        return {
            "download_authorized": False,
            "identifier": identifier,
            "reason": "local_path_not_found",
            "resolved_path": None,
            "source": "local",
            "status": "missing",
        }

    if source_root is not None:
        raise ValueError("source_root can only be used with source='local'")
    if allow_download:
        if revision is None:
            revision = _resolve_hub_revision(identifier, repo_type)
        elif not _COMMIT_SHA_RE.fullmatch(revision):
            raise ValueError("authorized Hub downloads require a 40-character commit SHA revision")

    snapshot_download, local_entry_not_found = _load_huggingface_hub()
    try:
        download_args: Dict[str, Any] = {
            "repo_id": identifier,
            "cache_dir": cache_dir,
            "local_files_only": not allow_download,
        }
        if repo_type != "model":
            download_args["repo_type"] = repo_type
        resolved_path = snapshot_download(
            revision=revision,
            **download_args,
        )
    except local_entry_not_found:
        if allow_download:
            raise
        return {
            "download_authorized": False,
            "identifier": identifier,
            "reason": "not_cached",
            "repo_type": repo_type,
            "resolved_path": None,
            "revision": revision,
            "source": "hub_cache",
            "status": "missing",
        }

    return {
        "download_authorized": allow_download,
        "identifier": identifier,
        "resolved_path": str(Path(resolved_path).resolve()),
        "repo_type": repo_type,
        "revision": revision,
        "source": "hub_download_or_cache" if allow_download else "hub_cache",
        "status": "available",
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("identifier", help="Local artifact path or Hugging Face Hub repository ID")
    parser.add_argument("--source", choices=("local", "hub"), required=True, help="Identifier source type")
    parser.add_argument(
        "--source-root",
        help="Absolute original source-project root used to resolve a relative local identifier",
    )
    parser.add_argument("--allow-download", action="store_true", help="Permit one normal cache-aware Hub download")
    parser.add_argument("--revision", help="Optional full Hub commit SHA; authorized downloads resolve it when omitted")
    parser.add_argument("--cache-dir", help="Optional Hugging Face cache directory")
    parser.add_argument(
        "--repo-type",
        choices=("model", "dataset"),
        default="model",
        help="Hugging Face Hub repository type; applies only to --source hub",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = resolve_model_snapshot(
        args.identifier,
        source=args.source,
        source_root=args.source_root,
        allow_download=args.allow_download,
        revision=args.revision,
        cache_dir=args.cache_dir,
        repo_type=args.repo_type,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
