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

"""Generate a reproducible final report for a stopped Auto-FL campaign."""

from __future__ import annotations

import argparse
import csv
import functools
import importlib.util
import inspect
import json
import math
import os
import re
import shlex
import sys
import tempfile
import textwrap
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import quote

try:
    import fcntl
except ImportError:  # pragma: no cover - NVFlare supports POSIX platforms
    fcntl = None

try:
    import yaml
except ImportError:  # pragma: no cover - NVFlare installs PyYAML
    yaml = None


SUMMARY_SCHEMA_VERSION = "nvflare.autofl.report.v1"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
ATTEMPT_STATUSES = {"candidate", "keep", "discard", "crash"}
FINALIZED_SCORE_STATUSES = {"baseline", "keep", "discard"}
RETAINED_STATUSES = {"baseline", "keep"}
TERMINAL_MANIFEST_STATUSES = {"abandoned", "crash", "discard", "keep"}
CAMPAIGN_LOCK_PATH = ".nvflare/autofl/campaign.lock"
FALLBACK_TRAINING_BUDGET_ARGS = {
    "aggregation_epochs",
    "aggregator",
    "aggregator_data_kind",
    "alpha",
    "batch_size",
    "cross_site_eval",
    "eval_batch_size",
    "final_eval_clients",
    "local_train_steps",
    "max_model_params",
    "min_clients",
    "model_arch",
    "n_clients",
    "num_clients",
    "num_rounds",
    "seed",
}
SPECIAL_COMPARISON_BUDGET_ARGS = {"aggregator_data_kind", "cross_site_eval"}
FIXED_BUDGET_TO_CLI = {
    "min_clients": "min_clients",
    "num_clients": "n_clients",
    "num_rounds": "num_rounds",
}
SOURCE_RE = re.compile(r"\[src:\s*([^\]]+)\]", re.IGNORECASE)
ARXIV_RE = re.compile(r"\barxiv\s*:\s*(\d{4}\.\d{4,5})", re.IGNORECASE)


def product_autofl_script_path(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "nvflare-autofl" / "scripts" / name


def _load_product_module(script_name: str, module_name: str, *, catch_system_exit: bool = False):
    module_path = product_autofl_script_path(script_name)
    if not module_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    except SystemExit:
        if not catch_system_exit:
            raise
        return None
    except Exception:
        return None
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module


def load_training_budget_args() -> set:
    """Use the installed producer's comparison vocabulary when it is available."""

    module = _load_product_module("run_job_campaign.py", "nvflare_autofl_report_runner_contract")
    if module is None:
        return set(FALLBACK_TRAINING_BUDGET_ARGS)
    mapping = getattr(module, "COMPARISON_BUDGET_TO_CLI", None)
    fixed_mapping = getattr(module, "FIXED_BUDGET_TO_CLI", None)
    mappings = (mapping, fixed_mapping)
    if not all(
        isinstance(candidate, dict)
        and all(isinstance(key, str) and isinstance(value, str) for key, value in candidate.items())
        for candidate in mappings
    ):
        return set(FALLBACK_TRAINING_BUDGET_ARGS)
    return (
        set(mapping)
        | set(mapping.values())
        | set(fixed_mapping)
        | set(fixed_mapping.values())
        | SPECIAL_COMPARISON_BUDGET_ARGS
    )


def load_campaign_guard_contract():
    """Load product comparison semantics when the Auto-FL producer skill is installed."""

    module = _load_product_module(
        "campaign_guard.py", "nvflare_autofl_report_campaign_guard_contract", catch_system_exit=True
    )
    if module is None or not all(
        callable(getattr(module, name, None)) for name in ("validate_mode", "better", "improvement_over_baseline")
    ):
        return None
    return module


@functools.lru_cache(maxsize=1)
def campaign_guard_contract():
    return load_campaign_guard_contract()


TRAINING_BUDGET_ARGS = load_training_budget_args()


@dataclass(frozen=True)
class RunRecord:
    index: int
    status: str
    name: str
    score: Optional[float]
    runtime_seconds: float
    changed_files: str
    diff_summary: str
    run_command: str
    artifacts: str
    failure_reason: str
    candidate_manifest: str
    base_candidate: str
    patch_sha256: str
    metric_name: str = ""
    metric_source: str = ""
    metric_artifact: str = ""
    candidate_kind: str = ""
    algorithm_family: str = ""
    literature_event_id: str = ""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", nargs="?", default=".", help="job directory containing Auto-FL artifacts")
    parser.add_argument("--results", default="results.tsv")
    parser.add_argument("--state", default=".nvflare/autofl/campaign_state.json")
    parser.add_argument("--autofl-yaml", default="autofl.yaml")
    parser.add_argument("--progress", default="progress.png")
    parser.add_argument("--output", default="autofl_final_report.md")
    parser.add_argument("--summary-json", default="autofl_report_summary.json")
    parser.add_argument(
        "--plotter",
        help="override path to nvflare-autofl plot_progress.py; relative paths resolve from campaign_dir",
    )
    parser.add_argument("--metric", help="override optimization metric label")
    parser.add_argument("--confirm-interrupted", action="store_true", help="confirm an abruptly interrupted campaign")
    parser.add_argument("--agent-model")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--agent-cost")
    parser.add_argument("--agent-context", help="optional JSON or text file with agent/tooling context")
    parser.add_argument("--max-milestones", type=int, default=12)
    parser.add_argument("--max-non-improvements", type=int, default=10)
    return parser.parse_args(argv)


def resolve_path(root: Path, value: str) -> Path:
    try:
        path = Path(value).expanduser()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot resolve report path {value!r}: {exc}") from exc
    return path if path.is_absolute() else root / path


def canonical_path(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot canonicalize report path {path}: {exc}") from exc


def path_comparison_key(path: Path) -> str:
    """Return a portable key that also catches aliases on case-insensitive filesystems."""

    return os.path.normcase(str(canonical_path(path))).casefold()


def paths_alias(left: Path, right: Path) -> bool:
    if path_comparison_key(left) == path_comparison_key(right):
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def validate_output_paths(outputs: Dict[str, Path], protected: Dict[str, Path]) -> None:
    output_items = list(outputs.items())
    for index, (left_name, left_path) in enumerate(output_items):
        for right_name, right_path in output_items[index + 1 :]:
            if paths_alias(left_path, right_path):
                raise ValueError(
                    f"report outputs --{left_name} and --{right_name} must be distinct: {canonical_path(left_path)}"
                )
        for protected_name, protected_path in protected.items():
            if paths_alias(left_path, protected_path):
                raise ValueError(
                    f"report output --{left_name} aliases protected campaign input {protected_name}: "
                    f"{canonical_path(left_path)}"
                )


def campaign_source_contract(root: Path, config: Dict[str, Any]) -> Tuple[Dict[str, Path], List[str]]:
    protected = {"job source": root / "job.py"}
    trust_contract = config.get("trust_contract", {})
    if trust_contract is None:
        trust_contract = {}
    if not isinstance(trust_contract, dict):
        raise ValueError("autofl.yaml trust_contract must be a mapping")

    allowed_edit_paths = trust_contract.get("allowed_edit_paths") or []
    if not isinstance(allowed_edit_paths, list):
        raise ValueError("autofl.yaml trust_contract.allowed_edit_paths must be a list")
    root_resolved = root.resolve()
    for index, value in enumerate(allowed_edit_paths, start=1):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("autofl.yaml trust_contract.allowed_edit_paths must contain non-empty strings")
        candidate = canonical_path(resolve_path(root, value))
        try:
            relative = candidate.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"trust-contract source path escapes the campaign directory: {value}") from exc
        if not relative.parts:
            raise ValueError(f"trust-contract source path must name a file: {value}")
        protected[f"managed source {index} ({relative.as_posix()})"] = candidate

    allowed_create_patterns = trust_contract.get("allowed_create_patterns") or []
    if not isinstance(allowed_create_patterns, list) or not all(
        isinstance(pattern, str) and pattern for pattern in allowed_create_patterns
    ):
        raise ValueError("autofl.yaml trust_contract.allowed_create_patterns must be a list of patterns")
    return protected, list(dict.fromkeys(allowed_create_patterns))


def validate_output_create_patterns(outputs: Dict[str, Path], root: Path, patterns: Sequence[str]) -> None:
    root_resolved = root.resolve()
    for output_name, output_path in outputs.items():
        try:
            relative = canonical_path(output_path).relative_to(root_resolved)
        except ValueError:
            continue
        posix_path = PurePosixPath(relative.as_posix())
        if any(
            posix_path.match(pattern) or (pattern.startswith("**/") and posix_path.match(pattern[3:]))
            for pattern in patterns
        ):
            raise ValueError(
                f"report output --{output_name} matches trust_contract.allowed_create_patterns: {relative.as_posix()}"
            )


@contextmanager
def locked_campaign_workspace(workspace: Path, action: str) -> Iterator[None]:
    """Serialize report generation with Auto-FL lifecycle actions."""

    if fcntl is None:
        raise ValueError("platform without fcntl is unsupported for Auto-FL reporting")
    lock_path = workspace / CAMPAIGN_LOCK_PATH
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"cannot access Auto-FL campaign lock directory {lock_path.parent}: {exc}") from exc
    descriptor = None
    acquired = False
    lock_created = False
    acquisition_complete = False
    try:
        try:
            descriptor = os.open(lock_path, os.O_RDONLY)
        except FileNotFoundError:
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                lock_created = True
            except FileExistsError:
                descriptor = os.open(lock_path, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as exc:
            raise ValueError(
                f"Auto-FL campaign workspace is already in use: {workspace}; "
                "wait for the active lifecycle action to finish, then retry"
            ) from exc
        if lock_created:
            os.write(
                descriptor,
                (
                    json.dumps({"pid": os.getpid(), "action": action, "workspace": str(workspace.resolve())}) + "\n"
                ).encode("utf-8"),
            )
        acquisition_complete = True
        yield
    except OSError as exc:
        if acquisition_complete:
            raise
        raise ValueError(f"cannot acquire Auto-FL campaign lock {lock_path}: {exc}") from exc
    finally:
        if descriptor is not None:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_results(path: Path) -> List[RunRecord]:
    if not path.is_file():
        raise ValueError(f"Auto-FL ledger not found: {path}")
    records = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or not {"status", "score"}.issubset(reader.fieldnames):
            raise ValueError(f"Auto-FL ledger is missing required status/score columns: {path}")
        for index, row in enumerate(reader):
            records.append(
                RunRecord(
                    index=index,
                    status=(row.get("status") or "").strip().lower(),
                    name=(row.get("name") or row.get("candidate") or row.get("commit") or f"row_{index}").strip(),
                    score=finite_float(row.get("score")),
                    runtime_seconds=max(0.0, finite_float(row.get("runtime_seconds")) or 0.0),
                    changed_files=(row.get("changed_files") or "").strip(),
                    diff_summary=(row.get("diff_summary") or row.get("description") or "").strip(),
                    run_command=(row.get("run_command") or "").strip(),
                    artifacts=(row.get("artifacts") or "").strip(),
                    failure_reason=(row.get("failure_reason") or "").strip(),
                    candidate_manifest=(row.get("candidate_manifest") or "").strip(),
                    base_candidate=(row.get("base_candidate") or "").strip(),
                    patch_sha256=(row.get("patch_sha256") or "").strip(),
                    metric_name=(row.get("metric_name") or "").strip(),
                    metric_source=(row.get("metric_source") or "").strip(),
                    metric_artifact=(row.get("metric_artifact") or "").strip(),
                    candidate_kind=(row.get("candidate_kind") or "").strip().lower(),
                    algorithm_family=(row.get("algorithm_family") or "").strip().lower(),
                    literature_event_id=(row.get("literature_event_id") or "").strip(),
                )
            )
    if not records:
        raise ValueError(f"Auto-FL ledger has no rows: {path}")
    return records


def load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    if yaml is None:
        raise ValueError("PyYAML is required to read autofl.yaml")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read Auto-FL config from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return value


def validate_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    contract = campaign_guard_contract()
    if contract is not None:
        return contract.validate_mode(mode)
    if mode not in {"min", "max"}:
        raise ValueError(f"objective mode must be 'min' or 'max', but got {mode!r}")
    return mode


def better(score: Optional[float], incumbent: Optional[float], mode: str = "max") -> bool:
    contract = campaign_guard_contract()
    if contract is not None:
        return contract.better(score, incumbent, validate_mode(mode))
    mode = validate_mode(mode)
    if score is None:
        return False
    if incumbent is None:
        return True
    return score < incumbent if mode == "min" else score > incumbent


def normalize_contract_sections(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    normalized = dict(config)
    warnings = []
    for section in ("objective", "environment", "budget"):
        if section not in normalized:
            continue
        value = normalized[section]
        if not isinstance(value, dict):
            kind = "null" if value is None else type(value).__name__
            warnings.append(
                f"autofl.yaml section '{section}' is {kind}, not a mapping; it was treated as an empty section."
            )
            normalized[section] = {}
    return normalized, warnings


def metric_contract(
    config: Dict[str, Any], state: Dict[str, Any], args: argparse.Namespace
) -> Tuple[str, str, str, str]:
    objective = config.get("objective") if isinstance(config.get("objective"), dict) else {}
    metric = (
        args.metric or objective.get("optimization_metric") or objective.get("metric") or state.get("metric") or "score"
    )
    requested = objective.get("requested_metric") or objective.get("metric") or metric
    measurement_source = objective.get("metric_source") or "NVFlare metric artifacts"
    contract_source = objective.get("metric_contract_source") or "not declared"
    return str(metric), str(requested), str(measurement_source), str(contract_source)


def metric_mode(config: Dict[str, Any], state: Dict[str, Any]) -> str:
    objective = config.get("objective") if isinstance(config.get("objective"), dict) else {}
    direction = objective.get("direction")
    if direction is not None:
        direction = {"maximize": "max", "minimize": "min"}.get(str(direction).strip().lower(), direction)
    declared = [
        ("autofl.yaml objective.mode", objective.get("mode")),
        ("autofl.yaml objective.direction", direction),
        ("campaign state mode", state.get("mode")),
    ]
    resolved = []
    for source, value in declared:
        if value is not None:
            resolved.append((source, validate_mode(value)))
    modes = {value for _, value in resolved}
    if len(modes) > 1:
        details = ", ".join(f"{source}={value!r}" for source, value in resolved)
        raise ValueError(f"metric direction disagrees across campaign artifacts: {details}")
    mode = next(iter(modes), "max")
    if mode == "min" and not objective.get("mode_contract_source"):
        raise ValueError(
            "legacy minimization campaign lacks native metric-direction provenance; initialize a fresh campaign"
        )
    return mode


def verify_stopped(state_path: Path, confirm_interrupted: bool) -> Tuple[Dict[str, Any], str, List[str]]:
    warnings = []
    if not state_path.is_file():
        if not confirm_interrupted:
            raise ValueError(
                f"campaign state not found at {state_path}; use --confirm-interrupted only after confirming execution stopped"
            )
        warnings.append("Campaign state was unavailable; the user explicitly confirmed interruption.")
        return {}, "user_confirmed_interruption", warnings

    state = load_json(state_path)
    if state.get("final_response_allowed") is True:
        return state, str(state.get("reason") or "campaign_stopped"), warnings
    if not confirm_interrupted:
        raise ValueError(
            "campaign state still has final_response_allowed=false; stop the campaign or pass --confirm-interrupted "
            "after independently confirming its processes are no longer running"
        )
    warnings.append("Campaign state remained active; the user explicitly confirmed that execution was interrupted.")
    return state, "user_confirmed_interruption", warnings


def is_baseline(record: RunRecord) -> bool:
    return record.status == "baseline"


def is_finalized_scored_record(record: RunRecord) -> bool:
    return record.score is not None and record.status in FINALIZED_SCORE_STATUSES


def candidate_manifest_paths(campaign_root: Path, records: Sequence[RunRecord]) -> List[Path]:
    candidates_root = campaign_root / ".nvflare" / "autofl" / "candidates"
    paths = set(candidates_root.glob("*/candidate_manifest.json")) if candidates_root.is_dir() else set()
    paths.update(
        resolve_path(campaign_root, record.candidate_manifest) for record in records if record.candidate_manifest
    )
    return sorted(paths)


def candidate_manifest_evidence(
    campaign_root: Path, records: Sequence[RunRecord]
) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    unfinished = []
    unreadable = []
    for path in candidate_manifest_paths(campaign_root, records):
        if not path.is_file():
            continue
        try:
            manifest = load_json(path)
        except ValueError as exc:
            unreadable.append((path, str(exc)))
            continue
        status = str(manifest.get("status") or "").strip().lower()
        if status not in TERMINAL_MANIFEST_STATUSES:
            unfinished.append((path, status))
    return unfinished, unreadable


def verify_no_pending_candidates(campaign_root: Path, state: Dict[str, Any], records: Sequence[RunRecord]) -> None:
    evidence = []
    pending_count = finite_float(state.get("pending_candidates"))
    if pending_count is not None and pending_count > 0:
        evidence.append(f"campaign state reports pending_candidates={state.get('pending_candidates')}")
    if state.get("pending_candidate_manifest"):
        evidence.append(f"campaign state names {state['pending_candidate_manifest']}")

    ledger_rows = [record for record in records if record.status == "candidate"]
    if ledger_rows:
        rows = ", ".join(f"row {record.index + 1} ({record.name})" for record in ledger_rows[:5])
        if len(ledger_rows) > 5:
            rows += f", and {len(ledger_rows) - 5} more"
        evidence.append(f"results.tsv contains pending candidate rows: {rows}")

    manifests, unreadable_manifests = candidate_manifest_evidence(campaign_root, records)
    if manifests:
        paths = ", ".join(f"{path.resolve()} (status={status or '<missing>'})" for path, status in manifests[:3])
        if len(manifests) > 3:
            paths += f", and {len(manifests) - 3} more"
        evidence.append(f"candidate manifests have unfinished or unknown status: {paths}")
    if unreadable_manifests:
        details = "; ".join(f"{path.resolve()} ({error})" for path, error in unreadable_manifests[:3])
        if len(unreadable_manifests) > 3:
            details += f"; and {len(unreadable_manifests) - 3} more"
        evidence.append(f"candidate manifests could not be read and may contain unfinished work: {details}")

    if evidence:
        raise ValueError(
            "cannot finalize while candidate evidence is unfinished; finalize or abandon pending candidates first. "
            "--confirm-interrupted bypasses only stale campaign stop state, never unfinished candidate evidence. "
            + " ".join(evidence)
        )


def scored_records(records: Iterable[RunRecord]) -> List[RunRecord]:
    return [record for record in records if is_finalized_scored_record(record)]


def select_baseline(records: Sequence[RunRecord]) -> Optional[RunRecord]:
    return next(
        (record for record in records if record.status == "baseline" and is_finalized_scored_record(record)), None
    )


def select_best(records: Sequence[RunRecord], retained_only: bool = False, mode: str = "max") -> Optional[RunRecord]:
    candidates = [
        record
        for record in records
        if is_finalized_scored_record(record)
        and (record.status in RETAINED_STATUSES if retained_only else record.status in FINALIZED_SCORE_STATUSES)
    ]
    if not candidates:
        return None
    selector = min if validate_mode(mode) == "min" else max
    return selector(candidates, key=lambda record: record.score)


def running_best_milestones(records: Sequence[RunRecord], limit: int, mode: str = "max") -> List[Dict[str, Any]]:
    milestones = []
    incumbent = None
    for record in records:
        if not is_finalized_scored_record(record) or not better(record.score, incumbent, mode):
            continue
        previous = incumbent
        incumbent = record.score
        milestones.append(
            {
                "row": record.index + 1,
                "name": record.name,
                "status": record.status,
                "score": record.score,
                "delta_from_previous_best": None if previous is None else record.score - previous,
                "improvement_from_previous_best": (
                    None if previous is None else improvement_amount(record.score, previous, mode)
                ),
                "hypothesis": record.diff_summary,
                "changed_files": split_files(record.changed_files),
                "candidate_kind": inferred_candidate_kind(record),
                "algorithm_family": record.algorithm_family
                or ("baseline" if record.status == "baseline" else "unclassified"),
                "literature_event_id": record.literature_event_id,
            }
        )
    if limit > 0 and len(milestones) > limit:
        if limit == 1:
            return [milestones[-1]]
        selected = {0, len(milestones) - 1}
        middle = sorted(
            range(1, len(milestones) - 1),
            key=lambda index: (
                milestones[index]["improvement_from_previous_best"] or 0.0,
                -milestones[index]["row"],
            ),
            reverse=True,
        )
        selected.update(middle[: max(0, limit - 2)])
        return [milestones[index] for index in sorted(selected)]
    return milestones


def improvement_amount(score: float, incumbent: float, mode: str = "max") -> float:
    contract = campaign_guard_contract()
    if contract is not None:
        return contract.improvement_over_baseline(incumbent, score, validate_mode(mode))
    return incumbent - score if validate_mode(mode) == "min" else score - incumbent


def split_files(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip() and item.strip().lower() != "none"]


def candidate_lineage(best: Optional[RunRecord], records: Sequence[RunRecord]) -> Dict[str, Any]:
    if best is None:
        return {"candidates": [], "changed_files": [], "complete": False}
    by_name = {record.name: record for record in records if record.name}
    chain = []
    changed_files = []
    seen = set()
    current = best
    complete = True
    while current:
        if current.name in seen:
            complete = False
            break
        seen.add(current.name)
        chain.append(current.name)
        changed_files.extend(split_files(current.changed_files))
        if not current.base_candidate:
            complete = current.status == "baseline"
            break
        base_candidate = current.base_candidate
        current = by_name.get(base_candidate)
        if current is None:
            chain.append(base_candidate)
            complete = False
            break
    return {
        "candidates": list(reversed(chain)),
        "changed_files": sorted(set(changed_files)),
        "complete": complete,
    }


def parse_sources(text: str) -> List[str]:
    sources = []
    for match in SOURCE_RE.findall(text or ""):
        for item in re.split(r"\s*;\s*", match):
            if item and item not in sources:
                sources.append(item)
    for identifier in ARXIV_RE.findall(text or ""):
        source = f"arXiv:{identifier}"
        if not any(identifier in existing for existing in sources):
            sources.append(source)
    return sources


def is_literature_event(record: RunRecord) -> bool:
    return record.status == "literature"


def inferred_candidate_kind(record: RunRecord) -> str:
    if record.status == "baseline":
        return "baseline"
    if record.candidate_kind:
        return record.candidate_kind
    changed = record.changed_files.strip().lower()
    return "source_edit" if changed and changed != "none" else "argument_only"


def campaign_local_manifest_path(campaign_root: Path, candidate_id: str) -> Optional[Path]:
    candidate_id = candidate_id.strip()
    if (
        not candidate_id
        or candidate_id in {".", ".."}
        or Path(candidate_id).name != candidate_id
        or "/" in candidate_id
        or "\\" in candidate_id
    ):
        return None
    return campaign_root / ".nvflare" / "autofl" / "candidates" / candidate_id / "candidate_manifest.json"


def manifest_summary(record: Optional[RunRecord], campaign_root: Path) -> Dict[str, Any]:
    if record is None or not record.candidate_manifest:
        return {}
    recorded_path = record.candidate_manifest
    path = resolve_path(campaign_root, recorded_path)
    resolution = "recorded"
    if not path.is_file():
        local_path = campaign_local_manifest_path(campaign_root, record.name)
        if local_path is None:
            return {
                "path": str(path),
                "recorded_path": recorded_path,
                "resolved_path": None,
                "resolution": "unavailable",
                "available": False,
                "error": f"candidate name is not a safe manifest identifier: {record.name!r}",
            }
        if not local_path.is_file():
            return {
                "path": str(path),
                "recorded_path": recorded_path,
                "resolved_path": None,
                "resolution": "unavailable",
                "available": False,
            }
        path = local_path
        resolution = "campaign_local"

    resolved_path = str(path.resolve())
    provenance = {
        "path": resolved_path,
        "recorded_path": recorded_path,
        "resolved_path": resolved_path,
        "resolution": resolution,
    }
    try:
        manifest = load_json(path)
    except ValueError as exc:
        return {**provenance, "available": False, "error": str(exc)}
    manifest_candidate_id = str(manifest.get("candidate_id") or "")
    if manifest_candidate_id != record.name:
        return {
            **provenance,
            "available": False,
            "error": (f"candidate manifest ID mismatch: expected {record.name!r}, " f"found {manifest_candidate_id!r}"),
        }
    return {
        **provenance,
        "available": True,
        "schema_version": manifest.get("schema_version"),
        "candidate_id": manifest_candidate_id,
        "base_candidate": manifest.get("base_candidate"),
        "hypothesis": manifest.get("hypothesis"),
        "run_args": manifest.get("run_args") or [],
        "changed_files": manifest.get("changed_files") or [],
        "created_files": manifest.get("created_files") or [],
        "candidate_kind": manifest.get("candidate_kind"),
        "algorithm_family": manifest.get("algorithm_family"),
        "literature_event_id": manifest.get("literature_event_id"),
        "source_sha256": manifest.get("candidate_source_sha256") or manifest.get("base_source_sha256"),
        "budget_sha256": manifest.get("fixed_budget_sha256") or manifest.get("budget_sha256"),
        "patch_sha256": manifest.get("patch_sha256"),
        "status": manifest.get("status"),
        "artifacts": manifest.get("artifacts") or {},
        "result": manifest.get("result") or {},
    }


def validated_exploration_batch(exploration_batch: Any) -> Optional[Tuple[str, int, int]]:
    if not isinstance(exploration_batch, dict):
        return None
    event_id = exploration_batch.get("literature_event_id")
    completed = exploration_batch.get("completed")
    required = exploration_batch.get("required")
    if (
        not isinstance(event_id, str)
        or not event_id
        or not isinstance(completed, int)
        or isinstance(completed, bool)
        or not isinstance(required, int)
        or isinstance(required, bool)
        or completed < 0
        or required <= 0
    ):
        return None
    return event_id, completed, required


def literature_outcomes(
    records: Sequence[RunRecord],
    exploration_batch: Optional[Dict[str, Any]] = None,
    termination_reason: Optional[str] = None,
    mode: str = "max",
) -> List[Dict[str, Any]]:
    events = []
    for index, record in enumerate(records):
        if is_literature_event(record):
            events.append((index, record, record.literature_event_id))
    outcomes = []
    validated_batch = validated_exploration_batch(exploration_batch)
    for start, event, event_id in events:
        before = select_best(records[:start], retained_only=True, mode=mode)
        attempts = [
            record
            for index, record in enumerate(records)
            if record.status in ATTEMPT_STATUSES
            and not is_baseline(record)
            and index > start
            and bool(event_id)
            and record.literature_event_id == event_id
        ]
        scored = [record for record in attempts if is_finalized_scored_record(record)]
        segment_best = select_best(scored, mode=mode)
        completion = None
        if validated_batch and bool(event_id) and validated_batch[0] == event_id:
            _, completed, required = validated_batch
            completion = {
                "completed": completed,
                "required": required,
                "complete": completed >= required,
                "reason": None if completed >= required else termination_reason,
            }
        if completion and not completion["complete"]:
            outcome = "incomplete"
            delta = None if segment_best is None or before is None else segment_best.score - before.score
        elif segment_best is None:
            outcome = "failed" if attempts else "not_evaluated"
            delta = None
        elif before is None or better(segment_best.score, before.score, mode):
            outcome = "helped"
            delta = None if before is None else segment_best.score - before.score
        elif segment_best.score == before.score:
            outcome = "matched"
            delta = 0.0
        else:
            outcome = "not_confirmed"
            delta = segment_best.score - before.score
        outcomes.append(
            {
                "event": event.name,
                "literature_event_id": event_id,
                "row": event.index + 1,
                "hypothesis": event.diff_summary,
                "sources": parse_sources(event.diff_summary),
                "outcome": outcome,
                "incumbent_score": None if before is None else before.score,
                "best_candidate": None if segment_best is None else segment_best.name,
                "best_score": None if segment_best is None else segment_best.score,
                "delta_from_incumbent": delta,
                "improvement_from_incumbent": (
                    None
                    if before is None or segment_best is None
                    else improvement_amount(segment_best.score, before.score, mode)
                ),
                "candidate_attempts": [record.name for record in attempts],
                "completion": completion,
                "candidate_results": [
                    {
                        "name": record.name,
                        "status": record.status,
                        "score": record.score,
                        "candidate_kind": inferred_candidate_kind(record),
                        "algorithm_family": record.algorithm_family,
                        "metric_name": record.metric_name,
                        "metric_source": record.metric_source,
                        "metric_artifact": record.metric_artifact,
                    }
                    for record in attempts
                ],
                "failures": [record.name for record in attempts if record.status == "crash"],
            }
        )
    return outcomes


def literature_completion_warnings(
    reviews: Sequence[Dict[str, Any]],
    exploration_batch: Optional[Dict[str, Any]] = None,
    termination_reason: Optional[str] = None,
) -> List[str]:
    warnings = []
    for review in reviews:
        completion = review.get("completion")
        if not isinstance(completion, dict) or completion.get("complete") is not False:
            continue
        reason = completion.get("reason") or "campaign termination"
        warnings.append(
            f"Literature exploration batch {review['literature_event_id'] or review['event']} was incomplete at "
            f"{reason}: {completion['completed']} of {completion['required']} required candidates completed."
        )
    validated_batch = validated_exploration_batch(exploration_batch)
    review_event_ids = {review.get("literature_event_id") for review in reviews}
    if validated_batch and validated_batch[1] < validated_batch[2] and validated_batch[0] not in review_event_ids:
        event_id, completed, required = validated_batch
        reason = termination_reason or "campaign termination"
        warnings.append(
            f"Literature exploration batch {event_id} was incomplete at {reason}: {completed} of {required} required "
            "candidates completed, but no matching literature checkpoint was found in the results ledger."
        )
    return warnings


def candidate_outcome_payload(record: RunRecord, incumbent: Optional[RunRecord], mode: str = "max") -> Dict[str, Any]:
    delta = None if incumbent is None or record.score is None else record.score - incumbent.score
    return {
        "row": record.index + 1,
        "name": record.name,
        "status": record.status,
        "score": record.score,
        "incumbent_name": None if incumbent is None else incumbent.name,
        "incumbent_score": None if incumbent is None else incumbent.score,
        "delta_from_incumbent": delta,
        "improvement_from_incumbent": (
            None
            if incumbent is None or record.score is None
            else improvement_amount(record.score, incumbent.score, mode)
        ),
        "hypothesis": record.diff_summary,
        "changed_files": split_files(record.changed_files),
        "candidate_kind": inferred_candidate_kind(record),
        "algorithm_family": record.algorithm_family or "unclassified",
        "literature_event_id": record.literature_event_id,
        "failure_reason": record.failure_reason,
    }


def select_major_improvements(items: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[-1]]
    selected = {0, len(items) - 1}
    middle = sorted(
        range(1, len(items) - 1),
        key=lambda index: (items[index]["improvement_from_incumbent"] or 0.0, -items[index]["row"]),
        reverse=True,
    )
    selected.update(middle[: max(0, limit - 2)])
    return [items[index] for index in sorted(selected)]


def representative_non_improvements(items: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    representatives: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for item in items:
        family = item["algorithm_family"]
        event_id = item["literature_event_id"]
        if family != "unclassified" and event_id:
            key = ("family_literature", family, event_id)
            label = f"family={family}, literature={event_id}"
        elif family != "unclassified":
            key = ("family", family)
            label = f"family={family}"
        elif event_id:
            key = ("literature", event_id)
            label = f"literature={event_id}"
        else:
            key = ("candidate", item["name"])
            label = f"candidate={item['name']}"
        candidate = dict(item)
        candidate["representative_for"] = label
        previous = representatives.get(key)
        candidate_rank = candidate["improvement_from_incumbent"]
        previous_rank = None if previous is None else previous["improvement_from_incumbent"]
        if previous is None or (
            candidate_rank is not None and (previous_rank is None or candidate_rank > previous_rank)
        ):
            representatives[key] = candidate
    ranked = sorted(
        representatives.values(),
        key=lambda item: (
            item["improvement_from_incumbent"] is not None,
            (item["improvement_from_incumbent"] if item["improvement_from_incumbent"] is not None else -math.inf),
            -item["row"],
        ),
        reverse=True,
    )
    return ranked[: max(0, limit)]


def failure_outcomes(items: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        reason = " ".join((item["failure_reason"] or "no score or failure reason recorded").split())
        groups.setdefault(reason, []).append(item)
    outcomes = []
    for reason, failures in groups.items():
        outcomes.append(
            {
                "reason": reason,
                "count": len(failures),
                "candidates": [item["name"] for item in failures],
                "statuses": sorted({item["status"] for item in failures}),
                "algorithm_families": sorted({item["algorithm_family"] for item in failures}),
                "candidate_kinds": sorted({item["candidate_kind"] for item in failures}),
            }
        )
    outcomes.sort(key=lambda item: (-item["count"], item["reason"]))
    return outcomes[: max(0, limit)]


def family_outcomes(records: Sequence[RunRecord], helped_rows: set[int], mode: str = "max") -> List[Dict[str, Any]]:
    families: Dict[str, List[RunRecord]] = {}
    for record in records:
        if record.status not in {"keep", "discard", "crash"}:
            continue
        families.setdefault(record.algorithm_family or "unclassified", []).append(record)
    outcomes = []
    for family, attempts in families.items():
        counts = Counter(record.status for record in attempts)
        helped = [record for record in attempts if record.index in helped_rows]
        scored_discards = [record for record in attempts if record.status == "discard" and record.score is not None]
        scored_unhelpful_keeps = [
            record
            for record in attempts
            if record.status == "keep" and record.score is not None and record.index not in helped_rows
        ]
        unscored_keeps = [record for record in attempts if record.status == "keep" and record.score is None]
        if helped and (counts.get("discard", 0) or counts.get("crash", 0) or unscored_keeps):
            outcome = "mixed"
        elif helped:
            outcome = "helped"
        elif scored_discards or scored_unhelpful_keeps:
            outcome = "not_confirmed"
        else:
            outcome = "failed"
        retained = select_best(
            [record for record in attempts if record.status == "keep"], retained_only=True, mode=mode
        )
        observed = select_best(attempts, mode=mode)
        outcomes.append(
            {
                "algorithm_family": family,
                "outcome": outcome,
                "attempts": len(attempts),
                "kept": counts.get("keep", 0),
                "discarded": counts.get("discard", 0),
                "crashed": counts.get("crash", 0),
                "best_retained_score": None if retained is None else retained.score,
                "best_observed_score": None if observed is None else observed.score,
                "candidate_kinds": sorted({inferred_candidate_kind(record) for record in attempts}),
                "literature_event_ids": sorted(
                    {record.literature_event_id for record in attempts if record.literature_event_id}
                ),
            }
        )
    order = {"helped": 0, "mixed": 1, "not_confirmed": 2, "failed": 3}
    return sorted(outcomes, key=lambda item: (order[item["outcome"]], item["algorithm_family"]))


def literature_digest(reviews: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = dict(sorted(Counter(review["outcome"] for review in reviews).items()))
    helped = [
        review
        for review in reviews
        if review["outcome"] == "helped" and review["improvement_from_incumbent"] is not None
    ]
    strongest = None
    if helped:
        strongest = max(
            helped,
            key=lambda review: review["improvement_from_incumbent"],
        )
        strongest = {
            "event": strongest["event"],
            "literature_event_id": strongest["literature_event_id"],
            "best_candidate": strongest["best_candidate"],
            "best_score": strongest["best_score"],
            "incumbent_score": strongest["incumbent_score"],
            "delta_from_incumbent": strongest["delta_from_incumbent"],
            "improvement_from_incumbent": strongest["improvement_from_incumbent"],
            "sources": strongest["sources"],
        }
    return {"counts": counts, "strongest_helped": strongest}


def outcome_summary(
    records: Sequence[RunRecord],
    reviews: Sequence[Dict[str, Any]],
    max_milestones: int,
    max_non_improvements: int,
    mode: str = "max",
) -> Dict[str, Any]:
    incumbent = None
    helped = []
    non_improvements = []
    failures = []
    for record in records:
        if record.status in RETAINED_STATUSES and record.score is not None:
            if record.status == "keep" and incumbent is not None and better(record.score, incumbent.score, mode):
                helped.append(candidate_outcome_payload(record, incumbent, mode))
            if incumbent is None or better(record.score, incumbent.score, mode):
                incumbent = record
            continue
        if record.status == "discard":
            payload = candidate_outcome_payload(record, incumbent, mode)
            (non_improvements if record.score is not None else failures).append(payload)
        elif record.status == "crash":
            failures.append(candidate_outcome_payload(record, incumbent, mode))
    helped_rows = {item["row"] - 1 for item in helped}
    return {
        "helped": helped,
        "major_helped": select_major_improvements(helped, max_milestones),
        "not_confirmed": representative_non_improvements(non_improvements, max_non_improvements),
        "failures": failure_outcomes(failures, max_non_improvements),
        "families": family_outcomes(records, helped_rows, mode),
        "literature": literature_digest(reviews),
    }


def selection_summary(
    best: Optional[RunRecord], baseline: Optional[RunRecord], lineage: Dict[str, Any], mode: str = "max"
) -> Optional[Dict[str, Any]]:
    if best is None:
        return None
    delta = None if baseline is None else best.score - baseline.score
    improvement = None if baseline is None else improvement_amount(best.score, baseline.score, mode)
    return {
        "candidate": best.name,
        "score": best.score,
        "delta_from_baseline": delta,
        "improvement_from_baseline": improvement,
        "algorithm_family": best.algorithm_family or ("baseline" if best.status == "baseline" else "unclassified"),
        "candidate_kind": "baseline" if best.status == "baseline" else inferred_candidate_kind(best),
        "hypothesis": best.diff_summary,
        "base_candidate": best.base_candidate,
        "literature_event_id": best.literature_event_id,
        "lineage": lineage,
        "reason": (
            f"It is the best scored retained result under the {mode} objective."
            if best.status != "baseline"
            else "No retained candidate improved on the scored baseline."
        ),
    }


def command_options(command: str) -> Dict[str, Any]:
    if not command:
        return {}
    try:
        tokens = shlex.split(command)
    except ValueError:
        return {}
    options: Dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        key_value = token[2:].split("=", 1)
        key = key_value[0].replace("-", "_")
        if len(key_value) == 2:
            options[key] = key_value[1]
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            options[key] = tokens[index + 1]
            index += 1
        else:
            options[key] = True
        index += 1
    return options


def command_changes(baseline: Optional[RunRecord], best: Optional[RunRecord]) -> Dict[str, Dict[str, Any]]:
    if baseline is None or best is None:
        return {}
    base = command_options(baseline.run_command)
    candidate = command_options(best.run_command)
    changes = {}
    for key in sorted(set(base) | set(candidate)):
        if key == "name" or base.get(key) == candidate.get(key):
            continue
        changes[key] = {"baseline": base.get(key), "best": candidate.get(key)}
    return changes


def values_equal(left: Any, right: Any) -> bool:
    if str(left) == str(right):
        return True
    left_number = finite_float(left)
    right_number = finite_float(right)
    return left_number is not None and right_number is not None and math.isclose(left_number, right_number)


def comparability_warnings(
    config: Dict[str, Any],
    records: Sequence[RunRecord],
    baseline: Optional[RunRecord],
    best: Optional[RunRecord],
    metric: str,
    metric_source: str,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    warnings = []
    changes = command_changes(baseline, best)
    changed_budget = sorted(set(changes).intersection(TRAINING_BUDGET_ARGS))
    if changed_budget:
        warnings.append(
            "The best run changed executed training/comparison arguments relative to baseline: "
            + ", ".join(changed_budget)
            + ". Treat the gain as non-equal-compute unless this was explicitly approved."
        )

    budget = config.get("budget") if isinstance(config.get("budget"), dict) else {}
    fixed = budget.get("fixed_training_budget", {})
    baseline_options = command_options(baseline.run_command) if baseline else {}
    fixed_mismatches = []
    if isinstance(fixed, dict):
        for config_key, expected in fixed.items():
            cli_key = FIXED_BUDGET_TO_CLI.get(config_key, config_key)
            actual = baseline_options.get(cli_key)
            if actual is not None and not values_equal(actual, expected):
                fixed_mismatches.append(f"{config_key}: autofl.yaml={expected}, executed={actual}")
    if fixed_mismatches:
        warnings.append(
            "The imported fixed budget differs from the executed baseline command ("
            + "; ".join(fixed_mismatches)
            + ")."
        )

    scored_candidate_count = sum(
        record.status in FINALIZED_SCORE_STATUSES and record.status != "baseline" and record.score is not None
        for record in records
    )
    metric_text = f"{metric} {metric_source}".lower()
    if scored_candidate_count > 1 and "test" in metric_text:
        warnings.append(
            "Multiple candidates were selected against a test-like metric. Re-evaluate the chosen candidate once on an "
            "untouched holdout before making generalization claims."
        )
    if not config:
        warnings.append(
            "autofl.yaml was unavailable, so the declared objective and fixed-budget contract could not be verified."
        )
    return warnings, changes


def state_accounting(
    state: Dict[str, Any],
    candidate_attempts: int,
    baseline: Optional[RunRecord],
    best: Optional[RunRecord],
    results_path: Path,
    campaign_root: Path,
    mode: str = "max",
) -> Tuple[Dict[str, Any], List[str]]:
    ledger_baseline = None if baseline is None else baseline.score
    ledger_improvement = (
        None if baseline is None or best is None else improvement_amount(best.score, baseline.score, mode)
    )
    ledger_values = {
        "candidate_attempts": candidate_attempts,
        "baseline_score": ledger_baseline,
        "improvement": ledger_improvement,
    }
    state_values = {key: state.get(key) for key in ledger_values}
    state_values["abandoned_candidates"] = state.get("abandoned_candidates")
    state_values["results"] = state.get("results")
    warnings = []

    if "results" in state:
        state_results = state.get("results")
        if not isinstance(state_results, str) or not state_results.strip():
            warnings.append(f"Campaign state results is invalid: {state_results!r}.")
        else:
            try:
                state_results_path = canonical_path(resolve_path(campaign_root, state_results))
            except ValueError as exc:
                warnings.append(f"Campaign state results is invalid: {state_results!r} ({exc}).")
            else:
                parsed_results_path = canonical_path(results_path)
                if state_results_path != parsed_results_path:
                    warnings.append(
                        f"Campaign state names ledger {state_results_path} but the report was generated from "
                        f"{parsed_results_path}."
                    )

    if "candidate_attempts" in state:
        state_attempts = finite_float(state.get("candidate_attempts"))
        if state_attempts is None or not state_attempts.is_integer() or int(state_attempts) != candidate_attempts:
            warnings.append(
                "Campaign state candidate_attempts disagrees with the ledger "
                f"(state={state.get('candidate_attempts')!r}, ledger={candidate_attempts})."
            )

    for field, ledger_value in (("baseline_score", ledger_baseline), ("improvement", ledger_improvement)):
        if field not in state:
            continue
        state_value = finite_float(state.get(field))
        agrees = (
            state_value is None
            and ledger_value is None
            or state_value is not None
            and ledger_value is not None
            and math.isclose(state_value, ledger_value, rel_tol=1e-9, abs_tol=1e-12)
        )
        if not agrees:
            warnings.append(
                f"Campaign state {field} disagrees with the ledger "
                f"(state={state.get(field)!r}, ledger={ledger_value!r})."
            )

    abandoned_candidates = state.get("abandoned_candidates")
    if abandoned_candidates is not None:
        parsed_abandoned = finite_float(abandoned_candidates)
        if parsed_abandoned is None or not parsed_abandoned.is_integer() or parsed_abandoned < 0:
            warnings.append(
                f"Campaign state abandoned_candidates is invalid: {abandoned_candidates!r}; it was not reported."
            )
            abandoned_candidates = None
        else:
            abandoned_candidates = int(parsed_abandoned)

    return {
        "consistent": not warnings,
        "ledger": ledger_values,
        "campaign_state": state_values,
        "abandoned_candidates": abandoned_candidates,
    }, warnings


def default_plotter_path() -> Path:
    return product_autofl_script_path("plot_progress.py")


def refresh_plot(results: Path, output: Path, metric: str, plotter_path: Path, mode: str = "max") -> Optional[str]:
    if not plotter_path.is_file():
        return f"Auto-FL progress plotter not found at {plotter_path}; existing plot was preserved."
    spec = importlib.util.spec_from_file_location("nvflare_autofl_report_plotter", plotter_path)
    if spec is None or spec.loader is None:
        return f"Could not load Auto-FL progress plotter from {plotter_path}; existing plot was preserved."
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        parameters = inspect.signature(module.plot_progress).parameters
        accepts_mode = "mode" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        if mode == "min" and not accepts_mode:
            return (
                f"Progress plotter at {plotter_path} cannot represent mode='min'; existing plot was preserved. "
                "Use the current nvflare-autofl plotter to refresh this campaign."
            )
        kwargs = {"metric_label": metric}
        if accepts_mode:
            kwargs["mode"] = mode
        module.plot_progress(module.load_results(results), output, **kwargs)
    except Exception as exc:  # plotting should not destroy an otherwise useful stopped-campaign report
        return f"Could not refresh progress plot ({type(exc).__name__}: {exc}); existing plot was preserved."
    finally:
        if previous_module is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_module
    return None


def is_png(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE
    except OSError:
        return False


def compact_lineage(lineage: Sequence[str], edge_items: int = 4) -> str:
    if len(lineage) <= edge_items * 2 + 1:
        return " -> ".join(lineage)
    hidden = len(lineage) - edge_items * 2
    return " -> ".join([*lineage[:edge_items], f"... ({hidden} intermediate)", *lineage[-edge_items:]])


def read_agent_context(path: Optional[Path], args: argparse.Namespace) -> Dict[str, Any]:
    context = {}
    if path:
        if not path.is_file():
            raise ValueError(f"agent context file not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            context["notes"] = text
        else:
            context.update(parsed if isinstance(parsed, dict) else {"notes": parsed})
    if args.agent_model:
        context["model"] = args.agent_model
    if args.reasoning_effort:
        context["reasoning_effort"] = args.reasoning_effort
    if args.agent_cost:
        context["cost"] = args.agent_cost
    return context


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as f:
            temp_path = Path(f.name)
            f.write(text)
        os.replace(temp_path, path)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def md_cell(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def markdown_artifact_path(artifact: str, report: str) -> str:
    try:
        path = Path(os.path.relpath(artifact, start=Path(report).parent)).as_posix()
    except ValueError:
        path = Path(artifact).as_posix()
    return quote(path, safe="/:")


def literature_outcome_label(review: Dict[str, Any]) -> str:
    completion = review.get("completion")
    if isinstance(completion, dict) and completion.get("complete") is False:
        label = f"incomplete ({completion['completed']}/{completion['required']}"
        if completion.get("reason"):
            label += f"; {md_cell(completion['reason'], 60)}"
        return label + ")"
    return str(review["outcome"])


def format_score(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def format_delta(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+.6f}"


def format_runtime(seconds: float) -> str:
    if seconds <= 0:
        return ""
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def format_command(command: str) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command or "not recorded"
    if not tokens:
        return "not recorded"
    chunks = []
    current = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--") and current:
            chunks.append(current)
            current = []
        current.append(token)
        if token.startswith("--") and index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            index += 1
            current.append(tokens[index])
        index += 1
    if current:
        chunks.append(current)
    return " \\\n  ".join(shlex.join(chunk) for chunk in chunks)


def wrap_markdown_bullet(prefix: str, text: str, width: int = 120) -> List[str]:
    return textwrap.wrap(
        f"{prefix}{' '.join(text.split())}",
        width=width,
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=False,
    )


def report_markdown(summary: Dict[str, Any], records: Sequence[RunRecord]) -> str:
    baseline = summary["baseline"]
    best = summary["best"]
    counts = summary["status_counts"]
    runtime_label = format_runtime(summary["runtime_seconds"]) or "n/a"
    lines = [
        "# Auto-FL Final Report",
        "",
        "## Executive Summary",
        "",
        f"The stopped campaign optimized `{summary['objective']['optimization_metric']}` with "
        f"`{summary['objective']['mode']}` direction in the `{summary['environment']}` environment. "
        f"It evaluated `{summary['candidate_attempts']}` candidate attempts "
        f"across `{len(records)}` ledger rows in `{runtime_label}`.",
        "",
    ]
    if best:
        baseline_score = baseline["score"] if baseline else None
        improvement = (
            None
            if baseline_score is None
            else improvement_amount(best["score"], baseline_score, summary["objective"]["mode"])
        )
        lines.append(
            f"Best retained candidate: `{best['name']}` at "
            f"`{summary['objective']['optimization_metric']}={format_score(best['score'])}` "
            f"(baseline `{format_score(baseline_score)}`, objective improvement `{format_delta(improvement)}`)."
        )
    else:
        observed = summary["best_observed"]
        if observed:
            lines.append(
                "No scored result was retained. Best unretained observed result: "
                f"`{observed['name']}` at "
                f"`{summary['objective']['optimization_metric']}={format_score(observed['score'])}` "
                f"with status `{observed['status']}`."
            )
        else:
            lines.append("No finalized scored result was available.")
    progress_lines = (
        [
            "![Auto-FL progress]("
            f"{markdown_artifact_path(summary['artifacts']['progress_plot'], summary['artifacts']['report'])})"
        ]
        if summary["artifacts"]["progress_plot_available"]
        else ["Progress plot unavailable; see the validation and comparability warnings below."]
    )
    selection = summary["selection"]
    outcomes = summary["outcome_summary"]
    lines.extend(
        [
            "",
            f"Termination: `{summary['termination']['reason']}`. "
            f"Final campaign state allowed: `{summary['termination']['state_allowed_final_response']}`.",
            "",
            "## Selected Candidate And Why",
            "",
        ]
    )
    if selection:
        lines.append(
            f"Select `{selection['candidate']}` at `{summary['objective']['optimization_metric']}="
            f"{format_score(selection['score'])}`. {selection['reason']}"
        )
        lines.extend(
            [
                "",
                f"- Objective improvement from baseline: `{format_delta(selection['improvement_from_baseline'])}`",
                f"- Recorded algorithm family: `{selection['algorithm_family']}`",
                f"- Candidate kind: `{selection['candidate_kind']}`",
                f"- Hypothesis: {md_cell(selection['hypothesis'], 300) or 'not recorded'}",
                f"- Retained lineage: `{compact_lineage(selection['lineage']['candidates']) or 'unavailable'}`",
                f"- Cumulative changed files: `{', '.join(selection['lineage']['changed_files']) or 'none recorded'}`",
            ]
        )
    else:
        lines.append("No retained scored result was available for selection.")

    lines.extend(
        [
            "",
            "## What Helped",
            "",
            "The following candidates made strict measured improvements over the retained incumbent and were kept:",
            "",
        ]
    )
    if outcomes["major_helped"]:
        lines.extend(
            [
                "| Candidate | Family | Kind | Score | Objective improvement vs incumbent | Evidence |",
                "| --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for item in outcomes["major_helped"]:
            lines.append(
                f"| `{md_cell(item['name'], 50)}` | `{md_cell(item['algorithm_family'], 40)}` | "
                f"`{md_cell(item['candidate_kind'], 30)}` | {format_score(item['score'])} | "
                f"{format_delta(item['improvement_from_incumbent'])} | {md_cell(item['hypothesis'])} |"
            )
    else:
        lines.append("No candidate produced a strict retained improvement over the baseline.")

    lines.extend(["", "## What Did Not Help", ""])
    if outcomes["not_confirmed"]:
        lines.extend(
            [
                "Representative scored candidates that did not produce a retained improvement:",
                "",
                "| Candidate | Represents | Score | Delta vs incumbent | Evidence |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for item in outcomes["not_confirmed"]:
            lines.append(
                f"| `{md_cell(item['name'], 50)}` | `{md_cell(item['representative_for'], 70)}` | "
                f"{format_score(item['score'])} | {format_delta(item['delta_from_incumbent'])} | "
                f"{md_cell(item['hypothesis'])} |"
            )
    else:
        lines.append("No scored discarded candidates were recorded.")
    if outcomes["failures"]:
        lines.extend(["", "Grouped failed or unscored attempts:", ""])
        for item in outcomes["failures"]:
            candidates = ", ".join(item["candidates"])
            lines.extend(
                wrap_markdown_bullet(
                    f"- **{item['count']} attempt(s):** ",
                    f"{item['reason']} Candidates: `{md_cell(candidates, 240)}`.",
                )
            )

    if outcomes["families"]:
        lines.extend(
            [
                "",
                "Outcome by recorded algorithm family (`unclassified` means the ledger did not declare one):",
                "",
                "| Family | Outcome | Attempts | Kept | Discarded | Crashed |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in outcomes["families"]:
            lines.append(
                f"| `{md_cell(item['algorithm_family'], 50)}` | {item['outcome']} | {item['attempts']} | "
                f"{item['kept']} | {item['discarded']} | {item['crashed']} |"
            )

    lines.extend(
        [
            "",
            "## Campaign Contract",
            "",
            f"- Requested metric: `{summary['objective']['requested_metric']}`",
            f"- Optimization metric: `{summary['objective']['optimization_metric']}`",
            f"- Metric source: `{summary['objective']['metric_source']}`",
            f"- Metric contract source: `{summary['objective']['metric_contract_source']}`",
            f"- Direction: `{summary['objective']['mode']}`",
            f"- Candidate cap: `{summary['candidate_cap']}`",
            f"- Declared fixed budget: `{json.dumps(summary['declared_fixed_budget'], sort_keys=True)}`",
            "",
            "## Progress",
            "",
            *progress_lines,
            "",
            "## Optimization Trajectory",
            "",
            "Major running-best milestones are selected by first, final, and largest measured objective improvements.",
            "",
            "| Row | Candidate | Status | Family | Kind | Score | Delta from previous best | Hypothesis |",
            "| ---: | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for item in summary["milestones"]:
        lines.append(
            f"| {item['row']} | `{md_cell(item['name'], 50)}` | {item['status']} | "
            f"`{md_cell(item['algorithm_family'], 40)}` | `{md_cell(item['candidate_kind'], 30)}` | "
            f"{format_score(item['score'])} | {format_delta(item['delta_from_previous_best'])} | "
            f"{md_cell(item['hypothesis'])} |"
        )
    if not summary["milestones"]:
        lines.append("|  |  |  |  |  |  |  | No scored milestones. |")

    lines.extend(["", "## Best Candidate Provenance", ""])
    if best:
        lines.extend(
            [
                f"- Candidate: `{best['name']}`",
                f"- Status: `{best['status']}`",
                f"- Base lineage: `{compact_lineage(summary['best_lineage']['candidates']) or 'unavailable'}`",
                f"- Cumulative changed files: `{', '.join(summary['best_lineage']['changed_files']) or 'none recorded'}`",
                f"- Candidate manifest (recorded): `{best['candidate_manifest'] or 'not recorded'}`",
                f"- Candidate manifest (resolved): "
                f"`{summary['best_manifest'].get('resolved_path') or 'not available'}`",
                f"- Manifest resolution: `{summary['best_manifest'].get('resolution') or 'not available'}`",
                f"- Manifest available: `{summary['best_manifest'].get('available', False)}`",
                f"- Manifest budget SHA-256: `{summary['best_manifest'].get('budget_sha256') or 'not recorded'}`",
                f"- Patch SHA-256: `{best['patch_sha256'] or 'not recorded'}`",
                f"- Candidate kind: `{best['candidate_kind'] or 'not recorded'}`",
                f"- Algorithm family: `{best['algorithm_family'] or 'not recorded'}`",
                f"- Literature event: `{best['literature_event_id'] or 'not linked'}`",
                f"- Recorded metric: `{best['metric_name'] or summary['objective']['optimization_metric']}`",
                f"- Metric extraction source: `{best['metric_source'] or 'not recorded'}`",
                f"- Metric artifact: `{best['metric_artifact'] or 'not recorded'}`",
                f"- Artifacts: `{best['artifacts'] or 'not recorded'}`",
                "",
                "Executed baseline command:",
                "",
                "```text",
                format_command(baseline["run_command"]) if baseline else "not recorded",
                "```",
                "",
                "Executed best-candidate command:",
                "",
                "```text",
                format_command(best["run_command"]),
                "```",
            ]
        )
    else:
        lines.append("No retained scored candidate provenance was available.")

    lines.extend(["", "## Literature Review Outcomes", ""])
    if summary["literature_reviews"]:
        literature_digest = outcomes["literature"]
        lines.append(f"Outcome counts: `{json.dumps(literature_digest['counts'], sort_keys=True)}`.")
        strongest = literature_digest["strongest_helped"]
        if strongest:
            lines.append(
                f"Strongest literature-linked improvement: `{strongest['event']}` led to "
                f"`{strongest['best_candidate']}` with objective improvement "
                f"`{format_delta(strongest['improvement_from_incumbent'])}` versus its incumbent."
            )
        lines.extend(
            [
                "",
                "| Checkpoint | Sources | Outcome | Candidate evidence | Delta vs incumbent |",
                "| --- | --- | --- | --- | ---: |",
            ]
        )
        for item in summary["literature_reviews"]:
            evidence_items = []
            for candidate in item["candidate_results"]:
                if candidate["status"] == "crash":
                    result = "crash"
                else:
                    result = format_score(candidate["score"])
                evidence_items.append(f"{candidate['name']}={result}")
            evidence = "; ".join(evidence_items) or "no candidate recorded"
            outcome_label = literature_outcome_label(item)
            lines.append(
                f"| `{md_cell(item['event'], 45)}` (`{item['literature_event_id'] or 'not recorded'}`) | "
                f"{md_cell('; '.join(item['sources']) or 'not recorded', 100)} | "
                f"{outcome_label} | `{md_cell(evidence, 130)}` | {format_delta(item['delta_from_incumbent'])} |"
            )
        lines.extend(["", "Checkpoint hypotheses and decisions:", ""])
        for item in summary["literature_reviews"]:
            candidates = ", ".join(item["candidate_attempts"]) or "none recorded"
            outcome_label = literature_outcome_label(item)
            lines.extend(
                wrap_markdown_bullet(
                    f"- **{item['event']} ({outcome_label}):** ",
                    f"{item['hypothesis']} Measured candidates: `{md_cell(candidates, 300)}`.",
                )
            )
        lines.extend(
            [
                "",
                "Source labels above are campaign-recorded identifiers and were not independently verified by the report helper.",
            ]
        )
    else:
        lines.append("No literature checkpoints were recorded in the ledger.")

    lines.extend(
        [
            "",
            "## Runtime And Reliability",
            "",
            f"- Total recorded runtime: `{runtime_label}`",
            f"- Status counts: `{json.dumps(counts, sort_keys=True)}`",
            f"- Scored comparable runs: `{summary['scored_runs']}`",
            f"- Crashes: `{counts.get('crash', 0)}`",
            f"- Abandoned candidates: "
            f"`{summary['abandoned_candidates'] if summary['abandoned_candidates'] is not None else 'not recorded'}`",
        ]
    )

    lines.extend(["", "## Validation And Comparability Notes", ""])
    if summary["warnings"]:
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    else:
        lines.append("- No deterministic comparability warning was detected from the available artifacts.")
    if summary["best_command_changes"]:
        lines.extend(["", "Executed argument changes from baseline to best:", ""])
        for key, value in summary["best_command_changes"].items():
            lines.append(f"- `{key}`: `{value['baseline']}` -> `{value['best']}`")

    lines.extend(["", "## Agent And Tooling Context", ""])
    if summary["agent_context"]:
        lines.append("```json")
        lines.append(json.dumps(summary["agent_context"], indent=2, sort_keys=True))
        lines.append("```")
    else:
        lines.append("Agent model, reasoning effort, and cost were not supplied to the report generator.")

    lines.extend(
        [
            "",
            "## Reproduction Recommendations",
            "",
            "1. Re-run the baseline and selected candidate from the exact commands above in the same NVFLARE environment.",
            "2. Confirm the candidate on additional seeds or sites before treating a single-run improvement as robust.",
            "3. When a test-like metric guided selection, perform one final evaluation on an untouched holdout.",
            "4. Preserve the candidate manifest, patch hash, ledger, campaign config, and downloaded NVFLARE artifacts.",
            "",
            "## Report Artifacts",
            "",
            f"- Auto-FL config: `{summary['artifacts']['autofl_yaml']}`",
            f"- Results ledger: `{summary['artifacts']['results']}`",
            f"- Campaign state: `{summary['artifacts']['campaign_state']}`",
            f"- Progress plot: `{summary['artifacts']['progress_plot']}`",
            f"- Progress plot available: `{summary['artifacts']['progress_plot_available']}`",
            f"- Machine-readable summary: `{summary['artifacts']['summary_json']}`",
            f"- Markdown report: `{summary['artifacts']['report']}`",
            "",
            "## Technical Appendix",
            "",
            f"Generated at `{summary['generated_at']}` with schema `{summary['schema_version']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def record_payload(record: Optional[RunRecord]) -> Optional[Dict[str, Any]]:
    return asdict(record) if record else None


def generate_locked(args: argparse.Namespace, root: Path) -> Dict[str, Any]:
    results_path = resolve_path(root, args.results)
    state_path = resolve_path(root, args.state)
    config_path = resolve_path(root, args.autofl_yaml)
    progress_path = resolve_path(root, args.progress)
    report_path = resolve_path(root, args.output)
    summary_path = resolve_path(root, args.summary_json)
    agent_context_path = resolve_path(root, args.agent_context) if args.agent_context else None
    plotter_path = resolve_path(root, args.plotter).resolve() if args.plotter else default_plotter_path()

    records = load_results(results_path)
    state, termination_reason, warnings = verify_stopped(state_path, args.confirm_interrupted)
    config = load_config(config_path)
    config, config_warnings = normalize_contract_sections(config)
    warnings.extend(config_warnings)
    protected_sources, allowed_create_patterns = campaign_source_contract(root, config)
    protected_paths = {
        "results ledger": results_path,
        "campaign state": state_path,
        "Auto-FL config": config_path,
        "campaign lock": root / CAMPAIGN_LOCK_PATH,
        "progress plotter": plotter_path,
    }
    protected_paths.update(protected_sources)
    if agent_context_path:
        protected_paths["agent context"] = agent_context_path
    for index, manifest_path in enumerate(candidate_manifest_paths(root, records), start=1):
        protected_paths[f"candidate manifest {index}"] = manifest_path
    pending_manifest = state.get("pending_candidate_manifest")
    if isinstance(pending_manifest, str) and pending_manifest:
        protected_paths["state pending-candidate manifest"] = resolve_path(root, pending_manifest)
    outputs = {
        "progress": progress_path,
        "output": report_path,
        "summary-json": summary_path,
    }
    validate_output_paths(outputs, protected_paths)
    validate_output_create_patterns(outputs, root, allowed_create_patterns)
    verify_no_pending_candidates(root, state, records)
    mode = metric_mode(config, state)
    metric, requested_metric, metric_source, metric_contract_source = metric_contract(config, state, args)
    baseline = select_baseline(records)
    best = select_best(records, retained_only=True, mode=mode)
    observed_best = select_best(records, mode=mode)
    if best is None and observed_best is not None:
        warnings.append(
            f"Best observed row {observed_best.name} was not retained; no scored baseline or kept candidate is available."
        )
    elif best and observed_best and best.name != observed_best.name:
        warnings.append(
            f"Best observed row {observed_best.name} was not retained; the report identifies retained candidate {best.name}."
        )

    plot_warning = refresh_plot(
        results_path,
        progress_path,
        metric,
        plotter_path,
        mode,
    )
    if plot_warning:
        warnings.append(plot_warning)
    progress_plot_available = is_png(progress_path)
    if not progress_plot_available:
        warnings.append(
            f"Progress plot at {progress_path} is missing or is not a valid PNG; the report was generated without "
            "embedding it."
        )

    comparison_warnings, changes = comparability_warnings(config, records, baseline, best, metric, metric_source)
    warnings.extend(comparison_warnings)
    status_counts = dict(sorted(Counter(record.status or "unknown" for record in records).items()))
    scored = scored_records(records)
    candidate_attempts = len(
        [record for record in records if record.status in ATTEMPT_STATUSES and not is_baseline(record)]
    )
    accounting, accounting_warnings = state_accounting(
        state,
        candidate_attempts,
        baseline,
        best,
        results_path,
        root,
        mode,
    )
    warnings.extend(accounting_warnings)
    environment = config.get("environment") if isinstance(config.get("environment"), dict) else {}
    budget = config.get("budget") if isinstance(config.get("budget"), dict) else {}
    fixed_budget = budget.get("fixed_training_budget", {})
    cap = state.get("candidate_cap")
    cap_label = "uncapped" if cap in {None, "", 0} else cap
    lineage = candidate_lineage(best, records)
    reviews = literature_outcomes(
        records,
        exploration_batch=state.get("exploration_batch"),
        termination_reason=termination_reason,
        mode=mode,
    )
    warnings.extend(
        literature_completion_warnings(
            reviews,
            exploration_batch=state.get("exploration_batch"),
            termination_reason=termination_reason,
        )
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "termination": {
            "reason": termination_reason,
            "state_allowed_final_response": state.get("final_response_allowed") is True,
            "user_confirmed_interruption": bool(args.confirm_interrupted),
        },
        "objective": {
            "requested_metric": requested_metric,
            "optimization_metric": metric,
            "metric_source": metric_source,
            "metric_contract_source": metric_contract_source,
            "mode": mode,
            "mode_contract_source": str(config.get("objective", {}).get("mode_contract_source") or "not recorded"),
        },
        "environment": str(environment.get("requested") or state.get("environment") or "not declared"),
        "candidate_cap": cap_label,
        "candidate_cap_source": state.get("candidate_cap_source") or "not recorded",
        "abandoned_candidates": accounting["abandoned_candidates"],
        "declared_fixed_budget": fixed_budget if isinstance(fixed_budget, dict) else {},
        "candidate_attempts": candidate_attempts,
        "state_accounting": accounting,
        "scored_runs": len(scored),
        "runtime_seconds": sum(record.runtime_seconds for record in records),
        "status_counts": status_counts,
        "baseline": record_payload(baseline),
        "best": record_payload(best),
        "best_observed": record_payload(observed_best),
        "best_lineage": lineage,
        "best_manifest": manifest_summary(best, root),
        "best_command_changes": changes,
        "milestones": running_best_milestones(records, args.max_milestones, mode),
        "selection": selection_summary(best, baseline, lineage, mode),
        "outcome_summary": outcome_summary(
            records,
            reviews,
            args.max_milestones,
            args.max_non_improvements,
            mode,
        ),
        "literature_reviews": reviews,
        "warnings": warnings,
        "agent_context": read_agent_context(agent_context_path, args),
        "artifacts": {
            "autofl_yaml": str(config_path.resolve()),
            "results": str(results_path.resolve()),
            "campaign_state": str(state_path.resolve()),
            "progress_plot": str(progress_path.resolve()),
            "progress_plot_available": progress_plot_available,
            "summary_json": str(summary_path.resolve()),
            "report": str(report_path.resolve()),
        },
    }
    atomic_write_text(report_path, report_markdown(summary, records))
    atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def generate(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.campaign_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Auto-FL campaign directory not found: {root}")
    results_path = resolve_path(root, args.results)
    if not results_path.is_file():
        raise ValueError(f"Auto-FL ledger not found: {results_path}")
    with locked_campaign_workspace(root, "report"):
        return generate_locked(args, root)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = generate(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", "artifacts": summary["artifacts"], "best": summary["best"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
