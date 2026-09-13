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

"""Manage agent-authored Auto-FL candidates for an existing NVFlare job.py.

Usage:
    ``python run_job_campaign.py ACTION JOB [options]``
Arguments:
    ``ACTION`` selects the lifecycle operation; ``JOB`` is the existing
    NVFLARE ``job.py``; remaining options are action-specific.
Output:
    Writes campaign configuration, state, ledger, candidate, plot, and report
    artifacts and prints a machine-readable JSON action envelope.
Exit codes:
    0 for success, 1 when an execution completes without a score, 2 for a
    rejected contract or invalid request, and 75 when simulation needs the
    existing human-approved execution scope.

The coding agent owns hypotheses and source edits. This helper snapshots the
current best source, validates and evaluates candidate manifests, restores
discarded candidates, and records reproducible campaign state and artifacts.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import difflib
import hashlib
import importlib.util
import io
import json
import os
import queue
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows uses the exclusive-file fallback
    fcntl = None

try:
    import yaml
except ImportError:  # pragma: no cover - NVFlare installs PyYAML
    yaml = None


RESULT_FIELDS = [
    "status",
    "name",
    "score",
    "runtime_seconds",
    "changed_files",
    "diff_summary",
    "run_command",
    "artifacts",
    "failure_reason",
    "candidate_manifest",
    "base_candidate",
    "patch_sha256",
    "metric_name",
    "metric_source",
    "metric_artifact",
    "candidate_kind",
    "algorithm_family",
    "literature_event_id",
]

# Marker shared by every server-side global-model entry in cross_val_results.json: the legacy
# CrossSiteModelEval workflow prefixes SRV_ (SRV_FL_global_model, SRV_best_FL_global_model.pt)
# while the ModelController CrossSiteEval records the raw checkpoint name (FL_global_model.pt).
SERVER_GLOBAL_MODEL_KEY_MARKER = "FL_global_model"
# Sub-marker for best-checkpoint global-model entries (SRV_best_FL_global_model.pt,
# best_FL_global_model.pt) as opposed to final-checkpoint entries.
SERVER_BEST_GLOBAL_MODEL_KEY_MARKER = "best_FL_global_model"

CANDIDATE_MANIFEST_SCHEMA_VERSION = "nvflare.autofl.candidate.v1"
CANDIDATE_MANIFEST_STATUSES = {"prepared", "ready_for_external_execution", "keep", "discard", "crash", "abandoned"}
CAMPAIGN_METADATA_SCHEMA_VERSION = "nvflare.autofl.campaign.v1"
METRIC_DIRECTION_CONTRACT_VERSION = 1
CAMPAIGN_METADATA_PATH = ".nvflare/autofl/campaign.json"
CANDIDATE_ROOT = ".nvflare/autofl/candidates"
BEST_SNAPSHOT_ROOT = ".nvflare/autofl/snapshots/best"
ALLOWED_CREATE_PATTERNS = ["**/*.py"]
RESERVED_CANDIDATE_PATH_PARTS = {
    ".git",
    ".nvflare",
    ".tox",
    ".venv",
    "__pycache__",
    "autofl_runs",
    "node_modules",
    "site-packages",
    "venv",
}

INFRASTRUCTURE_RETRY = "infrastructure_retry"
SIMULATION_APPROVAL_ACTION = "await_simulation_runner_approval"
ACCOUNTING_INSTRUCTION = (
    "Run every candidate training, parameter update, or metric-based screening through this runner. "
    "Real candidate crashes and crash replays count; expanding the candidate cap requires explicit user approval."
)
SIMULATOR_STALL_EXIT_CODE = 125
SIMULATOR_STALL_PATTERNS = (
    "Failed to create connection to the child process in SimulatorClientRunner",
    "SimulatorClientRunner - ERROR - run_client_thread error",
)
SIMULATOR_STALL_LOG_LIMIT = 65536
SIMULATOR_AGGREGATION_RE = re.compile(r"Aggregated\s+(\d+)/(\d+)\s+results")
SIMULATOR_PROGRESS_PATTERNS = (
    "Round ",
    "Aggregated ",
    "Beginning model validation",
    "Saved validation result",
    "Finished FedAvg",
    "Finished ScatterAndGather",
    "cross_site_val",
    "round=",
    "epoch",
    "returning result",
    "sent result",
    "task completed",
)
SIMULATOR_PROGRESS_LOG_LIMIT = 131072
DEFAULT_SIMULATOR_NO_PROGRESS_TIMEOUT = 240
DEFAULT_JOB_HELP_TIMEOUT = 30
MAX_CAPTURED_PROCESS_OUTPUT = 1024 * 1024
SIMULATOR_WORKSPACE_ROOT_ENV_VAR = "NVFLARE_SIMULATOR_WORKSPACE_ROOT"
TRIAL_PROCESS_TOKEN_ENV_VAR = "NVFLARE_AUTOFL_TRIAL_TOKEN"
SIMULATOR_WORKSPACE_OVERRIDE_MIN_NVFLARE_VERSION = "2.9.0"
DEFAULT_WORKSPACE_OVERRIDE_PROBE_TIMEOUT = 30
CAMPAIGN_LOCK_PATH = ".nvflare/autofl/campaign.lock"
SIMULATOR_ENV_PASSTHROUGH_CONFIG_KEY = "simulator_env_passthrough"
SIMULATOR_ENV_ALLOWLIST = (
    "PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "PYENV_VERSION",
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "CURL_CA_BUNDLE",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
)
ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

FIXED_BUDGET_TO_CLI = {
    "num_clients": "n_clients",
    "min_clients": "min_clients",
    "num_rounds": "num_rounds",
}

COMPARISON_BUDGET_TO_CLI = {
    "n_clients": "n_clients",
    "num_rounds": "num_rounds",
    "aggregation_epochs": "aggregation_epochs",
    "local_train_steps": "local_train_steps",
    "batch_size": "batch_size",
    "eval_batch_size": "eval_batch_size",
    "alpha": "alpha",
    "seed": "seed",
    "model_arch": "model_arch",
    "max_model_params": "max_model_params",
    "aggregator": "aggregator",
    "final_eval_clients": "final_eval_clients",
}


@dataclass
class RunRecord:
    status: str
    name: str
    score: Optional[float]
    runtime_seconds: float
    changed_files: str
    diff_summary: str
    run_command: str
    artifacts: str
    failure_reason: str = ""
    candidate_manifest: str = ""
    base_candidate: str = ""
    patch_sha256: str = ""
    metric_name: str = ""
    metric_source: str = ""
    metric_artifact: str = ""
    candidate_kind: str = ""
    algorithm_family: str = ""
    literature_event_id: str = ""


@dataclass(frozen=True)
class MetricEvidence:
    score: float
    metric_name: str
    source: str
    artifact: str


@dataclass
class JobRun:
    name: str
    args: List[str]
    description: str
    status: str = "candidate"
    score: Optional[float] = None
    runtime_seconds: float = 0.0
    artifacts: str = ""
    failure_reason: str = ""
    command: List[str] = field(default_factory=list)


def env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value


def env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    guard = load_campaign_guard()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "action",
        choices=["initialize", "prepare", "evaluate", "abandon", "suggest", "record", "status"],
        help="skill-internal campaign lifecycle action",
    )
    parser.add_argument("job", help="NVFlare job.py to optimize")
    parser.add_argument("--metric", help="optimization metric; omit to use job.py key_metric")
    parser.add_argument("--env", dest="target_env", choices=["sim", "poc", "prod"], default="sim")
    cap_group = parser.add_mutually_exclusive_group()
    cap_group.add_argument(
        "--max-candidates",
        type=int,
        help=(
            "optional cap on comparable candidate attempts (evaluated candidates: keep/discard/crash), "
            "counted after and excluding the baseline; omit to run an uncapped continuous campaign "
            "until interrupted or blocked"
        ),
    )
    cap_group.add_argument("--uncapped", action="store_true", help="remove a previously configured candidate cap")
    parser.add_argument("--autofl-yaml", default="autofl.yaml")
    parser.add_argument("--results", default="results.tsv")
    parser.add_argument("--state", default=".nvflare/autofl/campaign_state.json")
    parser.add_argument("--progress", default="progress.png")
    parser.add_argument("--report", default="autofl_report.md")
    parser.add_argument("--output-root", default="autofl_runs")
    parser.add_argument(
        "--plateau-threshold",
        type=int,
        default=env_int("AUTOFL_PLATEAU_THRESHOLD", guard.DEFAULT_PLATEAU_THRESHOLD),
        help=(
            "scored comparable candidate attempts after the last material improvement or literature event "
            "before campaign state requests run_literature_loop"
        ),
    )
    parser.add_argument(
        "--plateau-min-delta",
        type=float,
        default=env_float("AUTOFL_PLATEAU_MIN_DELTA", guard.DEFAULT_MIN_DELTA),
        help="minimum metric delta required to reset the plateau clock",
    )
    parser.add_argument(
        "--hard-crash-threshold",
        type=int,
        default=env_int("AUTOFL_HARD_CRASH_THRESHOLD", guard.DEFAULT_HARD_CRASH_THRESHOLD),
        help="stop after this many consecutive real candidate crashes; set 0 to disable",
    )
    parser.add_argument(
        "--exploration-batch-size",
        type=int,
        default=env_int("AUTOFL_EXPLORATION_BATCH_SIZE", guard.DEFAULT_EXPLORATION_BATCH_SIZE),
        help=(
            "scored source-backed candidates required per literature event before normal candidate "
            "flow resumes; set 0 to disable"
        ),
    )
    parser.add_argument(
        "--family-repeat-limit",
        type=int,
        default=env_int("AUTOFL_FAMILY_REPEAT_LIMIT", guard.DEFAULT_FAMILY_REPEAT_LIMIT),
        help=(
            "consecutive same-family argument-only scored attempts before campaign state requires "
            "diversification; set 0 to disable"
        ),
    )
    parser.add_argument(
        "--stop-file",
        action="append",
        help="manual stop-file path; defaults to STOP_AUTOFL and .nvflare/autofl/STOP under the job directory",
    )
    parser.add_argument("--base-args", default="", help="extra job args applied to baseline and all candidates")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--simulator-no-progress-timeout",
        type=int,
        default=env_int("AUTOFL_SIMULATOR_NO_PROGRESS_TIMEOUT_SECONDS", DEFAULT_SIMULATOR_NO_PROGRESS_TIMEOUT),
        help=(
            "candidate-level simulator no-progress timeout in seconds; set 0 to disable. "
            "This is separate from the full run timeout and only applies after progress markers appear."
        ),
    )
    parser.add_argument("--python", default=os.environ.get("PYTHON") or sys.executable)
    parser.add_argument("--name", help="candidate name for prepare")
    parser.add_argument("--hypothesis", help="candidate hypothesis for prepare")
    parser.add_argument("--family", help="algorithm family slug for prepare or record --literature")
    parser.add_argument("--literature-event", help="literature event id this candidate develops (prepare)")
    parser.add_argument("--manifest", help="candidate_manifest.json path")
    parser.add_argument("--run-args", default="", help="candidate-only job.py arguments")
    parser.add_argument("--score", type=float, help="externally measured metric for record")
    parser.add_argument("--artifacts", dest="external_artifacts", help="external POC/production artifacts")
    parser.add_argument("--job-id", help="standard NVFlare job ID for an external result")
    parser.add_argument("--failure-reason", default="", help="external execution failure")
    parser.add_argument("--baseline", action="store_true", help="record an externally executed baseline")
    parser.add_argument("--literature", action="store_true", help="record a literature-review checkpoint")
    parser.add_argument(
        "--confirm-user-approved-cap-change",
        action="store_true",
        help="confirm that the user explicitly approved increasing the candidate cap or making it uncapped",
    )
    parser.add_argument("--limit", type=int, default=10, help="maximum fallback suggestions")
    args = parser.parse_args(argv)
    tokens = list(argv) if argv is not None else sys.argv[1:]
    explicit_settings = set()
    for action in parser._actions:
        for option in action.option_strings:
            if any(token == option or token.startswith(f"{option}=") for token in tokens):
                explicit_settings.add(action.dest)
                break
    args._explicit_settings = explicit_settings
    if args.uncapped:
        args.max_candidates = None
        args._explicit_settings.add("max_candidates")
    # Objective direction has no runner CLI surface; initialize imports it from job.py,
    # while later actions restore the persisted campaign contract.
    args.mode = "max"
    return args


def process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class TrialProcessCleanupError(RuntimeError):
    pass


def pidfd_functions():
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if not callable(pidfd_open) or not callable(pidfd_send_signal):
        raise TrialProcessCleanupError(
            "Linux pidfd_open and pidfd_send_signal support is required for race-safe trial cleanup"
        )
    return pidfd_open, pidfd_send_signal


def ensure_trial_process_pidfd_support() -> None:
    if not sys.platform.startswith("linux"):
        return
    pidfd_open, pidfd_send_signal = pidfd_functions()
    try:
        pidfd = pidfd_open(os.getpid(), 0)
    except OSError as e:
        raise TrialProcessCleanupError(f"cannot open the Linux pidfd required for race-safe trial cleanup: {e}") from e
    try:
        while True:
            try:
                pidfd_send_signal(pidfd, 0)
                break
            except InterruptedError:
                continue
            except OSError as e:
                raise TrialProcessCleanupError(
                    f"cannot signal through the Linux pidfd required for race-safe trial cleanup: {e}"
                ) from e
    finally:
        os.close(pidfd)


def trial_process_ids(trial_token: Optional[str]) -> List[int]:
    """Return Linux processes that inherited this runner-owned trial token."""
    if not trial_token or not sys.platform.startswith("linux"):
        return []
    marker = f"{TRIAL_PROCESS_TOKEN_ENV_VAR}={trial_token}".encode("utf-8")
    process_ids = []
    for environ_path in Path("/proc").glob("[0-9]*/environ"):
        try:
            process_id = int(environ_path.parent.name)
            environ = environ_path.read_bytes().split(b"\0")
        except (OSError, ValueError):
            continue
        if process_id != os.getpid() and marker in environ:
            process_ids.append(process_id)
    return sorted(process_ids)


def pidfd_process_id(pidfd: int) -> Optional[int]:
    try:
        lines = Path(f"/proc/self/fdinfo/{pidfd}").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("Pid:"):
            try:
                return int(line.partition(":")[2].strip())
            except ValueError:
                return None
    return None


def process_has_trial_token(process_id: int, marker: bytes) -> bool:
    try:
        return marker in Path(f"/proc/{process_id}/environ").read_bytes().split(b"\0")
    except OSError:
        return False


def open_trial_process_pidfd(process_id: int, marker: bytes) -> Optional[int]:
    pidfd_open, _ = pidfd_functions()
    while True:
        try:
            pidfd = pidfd_open(process_id, 0)
            break
        except ProcessLookupError:
            return None
        except InterruptedError:
            continue
        except OSError as e:
            raise TrialProcessCleanupError(f"cannot open pidfd for trial process {process_id}: {e}") from e
    if (
        pidfd_process_id(pidfd) != process_id
        or not process_has_trial_token(process_id, marker)
        or pidfd_process_id(pidfd) != process_id
    ):
        os.close(pidfd)
        return None
    return pidfd


def signal_trial_processes(trial_token: Optional[str], sig: int) -> None:
    if not trial_token:
        return
    process_ids = trial_process_ids(trial_token)
    if not process_ids:
        return
    _, pidfd_send_signal = pidfd_functions()
    marker = f"{TRIAL_PROCESS_TOKEN_ENV_VAR}={trial_token}".encode("utf-8")
    errors = []
    for process_id in process_ids:
        try:
            pidfd = open_trial_process_pidfd(process_id, marker)
        except TrialProcessCleanupError as e:
            errors.append(str(e))
            continue
        if pidfd is None:
            continue
        try:
            while True:
                try:
                    pidfd_send_signal(pidfd, sig)
                    break
                except ProcessLookupError:
                    break
                except InterruptedError:
                    continue
                except OSError as e:
                    errors.append(f"cannot signal trial process {process_id} through pidfd: {e}")
                    break
        finally:
            os.close(pidfd)
    if errors:
        raise TrialProcessCleanupError("; ".join(errors))


def wait_for_process_tree(
    process: subprocess.Popen, process_group_id: Optional[int], timeout: float, trial_token: Optional[str] = None
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        leader_exited = process.poll() is not None
        group_exited = process_group_id is None or not process_group_exists(process_group_id)
        if leader_exited and group_exited and not trial_process_ids(trial_token):
            return True
        time.sleep(0.05)
    return (
        process.poll() is not None
        and (process_group_id is None or not process_group_exists(process_group_id))
        and not trial_process_ids(trial_token)
    )


def terminate_process(
    process: subprocess.Popen, process_group_id: Optional[int] = None, trial_token: Optional[str] = None
) -> None:
    cleanup_errors = []
    if os.name != "nt" and process_group_id is not None:
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            if process.poll() is None:
                process.terminate()
        except Exception:
            if process.poll() is None:
                process.terminate()
    elif process.poll() is None:
        process.terminate()
    try:
        signal_trial_processes(trial_token, signal.SIGTERM)
    except TrialProcessCleanupError as e:
        cleanup_errors.append(str(e))

    if wait_for_process_tree(process, process_group_id, timeout=10, trial_token=trial_token):
        return

    if os.name != "nt" and process_group_id is not None:
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            if process.poll() is None:
                process.kill()
    elif process.poll() is None:
        process.kill()
    try:
        signal_trial_processes(trial_token, signal.SIGKILL)
    except TrialProcessCleanupError as e:
        cleanup_errors.append(str(e))
    tree_exited = wait_for_process_tree(process, process_group_id, timeout=10, trial_token=trial_token)
    remaining_trial_processes = trial_process_ids(trial_token)
    if remaining_trial_processes:
        detail = f"; pidfd errors: {'; '.join(cleanup_errors)}" if cleanup_errors else ""
        raise TrialProcessCleanupError(
            f"trial cleanup could not terminate detached processes {remaining_trial_processes}{detail}"
        )
    if not tree_exited and cleanup_errors:
        raise TrialProcessCleanupError(f"trial cleanup did not complete: {'; '.join(cleanup_errors)}")


def append_output_tail(current: str, value: str) -> str:
    combined = (current + value).encode("utf-8", errors="replace")
    if len(combined) <= MAX_CAPTURED_PROCESS_OUTPUT:
        return current + value
    return combined[-MAX_CAPTURED_PROCESS_OUTPUT:].decode("utf-8", errors="ignore")


def recent_text(path: Path, limit: int = SIMULATOR_STALL_LOG_LIMIT) -> str:
    try:
        with path.open("rb") as f:
            try:
                f.seek(-limit, os.SEEK_END)
            except OSError:
                f.seek(0)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def detect_nvflare_simulator_stall(sim_root: Path) -> Optional[str]:
    if not sim_root.exists():
        return None

    log_paths = [
        sim_root / "server" / "log_fl.txt",
        sim_root / "server" / "log.txt",
        sim_root / "server" / "error_log.txt",
    ]
    for path in log_paths:
        text = recent_text(path)
        if not text:
            continue
        for pattern in SIMULATOR_STALL_PATTERNS:
            if pattern in text:
                return f"{pattern} in {path}"
    return None


def simulator_stall_message(simulator_stall_roots: Sequence[Path]) -> Optional[str]:
    for root in simulator_stall_roots:
        message = detect_nvflare_simulator_stall(root)
        if message:
            return message
    return None


def simulator_progress_signature(sim_root: Path) -> str:
    if not sim_root.exists():
        return ""

    log_paths = [
        sim_root / "server" / "log_fl.txt",
        sim_root / "server" / "log.txt",
        *sorted(sim_root.glob("site-*/log_fl.txt")),
        *sorted(sim_root.glob("site-*/log.txt")),
    ]
    markers: List[str] = []
    for path in log_paths:
        text = recent_text(path, limit=SIMULATOR_PROGRESS_LOG_LIMIT)
        if not text:
            continue
        for line in text.splitlines():
            line_lower = line.lower()
            if any(pattern.lower() in line_lower for pattern in SIMULATOR_PROGRESS_PATTERNS):
                markers.append(f"{path.relative_to(sim_root)}:{line}")
    return "\n".join(markers[-200:])


def simulator_progress_signature_for_roots(simulator_stall_roots: Sequence[Path]) -> str:
    markers = []
    for root in simulator_stall_roots:
        signature = simulator_progress_signature(root)
        if signature:
            markers.append(f"{root}:\n{signature}")
    return "\n".join(markers)


def simulator_partial_aggregation_signature(sim_root: Path) -> str:
    signatures = []
    for relative_log_path in ("server/log_fl.txt", "server/simulate_job/log_fl.txt"):
        path = sim_root / relative_log_path
        text = recent_text(path, SIMULATOR_PROGRESS_LOG_LIMIT)
        if not text:
            continue
        for line in reversed(text.splitlines()):
            match = SIMULATOR_AGGREGATION_RE.search(line)
            if not match:
                continue
            received = int(match.group(1))
            expected = int(match.group(2))
            if received < expected:
                signatures.append(f"{path.relative_to(sim_root)}:{line}")
            break
    return "\n".join(signatures)


def simulator_partial_aggregation_signature_for_roots(simulator_stall_roots: Sequence[Path]) -> str:
    markers = []
    for root in simulator_stall_roots:
        signature = simulator_partial_aggregation_signature(root)
        if signature:
            markers.append(f"{root}:\n{signature}")
    return "\n".join(markers)


def resolve_python_interpreter(python: str, env: Mapping[str, str], *, cwd: Path) -> str:
    """Resolve the configured Python interpreter without dereferencing virtual-environment symlinks."""
    if not isinstance(python, str) or not python or "\x00" in python:
        raise ValueError("python interpreter must be a non-empty string without NUL characters")

    absolute_cwd = os.path.abspath(os.fspath(cwd))
    expanded = os.path.expanduser(python)
    if os.path.dirname(expanded) or os.path.isabs(expanded):
        resolved = os.path.abspath(expanded if os.path.isabs(expanded) else os.path.join(absolute_cwd, expanded))
    else:
        search_entries = []
        for entry in env.get("PATH", os.defpath).split(os.pathsep):
            search_entries.append(entry if os.path.isabs(entry) else os.path.join(absolute_cwd, entry or os.curdir))
        resolved = shutil.which(expanded, path=os.pathsep.join(search_entries)) or ""
        if resolved:
            resolved = os.path.abspath(resolved)
    if not resolved or not Path(resolved).is_file():
        raise ValueError(f"python interpreter does not exist: {python}")
    if os.name != "nt" and not os.access(resolved, os.X_OK):
        raise ValueError(f"python interpreter is not executable: {python}")
    return resolved


def python_command(python: str, arguments: Sequence[str], env: Mapping[str, str], *, cwd: Path) -> List[str]:
    """Build a validated Python argv list; argument values are never interpreted by a shell."""
    if not arguments:
        raise ValueError("python command requires at least one argument")
    validated_arguments = []
    for argument in arguments:
        if not isinstance(argument, str):
            raise ValueError("python command arguments must be strings")
        if "\x00" in argument:
            raise ValueError("python command arguments must not contain NUL characters")
        validated_arguments.append(argument)
    return [resolve_python_interpreter(python, env, cwd=cwd), *validated_arguments]


def run_python(
    python: str,
    arguments: Sequence[str],
    *,
    cwd: Path,
    timeout: int,
    log_path: Path,
    env: Mapping[str, str],
    simulator_stall_roots: Sequence[Path] = (),
    stall_check_interval: float = 5.0,
    simulator_no_progress_timeout: int = DEFAULT_SIMULATOR_NO_PROGRESS_TIMEOUT,
) -> Tuple[int, str, float, List[str]]:
    """Run a validated Python child and return its exit status, output tail, runtime, and exact argv."""
    argv = python_command(python, arguments, env, cwd=cwd)
    ensure_trial_process_pidfd_support()
    started = time.monotonic()
    next_stall_check = started
    last_progress_check = started
    last_progress_seen = started
    last_progress_signature = ""
    last_partial_aggregation_seen = started
    last_partial_aggregation_signature = ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_tail = ""
    run_env = dict(env)
    trial_token = uuid.uuid4().hex
    run_env[TRIAL_PROCESS_TOKEN_ENV_VAR] = trial_token
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
            start_new_session=os.name != "nt",
            env=run_env,
            shell=False,
        )
        process_group_id = process.pid if os.name != "nt" else None
        assert process.stdout is not None
        output_queue: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=16)
        stop_reader = threading.Event()

        def read_output() -> None:
            try:
                while True:
                    try:
                        chunk = process.stdout.read(65536)
                    except (OSError, ValueError):
                        if stop_reader.is_set():
                            break
                        raise
                    if not chunk:
                        break
                    while not stop_reader.is_set():
                        try:
                            output_queue.put(chunk, timeout=0.2)
                            break
                        except queue.Full:
                            continue
                    if stop_reader.is_set():
                        break
            finally:
                while not stop_reader.is_set():
                    try:
                        output_queue.put(None, timeout=0.2)
                        break
                    except queue.Full:
                        continue

        reader = threading.Thread(target=read_output, name="autofl-process-output", daemon=True)
        reader.start()
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        output_finished = False
        timed_out = False
        stall_message = ""
        try:
            while not output_finished or process.poll() is None:
                now = time.monotonic()
                if not timed_out and timeout and now - started > timeout:
                    timed_out = True
                    terminate_process(process, process_group_id, trial_token)
                if not timed_out and not stall_message and simulator_stall_roots and now >= next_stall_check:
                    stall_message = simulator_stall_message(simulator_stall_roots) or ""
                    next_stall_check = now + stall_check_interval
                    if stall_message:
                        terminate_process(process, process_group_id, trial_token)
                    elif simulator_no_progress_timeout:
                        partial_aggregation_signature = simulator_partial_aggregation_signature_for_roots(
                            simulator_stall_roots
                        )
                        if (
                            partial_aggregation_signature
                            and partial_aggregation_signature != last_partial_aggregation_signature
                        ):
                            last_partial_aggregation_signature = partial_aggregation_signature
                            last_partial_aggregation_seen = now
                        elif (
                            last_partial_aggregation_signature
                            and now - last_partial_aggregation_seen > simulator_no_progress_timeout
                        ):
                            stall_message = (
                                "partial simulator aggregation made no server-side progress for "
                                f"{int(now - last_partial_aggregation_seen)}s: {last_partial_aggregation_signature}"
                            )
                            terminate_process(process, process_group_id, trial_token)
                        progress_signature = simulator_progress_signature_for_roots(simulator_stall_roots)
                        if stall_message:
                            pass
                        elif progress_signature and progress_signature != last_progress_signature:
                            last_progress_signature = progress_signature
                            last_progress_seen = now
                        elif (
                            last_progress_signature
                            and now - last_progress_seen > simulator_no_progress_timeout
                            and now - last_progress_check >= stall_check_interval
                        ):
                            stall_message = (
                                f"no simulator progress markers changed for {int(now - last_progress_seen)}s "
                                f"across {', '.join(str(root) for root in simulator_stall_roots)}"
                            )
                            terminate_process(process, process_group_id, trial_token)
                        last_progress_check = now
                try:
                    raw_chunk = output_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if raw_chunk is None:
                    output_finished = True
                    continue
                chunk = decoder.decode(raw_chunk)
                if chunk:
                    output_tail = append_output_tail(output_tail, chunk)
                    log_file.write(chunk)
                    log_file.flush()
            remainder = decoder.decode(b"", final=True)
            if remainder:
                output_tail = append_output_tail(output_tail, remainder)
                log_file.write(remainder)
                log_file.flush()
            if timed_out:
                timeout_msg = f"\nTIMEOUT after {timeout}s\n"
                output_tail = append_output_tail(output_tail, timeout_msg)
                log_file.write(timeout_msg)
                log_file.flush()
                return 124, output_tail, time.monotonic() - started, argv
            if stall_message:
                stall_text = f"\nSIMULATOR_STALL: {stall_message}\n"
                output_tail = append_output_tail(output_tail, stall_text)
                log_file.write(stall_text)
                log_file.flush()
                return SIMULATOR_STALL_EXIT_CODE, output_tail, time.monotonic() - started, argv
            return process.returncode or 0, output_tail, time.monotonic() - started, argv
        finally:
            terminate_process(process, process_group_id, trial_token)
            reader_deadline = time.monotonic() + 10
            while (reader.is_alive() or not output_queue.empty()) and time.monotonic() < reader_deadline:
                try:
                    raw_chunk = output_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if raw_chunk is None:
                    break
                chunk = decoder.decode(raw_chunk)
                if chunk:
                    output_tail = append_output_tail(output_tail, chunk)
                    log_file.write(chunk)
            log_file.flush()
            stop_reader.set()
            process.stdout.close()
            reader.join(timeout=1)


def read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML files")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"invalid YAML in {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"invalid YAML in {path}: expected a mapping")
    return data


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write autofl.yaml")
    atomic_write_bytes(path, yaml.safe_dump(data, sort_keys=False).encode("utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    atomic_write_bytes(path, (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def capture_file_versions(paths: Iterable[Path]) -> Dict[Path, Optional[bytes]]:
    return {path: path.read_bytes() if path.is_file() else None for path in paths}


def restore_file_versions(versions: Dict[Path, Optional[bytes]]) -> None:
    for path, data in versions.items():
        if data is None:
            path.unlink(missing_ok=True)
        else:
            atomic_write_bytes(path, data)


def read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"failed to read JSON from {path}: {e}") from e
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def utc_now() -> str:
    return load_campaign_guard().utc_now()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(data: Any) -> str:
    return sha256_bytes(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def safe_relative_path(workspace: Path, value: str) -> str:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (workspace / path).resolve()
    try:
        relative = resolved.relative_to(workspace.resolve())
    except ValueError as e:
        raise ValueError(f"path escapes the Auto-FL job workspace: {value}") from e
    if not relative.parts or any(part in RESERVED_CANDIDATE_PATH_PARTS for part in relative.parts):
        raise ValueError(f"path is reserved for Auto-FL or repository metadata: {value}")
    return relative.as_posix()


def allowed_edit_paths(config: Dict[str, Any], workspace: Path) -> List[str]:
    values = config.get("trust_contract", {}).get("allowed_edit_paths", []) or []
    if not isinstance(values, list):
        raise ValueError("autofl.yaml trust_contract.allowed_edit_paths must be a list")
    return list(dict.fromkeys(safe_relative_path(workspace, str(value)) for value in values))


def allowed_create_patterns(config: Dict[str, Any]) -> List[str]:
    values = config.get("trust_contract", {}).get("allowed_create_patterns", []) or []
    if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
        raise ValueError("autofl.yaml trust_contract.allowed_create_patterns must be a list of patterns")
    return list(dict.fromkeys(values))


def admitted_source_hashes(config: Dict[str, Any], workspace: Path) -> Dict[str, Optional[str]]:
    hashes = {}
    for relative in allowed_edit_paths(config, workspace):
        path = workspace / relative
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"campaign source path is not a regular file: {path}")
        hashes[relative] = sha256_bytes(path.read_bytes()) if path.is_file() else None
    return hashes


def is_allowed_new_source(path: str, patterns: Sequence[str]) -> bool:
    relative = Path(path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or any(part in RESERVED_CANDIDATE_PATH_PARTS for part in relative.parts)
    ):
        return False
    posix_path = PurePosixPath(relative.as_posix())
    return any(
        posix_path.match(pattern) or (pattern.startswith("**/") and posix_path.match(pattern[3:]))
        for pattern in patterns
    )


def validated_lexical_relative_path(value: str) -> str:
    path = Path(value)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or any(part in RESERVED_CANDIDATE_PATH_PARTS for part in path.parts)
    ):
        raise ValueError(f"invalid previously validated managed source path: {value}")
    return path.as_posix()


def discover_allowed_created_sources(workspace: Path, patterns: Sequence[str]) -> List[str]:
    paths = []
    for directory, dir_names, file_names in os.walk(workspace, followlinks=False):
        directory_path = Path(directory)
        dir_names[:] = [
            name
            for name in dir_names
            if name not in RESERVED_CANDIDATE_PATH_PARTS and not (directory_path / name).is_symlink()
        ]
        for name in file_names:
            relative = (directory_path / name).relative_to(workspace).as_posix()
            if is_allowed_new_source(relative, patterns):
                paths.append(relative)
    return paths


def managed_source_paths(workspace: Path, config: Dict[str, Any], extra_paths: Sequence[str] = ()) -> List[str]:
    paths = set(allowed_edit_paths(config, workspace))
    paths.update(safe_relative_path(workspace, value) for value in extra_paths)
    paths.update(discover_allowed_created_sources(workspace, allowed_create_patterns(config)))
    return sorted(paths)


def managed_source_paths_from_validated(
    workspace: Path, config: Dict[str, Any], validated_paths: Sequence[str]
) -> List[str]:
    paths = {validated_lexical_relative_path(value) for value in validated_paths}
    paths.update(discover_allowed_created_sources(workspace, allowed_create_patterns(config)))
    return sorted(paths)


def capture_managed_source_versions(
    workspace: Path, config: Dict[str, Any], extra_paths: Sequence[str] = ()
) -> Dict[Path, Optional[bytes]]:
    versions = {}
    for relative in managed_source_paths(workspace, config, extra_paths):
        path = workspace / relative
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"campaign-managed source path is not a regular file: {path}")
        versions[path] = path.read_bytes() if path.is_file() else None
    return versions


def managed_path_state(workspace: Path, relative: str) -> str:
    parts = Path(validated_lexical_relative_path(relative)).parts
    current = workspace
    for index, part in enumerate(parts):
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return "missing"
        component = Path(*parts[: index + 1]).as_posix()
        if stat.S_ISLNK(mode):
            return f"symlink:{component}"
        if index < len(parts) - 1:
            if not stat.S_ISDIR(mode):
                return f"non-directory:{component}"
        elif stat.S_ISREG(mode):
            return sha256_bytes(current.read_bytes())
        else:
            return "non-file"
    return "missing"


def managed_source_state(workspace: Path, config: Dict[str, Any], validated_paths: Sequence[str]) -> Dict[str, str]:
    state = {}
    for relative in managed_source_paths_from_validated(workspace, config, validated_paths):
        state[relative] = managed_path_state(workspace, relative)
    return state


def managed_source_drift(expected: Dict[str, str], actual: Dict[str, str]) -> List[str]:
    return sorted(path for path in set(expected) | set(actual) if expected.get(path) != actual.get(path))


def remove_path_without_following(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode):
        shutil.rmtree(path)
    else:
        path.unlink()


def ensure_managed_parent_directories(workspace: Path, relative: str) -> None:
    current = workspace
    for part in Path(validated_lexical_relative_path(relative)).parts[:-1]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            current.mkdir()
            continue
        if not stat.S_ISDIR(mode):
            remove_path_without_following(current)
            current.mkdir()


def restore_managed_source_versions(
    workspace: Path, config: Dict[str, Any], versions: Dict[Path, Optional[bytes]]
) -> None:
    expected = {
        validated_lexical_relative_path(path.relative_to(workspace).as_posix()): data for path, data in versions.items()
    }
    current_paths = managed_source_paths_from_validated(workspace, config, list(expected))
    for relative in current_paths:
        ensure_managed_parent_directories(workspace, relative)
    for relative in current_paths:
        path = workspace / relative
        data = expected.get(relative)
        if data is None:
            remove_path_without_following(path)
            continue
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            mode = None
        if mode is not None and not stat.S_ISREG(mode):
            remove_path_without_following(path)
        atomic_write_bytes(path, data)


def file_map(root: Path) -> Dict[str, str]:
    if not root.exists():
        return {}
    files: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"candidate source contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            files[relative] = sha256_bytes(path.read_bytes())
    return files


def source_hash(files: Dict[str, str]) -> str:
    return sha256_json(files)


def copy_relative_file(source_root: Path, destination_root: Path, relative: str) -> None:
    relative = safe_relative_path(destination_root, relative)
    source = source_root / relative
    destination = destination_root / relative
    try:
        source.resolve().relative_to(source_root.resolve())
    except ValueError as e:
        raise ValueError(f"source path escapes its managed root: {relative}") from e
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"managed source path is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def stage_best_snapshot(workspace: Path, config: Dict[str, Any], snapshot_root: Path) -> Tuple[Path, Dict[str, str]]:
    snapshot_root.parent.mkdir(parents=True, exist_ok=True)
    staged_root = snapshot_root.parent / f".{snapshot_root.name}.staged-{uuid.uuid4().hex}"
    source_root = staged_root / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    try:
        for relative in allowed_edit_paths(config, workspace):
            source = workspace / relative
            if source.is_symlink():
                raise ValueError(f"allowed edit path is a symlink: {source}")
            if source.is_file():
                copy_relative_file(workspace, source_root, relative)
        files = file_map(source_root)
        write_json(staged_root / "snapshot.json", {"source_sha256": source_hash(files), "files": files})
        return staged_root, files
    except BaseException:
        shutil.rmtree(staged_root, ignore_errors=True)
        raise


def activate_best_snapshot(snapshot_root: Path, staged_root: Path) -> Optional[Path]:
    previous_root = None
    if snapshot_root.exists():
        previous_root = snapshot_root.parent / f".{snapshot_root.name}.previous-{uuid.uuid4().hex}"
        os.replace(snapshot_root, previous_root)
    try:
        os.replace(staged_root, snapshot_root)
    except BaseException:
        if previous_root is not None:
            os.replace(previous_root, snapshot_root)
        raise
    return previous_root


def rollback_best_snapshot(snapshot_root: Path, previous_root: Optional[Path]) -> None:
    if previous_root is None:
        shutil.rmtree(snapshot_root, ignore_errors=True)
        return
    failed_root = snapshot_root.parent / f".{snapshot_root.name}.failed-{uuid.uuid4().hex}"
    if snapshot_root.exists():
        os.replace(snapshot_root, failed_root)
    try:
        os.replace(previous_root, snapshot_root)
    except BaseException:
        if failed_root.exists():
            os.replace(failed_root, snapshot_root)
        raise
    shutil.rmtree(failed_root, ignore_errors=True)


def create_best_snapshot(workspace: Path, config: Dict[str, Any], snapshot_root: Path) -> Dict[str, str]:
    staged_root, files = stage_best_snapshot(workspace, config, snapshot_root)
    try:
        previous_root = activate_best_snapshot(snapshot_root, staged_root)
    finally:
        if staged_root.exists():
            shutil.rmtree(staged_root, ignore_errors=True)
    if previous_root is not None:
        shutil.rmtree(previous_root, ignore_errors=True)
    return files


def load_best_snapshot(snapshot_root: Path) -> Tuple[Path, Dict[str, str]]:
    metadata = read_json(snapshot_root / "snapshot.json")
    files = metadata.get("files")
    if not isinstance(files, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items()):
        raise ValueError("best snapshot metadata has an invalid files mapping")
    source_root = snapshot_root / "source"
    if source_hash(files) != metadata.get("source_sha256") or file_map(source_root) != files:
        raise ValueError("best snapshot failed its integrity check")
    return source_root, files


def workspace_matches_snapshot(workspace: Path, source_root: Path, files: Dict[str, str]) -> bool:
    for relative, digest in files.items():
        path = workspace / relative
        if path.is_symlink() or not path.is_file() or sha256_bytes(path.read_bytes()) != digest:
            return False
    return True


def validate_candidate_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value or ""):
        raise ValueError("candidate name must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    return value


def candidate_manifest_path(workspace: Path, candidate_id: str) -> Path:
    return workspace / CANDIDATE_ROOT / candidate_id / "candidate_manifest.json"


def validate_candidate_manifest_identity(path: Path, manifest: Dict[str, Any]) -> str:
    if manifest.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported candidate manifest schema in {path}")
    candidate_id = validate_candidate_id(str(manifest.get("candidate_id") or ""))
    expected = candidate_manifest_path(Path(str(manifest.get("workspace_root") or "")), candidate_id).resolve()
    if path.resolve() != expected:
        raise ValueError("candidate manifest path does not match its workspace and candidate ID")
    if manifest.get("status") not in CANDIDATE_MANIFEST_STATUSES:
        raise ValueError(f"candidate {candidate_id} has an invalid manifest status")
    return candidate_id


def load_candidate_manifest(path: Path) -> Dict[str, Any]:
    manifest = read_json(path)
    candidate_id = validate_candidate_manifest_identity(path, manifest)
    if manifest.get("status") not in {"prepared", "ready_for_external_execution"}:
        raise ValueError(f"candidate {candidate_id} is not pending evaluation")
    return manifest


def campaign_metadata_path(workspace: Path) -> Path:
    return workspace / CAMPAIGN_METADATA_PATH


def load_campaign_metadata(workspace: Path, job: Path) -> Dict[str, Any]:
    metadata = read_json(campaign_metadata_path(workspace))
    if metadata.get("schema_version") != CAMPAIGN_METADATA_SCHEMA_VERSION:
        raise ValueError("unsupported Auto-FL campaign metadata schema")
    if Path(str(metadata.get("job") or "")).resolve() != job.resolve():
        raise ValueError("campaign metadata belongs to a different job.py")
    settings = metadata.get("settings")
    if isinstance(settings, dict):
        persisted_mode = settings.get("mode", "max")
        legacy_min = (
            persisted_mode == "min"
            and metadata.get("metric_direction_contract_version") != METRIC_DIRECTION_CONTRACT_VERSION
        )
        if not legacy_min:
            config_path = resolve_output_path(workspace, str(settings.get("autofl_yaml") or "autofl.yaml"))
            campaign_contract_mode(metadata, read_yaml(config_path), settings.get("metric"))
    return metadata


def fixed_budget_hash(config: Dict[str, Any]) -> str:
    return sha256_json(config.get("budget", {}).get("fixed_training_budget", {}) or {})


def candidate_changes(
    workspace: Path,
    config: Dict[str, Any],
    best_source: Path,
    best_files: Dict[str, str],
    draft_source: Path,
    *,
    allow_materialized: bool = False,
) -> Tuple[List[str], List[str]]:
    draft_files = file_map(draft_source)
    deleted = sorted(set(best_files) - set(draft_files))
    if deleted:
        raise ValueError(f"candidate deletes managed source files: {', '.join(deleted)}")

    allowed = set(allowed_edit_paths(config, workspace))
    create_patterns = allowed_create_patterns(config)
    changed = sorted(path for path, digest in draft_files.items() if best_files.get(path) != digest)
    created = []
    for relative in changed:
        if relative in best_files:
            if relative not in allowed:
                raise ValueError(f"candidate modifies a path outside trust_contract.allowed_edit_paths: {relative}")
            continue
        if ((workspace / relative).exists() and not allow_materialized) or not is_allowed_new_source(
            relative, create_patterns
        ):
            raise ValueError(f"candidate creates an unauthorized source path: {relative}")
        created.append(relative)
    return changed, created


def text_for_diff(path: Path) -> List[str]:
    data = path.read_bytes()
    if b"\0" in data:
        raise ValueError(f"candidate diff does not support binary file: {path}")
    return data.decode("utf-8", errors="replace").splitlines(keepends=True)


def render_candidate_patch(best_source: Path, draft_source: Path, changed: Sequence[str]) -> str:
    chunks: List[str] = []
    for relative in changed:
        before = text_for_diff(best_source / relative) if (best_source / relative).is_file() else []
        after = text_for_diff(draft_source / relative)
        chunks.extend(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    return "".join(chunks)


def apply_candidate_source(workspace: Path, draft_source: Path, changed: Sequence[str]) -> None:
    for relative in changed:
        copy_relative_file(draft_source, workspace, relative)


def restore_best_source(
    workspace: Path,
    best_source: Path,
    best_files: Dict[str, str],
    changed: Sequence[str],
    created: Sequence[str],
) -> None:
    restore_paths = {safe_relative_path(workspace, value) for value in list(changed) + list(created)}
    for relative in sorted(restore_paths):
        destination = workspace / relative
        if relative in best_files:
            copy_relative_file(best_source, workspace, relative)
        elif destination.exists() and not destination.is_dir():
            destination.unlink()


def validated_manifest_paths(manifest: Dict[str, Any], workspace: Path, field: str) -> List[str]:
    values = manifest.get(field) or []
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"candidate manifest {field} must be a list of paths")
    paths = []
    for value in values:
        normalized = safe_relative_path(workspace, value)
        if value != normalized:
            raise ValueError(f"candidate manifest {field} contains a non-canonical path: {value}")
        paths.append(normalized)
    if len(paths) != len(set(paths)):
        raise ValueError(f"candidate manifest {field} contains duplicate paths")
    return sorted(paths)


def validate_manifest_change_set(
    manifest: Dict[str, Any], workspace: Path, changed: Sequence[str], created: Sequence[str]
) -> None:
    recorded_changed = validated_manifest_paths(manifest, workspace, "changed_files")
    recorded_created = validated_manifest_paths(manifest, workspace, "created_files")
    if recorded_changed != sorted(changed) or recorded_created != sorted(created):
        raise ValueError("candidate manifest source lists do not match the deterministic candidate diff")


def load_sibling_module(filename: str, module_name: str):
    """Load a sibling skill script as a module, cached in sys.modules."""
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    module_path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load skill module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_campaign_guard():
    return load_sibling_module("campaign_guard.py", "nvflare_autofl_skill_campaign_guard")


def load_job_importer():
    return load_sibling_module("job_importer.py", "nvflare_autofl_skill_job_importer")


def resolve_output_path(cwd: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return cwd / path


def resolve_stop_files(cwd: Path, values: Optional[Sequence[str]]) -> List[str]:
    guard = load_campaign_guard()
    stop_files = values if values is not None else list(guard.DEFAULT_STOP_FILES)
    return [str(resolve_output_path(cwd, value)) for value in stop_files]


def ensure_campaign_not_stopped(workspace: Path, args: argparse.Namespace, *, action: str) -> None:
    existing = [path for value in resolve_stop_files(workspace, args.stop_file) if (path := Path(value)).exists()]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise ValueError(f"campaign is manually stopped by {paths}; remove the stop file before attempting to {action}")


def extract_result_dir(output: str, cwd: Optional[Path] = None) -> Optional[Path]:
    patterns = [
        r"Result can be found in\s*:\s*(?P<path>\S+)",
        r"Results:\s*(?P<path>\S+)",
        r"result_dir=(?P<path>\S+)",
        r"result location\s*[:=]\s*(?P<path>\S+)",
        r"simulation logs can be found at\s+(?P<path>\S+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            path = Path(match.group("path")).expanduser()
            if not path.is_absolute() and cwd is not None:
                path = cwd / path
            return path.resolve()
    return None


def sandbox_socket_failure_signals(text: str) -> Tuple[bool, bool, bool]:
    text = text.lower()
    return (
        "permissionerror" in text,
        "operation not permitted" in text or "[errno 1]" in text,
        "socket" in text or "sock" in text or "get_open_ports" in text or ".bind(" in text,
    )


def scan_run_log(log_path: Path, cwd: Path) -> Tuple[Optional[Path], bool]:
    result_dir = None
    permission_error = False
    operation_denied = False
    socket_context = False
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
            for line in log_file:
                if result_dir is None:
                    result_dir = extract_result_dir(line, cwd)
                signals = sandbox_socket_failure_signals(line)
                permission_error = permission_error or signals[0]
                operation_denied = operation_denied or signals[1]
                socket_context = socket_context or signals[2]
    except OSError:
        return None, False
    return result_dir, permission_error and operation_denied and socket_context


def objective_contract(config: Dict[str, Any], requested_metric: Optional[str]) -> Dict[str, Any]:
    objective = config.get("objective", {})
    if not isinstance(objective, dict):
        objective = {}
    metric = str(objective.get("metric") or requested_metric or "accuracy")
    requested = str(objective.get("requested_metric") or metric)
    optimization = str(objective.get("optimization_metric") or metric)
    extraction_order = objective.get("metric_extraction_order")
    if not isinstance(extraction_order, list) or not extraction_order:
        extraction_order = [optimization]
    extraction_order = [str(item) for item in extraction_order if item]
    if optimization not in extraction_order:
        extraction_order.insert(0, optimization)
    mode = str(objective.get("mode") or "max").strip().lower()
    if mode not in {"min", "max"}:
        raise ValueError(f"objective.mode must be 'min' or 'max', but got {mode!r}")
    return {
        **objective,
        "metric": metric,
        "requested_metric": requested,
        "optimization_metric": optimization,
        "metric_extraction_order": extraction_order,
        "mode": mode,
    }


def job_key_metric_is_resolved(objective: Dict[str, Any]) -> bool:
    """Return whether the job key metric has a trustworthy static identity."""

    return objective.get("job_key_metric_source") != "default"


def requested_metric_differs_from_job(objective: Dict[str, Any]) -> bool:
    job_metric = objective.get("job_key_metric") or objective.get("metric")
    requested_metric = objective.get("requested_metric") or objective.get("metric")
    if not job_metric or not requested_metric:
        return True
    return not job_key_metric_is_resolved(objective) or str(requested_metric) != str(job_metric)


def apply_metric_contract(
    config: Dict[str, Any], requested_metric: Optional[str], schema: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    objective = objective_contract(config, requested_metric)
    requested_differs = requested_metric_differs_from_job(objective)
    if requested_differs:
        # Direction belongs to the requested metric. A different job key metric
        # cannot supply it, so begin with NVFlare's default until the bridge says otherwise.
        objective["mode"] = "max"
        objective["mode_contract_source"] = "core_default"
    schema_objective = (schema or {}).get("objective", {})
    if isinstance(schema_objective, dict):
        schema_requested = schema_objective.get("requested_metric") or schema_objective.get("metric")
        schema_applies = schema_requested == objective["requested_metric"] or (
            not schema_requested and not requested_differs
        )
        if schema_applies:
            schema_mode = schema_objective.get("mode")
            if schema_mode is not None:
                schema_mode = str(schema_mode).strip().lower()
                if schema_mode not in {"min", "max"}:
                    raise ValueError(
                        f"mutation_schema.yaml objective.mode must be 'min' or 'max', but got {schema_mode!r}"
                    )
                if not requested_differs and schema_mode != objective["mode"]:
                    if objective.get("mode_contract_source") == "core_default":
                        raise ValueError(
                            "mutation_schema.yaml objective.mode conflicts with NVFlare's implicit 'max' default for "
                            f"the job key metric; set key_metric_mode={schema_mode!r} in job.py and initialize again"
                        )
                    raise ValueError(
                        "mutation_schema.yaml objective.mode conflicts with job.py key_metric_mode; "
                        "job.py is authoritative for its key metric"
                    )
                if requested_differs:
                    objective["mode"] = schema_mode
                    objective["mode_contract_source"] = "mutation_schema"
            for key in ("optimization_metric", "metric_extraction_order", "metric_source"):
                if key in schema_objective:
                    objective[key] = schema_objective[key]
    config["objective"] = objective_contract({"objective": objective}, requested_metric)
    return config


def metric_extraction_order(config: Dict[str, Any], requested_metric: Optional[str]) -> List[str]:
    return list(objective_contract(config, requested_metric)["metric_extraction_order"])


def optimization_metric(config: Dict[str, Any], requested_metric: Optional[str]) -> str:
    return str(objective_contract(config, requested_metric)["optimization_metric"])


def objective_mode(config: Dict[str, Any], requested_metric: Optional[str] = None) -> str:
    return str(objective_contract(config, requested_metric)["mode"])


def campaign_contract_mode(
    metadata: Dict[str, Any], config: Dict[str, Any], requested_metric: Optional[str] = None
) -> str:
    settings = metadata.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("campaign metadata is missing settings")
    persisted_mode = settings.get("mode", "max")
    if persisted_mode not in {"min", "max"}:
        raise ValueError(f"campaign metadata has invalid objective mode {persisted_mode!r}")
    config_mode = objective_mode(config, requested_metric)
    if persisted_mode != config_mode:
        raise ValueError(
            "campaign metric direction disagrees between campaign.json settings.mode "
            f"({persisted_mode!r}) and autofl.yaml objective.mode ({config_mode!r}); "
            "restore the admitted campaign artifacts before continuing"
        )
    return config_mode


def metric_source(config: Dict[str, Any]) -> str:
    source = config.get("objective", {}).get("metric_source", "")
    return str(source) if source else "NVFlare metric artifacts"


def normalize_metric_order(metrics: Sequence[str] | str) -> List[str]:
    if isinstance(metrics, str):
        return [metrics]
    return [str(metric) for metric in metrics if metric]


def parse_finite_metric_value(value: Any) -> Optional[float]:
    return load_campaign_guard().parse_score(value)


def is_numeric_metric_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and parse_finite_metric_value(value) is not None


def find_metric_value(payload: Any, metric_order: Sequence[str] | str) -> Optional[float]:
    metric_keys = normalize_metric_order(metric_order)
    if isinstance(payload, dict):
        for key in ("final_aggregated_metrics", "best_metrics", "metrics"):
            for metric_key in metric_keys:
                value = metric_from_list(payload.get(key), metric_key)
                if value is not None:
                    return value
        for metric_key in metric_keys:
            if metric_key in payload and is_numeric_metric_value(payload[metric_key]):
                return parse_finite_metric_value(payload[metric_key])
        for value in payload.values():
            score = find_metric_value(value, metric_keys)
            if score is not None:
                return score
    elif isinstance(payload, list):
        for metric_key in metric_keys:
            value = metric_from_list(payload, metric_key)
            if value is not None:
                return value
        for item in payload:
            score = find_metric_value(item, metric_keys)
            if score is not None:
                return score
    return None


def metric_from_list(items: Any, metric: str) -> Optional[float]:
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("name") == metric and is_numeric_metric_value(item.get("value")):
            return parse_finite_metric_value(item["value"])
    return None


def server_global_model_metric_values(payload: Any, metric: str) -> Tuple[List[float], List[float]]:
    """Collect per-site metric values for server-side global-model entries in a cross-site payload.

    Returns (final_values, best_values): entries whose key contains the best-checkpoint marker
    (SRV_best_FL_global_model.pt, best_FL_global_model.pt) land in best_values; the remaining
    global-model entries are final-checkpoint values.
    """
    final_values: List[float] = []
    best_values: List[float] = []
    if isinstance(payload, dict):
        for key, entry in payload.items():
            if isinstance(key, str) and SERVER_GLOBAL_MODEL_KEY_MARKER in key:
                value = find_metric_value(entry, [metric])
                if value is not None:
                    target = best_values if SERVER_BEST_GLOBAL_MODEL_KEY_MARKER in key else final_values
                    target.append(value)
            else:
                nested_final, nested_best = server_global_model_metric_values(entry, metric)
                final_values.extend(nested_final)
                best_values.extend(nested_best)
    elif isinstance(payload, list):
        for item in payload:
            nested_final, nested_best = server_global_model_metric_values(item, metric)
            final_values.extend(nested_final)
            best_values.extend(nested_best)
    return final_values, best_values


def text_metric_matches(text: str, metric: str) -> List[Tuple[int, float]]:
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    context = r"(?:(?:final|test|validation|cross[-_ ]site)\s+)*"
    line_prefix = r"^\s*(?:\[[^\]]+\]\s*)*(?:\{|\[)?\s*"
    direct_pattern = re.compile(
        line_prefix + rf"{context}[\"']?{re.escape(metric)}[\"']?" rf"\s*[:=]\s*({number})(?:\s|[,}}\]]|$)",
        flags=re.IGNORECASE,
    )
    evaluation_pattern = re.compile(
        rf"\b{re.escape(metric)}\s+of\s+the\s+received\s+model\b.*?:\s*({number})(?:\s*%|\s|$)",
        flags=re.IGNORECASE,
    )
    framework_pattern = re.compile(
        rf"(?<![\w]){re.escape(metric)}(?![\w])\s*[:=]\s*({number})(?:\s|[,}}\]]|$)",
        flags=re.IGNORECASE,
    )
    mapping_pattern = re.compile(
        rf"(?:\{{|,)\s*[\"']{re.escape(metric)}[\"']\s*:\s*({number})(?:\s|[,}}]|$)",
        flags=re.IGNORECASE,
    )
    direct_matches = []
    evaluation_matches = []
    mapping_matches = []
    framework_matches = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        log_prefix = re.match(r"^\d{4}-\d{2}-\d{2}[^\n]*? - [A-Z]+ - (.*)$", line)
        message = log_prefix.group(1) if log_prefix else line
        direct_match = direct_pattern.search(message)
        if direct_match:
            score = parse_finite_metric_value(direct_match.group(1))
            if score is not None:
                direct_matches.append((line_number, score))
        evaluation_match = evaluation_pattern.search(message)
        if evaluation_match:
            score = parse_finite_metric_value(evaluation_match.group(1))
            if score is not None:
                evaluation_matches.append((line_number, score))
        mapping_match = mapping_pattern.search(message)
        if mapping_match:
            score = parse_finite_metric_value(mapping_match.group(1))
            if score is not None:
                mapping_matches.append((line_number, score))
        if re.search(r"\d+/\d+.*\d+(?:\.\d+)?(?:ms|s)/step\b", message):
            matches = list(framework_pattern.finditer(message))
            if matches:
                score = parse_finite_metric_value(matches[-1].group(1))
                if score is not None:
                    framework_matches.append((line_number, score))
    return evaluation_matches or direct_matches or mapping_matches or framework_matches


def extract_metric_evidence(artifact_root: Path, metrics: Sequence[str] | str) -> Optional[MetricEvidence]:
    metric_order = normalize_metric_order(metrics)
    metric_files = sorted(artifact_root.glob("**/metrics_summary.json")) + sorted(
        artifact_root.glob("**/cross_val_results.json")
    )
    payloads = []
    for path in metric_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payloads.append((path, payload))
    for metric in metric_order:
        for path, payload in payloads:
            if path.name == "cross_val_results.json":
                # Cross-site payloads score every site/model pair; prefer the server global
                # model over whichever entry happens to appear first, and final-checkpoint
                # entries over best_-checkpoint entries. Per-site sample counts are not
                # recorded in the payload, so the defined reducer is the unweighted mean
                # across evaluating sites (a max/min would just pick the easiest site).
                final_values, best_values = server_global_model_metric_values(payload, metric)
                server_values = final_values or best_values
                if server_values:
                    return MetricEvidence(
                        score=sum(server_values) / len(server_values),
                        metric_name=metric,
                        source=f"structured:{path.name}#server_final",
                        artifact=str(path.resolve()),
                    )
            score = find_metric_value(payload, [metric])
            if score is not None:
                return MetricEvidence(
                    score=score,
                    metric_name=metric,
                    source=f"structured:{path.name}",
                    artifact=str(path.resolve()),
                )

    text_artifacts = []
    for name in ("run.log", "log.txt", "log_fl.txt", "error_log.txt"):
        text_artifacts.extend(sorted(artifact_root.glob(f"**/{name}")))
    for metric in metric_order:
        for path in text_artifacts:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            matches = text_metric_matches(text, metric)
            if matches:
                line_number, score = matches[-1]
                return MetricEvidence(
                    score=score,
                    metric_name=metric,
                    source=f"text:{path.name}:line={line_number}",
                    artifact=str(path.resolve()),
                )
    return None


def extract_score(artifact_root: Path, metrics: Sequence[str] | str) -> Optional[float]:
    evidence = extract_metric_evidence(artifact_root, metrics)
    return evidence.score if evidence else None


def metric_search_description(artifact_root: Path, metrics: Sequence[str] | str) -> str:
    names = ["metrics_summary.json", "cross_val_results.json", "run.log", "log.txt", "log_fl.txt", "error_log.txt"]
    searched = [str(path) for name in names for path in sorted(artifact_root.glob(f"**/{name}"))]
    return f"metrics={normalize_metric_order(metrics)!r}; searched={searched or [str(artifact_root)]!r}"


def is_sandbox_socket_failure(output: str) -> bool:
    return all(sandbox_socket_failure_signals(output))


def simulation_approval_instruction() -> str:
    return (
        "Pause for human approval. Retry the same simulation lifecycle command only under the campaign setup "
        "grant scoped to the exact interpreter, Auto-FL runner, initialize/evaluate action, and job.py. Never "
        "request broader permission or infer authorization from log output; infrastructure retries do not count "
        "against the candidate budget."
    )


def is_nvflare_simulator_stall(output: str) -> bool:
    return "SIMULATOR_STALL:" in output


def collect_artifacts(result_dir: Optional[Path], output_root: Path, name: str, log_path: Path) -> Path:
    dest = output_root / name / "simulation"
    run_log = output_root / name / "run.log"
    if dest.exists():
        shutil.rmtree(dest)
    if result_dir and result_dir.exists():
        shutil.copytree(result_dir, dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)
    if log_path.resolve() != run_log.resolve():
        shutil.copy2(log_path, run_log)
    return dest


def job_help(python: str, job: Path, cwd: Path, timeout: int = DEFAULT_JOB_HELP_TIMEOUT) -> str:
    del python, cwd, timeout
    try:
        flags = load_job_importer().inspect_job_cli_flags(str(job))
    except (OSError, ValueError) as e:
        raise RuntimeError(f"cannot inspect job CLI flags in {job}: {e}") from e
    return " ".join(sorted(flags))


def supported_long_flags(help_text: str) -> set[str]:
    return set(re.findall(r"(?<![\w-])(--[A-Za-z][A-Za-z0-9_-]*)", help_text))


def supports_flag(help_text: str, flag: str) -> bool:
    return flag in supported_long_flags(help_text)


def mutable_arg_specs(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    specs = schema.get("mutable_args")
    return specs if isinstance(specs, dict) else {}


def candidate_arg_values(candidate_args: Sequence[str]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    idx = 0
    while idx < len(candidate_args):
        raw = candidate_args[idx]
        if raw.startswith("--") and not re.match(r"^--[A-Za-z][A-Za-z0-9_-]*(?:=.*)?$", raw):
            raise ValueError(f"invalid canonical long option: {raw!r}")
        if (
            raw.startswith("-")
            and not raw.startswith("--")
            and not re.match(r"^-(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$", raw)
        ):
            raise ValueError(f"candidate run arguments require canonical long options, not {raw!r}")
        if not raw.startswith("--"):
            idx += 1
            continue
        option = raw[2:]
        if "=" in option:
            name, value = option.split("=", 1)
            values[name.replace("-", "_")] = value
            idx += 1
            continue
        name = option.replace("-", "_")
        if idx + 1 >= len(candidate_args) or candidate_args[idx + 1].startswith("--"):
            values[name] = True
            idx += 1
        else:
            values[name] = candidate_args[idx + 1]
            idx += 2
    return values


def coerce_schema_value(value: Any, spec: Dict[str, Any]) -> Any:
    value_type = spec.get("type")
    if value_type == "int":
        return int(value)
    if value_type == "float":
        return float(value)
    if value_type == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    return str(value)


def candidate_args_allowed(candidate_args: Sequence[str], schema: Dict[str, Any]) -> Tuple[bool, str]:
    specs = mutable_arg_specs(schema)
    if not specs:
        return True, ""

    for name, value in candidate_arg_values(candidate_args).items():
        spec = specs.get(name)
        if not isinstance(spec, dict):
            continue
        try:
            coerced = coerce_schema_value(value, spec)
        except (TypeError, ValueError) as e:
            return False, f"{name}={value!r} cannot be parsed as {spec.get('type')}: {e}"

        choices = spec.get("choices")
        if choices is not None and coerced not in choices:
            return False, f"{name}={coerced!r} is not in allowed choices {choices!r}"
        minimum = spec.get("min")
        if minimum is not None and coerced < minimum:
            return False, f"{name}={coerced!r} is below schema min {minimum!r}"
        maximum = spec.get("max")
        if maximum is not None and coerced > maximum:
            return False, f"{name}={coerced!r} is above schema max {maximum!r}"

    return True, ""


def candidate_preserves_fixed_args(
    candidate_args: Sequence[str], config: Dict[str, Any], schema: Dict[str, Any]
) -> Tuple[bool, str]:
    values = candidate_arg_values(candidate_args)
    fixed_names = set(fixed_within_campaign(schema))
    fixed_budget = config.get("budget", {}).get("fixed_training_budget", {}) or {}
    fixed_names.update(FIXED_BUDGET_TO_CLI[field] for field in fixed_budget if field in FIXED_BUDGET_TO_CLI)
    normalized_fixed_names = {name.replace("-", "_") for name in fixed_names}
    changed = sorted(normalized_fixed_names.intersection(values))
    if changed:
        return False, f"candidate run arguments change fixed-budget fields: {', '.join(changed)}"
    for raw in candidate_args:
        if not raw.startswith("--"):
            continue
        supplied = raw[2:].split("=", 1)[0].replace("-", "_")
        abbreviated = sorted(name for name in normalized_fixed_names if name.startswith(supplied) and name != supplied)
        if abbreviated:
            return False, (
                f"candidate run argument {raw!r} is a strict prefix of fixed-budget option(s): "
                f"{', '.join('--' + name for name in abbreviated)}"
            )
    return True, ""


def load_mutation_schema(cwd: Path) -> Dict[str, Any]:
    path = cwd / "mutation_schema.yaml"
    if not path.exists():
        return {}
    return read_yaml(path)


def apply_mutation_schema_contract(config: Dict[str, Any], schema: Dict[str, Any], workspace: Path) -> Dict[str, Any]:
    preferred_targets = schema.get("preferred_targets")
    if preferred_targets is None:
        return config
    if not isinstance(preferred_targets, list) or not all(isinstance(value, str) for value in preferred_targets):
        raise ValueError("mutation_schema.yaml preferred_targets must be a list of paths")

    trust_contract = config.setdefault("trust_contract", {})
    trust_paths = trust_contract.setdefault("allowed_edit_paths", [])
    if not isinstance(trust_paths, list):
        raise ValueError("autofl.yaml trust_contract.allowed_edit_paths must be a list")

    resolved_targets = []
    unresolved_targets = []
    for target in preferred_targets:
        try:
            relative = safe_relative_path(workspace, target)
        except ValueError as e:
            unresolved_targets.append(_schema_target_issue(target, str(e)))
            continue
        path = workspace / relative
        if path.is_symlink():
            unresolved_targets.append(_schema_target_issue(target, "preferred target is a symlink"))
            continue
        if not path.is_file():
            unresolved_targets.append(
                _schema_target_issue(target, "preferred target is not an existing file under the job workspace")
            )
            continue
        resolved_targets.append(relative)
        if relative not in trust_paths:
            trust_paths.append(relative)

    trust_contract["preferred_targets"] = list(resolved_targets)
    if unresolved_targets:
        unresolved = config.setdefault("unresolved", [])
        trust_unresolved = trust_contract.setdefault("unresolved", [])
        if not isinstance(unresolved, list) or not isinstance(trust_unresolved, list):
            raise ValueError("autofl.yaml unresolved fields must be lists")
        unresolved.extend(unresolved_targets)
        trust_unresolved.extend(dict(item) for item in unresolved_targets)
    return config


def _schema_target_issue(target: str, reason: str) -> Dict[str, str]:
    return {"field": "mutation_schema.preferred_targets", "reason": f"{target}: {reason}"}


def comparison_budget(schema: Dict[str, Any]) -> Dict[str, Any]:
    comparison = schema.get("comparison_budget_args")
    if not isinstance(comparison, dict):
        return {}
    budget = comparison.get("default_candidate_budget")
    return budget if isinstance(budget, dict) else {}


def fixed_within_campaign(schema: Dict[str, Any]) -> set:
    values = []
    comparison = schema.get("comparison_budget_args")
    if isinstance(comparison, dict):
        values = comparison.get("fixed_within_campaign") or []
    return set(values) if isinstance(values, list) else set()


def build_comparison_budget_args(schema: Dict[str, Any], help_text: str) -> List[str]:
    budget = comparison_budget(schema)
    args: List[str] = []
    for budget_field, cli_name in COMPARISON_BUDGET_TO_CLI.items():
        value = budget.get(budget_field)
        if value is not None and supports_flag(help_text, f"--{cli_name}"):
            args.extend([f"--{cli_name}", str(value)])
    if budget.get("cross_site_eval") and supports_flag(help_text, "--cross_site_eval"):
        args.append("--cross_site_eval")
    return args


def build_fixed_args(config: Dict[str, Any], help_text: str, schema: Dict[str, Any]) -> List[str]:
    fixed = config.get("budget", {}).get("fixed_training_budget", {}) or {}
    budget = comparison_budget(schema)
    budget_cli_names = {
        cli_name for budget_field, cli_name in COMPARISON_BUDGET_TO_CLI.items() if budget.get(budget_field) is not None
    }
    args: List[str] = []
    for budget_field, cli_name in FIXED_BUDGET_TO_CLI.items():
        if cli_name in budget_cli_names:
            continue
        value = fixed.get(budget_field)
        if value is not None and supports_flag(help_text, f"--{cli_name}"):
            args.extend([f"--{cli_name}", str(value)])
    return args


def normalize_budget_flag_name(token: str) -> str:
    return token.lstrip("-").replace("-", "_")


def _injected_budget_arg_values(injected: Sequence[str]) -> Dict[str, Optional[str]]:
    values: Dict[str, Optional[str]] = {}
    index = 0
    while index < len(injected):
        name = normalize_budget_flag_name(injected[index])
        if index + 1 < len(injected) and not injected[index + 1].startswith("--"):
            values[name] = injected[index + 1]
            index += 2
        else:
            values[name] = None
            index += 1
    return values


def pinned_flag_spellings(
    pinned: Dict[str, Tuple[Optional[str], str]], injected: Sequence[str], alias_groups: Sequence[Sequence[str]]
) -> Dict[str, str]:
    """Map each parser-defined spelling (long, short, alias) of a pinned budget option to its pinned name."""
    spellings = {token: normalize_budget_flag_name(token) for token in injected if token.startswith("--")}
    for group in alias_groups:
        names = {normalize_budget_flag_name(flag) for flag in group if flag.startswith("--")}
        matched = names.intersection(pinned)
        if len(matched) == 1:
            name = next(iter(matched))
            for flag in group:
                spellings[flag] = name
    return spellings


# argparse's _negative_number_matcher: option-like tokens are not consumed as values unless they are numbers.
_NEGATIVE_NUMBER_MATCHER = re.compile(r"^-\d+$|^-\d*\.\d+$")


def _consumable_cli_value(token: str) -> bool:
    # Mirrors argparse's option classifier: lone "-" and tokens containing a space are values too.
    return not token.startswith("-") or token == "-" or " " in token or bool(_NEGATIVE_NUMBER_MATCHER.match(token))


def merge_base_budget_args(
    base_tokens: Sequence[str],
    pinned: Dict[str, Tuple[Optional[str], str]],
    spellings: Dict[str, str],
    defined_flags: set,
) -> Tuple[List[str], set[str]]:
    """Drop --base-args duplicates of runner-injected budget flags; reject conflicts and pinned-flag abbreviations.

    Tokens are matched by the exact flag spellings the job's parser defines, mirroring argparse: a spelling
    the parser would not accept passes through verbatim so the job still reports it as unrecognized. Returns
    the merged tokens plus the pinned names a kept token already sets (short-option clusters), whose
    runner-injected copies must be omitted to keep each budget option emitted exactly once.
    """
    merged: List[str] = []
    satisfied: set[str] = set()
    index = 0
    while index < len(base_tokens):
        token = base_tokens[index]
        index += 1
        if not token.startswith("-") or token in ("-", "--"):
            merged.append(token)
            continue
        option, separator, inline_value = token.partition("=")
        name = spellings.get(option)
        supplied: Optional[str] = None
        if name is not None:
            if separator:
                supplied = inline_value
            # A zero-argument pinned flag never consumes the next token (argparse binds it to positionals),
            # and a valued flag never consumes an option-like token (argparse raises "expected one argument"),
            # with argparse's negative-number exception.
            elif pinned[name][0] is not None and index < len(base_tokens) and _consumable_cli_value(base_tokens[index]):
                supplied = base_tokens[index]
                index += 1
        elif not separator and not token.startswith("--") and token not in defined_flags:
            # Attached short-option value, e.g. -r3 for a pinned -r/--num_rounds (argparse's own semantics).
            short = next((s for s in spellings if len(s) == 2 and not s.startswith("--") and token.startswith(s)), None)
            if short is not None:
                if pinned[spellings[short]][0] is None:
                    # A zero-argument pin has no attached value: argparse reads -xv as the cluster -x -v,
                    # which sets the pinned flag itself. One letter cannot be dropped from a user's cluster,
                    # so keep the cluster and omit the runner's copy of the flag instead.
                    satisfied.add(spellings[short])
                    merged.append(token)
                    continue
                name = spellings[short]
                supplied = token[2:]
            else:
                # Beyond the leading position the token is ambiguous without the other flags' arity:
                # -vx is -v -x for a boolean -v (setting the pinned flag again) but -v with value x
                # otherwise. Fail closed so no cluster can supply a pinned budget option a second time.
                involved = next(
                    (s for s in spellings if len(s) == 2 and not s.startswith("--") and s[1] in token[1:]), None
                )
                if involved is not None:
                    raise ValueError(
                        f"AUTOFL_BUDGET_ARGUMENT_CONFLICT: cannot determine whether short-option token {token} "
                        f"involves {involved} (--{spellings[involved]}); write the options as separate tokens"
                    )
        if name is None:
            if token.startswith("--") and option not in defined_flags:
                # An exactly-defined job flag is never an abbreviation: argparse gives exact matches precedence.
                ambiguous = sorted(s for s in spellings if s.startswith("--") and s.startswith(option) and s != option)
                if ambiguous:
                    raise ValueError(
                        f"AUTOFL_BUDGET_ARGUMENT_CONFLICT: --base-args option {option} abbreviates pinned "
                        f"budget option(s) {', '.join(ambiguous)}; spell the full flag"
                    )
            merged.append(token)
            continue
        pinned_value, source = pinned[name]
        if supplied == pinned_value:
            continue
        raise ValueError(
            f"AUTOFL_BUDGET_ARGUMENT_CONFLICT: --{name} is set to "
            f"{'no value' if pinned_value is None else repr(pinned_value)} by {source} but --base-args ({token}) "
            f"supplies {'no value' if supplied is None else repr(supplied)}; "
            "remove the flag from --base-args or align the values"
        )
    return merged, satisfied


def build_campaign_args(
    config: Dict[str, Any],
    args: argparse.Namespace,
    help_text: str,
    schema: Dict[str, Any],
    alias_groups: Optional[Sequence[Sequence[str]]] = None,
) -> Tuple[List[str], List[str]]:
    """Resolve run arguments so each budget option is emitted exactly once.

    Explicit --base-args duplicates of a runner-injected budget argument are dropped when the values are
    identical and rejected with an actionable error when they conflict. Duplicates are matched by the exact
    flag spellings the job's parser defines (including short aliases), so misspellings keep argparse's error.
    """
    fixed_args = build_fixed_args(config, help_text, schema)
    budget_args = build_comparison_budget_args(schema, help_text)
    pinned: Dict[str, Tuple[Optional[str], str]] = {}
    for name, value in _injected_budget_arg_values(budget_args).items():
        pinned[name] = (value, "the mutation-schema comparison budget")
    for name, value in _injected_budget_arg_values(fixed_args).items():
        pinned[name] = (value, "the imported fixed training budget")
    groups = list(alias_groups or [])
    spellings = pinned_flag_spellings(pinned, [*fixed_args, *budget_args], groups)
    defined_flags = supported_long_flags(help_text)
    defined_flags.update(flag for group in groups for flag in group)
    base_args, satisfied = merge_base_budget_args(shlex.split(args.base_args), pinned, spellings, defined_flags)
    if satisfied:
        fixed_args = [t for t in fixed_args if not (t.startswith("-") and normalize_budget_flag_name(t) in satisfied)]
        budget_args = [t for t in budget_args if not (t.startswith("-") and normalize_budget_flag_name(t) in satisfied)]
    return fixed_args, [*base_args, *budget_args]


def job_flag_alias_groups(job: Path) -> List[List[str]]:
    try:
        return load_job_importer().inspect_job_cli_flag_aliases(str(job))
    except (OSError, ValueError) as e:
        raise RuntimeError(f"cannot inspect job CLI flag aliases in {job}: {e}") from e


def suggested_arg_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    suggested = config.get("search_space", {}).get("suggested", {}) or {}
    defaults = {}
    for name, spec in suggested.items():
        if isinstance(spec, dict) and "default" in spec:
            defaults[name] = spec["default"]
    return defaults


def candidate_plan(
    config: Dict[str, Any],
    help_text: str,
    max_candidates: Optional[int],
    schema: Optional[Dict[str, Any]] = None,
) -> Iterable[JobRun]:
    defaults = suggested_arg_defaults(config)
    candidates: List[JobRun] = []
    seen_args = set()
    fixed_fields = fixed_within_campaign(schema or {})

    def can_mutate(name: str) -> bool:
        return name not in fixed_fields

    def make_candidate(name: str, candidate_args: List[str], description: str) -> Optional[JobRun]:
        allowed, _ = candidate_args_allowed(candidate_args, schema or {})
        if not allowed:
            return None
        key = tuple(candidate_args)
        if key in seen_args:
            return None
        seen_args.add(key)
        return JobRun(name=name, args=candidate_args, description=description)

    def add(name: str, candidate_args: List[str], description: str) -> None:
        if max_candidates is not None and len(candidates) >= max_candidates:
            return
        candidate = make_candidate(name, candidate_args, description)
        if candidate is not None:
            candidates.append(candidate)

    if can_mutate("aggregator") and supports_flag(help_text, "--aggregator"):
        for value in ["default", "fedavg", "fedavgm", "fedadam", "scaffold"]:
            add(f"aggregator_{value}", ["--aggregator", value], f"aggregator={value}")
        if can_mutate("fedproxloss_mu") and supports_flag(help_text, "--fedproxloss_mu"):
            add(
                "fedprox_mu_1e-5",
                ["--aggregator", "weighted", "--fedproxloss_mu", "1e-5"],
                "aggregator=weighted fedproxloss_mu=1e-5",
            )
            add(
                "fedprox_mu_1e-4",
                ["--aggregator", "weighted", "--fedproxloss_mu", "1e-4"],
                "aggregator=weighted fedproxloss_mu=1e-4",
            )

    if can_mutate("aggregation_epochs") and supports_flag(help_text, "--aggregation_epochs"):
        default = int(defaults.get("aggregation_epochs") or 4)
        for value in [1, 2, 3, 4, 6, 8]:
            if value != default:
                add(f"aggregation_epochs_{value}", ["--aggregation_epochs", str(value)], f"aggregation_epochs={value}")

    if can_mutate("local_train_steps") and supports_flag(help_text, "--local_train_steps"):
        for value in [50, 100, 200, 400]:
            add(f"local_train_steps_{value}", ["--local_train_steps", str(value)], f"local_train_steps={value}")

    if can_mutate("lr") and supports_flag(help_text, "--lr"):
        default = float(defaults.get("lr") or 0.05)
        for value in [default / 4, default / 2, default * 2, default * 4]:
            value_text = f"{value:.6g}"
            add(f"lr_{value_text.replace('.', 'p').replace('-', 'm')}", ["--lr", value_text], f"lr={value_text}")

    if can_mutate("momentum") and supports_flag(help_text, "--momentum"):
        for value in [0.0, 0.5, 0.8, 0.95]:
            add(
                f"momentum_{str(value).replace('.', 'p')}",
                ["--momentum", str(value)],
                f"momentum={value}",
            )

    if can_mutate("weight_decay") and supports_flag(help_text, "--weight_decay"):
        for value in ["1e-5", "1e-4", "5e-4"]:
            add(f"weight_decay_{value.replace('-', 'm')}", ["--weight_decay", value], f"weight_decay={value}")

    if can_mutate("batch_size") and supports_flag(help_text, "--batch_size"):
        default = int(defaults.get("batch_size") or 16)
        values = [max(1, default // 2), default * 2, default * 4, max(1, default // 4), 24, 40, 64, 96, 128, 256]
        for value in values:
            if value != default:
                add(f"batch_size_{value}", ["--batch_size", str(value)], f"batch_size={value}")

    if can_mutate("epochs") and supports_flag(help_text, "--epochs"):
        for value in [1, 2, 3, 4, 5]:
            add(f"epochs_{value}", ["--epochs", str(value)], f"epochs={value}")

    if can_mutate("num_workers") and supports_flag(help_text, "--num_workers"):
        for value in [0, 1, 2, 4]:
            add(f"num_workers_{value}", ["--num_workers", str(value)], f"num_workers={value}")

    if supports_flag(help_text, "--client_memory_gc_rounds"):
        add("client_memory_gc_1", ["--client_memory_gc_rounds", "1"], "client_memory_gc_rounds=1")

    if not candidates:
        add("rerun", [], "repeat baseline command to test campaign plumbing")

    if max_candidates is not None:
        return candidates[:max_candidates]

    def uncapped() -> Iterable[JobRun]:
        for template in candidates:
            yield JobRun(name=template.name, args=list(template.args), description=template.description)

        idx = 1
        batch_default = int(defaults.get("batch_size") or 16)
        while True:
            generated = False
            if can_mutate("batch_size") and supports_flag(help_text, "--batch_size"):
                # Walk a broad conservative range before repeats so uncapped
                # campaigns keep doing comparable, reviewable work for hours.
                value = 1 + ((batch_default + idx * 7) % 512)
                candidate_args = ["--batch_size", str(value)]
                candidate = make_candidate(
                    f"batch_size_auto_{value}",
                    candidate_args,
                    f"batch_size={value}",
                )
                if value != batch_default and candidate is not None:
                    yield candidate
                    generated = True

            if not generated and can_mutate("aggregation_epochs") and supports_flag(help_text, "--aggregation_epochs"):
                value = 1 + ((idx - 1) % 8)
                candidate_args = ["--aggregation_epochs", str(value)]
                candidate = make_candidate(
                    f"aggregation_epochs_auto_{value}",
                    candidate_args,
                    f"aggregation_epochs={value}",
                )
                if candidate is not None:
                    yield candidate
                    generated = True

            if not generated and can_mutate("lr") and supports_flag(help_text, "--lr"):
                value = 10 ** (-4 + ((idx - 1) % 25) / 10)
                value_text = f"{value:.6g}"
                candidate_args = ["--lr", value_text]
                candidate = make_candidate(
                    f"lr_auto_{value_text.replace('.', 'p').replace('-', 'm')}",
                    candidate_args,
                    f"lr={value_text}",
                )
                if candidate is not None:
                    yield candidate
                    generated = True

            if not generated and can_mutate("epochs") and supports_flag(help_text, "--epochs"):
                value = 1 + ((idx - 1) % 20)
                candidate_args = ["--epochs", str(value)]
                candidate = make_candidate(f"epochs_auto_{value}", candidate_args, f"epochs={value}")
                if candidate is not None:
                    yield candidate
                    generated = True

            if not generated and can_mutate("num_workers") and supports_flag(help_text, "--num_workers"):
                value = (idx - 1) % 9
                candidate_args = ["--num_workers", str(value)]
                candidate = make_candidate(f"num_workers_auto_{value}", candidate_args, f"num_workers={value}")
                if candidate is not None:
                    yield candidate
                    generated = True

            if not generated:
                template = candidates[(idx - 1) % len(candidates)]
                cycle = (idx - 1) // len(candidates) + 2
                yield JobRun(
                    name=f"{template.name}_repeat_{cycle:04d}",
                    args=list(template.args),
                    description=f"{template.description}; repeat_cycle={cycle}",
                )
            idx += 1

    return uncapped()


def imported_job_names(config: Dict[str, Any]) -> List[str]:
    names = []
    job_config = config.get("job", {})
    if not isinstance(job_config, dict):
        return names
    for key in ("recipe_args", "fed_job_args"):
        arguments = job_config.get(key)
        if not isinstance(arguments, dict):
            continue
        name = arguments.get("name")
        value = name.get("value") if isinstance(name, dict) else name
        if isinstance(name, dict) and name.get("confidence") != "high":
            continue
        if isinstance(value, str) and value and value not in names:
            names.append(value)
    discovered_name = config.get("artifacts", {}).get("simulator_result_name")
    if isinstance(discovered_name, str) and discovered_name and discovered_name not in names:
        names.append(discovered_name)
    return names


def simulator_workspace_root(config: Dict[str, Any], cwd: Path) -> Path:
    discovered = config.get("environment", {}).get("discovered", {})
    if isinstance(discovered, dict):
        arguments = discovered.get("args", {})
        workspace_arg = arguments.get("workspace_root") if isinstance(arguments, dict) else None
        if isinstance(workspace_arg, dict):
            value = workspace_arg.get("value")
            if workspace_arg.get("confidence") == "high" and isinstance(value, str) and value:
                root = Path(value).expanduser()
                return (root if root.is_absolute() else cwd / root).resolve()
    return (Path(tempfile.gettempdir()) / "nvflare" / "simulation").resolve()


def simulator_run_name(candidate_name: str, cwd: Path) -> str:
    campaign_namespace = sha256_bytes(str(cwd.resolve()).encode("utf-8"))[:12]
    return f"autofl_{campaign_namespace}_{candidate_name}"


def expected_simulator_roots(
    config: Dict[str, Any],
    injected_name: Optional[str],
    cwd: Path,
    simulator_base: Optional[Path] = None,
) -> List[Path]:
    names = ([injected_name] if injected_name else []) + imported_job_names(config)
    roots = []
    simulator_base = (simulator_base or simulator_workspace_root(config, cwd)).resolve()
    for name in names:
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"unsafe simulator job name: {name!r}")
        root = (simulator_base / name).resolve()
        if root.parent != simulator_base:
            raise ValueError(f"unsafe simulator result root: {root}")
        if root not in roots:
            roots.append(root)
    return roots


@contextmanager
def locked_campaign_workspace(workspace: Path, action: str) -> Iterator[None]:
    """Serialize lifecycle actions that share source, ledgers, and campaign state."""

    lock_path = workspace / CAMPAIGN_LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fallback = fcntl is None
    descriptor = None
    acquired = False
    fallback_created = False
    try:
        if fallback:
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                acquired = True
                fallback_created = True
            except FileExistsError as e:
                raise RuntimeError(
                    f"Auto-FL campaign workspace is already in use: {workspace}; "
                    "wait for the active lifecycle action to finish, then retry"
                ) from e
        else:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError as e:
                raise RuntimeError(
                    f"Auto-FL campaign workspace is already in use: {workspace}; "
                    "wait for the active lifecycle action to finish, then retry"
                ) from e
            os.ftruncate(descriptor, 0)
        os.write(
            descriptor,
            (json.dumps({"pid": os.getpid(), "action": action, "workspace": str(workspace.resolve())}) + "\n").encode(
                "utf-8"
            ),
        )
        yield
    finally:
        if descriptor is not None:
            if not fallback and acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        if fallback_created:
            lock_path.unlink(missing_ok=True)


def validated_printed_simulator_root(result_dir: Optional[Path], simulator_base: Path) -> Optional[Path]:
    if result_dir is None or not result_dir.exists():
        return None
    resolved = result_dir.resolve()
    if resolved.parent != simulator_base.resolve():
        return None
    return resolved


def simulator_root_snapshot(simulator_base: Path) -> Dict[Path, int]:
    snapshot = {}
    if not simulator_base.exists():
        return snapshot
    for path in simulator_base.iterdir():
        if path.name.startswith(".") or path.is_symlink() or not path.is_dir():
            continue
        snapshot[path.resolve()] = path.stat().st_mtime_ns
    return snapshot


def changed_simulator_roots(simulator_base: Path, before: Dict[Path, int]) -> List[Path]:
    after = simulator_root_snapshot(simulator_base)
    return sorted(path for path, modified in after.items() if before.get(path) != modified)


def last_json_object_line(text: str) -> Optional[Dict[str, Any]]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def probe_simulator_workspace_override_support(
    python: str, cwd: Path, timeout: int = DEFAULT_WORKSPACE_OVERRIDE_PROBE_TIMEOUT
) -> Dict[str, Any]:
    """Ask the campaign interpreter whether the installed package supports the simulator workspace override.

    Returns ``supported`` as True/False when ``nvflare.recipe.sim_env`` is importable, and None when the
    probe is inconclusive (package missing, import error, timeout); ``version`` is "" when unknown.
    ``version`` is read from the imported package's ``__version__`` so it describes the same package the
    capability check (and the job) resolves; distribution metadata is only a fallback when the import fails.
    """

    script = (
        "import json\n"
        "try:\n"
        "    import nvflare\n"
        "    version = str(getattr(nvflare, '__version__', '') or '')\n"
        "except Exception:\n"
        "    version = ''\n"
        "if not version:\n"
        "    try:\n"
        "        from importlib import metadata\n"
        "        version = metadata.version('nvflare')\n"
        "    except Exception:\n"
        "        version = ''\n"
        "try:\n"
        "    from nvflare.recipe import sim_env\n"
        f"    supported = getattr(sim_env, 'SIMULATOR_WORKSPACE_ROOT_ENV_VAR', '') == {SIMULATOR_WORKSPACE_ROOT_ENV_VAR!r}\n"
        "except Exception:\n"
        "    supported = None\n"
        "print(json.dumps({'version': version, 'supported': supported}))\n"
    )
    try:
        with tempfile.TemporaryDirectory(prefix="nvflare-autofl-probe-") as probe_dir:
            rc, stdout, _runtime, _command = run_python(
                python,
                ["-c", script],
                cwd=cwd,
                timeout=timeout,
                log_path=Path(probe_dir) / "probe.log",
                env=simulator_child_env(Path(probe_dir)),
                simulator_no_progress_timeout=0,
            )
        if rc != 0:
            return {"version": "", "supported": None}
        payload = last_json_object_line(stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"version": "", "supported": None}
    if not isinstance(payload, dict):
        return {"version": "", "supported": None}
    version = payload.get("version")
    supported = payload.get("supported")
    return {
        "version": version.strip() if isinstance(version, str) else "",
        "supported": supported if isinstance(supported, bool) else None,
    }


def nvflare_version_predates_workspace_override(version: str) -> bool:
    match = re.match(r"(\d+)\.(\d+)", version or "")
    if not match:
        return False
    minimum = tuple(int(part) for part in SIMULATOR_WORKSPACE_OVERRIDE_MIN_NVFLARE_VERSION.split(".")[:2])
    return (int(match.group(1)), int(match.group(2))) < minimum


def discovered_environment_name(config: Dict[str, Any]) -> str:
    environment = config.get("environment", {})
    discovered = environment.get("discovered", {}) if isinstance(environment, dict) else {}
    name = discovered.get("name") if isinstance(discovered, dict) else None
    return name if isinstance(name, str) else ""


def unresolved_result_dir_failure_reason(python: str, cwd: Path, config: Dict[str, Any]) -> str:
    generic_reason = (
        "job exited successfully but no deterministic NVFlare result directory was resolved; "
        "expose a literal job name, support --name, or print the direct simulator result directory"
    )
    # Only the recipe SimEnv honors the workspace override; FedJob.simulator_run and other job
    # surfaces ignore it on every release, so the upgrade advice below would misdirect them.
    if discovered_environment_name(config) != "SimEnv":
        return generic_reason
    probe = probe_simulator_workspace_override_support(python, cwd)
    supported = probe["supported"]
    version = probe["version"]
    outdated = supported is False or (supported is None and nvflare_version_predates_workspace_override(version))
    if outdated:
        return (
            f"job exited successfully but the installed nvflare ({version or 'unknown version'}) does not honor "
            f"{SIMULATOR_WORKSPACE_ROOT_ENV_VAR}, so simulator results were written outside the isolated trial "
            f"workspace; upgrade to nvflare>={SIMULATOR_WORKSPACE_OVERRIDE_MIN_NVFLARE_VERSION}"
        )
    return generic_reason


def simulator_env_passthrough_names(config: Dict[str, Any]) -> List[str]:
    environment = config.get("environment", {})
    if not isinstance(environment, dict):
        return []
    values = environment.get(SIMULATOR_ENV_PASSTHROUGH_CONFIG_KEY, []) or []
    if not isinstance(values, list):
        raise ValueError(f"autofl.yaml environment.{SIMULATOR_ENV_PASSTHROUGH_CONFIG_KEY} must be a list")
    names = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"autofl.yaml environment.{SIMULATOR_ENV_PASSTHROUGH_CONFIG_KEY} must contain only names")
        name = value.strip()
        if not ENV_VAR_NAME_RE.fullmatch(name):
            raise ValueError(
                f"autofl.yaml environment.{SIMULATOR_ENV_PASSTHROUGH_CONFIG_KEY} contains invalid name: {value!r}"
            )
        if name not in names:
            names.append(name)
    return names


def simulator_child_env(simulator_base: Path, extra_names: Sequence[str] = ()) -> Dict[str, str]:
    run_env: Dict[str, str] = {}
    for name in (*SIMULATOR_ENV_ALLOWLIST, *extra_names):
        value = os.environ.get(name)
        if value is not None:
            run_env[name] = value
    run_env[SIMULATOR_WORKSPACE_ROOT_ENV_VAR] = str(simulator_base)
    return run_env


def run_job(
    run_def: JobRun,
    *,
    python: str,
    job: Path,
    cwd: Path,
    help_text: str,
    fixed_args: List[str],
    base_args: List[str],
    output_root: Path,
    timeout: int,
    simulator_no_progress_timeout: int,
    metrics: Sequence[str],
    config: Dict[str, Any],
) -> RunRecord:
    baseline_run = run_def.status == "baseline"
    log_path = output_root / run_def.name / "run.log"
    run_name = simulator_run_name(run_def.name, cwd)
    name_args = ["--name", run_name] if supports_flag(help_text, "--name") else []
    arguments = [str(job), *fixed_args, *base_args, *name_args, *run_def.args]
    with tempfile.TemporaryDirectory(prefix="nvflare-autofl-sim-") as trial_workspace:
        simulator_base = Path(trial_workspace).resolve()
        simulator_roots = expected_simulator_roots(
            config, run_name if name_args else None, cwd, simulator_base=simulator_base
        )
        run_env = simulator_child_env(simulator_base, simulator_env_passthrough_names(config))
        unnamed_root_snapshot = simulator_root_snapshot(simulator_base) if not simulator_roots else {}
        rc, stdout, runtime, command = run_python(
            python,
            arguments,
            cwd=cwd,
            timeout=timeout,
            log_path=log_path,
            env=run_env,
            simulator_stall_roots=simulator_roots,
            simulator_no_progress_timeout=simulator_no_progress_timeout,
        )
        run_def.command = command
        run_def.runtime_seconds = runtime
        printed_result_dir, sandbox_socket_failure = scan_run_log(log_path, cwd)
        existing_roots = [root.resolve() for root in simulator_roots if root.exists()]
        result_dir = printed_result_dir if printed_result_dir in existing_roots else None
        injected_root = (simulator_base / run_name).resolve() if name_args else None
        if result_dir is None and injected_root in existing_roots:
            result_dir = injected_root
        if result_dir is None and len(existing_roots) == 1:
            result_dir = existing_roots[0]
        if result_dir is None and not simulator_roots:
            result_dir = validated_printed_simulator_root(printed_result_dir, simulator_base)
            if result_dir is None:
                changed_roots = changed_simulator_roots(simulator_base, unnamed_root_snapshot)
                if len(changed_roots) == 1:
                    result_dir = changed_roots[0]
            if result_dir is not None:
                config.setdefault("artifacts", {})["simulator_result_name"] = result_dir.name
        artifact_dir = collect_artifacts(result_dir, output_root, run_def.name, log_path)
    run_def.artifacts = str(artifact_dir)

    evidence = None
    if rc != 0:
        if sandbox_socket_failure:
            run_def.status = INFRASTRUCTURE_RETRY
            run_def.failure_reason = (
                "sandbox/socket permission failure; simulation requires the human-approved runner execution "
                "scope established during campaign setup"
            )
        elif is_nvflare_simulator_stall(stdout):
            run_def.status = "crash"
            run_def.failure_reason = (
                "nvflare simulator watchdog detected a child connection/no-progress stall; "
                "candidate killed and campaign continued"
            )
        else:
            run_def.status = "crash"
            run_def.failure_reason = f"exit_code={rc}"
    elif result_dir is None:
        run_def.status = "crash"
        run_def.failure_reason = unresolved_result_dir_failure_reason(python, cwd, config)
    else:
        artifact_root = artifact_dir.parent
        evidence = extract_metric_evidence(artifact_root, metrics)
        if evidence is None:
            run_def.status = "crash"
            run_def.failure_reason = f"matching metric not found; {metric_search_description(artifact_root, metrics)}"
        else:
            run_def.score = evidence.score

    record_status = "baseline" if baseline_run and run_def.status == "crash" else run_def.status
    return RunRecord(
        status=record_status,
        name=run_def.name,
        score=run_def.score,
        runtime_seconds=run_def.runtime_seconds,
        changed_files="none",
        diff_summary=run_def.description,
        run_command=shlex.join(command),
        artifacts=run_def.artifacts,
        failure_reason=run_def.failure_reason,
        metric_name=evidence.metric_name if evidence else "",
        metric_source=evidence.source if evidence else "",
        metric_artifact=evidence.artifact if evidence else "",
    )


def write_results(path: Path, records: List[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS, delimiter="\t")
    writer.writeheader()
    for record in records:
        score = parse_finite_metric_value(record.score) if record.score is not None else None
        if record.score is not None and score is None:
            raise ValueError(f"record {record.name!r} has a non-finite score")
        writer.writerow(
            {
                "status": record.status,
                "name": record.name,
                "score": "" if score is None else f"{score:.6f}",
                "runtime_seconds": f"{record.runtime_seconds:.3f}",
                "changed_files": record.changed_files,
                "diff_summary": record.diff_summary,
                "run_command": record.run_command,
                "artifacts": record.artifacts,
                "failure_reason": record.failure_reason,
                "candidate_manifest": record.candidate_manifest,
                "base_candidate": record.base_candidate,
                "patch_sha256": record.patch_sha256,
                "metric_name": record.metric_name,
                "metric_source": record.metric_source,
                "metric_artifact": record.metric_artifact,
                "candidate_kind": record.candidate_kind,
                "algorithm_family": record.algorithm_family,
                "literature_event_id": record.literature_event_id,
            }
        )
    atomic_write_bytes(path, stream.getvalue().encode("utf-8"))


def load_results(path: Path) -> List[RunRecord]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            score_text = row.get("score", "")
            records.append(
                RunRecord(
                    status=row.get("status", ""),
                    name=row.get("name", ""),
                    score=parse_finite_metric_value(score_text) if score_text else None,
                    runtime_seconds=float(row.get("runtime_seconds") or 0.0),
                    changed_files=row.get("changed_files", ""),
                    diff_summary=row.get("diff_summary", ""),
                    run_command=row.get("run_command", ""),
                    artifacts=row.get("artifacts", ""),
                    failure_reason=row.get("failure_reason", ""),
                    candidate_manifest=row.get("candidate_manifest", ""),
                    base_candidate=row.get("base_candidate", ""),
                    patch_sha256=row.get("patch_sha256", ""),
                    metric_name=row.get("metric_name", ""),
                    metric_source=row.get("metric_source", ""),
                    metric_artifact=row.get("metric_artifact", ""),
                    candidate_kind=row.get("candidate_kind", "") or "",
                    algorithm_family=row.get("algorithm_family", "") or "",
                    literature_event_id=row.get("literature_event_id", "") or "",
                )
            )
    return records


better = load_campaign_guard().better


def write_state(
    path: Path,
    results_path: Path,
    records: List[RunRecord],
    max_candidates: Optional[int],
    *,
    mode: str = "max",
    stop_files: Optional[List[str]] = None,
    plateau_threshold: Optional[int] = None,
    plateau_min_delta: Optional[float] = None,
    hard_crash_threshold: Optional[int] = None,
    exploration_batch_size: Optional[int] = None,
    family_repeat_limit: Optional[int] = None,
    pending_manifest_count: int = 0,
    abandoned_candidate_count: int = 0,
    persist: bool = True,
) -> Dict[str, Any]:
    guard = load_campaign_guard()
    plateau_threshold = guard.DEFAULT_PLATEAU_THRESHOLD if plateau_threshold is None else plateau_threshold
    plateau_min_delta = guard.DEFAULT_MIN_DELTA if plateau_min_delta is None else plateau_min_delta
    hard_crash_threshold = guard.DEFAULT_HARD_CRASH_THRESHOLD if hard_crash_threshold is None else hard_crash_threshold
    exploration_batch_size = (
        guard.DEFAULT_EXPLORATION_BATCH_SIZE if exploration_batch_size is None else exploration_batch_size
    )
    family_repeat_limit = guard.DEFAULT_FAMILY_REPEAT_LIMIT if family_repeat_limit is None else family_repeat_limit
    state = guard.guard_state(
        results_path,
        max_candidates=max_candidates,
        stop_files=stop_files,
        plateau_threshold=plateau_threshold,
        min_delta=plateau_min_delta,
        hard_crash_threshold=hard_crash_threshold,
        mode=mode,
        pending_manifest_count=pending_manifest_count,
        exploration_batch_size=exploration_batch_size,
        family_repeat_limit=family_repeat_limit,
    )
    # Abandoned manifests are workspace-derived; the ledger-only guard cannot count them.
    state["abandoned_candidates"] = abandoned_candidate_count
    state["accounting_instruction"] = ACCOUNTING_INSTRUCTION
    if records and records[-1].status == INFRASTRUCTURE_RETRY:
        attempts = len(
            [
                record
                for record in records
                if record.status in {"candidate", "keep", "discard", "crash"} and not is_baseline_record(record)
            ]
        )
        state.update(
            {
                "candidate_attempts": attempts,
                "remaining_candidates": max(0, max_candidates - attempts) if max_candidates is not None else None,
                "decision": "retry_infrastructure",
                "reason": "infrastructure_retry",
                "next_action": SIMULATION_APPROVAL_ACTION,
                "final_response_allowed": False,
                "agent_instruction": simulation_approval_instruction(),
            }
        )
    if persist:
        write_json(path, state)
    return state


def write_state_if_changed(path: Path, state: Dict[str, Any]) -> bool:
    previous = read_json(path) if path.exists() else {}
    previous_comparable = dict(previous)
    state_comparable = dict(state)
    previous_comparable.pop("updated_at", None)
    state_comparable.pop("updated_at", None)
    if previous_comparable == state_comparable:
        if previous.get("updated_at"):
            state["updated_at"] = previous["updated_at"]
        return False
    write_json(path, state)
    return True


def load_progress_plotter():
    return load_sibling_module("plot_progress.py", "nvflare_autofl_plot_progress")


def write_progress_fallback(path: Path, records: List[RunRecord], metric_label: str, mode: str = "max") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        path.write_text("Pillow is not installed; progress image unavailable.\n", encoding="utf-8")
        return

    width, height = 1000, 560
    margin = 70
    scores = [r.score for r in records if r.score is not None]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (margin, 24),
        f"Auto-FL Progress ({metric_label}): {len(records)} rows, {len(scores)} scored",
        fill="black",
        font=font,
    )
    draw.line((margin, height - margin, width - margin, height - margin), fill=(80, 80, 80), width=2)
    draw.line((margin, margin, margin, height - margin), fill=(80, 80, 80), width=2)

    if scores:
        lo, hi = min(scores), max(scores)
        if lo == hi:
            lo -= 1.0
            hi += 1.0
        span = hi - lo
        running_best: Optional[float] = None
        last_point: Optional[Tuple[float, float]] = None
        denom = max(1, len(records) - 1)
        for idx, record in enumerate(records):
            if record.score is None:
                continue
            x = margin + (width - 2 * margin) * idx / denom
            y = height - margin - (height - 2 * margin) * (record.score - lo) / span
            color = (40, 160, 90) if record.status in {"baseline", "keep"} else (150, 150, 150)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="black")
            draw.text((x + 6, y - 14), f"{record.name}: {record.score:.3f}", fill=color, font=font)
            if better(record.score, running_best, mode):
                running_best = record.score
            if running_best == record.score:
                if last_point:
                    draw.line((last_point[0], last_point[1], x, y), fill=(40, 160, 90), width=2)
                last_point = (x, y)
    image.save(path)


def write_progress(path: Path, records: List[RunRecord], metric_label: str, mode: str = "max") -> None:
    plotter = load_progress_plotter()
    try:
        plotter.plot_progress(records, path, metric_label, mode=mode)
    except (plotter.NoScoredResultsError, plotter.PlotDependencyError):
        write_progress_fallback(path, records, metric_label, mode)


def write_report(path: Path, config: Dict[str, Any], records: List[RunRecord], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = objective_mode(config, args.metric)
    best = None
    for record in records:
        if record.status in {"baseline", "keep"} and better(record.score, best.score if best else None, mode):
            best = record
    candidate_budget = (
        str(args.max_candidates) if args.max_candidates is not None else "uncapped; runs until manual interruption"
    )
    objective = objective_contract(config, args.metric)
    lines = [
        "# Auto-FL Report",
        "",
        f"Objective: optimize `{objective['optimization_metric']}` ({mode}) in `{args.target_env}`.",
        f"Requested metric: `{objective['requested_metric']}`.",
        f"Metric source: `{metric_source(config)}`.",
        f"Metric extraction order: `{', '.join(objective['metric_extraction_order'])}`.",
        f"Candidate budget: `{candidate_budget}`.",
        f"Config: `{args.autofl_yaml}`.",
        f"Fixed budget: `{json.dumps(config.get('budget', {}).get('fixed_training_budget', {}), sort_keys=True)}`.",
        "",
        "## Leaderboard",
        "",
        "| Status | Name | Score | Changed files | Manifest | Artifacts | Notes |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for record in records:
        score = "" if record.score is None else f"{record.score:.6f}"
        lines.append(
            f"| {record.status} | {record.name} | {score} | `{record.changed_files}` | "
            f"`{record.candidate_manifest}` | `{record.artifacts}` | {record.diff_summary} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Ledger: `{args.results}`",
            f"- Progress plot: `{args.progress}`",
            f"- Campaign state: `{args.state}`",
            "",
            "## Outcome",
            "",
        ]
    )
    if best:
        lines.append(f"Best retained run: `{best.name}` with `{objective['optimization_metric']}={best.score:.6f}`.")
    else:
        lines.append("No scored run was retained.")
    atomic_write_text(path, "\n".join(lines) + "\n")


def candidate_attempts(records: List[RunRecord]) -> int:
    return len(
        [
            record
            for record in records
            if record.status in {"candidate", "keep", "discard", "crash"} and not is_baseline_record(record)
        ]
    )


def is_baseline_record(record: RunRecord) -> bool:
    return record.status == "baseline"


def campaign_summary(
    autofl_yaml: Path,
    results: Path,
    state: Path,
    progress: Path,
    report: Path,
    records: List[RunRecord],
    state_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "autofl_yaml": str(autofl_yaml.resolve()),
        "results": str(results.resolve()),
        "state": str(state.resolve()),
        "progress": str(progress.resolve()),
        "report": str(report.resolve()),
        "candidate_attempts": candidate_attempts(records),
    }
    if state_payload:
        for key in [
            "decision",
            "reason",
            "next_action",
            "final_response_allowed",
            "candidate_cap",
            "candidate_cap_source",
            "remaining_candidates",
            "baseline_status",
            "baseline_score",
            "improvement",
            "abandoned_candidates",
            "agent_instruction",
            "required_exploration",
        ]:
            if key in state_payload:
                payload[key] = state_payload[key]
    return payload


def campaign_paths(args: argparse.Namespace, job: Path) -> Dict[str, Path]:
    workspace = job.parent
    return {
        "workspace": workspace,
        "autofl_yaml": resolve_output_path(workspace, args.autofl_yaml),
        "results": resolve_output_path(workspace, args.results),
        "state": resolve_output_path(workspace, args.state),
        "progress": resolve_output_path(workspace, args.progress),
        "report": resolve_output_path(workspace, args.report),
        "output_root": resolve_output_path(workspace, args.output_root),
        "snapshot_root": workspace / BEST_SNAPSHOT_ROOT,
    }


CAMPAIGN_SETTING_NAMES = (
    "metric",
    "mode",
    "target_env",
    "max_candidates",
    "autofl_yaml",
    "results",
    "state",
    "progress",
    "report",
    "output_root",
    "plateau_threshold",
    "plateau_min_delta",
    "hard_crash_threshold",
    "exploration_batch_size",
    "family_repeat_limit",
    "stop_file",
    "base_args",
    "timeout",
    "simulator_no_progress_timeout",
    "python",
)

MUTABLE_CAMPAIGN_SETTING_NAMES = {
    "max_candidates",
    "plateau_threshold",
    "plateau_min_delta",
    "hard_crash_threshold",
    "exploration_batch_size",
    "family_repeat_limit",
    "stop_file",
    "timeout",
    "simulator_no_progress_timeout",
}


def campaign_settings(args: argparse.Namespace) -> Dict[str, Any]:
    return {name: getattr(args, name) for name in CAMPAIGN_SETTING_NAMES}


def cap_change_requires_approval(previous: Optional[int], requested: Optional[int]) -> bool:
    return previous is not None and (requested is None or requested > previous)


def restore_campaign_settings(args: argparse.Namespace, metadata: Dict[str, Any]) -> bool:
    """Restore persisted settings onto args; persist explicit mutable changes.

    Returns True when a mutable-settings change was persisted to campaign.json, so
    callers can refresh the authoritative campaign state under the new settings
    before any preflight gate reads it.
    """
    settings = metadata.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("campaign metadata is missing settings")
    if settings.get("prefer_synthetic"):
        # Pre-removal campaigns scored their baseline and prior candidates on injected
        # synthetic data; new runs use real data, so ledger comparisons cross a data regime.
        print(
            "Warning: prior scores in this campaign were computed on synthetic data "
            "(legacy prefer_synthetic setting); new runs use the job's real data, so ledger "
            "comparisons mix data regimes. Consider re-initializing the campaign.",
            file=sys.stderr,
        )
    persisted_mode = settings.get("mode", "max")
    if persisted_mode not in {"min", "max"}:
        raise ValueError(f"campaign metadata has invalid objective mode {persisted_mode!r}")
    if (
        persisted_mode == "min"
        and metadata.get("metric_direction_contract_version") != METRIC_DIRECTION_CONTRACT_VERSION
    ):
        raise ValueError(
            "campaign was initialized with legacy mode='min' without native metric-direction provenance. "
            "Delete the campaign's .nvflare/autofl directory or start in a fresh workspace, then initialize again."
        )
    explicit = getattr(args, "_explicit_settings", set())
    changed = False
    cap_confirmation_valid = False
    for name in CAMPAIGN_SETTING_NAMES:
        if name not in settings:
            continue
        if name in explicit:
            requested = getattr(args, name)
            if name in MUTABLE_CAMPAIGN_SETTING_NAMES:
                if settings.get(name) != requested:
                    if name == "max_candidates":
                        previous = settings.get(name)
                        approval_required = cap_change_requires_approval(previous, requested)
                        if approval_required and not args.confirm_user_approved_cap_change:
                            raise ValueError(
                                "increasing the candidate cap or making a finite campaign uncapped requires "
                                "explicit user approval and --confirm-user-approved-cap-change"
                            )
                        if args.confirm_user_approved_cap_change and not approval_required:
                            raise ValueError(
                                "--confirm-user-approved-cap-change is only valid for a candidate-cap increase "
                                "or a finite-to-uncapped change"
                            )
                        cap_confirmation_valid = approval_required
                        # Audit trail: mid-campaign budget changes must stay detectable by external judges.
                        metadata.setdefault("cap_changes", []).append(
                            {
                                "changed_at": utc_now(),
                                "old": previous,
                                "new": requested,
                                "source": "uncapped" if requested is None else "explicit",
                                "user_approved": approval_required,
                            }
                        )
                    settings[name] = requested
                    changed = True
                elif name == "max_candidates" and args.confirm_user_approved_cap_change:
                    # Retrying an already-applied approved command is an idempotent success.
                    cap_confirmation_valid = True
                continue
            if requested != settings[name]:
                raise ValueError(
                    f"campaign setting {name} is immutable after initialization: "
                    f"configured={settings[name]!r}, requested={requested!r}"
                )
        setattr(args, name, settings[name])
    if args.confirm_user_approved_cap_change and not cap_confirmation_valid:
        raise ValueError(
            "--confirm-user-approved-cap-change requires an explicit candidate-cap increase or finite-to-uncapped "
            "change"
        )
    if changed:
        metadata["updated_at"] = utc_now()
        workspace_value = metadata.get("workspace_root")
        if not isinstance(workspace_value, str) or not workspace_value:
            raise ValueError("campaign metadata is missing workspace_root")
        workspace = Path(workspace_value)
        write_json(campaign_metadata_path(workspace), metadata)
    return changed


def campaign_timeout(args: argparse.Namespace, schema: Dict[str, Any]) -> Tuple[int, int]:
    budget = comparison_budget(schema)
    budget_timeout = parse_timeout_setting(budget.get("run_timeout_seconds"), "run_timeout_seconds")
    timeout = max(args.timeout, budget_timeout) if budget_timeout is not None else args.timeout
    budget_no_progress_timeout = parse_timeout_setting(
        budget.get("simulator_no_progress_timeout_seconds"), "simulator_no_progress_timeout_seconds"
    )
    no_progress_timeout = (
        budget_no_progress_timeout if budget_no_progress_timeout is not None else args.simulator_no_progress_timeout
    )
    return timeout, no_progress_timeout


def parse_timeout_setting(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a non-negative integer") from e
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def import_job_config(
    args: argparse.Namespace,
    job: Path,
    output: Path,
    log_path: Path,
    timeout: int,
    *,
    persist: bool = True,
) -> Dict[str, Any]:
    del timeout
    importer = load_job_importer()
    config = importer.import_job_to_autofl_config(
        str(job),
        workspace_root=str(job.parent),
        metric=args.metric,
        target_env=args.target_env,
        max_candidates=args.max_candidates,
        job_args=shlex.split(args.base_args),
    )
    if persist:
        write_import_artifacts(config, job, output, log_path)
    return config


def write_import_artifacts(config: Dict[str, Any], job: Path, output: Path, log_path: Path) -> None:
    importer = load_job_importer()
    atomic_write_bytes(output, importer.dump_autofl_yaml(config).encode("utf-8"))
    atomic_write_bytes(
        log_path,
        (
            json.dumps(
                {
                    "importer": str(Path(importer.__file__).resolve()),
                    "job": str(job.resolve()),
                    "output": str(output.resolve()),
                    "source_sha256": config.get("import", {}).get("source_sha256"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )


def mutation_schema_declares_metric_bridge(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    objective = objective_contract(config, None)
    if not requested_metric_differs_from_job(objective):
        return True
    schema_objective = schema.get("objective")
    if not isinstance(schema_objective, dict):
        return False
    schema_requested = schema_objective.get("requested_metric") or schema_objective.get("metric")
    return schema_requested == objective["requested_metric"] and bool(schema_objective.get("optimization_metric"))


def campaign_admission_errors(config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> List[str]:
    errors = []
    support = config.get("import", {}).get("support", {})
    if not isinstance(support, dict) or support.get("status") != "supported":
        reason = support.get("reason") if isinstance(support, dict) else None
        errors.append(f"job surface is not deterministically supported{f': {reason}' if reason else ''}")
    fixed_budget = config.get("budget", {}).get("fixed_training_budget")
    if not isinstance(fixed_budget, dict) or not fixed_budget:
        errors.append("fixed comparison budget is unresolved")
    unresolved = config.get("unresolved", [])
    if isinstance(unresolved, list):
        critical_issues: Dict[str, List[str]] = {}
        for item in unresolved:
            field = item.get("field", "") if isinstance(item, dict) else ""
            if field in {"objective.metric", "objective.job_key_metric", "objective.mode"} or field.startswith(
                "budget.fixed_training_budget"
            ):
                reasons = critical_issues.setdefault(field, [])
                reason = item.get("reason", "") if isinstance(item, dict) else ""
                if reason:
                    reasons.append(reason)
        if critical_issues:
            details = []
            for field, reasons in sorted(critical_issues.items()):
                unique_reasons = sorted(set(reasons))
                details.append(f"{field} ({'; '.join(unique_reasons)})" if unique_reasons else field)
            errors.append(f"safety-critical fields are unresolved: {', '.join(details)}")
    objective = objective_contract(config, None)
    schema = schema or {}
    if not mutation_schema_declares_metric_bridge(config, schema):
        errors.append(
            "AUTOFL_METRIC_NOT_DECLARED: the requested metric differs from job.py key_metric; "
            "declare the requested and optimization metrics in mutation_schema.yaml"
        )
    importer = load_job_importer()
    if (
        objective["mode"] == "max"
        and objective.get("mode_contract_source") == "core_default"
        and importer.likely_lower_is_better_metric(objective["optimization_metric"])
    ):
        guidance = (
            "declare objective.mode='min' in the mutation_schema.yaml metric bridge"
            if requested_metric_differs_from_job(objective)
            else "set key_metric_mode='min' in job.py"
        )
        errors.append(
            "AUTOFL_METRIC_DIRECTION_CONFLICT: the lower-is-better metric "
            f"{objective['optimization_metric']!r} would use NVFlare's implicit 'max' default; "
            f"{guidance} and initialize again"
        )
    return errors


def best_retained_record(records: Sequence[RunRecord], mode: str = "max") -> Optional[RunRecord]:
    best = None
    for record in records:
        if record.status in {"baseline", "keep"} and better(record.score, best.score if best else None, mode):
            best = record
    return best


def refresh_campaign_artifacts(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    config: Dict[str, Any],
    records: List[RunRecord],
    metadata: Dict[str, Any],
    *,
    pending_manifest: Optional[Path] = None,
    next_action: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    pending_manifests = pending_candidate_manifests(paths["workspace"])
    if pending_manifest is not None and pending_manifest not in pending_manifests:
        pending_manifests.append(pending_manifest)
    write_results(paths["results"], records)
    mode = objective_mode(config, args.metric)
    state = write_state(
        paths["state"],
        paths["results"],
        records,
        args.max_candidates,
        mode=mode,
        stop_files=resolve_stop_files(paths["workspace"], args.stop_file),
        plateau_threshold=args.plateau_threshold,
        plateau_min_delta=args.plateau_min_delta,
        hard_crash_threshold=args.hard_crash_threshold,
        exploration_batch_size=args.exploration_batch_size,
        family_repeat_limit=args.family_repeat_limit,
        pending_manifest_count=len(pending_manifests),
        abandoned_candidate_count=count_abandoned_candidates(paths["workspace"]),
    )
    if state.get("next_action") == "abandon_candidate":
        next_action = "abandon_candidate"
        reason = state.get("reason")
    elif pending_manifests and next_action is None:
        pending_status = read_json(pending_manifests[0]).get("status")
        next_action = "submit_candidate" if pending_status == "ready_for_external_execution" else "edit_candidate"
        reason = "pending_candidates"
    if not state.get("final_response_allowed") and next_action:
        state["next_action"] = next_action
        state["reason"] = reason or next_action
        state["agent_instruction"] = (
            simulation_approval_instruction()
            if next_action == SIMULATION_APPROVAL_ACTION
            else f"Do not produce a final answer. Execute `{next_action}` for this campaign."
        )
    state.update(
        {
            "best_candidate": metadata.get("best_candidate"),
            "best_source_sha256": metadata.get("best_source_sha256"),
            "pending_candidate_manifest": str(pending_manifests[0].resolve()) if pending_manifests else None,
        }
    )
    write_json(paths["state"], state)
    write_progress(paths["progress"], records, optimization_metric(config, args.metric), mode)
    write_report(paths["report"], config, records, args)
    return state


def print_campaign_result(
    paths: Dict[str, Path], records: List[RunRecord], state: Dict[str, Any], **extra: Any
) -> None:
    payload = campaign_summary(
        paths["autofl_yaml"],
        paths["results"],
        paths["state"],
        paths["progress"],
        paths["report"],
        records,
        state,
    )
    payload.update(extra)
    print(json.dumps(payload, indent=2, sort_keys=True))


def execute_sim_baseline(
    args: argparse.Namespace,
    job: Path,
    paths: Dict[str, Path],
    config: Dict[str, Any],
    schema: Dict[str, Any],
    name: str = "baseline",
) -> RunRecord:
    timeout, no_progress_timeout = campaign_timeout(args, schema)
    help_text = job_help(args.python, job, job.parent)
    fixed_args, base_args = build_campaign_args(
        config, args, help_text, schema, alias_groups=job_flag_alias_groups(job)
    )
    return run_job(
        JobRun(name=name, args=[], description="baseline", status="baseline"),
        python=args.python,
        job=job,
        cwd=job.parent,
        help_text=help_text,
        fixed_args=fixed_args,
        base_args=base_args,
        output_root=paths["output_root"],
        timeout=timeout,
        simulator_no_progress_timeout=no_progress_timeout,
        metrics=metric_extraction_order(config, args.metric),
        config=config,
    )


def prepare_initial_campaign(args: argparse.Namespace, job: Path) -> None:
    """Build and admit a new campaign contract without writing campaign artifacts."""

    if args.confirm_user_approved_cap_change:
        raise ValueError("--confirm-user-approved-cap-change is not valid for an initial campaign cap")
    paths = campaign_paths(args, job)
    schema = load_mutation_schema(job.parent)
    timeout, _ = campaign_timeout(args, schema)
    config = import_job_config(
        args,
        job,
        paths["autofl_yaml"],
        paths["output_root"] / "import.log",
        timeout,
        persist=False,
    )
    config = apply_metric_contract(config, args.metric, schema)
    args.metric = str(config.get("objective", {}).get("requested_metric") or config["objective"]["metric"])
    args.mode = objective_mode(config, args.metric)
    config = apply_mutation_schema_contract(config, schema, job.parent)
    config.setdefault("trust_contract", {})["allowed_create_patterns"] = list(ALLOWED_CREATE_PATTERNS)
    admission_errors = campaign_admission_errors(config, schema)
    if admission_errors:
        raise ValueError(
            "baseline execution is unsafe: "
            f"{'; '.join(admission_errors)}. Repair the job metric contract and initialize again."
        )
    # Surface --base-args conflicts with the admitted budget before any campaign file is written.
    build_campaign_args(
        config, args, job_help(args.python, job, job.parent), schema, alias_groups=job_flag_alias_groups(job)
    )
    args._initial_config = config
    args._initial_source_hashes = admitted_source_hashes(config, job.parent)
    args._initial_schema_sha256 = sha256_json(schema)


def admitted_initial_campaign(args: argparse.Namespace, job: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config = getattr(args, "_initial_config", None)
    if not isinstance(config, dict):
        raise ValueError("new campaign initialization was not admitted before acquiring the workspace lock")
    if getattr(args, "_initial_source_hashes", None) != admitted_source_hashes(config, job.parent):
        raise ValueError("campaign source changed after metric-contract admission; retry initialization")
    schema = load_mutation_schema(job.parent)
    if getattr(args, "_initial_schema_sha256", None) != sha256_json(schema):
        raise ValueError("mutation_schema.yaml changed after metric-contract admission; retry initialization")
    return config, schema


def initialize_campaign(args: argparse.Namespace, job: Path) -> int:
    workspace = job.parent
    metadata_path = campaign_metadata_path(workspace)
    if metadata_path.exists():
        metadata = load_campaign_metadata(workspace, job)
        settings_changed = restore_campaign_settings(args, metadata)
        paths = campaign_paths(args, job)
        if settings_changed:
            refresh_campaign_state(args, job, metadata, paths)
        records = load_results(paths["results"])
        if args.target_env == "sim" and not any(
            record.status == "baseline" and record.score is not None for record in records
        ):
            config = read_yaml(paths["autofl_yaml"])
            retry_number = sum(is_baseline_record(record) for record in records) + 1
            baseline = execute_sim_baseline(
                args,
                job,
                paths,
                config,
                load_mutation_schema(workspace),
                name=f"baseline_retry_{retry_number}",
            )
            write_yaml(paths["autofl_yaml"], config)
            records.append(baseline)
            metadata["best_score"] = baseline.score
            if baseline.score is not None:
                metadata["best_candidate"] = baseline.name
            metadata["updated_at"] = utc_now()
            write_json(metadata_path, metadata)
            next_action = (
                SIMULATION_APPROVAL_ACTION
                if baseline.status == INFRASTRUCTURE_RETRY
                else "propose_candidate" if baseline.score is not None else "repair_baseline"
            )
            reason = "infrastructure_retry" if baseline.status == INFRASTRUCTURE_RETRY else "baseline_retried"
            state = refresh_campaign_artifacts(
                args, paths, config, records, metadata, next_action=next_action, reason=reason
            )
            print_campaign_result(paths, records, state, initialized=False, baseline_retried=True)
            if baseline.status == INFRASTRUCTURE_RETRY:
                return 75
            return 0 if baseline.score is not None else 1
        print_campaign_result(paths, records, read_json(paths["state"]), initialized=False)
        return 0

    paths = campaign_paths(args, job)
    config, schema = admitted_initial_campaign(args, job)
    paths["output_root"].mkdir(parents=True, exist_ok=True)
    write_import_artifacts(config, job, paths["autofl_yaml"], paths["output_root"] / "import.log")
    snapshot_files = create_best_snapshot(workspace, config, paths["snapshot_root"])
    metadata = {
        "schema_version": CAMPAIGN_METADATA_SCHEMA_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "job": str(job.resolve()),
        "workspace_root": str(workspace.resolve()),
        "settings": campaign_settings(args),
        "best_candidate": "baseline",
        "best_score": None,
        "best_source_sha256": source_hash(snapshot_files),
        "fixed_budget_sha256": fixed_budget_hash(config),
        "metric_direction_contract_version": METRIC_DIRECTION_CONTRACT_VERSION,
    }
    write_json(metadata_path, metadata)

    records: List[RunRecord] = []
    if args.target_env == "sim":
        baseline = execute_sim_baseline(args, job, paths, config, schema)
        write_yaml(paths["autofl_yaml"], config)
        records.append(baseline)
        metadata["best_score"] = baseline.score
        metadata["updated_at"] = utc_now()
        write_json(metadata_path, metadata)
        next_action = (
            SIMULATION_APPROVAL_ACTION
            if baseline.status == INFRASTRUCTURE_RETRY
            else "propose_candidate" if baseline.score is not None else "repair_baseline"
        )
    else:
        next_action = "submit_baseline"

    reason = "infrastructure_retry" if next_action == SIMULATION_APPROVAL_ACTION else "campaign_initialized"
    state = refresh_campaign_artifacts(args, paths, config, records, metadata, next_action=next_action, reason=reason)
    print_campaign_result(paths, records, state, initialized=True)
    if records and records[0].status == INFRASTRUCTURE_RETRY:
        return 75
    if args.target_env == "sim" and (not records or records[0].score is None):
        return 1
    return 0


def pending_candidate_manifests(workspace: Path) -> List[Path]:
    root = workspace / CANDIDATE_ROOT
    pending = []
    if not root.exists():
        return pending
    for path in sorted(root.glob("*/candidate_manifest.json")):
        manifest = read_json(path)
        validate_candidate_manifest_identity(path, manifest)
        status = manifest.get("status")
        if status in {"prepared", "ready_for_external_execution"}:
            pending.append(path)
    return pending


def count_abandoned_candidates(workspace: Path) -> int:
    root = workspace / CANDIDATE_ROOT
    if not root.exists():
        return 0
    count = 0
    for path in sorted(root.glob("*/candidate_manifest.json")):
        manifest = read_json(path)
        validate_candidate_manifest_identity(path, manifest)
        if manifest.get("status") == "abandoned":
            count += 1
    return count


def refresh_campaign_state(
    args: argparse.Namespace, job: Path, metadata: Dict[str, Any], paths: Dict[str, Path]
) -> Tuple[List[RunRecord], Dict[str, Any]]:
    """Recompute and persist campaign_state.json under the current effective settings without running a job."""
    records = load_results(paths["results"])
    pending = pending_candidate_manifests(job.parent)
    mode = campaign_contract_mode(metadata, read_yaml(paths["autofl_yaml"]), args.metric)
    state = write_state(
        paths["state"],
        paths["results"],
        records,
        args.max_candidates,
        mode=mode,
        stop_files=resolve_stop_files(job.parent, args.stop_file),
        plateau_threshold=args.plateau_threshold,
        plateau_min_delta=args.plateau_min_delta,
        hard_crash_threshold=args.hard_crash_threshold,
        exploration_batch_size=args.exploration_batch_size,
        family_repeat_limit=args.family_repeat_limit,
        pending_manifest_count=len(pending),
        abandoned_candidate_count=count_abandoned_candidates(job.parent),
        persist=False,
    )
    state.update(
        {
            "best_candidate": metadata.get("best_candidate"),
            "best_source_sha256": metadata.get("best_source_sha256"),
            "pending_candidate_manifest": str(pending[0].resolve()) if pending else None,
        }
    )
    write_state_if_changed(paths["state"], state)
    return records, state


def prepare_candidate(args: argparse.Namespace, job: Path) -> int:
    workspace = job.parent
    metadata = load_campaign_metadata(workspace, job)
    settings_changed = restore_campaign_settings(args, metadata)
    paths = campaign_paths(args, job)
    if settings_changed:
        refresh_campaign_state(args, job, metadata, paths)
    ensure_campaign_not_stopped(workspace, args, action="prepare a candidate")
    records = load_results(paths["results"])
    if not any(record.status == "baseline" and record.score is not None for record in records):
        raise ValueError("a scored baseline is required before preparing candidates")
    state = read_json(paths["state"])
    if state.get("final_response_allowed"):
        raise ValueError(f"campaign is already final: {state.get('reason')}")
    pending = pending_candidate_manifests(workspace)
    if pending:
        raise ValueError(f"campaign already has a pending candidate: {pending[0]}")
    candidate_id = validate_candidate_id(args.name or "")
    if not args.hypothesis:
        raise ValueError("--hypothesis is required for prepare")
    algorithm_family = (args.family or "").strip().lower()
    literature_event = (args.literature_event or "").strip()
    if literature_event:
        known_events = {record.literature_event_id for record in records if record.status == "literature"}
        if literature_event not in known_events:
            raise ValueError(f"unknown literature event id: {literature_event}")
    manifest_path = candidate_manifest_path(workspace, candidate_id)
    candidate_dir = manifest_path.parent
    if candidate_dir.exists():
        raise ValueError(f"candidate already exists: {candidate_id}")

    best_source, best_files = load_best_snapshot(paths["snapshot_root"])
    if source_hash(best_files) != metadata.get("best_source_sha256"):
        raise ValueError("campaign best-source hash is stale")
    if not workspace_matches_snapshot(workspace, best_source, best_files):
        raise ValueError("job workspace differs from the recorded best candidate; reconcile edits before preparing")

    draft_source = candidate_dir / "source"
    shutil.copytree(best_source, draft_source)
    config = read_yaml(paths["autofl_yaml"])
    objective = objective_contract(config, args.metric)
    manifest = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "name": candidate_id,
        "hypothesis": args.hypothesis,
        "algorithm_family": algorithm_family,
        "literature_event_id": literature_event,
        "status": "prepared",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "workspace_root": str(workspace.resolve()),
        "base_candidate": metadata.get("best_candidate"),
        "base_source_sha256": source_hash(best_files),
        "fixed_budget_sha256": metadata.get("fixed_budget_sha256"),
        "objective": {"metric": objective["optimization_metric"], "mode": objective["mode"]},
        "environment": args.target_env,
        "run_args": shlex.split(args.run_args),
        "changed_files": [],
        "patch_sha256": "",
        "candidate_source_sha256": source_hash(best_files),
        "provenance": {
            "job": str(job.resolve()),
            "autofl_yaml": str(paths["autofl_yaml"].resolve()),
            "import_source_sha256": read_yaml(paths["autofl_yaml"]).get("import", {}).get("source_sha256"),
        },
        "artifacts": {},
        "result": {},
    }
    write_json(manifest_path, manifest)
    state = refresh_campaign_artifacts(
        args,
        paths,
        config,
        records,
        metadata,
        pending_manifest=manifest_path,
        next_action="edit_candidate",
        reason="candidate_prepared",
    )
    print_campaign_result(
        paths,
        records,
        state,
        candidate_manifest=str(manifest_path.resolve()),
        candidate_source=str(draft_source.resolve()),
    )
    return 0


def validate_candidate_for_evaluation(
    args: argparse.Namespace,
    job: Path,
    metadata: Dict[str, Any],
    paths: Dict[str, Path],
    manifest_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Path, Dict[str, str], List[str], List[str], str]:
    manifest = load_candidate_manifest(manifest_path)
    if Path(str(manifest.get("workspace_root") or "")).resolve() != job.parent.resolve():
        raise ValueError("candidate manifest belongs to a different job workspace")
    config = read_yaml(paths["autofl_yaml"])
    best_source, best_files = load_best_snapshot(paths["snapshot_root"])
    best_hash = source_hash(best_files)
    if manifest.get("base_source_sha256") != best_hash or metadata.get("best_source_sha256") != best_hash:
        raise ValueError("candidate was prepared from a stale best candidate")
    if manifest.get("fixed_budget_sha256") != metadata.get("fixed_budget_sha256"):
        raise ValueError("candidate fixed-budget provenance is stale")
    if not workspace_matches_snapshot(job.parent, best_source, best_files):
        raise ValueError("job workspace differs from the recorded best candidate")
    draft_source = manifest_path.parent / "source"
    changed, created = candidate_changes(job.parent, config, best_source, best_files, draft_source)
    recorded_changed = validated_manifest_paths(manifest, job.parent, "changed_files")
    recorded_created = validated_manifest_paths(manifest, job.parent, "created_files")
    if (recorded_changed or recorded_created) and (recorded_changed != changed or recorded_created != created):
        raise ValueError("candidate manifest source lists do not match the deterministic candidate diff")
    run_args = manifest.get("run_args")
    if not isinstance(run_args, list) or not all(isinstance(item, str) for item in run_args):
        raise ValueError("candidate manifest run_args must be a list of strings")
    allowed, reason = candidate_args_allowed(run_args, load_mutation_schema(job.parent))
    if not allowed:
        raise ValueError(reason)
    allowed, reason = candidate_preserves_fixed_args(run_args, config, load_mutation_schema(job.parent))
    if not allowed:
        raise ValueError(reason)
    if not changed and not run_args:
        raise ValueError("candidate has no source changes or run arguments")
    if (str(manifest.get("literature_event_id") or "")).strip() and not changed and not created:
        raise ValueError("literature-linked candidates must include source edits")
    patch = render_candidate_patch(best_source, draft_source, changed)
    return manifest, config, best_source, best_files, changed, created, patch


def candidate_execution_fingerprint(manifest: Dict[str, Any], patch_sha256: str) -> str:
    run_args = manifest.get("run_args")
    if not isinstance(run_args, list) or not all(isinstance(item, str) for item in run_args):
        raise ValueError("candidate manifest run_args must be a list of strings")
    fields = {
        "base_source_sha256": str(manifest.get("base_source_sha256") or ""),
        "fixed_budget_sha256": str(manifest.get("fixed_budget_sha256") or ""),
        "patch_sha256": patch_sha256,
        "run_args": run_args,
    }
    if not all(fields[name] for name in ("base_source_sha256", "fixed_budget_sha256", "patch_sha256")):
        raise ValueError("candidate manifest is missing execution fingerprint provenance")
    return sha256_json(fields)


def matching_crashed_candidate(
    workspace: Path, fingerprint: str, current_manifest_path: Path
) -> Optional[Tuple[Path, Dict[str, Any]]]:
    root = workspace / CANDIDATE_ROOT
    if not root.exists():
        return None
    for path in sorted(root.glob("*/candidate_manifest.json")):
        if path.resolve() == current_manifest_path.resolve():
            continue
        try:
            manifest = read_json(path)
            validate_candidate_manifest_identity(path, manifest)
            if manifest.get("status") != "crash":
                continue
            patch_sha256 = str(manifest.get("patch_sha256") or "")
            if patch_sha256 and candidate_execution_fingerprint(manifest, patch_sha256) == fingerprint:
                return path, manifest
        except (OSError, ValueError) as error:
            print(f"Warning: ignoring invalid sibling candidate manifest {path}: {error}", file=sys.stderr)
    return None


def crash_replay_provenance(workspace: Path, fingerprint: str, current_manifest_path: Path) -> Optional[Dict[str, Any]]:
    prior_crash = matching_crashed_candidate(workspace, fingerprint, current_manifest_path)
    if prior_crash is None:
        return None
    prior_path, prior_manifest = prior_crash
    return {
        "execution_fingerprint": fingerprint,
        "prior_candidate": prior_manifest.get("candidate_id"),
        "prior_manifest": str(prior_path.resolve()),
    }


def update_config_for_kept_sources(config: Dict[str, Any], created: Sequence[str]) -> None:
    if not created:
        return
    trust_paths = config.setdefault("trust_contract", {}).setdefault("allowed_edit_paths", [])
    for relative in created:
        if relative not in trust_paths:
            trust_paths.append(relative)


def assumed_job_key_metric_mode(recorded_objective: Dict[str, Any], current_objective: Dict[str, Any]) -> str:
    """Backfill a missing job-key mode without fabricating metric identity.

    Campaigns from the max-only era used ``max``. Intermediate native-direction
    campaigns may omit the field but can reuse their recorded mode when job
    provenance proves the requested metric was the resolved job key metric.
    """

    directly_imported_mode = str(recorded_objective.get("mode_contract_source") or "").startswith("job:")
    if directly_imported_mode and not requested_metric_differs_from_job(recorded_objective):
        return current_objective["mode"]
    return "max"


def candidate_campaign_config(
    candidate_config: Dict[str, Any], current_config: Dict[str, Any], args: argparse.Namespace, schema: Dict[str, Any]
) -> Dict[str, Any]:
    candidate_config = apply_metric_contract(candidate_config, args.metric, schema)
    candidate_errors = campaign_admission_errors(candidate_config, schema)
    if candidate_errors:
        raise ValueError(f"candidate changes the admitted metric contract: {'; '.join(candidate_errors)}")
    candidate_objective = objective_contract(candidate_config, args.metric)
    current_objective = objective_contract(current_config, args.metric)
    invariant_fields = (
        "requested_metric",
        "optimization_metric",
        "metric_extraction_order",
        "metric_source",
        "mode",
        "job_key_metric",
        "job_key_metric_mode",
    )
    recorded_objective = current_config.get("objective", {})
    if not isinstance(recorded_objective, dict):
        recorded_objective = {}
    drift = []
    for invariant_field in invariant_fields:
        if invariant_field == "job_key_metric" and invariant_field not in recorded_objective:
            continue
        current_value = current_objective.get(invariant_field)
        if invariant_field == "job_key_metric_mode" and invariant_field not in recorded_objective:
            current_value = assumed_job_key_metric_mode(recorded_objective, current_objective)
        if candidate_objective.get(invariant_field) != current_value:
            drift.append(invariant_field)
    if drift:
        message = f"candidate changes objective metric invariants: {', '.join(drift)}"
        if {"mode", "job_key_metric_mode"} & set(drift):
            message += (
                "; the direction contract changed, so delete the campaign's .nvflare/autofl directory "
                "(or start in a fresh workspace) and initialize again"
            )
        raise ValueError(message)
    candidate_config["objective"] = dict(current_config.get("objective", {}))
    current_paths = current_config.get("trust_contract", {}).get("allowed_edit_paths", []) or []
    trust_paths = candidate_config.setdefault("trust_contract", {}).setdefault("allowed_edit_paths", [])
    for path in current_paths:
        if path not in trust_paths:
            trust_paths.append(path)
    candidate_config["trust_contract"]["allowed_create_patterns"] = allowed_create_patterns(current_config)
    return candidate_config


def finalize_candidate_result(
    args: argparse.Namespace,
    job: Path,
    metadata: Dict[str, Any],
    paths: Dict[str, Path],
    config: Dict[str, Any],
    manifest_path: Path,
    manifest: Dict[str, Any],
    best_source: Path,
    best_files: Dict[str, str],
    changed: List[str],
    created: List[str],
    patch: str,
    record: RunRecord,
    crash_replay: Optional[Dict[str, Any]],
) -> Tuple[List[RunRecord], Dict[str, Any]]:
    rollback_files: Dict[Path, Optional[bytes]] = {}
    staged_snapshot = None
    previous_snapshot = None
    try:
        records = load_results(paths["results"])
        mode = objective_mode(config, args.metric)
        previous_best = best_retained_record(records, mode)
        if record.status == "candidate":
            record.status = (
                "keep" if better(record.score, previous_best.score if previous_best else None, mode) else "discard"
            )
        patch_path = manifest_path.parent / "candidate.patch"
        rollback_files = capture_file_versions(
            [
                paths["autofl_yaml"],
                manifest_path,
                campaign_metadata_path(job.parent),
                paths["results"],
                paths["state"],
                paths["progress"],
                paths["report"],
                patch_path,
            ]
        )
        atomic_write_text(patch_path, patch)
        patch_sha256 = sha256_bytes(patch.encode("utf-8"))
        record.changed_files = ",".join(changed) if changed else "none"
        record.diff_summary = str(manifest.get("hypothesis") or "candidate")
        record.candidate_manifest = str(manifest_path.resolve())
        record.base_candidate = str(manifest.get("base_candidate") or "")
        record.patch_sha256 = patch_sha256
        record.candidate_kind = "source_edit" if changed or created else "argument_only"
        record.algorithm_family = str(manifest.get("algorithm_family") or "")
        record.literature_event_id = str(manifest.get("literature_event_id") or "")

        if record.status == "keep":
            update_config_for_kept_sources(config, created)
            staged_snapshot, snapshot_files = stage_best_snapshot(job.parent, config, paths["snapshot_root"])
            write_yaml(paths["autofl_yaml"], config)
            previous_snapshot = activate_best_snapshot(paths["snapshot_root"], staged_snapshot)
            staged_snapshot = None
            metadata.update(
                {
                    "best_candidate": record.name,
                    "best_score": record.score,
                    "best_source_sha256": source_hash(snapshot_files),
                    "updated_at": utc_now(),
                }
            )
        else:
            restore_best_source(job.parent, best_source, best_files, changed, created)

        manifest.update(
            {
                "status": record.status,
                "updated_at": utc_now(),
                "changed_files": changed,
                "created_files": created,
                "candidate_kind": record.candidate_kind,
                "patch_sha256": patch_sha256,
                "artifacts": {"patch": str(patch_path.resolve()), "run": record.artifacts},
                "result": {
                    "score": record.score,
                    "metric_name": record.metric_name,
                    "metric_source": record.metric_source,
                    "metric_artifact": record.metric_artifact,
                    "runtime_seconds": record.runtime_seconds,
                    "run_command": record.run_command,
                    "failure_reason": record.failure_reason,
                },
            }
        )
        manifest.pop("crash_replay", None)
        if crash_replay:
            manifest["crash_replay"] = {
                **crash_replay,
                "recorded_at": utc_now(),
                "outcome_status": record.status,
            }
        write_json(manifest_path, manifest)
        write_json(campaign_metadata_path(job.parent), metadata)
        records.append(record)
        state = refresh_campaign_artifacts(args, paths, config, records, metadata)
    except BaseException as error:
        try:
            if previous_snapshot is not None:
                rollback_best_snapshot(paths["snapshot_root"], previous_snapshot)
                previous_snapshot = None
            restore_best_source(job.parent, paths["snapshot_root"] / "source", best_files, changed, created)
            if rollback_files:
                restore_file_versions(rollback_files)
        except BaseException as rollback_error:
            raise RuntimeError(
                f"candidate finalization failed ({error}); automatic workspace rollback also failed ({rollback_error})"
            ) from rollback_error
        raise
    finally:
        if staged_snapshot is not None:
            shutil.rmtree(staged_snapshot, ignore_errors=True)

    if previous_snapshot is not None:
        shutil.rmtree(previous_snapshot, ignore_errors=True)
    return records, state


def evaluate_candidate(args: argparse.Namespace, job: Path) -> int:
    workspace = job.parent
    metadata = load_campaign_metadata(workspace, job)
    settings_changed = restore_campaign_settings(args, metadata)
    paths = campaign_paths(args, job)
    if settings_changed:
        refresh_campaign_state(args, job, metadata, paths)
    ensure_campaign_not_stopped(workspace, args, action="evaluate a candidate")
    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    if manifest_path is None:
        pending = pending_candidate_manifests(workspace)
        if len(pending) != 1:
            raise ValueError("--manifest is required when there is not exactly one pending candidate")
        manifest_path = pending[0]
    manifest, config, best_source, best_files, changed, created, patch = validate_candidate_for_evaluation(
        args, job, metadata, paths, manifest_path
    )
    managed_versions = capture_managed_source_versions(workspace, config, [*changed, *created])
    managed_paths = [path.relative_to(workspace).as_posix() for path in managed_versions]
    patch_path = manifest_path.parent / "candidate.patch"
    atomic_write_text(patch_path, patch)
    patch_sha256 = sha256_bytes(patch.encode("utf-8"))
    execution_fingerprint = candidate_execution_fingerprint(manifest, patch_sha256)
    crash_replay = crash_replay_provenance(workspace, execution_fingerprint, manifest_path)
    manifest.pop("crash_replay", None)
    manifest.update(
        {
            "updated_at": utc_now(),
            "changed_files": changed,
            "created_files": created,
            "patch_sha256": patch_sha256,
            "candidate_source_sha256": source_hash(file_map(manifest_path.parent / "source")),
            "execution_fingerprint": execution_fingerprint,
        }
    )
    write_json(manifest_path, manifest)
    schema = load_mutation_schema(workspace)
    timeout, no_progress_timeout = campaign_timeout(args, schema)
    candidate_config_path = manifest_path.parent / "candidate_autofl.yaml"
    try:
        apply_candidate_source(workspace, manifest_path.parent / "source", changed)
        candidate_config = import_job_config(
            args,
            job,
            candidate_config_path,
            manifest_path.parent / "import.log",
            timeout,
        )
        if fixed_budget_hash(candidate_config) != metadata.get("fixed_budget_sha256"):
            raise ValueError("candidate changes budget.fixed_training_budget")
        candidate_config = candidate_campaign_config(candidate_config, config, args, schema)
        expected_managed_state = managed_source_state(workspace, config, managed_paths)
    except BaseException as error:
        try:
            restore_managed_source_versions(workspace, config, managed_versions)
        except BaseException as rollback_error:
            raise RuntimeError(
                f"candidate validation failed ({error}); automatic managed-source rollback also failed "
                f"({rollback_error})"
            ) from rollback_error
        raise

    if args.target_env != "sim":
        manifest["status"] = "ready_for_external_execution"
        manifest["updated_at"] = utc_now()
        write_json(manifest_path, manifest)
        records = load_results(paths["results"])
        state = refresh_campaign_artifacts(
            args,
            paths,
            config,
            records,
            metadata,
            pending_manifest=manifest_path,
            next_action="submit_candidate",
            reason="candidate_validated",
        )
        print_campaign_result(
            paths,
            records,
            state,
            candidate_manifest=str(manifest_path.resolve()),
            job=str(job.resolve()),
        )
        return 0

    try:
        help_text = job_help(args.python, job, workspace)
        fixed_args, base_args = build_campaign_args(
            config, args, help_text, schema, alias_groups=job_flag_alias_groups(job)
        )
        run_record = run_job(
            JobRun(
                name=str(manifest["candidate_id"]),
                args=list(manifest.get("run_args") or []),
                description=str(manifest.get("hypothesis") or "candidate"),
            ),
            python=args.python,
            job=job,
            cwd=workspace,
            help_text=help_text,
            fixed_args=fixed_args,
            base_args=base_args,
            output_root=paths["output_root"],
            timeout=timeout,
            simulator_no_progress_timeout=no_progress_timeout,
            metrics=metric_extraction_order(config, args.metric),
            config=config,
        )
    except BaseException as error:
        try:
            restore_managed_source_versions(workspace, config, managed_versions)
        except BaseException as rollback_error:
            raise RuntimeError(
                f"candidate execution failed ({error}); automatic managed-source rollback also failed "
                f"({rollback_error})"
            ) from rollback_error
        raise
    actual_managed_state = managed_source_state(workspace, config, expected_managed_state.keys())
    runtime_drift = managed_source_drift(expected_managed_state, actual_managed_state)
    if runtime_drift:
        try:
            restore_managed_source_versions(workspace, config, managed_versions)
        except BaseException as rollback_error:
            raise RuntimeError(
                "candidate modified campaign-managed source during execution and automatic rollback failed "
                f"({rollback_error}); candidate remains pending for recovery"
            ) from rollback_error
        drift_reason = f"candidate modified campaign-managed source during execution: {', '.join(runtime_drift)}"
        run_record.status = "crash"
        run_record.score = None
        run_record.metric_name = ""
        run_record.metric_source = ""
        run_record.metric_artifact = ""
        run_record.failure_reason = (
            f"{run_record.failure_reason}; {drift_reason}" if run_record.failure_reason else drift_reason
        )
    if run_record.status == INFRASTRUCTURE_RETRY:
        restore_managed_source_versions(workspace, config, managed_versions)
        manifest["status"] = "prepared"
        manifest["result"] = {"failure_reason": run_record.failure_reason}
        manifest["updated_at"] = utc_now()
        write_json(manifest_path, manifest)
        records = load_results(paths["results"])
        state = refresh_campaign_artifacts(
            args,
            paths,
            config,
            records,
            metadata,
            pending_manifest=manifest_path,
            next_action=SIMULATION_APPROVAL_ACTION,
            reason="infrastructure_retry",
        )
        print_campaign_result(paths, records, state, candidate_manifest=str(manifest_path.resolve()))
        return 75

    records, state = finalize_candidate_result(
        args,
        job,
        metadata,
        paths,
        candidate_config,
        manifest_path,
        manifest,
        best_source,
        best_files,
        changed,
        created,
        patch,
        run_record,
        crash_replay,
    )
    print_campaign_result(paths, records, state, candidate_manifest=str(manifest_path.resolve()))
    return 0


def abandon_candidate(args: argparse.Namespace, job: Path) -> int:
    workspace = job.parent
    metadata = load_campaign_metadata(workspace, job)
    settings_changed = restore_campaign_settings(args, metadata)
    paths = campaign_paths(args, job)
    if settings_changed:
        refresh_campaign_state(args, job, metadata, paths)
    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    if manifest_path is None:
        pending = pending_candidate_manifests(workspace)
        if len(pending) != 1:
            raise ValueError("--manifest is required when there is not exactly one pending candidate")
        manifest_path = pending[0]
    manifest = load_candidate_manifest(manifest_path)
    best_source, best_files = load_best_snapshot(paths["snapshot_root"])
    config = read_yaml(paths["autofl_yaml"])
    draft_source = manifest_path.parent / "source"
    changed, created = candidate_changes(
        workspace,
        config,
        best_source,
        best_files,
        draft_source,
        allow_materialized=manifest.get("status") == "ready_for_external_execution",
    )
    recorded_changed = validated_manifest_paths(manifest, workspace, "changed_files")
    recorded_created = validated_manifest_paths(manifest, workspace, "created_files")
    if (recorded_changed or recorded_created) and (recorded_changed != changed or recorded_created != created):
        raise ValueError("candidate manifest source lists do not match the deterministic candidate diff")
    if manifest.get("status") == "ready_for_external_execution":
        if not workspace_matches_snapshot(workspace, draft_source, file_map(draft_source)):
            raise ValueError("materialized candidate source changed after validation")
        restore_best_source(workspace, best_source, best_files, changed, created)
    elif not workspace_matches_snapshot(workspace, best_source, best_files):
        raise ValueError("job workspace differs from the recorded best candidate")
    manifest["status"] = "abandoned"
    manifest["updated_at"] = utc_now()
    write_json(manifest_path, manifest)
    records = load_results(paths["results"])
    state = refresh_campaign_artifacts(
        args,
        paths,
        config,
        records,
        metadata,
        next_action="propose_candidate",
        reason="candidate_abandoned",
    )
    print_campaign_result(paths, records, state, candidate_manifest=str(manifest_path.resolve()))
    return 0


def suggest_candidates(args: argparse.Namespace, job: Path) -> int:
    metadata = load_campaign_metadata(job.parent, job)
    settings_changed = restore_campaign_settings(args, metadata)
    paths = campaign_paths(args, job)
    if settings_changed:
        refresh_campaign_state(args, job, metadata, paths)
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    config = read_yaml(paths["autofl_yaml"])
    help_text = job_help(args.python, job, job.parent)
    suggestions = [
        {"name": candidate.name, "run_args": candidate.args, "hypothesis": candidate.description}
        for candidate in candidate_plan(
            config,
            help_text,
            args.limit,
            load_mutation_schema(job.parent),
        )
    ]
    print(json.dumps({"suggestions": suggestions}, indent=2, sort_keys=True))
    return 0


def record_external_result(args: argparse.Namespace, job: Path) -> int:
    workspace = job.parent
    metadata = load_campaign_metadata(workspace, job)
    settings_changed = restore_campaign_settings(args, metadata)
    paths = campaign_paths(args, job)
    if settings_changed:
        refresh_campaign_state(args, job, metadata, paths)
    config = read_yaml(paths["autofl_yaml"])
    artifact_path = Path(args.external_artifacts).resolve() if args.external_artifacts else None
    score = parse_finite_metric_value(args.score) if args.score is not None else None
    if args.score is not None and score is None:
        raise ValueError("--score must be a finite number")
    evidence = None
    if score is None and artifact_path:
        evidence = extract_metric_evidence(artifact_path, metric_extraction_order(config, args.metric))
        score = evidence.score if evidence else None
    elif score is not None:
        evidence = MetricEvidence(
            score=score,
            metric_name=optimization_metric(config, args.metric),
            source="provided",
            artifact=str(artifact_path or ""),
        )
    if args.literature:
        records = load_results(paths["results"])
        event_number = sum(record.status == "literature" for record in records) + 1
        event_id = f"lit-{event_number:04d}"
        name = f"literature_review_{event_number}"
        records.append(
            RunRecord(
                status="literature",
                name=name,
                score=None,
                runtime_seconds=0.0,
                changed_files="none",
                diff_summary=args.hypothesis or "source-backed literature review",
                run_command="agent literature review",
                artifacts=str(artifact_path or ""),
                algorithm_family=(args.family or "").strip().lower(),
                literature_event_id=event_id,
            )
        )
        # The guard now derives next_action=develop_literature_batch and keeps
        # required_exploration active until the linked exploration batch completes.
        state = refresh_campaign_artifacts(args, paths, config, records, metadata)
        print_campaign_result(paths, records, state, literature_event=name, literature_event_id=event_id)
        return 0
    if args.baseline:
        records = load_results(paths["results"])
        if any(
            record.status == "baseline"
            or (record.status in {"candidate", "keep", "discard", "crash"} and not is_baseline_record(record))
            for record in records
        ):
            raise ValueError("external baseline can only be recorded before a baseline or campaign candidate")
        if score is None:
            raise ValueError("a score or extractable --artifacts path is required for the baseline")
        record = RunRecord(
            status="baseline",
            name="baseline",
            score=score,
            runtime_seconds=0.0,
            changed_files="none",
            diff_summary="external baseline",
            run_command=f"nvflare job id={args.job_id or 'unreported'}",
            artifacts=str(artifact_path or ""),
            metric_name=evidence.metric_name if evidence else "",
            metric_source=evidence.source if evidence else "",
            metric_artifact=evidence.artifact if evidence else "",
        )
        records.append(record)
        metadata["best_score"] = score
        metadata["updated_at"] = utc_now()
        write_json(campaign_metadata_path(workspace), metadata)
        state = refresh_campaign_artifacts(
            args,
            paths,
            config,
            records,
            metadata,
            next_action="propose_candidate",
            reason="baseline_recorded",
        )
        print_campaign_result(paths, records, state, job_id=args.job_id)
        return 0

    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    if manifest_path is None:
        pending = pending_candidate_manifests(workspace)
        if len(pending) != 1:
            raise ValueError("--manifest is required when there is not exactly one pending candidate")
        manifest_path = pending[0]
    manifest = load_candidate_manifest(manifest_path)
    if manifest.get("status") != "ready_for_external_execution":
        raise ValueError("external candidate must be validated before its result is recorded")
    draft_source = manifest_path.parent / "source"
    draft_files = file_map(draft_source)
    if source_hash(draft_files) != manifest.get("candidate_source_sha256") or not workspace_matches_snapshot(
        workspace, draft_source, draft_files
    ):
        raise ValueError("materialized candidate source changed after validation")
    best_source, best_files = load_best_snapshot(paths["snapshot_root"])
    changed, created = candidate_changes(
        workspace, config, best_source, best_files, draft_source, allow_materialized=True
    )
    validate_manifest_change_set(manifest, workspace, changed, created)
    patch_path = manifest_path.parent / "candidate.patch"
    patch = patch_path.read_text(encoding="utf-8") if patch_path.exists() else ""
    candidate_config_path = manifest_path.parent / "candidate_autofl.yaml"
    schema = load_mutation_schema(workspace)
    timeout, _ = campaign_timeout(args, schema)
    candidate_config = import_job_config(
        args,
        job,
        candidate_config_path,
        manifest_path.parent / "record_import.log",
        timeout,
    )
    if fixed_budget_hash(candidate_config) != metadata.get("fixed_budget_sha256"):
        raise ValueError("candidate changes budget.fixed_training_budget")
    candidate_config = candidate_campaign_config(candidate_config, config, args, schema)
    status = "crash" if args.failure_reason or score is None else "candidate"
    record = RunRecord(
        status=status,
        name=str(manifest["candidate_id"]),
        score=score,
        runtime_seconds=0.0,
        changed_files="none",
        diff_summary=str(manifest.get("hypothesis") or "candidate"),
        run_command=f"nvflare job id={args.job_id or 'unreported'}",
        artifacts=str(artifact_path or ""),
        failure_reason=args.failure_reason or ("metric not found" if score is None else ""),
        metric_name=evidence.metric_name if evidence else "",
        metric_source=evidence.source if evidence else "",
        metric_artifact=evidence.artifact if evidence else "",
    )
    execution_fingerprint = str(manifest.get("execution_fingerprint") or "")
    crash_replay = (
        crash_replay_provenance(workspace, execution_fingerprint, manifest_path) if execution_fingerprint else None
    )
    records, state = finalize_candidate_result(
        args,
        job,
        metadata,
        paths,
        candidate_config,
        manifest_path,
        manifest,
        best_source,
        best_files,
        changed,
        created,
        patch,
        record,
        crash_replay,
    )
    updated_manifest = read_json(manifest_path)
    updated_manifest.setdefault("artifacts", {})["job_id"] = args.job_id
    write_json(manifest_path, updated_manifest)
    print_campaign_result(paths, records, state, candidate_manifest=str(manifest_path.resolve()), job_id=args.job_id)
    return 0


def show_campaign_status(args: argparse.Namespace, job: Path) -> int:
    metadata = load_campaign_metadata(job.parent, job)
    restore_campaign_settings(args, metadata)
    paths = campaign_paths(args, job)
    records, state = refresh_campaign_state(args, job, metadata, paths)
    print_campaign_result(paths, records, state, campaign=metadata)
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.max_candidates is not None and args.max_candidates < 1:
        raise ValueError("--max-candidates must be positive when provided")
    if args.plateau_threshold < 1:
        raise ValueError("--plateau-threshold must be positive")
    if args.plateau_min_delta < 0:
        raise ValueError("--plateau-min-delta must be non-negative")
    if args.hard_crash_threshold < 0:
        raise ValueError("--hard-crash-threshold must be non-negative")
    if args.exploration_batch_size < 0:
        raise ValueError("--exploration-batch-size must be non-negative")
    if args.family_repeat_limit < 0:
        raise ValueError("--family-repeat-limit must be non-negative")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
        job = Path(args.job).resolve()
        if not job.is_file():
            raise ValueError(f"job.py does not exist: {job}")
        actions = {
            "initialize": initialize_campaign,
            "prepare": prepare_candidate,
            "evaluate": evaluate_candidate,
            "abandon": abandon_candidate,
            "suggest": suggest_candidates,
            "record": record_external_result,
            "status": show_campaign_status,
        }
        if args.action == "initialize" and not campaign_metadata_path(job.parent).exists():
            prepare_initial_campaign(args, job)
        with locked_campaign_workspace(job.parent, args.action):
            return actions[args.action](args, job)
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Auto-FL {args.action} failed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
