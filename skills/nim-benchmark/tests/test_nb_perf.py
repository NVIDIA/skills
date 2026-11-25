# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/nb_perf.py: plans, AIPerf commands, export checks and whole runs."""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import nb_common  # noqa: E402
import nb_perf  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "aiperf_export_1_4.json"
# what the fake AIPerf does for one run: (load, repeat, attempt, has_salt) -> export, or None
Behavior = Callable[[int, int, int, bool], dict[str, Any] | None]


def export_for(requested: int, **changes: Any) -> dict[str, Any]:
    """The real AIPerf 0.13.0 export, with the request count set to what the plan asked."""
    doc: dict[str, Any] = copy.deepcopy(json.loads(FIXTURE.read_text()))
    doc["request_count"]["avg"] = float(requested)
    doc.update(changes)
    return doc


def make_plan(**changes: Any) -> dict[str, Any]:
    plan = nb_perf.build_plan(
        endpoint="http://gpu-01:8000",
        model="qwen7b",
        tokenizer="Qwen/Qwen2.5-7B-Instruct",
        loads=[2, 4],
        repeats=3,
        isl=64,
        osl=32,
        key_env=None,
        target="docker",
        image="python:3.12-slim",
        namespace=None,
        pod=None,
    )
    plan.update(changes)
    plan["plan_id"] = "abc123"
    return plan


class FakeDocker:
    """Stands in for run_command: it plays AIPerf by writing an export into the mounted folder."""

    def __init__(self, behavior: Behavior, console: str = "") -> None:
        self.behavior = behavior
        self.console = console
        self.calls: list[list[str]] = []
        self.attempts: dict[tuple[int, int], int] = {}

    def __call__(
        self, argv: list[str], timeout: float = 0, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        if argv[:2] != ["docker", "run"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        mount = argv[argv.index("-v") + 1].split(":")[0]
        line = argv[-1]
        load = int(re.search(r"--concurrency (\d+)", line).group(1))  # type: ignore[union-attr]
        seed = int(re.search(r"--random-seed (\d+)", line).group(1))  # type: ignore[union-attr]
        rel = re.search(r"--artifact-dir /out/(\S+)", line).group(1)  # type: ignore[union-attr]
        repeat = seed - 100 * load
        attempt = self.attempts.get((load, repeat), 0)
        self.attempts[(load, repeat)] = attempt + 1
        doc = self.behavior(load, repeat, attempt, "cache_salt:" in line)
        folder = Path(mount) / rel
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "inputs.json").write_text('{"prompts": ["private prompt"]}')
        (folder / "profile_export.jsonl").write_text('{"answer": "private answer"}')
        if doc is not None:
            (folder / nb_perf.EXPORT_NAME).write_text(json.dumps(doc))
        return subprocess.CompletedProcess(argv, 0, self.console, "")


def install(monkeypatch: pytest.MonkeyPatch, fake: FakeDocker) -> FakeDocker:
    monkeypatch.setattr(nb_common, "run_command", fake)
    return fake


def good(load: int, _r: int, _a: int, _s: bool) -> dict[str, Any]:
    return export_for(max(40, 10 * load))


def test_plan_has_documented_seeds_and_request_counts() -> None:
    plan = make_plan()
    assert plan["seeds"] == [201, 202, 203, 401, 402, 403]
    assert plan["requests_per_load"] == {"2": 40, "4": 40}
    assert nb_perf.requests_for(128) == 1280
    assert plan["aiperf_version"] == "0.13.0"


def test_aiperf_args_pin_the_workload_and_hold_no_secret() -> None:
    plan = make_plan(key_env="MY_KEY")
    args = nb_perf.aiperf_args(plan, 4, 2, "/out/perf/c4-r2", "salt1")
    joined = " ".join(args)
    for part in ("--concurrency 4", "--random-seed 402", "--isl 64", "--osl 32", "--streaming"):
        assert part in joined
    assert "cache_salt:salt1" in joined and "ignore_eos:true" in joined
    assert "MY_KEY" not in joined and "--api-key" not in joined
    assert "cache_salt" not in " ".join(nb_perf.aiperf_args(plan, 4, 2, "/o", None))


def test_the_key_is_a_variable_reference_in_the_shell_line() -> None:
    line = nb_perf.shell_line(["aiperf", "profile", "--model", "a b"], "MY_KEY")
    assert line.endswith('--api-key "$MY_KEY"') and "'a b'" in line
    assert "--api-key" not in nb_perf.shell_line(["aiperf"], None)


def test_client_argv_for_each_target(tmp_path: Path) -> None:
    docker = nb_perf.client_argv(make_plan(key_env="MY_KEY"), "aiperf x", tmp_path, "nm")
    assert docker[:5] == ["docker", "run", "--rm", "--name", "nm"]
    assert "-e" in docker and "MY_KEY" in docker and "HF_TOKEN" in docker
    assert f"{tmp_path.resolve()}:/out" in docker
    assert docker[-1] == "pip install -q aiperf==0.13.0 && aiperf x"

    k8s = make_plan(client_target="kubernetes", namespace="ns", pod="pod-1")
    assert nb_perf.client_argv(k8s, "aiperf x", tmp_path, "nm")[:5] == [
        "kubectl",
        "exec",
        "-n",
        "ns",
        "pod-1",
    ]
    local = nb_perf.client_argv(make_plan(client_target="local"), "aiperf x", tmp_path, "nm")
    assert local[:2] == ["sh", "-c"] and local[2].endswith("&& aiperf x")


def test_check_export_accepts_a_real_export() -> None:
    assert nb_perf.check_export(export_for(8), 8) == ("ok", "")


@pytest.mark.parametrize(
    ("changes", "kind", "word"),
    [
        ({"schema_version": "2.0"}, "tool_failed", "schema_version"),
        ({"schema_version": None}, "tool_failed", "schema_version"),
        ({"was_cancelled": True}, "tool_failed", "cancelled"),
        ({"is_complete": False}, "tool_failed", "finish"),
        ({"time_to_first_token": None}, "tool_failed", "time_to_first_token"),
        ({"request_count": {"avg": 5.0}}, "tool_failed", "5.0 of 8"),
        ({"request_error_rate": {"avg": 12.5}}, "request_errors", "failed"),
        ({"error_summary": [{"code": 500}]}, "request_errors", "failed"),
    ],
)
def test_check_export_refuses_a_bad_export(changes: dict[str, Any], kind: str, word: str) -> None:
    got_kind, why = nb_perf.check_export(export_for(8, **changes), 8)
    assert got_kind == kind and word in why


def test_check_export_needs_a_p99_for_every_latency() -> None:
    doc = export_for(8)
    del doc["request_latency"]["p99"]
    assert nb_perf.check_export(doc, 8)[0] == "tool_failed"
    assert nb_perf.check_export([], 8)[0] == "tool_failed"


def test_summarize_and_aggregate_use_the_real_numbers() -> None:
    summary = nb_perf.summarize_run(export_for(8))
    assert summary["ttft_avg"] == pytest.approx(36.462, abs=1e-3)
    assert summary["itl_p99"] == pytest.approx(2.9878, abs=1e-3)
    assert "otps_avg" in summary and "rps_avg" in summary
    both = nb_perf.aggregate([{"x": 1.0}, {"x": 3.0}])
    assert both["x"] == {"mean": 2.0, "min": 1.0, "max": 3.0, "values": [1.0, 3.0]}
    assert nb_perf.aggregate([]) == {}


def test_a_full_run_counts_three_repeats_and_removes_raw_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, FakeDocker(good))
    result = nb_perf.run_plan(make_plan(), tmp_path)
    assert result["problems"] == [] and result["incomplete"] == []
    assert sorted(result["loads"], key=int) == ["2", "4"]
    assert result["loads"]["2"]["repeats"] == 3
    assert len(fake.calls) == 6
    assert result["cache_salt"] == "used"
    for folder in (tmp_path / "perf").glob("c*-r*"):
        assert (folder / nb_perf.EXPORT_NAME).exists()
        assert not (folder / "inputs.json").exists()
        assert not (folder / "profile_export.jsonl").exists()


def test_every_run_gets_its_own_seed_and_a_fresh_salt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, FakeDocker(good))
    nb_perf.run_plan(make_plan(), tmp_path)
    lines = [c[-1] for c in fake.calls]
    seeds = [re.search(r"--random-seed (\d+)", ln).group(1) for ln in lines]  # type: ignore[union-attr]
    salts = [re.search(r"cache_salt:(\w+)", ln).group(1) for ln in lines]  # type: ignore[union-attr]
    assert seeds == ["201", "202", "203", "401", "402", "403"]
    assert len(set(salts)) == 6


def test_a_run_with_failed_requests_is_retried_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def flaky(load: int, repeat: int, attempt: int, _s: bool) -> dict[str, Any]:
        bad = (load, repeat) == (2, 1) and attempt == 0
        return export_for(40, request_error_rate={"avg": 5.0}) if bad else good(load, 0, 0, False)

    install(monkeypatch, FakeDocker(flaky))
    result = nb_perf.run_plan(make_plan(), tmp_path)
    assert result["loads"]["2"]["repeats"] == 3
    assert result["run_had_request_errors"] == ["c2-r1"]
    assert result["incomplete"] == []


def test_a_load_with_too_few_counted_repeats_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def always_bad(load: int, repeat: int, _a: int, _s: bool) -> dict[str, Any]:
        if (load, repeat) == (4, 3):
            return export_for(40, request_error_rate={"avg": 5.0})
        return good(load, 0, 0, False)

    install(monkeypatch, FakeDocker(always_bad))
    result = nb_perf.run_plan(make_plan(), tmp_path)
    assert "4" not in result["loads"]
    assert result["incomplete"] == [{"load": 4, "counted": 2, "needed": 3}]
    assert result["run_had_request_errors"] == ["c4-r3", "c4-r3"]


def test_a_missing_export_is_a_tool_failure_whatever_the_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, FakeDocker(lambda *_a: None))
    result = nb_perf.run_plan(make_plan(loads=[2], seeds=[201, 202, 203]), tmp_path)
    assert result["loads"] == {}
    assert {p["code"] for p in result["problems"]} == {"tool_failed"}
    assert "aiperf.log" in result["problems"][0]["fix"]


def test_a_rejected_cache_salt_is_dropped_for_every_later_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def picky(load: int, _r: int, _a: int, has_salt: bool) -> dict[str, Any] | None:
        return None if has_salt else good(load, 0, 0, False)

    fake = install(monkeypatch, FakeDocker(picky, console="unknown field cache_salt"))
    result = nb_perf.run_plan(make_plan(loads=[2], seeds=[201, 202, 203]), tmp_path)
    assert result["cache_salt"] == "dropped"
    assert result["loads"]["2"]["repeats"] == 3
    assert all("cache_salt" not in c[-1] for c in fake.calls[2:])


def test_a_timeout_removes_the_container(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def slow(argv: list[str], timeout: float = 0, env: dict[str, str] | None = None) -> Any:
        calls.append(argv)
        if argv[1] == "run":
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(nb_common, "run_command", slow)
    verdict, why, doc = nb_perf.run_once(make_plan(), tmp_path, 2, 1, "s")
    assert (verdict, doc) == ("tool_failed", None) and "timed out" in why
    assert calls[-1][:3] == ["docker", "rm", "-f"]


def test_a_kubernetes_run_fetches_the_export_from_the_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def kubectl(argv: list[str], timeout: float = 0, env: dict[str, str] | None = None) -> Any:
        calls.append(argv)
        if "cat" in argv:
            return subprocess.CompletedProcess(argv, 0, json.dumps(export_for(40)), "")
        return subprocess.CompletedProcess(argv, 0, "console", "")

    monkeypatch.setattr(nb_common, "run_command", kubectl)
    plan = make_plan(client_target="kubernetes", namespace="ns", pod="pod-1", client_image=None)
    verdict, _why, doc = nb_perf.run_once(plan, tmp_path, 2, 1, "s")
    assert verdict == "ok" and doc is not None
    assert calls[0][:6] == ["kubectl", "exec", "-n", "ns", "pod-1", "--"]
    adir = nb_perf.artifact_dir_for(plan, tmp_path, "perf/c2-r1")
    assert adir.startswith("/tmp/nim-benchmark/") and adir.endswith("/perf/c2-r1")
    assert calls[1][-1] == f"{adir}/profile_export_aiperf.json"
    # the raw prompt and answer files are removed in the pod too
    assert calls[2][6:] == ["rm", "-f", f"{adir}/inputs.json", f"{adir}/profile_export.jsonl"]


def k8s_plan() -> dict[str, Any]:
    return make_plan(client_target="kubernetes", namespace="ns", pod="pod-1", client_image=None)


def test_the_baseline_never_shares_a_pod_folder_with_the_candidate(tmp_path: Path) -> None:
    plan = k8s_plan()
    candidate = nb_perf.artifact_dir_for(plan, tmp_path, "perf/c2-r1")
    baseline = nb_perf.artifact_dir_for(plan, tmp_path / "baseline", "perf/c2-r1")
    assert candidate != baseline


def test_an_old_export_never_counts_for_a_new_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = tmp_path / "perf" / "c2-r1" / nb_perf.EXPORT_NAME
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps(export_for(40)))

    # docker or local: the new AIPerf writes nothing, the old export is gone
    install(monkeypatch, FakeDocker(lambda *_a: None))
    verdict, why, _doc = nb_perf.run_once(make_plan(), tmp_path, 2, 1, None)
    assert verdict == "tool_failed" and "no export" in why

    # kubernetes: the pod export is deleted first, and a failed fetch is not the old file
    stale.write_text(json.dumps(export_for(40)))
    calls: list[list[str]] = []

    def kubectl(argv: list[str], timeout: float = 0, env: dict[str, str] | None = None) -> Any:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1 if "cat" in argv else 0, "", "")

    monkeypatch.setattr(nb_common, "run_command", kubectl)
    plan = k8s_plan()
    verdict, why, _doc = nb_perf.run_once(plan, tmp_path, 2, 1, None)
    assert verdict == "tool_failed" and "Pod export read failed (exit status 1)" in why
    adir = nb_perf.artifact_dir_for(plan, tmp_path, "perf/c2-r1")
    assert calls[0][-1].startswith(f"rm -f {adir}/{nb_perf.EXPORT_NAME} && aiperf profile")


def test_a_kubernetes_timeout_stops_aiperf_and_cleans_the_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def slow(argv: list[str], timeout: float = 0, env: dict[str, str] | None = None) -> Any:
        calls.append(argv)
        if argv[-2:-1] == ["-c"]:
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(nb_common, "run_command", slow)
    plan = k8s_plan()
    adir = nb_perf.artifact_dir_for(plan, tmp_path, "perf/c2-r1")
    verdict, why, doc = nb_perf.run_once(plan, tmp_path, 2, 1, None)
    assert (verdict, doc) == ("tool_failed", None) and why == "AIPerf timed out"
    assert calls[1][6:] == ["python3", "-c", nb_perf.POD_STOP_SCRIPT, f"artifact-dir {adir}"]
    assert f"--artifact-dir {adir}" in calls[0][-1]  # the pattern matches this run only
    assert calls[2][6:8] == ["rm", "-f"] and f"{adir}/inputs.json" in calls[2]


def test_a_kubernetes_timeout_says_when_aiperf_may_still_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stop_fails(argv: list[str], timeout: float = 0, env: dict[str, str] | None = None) -> Any:
        if argv[-2:-1] == ["-c"]:
            raise subprocess.TimeoutExpired(argv, timeout)
        stop = nb_perf.POD_STOP_SCRIPT in argv
        return subprocess.CompletedProcess(argv, 127 if stop else 0, "", "")

    monkeypatch.setattr(nb_common, "run_command", stop_fails)
    _verdict, why, _doc = nb_perf.run_once(k8s_plan(), tmp_path, 2, 1, None)
    assert "may still run in the pod" in why


def test_a_load_is_complete_only_at_the_planned_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def one_bad(load: int, repeat: int, _a: int, _s: bool) -> dict[str, Any]:
        if (load, repeat) == (2, 5):
            return export_for(40, request_error_rate={"avg": 5.0})
        return good(load, 0, 0, False)

    install(monkeypatch, FakeDocker(one_bad))
    result = nb_perf.run_plan(make_plan(repeats=5), tmp_path)
    assert "2" not in result["loads"]
    assert result["incomplete"] == [{"load": 2, "counted": 4, "needed": 5}]
    assert result["loads"]["4"]["repeats"] == 5


def test_the_named_key_value_is_masked_in_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NIM_EP_BEARER", "endpoint-bearer-value-42")
    monkeypatch.setattr(nb_common, "SCRUBBER", nb_common.Scrubber())
    install(monkeypatch, FakeDocker(good, console="sent endpoint-bearer-value-42"))
    nb_perf.run_plan(make_plan(loads=[2], key_env="NIM_EP_BEARER"), tmp_path)
    log = (tmp_path / "perf" / "c2-r1" / nb_perf.LOG_NAME).read_text()
    assert "endpoint-bearer-value-42" not in log and nb_common.MASK in log


def test_script_prints_the_defaults() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "nb_perf.py"), "--defaults"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    facts = json.loads(proc.stdout)["facts"]
    assert facts["loads"] == [8, 32, 128] and facts["repeats"] == 3


def vision_plan(**changes: Any) -> dict[str, Any]:
    return make_plan(
        images_per_request=2,
        image_width=224,
        image_height=224,
        image_format="jpeg",
        image_source="noise",
        **changes,
    )


def test_a_text_plan_has_no_image_settings_and_a_vision_plan_does() -> None:
    text = make_plan()
    assert text["images_per_request"] == 0
    assert text["image_width"] is None and text["image_source"] is None
    built = nb_perf.build_plan(
        endpoint="http://x",
        model="m",
        tokenizer="t",
        loads=[2],
        repeats=3,
        isl=64,
        osl=32,
        key_env=None,
        target="local",
        image="i",
        namespace=None,
        pod=None,
        images=2,
    )
    assert built["images_per_request"] == 2
    assert built["image_width"] == built["image_height"] == 512
    assert built["image_format"] == "jpeg" and built["image_source"] == "noise"
    assert nb_common.plan_id(built) != nb_common.plan_id({**built, "images_per_request": 3})


def test_image_flags_reach_aiperf_only_for_a_vision_model() -> None:
    plain = " ".join(nb_perf.aiperf_args(make_plan(), 2, 1, "/o", None))
    assert "--image" not in plain
    args = nb_perf.aiperf_args(vision_plan(), 2, 1, "/o", None)
    pairs = {
        a: args[i + 1]
        for i, a in enumerate(args[:-1])
        if a.startswith("--") and not args[i + 1].startswith("--")
    }
    assert pairs["--image-batch-size"] == "2"
    assert pairs["--image-width-mean"] == pairs["--image-height-mean"] == "224"
    assert pairs["--image-width-stddev"] == pairs["--image-height-stddev"] == "0"
    assert pairs["--image-format"] == "jpeg" and pairs["--image-source"] == "noise"


def test_runs_with_different_images_cannot_be_compared() -> None:
    fields = {"images_per_request", "image_width", "image_height", "image_format", "image_source"}
    assert fields <= set(nb_perf.MATCH_FIELDS)


def test_a_vision_run_counts_like_a_text_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, FakeDocker(good))
    result = nb_perf.run_plan(vision_plan(loads=[2], seeds=[201, 202, 203]), tmp_path)
    assert result["problems"] == [] and result["loads"]["2"]["repeats"] == 3
    assert all("--image-batch-size 2" in c[-1] for c in fake.calls)


@pytest.mark.parametrize("version", [None, "0.12.0"])
def test_export_must_identify_the_pinned_aiperf_version(version: str | None) -> None:
    """A compatible export schema does not prove the required tool version ran."""
    status, why = nb_perf.check_export(export_for(40, aiperf_version=version), 40)
    assert status == "tool_failed" and "aiperf_version" in why


def test_cleanup_attempts_all_raw_files_before_reporting_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One protected raw file must not prevent deletion of another raw file."""
    for name in nb_perf.RAW_FILES:
        (tmp_path / name).write_text("private")
    unlink = Path.unlink

    def remove(path: Path, missing_ok: bool = False) -> None:
        if path.name == "inputs.json":
            raise PermissionError("read-only file")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", remove)
    with pytest.raises(PermissionError):
        nb_perf.drop_raw_files(tmp_path)
    assert not (tmp_path / "profile_export.jsonl").exists()



@pytest.mark.parametrize("timed_out", [False, True])
def test_remote_cleanup_failures_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timed_out: bool
) -> None:
    """A pod cleanup error cannot be reported as a clean successful attempt."""

    def command(argv: list[str], timeout: float = 0, env: Any = None) -> Any:
        if "rm" in argv:
            return subprocess.CompletedProcess(argv, 1, "", "permission denied")
        if "cat" in argv:
            return subprocess.CompletedProcess(argv, 0, json.dumps(export_for(40)), "")
        if argv[6:8] == ["sh", "-c"] and timed_out:
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(nb_common, "run_command", command)
    status, why, doc = nb_perf.run_once(k8s_plan(), tmp_path, 2, 1, None)
    assert status == "tool_failed" and "cleanup" in why.lower() and doc is None


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_a_failed_stop_preserves_all_errors_and_prevents_more_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool
) -> None:
    """An unconfirmed stop must halt every remaining load, even when cleanup also fails."""
    starts: list[list[str]] = []

    def command(argv: list[str], timeout: float = 0, env: Any = None) -> Any:
        if argv[6:8] == ["sh", "-c"]:
            starts.append(argv)
            raise subprocess.TimeoutExpired(argv, timeout)
        if nb_perf.POD_STOP_SCRIPT in argv or (cleanup_fails and "rm" in argv):
            return subprocess.CompletedProcess(argv, 1, "", "access denied")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def cleanup(_folder: Path) -> None:
        raise PermissionError("local file is protected")

    monkeypatch.setattr(nb_common, "run_command", command)
    if cleanup_fails:
        monkeypatch.setattr(nb_perf, "drop_raw_files", cleanup)
    result = nb_perf.run_plan(k8s_plan(), tmp_path)
    message = result["problems"][0]["message"]
    assert "may still run in the pod; stop it" in message
    if cleanup_fails:
        assert "local artifact cleanup failed" in message
        assert "Kubernetes artifact cleanup failed" in message
    assert len(starts) == 1
    assert result["loads"] == {}
    assert result["incomplete"] == [
        {"load": 2, "counted": 0, "needed": 3},
        {"load": 4, "counted": 0, "needed": 3},
    ]


def test_a_failed_docker_removal_prevents_more_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed Docker removal must not certify that a timed-out load client stopped."""
    starts: list[list[str]] = []

    def command(argv: list[str], timeout: float = 0, env: Any = None) -> Any:
        if argv[:2] == ["docker", "run"]:
            starts.append(argv)
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, 1, "", "cannot remove container")

    monkeypatch.setattr(nb_common, "run_command", command)
    result = nb_perf.run_plan(make_plan(), tmp_path)
    assert len(starts) == 1
    assert "may still run in the container; stop it" in result["problems"][0]["message"]
    assert result["loads"] == {}


@pytest.mark.parametrize(
    ("target", "failure", "stopped"),
    [
        ("kubernetes", "timeout", True),
        ("kubernetes", "timeout", False),
        ("docker", "timeout", False),
        ("local", "cleanup", True),
        ("kubernetes", "cleanup", True),
        ("kubernetes", "timeout_and_cleanup", False),
    ],
)
def test_failure_fixes_name_the_required_recovery_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    target: str, failure: str, stopped: bool
) -> None:
    """Failures need recovery steps that match the stopped client and remaining raw files."""
    plan = make_plan(client_target=target, namespace="default", pod="bench-client", loads=[2])

    def command(argv: list[str], timeout: float = 0, env: Any = None) -> Any:
        if nb_perf.POD_STOP_SCRIPT in argv or argv[:3] == ["docker", "rm", "-f"]:
            return subprocess.CompletedProcess(argv, 0 if stopped else 1, "", "")
        if "rm" in argv:
            return subprocess.CompletedProcess(argv, 1 if "cleanup" in failure else 0, "", "")
        if "timeout" in failure:
            raise subprocess.TimeoutExpired(argv, timeout)
        if target == "kubernetes" and "cat" in argv:
            return subprocess.CompletedProcess(argv, 0, json.dumps(export_for(40)), "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    unlink = Path.unlink

    def remove(path: Path, missing_ok: bool = False) -> None:
        if "cleanup" in failure and path.name in nb_perf.RAW_FILES:
            raise PermissionError("raw file is protected")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(nb_common, "run_command", command)
    monkeypatch.setattr(Path, "unlink", remove)
    result = nb_perf.run_plan(plan, tmp_path)
    fix = result["problems"][0]["fix"]
    assert "aiperf.log" not in fix
    if not stopped:
        assert "Stop" in fix and "before" in fix and "rerun" in fix
        if target == "kubernetes":
            assert "default/bench-client" in fix
        else:
            assert "nim-benchmark-abc123-c2-r1" in fix
    if "cleanup" in failure:
        assert "Restore" in fix and "raw" in fix
        assert str(tmp_path / "perf/c2-r1") in fix
        if target == "kubernetes":
            assert "default/bench-client" in fix
    elif stopped:
        assert "timeout" in fix.lower()
        assert not (tmp_path / "perf/c2-r1/aiperf.log").exists()


def test_other_failures_keep_the_saved_log_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed export still points to the console log produced by that attempt."""
    fake = install(monkeypatch, FakeDocker(lambda *_: export_for(40, schema_version="2.0")))
    result = nb_perf.run_plan(make_plan(loads=[2]), tmp_path)
    assert fake.calls
    log = tmp_path / "perf/c2-r1/aiperf.log"
    assert log.exists() and result["problems"][0]["fix"] == f"Read {log}"


@pytest.mark.parametrize("target", ["local", "docker", "kubernetes"])
def test_startup_oserror_has_a_client_fix_instead_of_a_missing_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """A process that never started cannot have produced this attempt's console log."""
    def command(argv: list[str], timeout: float = 0, env: Any = None) -> Any:
        if target == "kubernetes" and "rm" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise PermissionError("client cannot execute")

    monkeypatch.setattr(nb_common, "run_command", command)
    plan = make_plan(client_target=target, namespace="default", pod="bench-client", loads=[2])
    result = nb_perf.run_plan(plan, tmp_path)
    problem = result["problems"][0]
    assert problem["code"] == "tool_failed"
    assert "aiperf.log" not in problem["fix"]
    assert "executable" in problem["fix"] and "permissions" in problem["fix"]
    assert not (tmp_path / "perf/c2-r1/aiperf.log").exists()


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs Linux process state")
@pytest.mark.parametrize("behavior", ["exit", "delayed", "ignore", "unrelated"])
def test_pod_stop_confirms_termination_of_the_selected_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, behavior: str
) -> None:
    """A successful signal delivery is not confirmation that a load process has exited."""
    import time

    ready = tmp_path / "ready"
    marker = f"artifact-dir {tmp_path}/selected"
    program = '''
import signal, sys, time
from pathlib import Path
if sys.argv[2] == "ignore":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
elif sys.argv[2] == "delayed":
    def finish(*args):
        time.sleep(0.25)
        sys.exit(0)
    signal.signal(signal.SIGTERM, finish)
Path(sys.argv[1]).touch()
while True:
    time.sleep(0.1)
'''
    proc = subprocess.Popen([sys.executable, "-c", program, str(ready), behavior, marker])
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()

        def local_pod(_plan: Any, *argv: str) -> Any:
            return subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)

        monkeypatch.setattr(nb_perf, "pod_exec", local_pod)
        artifact_dir = str(tmp_path / ("unrelated" if behavior == "unrelated" else "selected"))
        stopped = nb_perf.stop_client(k8s_plan(), "unused", artifact_dir)
        if behavior == "ignore":
            assert not stopped and proc.poll() is None
        elif behavior == "unrelated":
            assert stopped and proc.poll() is None
        else:
            assert stopped and proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=3)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs Linux process state")
def test_pod_stop_includes_worker_descendants_without_the_artifact_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A load worker cannot be left running just because its argv omits its parent's folder."""
    import os
    import signal
    import time

    ready = tmp_path / "worker-pid"
    marker = f"artifact-dir {tmp_path}/selected"
    program = '''
import subprocess, sys, time
from pathlib import Path
worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
ready = Path(sys.argv[1])
pending = ready.with_suffix(".tmp")
pending.write_text(str(worker.pid))
pending.replace(ready)
while True:
    time.sleep(0.1)
'''
    proc = subprocess.Popen([sys.executable, "-c", program, str(ready), marker])
    worker_pid = None
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        worker_pid = int(ready.read_text())

        def local_pod(_plan: Any, *argv: str) -> Any:
            return subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)

        monkeypatch.setattr(nb_perf, "pod_exec", local_pod)
        assert nb_perf.stop_client(k8s_plan(), "unused", str(tmp_path / "selected"))
        state_path = Path(f"/proc/{worker_pid}/stat")
        try:
            state = state_path.read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            pass  # the worker exited and its proc entry was removed
        else:
            assert state in ("Z", "X")
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=3)
        if worker_pid is not None:
            try:
                os.kill(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("stage", ["prepare", "save", "read"])
def test_export_io_errors_return_output_recovery_instead_of_escaping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """Preparing, saving, and reading the current export all need the same safe I/O boundary."""
    def command(argv: list[str], timeout: float = 0, env: Any = None) -> Any:
        return subprocess.CompletedProcess(argv, 0, json.dumps(export_for(40)) if "cat" in argv else "", "")

    write_text = nb_common.write_text
    read_json = nb_common.read_json
    unlink = Path.unlink

    def write(path: Path, text: str) -> None:
        if stage == "save" and path.name == nb_perf.EXPORT_NAME:
            raise PermissionError("cannot save export")
        write_text(path, text)

    def read(path: Path) -> Any:
        if stage == "read" and path.name == nb_perf.EXPORT_NAME:
            raise PermissionError("cannot read export")
        return read_json(path)

    def remove(path: Path, missing_ok: bool = False) -> None:
        if stage == "prepare" and path.name == nb_perf.EXPORT_NAME:
            raise PermissionError("cannot clear old export")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(nb_common, "run_command", command)
    monkeypatch.setattr(nb_common, "write_text", write)
    monkeypatch.setattr(nb_common, "read_json", read)
    monkeypatch.setattr(Path, "unlink", remove)
    result = nb_perf.run_plan(k8s_plan(), tmp_path)
    problem = result["problems"][0]
    assert problem["code"] == "tool_failed"
    assert "output-folder permissions" in problem["fix"]
    assert result["loads"] == {}


@pytest.mark.parametrize(
    ("status", "failure", "missing"),
    [
        (1, "permission", False),
        (126, "exec", False),
        (137, "empty", False),
        (1, "gnu_missing", True),
        (1, "busybox_missing", True),
        (127, "reader_missing", False),
    ],
)
def test_pod_export_read_failure_preserves_status_and_access_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    status: int, failure: str, missing: bool
) -> None:
    """A failed read is not proof that AIPerf failed to produce its export."""
    calls: list[list[str]] = []

    def command(argv: list[str], timeout: float = 0, env: Any = None) -> Any:
        calls.append(argv)
        if "cat" in argv:
            path = argv[-1]
            stderr = {
                "permission": f"cat: {path}: Permission denied\nprivate diagnostic",
                "exec": "error: exec access was denied\nprivate diagnostic",
                "empty": "",
                "gnu_missing": f"cat: {path}: No such file or directory",
                "busybox_missing": f"cat: can't open '{path}': No such file or directory",
                "reader_missing": "exec: cat: No such file or directory",
            }[failure]
            return subprocess.CompletedProcess(argv, status, "", stderr)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(nb_common, "run_command", command)
    plan = k8s_plan()
    result = nb_perf.run_plan(plan, tmp_path)
    problem = result["problems"][0]
    assert result["loads"] == {} and problem["code"] == "tool_failed"
    if missing:
        assert "AIPerf wrote no export file" in problem["message"]
    else:
        assert f"exit status {status}" in problem["message"]
        assert "AIPerf wrote no export file" not in problem["message"]
        assert "ns/pod-1" in problem["fix"]
        assert nb_perf.artifact_dir_for(plan, tmp_path, "perf/c2-r1") in problem["fix"]
        assert "read" in problem["fix"] and "access" in problem["fix"]
        assert "aiperf.log" not in problem["fix"]
        assert "private diagnostic" not in json.dumps(problem)
    assert any(call[6:8] == ["rm", "-f"] for call in calls)
