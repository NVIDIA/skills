# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/nimbench.py: the commands, the approval gate and whole runs."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from secrets import token_hex
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import nb_common  # noqa: E402
import nimbench  # noqa: E402
from test_nb_accuracy import IMAGE, FakeGym, items  # noqa: E402
from test_nb_perf import FakeDocker, export_for  # noqa: E402

CAND, BASE = "http://gpu-01:8000/v1", "http://gpu-02:8000/v1"


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    """Run main() in this process. Returns the exit code and the one JSON object it printed."""
    with pytest.raises(SystemExit) as stop:
        nimbench.main(argv)
    out = capsys.readouterr().out
    return int(stop.value.code or 0), json.loads(out)


@pytest.fixture(autouse=True)
def ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both endpoints answer, and Docker is installed."""
    monkeypatch.setattr(nb_common, "check_endpoint", lambda base, model, key=None: ([model], []))
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")


def perf_args(out: Path, *extra: str) -> list[str]:
    return [
        "perf",
        "--endpoint",
        CAND,
        "--model",
        "qwen7b",
        "--tokenizer",
        "tok",
        "--loads",
        "8",
        "--out",
        str(out),
        *extra,
    ]


def acc_args(out: Path, *extra: str) -> list[str]:
    return [
        "accuracy",
        "--endpoint",
        CAND,
        "--model",
        "qwen7b",
        "--gym-image",
        IMAGE,
        "--sets",
        "gsm8k",
        "--out",
        str(out),
        *extra,
    ]


class UrlDocker(FakeDocker):
    """A fake AIPerf whose speed depends on the endpoint URL it was pointed at."""

    def __call__(self, argv: list[str], timeout: float = 0, env: Any = None) -> Any:
        found = re.search(r"--url (\S+)", argv[-1])
        self.url = found.group(1) if found else ""
        return super().__call__(argv, timeout, env)

    url = ""


def speed(fake_url: Callable[[], str], cand_ttft: float, base_ttft: float) -> Any:
    def behavior(load: int, repeat: int, _a: int, _s: bool) -> dict[str, Any]:
        ttft = cand_ttft if "gpu-01" in fake_url() else base_ttft
        doc = export_for(max(40, 10 * load))
        doc["time_to_first_token"]["p99"] = ttft + repeat * 0.1
        doc["time_to_first_token"]["avg"] = ttft + repeat * 0.1
        return doc

    return behavior


from collections.abc import Callable  # noqa: E402


def install_perf(monkeypatch: pytest.MonkeyPatch, cand: float, base: float) -> UrlDocker:
    holder: dict[str, UrlDocker] = {}
    fake = UrlDocker(speed(lambda: holder["f"].url, cand, base))
    holder["f"] = fake
    monkeypatch.setattr(nb_common, "run_command", fake)
    return fake


def test_a_dry_run_prints_a_stable_plan_id_and_sends_no_load(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_perf(monkeypatch, 40, 40)
    code1, first = run(perf_args(tmp_path / "a", "--dry-run"), capsys)
    code2, second = run(perf_args(tmp_path / "a", "--dry-run"), capsys)
    assert code1 == code2 == 0 and first["state"] == "planned"
    assert first["facts"]["plan_id"] == second["facts"]["plan_id"]
    assert first["next"] == "ask_user_to_approve" and fake.calls == []
    assert not (tmp_path / "a").exists()
    _c, other = run(perf_args(tmp_path / "a", "--loads", "16", "--dry-run"), capsys)
    assert other["facts"]["plan_id"] != first["facts"]["plan_id"]


def test_the_plan_id_covers_the_baseline_endpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _c, without = run(perf_args(tmp_path / "a", "--dry-run"), capsys)
    _c, with_base = run(perf_args(tmp_path / "a", "--baseline", BASE, "--dry-run"), capsys)
    _c, other_base = run(
        perf_args(tmp_path / "a", "--baseline", "http://gpu-03:8000/v1", "--dry-run"), capsys
    )
    ids = {
        without["facts"]["plan_id"],
        with_base["facts"]["plan_id"],
        other_base["facts"]["plan_id"],
    }
    assert len(ids) == 3
    assert with_base["facts"]["plan"]["baseline"]["endpoint"] == "http://gpu-02:8000"


def test_a_live_run_without_the_approved_plan_id_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_perf(monkeypatch, 40, 40)
    for extra in ([], ["--approved", "wrong-id"]):
        code, payload = run(perf_args(tmp_path / "a", *extra), capsys)
        assert code == 5 and payload["errors"][0]["code"] == "not_approved"
    assert fake.calls == [] and not (tmp_path / "a").exists()


def test_a_changed_option_voids_the_approval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_perf(monkeypatch, 40, 40)
    _c, plan = run(perf_args(tmp_path / "a", "--dry-run"), capsys)
    approved = plan["facts"]["plan_id"]
    code, payload = run(perf_args(tmp_path / "a", "--repeats", "4", "--approved", approved), capsys)
    assert code == 5 and payload["state"] == "not_approved" and fake.calls == []


def test_perf_with_a_baseline_endpoint_runs_both_and_gives_a_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_perf(monkeypatch, 40, 80)
    out = tmp_path / "run"
    _c, plan = run(perf_args(out, "--baseline", BASE, "--dry-run"), capsys)
    code, done = run(
        perf_args(out, "--baseline", BASE, "--approved", plan["facts"]["plan_id"]), capsys
    )
    assert code == 0 and done["state"] == "done", done["errors"]
    assert len(fake.calls) == 6  # 3 repeats on each of the two endpoints
    assert done["facts"]["verdicts"]["perf"]["loads"]["8"]["verdict"] == "better"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["perf"]["aiperf_version"] == "0.13.0" and manifest["plan_ids"]["perf"]
    assert "endpoint" in manifest and (out / "baseline" / "manifest.json").exists()
    report = (out / "report.md").read_text()
    assert "| 8 |" in report and "better" in report


def test_perf_without_a_baseline_reports_numbers_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_perf(monkeypatch, 40, 40)
    out = tmp_path / "run"
    _c, plan = run(perf_args(out, "--dry-run"), capsys)
    code, done = run(perf_args(out, "--approved", plan["facts"]["plan_id"]), capsys)
    assert code == 0 and done["facts"]["verdicts"] == {}
    assert "no baseline" in (out / "report.md").read_text()
    assert not (out / "baseline").exists()


def test_a_failed_tool_run_exits_4_and_still_writes_a_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nb_common, "run_command", FakeDocker(lambda *_a: None))
    out = tmp_path / "run"
    _c, plan = run(perf_args(out, "--dry-run"), capsys)
    code, done = run(perf_args(out, "--approved", plan["facts"]["plan_id"]), capsys)
    assert code == 4 and done["errors"][0]["code"] == "tool_failed"
    assert (out / "report.md").exists()


def test_a_saved_baseline_run_is_compared_and_a_mismatch_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_perf(monkeypatch, 80, 80)
    old = tmp_path / "old"
    _c, plan = run(perf_args(old, "--dry-run"), capsys)
    run(perf_args(old, "--approved", plan["facts"]["plan_id"]), capsys)

    install_perf(monkeypatch, 40, 40)
    new = tmp_path / "new"
    _c, plan = run(perf_args(new, "--baseline", str(old), "--dry-run"), capsys)
    code, done = run(
        perf_args(new, "--baseline", str(old), "--approved", plan["facts"]["plan_id"]), capsys
    )
    assert code == 0 and done["facts"]["verdicts"]["perf"]["loads"]["8"]["verdict"] == "better"

    other = tmp_path / "other"
    args = [*perf_args(other, "--baseline", str(old)), "--isl", "512"]
    _c, plan = run([*args, "--dry-run"], capsys)
    code, done = run([*args, "--approved", plan["facts"]["plan_id"]], capsys)
    assert code == 5 and done["errors"][0]["code"] == "baseline_mismatch"
    assert "isl" in done["errors"][0]["message"]
    assert (other / "report.md").exists()  # the numbers are still reported, without a verdict


def install_gym(monkeypatch: pytest.MonkeyPatch, cand_wrong: int, base_wrong: int) -> FakeGym:
    holder: dict[str, FakeGym] = {}

    def make(_name: str) -> tuple[list[dict[str, Any]], Any]:
        url = holder["f"].calls[-1][holder["f"].calls[-1].index("--model-url") + 1]
        return items(1000, wrong=cand_wrong if "gpu-01" in url else base_wrong)

    fake = FakeGym(make)
    holder["f"] = fake
    monkeypatch.setattr(nb_common, "run_command", fake)
    return fake


def test_accuracy_with_a_baseline_endpoint_gives_a_paired_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_gym(monkeypatch, cand_wrong=130, base_wrong=100)
    out = tmp_path / "run"
    _c, plan = run(acc_args(out, "--baseline", BASE, "--dry-run"), capsys)
    code, done = run(
        acc_args(out, "--baseline", BASE, "--approved", plan["facts"]["plan_id"]), capsys
    )
    assert code == 0, done["errors"]
    assert len(fake.calls) == 2
    sets = done["facts"]["verdicts"]["accuracy"]["sets"]
    assert sets["gsm8k"]["verdict"] == "fail" and sets["gsm8k"]["difference"] == -3.0
    assert done["facts"]["sets"]["gsm8k"]["score"] == 87.0
    assert (out / "baseline" / "accuracy" / "summary.json").exists()


def test_accuracy_without_a_baseline_reports_scores_with_ranges(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_gym(monkeypatch, 100, 100)
    out = tmp_path / "run"
    _c, plan = run(acc_args(out, "--dry-run"), capsys)
    code, done = run(acc_args(out, "--approved", plan["facts"]["plan_id"]), capsys)
    entry = done["facts"]["sets"]["gsm8k"]
    assert code == 0 and entry["low"] < entry["score"] < entry["high"]
    assert done["facts"]["verdicts"] == {}
    assert "no baseline" in (out / "report.md").read_text()


def test_perf_and_accuracy_share_one_run_folder_and_one_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "run"
    install_perf(monkeypatch, 40, 40)
    _c, plan = run(perf_args(out, "--dry-run"), capsys)
    run(perf_args(out, "--approved", plan["facts"]["plan_id"]), capsys)
    install_gym(monkeypatch, 100, 100)
    _c, plan = run(acc_args(out, "--dry-run"), capsys)
    run(acc_args(out, "--approved", plan["facts"]["plan_id"]), capsys)
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["plan_ids"]) == {"perf", "accuracy"}
    report = (out / "report.md").read_text()
    assert "## Speed" in report and "## Accuracy" in report


def test_compare_checks_two_run_folders(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_gym(monkeypatch, 100, 100)
    for name in ("a", "b"):
        _c, plan = run(acc_args(tmp_path / name, "--dry-run"), capsys)
        run(acc_args(tmp_path / name, "--approved", plan["facts"]["plan_id"]), capsys)
    code, done = run(["compare", str(tmp_path / "a"), "--baseline", str(tmp_path / "b")], capsys)
    assert code == 0 and done["facts"]["verdicts"]["accuracy"]["sets"]["gsm8k"]["verdict"] == "pass"
    code, done = run(["compare", str(tmp_path / "a"), "--baseline", BASE], capsys)
    assert code == 2 and "run folders" in done["errors"][0]["message"]


@pytest.mark.parametrize(
    "bad",
    ["hf_abcdefghijklmnopqrstuvwx", "sk-abcdefghijklmnopqrstuvwxyz", "has space", "1BAD"],
)
def test_a_key_given_as_a_value_is_refused(
    bad: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = run(perf_args(tmp_path / "a", "--api-key-env", bad, "--dry-run"), capsys)
    assert code == 2 and payload["errors"][0]["code"] == "literal_secret"
    assert bad not in json.dumps(payload)


def test_a_credential_in_any_flag_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Generated at runtime, so the source holds no credential-like value for the PII scan.
    secret = "hf_" + token_hex(12)
    code, payload = run(perf_args(tmp_path / "a", "--tokenizer", secret, "--dry-run"), capsys)
    assert code == 2 and payload["errors"][0]["code"] == "literal_secret"


@pytest.mark.parametrize(
    ("extra", "code_word"),
    [
        (["--repeats", "2"], "Use at least 3"),
        (["--loads", "8,x"], "positive whole numbers"),
        (["--loads", "0"], "positive whole numbers"),
        (["--client", "kubernetes"], "--namespace and --pod"),
    ],
)
def test_bad_performance_flags_are_refused_before_any_work(
    extra: list[str], code_word: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = run([*perf_args(tmp_path / "a"), *extra, "--dry-run"], capsys)
    assert code == 2 and payload["errors"][0]["code"] == "bad_input"
    assert code_word in payload["errors"][0]["message"] + payload["errors"][0]["fix"]


def test_bad_accuracy_flags_are_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    unpinned = acc_args(tmp_path / "a", "--dry-run")
    unpinned[unpinned.index("--gym-image") + 1] = "nvcr.io/nvidia/nemo-gym:latest"
    code, payload = run(unpinned, capsys)
    assert code == 2 and payload["errors"][0]["code"] == "image_not_pinned"
    code, payload = run([*acc_args(tmp_path / "a"), "--sets", "gsm8k,nope", "--dry-run"], capsys)
    assert code == 2 and "nope" in payload["errors"][0]["message"]
    code, payload = run([*acc_args(tmp_path / "a"), "--limit", "0", "--dry-run"], capsys)
    assert code == 2


def test_accuracy_needs_docker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    code, payload = run(acc_args(tmp_path / "a", "--dry-run"), capsys)
    assert code == 3 and payload["errors"][0]["code"] == "docker_missing"


def test_an_unreachable_or_wrong_model_stops_the_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = [nb_common.error("model_not_served", "qwen7b is not served", "Use a served name")]
    monkeypatch.setattr(nb_common, "check_endpoint", lambda base, model, key=None: (["x"], bad))
    code, payload = run(perf_args(tmp_path / "a", "--dry-run"), capsys)
    assert code == 3 and payload["errors"][0]["code"] == "model_not_served"
    assert payload["next"] == "fix_errors"


def test_a_baseline_run_folder_without_a_manifest_is_bad_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = run(perf_args(tmp_path / "a", "--baseline", str(tmp_path), "--dry-run"), capsys)
    assert code == 3 and "no manifest.json" in payload["errors"][0]["message"]


def test_local_and_kubernetes_clients_need_their_tools(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    code, payload = run(perf_args(tmp_path / "a", "--client", "local", "--dry-run"), capsys)
    assert code == 3 and payload["errors"][0]["code"] == "client_missing"
    args = [*perf_args(tmp_path / "a"), "--client", "kubernetes", "--namespace", "n", "--pod", "p"]
    code, payload = run([*args, "--dry-run"], capsys)
    assert code == 3 and "kubectl" in payload["errors"][0]["message"]


def test_check_lists_the_served_models_and_sends_no_load(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_perf(monkeypatch, 40, 40)
    code, payload = run(["check", "--endpoint", CAND, "--model", "qwen7b"], capsys)
    assert code == 0 and payload["facts"]["served_models"] == ["qwen7b"]
    assert payload["facts"]["docker"] is True and fake.calls == []


def test_the_output_never_holds_a_key_from_the_environment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Generated at runtime, so the source holds no credential-like value for the PII scan.
    secret = "scrub-me-" + token_hex(8)
    monkeypatch.setenv("MY_API_KEY", secret)
    nb_common.SCRUBBER.add(secret)
    fake = install_perf(monkeypatch, 40, 40)
    args = perf_args(tmp_path / "a", "--api-key-env", "MY_API_KEY")
    _c, plan = run([*args, "--dry-run"], capsys)
    code, done = run([*args, "--approved", plan["facts"]["plan_id"]], capsys)
    assert code == 0
    everything = json.dumps(done) + json.dumps(plan)
    everything += "".join(
        p.read_text(errors="ignore") for p in (tmp_path / "a").rglob("*") if p.is_file()
    )
    assert secret not in everything
    assert all(secret not in " ".join(call) for call in fake.calls)
    assert any("MY_API_KEY" in " ".join(call) for call in fake.calls)  # the name, not the value


def test_script_help_works_and_a_bad_command_exits_2() -> None:
    script = str(SCRIPTS / "nimbench.py")
    helped = subprocess.run(
        [sys.executable, script, "--help"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert helped.returncode == 0 and "perf" in helped.stdout
    bad = subprocess.run(
        [sys.executable, script, "nope"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert bad.returncode == 2 and json.loads(bad.stdout)["errors"][0]["code"] == "bad_input"


JUDGE_FLAGS = ["--judge-endpoint", "http://judge:8000/v1", "--judge-model", "big-judge"]


def vlm_args(out: Path, *extra: str) -> list[str]:
    return [
        "accuracy",
        "--endpoint",
        CAND,
        "--model",
        "qwen7b",
        "--gym-image",
        IMAGE,
        "--vlm",
        "--out",
        str(out),
        *extra,
    ]


def test_a_vision_speed_plan_carries_the_image_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    flags = ["--images-per-request", "2", "--image-width", "224", "--image-height", "224"]
    code, plan = run(perf_args(tmp_path / "a", *flags, "--dry-run"), capsys)
    assert code == 0
    facts = plan["facts"]["plan"]
    assert facts["images_per_request"] == 2 and facts["image_width"] == 224
    assert facts["image_format"] == "jpeg" and facts["image_source"] == "noise"
    _c, text = run(perf_args(tmp_path / "a", "--dry-run"), capsys)
    assert text["facts"]["plan"]["images_per_request"] == 0
    assert text["facts"]["plan_id"] != plan["facts"]["plan_id"]


@pytest.mark.parametrize(
    "extra",
    [["--image-width", "224"], ["--image-height", "224"], ["--images-per-request", "-1"]],
)
def test_image_flags_need_a_vision_model_and_sane_values(
    extra: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = run(perf_args(tmp_path / "a", *extra, "--dry-run"), capsys)
    assert code == 2 and payload["errors"][0]["code"] == "bad_input"


def test_a_zero_image_size_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    flags = ["--images-per-request", "1", "--image-width", "0"]
    code, payload = run(perf_args(tmp_path / "a", *flags, "--dry-run"), capsys)
    assert code == 2 and "1 or more" in payload["errors"][0]["message"]


def test_vlm_defaults_to_the_vision_sets_and_asks_for_a_judge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = run(vlm_args(tmp_path / "a", "--dry-run"), capsys)
    assert code == 2 and payload["errors"][0]["code"] == "judge_required"
    assert "labbench2_vlm, gdp_pdf" in payload["errors"][0]["message"]
    code, payload = run(vlm_args(tmp_path / "a", *JUDGE_FLAGS, "--dry-run"), capsys)
    assert code == 0
    plan = payload["facts"]["plan"]
    assert plan["sets"] == ["labbench2_vlm", "gdp_pdf"]
    assert plan["judge"] == {"endpoint": "http://judge:8000/v1", "model": "big-judge"}


def test_text_sets_are_the_default_without_vlm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = acc_args(tmp_path / "a", "--dry-run")
    del args[args.index("--sets") : args.index("--sets") + 2]
    _c, payload = run(args, capsys)
    assert payload["facts"]["plan"]["sets"] == ["gsm8k", "ifeval", "mmlu_pro"]
    assert payload["facts"]["plan"]["judge"] is None


def test_the_judge_flags_are_refused_when_no_selected_set_uses_a_judge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = run(acc_args(tmp_path / "a", *JUDGE_FLAGS, "--dry-run"), capsys)
    assert code == 2 and "no selected test set uses a judge" in payload["errors"][0]["message"]


def test_half_a_judge_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    flags = ["--judge-endpoint", "http://judge:8000/v1"]
    code, payload = run(vlm_args(tmp_path / "a", *flags, "--dry-run"), capsys)
    assert code == 2 and payload["errors"][0]["code"] == "judge_required"


@pytest.mark.parametrize(
    "extra",
    [
        ["--judge-endpoint", CAND, "--judge-model", "qwen7b"],
        ["--judge-endpoint", BASE, "--judge-model", "qwen7b", "--baseline", BASE],
        [
            "--judge-endpoint",
            BASE,
            "--judge-model",
            "other",
            "--baseline",
            BASE,
            "--baseline-model",
            "other",
        ],
    ],
)
def test_a_model_may_not_judge_itself(
    extra: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = run(vlm_args(tmp_path / "a", *extra, "--dry-run"), capsys)
    assert code == 2 and payload["errors"][0]["code"] == "self_judging"


def test_the_same_model_name_on_another_endpoint_may_judge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    flags = ["--judge-endpoint", "http://judge:8000/v1", "--judge-model", "qwen7b"]
    code, _payload = run(vlm_args(tmp_path / "a", *flags, "--dry-run"), capsys)
    assert code == 0


def test_the_judge_is_checked_before_any_load_and_named_in_the_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    down = [nb_common.error("endpoint_unreachable", "cannot read /v1/models", "Check the URL")]

    def check(base: str, model: str, key: str | None = None) -> tuple[list[str], Any]:
        return ([], down) if "judge" in base else ([model], [])

    monkeypatch.setattr(nb_common, "check_endpoint", check)
    code, payload = run(vlm_args(tmp_path / "a", *JUDGE_FLAGS, "--dry-run"), capsys)
    assert code == 3 and payload["errors"][0]["message"].startswith("judge: ")


def test_the_judge_is_part_of_the_plan_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ids = set()
    for model in ("judge-a", "judge-b"):
        flags = ["--judge-endpoint", "http://judge:8000/v1", "--judge-model", model]
        _c, payload = run(vlm_args(tmp_path / "a", *flags, "--dry-run"), capsys)
        ids.add(payload["facts"]["plan_id"])
    assert len(ids) == 2


def test_a_judge_key_given_as_a_value_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    flags = [*JUDGE_FLAGS, "--judge-api-key-env", "hf_abcdefghijklmnopqrstuvwx"]
    code, payload = run(vlm_args(tmp_path / "a", *flags, "--dry-run"), capsys)
    assert code == 2 and payload["errors"][0]["code"] == "literal_secret"


def test_a_vision_run_with_a_baseline_uses_one_judge_for_both_endpoints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_gym(monkeypatch, cand_wrong=130, base_wrong=100)
    monkeypatch.setenv("JUDGE_KEY", "judge-key-value-1234")
    nb_common.SCRUBBER.add("judge-key-value-1234")
    out = tmp_path / "run"
    args = vlm_args(out, *JUDGE_FLAGS, "--judge-api-key-env", "JUDGE_KEY", "--baseline", BASE)
    _c, plan = run([*args, "--dry-run"], capsys)
    code, done = run([*args, "--approved", plan["facts"]["plan_id"]], capsys)
    assert code == 0, done["errors"]
    assert len(fake.calls) == 4  # two vision sets on each of the two endpoints
    for call in fake.calls:
        assert "+judge_base_url=http://judge:8000/v1" in call
        assert "+judge_model_name=big-judge" in call
        assert "+judge_api_key=${oc.env:JUDGE_KEY}" in call
        assert not any("judge-key-value-1234" in part for part in call)
    sets = done["facts"]["verdicts"]["accuracy"]["sets"]
    assert {s["verdict"] for s in sets.values()} == {"fail"} and set(sets) == {
        "labbench2_vlm",
        "gdp_pdf",
    }
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["accuracy"]["judge"] == {
        "endpoint": "http://judge:8000/v1",
        "model": "big-judge",
    }
    assert "judge_key_env" not in manifest["accuracy"]
    for side in (out, out / "baseline"):
        for name in ("labbench2_vlm", "gdp_pdf"):
            assert not (side / "accuracy" / name / "rollouts.jsonl").exists()
            assert (side / "accuracy" / name / "scores.json").exists()


def test_one_failed_set_is_listed_without_a_verdict_and_the_report_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    holder: dict[str, FakeGym] = {}

    def make(name: str) -> tuple[list[dict[str, Any]], Any]:
        return items(1000, wrong=100, cut=200 if name == "ifeval" else 0)

    holder["f"] = FakeGym(make)
    monkeypatch.setattr(nb_common, "run_command", holder["f"])
    out = tmp_path / "run"
    args = [*acc_args(out, "--baseline", BASE), "--sets", "gsm8k,ifeval"]
    _c, plan = run([*args, "--dry-run"], capsys)
    code, done = run([*args, "--approved", plan["facts"]["plan_id"]], capsys)
    assert code == 4 and {e["code"] for e in done["errors"]} == {"answers_truncated"}
    sets = done["facts"]["verdicts"]["accuracy"]["sets"]
    assert sets["gsm8k"]["verdict"] == "pass" and sets["ifeval"]["verdict"] == "no verdict"
    report = (out / "report.md").read_text()
    assert "Test set ifeval has no score in this run. It has no verdict." in report


def test_a_rerun_replaces_the_verdicts_of_its_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "run"
    install_perf(monkeypatch, 40, 80)
    _c, plan = run(perf_args(out, "--baseline", BASE, "--dry-run"), capsys)
    run(
        perf_args(out, "--baseline", BASE, "--approved", plan["facts"]["plan_id"]),
        capsys,
    )
    install_gym(monkeypatch, cand_wrong=130, base_wrong=100)
    _c, plan = run(acc_args(out, "--baseline", BASE, "--dry-run"), capsys)
    run(
        acc_args(out, "--baseline", BASE, "--approved", plan["facts"]["plan_id"]),
        capsys,
    )
    saved = json.loads((out / "compare.json").read_text())
    assert saved["accuracy"]["sets"]["gsm8k"]["verdict"] == "fail" and "perf" in saved

    # the same folder again without a baseline: the old accuracy verdict must not survive
    _c, plan = run(acc_args(out, "--dry-run"), capsys)
    code, done = run(acc_args(out, "--approved", plan["facts"]["plan_id"]), capsys)
    assert code == 0 and done["facts"]["verdicts"] == {}
    saved = json.loads((out / "compare.json").read_text())
    assert "accuracy" not in saved and saved["perf"]["loads"]["8"]["verdict"] == "better"
    report = (out / "report.md").read_text()
    assert "| gsm8k |" in report and "fail" not in report and "no baseline" in report

    # a refused baseline clears the perf verdict, and the accuracy section is not compared
    old = tmp_path / "old"
    install_perf(monkeypatch, 80, 80)
    args = [*perf_args(old), "--isl", "512"]
    _c, plan = run([*args, "--dry-run"], capsys)
    run([*args, "--approved", plan["facts"]["plan_id"]], capsys)
    _c, plan = run(perf_args(out, "--baseline", str(old), "--dry-run"), capsys)
    code, done = run(
        perf_args(out, "--baseline", str(old), "--approved", plan["facts"]["plan_id"]),
        capsys,
    )
    assert code == 5 and [e["code"] for e in done["errors"]] == ["baseline_mismatch"]
    assert json.loads((out / "compare.json").read_text()) == {}
    assert "better" not in (out / "report.md").read_text()


def test_the_margin_is_part_of_the_plan_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_gym(monkeypatch, 100, 100)
    _c, plan = run(acc_args(tmp_path / "a", "--baseline", BASE, "--dry-run"), capsys)
    _c, wider = run(
        acc_args(tmp_path / "a", "--baseline", BASE, "--margin", "5", "--dry-run"),
        capsys,
    )
    assert plan["facts"]["plan_id"] != wider["facts"]["plan_id"]
    assert wider["facts"]["plan"]["margin"] == 5.0
    approved = plan["facts"]["plan_id"]
    code, payload = run(
        acc_args(tmp_path / "a", "--baseline", BASE, "--margin", "5", "--approved", approved),
        capsys,
    )
    assert code == 5 and payload["state"] == "not_approved" and fake.calls == []


def test_other_mode_verdicts_survive_only_while_their_results_match(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rerunning speed preserves a valid accuracy verdict but drops one with changed evidence."""
    install_gym(monkeypatch, 100, 100)
    args = acc_args(tmp_path, "--baseline", BASE)
    _code, plan = run([*args, "--dry-run"], capsys)
    assert run([*args, "--approved", plan["facts"]["plan_id"]], capsys)[0] == 0
    install_perf(monkeypatch, 40, 80)
    args = perf_args(tmp_path)
    _code, plan = run([*args, "--dry-run"], capsys)
    assert run([*args, "--approved", plan["facts"]["plan_id"]], capsys)[0] == 0
    saved = json.loads((tmp_path / "compare.json").read_text())
    assert saved["accuracy"]["sets"]["gsm8k"]["verdict"] == "pass"
    scores = tmp_path / "accuracy/gsm8k/scores.json"
    changed_scores = json.loads(scores.read_text())
    changed_scores["rewards"][0] = 0.5
    scores.write_text(json.dumps(changed_scores))
    assert run([*args, "--approved", plan["facts"]["plan_id"]], capsys)[0] == 0
    assert "accuracy" not in json.loads((tmp_path / "compare.json").read_text())
    assert "pass" not in (tmp_path / "report.md").read_text()


@pytest.mark.parametrize("artifact", ["manifest.json", "perf/summary.json"])
def test_changed_baseline_invalidates_a_preserved_verdict(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    """A speed verdict depends on the baseline evidence as well as candidate results."""
    install_perf(monkeypatch, 40, 80)
    args = perf_args(tmp_path, "--baseline", BASE)
    _code, plan = run([*args, "--dry-run"], capsys)
    assert run([*args, "--approved", plan["facts"]["plan_id"]], capsys)[0] == 0
    path = tmp_path / "baseline" / artifact
    changed = json.loads(path.read_text())
    if artifact == "manifest.json":
        changed["perf"]["isl"] = 512
    else:
        changed["loads"]["8"]["metrics"]["ttft_p99"]["mean"] = 1.0
    path.write_text(json.dumps(changed))
    install_gym(monkeypatch, 100, 100)
    args = acc_args(tmp_path)
    _code, plan = run([*args, "--dry-run"], capsys)
    assert run([*args, "--approved", plan["facts"]["plan_id"]], capsys)[0] == 0
    assert "perf" not in json.loads((tmp_path / "compare.json").read_text())



@pytest.mark.parametrize("version", ["0.12.0", "0.13.0"])
def test_local_planning_requires_the_pinned_aiperf_version(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    version: str,
) -> None:
    """Planning must reject a different local tool version before approval."""
    monkeypatch.setattr(
        nb_common,
        "run_command",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, f"AIPerf {version}", ""),
    )
    code, result = run(perf_args(tmp_path, "--client", "local", "--dry-run"), capsys)
    assert code == (0 if version == "0.13.0" else 3)
    if code:
        assert result["errors"][0]["code"] == "client_version_mismatch"


def test_a_failed_client_stop_does_not_start_the_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live client left by the candidate must prevent baseline load and comparison."""
    starts: list[list[str]] = []

    def command(argv: list[str], timeout: float = 0, env: Any = None) -> Any:
        if argv[6:8] == ["sh", "-c"]:
            starts.append(argv)
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, 1, "", "access denied")

    monkeypatch.setattr(nb_common, "check_client_pod", lambda *_: (True, []))
    monkeypatch.setattr(nb_common, "check_endpoint", lambda *a, **kw: (["qwen7b"], []))
    monkeypatch.setattr(nb_common, "run_command", command)
    args = perf_args(tmp_path, "--client", "kubernetes", "--namespace", "default",
                     "--pod", "bench-client", "--baseline", BASE)
    _code, planned = run([*args, "--dry-run"], capsys)
    code, result = run([*args, "--approved", planned["facts"]["plan_id"]], capsys)
    assert code == 4 and result["state"] == "tool_failed"
    assert "may still run in the pod; stop it" in result["errors"][0]["message"]
    assert "cleanup failed" in result["errors"][0]["message"]
    assert len(starts) == 1
    assert result["facts"]["verdicts"] == {}
    assert not (tmp_path / "baseline").exists()


@pytest.mark.parametrize("bad", [None, [], "broken", {"perf": {}, "plan_ids": None}, {"perf": {}, "plan_ids": []}, {"perf": []}])
def test_malformed_baseline_evidence_drops_saved_verdict_without_crashing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, bad: Any
) -> None:
    """Invalid baseline provenance must drop the old mode and still write the new report."""
    install_perf(monkeypatch, 40, 80)
    args = perf_args(tmp_path, "--baseline", BASE)
    _code, plan = run([*args, "--dry-run"], capsys)
    assert run([*args, "--approved", plan["facts"]["plan_id"]], capsys)[0] == 0
    (tmp_path / "baseline/manifest.json").write_text(json.dumps(bad))
    install_gym(monkeypatch, 100, 100)
    args = acc_args(tmp_path)
    _code, plan = run([*args, "--dry-run"], capsys)
    assert run([*args, "--approved", plan["facts"]["plan_id"]], capsys)[0] == 0
    assert "perf" not in json.loads((tmp_path / "compare.json").read_text())
    assert "better" not in (tmp_path / "report.md").read_text()


@pytest.mark.parametrize("bad", [None, [], "broken"])
@pytest.mark.parametrize("field", ["_provenance", "baseline"])
def test_malformed_saved_provenance_does_not_abort_the_next_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
    field: str, bad: Any
) -> None:
    """Corrupt provenance metadata is unusable evidence, rather than a fatal runtime error."""
    install_perf(monkeypatch, 40, 80)
    args = perf_args(tmp_path, "--baseline", BASE)
    _code, plan = run([*args, "--dry-run"], capsys)
    assert run([*args, "--approved", plan["facts"]["plan_id"]], capsys)[0] == 0
    path = tmp_path / "compare.json"
    saved = json.loads(path.read_text())
    if field == "_provenance":
        saved[field] = bad
    else:
        saved["_provenance"]["perf"][field] = bad
    path.write_text(json.dumps(saved))
    install_gym(monkeypatch, 100, 100)
    args = acc_args(tmp_path)
    _code, plan = run([*args, "--dry-run"], capsys)
    assert run([*args, "--approved", plan["facts"]["plan_id"]], capsys)[0] == 0
    assert "perf" not in json.loads(path.read_text())
    assert (tmp_path / "report.md").exists()


def test_unconfirmed_gym_stop_does_not_launch_the_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The accuracy baseline must wait until the candidate's Gym container has stopped."""
    starts: list[list[str]] = []

    def command(argv: list[str], timeout: float = 0, env: Any = None) -> Any:
        if argv[:2] == ["docker", "run"]:
            starts.append(argv)
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, 1, "", "cannot remove client")

    monkeypatch.setattr(nb_common, "run_command", command)
    args = acc_args(tmp_path, "--baseline", BASE)
    _code, plan = run([*args, "--dry-run"], capsys)
    code, result = run([*args, "--approved", plan["facts"]["plan_id"]], capsys)
    assert code == 4 and len(starts) == 1
    assert any("may still run" in p["message"] for p in result["errors"])
    assert not (tmp_path / "baseline").exists()
    assert result["facts"]["verdicts"] == {}
