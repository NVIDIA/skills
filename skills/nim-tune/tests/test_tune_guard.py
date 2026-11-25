# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for scripts/tune_guard.py, the PreToolUse hook for nim-tune's must-never rules."""

from __future__ import annotations

import importlib.util
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from testenv import base_env

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tune_guard.py"


def hook(command: str) -> int:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=json.dumps({"tool_input": {"command": command}}),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return proc.returncode


def test_direct_target_changes_are_denied() -> None:
    for c in (
        "kubectl -n x apply -f a.yaml",
        "kubectl delete nimservice nimtune-me-before",
        "sudo -n docker rm -f nimtune-me-before",
        "ssh host 'x' ; docker run --gpus all img",
        "helm install x chart",
    ):
        assert hook(c) == 2, c


def test_wrapped_and_remote_changes_are_denied() -> None:
    for c in (
        "timeout 60 kubectl delete pod x",
        "timeout -s KILL 30 docker run --gpus all img",
        "sudo -u root kubectl delete ns x",
        "env A=1 -u B docker rm x",
        "nice -n 5 helm upgrade x chart",
        "ssh host kubectl delete pod x",
        "ssh -p 22 -i k.pem user@host 'docker run --gpus all img'",
        "sh -c 'docker rm x'",
        "bash -lc 'kubectl -n x apply -f a.yaml'",
        "sudo sh -c 'docker rm -f x'",
        "timeout 60 ssh host 'sudo docker stop x'",
        "eval 'kubectl delete pod x'",
        'echo "$(kubectl delete pod x)"',
        "echo `docker rm x`",
        "kubectl get pods -o name | xargs -n 1 kubectl delete",
        "watch -n 5 'kubectl delete pod x'",
        "/usr/local/bin/kubectl delete pod x",
    ):
        assert hook(c) == 2, c


def test_docker_management_groups_are_checked() -> None:
    for c in (
        "docker container rm x",
        "docker container run --gpus all img",
        "docker image rm img",
        "docker image prune -a",
        "docker -H tcp://h:2375 container stop x",
        "docker network create n",
        "docker volume rm v",
        "docker system prune -f",
        "docker compose up -d",
        "docker builder prune",
        "kubectl auth reconcile -f rbac.yaml",
        "helm test x",
    ):
        assert hook(c) == 2, c
    for c in (
        "docker container ls -a",
        "docker image inspect img",
        "docker network ls",
        "docker compose config",
        "docker system df",
        "docker container logs x",
        "kubectl auth can-i get pods",
    ):
        assert hook(c) == 0, c


def test_only_a_real_dry_run_is_exempt() -> None:
    for c in (
        "kubectl delete pod x --dry-run=none",
        "kubectl apply --dry-run=false -f a.yaml",
        "kubectl apply -f --dry-run",  # a file named --dry-run is not a dry run
        "kubectl apply -f a-dry-run.yaml",
        "helm install x chart --dry-run=false",
    ):
        assert hook(c) == 2, c
    for c in (
        "kubectl apply --dry-run=server -f a.yaml",
        "kubectl delete pod x --dry-run=client",
        "helm install x chart --dry-run",
    ):
        assert hook(c) == 0, c


def test_reads_and_dry_runs_pass() -> None:
    for c in (
        "kubectl get pods -n x --context c",
        "kubectl apply --dry-run=server -f a.yaml",
        "docker ps -a",
        "python3 s/run_campaign.py c.json --out . --dry-run",
        "vllm serve --max-num-batched-tokens 4096",
        "ssh host nvidia-smi",
        "sudo -u me docker ps",
        "timeout 60 kubectl logs pod/x",
        "sh -c 'kubectl get pods | grep x'",
        "git commit -m 'note: kubectl delete is denied by the hook'",
    ):
        assert hook(c) == 0, c


def test_a_live_campaign_needs_an_approved_plan() -> None:
    assert hook("python3 s/run_campaign.py c.json --out .") == 2
    assert hook("nohup python3 s/run_campaign.py c.json --out . --approved 0123abcd &") == 0


def test_credentials_on_a_command_line_are_denied() -> None:
    # fake values built at run time, so no line of this file looks like a credential to gitleaks
    fake = "x" * 4 + "Y" * 20
    client = "cu" + "rl"
    for c in (
        f"NGC_API_KEY=nvapi-{fake} python3 x.py",
        f"{client} -H 'Authorization: Bearer {fake}' u",
        f"tool --api-key {fake}",
    ):
        assert hook(c) == 2, c
    assert hook("export NGC_API_KEY=$NGC_API_KEY") == 0  # a reference, not a value


def test_check_mode_prints_contract_json() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "kubectl delete ns prod"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    out = json.loads(proc.stdout)
    assert proc.returncode == 2 and set(out) == {"ok", "state", "next", "facts", "errors"}


def test_hook_doc_table_rows_have_the_same_column_count() -> None:
    doc = Path(__file__).resolve().parents[1] / "references" / "hook.md"
    rows = [ln for ln in doc.read_text().splitlines() if ln.startswith("|")]
    counts = {len(re.split(r"(?<!\\)\|", ln)) for ln in rows}
    assert len(counts) == 1, counts


def test_unknown_value_flags_cannot_hide_the_verb() -> None:
    for c in (
        "kubectl -v 6 delete po x",
        "kubectl --as admin delete po",
        "docker --tlscacert ca rm x",
        "kubectl --as-group g --as admin apply -f a.yaml",
        "kubectl --token abc scale deploy x --replicas 0",
        "helm --repository-config r.yaml upgrade x chart",
        "docker --tlscacert ca container rm x",
        "kubectl --certificate-authority ca.pem rollout restart deploy/x",
    ):
        assert hook(c) == 2, c
    for c in (
        "kubectl -v 6 get pods",
        "kubectl --as admin get pods -n x",
        "kubectl -n x logs nim-tune-before-0 --tail 100",
        "kubectl -n x get nimservice nim-tune-before -o yaml",
        "kubectl -n x describe pod nim-tune-before-0",
        "kubectl -n x wait --for=condition=Ready nimservice/x --timeout 60s",
        "kubectl -n x exec nim-tune-before-0 -- nvidia-smi",
        "docker --tlscacert ca ps",
        "docker logs --tail 100 nim-tune-before",
        "docker inspect nim-tune-before",
        "helm -n x status release",
        "helm list -A",
    ):
        assert hook(c) == 0, c


def test_only_rollout_changes_are_denied() -> None:
    for c in (
        "kubectl rollout restart deploy/x",
        "kubectl -n x rollout undo deploy/x",
        "kubectl rollout pause deploy/x",
        "kubectl rollout resume deploy/x",
    ):
        assert hook(c) == 2, c
    for c in (
        "kubectl rollout status deploy/x",
        "kubectl -n x rollout history deploy/x",
        "kubectl rollout status deploy/x --timeout 120s",
    ):
        assert hook(c) == 0, c


def run_hook(stdin: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_a_crash_in_the_guard_blocks_the_command(monkeypatch: Any, capsys: Any) -> None:
    spec = importlib.util.spec_from_file_location("tune_guard", SCRIPT)
    assert spec and spec.loader
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)

    def boom(command: str, depth: int = 0) -> str | None:
        raise RuntimeError("odd edge case")

    monkeypatch.setattr(guard, "decide", boom)
    monkeypatch.setattr(sys, "argv", ["tune_guard.py"])
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"tool_input": {"command": "ls"}}'))
    try:
        guard.main()
    except SystemExit as exit_:
        assert exit_.code == 2  # 1 would let Claude Code run the command
    else:
        raise AssertionError("the guard let a crash through")
    assert "cannot decide" in capsys.readouterr().err


def test_unreadable_hook_input_blocks_and_other_tools_pass() -> None:
    for bad in ("", "not json", "[]", "null", '{"tool_input": "ls"}', '{"tool_input": ["ls"]}'):
        proc = run_hook(bad)
        assert proc.returncode == 2 and "nim-tune guard" in proc.stderr, bad
    # a valid event without a Bash command (another tool) is allowed
    for ok in ("{}", '{"tool_name": "Read", "tool_input": {"file_path": "/x"}}'):
        assert run_hook(ok).returncode == 0, ok


def test_a_bash_event_without_a_command_string_blocks() -> None:
    for bad in (
        {"tool_name": "Bash", "tool_input": {}},
        {"tool_name": "Bash", "tool_input": {"command": ""}},
        {"tool_name": "Bash", "tool_input": {"command": ["kubectl", "delete", "pod", "x"]}},
        {"tool_input": {"command": None}},
        {"tool_input": {"command": 5}},
    ):
        proc = run_hook(json.dumps(bad))
        assert proc.returncode == 2 and "nim-tune guard" in proc.stderr, bad
    ls = {"tool_name": "Bash", "tool_input": {"command": "ls"}}
    assert run_hook(json.dumps(ls)).returncode == 0


def test_a_resource_named_like_a_verb_is_not_the_verb() -> None:
    for c in (
        "kubectl get pod delete",
        "kubectl -n x logs delete --tail 10",
        "kubectl --context=c get nimservice apply -o yaml",
        "docker logs rm",
        "docker inspect stop",
        "helm status uninstall",
    ):
        assert hook(c) == 0, c
    for c in (
        "kubectl -n x delete pod get",
        "kubectl --context=c --as admin delete pod x",
        "kubectl -v 6 delete po x",  # an unknown flag before the verb: still found
        "docker -H tcp://h rm x",
        "helm plugin install ./plugin",  # a group's changing subcommand is still found
        "helm template ./chart --post-renderer=kubectl --post-renderer-args delete",
        "helm template ./chart --post-renderer ./hook.sh",
        "helm upgrade x ./chart --dry-run --post-renderer ./hook.sh",  # runs even in a dry run
        "helm template ./chart --output-dir -n --post-renderer ./hook.sh",  # -n is the folder
        "kubectl config set-context x && kubectl delete pod y",
    ):
        assert hook(c) == 2, c


def hook_command() -> str:
    doc = Path(__file__).resolve().parents[1] / "references" / "hook.md"
    text = doc.read_text()
    assert "onFailure" not in text  # not a Claude Code hook option
    block = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    assert block
    entry = json.loads(block.group(1))["hooks"]["PreToolUse"][0]["hooks"][0]
    assert set(entry) == {"type", "command"}
    return str(entry["command"])


def run_hook_command(project: Path, stdin: str, path: str | None = None) -> int:
    env = base_env({"CLAUDE_PROJECT_DIR": str(project)})
    if path is not None:
        env["PATH"] = path
    return subprocess.run(
        ["/bin/sh", "-c", hook_command()],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    ).returncode


def test_the_documented_hook_command_fails_closed(tmp_path: Path) -> None:
    # a project with the skill installed where the documented command looks for it, whatever
    # folder this checkout keeps the skill in
    project = tmp_path / "project"
    (project / ".agents" / "skills").mkdir(parents=True)
    (project / ".agents" / "skills" / "nim-tune").symlink_to(Path(__file__).resolve().parents[1])
    assert (project / ".agents/skills/nim-tune/scripts/tune_guard.py").is_file()
    ls = json.dumps({"tool_input": {"command": "ls"}})
    rm = json.dumps({"tool_input": {"command": "kubectl delete pod x"}})
    assert run_hook_command(project, ls) == 0  # allow stays 0
    assert run_hook_command(project, rm) == 2  # deny stays 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run_hook_command(empty, ls) == 2  # the script is missing
    assert run_hook_command(project, ls, path=str(empty)) == 2  # python3 is missing (127)
    crashing = empty / ".agents/skills/nim-tune/scripts"
    crashing.mkdir(parents=True)
    (crashing / "tune_guard.py").write_text("import sys\nsys.exit(1)\n")  # a plain crash
    assert run_hook_command(empty, ls) == 2


def test_rollout_changes_are_found_behind_unknown_flags() -> None:
    for c in (
        "kubectl rollout --as admin restart deployment/foo",
        "kubectl --as admin rollout restart deployment/foo",
        "kubectl rollout -v 6 undo deployment/foo",
        "kubectl rollout --some-new-flag x --other y pause deployment/foo",
        "kubectl -n x rollout --as-group g --as admin resume deployment/foo",
    ):
        assert hook(c) == 2, c
    for c in (
        "kubectl rollout status deployment/foo --timeout=60s",
        "kubectl --as admin rollout status deployment/foo",
        "kubectl rollout history deployment/foo --revision 2",
        "kubectl --certificate-authority ca.pem rollout status deployment/foo",
    ):
        assert hook(c) == 0, c
