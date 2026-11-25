# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/shellparse.py: splitting a Bash command line into simple commands."""

from __future__ import annotations

import json

import pytest
import shellparse
from shellparse import GuardError, parse


def norms(text: str) -> list[list[str]]:
    return [[command.norm for command in pipeline] for pipeline in parse(text)]


def test_splits_on_operators_and_pipes() -> None:
    assert norms("kubectl get pods | head -n 3 && echo done; ls") == [
        ["kubectl get pods", "head -n 3"],
        ["echo done"],
        ["ls"],
    ]


@pytest.mark.parametrize(
    "text",
    [
        "env --split-string='docker system prune -af'",
        "env --split-string 'docker system prune -af'",
        "env -S 'docker system prune -af'",
        "env -S'docker system prune -af'",
        "env -iS 'docker system prune -af'",
        "sudo env -u X -S 'docker system prune -af'",
    ],
)
def test_env_split_string_is_parsed_as_the_command_it_runs(text: str) -> None:
    assert norms(text) == [["docker system prune -af"]]


def test_env_split_string_keeps_the_arguments_after_it() -> None:
    assert norms("env -S 'kubectl get' pods -n ns") == [["kubectl get pods -n ns"]]


@pytest.mark.parametrize(
    "text",
    ["env -S 'docker \"x'", "env -S '${CMD} prune'", "env --split-string", "env -S"],
)
def test_env_split_string_that_cannot_be_split_safely_raises(text: str) -> None:
    with pytest.raises(GuardError):
        parse(text)


def test_newlines_separate_commands_and_comments_are_dropped() -> None:
    assert norms("echo a # kubectl delete x\nls /tmp") == [["echo a"], ["ls /tmp"]]
    assert norms("ls \\\n-la") == [["ls -la"]]


def test_prefixes_and_wrappers_are_removed() -> None:
    (pipeline,) = parse("FOO=1 sudo -u root env -i timeout 5 /usr/bin/kubectl get pods")
    (command,) = pipeline
    assert command.norm == "kubectl get pods"
    assert command.wrappers == ["sudo", "env", "timeout"]
    assert command.env_names == {"FOO"}


def test_substitutions_and_nested_shells_are_parsed() -> None:
    assert ["kubectl delete ns x"] in norms("echo $(kubectl delete ns x)")
    assert ["docker rm -f a"] in norms("echo `docker rm -f a`")
    assert ["helm uninstall r -n ns"] in norms("bash -c 'helm uninstall r -n ns'")
    assert ["kubectl apply -f x"] in norms("sh -lc 'kubectl apply -f x'")
    assert ["kubectl apply -f x"] in norms("eval kubectl apply -f x")
    assert ["docker stop a"] in norms("su root -c 'docker stop a'")
    assert norms("echo '$(kubectl get x)'") == [["echo $(kubectl get x)"]]  # single quotes: literal


def test_redirections_do_not_become_arguments() -> None:
    assert norms("kubectl get pods > out.txt 2>&1") == [["kubectl get pods"]]


def test_a_shell_script_argument_is_not_a_payload() -> None:
    assert norms("bash run.sh -c extra") == [["bash run.sh -c extra"]]


def test_xargs_is_recorded_as_a_wrapper() -> None:
    (pipeline,) = parse("echo a | xargs -n1 docker rm")
    assert pipeline[1].wrappers == ["xargs"] and pipeline[1].norm == "docker rm"


@pytest.mark.parametrize(
    "text",
    [
        "echo 'unterminated",
        'echo "x',
        "echo $(kubectl get",
        "echo `x",
        "a " * 3 + "$(" * 8 + "x" + ")" * 8,
    ],
)
def test_unparseable_commands_raise(text: str) -> None:
    with pytest.raises(GuardError):
        parse(text)


def test_cli_check_prints_the_split(capsys: pytest.CaptureFixture[str]) -> None:
    shellparse.main(["--check", "sudo kubectl get pods | head"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["facts"]["pipelines"] == [["kubectl get pods", "head"]]
    with pytest.raises(SystemExit) as exit_info:
        shellparse.main(["--check", "echo 'x"])
    assert exit_info.value.code == 2
