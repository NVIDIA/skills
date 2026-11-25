# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The security rules of references/security.md, checked by parsing the scripts and by runs."""

from __future__ import annotations

import ast
import json
import os
import re
import socket
import stat
import textwrap
from pathlib import Path

import nimsmoke
import nsc_access as nsa
import nsc_checks as nc
import nsc_common as nsc
import nsc_testkit as kit
import pytest

SCRIPTS = kit.SCRIPTS
LLAMA = "meta/llama-3.1-8b-instruct"
NIM_ROUTES = {"/v1/health/live", "/v1/health/ready", "/health", "/v1/models", "/v1/version",
              "/v1/metadata", "/v1/chat/completions"}


def sources() -> dict[str, ast.Module]:
    return {path.name: ast.parse(path.read_text(encoding="utf-8"))
            for path in sorted(SCRIPTS.glob("*.py"))}


def call_name(node: ast.Call) -> str:
    return getattr(node.func, "attr", getattr(node.func, "id", ""))


def enclosing_functions(tree: ast.Module) -> dict[ast.AST, str]:
    """Each node -> the name of the function it is in ("<module>" at the top)."""
    owner: dict[ast.AST, str] = {}

    def visit(node: ast.AST, name: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef,
                                                     ast.AsyncFunctionDef)) else name
            owner[child] = inner
            visit(child, inner)

    visit(tree, "<module>")
    return owner


# --- AST: one opener, one program runner, no dynamic code ---------------------------------------


def test_every_request_goes_through_open_url() -> None:
    """nsc_common.open_url is the one opener, and it follows no redirect; urlopen or another
    opener would follow them and forward the Authorization header."""
    found = []
    for name, tree in sources().items():
        owner = enclosing_functions(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and call_name(node) in (
                "urlopen", "build_opener", "OpenerDirector", "HTTPConnection",
                "HTTPSConnection", "create_connection", "urlretrieve",
            ):
                found.append((name, owner[node], call_name(node)))
    assert found == [
        ("nsc_access.py", "_connects", "create_connection"),  # the port-forward's TCP probe
        ("nsc_common.py", "open_url", "build_opener"),
    ]
    common = (SCRIPTS / "nsc_common.py").read_text(encoding="utf-8")
    assert "NoRedirect()" in common and "import requests" not in common


def test_every_program_starts_in_run_tool() -> None:
    """Only nsc_access.run_tool starts a program, never with a shell; nimsmoke.py imports no
    subprocess at all."""
    starters = {"run", "Popen", "call", "check_call", "check_output", "getoutput",
                "getstatusoutput"}
    found = []
    for name, tree in sources().items():
        owner = enclosing_functions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            base = getattr(getattr(func, "value", None), "id", "")
            if base == "subprocess" and call_name(node) in starters:
                found.append((name, owner[node], call_name(node)))
            if base == "os" and re.match(r"(system|popen|exec|spawn|posix_spawn)", call_name(node)):
                found.append((name, owner[node], f"os.{call_name(node)}"))
            for keyword in node.keywords:
                if keyword.arg == "shell":
                    assert isinstance(keyword.value, ast.Constant), (name, node.lineno)
                    assert keyword.value.value is False, (name, node.lineno)
    assert sorted(found) == [("nsc_access.py", "run_tool", "Popen"),
                             ("nsc_access.py", "run_tool", "run")]
    smoke = sources()["nimsmoke.py"]
    imported = {alias.name for node in ast.walk(smoke)
                if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    imported |= {node.module for node in ast.walk(smoke)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert not imported & {"subprocess", "socket", "urllib.request", "http.client"}


def test_no_dynamic_code_or_unsafe_loader() -> None:
    for name, tree in sources().items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                bare = isinstance(node.func, ast.Name)
                assert not (bare and node.func.id in ("eval", "exec", "compile", "__import__")
                            ), (name, node.lineno)
                assert call_name(node) != "import_module", (name, node.lineno)
                assert not (isinstance(node.func, ast.Attribute) and node.func.attr == "load"
                            and getattr(node.func.value, "id", "") in ("pickle", "marshal",
                                                                       "yaml")), name
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = {alias.name for alias in node.names}
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.add(node.module)
                assert not names & {"pickle", "marshal", "shelve", "yaml", "ctypes"}, name


@pytest.mark.parametrize(
    "argv",
    [
        ["kubectl", "delete", "pod", "llama-0", "--context", "lab", "-n", "nim"],
        ["kubectl", "apply", "-f", "x.yaml", "--context", "lab", "-n", "nim"],
        ["kubectl", "exec", "llama-0", "--context", "lab", "-n", "nim"],
        ["kubectl", "get", "secret", "ngc", "-o", "json", "--context", "lab", "-n", "nim"],
        ["kubectl", "get", "pods", "-o", "json"],
        ["kubectl", "patch", "nimservice", "llama", "--context", "lab", "-n", "nim"],
        ["docker", "run", "--rm", "nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.5"],
        ["docker", "rm", "-f", "llama"],
        ["docker", "exec", "llama", "env"],
        ["docker", "container", "stop", "llama"],
        ["docker", "image", "pull", "nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.5"],
    ],
)
def test_run_tool_refuses_anything_off_the_read_only_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    tools = kit.make_fake_tools(tmp_path)
    monkeypatch.setenv("PATH", str(tools.bin))
    argv = [str(tools.bin / argv[0]), *argv[1:]]
    with pytest.raises(nsc.ScriptError) as info:
        nsa.run_tool(argv, timeout=5)
    assert (info.value.code, info.value.exit_code) == ("refused_command", nsc.EXIT_REFUSED)
    assert tools.calls() == [], "a refused command never starts"


# --- AST: the environment variables ---------------------------------------------------------------

# What each script reads (references/security.md lists the same names). HOME is read through
# Path.home() and expanduser(). nsc_common.read_key reads the variable --api-key-env names (or
# NVIDIA_API_KEY for the hosted API), the one read whose name is not a literal.
ENV_READS = {
    "nsc_access": {"PATH", "DOCKER_HOST"},
    "nsc_common": {"HOME"},
    "nimsmoke": {"HOME"},
}
NAMED_READ = "nsc_common.read_key"
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """The nodes of one scope, without those of the functions and classes inside it."""
    found: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        found.append(node)
        if not isinstance(node, _SCOPES):
            stack.extend(ast.iter_child_nodes(node))
    return found


def environment_reads(source: str, module: str) -> tuple[set[str], list[str]]:
    """The environment variables a script reads, and the reads this check cannot name, as
    "module.function: why".

    An environment mapping is `os.environ` (any `.environ` or `.environb`), a name `environ`, or
    a name assigned from either in the same function or one around it. A read is a subscript,
    `.get`, `.pop`, `.setdefault`, `in`, or os.getenv; its key must be a literal, or a loop
    variable over literals. Any other use of the mapping as a whole (copy, items, dict(), **,
    env=) cannot be named, nor can a write. Path.home() and expanduser() read HOME."""
    reads: set[str] = set()
    problems: list[str] = []

    def visit(scope: ast.AST, inherited: set[str], where_name: str) -> None:
        nodes = _own_nodes(scope)
        aliases = set(inherited)
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = scope.args
            every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
            aliases -= {arg.arg for arg in every}

        def is_env(node: ast.AST) -> bool:
            if isinstance(node, ast.Attribute):
                return node.attr in ("environ", "environb")
            return isinstance(node, ast.Name) and (node.id == "environ" or node.id in aliases)

        def holds_env(value: ast.AST) -> bool:
            if isinstance(value, ast.IfExp):
                return holds_env(value.body) or holds_env(value.orelse)
            if isinstance(value, ast.BoolOp):
                return any(holds_env(item) for item in value.values)
            return is_env(value)

        changed = True
        while changed:
            changed = False
            for node in nodes:
                if isinstance(node, ast.Assign):
                    targets, value = list(node.targets), node.value
                elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
                    targets, value = [node.target], node.value
                else:
                    continue
                if holds_env(value):
                    for target in targets:
                        if isinstance(target, ast.Name) and target.id not in aliases:
                            aliases.add(target.id)
                            changed = True
        loops = {
            node.target.id: [item.value for item in node.iter.elts]
            for node in nodes
            if isinstance(node, (ast.For, ast.comprehension))
            and isinstance(node.target, ast.Name)
            and isinstance(node.iter, (ast.Tuple, ast.List, ast.Set))
            and all(isinstance(item, ast.Constant) and isinstance(item.value, str)
                    for item in node.iter.elts)
        }
        where = f"{module}.{where_name}"

        def read(key: ast.AST | None) -> None:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                reads.add(key.value)
            elif isinstance(key, ast.Name) and key.id in loops:
                reads.update(loops[key.id])
            else:
                problems.append(f"{where}: name not literal")

        for node in nodes:
            if isinstance(node, ast.Subscript) and is_env(node.value):
                read(node.slice)
            elif (
                isinstance(node, ast.Compare)
                and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
                and any(is_env(item) for item in node.comparators)
            ):
                read(node.left)
            elif isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if isinstance(func, ast.Attribute) and is_env(func.value):
                    if name in ("get", "pop", "setdefault"):
                        read(node.args[0] if node.args else None)
                    else:
                        problems.append(f"{where}: the environment as a whole (.{name})")
                elif name in ("getenv", "getenvb"):
                    read(node.args[0] if node.args else None)
                elif name in ("putenv", "unsetenv"):
                    problems.append(f"{where}: an environment write ({name})")
                elif name in ("home", "expanduser"):
                    reads.add("HOME")
                elif any(is_env(arg) for arg in node.args) and name in ("dict", "list", "sorted"):
                    problems.append(f"{where}: the environment as a whole ({name}())")
                if any(is_env(k.value) for k in node.keywords if k.arg is None):
                    problems.append(f"{where}: the environment as a whole (**)")
                if any(is_env(k.value) for k in node.keywords if k.arg == "env"):
                    problems.append(f"{where}: the environment passed whole to a program")
            elif isinstance(node, ast.Dict) and any(
                key is None and is_env(value)
                for key, value in zip(node.keys, node.values, strict=False)
            ):
                problems.append(f"{where}: the environment as a whole (**)")
        for node in nodes:
            if isinstance(node, _SCOPES):
                inner = getattr(node, "name", "<lambda>")
                visit(node, aliases, inner if where_name == "<module>" else
                      f"{where_name}.{inner}")

    visit(ast.parse(source), set(), "<module>")
    return reads, problems


def documented_variables() -> set[str]:
    """The names in the first column of the table under 'What the scripts read from the
    environment' in references/security.md."""
    text = (kit.SKILL / "references" / "security.md").read_text(encoding="utf-8")
    section = text.split("## What the scripts read from the environment", 1)[1].split("\n## ")[0]
    rows = [line for line in section.splitlines() if line.startswith("| ")]
    return {name for row in rows for name in re.findall(r"`([A-Z][A-Z0-9_]*)`", row.split("|")[1])}


def test_the_scripts_read_only_the_listed_environment_variables() -> None:
    """Every environment read is one the security notes list; the only read by a name that is
    not a literal is read_key's, of the variable the user names."""
    found: dict[str, set[str]] = {}
    problems: list[str] = []
    for path in sorted(SCRIPTS.glob("*.py")):
        reads, unnamed = environment_reads(path.read_text(encoding="utf-8"), path.stem)
        problems.extend(unnamed)
        if reads:
            found[path.stem] = reads
    assert problems == [f"{NAMED_READ}: name not literal"]
    assert found == ENV_READS
    hosted = {nimsmoke.HOSTED_KEY_ENV}
    assert documented_variables() == set().union(*ENV_READS.values()) | hosted
    section = (kit.SKILL / "references" / "security.md").read_text(encoding="utf-8")
    assert "the variable named by `--api-key-env`" in section


def test_read_key_is_called_only_with_the_named_variable_or_the_hosted_one() -> None:
    tree = sources()["nimsmoke.py"]
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and call_name(node) == "read_key"
             and isinstance(node.func, ast.Attribute)]
    shown = sorted(ast.unparse(node.args[0]) for node in calls)
    assert shown == ["HOSTED_KEY_ENV", "args.api_key_env"]
    assert nimsmoke.HOSTED_KEY_ENV == "NVIDIA_API_KEY"


@pytest.mark.parametrize(
    "line",
    [
        'base = os.environ.get("NIM_SMOKE_URL", DEFAULT)',
        'base = os.environ["SMOKE_ROOT"]',
        'base = os.getenv("NIM_SMOKE_" + "URL")',
        'env = os.environ\nbase = env.get("SMOKE_URL")',
        'from os import environ\nbase = environ.get("X") if "X" in environ else None',
        "copied = dict(os.environ)",
        "subprocess.run(argv, env=os.environ)",
    ],
)
def test_the_environment_check_finds_a_hidden_override(line: str) -> None:
    source = "import os\n\n\ndef base_url():\n" + textwrap.indent(line, "    ") + "\n"
    reads, problems = environment_reads(source, "mutant")
    assert reads or problems, line


# --- runtime: redirects, keys, files, isolation -------------------------------------------------


def test_a_redirect_is_never_followed_and_the_key_never_reaches_another_host(
    tmp_path: Path,
) -> None:
    key = kit.fake_key()
    with kit.FakeNim(LLAMA, require_key=key) as elsewhere:
        with kit.FakeNim(LLAMA, redirect_to=elsewhere.url) as first:
            code, payload, _ = kit.run_cli(
                "nimsmoke.py", "run", "--url", first.url, "--model", LLAMA, "--api-key-env",
                "SMOKE_KEY", "--checks", "health,models,chat,stream", "--out",
                str(tmp_path / "out"), env={"SMOKE_KEY": key},
            )
            assert first.requests, "the first host got the requests"
        assert elsewhere.requests == [], "the host a redirect names gets no request"
    assert (code, payload["state"]) == (4, "failed")
    reasons = {r["id"]: r["reason"] for r in payload["facts"]["results"]}
    assert reasons["chat"] == reasons["stream"] == "redirect_refused"
    assert key not in json.dumps(payload)


def test_a_health_url_elsewhere_never_gets_the_key(tmp_path: Path) -> None:
    """The key goes only to the scheme, host and port of --url: a --health-url on another port
    gets the health GETs without Authorization, and the URL gets every request with it."""
    key = kit.fake_key()
    with kit.FakeNim(LLAMA, require_key=key) as api, kit.FakeNim(LLAMA) as health:
        code, payload, stderr = kit.run_cli(
            "nimsmoke.py", "run", "--url", api.url, "--health-url", health.url, "--api-key-env",
            "SMOKE_KEY", "--checks", "health,models,chat", "--out", str(tmp_path / "out"),
            env={"SMOKE_KEY": key},
        )
        assert len(health.gets()) == 2 and health.posts() == []
        assert all("authorization" not in seen.headers for seen in health.requests)
        assert api.requests and all(
            seen.headers.get("authorization") == f"Bearer {key}" for seen in api.requests
        )
    assert code == 0, payload
    assert key not in json.dumps(payload) + stderr


def test_a_health_host_with_a_final_dot_never_gets_the_key() -> None:
    """gw.example.com. is looked up as written, while gw.example.com may get the resolver's
    search domains first, so the two can reach different servers: a --health-url that differs
    from --url only by the final dot gets its health GETs without Authorization."""
    auth = {"Authorization": f"Bearer {kit.fake_key()}"}
    for base, health in (("https://gw.example.com", "https://gw.example.com."),
                         ("https://gw.example.com.", "https://gw.example.com")):
        ep = nc.Endpoint(base=base, health_base=health, headers=dict(auth))
        for path in ("/v1/health/live", "/v1/health/ready", "/health"):
            url = ep.health_url(path)
            assert ep.headers_for(url) == {} and not ep.sends_key_to(url), url
        assert ep.headers_for(ep.url("/v1/models")) == auth


def test_a_key_too_short_to_mask_is_refused_before_any_request(tmp_path: Path) -> None:
    """The masker skips values shorter than MIN_SECRET characters (they would mask common
    words), so such a key would show wherever the endpoint echoed it: bad_input before any
    request, and the key is not shown."""
    key = ("Qz7" + "a9")[: nsc.MIN_SECRET - 1]
    with kit.FakeNim(LLAMA, require_key=key) as nim:
        code, payload, stderr = kit.run_cli(
            "nimsmoke.py", "run", "--url", nim.url, "--api-key-env", "SMOKE_KEY", "--out",
            str(tmp_path / "out"), env={"SMOKE_KEY": key},
        )
        assert nim.requests == []
    assert (code, payload["errors"][0]["code"]) == (2, "bad_input"), payload
    assert f"shorter than {nsc.MIN_SECRET} characters" in payload["errors"][0]["message"]
    assert key not in json.dumps(payload) + stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("flag", ["--api-key={}", "--api-key {}", "--api {}", "--api-key-e {}"])
def test_a_short_form_of_a_flag_is_refused_and_the_key_not_echoed(
    tmp_path: Path, flag: str
) -> None:
    """argparse takes a unique prefix of a flag by default: --api-key=KEY was --api-key-env KEY.
    A flag counts only in full, so a short form is a usage error that does not echo the key,
    and nothing is sent."""
    key = kit.fake_token()
    with kit.FakeNim(LLAMA) as nim:
        code, payload, stderr = kit.run_cli(
            "nimsmoke.py", "run", "--url", nim.url, *flag.format(key).split(" "), "--out",
            str(tmp_path / "out"),
        )
        assert nim.requests == []
    assert (code, payload["errors"][0]["code"]) == (2, "bad_input")
    assert "unrecognized arguments" in payload["errors"][0]["message"]
    assert key not in json.dumps(payload) + stderr
    assert not (tmp_path / "out").exists()


def test_the_key_never_appears_in_any_output(tmp_path: Path) -> None:
    """A hosted-shape key and a token of no known shape: neither is in stdout, stderr,
    results.json, report.md or any evidence file, though every request carried it."""
    for key in (kit.fake_key(), kit.fake_token()):
        out = tmp_path / f"out-{len(key)}"
        with kit.FakeNim(LLAMA, require_key=key) as nim:
            code, payload, stderr = kit.run_cli(
                "nimsmoke.py", "run", "--url", nim.url, "--api-key-env", "SMOKE_KEY",
                "--extended", "--out", str(out), env={"SMOKE_KEY": key},
            )
            assert all(s.headers.get("authorization") == f"Bearer {key}" for s in nim.requests)
        assert code == 0, payload
        assert key not in json.dumps(payload) and key not in stderr
        assert kit.files_containing(out, key) == []
        assert "SMOKE_KEY" in json.dumps(payload)  # the name, never the value
        evidence = json.loads(next((out / "evidence").glob("chat-*.json")).read_text("utf-8"))
        assert key not in json.dumps(evidence)


@pytest.mark.parametrize("bad", [kit.fake_key(), "Bearer " + "x" * 24, "Mk4" * 7 + "q",
                                 "not a name", "gsk" + "_" + "Ab3" * 17 + "c",
                                 "sk" + "_live_" + "Zx9" * 8, "gw" + "_prod_" + "Kp4" * 13 + "a"])
def test_a_key_given_as_a_value_is_refused_and_not_echoed(tmp_path: Path, bad: str) -> None:
    with kit.FakeNim(LLAMA) as nim:
        code, payload, stderr = kit.run_cli(
            "nimsmoke.py", "run", "--url", nim.url, "--api-key-env", bad, "--out",
            str(tmp_path / "out"),
        )
        assert nim.requests == []
    assert (code, payload["errors"][0]["code"]) == (2, "literal_secret")
    assert bad not in json.dumps(payload) and bad not in stderr
    assert not (tmp_path / "out").exists()


EVAL_NAMES = ("id", "expected_skill", "expected_script")  # kebab-case names, not text
EVAL_TEXTS = ("question", "ground_truth")


def test_the_evals_hold_no_key_shaped_value() -> None:
    """evals/evals.json is public, and an eval asks the agent to handle a key without the key:
    no string in it has a well-known token shape or a bearer token, and no word of a question,
    a ground truth or an expected behavior looks like a key (nsc.looks_like_key). Words with
    / . : = or $, or that start with -, are URLs, model ids, files, flags or variables. A hit
    names the case and the field, never the word."""
    cases = json.loads((kit.SKILL / "evals" / "evals.json").read_text(encoding="utf-8"))["cases"]
    hits = []
    for case in cases:
        fields = [(name, case.get(name)) for name in (*EVAL_NAMES, *EVAL_TEXTS)]
        fields += [(f"expected_behavior[{number}]", item)
                   for number, item in enumerate(case.get("expected_behavior") or [])]
        for name, value in fields:
            if not isinstance(value, str):
                continue
            if nsc.TOKEN_SHAPES.search(value) or nsc.BEARER.search(value):
                hits.append((case.get("id"), name, "a token shape"))
            if name in EVAL_NAMES:
                continue
            for word in value.split():
                word = word.strip("\"'`()[]{}<>,;!?").rstrip(".")
                if (word and not word.startswith("-") and not set("/.:=$") & set(word)
                        and nsc.looks_like_key(word)):
                    hits.append((case.get("id"), name, f"a key of {len(word)} characters"))
    assert len(cases) >= 7 and hits == []


def _longest_prefix(secret: str, text: str) -> int:
    """The length of the longest start of `secret` (6 characters or more) found in `text`."""
    found = [n for n in range(nsc.MIN_SECRET, len(secret) + 1) if secret[:n] in text]
    return max(found, default=0)


def test_an_echoed_key_is_masked_before_any_text_is_cut(tmp_path: Path) -> None:
    """A gateway echoes the key bare (no Bearer, no known shape) across the 200-character cut of
    an error message and the 2,000-character cut of an evidence field: no part of it is in
    stdout, stderr, results.json, report.md or the evidence."""
    token = "Qm7" + "x4K" * 12 + "z"  # 40 characters of no known shape, built at run time
    message = "x" * 175 + " key " + token + " " + "y" * 1750 + " " + token + " is not allowed"
    assert message.index(token) < 200 < message.index(token) + len(token)
    second = message.rindex(token)
    assert second < 2000 < second + len(token)
    error = {"error": {"message": message, "type": "BadRequestError", "code": 400}}
    out = tmp_path / "out"
    with kit.FakeNim(LLAMA, routes={"/v1/chat/completions": (400, error)}) as nim:
        code, payload, stderr = kit.run_cli(
            "nimsmoke.py", "run", "--url", nim.url, "--api-key-env", "GW_TOKEN", "--checks",
            "health,models,chat", "--out", str(out), env={"GW_TOKEN": token},
        )
        assert all(s.headers.get("authorization") == f"Bearer {token}" for s in nim.requests)
    assert (code, payload["state"]) == (4, "failed")
    assert {r["id"]: r["reason"] for r in payload["facts"]["results"]}["chat"] == "http_error"
    texts = {"stdout": json.dumps(payload), "stderr": stderr}
    texts.update({str(path.relative_to(out)): path.read_text(encoding="utf-8")
                  for path in out.rglob("*") if path.is_file()})
    assert "evidence/chat-1.json" in texts and "results.json" in texts
    assert {name: _longest_prefix(token, text) for name, text in texts.items()
            if _longest_prefix(token, text)} == {}


@pytest.mark.parametrize(
    ("route", "depth", "expected"),
    [
        ("/v1/models", 100_000, {"models": ("fail", "not_json")}),
        ("/v1/models", 950, {"models": ("fail", "not_json")}),
        ("/v1/metadata", 100_000, {"models": ("pass", "ok"), "chat": ("pass", "ok")}),
        ("/v1/chat/completions", 100_000, {"chat": ("fail", "not_json")}),
        ("/v1/chat/completions", 2000, {"chat": ("fail", "not_json")}),
    ],
)
def test_json_nested_too_deeply_never_ends_the_run(
    tmp_path: Path, route: str, depth: int, expected: dict[str, tuple[str, str]]
) -> None:
    """security-5: the endpoint's answers are data. JSON nested deeply enough to exhaust the
    recursion limit (json.loads, or the walkers that mask and cut evidence) is "not JSON": the
    run still prints its one JSON object and writes results.json, report.md and the evidence."""
    body = "[" * depth + "]" * depth
    out = tmp_path / "out"
    with kit.FakeNim(LLAMA, routes={route: (200, body, "application/json")}) as nim:
        code, payload, stderr = kit.run_cli("nimsmoke.py", "run", "--url", nim.url, "--checks",
                                            "health,models,chat", "--out", str(out))
    assert "Traceback" not in stderr and "RecursionError" not in stderr
    assert code == (0 if route == "/v1/metadata" else 4), payload
    found = {r["id"]: (r["status"], r["reason"]) for r in payload["facts"]["results"]}
    assert {check: found.get(check) for check in expected} == expected
    assert (out / "results.json").is_file() and (out / "report.md").is_file()
    assert all((out / path).is_file() for r in payload["facts"]["results"]
               for path in r["evidence"])


@pytest.mark.parametrize("odd_id", ["7", '{"name": "x"}'])
def test_a_model_card_whose_id_is_not_text_never_ends_the_run(tmp_path: Path, odd_id: str) -> None:
    """security-6: /v1/models with {"id": 7} broke the card lookup (StopIteration, no JSON on
    stdout). Now models fails bad_response_shape, no POST is sent, and the run reports it."""
    cards = '{"object": "list", "data": [{"id": ' + odd_id + ', "parent": null}]}'
    for extra in ([], ["--model", "7"]):
        out = tmp_path / f"out{len(extra)}"
        with kit.FakeNim(LLAMA, routes={"/v1/models": (200, cards, "application/json")}) as nim:
            code, payload, stderr = kit.run_cli("nimsmoke.py", "run", "--url", nim.url,
                                                "--checks", "health,models,chat", "--out",
                                                str(out), *extra)
            posts = nim.posts()
        assert "Traceback" not in stderr and "StopIteration" not in stderr
        assert (code, payload["state"]) == (4, "failed"), payload
        models = {r["id"]: r for r in payload["facts"]["results"]}["models"]
        assert (models["status"], models["reason"]) == ("fail", "bad_response_shape")
        assert (out / "results.json").is_file()
        assert posts == [] or extra  # without --model, nothing names a model to test


def run_main(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
             *argv: str) -> tuple[int, dict, str]:
    """nimsmoke.main(["run", *argv]) in this process, with a fresh SCRUBBER (the key a run
    registers masks nothing in another test): (exit code, the JSON object, stderr)."""
    monkeypatch.setattr(nsc, "SCRUBBER", nsc.Scrubber())
    with pytest.raises(SystemExit) as info:
        nimsmoke.main(["run", *argv])
    printed = capsys.readouterr()
    payload = json.loads(printed.out)
    kit.assert_contract(payload)
    return int(info.value.code), payload, printed.err


def wide(text: str) -> str:
    """`text` in full-width letters, which the name lookup reads as ASCII (the idna codec)."""
    return "".join(chr(ord(char) + 0xFEE0) for char in text)


def test_no_network_fails_every_lookup_and_connection(no_network: None) -> None:
    """The no_network fixture, which the tests that must send nothing use, fails a name lookup
    or anything sent out by any route of the socket module, not only the ones the scripts take
    (getaddrinfo, create_connection); a local fake server still learns its own name."""
    attempts = [
        lambda: socket.getaddrinfo("localhost", 80),
        lambda: socket.gethostbyname("localhost"),
        lambda: socket.gethostbyname_ex("localhost"),
        lambda: socket.gethostbyaddr("192.0.2.7"),
        lambda: socket.getnameinfo(("127.0.0.1", 9), 0),
        lambda: socket.create_connection(("127.0.0.1", 9)),
    ]
    for attempt in attempts:
        with pytest.raises(AssertionError, match="no network"):
            attempt()
    assert socket.gethostbyaddr("127.0.0.1") == ("localhost", [], ["127.0.0.1"])
    with socket.socket() as sock:
        with pytest.raises(AssertionError, match="no network"):
            sock.connect(("127.0.0.1", 9))
        with pytest.raises(AssertionError, match="no network"):
            sock.connect_ex(("127.0.0.1", 9))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(AssertionError, match="no network"):
            sock.sendto(b"x", ("127.0.0.1", 9))
        with pytest.raises(AssertionError, match="no network"):
            sock.sendmsg([b"x"], [], 0, ("127.0.0.1", 9))


def test_the_hosted_api_over_http_is_refused_before_any_key_is_read(
    monkeypatch: pytest.MonkeyPatch, no_network: None,
) -> None:
    """http://integrate.api.nvidia.com would send NVIDIA_API_KEY in clear text: bad input from
    the input checks, before the key is read (the key readers fail when called) or a name is
    resolved (sockets are blocked), also in a spelling the name lookup reads as that host
    (full-width letters, an ideographic full stop). --allow-plain-http-key does not change that,
    and a percent-encoded final dot is bad input as well."""
    def no_key(*args: object, **kwargs: object) -> None:
        raise AssertionError("the key was read before the URL was refused")

    monkeypatch.setattr(nsc, "read_key", no_key)
    monkeypatch.setattr(nimsmoke, "read_key", no_key)
    spelled = f"{wide('integrate')}.api\u3002nvidia.com"
    for argv in (["--url", "http://integrate.api.nvidia.com/v1"],
                 ["--url", "HTTP://Integrate.API.nvidia.com."],
                 ["--url", "https://integrate.api.nvidia.com/v1", "--health-url",
                  "http://integrate.api.nvidia.com"],
                 ["--url", "http://integrate.api.nvidia.com/v1", "--allow-plain-http-key"],
                 ["--url", f"http://{spelled}/v1", "--allow-plain-http-key"]):
        args = nimsmoke.build_parser().parse_args(["run", *argv])
        with pytest.raises(nsc.ScriptError) as info:
            nimsmoke.validate(args)
        assert (info.value.code, info.value.exit_code) == ("bad_input", nsc.EXIT_INPUT), argv
        assert "https://integrate.api.nvidia.com" in info.value.fix
    args = nimsmoke.build_parser().parse_args(
        ["run", "--url", "http://integrate.api.nvidia.com%2E/v1", "--allow-plain-http-key"])
    with pytest.raises(nsc.ScriptError) as info:
        nimsmoke.validate(args)
    assert info.value.code == "bad_input" and "percent-encoding" in info.value.message
    assert nsa.from_url("https://integrate.api.nvidia.com/v1").place == "hosted"
    assert nsa.from_url(f"https://{spelled}/v1").place == "hosted"
    with pytest.raises(nsc.ScriptError):
        nsa.from_url("http://integrate.api.nvidia.com:80/v1")


def test_a_key_over_plain_http_to_another_host_is_refused_unless_allowed() -> None:
    """A key that would go over plain http to a host that is not this machine stops the run
    (key_over_plain_http, exit 2); --allow-plain-http-key lets it go, and the run keeps the
    warning. This machine (the name localhost, 127.0.0.0/8 and ::1, also as 127.1 or
    ::ffff:127.0.0.1, and 0.0.0.0 or ::, which connect to it; so a port-forward, and a Docker
    port published on all addresses or on loopback) and https are neither refused nor warned
    about. A *.localhost name or a host with a final dot is another host here: the resolver may
    look it up in DNS. The key goes only to the URL's scheme, host and port, so a health URL on
    another host, which gets no key (test_a_health_url_elsewhere_never_gets_the_key), is never
    the reason for a refusal."""
    cases = [
        (nsa.from_url("http://gateway.example.com:8000", "http://192.0.2.7:8001"),
         "gateway.example.com"),
        (nsa.from_url("http://gateway.example.com:8000", "http://gateway.example.com:8000/h"),
         "gateway.example.com"),
        (nsa.from_url("HTTP://Gateway.Example.com.:8000"), "gateway.example.com."),
        (nsa.from_url("http://192.0.2.7:8000/v1"), "192.0.2.7"),
        (nsa.from_url("http://[2001:db8::5]:8000"), "2001:db8::5"),
        (nsa.from_url("http://[::ffff:192.0.2.7]:8000"), "::ffff:192.0.2.7"),
        (nsa.from_url("http://gw.localhost:8000"), "gw.localhost"),
        (nsa.from_url("http://localhost.:8000"), "localhost."),
        (nsa.from_url("http://127.0.0.1.:8000"), "127.0.0.1."),
        (nsa.from_url("http://127.1.:8000"), "127.1."),
        (nsa.from_url("https://gateway.example.com", "http://192.0.2.7:8001"), None),
        (nsa.from_url("http://127.0.0.1:8000", "http://192.0.2.7:8001"), None),
        (nsa.from_url("http://127.8.9.10:8000"), None),
        (nsa.from_url("http://127.1:8000"), None),
        (nsa.from_url("http://localhost:8000"), None),
        (nsa.from_url("http://LOCALHOST:8000"), None),
        (nsa.from_url("http://[::1]:8000"), None),
        (nsa.from_url("http://[::ffff:127.0.0.1]:8000"), None),
        (nsa.from_url("http://0.0.0.0:8000"), None),
        (nsa.from_url("http://[::]:8000"), None),
        (nsa.from_url("https://gateway.example.com"), None),
    ]
    for allow in ([], ["--allow-plain-http-key"]):
        args = nimsmoke.build_parser().parse_args(["run", "--url", "http://gateway.example.com",
                                                   *allow])
        for access, host in cases:
            run = nimsmoke.Run(args)
            run.access = access
            if host is None:
                run.check_cleartext()
                assert run.warnings == [], access.url
            elif not allow:
                with pytest.raises(nsc.ScriptError) as info:
                    run.check_cleartext()
                error = info.value
                assert (error.code, error.exit_code) == ("key_over_plain_http", nsc.EXIT_INPUT)
                assert f"to {host} over plain http" in error.message and run.warnings == []
                assert "https://" in error.fix and "loopback" in error.fix
                assert error.fix.endswith("add --allow-plain-http-key")
            else:
                run.check_cleartext()
                assert len(run.warnings) == 1, access.url
                assert f"to {host} over plain http (--allow-plain-http-key)" in run.warnings[0]
                assert "use an https URL" in run.warnings[0]


@pytest.mark.parametrize("url", ["http://127.0.0.1%2Egw.example.com:8000",
                                 "http://0.0.0.0%2Egw.example.com:8000",
                                 "http://localhost%2Egw.example.com:8000",
                                 "http://gw%2Elocalhost:8000"])
def test_a_percent_encoded_host_is_refused_before_any_name_is_looked_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    no_network: None, url: str,
) -> None:
    """urllib decodes a percent-encoded host before it connects, so 127.0.0.1%2Egw.example.com
    is the name 127.0.0.1.gw.example.com, which DNS may send anywhere. The URL is bad input
    before the key is read or a name is looked up (sockets fail when used), with or without
    --allow-plain-http-key. Behind that check, nsc.is_loopback and key_in_clear read the host
    as the connection does, so such a host is never this machine."""
    def no_key(*args: object, **kwargs: object) -> None:
        raise AssertionError("the key was read before the URL was refused")

    monkeypatch.setattr(nsc, "read_key", no_key)
    monkeypatch.setenv("SMOKE_KEY", kit.fake_token())
    for extra in ([], ["--allow-plain-http-key"]):
        code, payload, _ = run_main(monkeypatch, capsys, "--url", url, "--api-key-env",
                                    "SMOKE_KEY", "--out", str(tmp_path / "out"), *extra)
        assert (code, payload["errors"][0]["code"]) == (2, "bad_input"), payload
        assert "percent-encoding" in payload["errors"][0]["message"]
    assert not (tmp_path / "out").exists()
    assert nimsmoke.key_in_clear(url)
    assert nsc.is_loopback(url) is url.startswith("http://gw%2E")  # gw.localhost skips the proxy


@pytest.mark.parametrize("url", ["http://gateway.example.com:8000", "http://192.0.2.7:8000/v1",
                                 "http://[2001:db8::5]:8000", "http://gw.localhost:8000"])
def test_a_key_over_plain_http_is_refused_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    no_network: None, url: str,
) -> None:
    """A run, or a dry run, with a key and a plain http URL of another host stops with
    key_over_plain_http (exit 2, ask the user) before any name is looked up or any connection
    made (sockets fail when used): nothing is sent or written, and the key is in no output."""
    key = kit.fake_token()
    monkeypatch.setenv("SMOKE_KEY", key)
    for extra in ([], ["--dry-run"]):
        code, payload, stderr = run_main(
            monkeypatch, capsys, "--url", url, "--api-key-env", "SMOKE_KEY", "--out",
            str(tmp_path / "out"), *extra,
        )
        assert (code, payload["state"], payload["next"]) == (2, "input", "ask_user")
        error = payload["errors"][0]
        assert error["code"] == "key_over_plain_http"
        assert "over plain http" in error["message"] and "before any request" in error["message"]
        assert error["fix"].endswith("--allow-plain-http-key"), error["fix"]
        assert payload["facts"]["access"]["auth"] == "bearer from $SMOKE_KEY"
        assert key not in json.dumps(payload) + stderr
    assert not (tmp_path / "out").exists()


def test_the_flag_sends_the_key_over_plain_http_and_the_run_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The fake NIM stands for a host that is not this machine (key_in_clear says so for every
    URL). Without --allow-plain-http-key it gets no request; with the flag every request
    carries the key and the warning is in facts.warnings, results.json and report.md; a run with
    no key is never refused, as it sends nothing secret."""
    monkeypatch.setattr(nimsmoke, "key_in_clear", lambda url: True)
    key = kit.fake_token()
    monkeypatch.setenv("SMOKE_KEY", key)
    with kit.FakeNim(LLAMA) as nim:
        keyed = ["--url", nim.url, "--api-key-env", "SMOKE_KEY", "--checks", "health,models,chat"]
        code, payload, _ = run_main(monkeypatch, capsys, *keyed, "--out",
                                    str(tmp_path / "refused"))
        assert (code, payload["errors"][0]["code"]) == (2, "key_over_plain_http")
        assert nim.requests == [] and not (tmp_path / "refused").exists()
        out = tmp_path / "allowed"
        code, payload, stderr = run_main(monkeypatch, capsys, *keyed, "--allow-plain-http-key",
                                         "--out", str(out))
        assert (code, payload["state"]) == (0, "passed"), payload
        assert nim.posts() and all(
            seen.headers.get("authorization") == f"Bearer {key}" for seen in nim.requests)
        nim.reset()
        code, plain, _ = run_main(monkeypatch, capsys, "--url", nim.url, "--checks",
                                  "health,models,chat", "--out", str(tmp_path / "nokey"))
        assert (code, plain["state"]) == (0, "passed"), plain
        assert nim.posts() and all("authorization" not in s.headers for s in nim.requests)
    warning = ("The key goes to 127.0.0.1 over plain http (--allow-plain-http-key), so anyone "
               "on the network path can read it; use an https URL.")
    assert warning in payload["facts"]["warnings"] and warning in stderr
    assert warning in json.loads((out / "results.json").read_text(encoding="utf-8"))["warnings"]
    report = (out / "report.md").read_text(encoding="utf-8")
    assert f"## Warnings\n\n- {warning}" in report
    assert kit.files_containing(out, key) == [] and key not in json.dumps(payload)
    assert not any("plain http" in text for text in plain["facts"]["warnings"])


def test_the_flag_keeps_its_warning_when_the_run_stops_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A gateway in front of several models stops the run while it finds the model
    (model_ambiguous), after the key went out over plain http and before any result is
    written: facts.warnings still names that send. The fake NIM stands for the gateway."""
    monkeypatch.setattr(nimsmoke, "key_in_clear", lambda url: True)
    key = kit.fake_token()
    monkeypatch.setenv("SMOKE_KEY", key)
    with kit.FakeNim(LLAMA, extra_models=("meta/llama-3.3-70b-instruct",)) as nim:
        code, payload, _ = run_main(monkeypatch, capsys, "--url", nim.url, "--api-key-env",
                                    "SMOKE_KEY", "--allow-plain-http-key", "--out",
                                    str(tmp_path / "out"))
        assert nim.requests and all(
            seen.headers.get("authorization") == f"Bearer {key}" for seen in nim.requests)
    assert (code, payload["errors"][0]["code"]) == (3, "model_ambiguous"), payload
    assert [text for text in payload["facts"]["warnings"] if "plain http" in text] == [
        "The key goes to 127.0.0.1 over plain http (--allow-plain-http-key), so anyone on the "
        "network path can read it; use an https URL."]
    assert not (tmp_path / "out").exists() and key not in json.dumps(payload)


def test_a_key_over_plain_http_from_docker_is_refused_like_any_other_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    no_network: None,
) -> None:
    """--docker is treated like any other URL. A remote daemon (DOCKER_HOST tcp://) stops the
    run with docker_remote (exit 3) before docker runs, with or without the flag; with a key its
    fix says that the --url it names is plain http to that host, refused with a key unless the
    user agrees to the flag, and prefers the Docker host. A container that publishes its port
    on one address that is not loopback gives that URL, refused too, after only the read-only
    docker reads and before any request."""
    key = kit.fake_token()
    record = kit.docker_container("llama", f"nvcr.io/nim/{LLAMA}:2.1.5", host_port=8008,
                                  host_ip="192.0.2.100")
    tools = kit.make_fake_tools(tmp_path, docker_fixtures={
        "container inspect llama": record, "context inspect": kit.docker_context()})
    for allow in ([], ["--allow-plain-http-key"]):
        code, payload, _ = kit.run_cli(
            "nimsmoke.py", "run", "--docker", "llama", "--api-key-env", "SMOKE_KEY", *allow,
            "--out", str(tmp_path / "remote"),
            env=tools.env({"DOCKER_HOST": "tcp://192.0.2.5:2376", "SMOKE_KEY": key}),
        )
        assert (code, payload["errors"][0]["code"]) == (3, "docker_remote"), payload
        fix = payload["errors"][0]["fix"]
        assert "--url" in fix and "http://192.0.2.5:8000" in fix and "Docker host" in fix
        assert "key_over_plain_http" in fix and "the key stays on loopback" in fix
        assert fix.endswith("only when the user agrees, add --allow-plain-http-key"), fix
        assert nsc.MASK not in fix and tools.calls() == []
    code, payload, _ = kit.run_cli(
        "nimsmoke.py", "run", "--docker", "llama", "--out", str(tmp_path / "remote"),
        env=tools.env({"DOCKER_HOST": "tcp://192.0.2.5:2376"}),
    )
    assert (code, payload["errors"][0]["code"]) == (3, "docker_remote"), payload
    assert "key_over_plain_http" not in payload["errors"][0]["fix"]
    monkeypatch.setenv("SMOKE_KEY", key)
    code, payload, _ = run_main(monkeypatch, capsys, "--url", "http://192.0.2.5:8000",
                                "--api-key-env", "SMOKE_KEY", "--out", str(tmp_path / "url"))
    assert (code, payload["errors"][0]["code"]) == (2, "key_over_plain_http")
    assert "Docker host" in payload["errors"][0]["fix"]
    monkeypatch.setenv("PATH", str(tools.bin))
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    code, payload, _ = run_main(monkeypatch, capsys, "--docker", "llama", "--api-key-env",
                                "SMOKE_KEY", "--out", str(tmp_path / "local"))
    assert (code, payload["errors"][0]["code"]) == (2, "key_over_plain_http"), payload
    assert payload["facts"]["url"] == "http://192.0.2.100:8008"
    tools.assert_read_only()
    assert [argv[:2] for argv in tools.argvs("docker")] == [["context", "inspect"],
                                                            ["container", "inspect"]]
    assert not any((tmp_path / name).exists() for name in ("remote", "url", "local"))


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX owners only")
def test_out_refuses_a_folder_other_users_can_change(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    with kit.FakeNim(LLAMA) as nim:
        code, payload, _ = kit.run_cli("nimsmoke.py", "run", "--url", nim.url, "--out",
                                       str(shared / "run"))
        assert nim.requests == [], "refused before any request"
    assert (code, payload["errors"][0]["code"]) == (2, "out_not_private")
    assert not (shared / "run").exists()


def test_out_files_are_private_and_a_planted_symlink_is_not_followed(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with kit.FakeNim(LLAMA) as nim:
        code, payload, _ = kit.run_cli("nimsmoke.py", "run", "--url", nim.url, "--out",
                                       str(out), "--checks", "chat,tools")
    assert code == 0, payload
    for path in [out, *out.rglob("*")]:
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == (0o700 if path.is_dir() else 0o600), (path, oct(mode))
    victim = tmp_path / "victim.txt"
    victim.write_text("precious\n", encoding="utf-8")
    (out / "results.json").unlink()
    (out / "results.json").symlink_to(victim)
    nsc.write_json_file(out / "results.json", {"a": 1})
    assert victim.read_text(encoding="utf-8") == "precious\n"
    assert not (out / "results.json").is_symlink()


def test_report_does_not_follow_a_symlinked_results_file(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps({"schema": nimsmoke.RESULTS_SCHEMA, "results": []}), "utf-8")
    folder = tmp_path / "run"
    folder.mkdir(mode=0o700)
    (folder / "results.json").symlink_to(target)
    code, payload, _ = kit.run_cli("nimsmoke.py", "report", str(folder))
    assert (code, payload["errors"][0]["code"]) == (2, "bad_input")


def test_a_run_sends_only_get_and_post_to_the_nim_routes(tmp_path: Path) -> None:
    with kit.FakeNim("qwen/qwen3.5-122b-a10b", reasoning="kwargs", vlm=True) as nim:
        code, payload, _ = kit.run_cli("nimsmoke.py", "run", "--url", nim.url, "--extended",
                                       "--out", str(tmp_path / "out"))
        seen = {(s.method, s.path) for s in nim.requests}
    assert code == 0, payload
    assert {method for method, _ in seen} == {"GET", "POST"}
    assert {path for _, path in seen} <= NIM_ROUTES
    assert {path for method, path in seen if method == "POST"} == {"/v1/chat/completions"}


def test_security_md_names_only_what_results_json_holds(tmp_path: Path) -> None:
    """docs-14: the Files table says what results.json holds; each field it names is in the
    file a run writes (there is no plan in it)."""
    with kit.FakeNim(LLAMA) as nim:
        code, payload, _ = kit.run_cli("nimsmoke.py", "run", "--url", nim.url, "--checks",
                                       "health,models,chat", "--out", str(tmp_path / "out"))
    assert code == 0, payload
    doc = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    text = (kit.SKILL / "references" / "security.md").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| `results.json` |"))
    claim = re.sub(r"\([^)]*\)", "", row.split("|")[3].split("the run:", 1)[1].split(";")[0])
    names = [part.strip(" `") for part in re.split(r",| and ", claim) if part.strip(" `")]
    names = ["results" if name == "every result" else name for name in names]
    assert len(names) >= 4 and [name for name in names if name not in doc] == [], names
    assert "`not_run`" in row and "not_run" in doc


def test_the_scripts_run_isolated(tmp_path: Path) -> None:
    """SKILL.md runs python3 -I: a sitecustomize.py on PYTHONPATH or in the current folder, or a
    planted json.py, is never loaded."""
    planted = tmp_path / "planted"
    planted.mkdir()
    marker = tmp_path / "planted-ran"
    for name in ("sitecustomize.py", "usercustomize.py", "json.py"):
        (planted / name).write_text(f"open({str(marker)!r}, 'w').close()\n", encoding="utf-8")
    code, payload, _ = kit.run_cli(
        "nimsmoke.py", "features", "--model", LLAMA, cwd=planted,
        env={"PYTHONPATH": str(planted), "PYTHONSTARTUP": str(planted / "json.py")},
    )
    assert code == 0 and payload["facts"]["record"] == "llama-3.1-instruct"
    assert not marker.exists()
    first = (SCRIPTS / "nimsmoke.py").read_text(encoding="utf-8").splitlines()
    assert first[0] == "#!/usr/bin/env python3"
    assert any("SPDX-License-Identifier: Apache-2.0" in line for line in first[:5])
