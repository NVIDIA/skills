# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Machine checks for the nvalchemi-toolkit task set (SkillEvaluator custom grader).

SkillEvaluator runs this file in the verifier with ``grading.mode:
default_plus_custom``, after its own judges. It reads the task's ``checks``
list from ``entry.json``, runs them on the agent's final code, on the
output of the agent's own runs, and on the skill notes it opened (scope
``reads``), and reports five custom metrics:

``batched``       all systems in one batch, no Python loop over systems
``device_aware``  the code picks cuda when a GPU exists, one device throughout
``loop_syncs``    no avoidable host-device syncs inside a step loop or hook
``reuse``         the toolkit component is used, no hand-written replacement
``outcome``       the run shows the expected result

Each metric is the fraction of that task's checks with that metric which
pass. A metric with no check in the task is left out (not applicable). The
merged reward keeps SkillEvaluator's own ``overall``; the report combines
the custom metrics with it (see the task-set report).

Checks marked ``needs_skill`` are skipped in the baseline arm (entry.json
``has_skill`` false), so that arm is not marked down for skill files it
never had. The grader never runs agent code. ``run`` checks read the output
of the agent's own commands from the trajectory. It also copies the agent's
workspace to ``<trial>/kept-workspace/`` for the GPU rerun (``gpu_check``),
because Harbor deletes the workspace after grading.

Offline use, for testing the checks on a script::

    python grader.py --offline --entry case.json --code DIR_OR_FILE \
        [--trajectory trajectory.json] [--output run_output.txt]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

METRICS = ("batched", "device_aware", "loop_syncs", "reuse", "outcome")
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 200 * 1024 * 1024
MAX_CODE_FILE_BYTES = 512 * 1024
SKIP_DIRS = {"input", "skills", "__pycache__", ".git", ".venv", "node_modules"}
# evals/setup_cpu.sh builds the toolkit venv in /workspace/.nvalchemi-env.*
# and links /workspace/.venv to it. Neither is the agent's code.
ENV_PREFIX = ".nvalchemi-env."


def _skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.startswith(ENV_PREFIX)


# Calls that copy a tensor to the host or wait for the device.
SYNC_METHODS = {"item", "cpu", "tolist", "numpy"}
SYNC_FUNCS = {"float", "int", "bool"}
SYNC_QUALIFIED = {"torch.cuda.synchronize", "torch.equal", "torch.allclose"}
# Calls that mark a loop body as a step loop when written by hand.
STEP_CALL_ATTRS = {"step", "run", "backward"}
HOOK_METHODS = {"__call__", "pre_update", "post_update"}


# --------------------------------------------------------------------------
# Workspace copy
# --------------------------------------------------------------------------


def keep_workspace(workspace: Path, dest: Path) -> dict:
    """Copy regular files under ``workspace`` (without ``skills/``) to ``dest``."""
    kept, skipped, total = [], [], 0
    for root, dirs, files in os.walk(workspace):
        rel_root = Path(root).relative_to(workspace)
        if rel_root == Path("."):
            dirs[:] = [
                d
                for d in dirs
                if d not in {"skills", ".venv"} and not d.startswith(ENV_PREFIX)
            ]
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git"}]
        for name in sorted(files):
            src = Path(root) / name
            rel = (rel_root / name).as_posix()
            if src.is_symlink() or not src.is_file():
                continue
            size = src.stat().st_size
            if size > MAX_FILE_BYTES or total + size > MAX_TOTAL_BYTES:
                skipped.append({"path": rel, "bytes": size})
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            kept.append(rel)
            total += size
    return {
        "workspace": str(workspace),
        "kept": kept,
        "skipped": skipped,
        "bytes": total,
    }


# --------------------------------------------------------------------------
# Evidence: final code, commands, run output, final message
# --------------------------------------------------------------------------


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_text(v) for v in value)
    if isinstance(value, dict):
        for key in ("content", "text", "output", "results"):
            if key in value:
                return _text(value[key])
        return json.dumps(value)
    return "" if value is None else str(value)


_HEREDOC = re.compile(
    r"(?:cat|tee)\s+>{1,2}\s*(?P<path>[^\s<]+\.py)\s*<<-?\s*['\"]?(?P<tag>\w+)['\"]?\n(?P<body>.*?)\n(?P=tag)\b",
    re.S,
)
_PY_HEREDOC = re.compile(
    r"python3?\s+-[^\n<]*<<-?\s*['\"]?(?P<tag>\w+)['\"]?\n(?P<body>.*?)\n(?P=tag)\b",
    re.S,
)
_PATCH_FILE = re.compile(
    r"\*\*\* (?P<op>Add|Update) File: (?P<path>\S+\.py)\n(?P<body>.*?)(?=\n\*\*\* |\Z)",
    re.S,
)


def _apply_update(old: str, body: str) -> str:
    """Apply the hunks of an apply_patch ``Update File`` section to ``old``."""
    text = old
    for hunk in re.split(r"(?m)^@@.*$", body):
        lines = hunk.splitlines()
        before = [ln[1:] for ln in lines if ln[:1] in (" ", "-")]
        after = [ln[1:] for ln in lines if ln[:1] in (" ", "+")]
        if not before and not after:
            continue
        a, b = "\n".join(before), "\n".join(after)
        if a and a in text:
            text = text.replace(a, b, 1)
        elif not a:
            text = text + "\n" + b
    return text


# Commands that show a file's text. A grep or rg on a note shows the lines the
# agent looked for, so it counts as a read too.
_READ_VERB = re.compile(r"^(?:cat|sed|head|tail|less|more|nl|awk|grep|egrep|rg)$")
READ_TOOLS = {"read", "read_file", "view", "open", "cat"}


def _expand(name: str) -> list[str]:
    """Expand one shell brace group: ``{data,models}`` -> data, models."""
    if name.startswith("{") and name.endswith("}"):
        return [n for n in name[1:-1].split(",") if n]
    return [name]


def _segments(cmd: str) -> list[list[str]]:
    """Split a shell command into simple commands, respecting quotes."""
    lexer = shlex.shlex(cmd, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    lexer.commenters = ""
    segments: list[list[str]] = [[]]
    try:
        for tok in lexer:
            if tok and set(tok) <= set(";&|"):
                segments.append([])
            else:
                segments[-1].append(tok)
    except ValueError:  # unbalanced quotes: fall back to a plain split
        segments = [seg.split() for seg in re.split(r"\|\||&&|[;|\n]", cmd)]
    return [seg for seg in segments if seg]


def command_reads(cmd: str, workdir: str = "") -> list[str]:
    """Notes under ``references/`` that a shell command prints.

    Counts ``cat``, ``sed``, ``head``, ``tail``, ``less``, ``grep`` and similar
    on ``references/<note>.md``, including brace groups such as
    ``references/{data,models}.md``, a shell variable set to a ``references``
    folder (``S=.../references; cat $S/models.md``), and a bare ``<note>.md``
    after a ``cd`` into ``references`` or with a ``workdir`` there.
    """
    found: list[str] = []
    in_refs = workdir.rstrip("/").endswith("references")
    ref_vars: set[str] = set()
    for words in _segments(cmd.replace("\n", ";")):
        assign = (
            re.fullmatch(r"(\w+)=(\S*)", words[-1])
            if words[0] in {words[-1], "export"}
            else None
        )
        if assign:
            if assign.group(2).rstrip("/").endswith("references"):
                ref_vars.add(assign.group(1))
            continue
        if words[0] == "cd" and len(words) > 1:
            in_refs = words[1].rstrip("/").endswith("references")
            continue
        if not _READ_VERB.match(words[0]):
            continue
        for arg in words[1:]:
            for var in ref_vars:  # $S/models.md and ${S}/models.md
                arg = re.sub(rf"^\$\{{?{var}\}}?/", "references/", arg)
            m = re.fullmatch(r"(.*/)?(\{[\w,.-]+\}|[\w.-]+)\.md", arg)
            if not m:
                continue
            head = (m.group(1) or "").rstrip("/")
            if head.endswith("references") or (not head and in_refs):
                found += [f"references/{n}.md" for n in _expand(m.group(2))]
    return found


def trajectory_evidence(traj: dict) -> dict:
    """Return written files, commands, run output, notes read, and final message."""
    written: dict[str, str] = {}
    reads: list[str] = []
    snippets: list[str] = []
    commands: list[str] = []
    outputs: list[str] = []
    # The reply is every agent message after the last tool call, so a reply
    # split over several messages counts whole. Agent CLI error reports
    # (model_name "<synthetic>", or extra.is_error) are not the reply.
    reply: list[str] = []
    last_message = ""  # fallback when the run ended on a tool call
    session_python = False  # a python process may still be streaming output
    for step in traj.get("steps", []) if isinstance(traj, dict) else []:
        if step.get("source") not in {"agent", "assistant"}:
            continue
        if step.get("tool_calls"):
            reply = []
        message = _text(step.get("message")).strip()
        extra = step.get("extra")
        cli_error = step.get("model_name") == "<synthetic>" or (
            isinstance(extra, dict) and extra.get("is_error") is True
        )
        if message and not cli_error:
            last_message = message
            if not step.get("tool_calls"):  # a note before a tool call is not the reply
                reply.append(message)
        ran_python = False
        obs_by_call = {}
        obs_val = step.get("observation")
        if isinstance(obs_val, dict):
            for res in obs_val.get("results") or []:
                if isinstance(res, dict):
                    obs_by_call[str(res.get("source_call_id"))] = _text(
                        res.get("content")
                    )
        for call in step.get("tool_calls") or []:
            call_out = obs_by_call.get(str(call.get("tool_call_id")), "")
            write_failed = bool(
                re.search(
                    r"apply_patch: (command )?not found|No such file or directory: .*apply_patch"
                    r"|requires a UTF-8 PATCH|Failed to (apply|find)|patch failed",
                    call_out,
                )
            )
            if (
                str(call.get("function_name", "")).lower() == "write_stdin"
                and session_python
            ):
                ran_python = True
            args = call.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            path = str(args.get("file_path") or args.get("path") or "")
            name = str(call.get("function_name", "")).lower()
            if path and name in READ_TOOLS:
                reads.append(path)
            body = args.get("content")
            if path.endswith(".py") and isinstance(body, str):
                written[path] = body
            cmd = str(args.get("command") or args.get("cmd") or args.get("patch") or "")
            if not cmd:
                continue
            commands.append(cmd)
            reads += command_reads(cmd, str(args.get("workdir") or ""))
            if re.search(r"(^|[\s;&|(/])python3?\b|\buv run\b", cmd) and not re.search(
                r"python3?\s+-c\s+['\"]import\s+\w+;\s*print\(\w+\.__(version|file)__",
                cmd,
            ):
                ran_python = session_python = True
            for m in _HEREDOC.finditer(cmd):
                written[m.group("path")] = m.group("body")
            for m in [] if write_failed else _PATCH_FILE.finditer(cmd):
                path_, body_ = m.group("path"), m.group("body")
                if m.group("op") == "Add":
                    lines = [ln[1:] for ln in body_.splitlines() if ln.startswith("+")]
                    written[path_] = "\n".join(lines)
                elif path_ in written:
                    written[path_] = _apply_update(written[path_], body_)
            snippets.extend(m.group("body") for m in _PY_HEREDOC.finditer(cmd))
        obs = step.get("observation")
        if obs and ran_python:  # program output, not source the agent read
            outputs.append(_text(obs))
    return {
        "written": written,
        "snippets": snippets,
        "commands": commands,
        "outputs": outputs,
        "reads": reads,
        "final": "\n\n".join(reply) or last_message,
    }


def workspace_code(root: Path, fixtures: set[str] = frozenset()) -> dict[str, str]:
    """Return the agent's .py files under ``root`` (not task fixtures or skills)."""
    files: dict[str, str] = {}
    if root.is_file():
        return {root.name: root.read_text(errors="replace")}
    if not root.is_dir():
        return files
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if (
            any(_skip_dir(part) and part != "input" for part in rel.parts[:-1])
            or path.name in fixtures
        ):
            continue
        if path.is_symlink() or path.stat().st_size > MAX_CODE_FILE_BYTES:
            continue
        files[rel.as_posix()] = path.read_text(errors="replace")
    return files


def main_code(
    files: dict[str, str], evidence: dict, fixtures: set[str] = frozenset()
) -> dict[str, str]:
    """Pick the agent's deliverable code.

    Workspace files win (they are the end state). Scripts the agent wrote
    outside the workspace (for example under /tmp) are added from the
    trajectory, with apply_patch updates applied in order.
    """
    code = dict(files)
    names = {Path(p).name for p in code}
    code.update(
        {
            path: body
            for path, body in evidence.get("written", {}).items()
            if Path(path).name not in names and Path(path).name not in fixtures
        }
    )
    if not code and evidence.get("snippets"):  # only inline `python - <<EOF` programs
        code = {"<heredoc>": max(evidence["snippets"], key=len)}
    return code


# --------------------------------------------------------------------------
# AST analysers
# --------------------------------------------------------------------------


def _parse(src: str) -> ast.AST | None:
    try:
        return ast.parse(src)
    except SyntaxError:
        return None


def _qualname(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_qualname(node.value)}.{node.attr}"
    return ""


def _sync_sites(nodes: list[ast.stmt]) -> list[tuple[int, str]]:
    """Return (line, text) of host-sync calls in a list of statements."""
    sites = []
    for stmt in nodes:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = _qualname(func)
            if (
                isinstance(func, ast.Attribute)
                and func.attr in SYNC_METHODS
                and not node.args
            ):
                sites.append((node.lineno, f".{func.attr}()"))
            elif name in SYNC_QUALIFIED:
                sites.append((node.lineno, name))
            elif (
                isinstance(func, ast.Name)
                and func.id in SYNC_FUNCS
                and len(node.args) == 1
                and isinstance(node.args[0], (ast.Attribute, ast.Subscript, ast.Call))
                and not _looks_host(node.args[0])
            ):
                sites.append((node.lineno, f"{func.id}(tensor)"))
        for node in ast.walk(stmt):
            # `if tensor.any():` / `assert (x < y).all()` read a device bool.
            test = getattr(node, "test", None)
            if isinstance(node, (ast.If, ast.While, ast.Assert)) and isinstance(
                test, ast.Call
            ):
                attr = getattr(test.func, "attr", "")
                if attr in {"any", "all"}:
                    sites.append((node.lineno, f"branch on .{attr}()"))
    return sites


def _looks_host(node: ast.AST) -> bool:
    """True for expressions that are plain Python values (len(), os.environ...)."""
    name = _qualname(node.func) if isinstance(node, ast.Call) else _qualname(node)
    return any(
        name.startswith(p)
        for p in (
            "len",
            "os.",
            "math.",
            "np.",
            "numpy.",
            "self.threshold",
            "args.",
            "str",
        )
    ) or name.endswith(
        ("num_graphs", "num_nodes", "step_count", "n_steps", "frequency")
    )


def _modulo_gate(body: list[ast.stmt]) -> int | None:
    """Line after an early ``if step % n: return`` gate, if the body has one."""
    for stmt in body:
        if (
            isinstance(stmt, ast.If)
            and any(
                isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod)
                for n in ast.walk(stmt.test)
            )
            and any(isinstance(s, ast.Return) for s in stmt.body)
        ):
            return stmt.end_lineno
    return None


def _class_frequency(cls: ast.ClassDef) -> int | None:
    for stmt in cls.body:
        targets = []
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            targets, value = [stmt.target], stmt.value
        for t in targets:
            names = [t] if isinstance(t, ast.Name) else list(getattr(t, "elts", []))
            for i, n in enumerate(names):
                if isinstance(n, ast.Name) and n.id == "frequency":
                    v = value.elts[i] if isinstance(value, ast.Tuple) else value
                    if isinstance(v, ast.Constant) and isinstance(v.value, int):
                        return v.value
    return None


def loop_sync_report(code: dict[str, str]) -> dict:
    """Count host syncs inside hand-written step loops and per-step hooks.

    Hook methods (``__call__``, ``pre_update``, ``post_update``) of classes
    with a class-level ``frequency`` above 1, and code after an early
    ``if ... % n: return`` gate, are frequency-gated and do not count.
    Loops count as step loops when their body calls ``.step()``,
    ``.run()``, or ``.backward()``.
    """
    sites: list[str] = []
    gated: list[str] = []
    for path, src in code.items():
        tree = _parse(src)
        if tree is None:
            continue
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            freq = _class_frequency(cls)
            for fn in cls.body:
                if not isinstance(fn, ast.FunctionDef) or fn.name not in HOOK_METHODS:
                    continue
                gate = _modulo_gate(fn.body)
                for line, what in _sync_sites(fn.body):
                    tag = f"{path}:{line} {what} in {cls.name}.{fn.name}"
                    if (freq is not None and freq > 1) or (
                        gate is not None and line > gate
                    ):
                        gated.append(tag)
                    else:
                        sites.append(tag)
        for loop in [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]:
            calls = [
                n
                for s in loop.body
                for n in ast.walk(s)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in STEP_CALL_ATTRS
            ]
            if not calls:
                continue
            for line, what in _sync_sites(loop.body):
                sites.append(f"{path}:{line} {what} in a step loop")
    return {"sites": sorted(set(sites)), "gated": sorted(set(gated))}


def system_loops(code: dict[str, str], calls: list[str]) -> list[str]:
    """Python loops whose body runs an engine or model (one run per system)."""
    found = []
    pattern = re.compile("|".join(calls)) if calls else None
    for path, src in code.items():
        tree = _parse(src)
        if tree is None:
            continue
        for loop in [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]:
            body = "\n".join(ast.get_source_segment(src, s) or "" for s in loop.body)
            if pattern and pattern.search(body):
                found.append(f"{path}:{loop.lineno}")
    return found


def loops_in_methods(code: dict[str, str], methods: set[str]) -> list[str]:
    """Python for/while loops inside per-step methods (hooks, updates, forward).

    Loops in a hook class with a class-level ``frequency`` above 1, or after
    an early ``if ... % n: return`` gate, run only every N steps and do not
    count. Loops over a constant range (for example a fixed number of
    denoising steps) do not count either; only loops whose iterable is a
    tensor, a ``range`` over a batch size, or a ``.tolist()`` do.
    """
    found = []
    for path, src in code.items():
        tree = _parse(src)
        if tree is None:
            continue
        owners = {}
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef):
                    owners[id(fn)] = _class_frequency(cls)
        for fn in [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in methods
        ]:
            freq = owners.get(id(fn))
            if freq is not None and freq > 1:
                continue
            gate = _modulo_gate(fn.body)
            for node in ast.walk(fn):
                if not isinstance(node, (ast.For, ast.While)):
                    continue
                if gate is not None and node.lineno > gate:
                    continue
                it = (
                    ast.get_source_segment(src, node.iter)
                    if isinstance(node, ast.For)
                    else "while"
                )
                if isinstance(node, ast.For) and re.fullmatch(
                    r"range\(\s*[\w.]*?\d+\s*\)|range\(self\.\w*steps\w*\)", it or ""
                ):
                    continue
                if isinstance(node, ast.For) and not re.search(
                    r"tolist|range\(|num_graphs|num_nodes|batch_ptr|batch_idx|unique|nonzero|split|positions|zip\(",
                    it or "",
                ):
                    continue  # loops over dicts, keys, or config lists are not per-atom work
                found.append(f"{path}:{node.lineno} in {fn.name}: {it}")
    return found


def calls_missing_kw(code: dict[str, str], func: str, kw: str) -> list[str]:
    """Calls to ``func`` that do not pass keyword ``kw``."""
    missing = []
    for path, src in code.items():
        tree = _parse(src)
        if tree is None:
            continue
        missing.extend(
            f"{path}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _qualname(node.func).split(".")[-1] == func
            and kw not in {k.arg for k in node.keywords}
        )
    return missing


def hardcoded_cpu(code: dict[str, str]) -> list[str]:
    """Places that pin ``cpu`` outside a cuda-or-cpu conditional."""
    found = []
    for path, src in code.items():
        tree = _parse(src)
        if tree is None:
            continue
        cond_ranges = [
            (n.lineno, n.end_lineno)
            for n in ast.walk(tree)
            if isinstance(n, (ast.IfExp, ast.If))
            and "cuda" in (ast.get_source_segment(src, n.test) or "")
        ]
        for node in ast.walk(tree):
            hit = False
            if isinstance(node, ast.keyword) and node.arg in {"device", "device_type"}:
                v = node.value
                hit = isinstance(v, ast.Constant) and v.value == "cpu"
                hit = hit or (
                    isinstance(v, ast.Call)
                    and _qualname(v.func) == "torch.device"
                    and v.args
                    and isinstance(v.args[0], ast.Constant)
                    and v.args[0].value == "cpu"
                )
            elif isinstance(node, ast.Call) and _qualname(node.func) == "torch.device":
                hit = (
                    bool(node.args)
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "cpu"
                )
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                hit = node.value.value == "cpu" and any(
                    "device" in _qualname(t).lower() for t in node.targets
                )
            line = getattr(node, "lineno", None) or getattr(
                getattr(node, "value", None), "lineno", 0
            )
            if hit and not any(a <= line <= b for a, b in cond_ranges):
                found.append(f"{path}:{line}")
    return sorted(set(found))


# --------------------------------------------------------------------------
# Check runner
# --------------------------------------------------------------------------


def _scope_text(scope: str, code: dict[str, str], ev: dict) -> str:
    if scope == "code":
        return "\n".join(code.values())
    if scope == "commands":
        return "\n".join(ev["commands"] + ev["snippets"])
    if scope == "output":
        return "\n".join(ev["outputs"])
    if scope == "final":
        return ev["final"]
    if scope == "reads":
        return "\n".join(ev.get("reads", []))
    if scope == "run":
        return "\n".join([*ev["outputs"], ev["final"]])
    if scope == "all":
        return "\n".join([*code.values(), *ev["commands"], *ev["outputs"], ev["final"]])
    raise ValueError(f"unknown scope {scope}")


def run_check(check: dict, code: dict[str, str], ev: dict) -> tuple[bool, str]:
    """Return (passed, reason) for one check."""
    scope = check.get("scope", "code" if check["kind"] == "static" else "output")
    analyser = check.get("analyser")
    if analyser == "no_loop_syncs":
        rep = loop_sync_report(code)
        limit = int(check.get("max", 0))
        ok = len(rep["sites"]) <= limit and bool(code)
        return (
            ok,
            f"{len(rep['sites'])} loop sync site(s) {rep['sites'][:6]}; gated {len(rep['gated'])}",
        )
    if analyser == "no_system_loop":
        found = system_loops(code, check.get("calls", []))
        return (
            not found and bool(code)
        ), f"loops over systems: {found}" if found else "none"
    if analyser == "no_loop_in_methods":
        found = loops_in_methods(code, set(check.get("methods", sorted(HOOK_METHODS))))
        return (not found and bool(code)), f"loops: {found}" if found else "none"
    if analyser == "calls_have_kw":
        missing = calls_missing_kw(code, check["func"], check["kw"])
        return (
            not missing,
            f"{check['func']} without {check['kw']}=: {missing}" if missing else "ok",
        )
    if analyser == "device_aware":
        if not code or set(code) == {"<heredoc>"}:
            return False, "no script written"
        text = "\n".join(code.values())
        picks = re.search(
            r"cuda\.is_available\(\)|resolve_device\(|device_count\(\)", text
        )
        pinned = hardcoded_cpu(code)
        ok = bool(picks) and not pinned
        return ok, f"picks cuda: {bool(picks)}; cpu pinned at {pinned}"
    text = _scope_text(scope, code, ev)
    missing = [p for p in check.get("require", []) if not re.search(p, text, re.M)]
    present = [p for p in check.get("forbid", []) if re.search(p, text, re.M)]
    if "expect" in check:  # numeric run check: first capture group near expect
        values = [float(v) for v in re.findall(check["pattern"], text, re.M)]
        tol = float(check.get("tol", 0))
        ok = any(abs(v - float(check["expect"])) <= tol for v in values)
        return ok, f"found {values[-5:]}, expect {check['expect']} +- {tol}"
    ok = not missing and not present and (bool(text) or not check.get("require"))
    reason = []
    if missing:
        reason.append(f"missing {missing}")
    if present:
        reason.append(f"forbidden {present}")
    return ok, "; ".join(reason) or "ok"


SKILL_NAME = "nvalchemi-toolkit"


def skill_present(entry: dict) -> bool:
    """True unless entry.json says this trial ran without this skill.

    SkillEvaluator writes ``has_skill`` into entry.json: false in the
    baseline arm. A run that installs a different skill under test names it
    in ``expected_skill``. Offline runs without ``has_skill`` count as present.
    """
    if entry.get("has_skill") is False:
        return False
    return entry.get("expected_skill") in (None, SKILL_NAME)


def grade(entry: dict, code: dict[str, str], ev: dict) -> dict:
    """Run every check of ``entry`` and return metrics and details.

    Checks marked ``"needs_skill": true`` (for example which notes the
    agent read) are skipped when the skill is absent, so the baseline arm
    is not marked down for files it never had.
    """
    results = []
    skipped = []
    with_skill = skill_present(entry)
    for check in entry.get("checks", []):
        if check.get("kind") == "gpu":  # graded after the trial by gpu/gpu_check.py
            continue
        if check.get("needs_skill") and not with_skill:
            skipped.append(check["id"])
            continue
        try:
            ok, reason = run_check(check, code, ev)
        except Exception as exc:  # a broken check must not kill the verifier
            ok, reason = False, f"check error: {exc}"
        results.append(
            {
                "id": check["id"],
                "metric": check["metric"],
                "passed": ok,
                "reason": reason,
            }
        )
    metrics: dict[str, float] = {}
    details: dict[str, dict] = {}
    for m in METRICS:
        rows = [r for r in results if r["metric"] == m]
        if rows:
            metrics[m] = round(sum(r["passed"] for r in rows) / len(rows), 4)
            details[m] = {
                "score": metrics[m],
                "reason": "; ".join(
                    f"{r['id']}={'pass' if r['passed'] else 'FAIL'}" for r in rows
                ),
            }
    # SkillEvaluator's reward contract needs every declared metric in every
    # reward. A metric this task does not check is reported as 1.0 and marked
    # not_checked, so reports can leave it out of their averages.
    reward_metrics = dict(metrics)
    for m in METRICS:
        if m not in reward_metrics:
            reward_metrics[m] = 1.0
            details[m] = {"score": 1.0, "not_checked": True, "reason": "no check"}
    for m in METRICS:  # say which checks the baseline arm skipped
        ids = [
            c["id"]
            for c in entry.get("checks", [])
            if c["id"] in skipped and c["metric"] == m
        ]
        if ids:
            details[m]["reason"] += "; " + "; ".join(
                f"{i}=not_checked (skill absent)" for i in ids
            )
    return {
        "metrics": metrics,
        "reward_metrics": reward_metrics,
        "details": details,
        "checks": results,
        "skipped_without_skill": skipped,
        "code_files": sorted(code),
    }


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def verifier_main() -> int:
    """Entry point inside the Harbor verifier; always writes a reward."""
    try:
        return _verifier_main()
    except Exception as exc:  # never zero a trial because the grader broke
        verifier = Path(os.environ.get("HARBOR_VERIFIER_DIR", "/logs/verifier"))
        reward = Path(os.environ.get("HARBOR_REWARD_JSON", verifier / "reward.json"))
        reward.parent.mkdir(parents=True, exist_ok=True)
        reward.write_text(
            json.dumps(
                {
                    "custom_metrics": {m: 0.0 for m in METRICS},
                    "grader_error": str(exc)[:500],
                }
            )
        )
        return 0


def _verifier_main() -> int:
    workspace = Path(os.environ.get("HARBOR_WORKSPACE_DIR", "/workspace"))
    trial = Path(os.environ.get("HARBOR_LOGS_DIR", "/logs"))
    verifier = Path(os.environ.get("HARBOR_VERIFIER_DIR", trial / "verifier"))
    agent_dir = Path(os.environ.get("HARBOR_AGENT_LOGS_DIR", trial / "agent"))
    atif = Path(os.environ.get("HARBOR_ATIF_PATH", agent_dir / "trajectory.json"))
    entry_path = Path(os.environ.get("HARBOR_ENTRY_JSON", "/tests/entry.json"))
    reward = Path(os.environ.get("HARBOR_REWARD_JSON", verifier / "reward.json"))

    try:
        summary = keep_workspace(workspace, trial / "kept-workspace")
    except OSError as exc:
        summary = {"workspace": str(workspace), "error": str(exc)}
    try:
        (trial / "kept-workspace.json").write_text(json.dumps(summary, indent=2))
    except OSError:
        pass

    entry = _load(entry_path)
    ev = trajectory_evidence(_load(atif))
    fixtures = {Path(f).name for f in entry.get("files", []) or []}
    code = main_code(workspace_code(workspace, fixtures), ev, fixtures)
    result = grade(entry, code, ev)
    verifier.mkdir(parents=True, exist_ok=True)
    (verifier / "nvalchemi_checks.json").write_text(json.dumps(result, indent=2))
    reward.parent.mkdir(parents=True, exist_ok=True)
    reward.write_text(
        json.dumps(
            {"custom_metrics": result["reward_metrics"], "details": result["details"]},
            indent=2,
        )
    )
    return 0


def offline_main(argv: list[str]) -> int:
    """Grade a script or directory outside Harbor and print the result."""
    p = argparse.ArgumentParser()
    p.add_argument("--offline", action="store_true")
    p.add_argument("--entry", required=True, help="case JSON, or evals.json plus --id")
    p.add_argument("--id")
    p.add_argument("--code", required=True)
    p.add_argument("--trajectory")
    p.add_argument("--output", help="text file with the run output")
    p.add_argument("--final", help="text file with the final message")
    a = p.parse_args(argv)
    entry = _load(Path(a.entry))
    if "evals" in entry:
        entry = next(c for c in entry["evals"] if c["id"] == a.id)
    ev = (
        trajectory_evidence(_load(Path(a.trajectory)))
        if a.trajectory
        else {
            "written": {},
            "snippets": [],
            "commands": [],
            "outputs": [],
            "reads": [],
            "final": "",
        }
    )
    if a.output:
        ev["outputs"].append(Path(a.output).read_text(errors="replace"))
    if a.final:
        ev["final"] = Path(a.final).read_text(errors="replace")
    fixtures = {Path(f).name for f in entry.get("files", []) or []}
    code = main_code(workspace_code(Path(a.code), fixtures), ev, fixtures)
    print(json.dumps(grade(entry, code, ev), indent=2))
    return 0


if __name__ == "__main__":
    if "--offline" in sys.argv:
        sys.exit(offline_main(sys.argv[1:]))
    sys.exit(verifier_main())
