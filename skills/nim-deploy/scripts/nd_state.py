#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run state and the output contract for the nim-deploy scripts (imported through nd_common).

Every script prints one JSON object (see ``emit``) matching references/script-output.schema.json of
the write-skill skill. Run state lives in ``<runs_root>/runs/<run_id>/state.json`` and is the
checkpoint that lets a run resume after the chat context is lost. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runners

EXIT_OK, EXIT_INPUT, EXIT_PRECONDITION, EXIT_EXTERNAL, EXIT_REFUSED = 0, 2, 3, 4, 5

STATES = (
    "intake",
    "plan",
    "render",
    "validate",
    "approve",
    "apply",
    "wait_ready",
    "verify",
    "record",
    "done",
    "diagnose",
    "rollback",
)
EDGES: dict[str, tuple[str, ...]] = {
    "intake": ("plan",),
    "plan": ("render",),
    "render": ("validate",),
    "validate": ("approve", "render"),
    "approve": ("apply", "render"),  # render: an input changed before the user approved
    "apply": ("wait_ready", "diagnose"),
    "wait_ready": ("verify", "diagnose"),
    "verify": ("record", "diagnose"),
    "diagnose": ("rollback", "render", "record"),
    "rollback": ("record",),
    "record": ("done",),
    "done": ("diagnose",),  # a finished deployment can still be stopped: diagnose > rollback
}
# The supported targets.
# The docker target's required inputs come from its runner (runners/__init__.py).
REQUIRED: dict[str, tuple[str, ...]] = {
    "helm": ("image", "namespace", "context", "name", "ngc_secret", "gpus"),
    "nim-operator": ("image", "namespace", "context", "name", "ngc_secret", "gpus"),
}
TARGETS = ("docker", *REQUIRED)
MANIFESTS = {"nim-operator": "nimservice.yaml"}  # targets deployed with kubectl apply -f FILE
MAX_VISITS = 3
APPROVAL_TTL = timedelta(minutes=60)
DOCS_TTL = timedelta(minutes=60)  # a docs check older than this is stale at approval time
# Input sources the skill chose itself (recomputed when the image changes); others are the
# user's ("user") or fetched facts handed in by the agent ("docs") and are never overwritten.
AUTO_SOURCES = ("default", "image-metadata", "documented-default")
# After approve the inputs are frozen: the approval record binds them.
APPROVED_STATES = ("apply", "wait_ready", "verify", "diagnose", "rollback", "record", "done")

TRUE_TEXT, FALSE_TEXT = {"true", "1", "yes", "on"}, {"false", "0", "no", "off"}
SECRET_KEY = re.compile(r"(?i)key|token|password|secret")
SECRET_VALUE = re.compile(r"nvapi-|^hf_[A-Za-z0-9]{8,}|^sk-[A-Za-z0-9_-]{8,}|^Bearer\s", re.I)
SECRET_FIX = (
    "Do not put secret values in the run. Store the value in a Kubernetes Secret (or export it as "
    "an environment variable) and pass only the secret NAME, e.g. ngc_secret=ngc-api."
)
DNS_LABEL = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?")
OBJECT_NAME = re.compile(r"[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?")
IMAGE_REFERENCE = re.compile(
    r"(?:[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?/)?"
    r"(?:[a-z0-9]+(?:[._-][a-z0-9]+)*/)*"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*"
    r"(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?"
    r"(?:@sha256:[A-Fa-f0-9]{64})?"
)
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
HANDOFF_CONFIRM = "confirm_handoff"  # the user's yes to result files that need a question


# --- output contract ------------------------------------------------------------------------


SCRIPTS_DIR = Path(__file__).resolve().parent
BARE_SCRIPT = re.compile(
    r"(?<![\w/.\-])(nimdeploy|verify|validate|render_docker|render_k8s|approvals|guard)\.py\b"
)


def script_command(name: str) -> str:
    """`python3 <absolute, shell-quoted path>` for a script of this skill (NAME like verify.py)."""
    return f"python3 {shlex.quote(str(SCRIPTS_DIR / name))}"


def absolute_scripts(text: str) -> str:
    """Replace a bare `nimdeploy.py ...` in advice text with the runnable absolute command, so the
    advice works from any folder and any install location."""
    return BARE_SCRIPT.sub(lambda match: script_command(match.group(0)), text)


def error(code: str, message: str, fix: str) -> dict[str, str]:
    return {"code": code, "message": absolute_scripts(message), "fix": absolute_scripts(fix)}


def emit(
    ok: bool,
    state: str,
    next_: str,
    facts: dict[str, Any] | None = None,
    errors: list[dict[str, str]] | None = None,
    code: int | None = None,
) -> NoReturn:
    payload = {
        "ok": ok,
        "state": state,
        "next": next_,
        "facts": facts or {},
        "errors": errors or [],
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()
    raise SystemExit(code if code is not None else (EXIT_OK if ok else EXIT_INPUT))


def log(message: str) -> None:
    print(message, file=sys.stderr)


class JsonArgumentParser(argparse.ArgumentParser):
    """Report usage errors as the standard JSON result (exit 2) instead of argparse text."""

    def error(self, message: str) -> NoReturn:
        fix = f"See: {self.prog} --help"
        emit(False, "input", "ask_user", errors=[error("bad_input", message, fix)])


# --- strict inputs --------------------------------------------------------------------------


def flag(value: Any) -> bool:
    """Strict boolean: bool, 0/1, or true/false/yes/no/on/off in any case; else ValueError."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    text = value.strip().lower() if isinstance(value, str) else None
    if text in TRUE_TEXT:
        return True
    if text in FALSE_TEXT:
        return False
    raise ValueError(f"expected true or false, got {value!r}")


def looks_secret(key: str, value: Any) -> bool:
    """A secret-shaped value, or a secret-named key whose value is not a Secret NAME or a switch."""
    if isinstance(value, str) and SECRET_VALUE.search(value):
        return True
    if not SECRET_KEY.search(key):
        return False
    if isinstance(value, bool) or str(value).lower() in TRUE_TEXT | FALSE_TEXT:
        return False
    return not (key.endswith("_secret") and OBJECT_NAME.fullmatch(str(value)) is not None)


def is_dns_label(value: Any) -> bool:
    return isinstance(value, str) and len(value) <= 63 and bool(DNS_LABEL.fullmatch(value))


# --- run state ------------------------------------------------------------------------------


def ensure_runs_root(root: Path) -> Path:
    """Create the runs root (mode 0700) with a .gitignore of '*' so run state never reaches git."""
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    ignore = root / ".gitignore"
    if not ignore.exists():
        ignore.write_text("*\n")
    return root


def default_runs_root(cwd: Path | None = None) -> Path:
    """Where the runs live when NIM_DEPLOY_HOME is not set, wherever the skill is installed.

    1. $CLAUDE_PROJECT_DIR/.nim-deploy when that variable is set;
    2. the nearest folder from the current one upwards that has a `.git` or a `.nim-deploy`;
    3. else <current folder>/.nim-deploy.
    A current folder inside a skill folder (one with a SKILL.md) is replaced by the folder that
    holds the skill, so the run store never lands inside an install; with no project there it is
    ~/.nim-deploy."""
    project = os.environ.get("CLAUDE_PROJECT_DIR")
    if project:
        return Path(project) / ".nim-deploy"
    start = (cwd or Path.cwd()).resolve()
    inside_skill = next((f for f in (start, *start.parents) if (f / "SKILL.md").is_file()), None)
    if inside_skill is not None:
        start = inside_skill.parent
    for folder in (start, *start.parents):
        if (folder / ".git").exists() or (folder / ".nim-deploy").is_dir():
            return folder / ".nim-deploy"
    return (Path.home() if inside_skill is not None else start) / ".nim-deploy"


def runs_root() -> Path:
    home = os.environ.get("NIM_DEPLOY_HOME")
    return ensure_runs_root(Path(home) if home else default_runs_root())


def run_dir(run_id: str) -> Path:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError(f"invalid run id: {run_id!r}")
    path = runs_root() / "runs" / run_id
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def utc_now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - datetime.UTC needs Python 3.11


def new_run_id(target: str, model: str, now: datetime) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")[:24].strip("-")
    return f"{now:%Y%m%dT%H%M%S}-{target}-{slug}"


def write_atomic(path: Path, text: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state(run_id: str) -> dict[str, Any]:
    path = run_dir(run_id) / "state.json"
    if not path.exists():
        raise FileNotFoundError(f"no run {run_id!r} under {runs_root()}")
    state: dict[str, Any] = json.loads(path.read_text())
    return state


def save_state(state: dict[str, Any]) -> None:
    text = json.dumps(state, indent=2, sort_keys=True) + "\n"
    write_atomic(run_dir(state["run_id"]) / "state.json", text)


def input_value(state: dict[str, Any], key: str) -> Any:
    entry = state.get("inputs", {}).get(key)
    return entry.get("value") if isinstance(entry, dict) else None


def run_inputs(state: dict[str, Any]) -> dict[str, Any]:
    """The non-empty inputs plus run_id: what the renderers read."""
    inputs = {key: input_value(state, key) for key in state.get("inputs", {})}
    inputs = {key: value for key, value in inputs.items() if value not in (None, "")}
    return {**inputs, "run_id": state["run_id"]}


def missing_inputs(state: dict[str, Any]) -> list[str]:
    extra: list[str] = []
    if state["target"] == "docker":
        runner = runners.runner_of(state)
        required = runner.required_inputs()
        further = getattr(runner, "missing_inputs", None)
        extra = further(run_inputs(state)) if further else []
    else:
        required = REQUIRED.get(state["target"], ())
    missing = [key for key in required if input_value(state, key) in (None, "")]
    missing += [key for key in extra if key not in missing]
    return missing + [i["key"] for i in handoff_pending(state) if i["key"] not in missing]


# Any one of these, given by the user, answers the "placement" question of a select file.
PLACEMENT_ANSWER = ("placement", "node_selector", "node_hostnames")


def placement_answered(state: dict[str, Any]) -> bool:
    # Imported at use time: render_k8s itself imports the shared state helpers.
    import render_k8s

    answer = {
        key: entry.get("value")
        for key in PLACEMENT_ANSWER
        if (entry := state.get("inputs", {}).get(key) or {}).get("source") == "user"
    }
    try:
        selector = render_k8s.node_selector(answer)
        hostnames = render_k8s.node_hostnames(answer)
        return answer.get("placement") == "none" or bool(selector or hostnames)
    except ValueError:
        return False


def handoff_pending(state: dict[str, Any]) -> list[dict[str, str]]:
    """Questions raised by `new --from` files (scripts/handoff.py) that the user has not answered:
    HANDOFF_CONFIRM needs a yes; any other key needs a value the user gave."""
    open_items = []
    for item in (state.get("handoff") or {}).get("pending", []):
        entry = state.get("inputs", {}).get(item["key"]) or {}
        if item["key"] == HANDOFF_CONFIRM:
            try:
                answered = flag(entry.get("value", False))
            except ValueError:
                answered = False
        elif item["key"] == PLACEMENT_ANSWER[0]:
            answered = placement_answered(state)
        else:
            answered = entry.get("source") == "user" and entry.get("value") not in (None, "")
        if not answered:
            open_items.append(item)
    return open_items + [
        item for item in overridden_pending(state)
        if item["key"] not in {pending["key"] for pending in open_items}
    ]


def gpu_count(state: dict[str, Any]) -> int | None:
    """How many GPUs the `gpus` input names, or None when it cannot be told (`all`)."""
    text = str(input_value(state, "gpus") or "").strip().strip("\"'")
    if text.startswith("count="):
        text = text[len("count=") :]
    elif "," in text or text.startswith("device=") or runners.name_of(state) == "nimcli":
        return len([i for i in text.removeprefix("device=").split(",") if i.strip()]) or None
    return int(text) if text.isdigit() else None


def overridden_pending(state: dict[str, Any]) -> list[dict[str, str]]:
    """Questions for a user change after `new --from` that the result files do not cover.

    A profile id belongs to one image version: a new image needs the profile to be set too (any
    value the user gives counts). Fewer GPUs than the profile needs needs a yes."""
    record = state.get("handoff") or {}
    given = record.get("values", {})
    inputs = state.get("inputs", {})
    found = []
    placed = any(given.get(key) for key in ("node_selector", "node_hostnames"))
    if state["target"] in ("helm", "nim-operator") and (placed or state.get("placement_pending")):
        scope_changed = state.get("placement_pending") or (
            given.get("context") and input_value(state, "context") != given["context"]["value"]
        )
        by_user = placement_answered(state)
        if scope_changed and not by_user:
            found.append(
                {
                    "key": "placement",
                    "question": "The Kubernetes target changed, so the placement nim-select "
                    "found for the old cluster does not apply. Inspect this cluster and set "
                    "node_selector (or node_hostnames), or set placement=none.",
                }
            )
    image = (given.get("image") or {}).get("value")
    profile_by_user = (inputs.get("model_profile") or {}).get("source") == "user"
    if image and input_value(state, "image") != image and not profile_by_user:
        found.append(
            {
                "key": "model_profile",
                "question": f"The image changed from {image} to {input_value(state, 'image')}, "
                "but profile ids belong to one image version. Which profile should it use on "
                "the new image (set model_profile=ID; the same id only if you checked it)?",
            }
        )
    needed = (record.get("profile") or {}).get("gpus_needed")
    count = gpu_count(state) if (inputs.get("gpus") or {}).get("source") == "user" else None
    if needed and count is not None and count < needed:
        try:
            answered = flag((inputs.get(HANDOFF_CONFIRM) or {}).get("value", False))
        except ValueError:
            answered = False
        if not answered:
            found.append(
                {
                    "key": HANDOFF_CONFIRM,
                    "question": f"gpus names {count} GPU(s), but the profile needs {needed}. "
                    f"Use fewer GPUs anyway? If yes: set {HANDOFF_CONFIRM}=yes",
                }
            )
    return found


def approval_bindings(state: dict[str, Any]) -> dict[str, Any]:
    """The inputs an approval record carries; the guard reads only these, never live inputs."""
    keys = ("name", "image", "model", "port", "container_port", "gpus", "model_profile")
    bound = {key: input_value(state, key) for key in keys}
    local = state["target"] == "docker"  # frozen now, so a later env change cannot swap the runner
    return {
        "target": state["target"],
        **bound,
        "runner": runners.name_of(state) if local else None,
        "release": bound["name"] if state["target"] == "helm" else None,
        "context": input_value(state, "context") or "",
        "namespace": input_value(state, "namespace") or "",
    }


def docs_problem(state: dict[str, Any], now: datetime | None = None) -> tuple[str, str, str] | None:
    """(code, message, fix) when the run has no fresh docs check; the gate before approval."""
    check = state.get("docs_check")
    fix = (
        "Fetch the official docs listed for this target in references/docs-sources.md, then run "
        "nimdeploy.py docs-check RUN --url URL --doc-version V --note TEXT (or, only after the "
        "user agreed the docs cannot be reached: --offline --reason TEXT)."
    )
    if not check:
        return "docs_check", "no docs check recorded for this run", fix
    fetched = datetime.fromisoformat(check["fetched_at"])
    if (now or utc_now()) - fetched > DOCS_TTL:
        minutes = int(DOCS_TTL.total_seconds() // 60)
        return "docs_stale", f"the docs check is older than {minutes} minutes", fix
    return None


def transition(state: dict[str, Any], to: str, artifacts: list[str] | None = None) -> None:
    """Move along a legal edge, hash artifacts, count visits (3 per state), then save."""
    current = state["state"]
    if to not in EDGES.get(current, ()):
        raise ValueError(f"illegal transition {current} -> {to}")
    visits = state.setdefault("attempts", {}).get(to, 0)
    if visits >= MAX_VISITS:
        raise ValueError(
            f"state {to!r} already had {visits} visit(s); ask the user before retrying"
        )
    for artifact in artifacts or []:
        state.setdefault("artifacts", {})[str(artifact)] = sha256_file(artifact)
    state["attempts"][to] = visits + 1
    state.setdefault("history", []).append({"from": current, "to": to, "at": utc_now().isoformat()})
    state["state"] = to
    save_state(state)


if __name__ == "__main__":
    emit(True, "library", "done", {"note": "import this module; it has no command line"})
