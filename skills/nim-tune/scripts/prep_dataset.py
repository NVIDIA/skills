#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""prep_dataset: write a real-text AIPerf multi-turn JSONL for a tuning campaign.

  prep_dataset.py ultrachat --tokenizer Qwen/Qwen2.5-7B-Instruct --out datasets/ultrachat.jsonl

Reads UltraChat 200k (HuggingFaceH4/ultrachat_200k) through the Hugging Face dataset viewer API,
one page of rows at a time, and writes one line per conversation:
{"session_id": ..., "turns": [{"text": USER_MESSAGE, "output_length": N}, ...]}. Each turn's
output_length is the token count of the real assistant reply that followed it, counted with the
model's own tokenizer, so the benchmark keeps the dataset's output-length spread. The default split
is test_sft: speculative drafts are often trained on the train split, which would make their
acceptance look better than on unseen text.

Point the campaign at it with workload.dataset {"file": ..., "type": "multi_turn"}.

Prints one JSON object: {"ok", "state", "next", "facts", "errors": [{"code", "message", "fix"}]}.
Exit codes: 0 ok, 2 bad input, 3 precondition failed, 4 external service failed. Never prompts.
Needs the `tokenizers` package for token counts (`transformers` installs it); everything else is
stdlib. Pass --tokenizer-revision COMMIT_SHA to count with fixed tokenizer files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NoReturn

EXIT_OK, EXIT_INPUT, EXIT_PRECONDITION, EXIT_EXTERNAL = 0, 2, 3, 4
ROWS_API = "https://datasets-server.huggingface.co/rows"
ULTRACHAT = "HuggingFaceH4/ultrachat_200k"
PAGE = 100  # the viewer API's largest page


def open_https(url: str, timeout: float) -> Any:
    """Open an https:// URL only: the opener has no file:, ftp: or plain http: handler, so neither
    the URL nor a redirect can reach another scheme. Proxy settings of the environment apply."""
    if not url.startswith("https://"):
        raise urllib.error.URLError(f"not an https URL: {url}")
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler(),
        urllib.request.HTTPSHandler(),
        urllib.request.HTTPDefaultErrorHandler(),
        urllib.request.HTTPRedirectHandler(),
        urllib.request.HTTPErrorProcessor(),
        urllib.request.UnknownHandler(),  # any other scheme, also after a redirect: URLError
    ):
        opener.add_handler(handler)
    return opener.open(url, timeout=timeout)


class MalformedResponse(ValueError):
    """The viewer API answered, but not with {"rows": [{"row": {...}}, ...]}."""


def err(code: str, message: str, fix: str) -> dict[str, str]:
    return {"code": code, "message": message, "fix": fix}


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
    raise SystemExit(code if code is not None else (EXIT_OK if ok else EXIT_INPUT))


class ContractParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        emit(False, "error", "fix_errors", errors=[err("bad_usage", message, "See --help.")])


def token_counter(tokenizer: str, revision: str) -> Callable[[str], int]:
    """Count tokens with the model's tokenizer files at `revision` (a commit sha pins them; a branch
    such as main can move). Only `tokenizers` is needed: `transformers` depends on it, and loads
    no code from the repository here."""
    from tokenizers import Tokenizer  # type: ignore[import-not-found,unused-ignore]

    tok = Tokenizer.from_pretrained(tokenizer, revision=revision)
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)


def fetch_rows(split: str, start: int, count: int) -> Iterator[dict[str, Any]]:
    """Yield UltraChat rows [start, start+count) from the dataset viewer API, retrying each page."""
    offset = start
    while offset < start + count:
        query = urllib.parse.urlencode(
            {
                "dataset": ULTRACHAT,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": min(PAGE, start + count - offset),
            }
        )
        body: Any = None
        for attempt in range(4):
            try:
                with open_https(f"{ROWS_API}?{query}", timeout=60) as resp:
                    body = json.load(resp)
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError):
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        rows = body.get("rows") if isinstance(body, dict) else None  # no "rows" is malformed
        if not isinstance(rows, list) or not all(well_formed(r) for r in rows):
            raise MalformedResponse(f"unexpected viewer response at offset {offset}")
        if not rows:
            return
        for r in rows:
            yield r["row"]
        offset += len(rows)


def well_formed(wrapper: Any) -> bool:
    """A viewer row wrapper {"row": {"messages": [{...}, ...], ...}} that to_session can read."""
    row = wrapper.get("row") if isinstance(wrapper, dict) else None
    # an absent or null messages is no conversation; any other non-list value is malformed
    msgs = ([] if row.get("messages") is None else row["messages"]) if isinstance(row, dict) else None
    return isinstance(msgs, list) and all(isinstance(m, dict) for m in msgs)


def to_session(
    row: dict[str, Any], count: Callable[[str], int], max_turns: int, max_output: int
) -> dict[str, Any] | None:
    """One conversation as an AIPerf session; each user turn is sized by the reply that follows."""
    turns: list[dict[str, Any]] = []
    msgs = row.get("messages") or []
    for i, m in enumerate(msgs):
        if m.get("role") != "user" or not str(m.get("content") or "").strip():
            continue
        nxt = msgs[i + 1] if i + 1 < len(msgs) else {}
        reply = str(nxt.get("content") or "") if nxt.get("role") == "assistant" else ""
        try:  # JSON allows an escaped lone surrogate; the tokenizer may not
            reply.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise MalformedResponse(f"a reply has text that is not valid Unicode: {exc.reason}")
        n = count(reply)
        if n == 0:
            break  # a turn with no real reply has no real output length
        turns.append({"text": str(m["content"]).strip(), "output_length": min(n, max_output)})
        if len(turns) == max_turns:
            break
    if not turns:
        return None
    return {"session_id": str(row.get("prompt_id") or len(turns)), "turns": turns}


def main() -> None:
    p = ContractParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", choices=["ultrachat"])
    p.add_argument("--tokenizer", required=True, help="Hugging Face id of the served model")
    p.add_argument(
        "--tokenizer-revision",
        default="main",
        help="commit sha (or branch) of the tokenizer files; a sha makes the counts reproducible",
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--split", default="test_sft", help="test_sft (default) or train_sft")
    p.add_argument("--sessions", type=int, default=2000, help="conversations to write")
    p.add_argument("--start", type=int, default=0, help="first row of the split")
    p.add_argument("--max-turns", type=int, default=4, help="user turns kept per conversation")
    p.add_argument("--max-output", type=int, default=2048, help="cap on one turn's output")
    a = p.parse_args()
    if a.sessions < 1 or a.max_turns < 1 or a.max_output < 1:
        emit(False, "error", "fix_errors", errors=[err("bad_value", "counts must be >= 1", "Fix.")])
    if a.start < 0:
        emit(False, "error", "fix_errors", errors=[err("bad_value", "start must be >= 0", "Fix.")])
    try:
        count = token_counter(a.tokenizer, a.tokenizer_revision)
    except ImportError:
        emit(
            False,
            "error",
            "fix_errors",
            errors=[
                err(
                    "tokenizer_missing",
                    "the `tokenizers` package is not installed",
                    "pip install tokenizers (AIPerf's environment already has it), then rerun.",
                )
            ],
            code=EXIT_PRECONDITION,
        )
    except Exception as exc:  # a gated or misspelled tokenizer id
        emit(
            False,
            "error",
            "fix_errors",
            errors=[
                err("tokenizer_unavailable", f"{a.tokenizer}: {exc}", "Check the id/HF_TOKEN.")
            ],
            code=EXIT_PRECONDITION,
        )
    sessions: list[dict[str, Any]] = []
    try:
        # read a little extra: conversations without a real reply are skipped
        for row in fetch_rows(a.split, a.start, a.sessions * 2):
            s = to_session(row, count, a.max_turns, a.max_output)
            if s:
                try:  # JSON allows an escaped lone surrogate, which no UTF-8 file can hold
                    json.dumps(s, ensure_ascii=False).encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise MalformedResponse(f"a row has text that is not valid Unicode: {exc.reason}")
                sessions.append(s)
            if len(sessions) == a.sessions:
                break
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        MalformedResponse,
    ) as exc:
        emit(
            False,
            "error",
            "fix_errors",
            errors=[err("download_failed", str(exc), "Check access to huggingface.co; retry.")],
            code=EXIT_EXTERNAL,
        )
    if len(sessions) < a.sessions:
        emit(
            False,
            "error",
            "fix_errors",
            {"written": 0, "found": len(sessions)},
            [
                err(
                    "too_few_rows",
                    f"only {len(sessions)} usable conversations",
                    "Lower --sessions.",
                )
            ],
            EXIT_PRECONDITION,
        )
    data = "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in sessions).encode("utf-8")
    tmp = None
    try:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        # a temporary file of our own beside --out: never a half-written dataset, and a second
        # writer to the same --out never shares it
        fd, tmp = tempfile.mkstemp(prefix=f".{a.out.name}.", suffix=".tmp", dir=a.out.parent)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, a.out)
        tmp = None
    except OSError as exc:
        if tmp:
            try:  # best effort: the write error is what the user needs to see
                Path(tmp).unlink(missing_ok=True)
            except OSError:
                pass
        emit(
            False,
            "error",
            "fix_errors",
            errors=[err("write_failed", f"{a.out}: {exc}", "Pass a writable file path to --out.")],
            code=EXIT_PRECONDITION,
        )
    outs = sorted(t["output_length"] for s in sessions for t in s["turns"])
    emit(
        True,
        "written",
        "set_campaign_dataset",
        {
            "file": str(a.out.resolve()),
            "sha256": hashlib.sha256(data).hexdigest(),  # the bytes this run wrote
            "source": f"{ULTRACHAT}:{a.split}[{a.start}:]",
            "tokenizer": f"{a.tokenizer}@{a.tokenizer_revision}",
            "sessions": len(sessions),
            "turns": len(outs),
            "output_length_p50": outs[len(outs) // 2],
            "output_length_p90": outs[int(len(outs) * 0.9)],
            # absolute: the campaign file may live in another folder than this run's
            "campaign_field": {"dataset": {"file": str(a.out.resolve()), "type": "multi_turn"}},
        },
    )


if __name__ == "__main__":
    main()
