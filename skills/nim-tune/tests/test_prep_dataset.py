# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for scripts/prep_dataset.py; no network: rows are built in the test."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prep_dataset.py"
spec = importlib.util.spec_from_file_location("prep_dataset", SCRIPT)
assert spec and spec.loader
pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd)


def words(text: str) -> int:
    return len(text.split())


def test_each_user_turn_is_sized_by_its_real_reply() -> None:
    row = {
        "prompt_id": "p1",
        "messages": [
            {"role": "user", "content": " Hi there "},
            {"role": "assistant", "content": "one two three"},
            {"role": "user", "content": "More?"},
            {"role": "assistant", "content": "a b c d e f"},
        ],
    }
    s = pd.to_session(row, words, max_turns=4, max_output=5)
    assert s == {
        "session_id": "p1",
        "turns": [
            {"text": "Hi there", "output_length": 3},
            {"text": "More?", "output_length": 5},  # capped at max_output
        ],
    }
    assert pd.to_session(row, words, max_turns=1, max_output=99)["turns"] == [
        {"text": "Hi there", "output_length": 3}
    ]


def test_a_turn_without_a_real_reply_ends_the_session() -> None:
    row = {"messages": [{"role": "user", "content": "q"}, {"role": "user", "content": "q2"}]}
    assert pd.to_session(row, words, max_turns=4, max_output=9) is None


def test_a_non_string_user_message_is_read_as_text() -> None:
    row = {
        "prompt_id": "p3",
        "messages": [{"role": "user", "content": 123}, {"role": "assistant", "content": "a b"}],
    }
    assert pd.to_session(row, words, max_turns=4, max_output=9) == {
        "session_id": "p3",
        "turns": [{"text": "123", "output_length": 2}],
    }


def test_bad_usage_is_contract_json() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "sharegpt"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    out = json.loads(proc.stdout)
    assert proc.returncode == 2 and out["errors"][0]["code"] == "bad_usage"


def test_an_unwritable_output_is_contract_json(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    rows = [
        {
            "prompt_id": "p",
            "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a b"}],
        }
    ]
    monkeypatch.setattr(pd, "token_counter", lambda tok, rev: words)
    monkeypatch.setattr(pd, "fetch_rows", lambda split, start, count: iter(rows))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prep_dataset.py",
            "ultrachat",
            "--tokenizer",
            "t",
            "--sessions",
            "1",
            "--out",
            str(tmp_path),
        ],  # a folder, not a file
    )
    try:
        pd.main()
    except SystemExit as exc:
        assert exc.code == 3
    out = json.loads(capsys.readouterr().out)
    assert out["errors"][0]["code"] == "write_failed"


ROWS = [
    {
        "prompt_id": "p1",
        "messages": [
            {"role": "user", "content": "Grüße, q"},
            {"role": "assistant", "content": "a b c"},
        ],
    },
    {
        "prompt_id": "p2",
        "messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a"}],
    },
]


def run_main(monkeypatch: Any, capsys: Any, *argv: str) -> tuple[int, dict[str, Any]]:
    monkeypatch.setattr(sys, "argv", ["prep_dataset.py", "ultrachat", "--tokenizer", "t", *argv])
    try:
        pd.main()
    except SystemExit as exc:
        return int(exc.code or 0), json.loads(capsys.readouterr().out)
    raise AssertionError("main() did not exit")


def test_a_written_dataset_hands_off_its_own_bytes(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(pd, "token_counter", lambda tok, rev: words)
    monkeypatch.setattr(pd, "fetch_rows", lambda split, start, count: iter(ROWS))
    real_replace = os.replace

    def replace_then_overwrite(src: Any, dst: Any) -> None:
        real_replace(src, dst)
        Path(dst).write_text("another writer's dataset\n")  # a second writer, right after

    monkeypatch.setattr(os, "replace", replace_then_overwrite)
    code, out = run_main(monkeypatch, capsys, "--sessions", "2", "--out", "datasets/chat.jsonl")
    assert code == 0 and out["state"] == "written", out
    want = "".join(json.dumps(pd.to_session(r, words, 4, 2048), ensure_ascii=False) + "\n"
                   for r in ROWS).encode("utf-8")  # fmt: skip
    # the hash is of the bytes this run wrote, never of whatever the path holds afterwards
    assert out["facts"]["sha256"] == hashlib.sha256(want).hexdigest()
    assert "Grüße" in want.decode("utf-8")
    field = Path(out["facts"]["campaign_field"]["dataset"]["file"])
    # a campaign in another folder still finds it
    assert field.is_absolute() and field == work / "datasets" / "chat.jsonl"
    assert [p.name for p in (work / "datasets").iterdir()] == ["chat.jsonl"]  # no temp left


def test_the_written_file_is_utf8_jsonl(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(pd, "token_counter", lambda tok, rev: words)
    monkeypatch.setattr(pd, "fetch_rows", lambda split, start, count: iter(ROWS))
    out_file = tmp_path / "d.jsonl"
    code, out = run_main(monkeypatch, capsys, "--sessions", "2", "--out", str(out_file))
    data = out_file.read_bytes()
    assert code == 0 and out["facts"]["sha256"] == hashlib.sha256(data).hexdigest()
    lines = [json.loads(ln) for ln in data.decode("utf-8").splitlines()]
    assert [s["session_id"] for s in lines] == ["p1", "p2"]
    assert lines[0]["turns"][0] == {"text": "Grüße, q", "output_length": 3}


LATIN1_RUN = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("pd", sys.argv[1])
pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd)
pd.token_counter = lambda tok, rev: (lambda text: len(text.split()))
rows = json.loads(sys.argv[2])
pd.fetch_rows = lambda split, start, count: iter(rows)
sys.argv = ["prep_dataset.py", "ultrachat", "--tokenizer", "t", "--sessions", "2",
            "--out", sys.argv[3]]
pd.main()
"""


def test_the_file_is_utf8_even_when_the_locale_is_not(tmp_path: Path) -> None:
    # a non-UTF-8 default encoding (Latin-1 here, ASCII where that locale is missing) must not
    # change the bytes written: the file is UTF-8 whatever the locale
    out_file = tmp_path / "d.jsonl"
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0",
           "LC_ALL": "en_US.ISO8859-1"}  # fmt: skip
    proc = subprocess.run(
        # -S: no site startup files, so nothing from the developer's home is loaded
        [sys.executable, "-S", "-c", LATIN1_RUN, str(SCRIPT), json.dumps(ROWS), str(out_file)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=False,
        env=env,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-500:]
    first = json.loads(out_file.read_bytes().decode("utf-8").splitlines()[0])
    assert first["turns"][0]["text"] == "Grüße, q"


def test_a_negative_start_is_bad_input_before_any_external_work(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    def no_tokenizer(tok: str, rev: str) -> Any:
        raise AssertionError("the tokenizer was loaded")

    monkeypatch.setattr(pd, "token_counter", no_tokenizer)
    code, out = run_main(monkeypatch, capsys, "--start", "-1", "--out", str(tmp_path / "d.jsonl"))
    assert code == 2 and out["errors"][0]["code"] == "bad_value"
    code, out = run_main(monkeypatch, capsys, "--start", "0", "--sessions", "0",
                         "--out", str(tmp_path / "d.jsonl"))  # fmt: skip
    assert code == 2 and "counts" in out["errors"][0]["message"]  # 0 is a valid start


class FakeResponse(io.BytesIO):
    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def test_a_malformed_viewer_response_is_download_failed(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setattr(pd, "token_counter", lambda tok, rev: words)
    for body in (
        {"rows": [{"row_idx": 0}]},  # a wrapper without "row"
        {"rows": [{"row": {"messages": "text"}}]},
        {"rows": "none"},
        {},  # no "rows" at all
        {"rows": [{"row": {"messages": {}}}]},  # false-valued, but not a list
        {"rows": [{"row": {"messages": ""}}]},
        {"rows": [{"row": {"messages": 0}}]},
        {"rows": [{"row": {"messages": False}}]},
        ["not", "an", "object"],
    ):
        payload = json.dumps(body).encode()
        monkeypatch.setattr(pd, "open_https",
                            lambda url, timeout, p=payload: FakeResponse(p))  # fmt: skip
        code, out = run_main(monkeypatch, capsys, "--sessions", "1",
                             "--out", str(tmp_path / "d.jsonl"))  # fmt: skip
        assert code == 4 and out["errors"][0]["code"] == "download_failed", body
    # a valid but empty answer stays too_few_rows
    monkeypatch.setattr(pd, "open_https",
                        lambda url, timeout: FakeResponse(b'{"rows": []}'))  # fmt: skip
    code, out = run_main(monkeypatch, capsys, "--sessions", "1", "--out", str(tmp_path / "d.jsonl"))
    assert code == 3 and out["errors"][0]["code"] == "too_few_rows"


def test_a_failed_write_whose_cleanup_also_fails_is_write_failed(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setattr(pd, "token_counter", lambda tok, rev: words)
    row = {"row": {"prompt_id": "p", "messages": [{"role": "user", "content": "q"},
                                                   {"role": "assistant", "content": "a"}]}}
    payload = json.dumps({"rows": [row]}).encode()
    monkeypatch.setattr(pd, "open_https", lambda url, timeout: FakeResponse(payload))

    def replace_denied(*a: Any, **k: Any) -> None:
        raise PermissionError("replace denied")

    cleanups: list[Any] = []

    def unlink_denied(self: Any, *a: Any, **k: Any) -> None:
        cleanups.append(self)
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(pd.os, "replace", replace_denied)
    monkeypatch.setattr(pd.Path, "unlink", unlink_denied)
    code, out = run_main(monkeypatch, capsys, "--sessions", "1", "--out", str(tmp_path / "d.jsonl"))
    assert code == 3 and out["errors"][0]["code"] == "write_failed"
    assert len(cleanups) == 1  # cleanup was tried, and its own error did not replace the cause
    assert "replace denied" in out["errors"][0]["message"]
    assert "cleanup denied" not in out["errors"][0]["message"]


def test_a_response_that_is_not_utf8_is_download_failed(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setattr(pd, "token_counter", lambda tok, rev: words)
    monkeypatch.setattr(pd.time, "sleep", lambda s: None)
    monkeypatch.setattr(pd, "open_https",
                        lambda url, timeout: FakeResponse(b'{"rows": "\xff\xfe"}'))  # fmt: skip
    code, out = run_main(monkeypatch, capsys, "--sessions", "1", "--out", str(tmp_path / "d.jsonl"))
    assert code == 4 and out["errors"][0]["code"] == "download_failed"


def test_a_lone_surrogate_in_a_row_is_download_failed(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setattr(pd, "token_counter", lambda tok, rev: words)
    msgs = '[{"role": "user", "content": "q \\ud800"}, {"role": "assistant", "content": "a"}]'
    payload = ('{"rows": [{"row": {"prompt_id": "p", "messages": ' + msgs + "}}]}").encode()
    monkeypatch.setattr(pd, "open_https", lambda url, timeout: FakeResponse(payload))
    code, out = run_main(monkeypatch, capsys, "--sessions", "1", "--out", str(tmp_path / "d.jsonl"))
    assert code == 4 and out["errors"][0]["code"] == "download_failed"
    assert not (tmp_path / "d.jsonl").exists()


def test_a_lone_surrogate_in_a_reply_never_reaches_the_tokenizer(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    def strict(tok: str, rev: str) -> Any:
        return lambda text: len(text.encode("utf-8").split())  # a tokenizer that rejects it

    monkeypatch.setattr(pd, "token_counter", strict)
    msgs = '[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a \\udc00"}]'
    payload = ('{"rows": [{"row": {"prompt_id": "p", "messages": ' + msgs + "}}]}").encode()
    monkeypatch.setattr(pd, "open_https", lambda url, timeout: FakeResponse(payload))
    code, out = run_main(monkeypatch, capsys, "--sessions", "1", "--out", str(tmp_path / "d.jsonl"))
    assert code == 4 and out["errors"][0]["code"] == "download_failed"


def test_the_tokenizer_revision_reaches_the_loader_and_the_facts(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    seen = []

    def counter(tok: str, rev: str) -> Any:
        seen.append((tok, rev))
        return lambda text: len(text.split())

    monkeypatch.setattr(pd, "token_counter", counter)
    monkeypatch.setattr(pd, "fetch_rows", lambda split, start, count: iter(ROWS))
    sha = "0123456789abcdef0123456789abcdef01234567"
    out_file = str(tmp_path / "d.jsonl")
    code, out = run_main(
        monkeypatch, capsys, "--sessions", "2", "--tokenizer-revision", sha, "--out", out_file
    )
    assert code == 0 and seen == [("t", sha)]
    assert out["facts"]["tokenizer"] == f"t@{sha}"


def test_open_https_refuses_any_other_scheme() -> None:
    for url in ("file:///tmp/rows.json", "http://example.com/", "ftp://example.com/x"):
        try:
            pd.open_https(url, timeout=1)
        except pd.urllib.error.URLError:
            continue
        raise AssertionError(f"{url} was opened")


def test_the_https_opener_has_no_handler_for_another_scheme(monkeypatch: Any) -> None:
    import urllib.request

    # a configured http proxy adds a proxy-only http_open to ProxyHandler: keep it out
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {})
    seen = {}

    class Recorder(urllib.request.OpenerDirector):
        def open(self, url: Any, data: Any = None, timeout: Any = None) -> Any:
            seen["schemes"] = {
                name[: -len("_open")]
                for h in self.handlers
                for name in dir(h)
                if name.endswith("_open") and not name.startswith(("default", "unknown"))
            }
            return None

    real = urllib.request.OpenerDirector
    urllib.request.OpenerDirector = Recorder  # type: ignore[misc]
    try:
        pd.open_https("https://example.com/", timeout=1)
    finally:
        urllib.request.OpenerDirector = real  # type: ignore[misc]
    assert "https" in seen["schemes"]
    assert not seen["schemes"] & {"file", "http", "ftp", "data"}


def test_a_redirect_to_plain_http_is_refused_not_followed() -> None:
    import http.client
    import io
    import urllib.request

    class Redirect(urllib.request.HTTPSHandler):
        def https_open(self, req: Any) -> Any:
            msg = http.client.HTTPMessage()
            msg["Location"] = "http://example.com/plain"
            resp = urllib.request.addinfourl(io.BytesIO(b""), msg, req.full_url, 302)
            resp.msg = "Found"
            return resp

    real = urllib.request.HTTPSHandler
    urllib.request.HTTPSHandler = Redirect  # type: ignore[misc]
    try:
        pd.open_https("https://example.com/", timeout=1)
    except pd.urllib.error.URLError as exc:
        assert "unknown url type" in str(exc.reason)
    else:
        raise AssertionError("the redirect to http was followed")
    finally:
        urllib.request.HTTPSHandler = real  # type: ignore[misc]
