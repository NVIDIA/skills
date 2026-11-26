# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/ns_common.py: the output contract and the small parsers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import ns_common as ns
import pytest
from ns_testkit import run_cli


def test_running_it_prints_a_library_notice() -> None:
    code, payload = run_cli("ns_common.py")
    assert code == 0
    assert payload["state"] == "library"


def test_parse_version_reads_leading_dotted_integers() -> None:
    assert ns.parse_version("580.126.09") == (580, 126, 9)
    assert ns.parse_version("535") == (535,)
    assert ns.parse_version("2.1.4-rc1") == (2, 1, 4)
    assert ns.parse_version("latest") is None
    assert ns.parse_version(None) is None


def test_as_list_accepts_absent_string_and_list() -> None:
    assert ns.as_list(None) == []
    assert ns.as_list("2330:10de") == ["2330:10de"]
    assert ns.as_list(["2330:10de", "", 7, " 2335:10de "]) == ["2330:10de", "2335:10de"]


def test_load_json_file_reports_the_file(tmp_path: Path) -> None:
    good = tmp_path / "env.json"
    ns.write_json_file(good, {"a": 1})
    assert ns.load_json_file(good) == {"a": 1}
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(ns.ScriptError) as caught:
        ns.load_json_file(bad, code="env_file_invalid")
    assert caught.value.code == "env_file_invalid"
    assert str(bad) in caught.value.message
    with pytest.raises(ns.ScriptError):
        ns.load_json_file(tmp_path / "missing.json")


def test_a_file_that_is_not_utf8_is_bad_input_never_a_traceback(tmp_path: Path) -> None:
    """Windows PowerShell's > writes UTF-16 (with a byte order mark): bad input (exit 2) that names
    the file and says how to save it. A UTF-8 byte order mark (Set-Content -Encoding utf8 in
    Windows PowerShell) is read."""
    utf16 = tmp_path / "env-utf16.json"
    utf16.write_text('{"a": 1}', encoding="utf-16")
    with pytest.raises(ns.ScriptError) as caught:
        ns.load_json_file(utf16, code="env_file_invalid")
    error = caught.value
    assert (error.code, error.exit_code) == ("env_file_invalid", ns.EXIT_INPUT)
    assert str(utf16) in error.message and "not UTF-8" in error.message
    assert "UTF-8" in error.fix and "PowerShell" in error.fix
    latin = tmp_path / "latin-1.json"
    latin.write_bytes('{"name": "Café"}'.encode("latin-1"))
    with pytest.raises(ns.ScriptError) as caught:
        ns.load_json_file(latin)
    assert "not UTF-8" in caught.value.message
    bom = tmp_path / "env-bom.json"
    bom.write_text('{"a": 1}', encoding="utf-8-sig")
    assert ns.load_json_file(bom) == {"a": 1}
    # other JSON the parser refuses: nesting too deep, an integer too long (Python's limit on the
    # digits of an int, where it has one)
    refused = [("deep.json", "[" * 100_000)]
    limit = sys.get_int_max_str_digits() if hasattr(sys, "get_int_max_str_digits") else 0
    if limit:
        refused.append(("long.json", "1" * (limit + 1)))
    for name, text in refused:
        odd = tmp_path / name
        odd.write_text(text, encoding="utf-8")
        with pytest.raises(ns.ScriptError) as caught:
            ns.load_json_file(odd)
        assert caught.value.exit_code == ns.EXIT_INPUT and str(odd) in caught.value.message


def test_emit_prints_one_contract_object(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        ns.emit(False, "checked", "fix_errors", {"x": 1}, [ns.error("bad_input", "m", "f")], 2)
    assert caught.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": False,
        "state": "checked",
        "next": "fix_errors",
        "facts": {"x": 1},
        "errors": [{"code": "bad_input", "message": "m", "fix": "f"}],
    }
