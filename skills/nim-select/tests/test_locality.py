# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/locality.py, `nimselect.py locate` and `nimselect.py country`: the timezone,
the language and the Google check that hint whether this machine is in China, and the country
lookup the agent runs only after the user agrees. Local HTTP servers stand in for
https://www.google.com/generate_204 and https://www.cloudflare.com/cdn-cgi/trace (TEST_LOCATE_URL
and TEST_COUNTRY_URL of ns_testkit.run_cli); otherwise conftest.py and the runner of run_cli point
both at a closed port."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import locality
import pytest
from ns_testkit import SCRIPTS, run_cli, serve, use_built_in_endpoints


def etc(tmp_path: Path, timezone: str | None = None, link: str | None = None) -> Path:
    folder = tmp_path / "etc"
    folder.mkdir(exist_ok=True)
    if timezone is not None:
        (folder / "timezone").write_text(timezone + "\n", encoding="utf-8")
    if link is not None:
        (folder / "localtime").symlink_to(link)
    return folder


# --- the timezone and the language ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "zone"),
    [
        ("Asia/Shanghai", "Asia/Shanghai"),
        (":Asia/Shanghai", "Asia/Shanghai"),
        ("/usr/share/zoneinfo/Asia/Urumqi", "Asia/Urumqi"),
        ("PRC", "PRC"),
        # the same zones with and without leap seconds
        ("posix/Asia/Shanghai", "Asia/Shanghai"),
        ("/usr/share/zoneinfo/right/Asia/Shanghai", "Asia/Shanghai"),
        ("", None),
        (None, None),
    ],
)
def test_zone_name(value: str | None, zone: str | None) -> None:
    assert locality.zone_name(value) == zone


def test_the_timezone_comes_from_tz_then_etc_timezone_then_the_localtime_link(
    tmp_path: Path,
) -> None:
    folder = etc(tmp_path, "Asia/Harbin", "/usr/share/zoneinfo/Asia/Chongqing")
    assert locality.local_timezone({"TZ": "Europe/Berlin"}, folder) == "Europe/Berlin"
    assert locality.local_timezone({}, folder) == "Asia/Harbin"
    (folder / "timezone").unlink()
    assert locality.local_timezone({}, folder) == "Asia/Chongqing"
    (folder / "localtime").unlink()
    assert locality.local_timezone({}, folder) is None


def test_tz_naming_the_localtime_link_follows_it(tmp_path: Path) -> None:
    # glibc's TZ=:/etc/localtime: the zone is the file the link names, never "/etc/localtime"
    folder = etc(tmp_path, link="/usr/share/zoneinfo/Asia/Shanghai")
    link = folder / "localtime"
    for value in (f":{link}", str(link)):
        zone = locality.local_timezone({"TZ": value}, folder)
        assert zone == "Asia/Shanghai"
        # a China zone with Google reachable (a VPN) is unknown: the agent asks
        assert locality.result(zone, None, True) == "unknown"
    # a link to a link, then a posix/ zone
    (tmp_path / "zone").symlink_to("/usr/share/zoneinfo/posix/Asia/Urumqi")
    link.unlink()
    link.symlink_to(tmp_path / "zone")
    assert locality.local_timezone({"TZ": f":{link}"}, folder) == "Asia/Urumqi"
    assert locality.local_timezone({}, folder) == "Asia/Urumqi"
    # a zone file that is no link names no zone: the next source is read
    link.unlink()
    link.write_bytes(b"TZif")
    (folder / "timezone").write_text("Asia/Harbin\n", encoding="utf-8")
    assert locality.local_timezone({"TZ": f":{link}"}, folder) == "Asia/Harbin"


def test_the_language_is_lc_all_then_lang() -> None:
    assert locality.local_lang({"LC_ALL": "zh_CN.UTF-8", "LANG": "en_US.UTF-8"}) == "zh_CN.UTF-8"
    assert locality.local_lang({"LC_ALL": "", "LANG": "en_US.UTF-8"}) == "en_US.UTF-8"
    assert locality.local_lang({}) is None


@pytest.mark.parametrize(
    ("zone", "china"),
    [
        ("Asia/Shanghai", True),
        ("Asia/Chongqing", True),
        ("Asia/Harbin", True),
        ("Asia/Urumqi", True),
        ("PRC", True),
        ("Asia/Hong_Kong", False),
        ("Asia/Taipei", False),
        ("Asia/Kolkata", False),
        (None, False),
    ],
)
def test_mainland_china_zones(zone: str | None, china: bool) -> None:
    assert locality.is_china_zone(zone) is china


@pytest.mark.parametrize(
    ("lang", "china"),
    [
        ("zh_CN.UTF-8", True),
        ("zh_CN", True),
        ("zh-CN", True),
        ("zh_TW.UTF-8", False),
        (None, False),
    ],
)
def test_a_zh_cn_language(lang: str | None, china: bool) -> None:
    assert locality.is_china_lang(lang) is china


@pytest.mark.parametrize(
    ("zone", "lang", "reachable", "result"),
    [
        ("Asia/Shanghai", "en_US.UTF-8", False, "china"),
        ("Europe/Berlin", "zh_CN.UTF-8", False, "china"),
        ("Asia/Kolkata", "en_US.UTF-8", True, "outside_china"),
        (None, None, True, "outside_china"),
        ("Asia/Shanghai", "zh_CN.UTF-8", True, "unknown"),  # a VPN, or a China clock abroad
        ("Europe/Berlin", "en_US.UTF-8", False, "unknown"),  # offline or a proxy
        (None, None, False, "unknown"),
    ],
)
def test_the_locality_result(
    zone: str | None, lang: str | None, reachable: bool, result: str
) -> None:
    assert locality.result(zone, lang, reachable) == result


# --- the Google check ---------------------------------------------------------------------------


def test_only_a_204_counts_as_google_reachable() -> None:
    with serve({"/generate_204": (204, ""), "/portal": (200, "<html>login</html>")}) as server:
        root = server.base.removesuffix("/v2")
        assert locality.google_reachable(f"{root}/generate_204") is True
        # a captive portal or a block page answers, but not 204
        assert locality.google_reachable(f"{root}/portal") is False
        assert locality.google_reachable(f"{root}/missing") is False
        assert server.requests == ["/generate_204", "/portal", "/missing"]


def test_a_redirect_is_not_followed_and_not_reachable() -> None:
    # generate_204 never redirects: a redirect is a proxy or a portal, even to a page that is 204
    requests: list[str] = []

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            self.send_response(302 if self.path == "/generate_204" else 204)
            self.send_header("Location", "/real_204")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/generate_204"
        assert locality.google_reachable(url) is False
    finally:
        server.shutdown()
        server.server_close()
    assert requests == ["/generate_204"]


def test_google_not_answering_in_time_is_not_reachable() -> None:
    import socket

    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(8)
        start = time.monotonic()
        url = f"http://127.0.0.1:{silent.getsockname()[1]}/generate_204"
        assert locality.google_reachable(url, budget=0.5) is False
        assert time.monotonic() - start < 2.0
    with socket.socket() as closed:  # bound, not listening: a connection to it is refused
        closed.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{closed.getsockname()[1]}/generate_204"
        assert locality.google_reachable(url) is False


# --- the country lookup -------------------------------------------------------------------------

# What Cloudflare's trace page answers (the address is a documentation one, RFC 5737).
TRACE = "fl=1f1\nh=www.cloudflare.com\nip=203.0.113.9\nts=1.0\nloc=DE\ncolo=FRA\nwarp=off\n"


@pytest.mark.parametrize(
    ("text", "code"),
    [
        (TRACE, "DE"),
        ("loc=in\n", "IN"),
        (" loc = US \n", "US"),
        ("loc=XK\n", "XK"),  # Kosovo: a user-assigned code in common use, not in ISO 3166-1
        # Cloudflare's values that name no country: not known, a Tor exit
        ("loc=XX\n", None),
        ("loc=T1\n", None),
        ("loc=ZZ\n", None),  # two letters, but no country has the code
        ("loc=DEU\n", None),
        ("loc=\n", None),
        ("ip=203.0.113.9\n", None),
        ("<html>blocked</html>", None),
        ("", None),
    ],
)
def test_trace_country(text: str, code: str | None) -> None:
    assert locality.trace_country(text) == code


def test_the_country_lookup_keeps_only_the_code() -> None:
    with serve({"/cdn-cgi/trace": (200, TRACE)}) as server:
        url = server.base.removesuffix("/v2") + "/cdn-cgi/trace"
        assert locality.lookup_country(url) == ("DE", None)
    assert server.requests == ["/cdn-cgi/trace"]
    [sent] = server.headers
    headers = {key.lower(): value for key, value in sent.items()}
    # an anonymous GET: nothing about the user rides along
    assert headers["user-agent"] == "nim-select/0.1"
    assert not {"authorization", "cookie"} & set(headers)


@pytest.mark.parametrize(
    ("answer", "problem"),
    [
        ((403, "forbidden"), "answered HTTP 403"),
        ((200, "loc=XX\n"), "named no country"),
        ((200, "loc=ZZ\n"), "named no country"),
        ((200, "<html>sign in</html>"), "named no country"),
        ((200, "loc=DE\n" + "x" * 5000), "more than 4096 bytes"),
    ],
)
def test_a_country_lookup_that_names_no_country_fails(
    answer: tuple[int, str], problem: str
) -> None:
    with serve({"/cdn-cgi/trace": answer}) as server:
        url = server.base.removesuffix("/v2") + "/cdn-cgi/trace"
        code, why = locality.lookup_country(url)
    assert code is None and why is not None and problem in why, why


def test_the_country_lookup_follows_no_redirect() -> None:
    with serve({"/cdn-cgi/trace": (200, TRACE)}) as elsewhere:
        target = elsewhere.base.removesuffix("/v2") + "/cdn-cgi/trace"
        with serve({"*": (302, target)}) as first:
            url = first.base.removesuffix("/v2") + "/cdn-cgi/trace"
            code, why = locality.lookup_country(url)
    assert code is None and "HTTP 302" in str(why)
    assert elsewhere.requests == []


def test_the_country_lookup_stops_at_its_budget() -> None:
    import socket

    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(8)
        start = time.monotonic()
        url = f"http://127.0.0.1:{silent.getsockname()[1]}/cdn-cgi/trace"
        code, why = locality.lookup_country(url, budget=0.5)
        assert time.monotonic() - start < 2.0
    assert code is None and "within 0.5 s" in str(why)
    with socket.socket() as closed:  # bound, not listening: a connection to it is refused
        closed.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{closed.getsockname()[1]}/cdn-cgi/trace"
        code, why = locality.lookup_country(url)
    assert code is None and "did not answer" in str(why)


def test_country_command_gives_the_code_for_region() -> None:
    with serve({"/cdn-cgi/trace": (200, TRACE)}) as server:
        url = server.base.removesuffix("/v2") + "/cdn-cgi/trace"
        code, payload = run_cli("nimselect.py", "country", env={"TEST_COUNTRY_URL": url})
    assert code == 0, payload
    assert (payload["state"], payload["next"]) == ("country_found", "rerun_with_region")
    assert payload["facts"] == {
        "country": "DE",
        "checked": "the public IP address of the machine the agent runs on",
        "lookup_host": "127.0.0.1",
    }
    # the address the trace page shows is never printed
    assert "203.0.113.9" not in json.dumps(payload)


def test_country_command_failing_asks_the_user() -> None:
    # the runner of run_cli points the lookup at a closed port
    code, payload = run_cli("nimselect.py", "country")
    assert code == 4, payload
    assert (payload["state"], payload["next"]) == ("country_unknown", "ask_user")
    assert payload["facts"]["country"] is None
    [error] = payload["errors"]
    assert error["code"] == "country_lookup_failed"
    assert "ask which country the NIM will run in" in error["fix"]


def test_country_help_says_it_runs_only_after_the_user_agrees() -> None:
    text = help_text(["country"])
    assert "run it only after the user agrees to the lookup" in text
    assert "otherwise ask which country the nim will run in" in text
    assert "cloudflare" in text and "only the country code is kept" in text
    assert "not used to choose ngc or a china nim distributor" in text


# --- nimselect.py locate ------------------------------------------------------------------------


def locate(env: dict[str, str]) -> tuple[int, dict]:
    clean = {key: "" for key in ("TZ", "LANG", "LC_ALL")}
    return run_cli("nimselect.py", "locate", env={**clean, **env})


def test_locate_outside_china_suggests_ngc() -> None:
    with serve({"/generate_204": (204, "")}) as server:
        url = server.base.removesuffix("/v2") + "/generate_204"
        code, payload = locate(
            {"TZ": "Europe/Berlin", "LANG": "de_DE.UTF-8", "TEST_LOCATE_URL": url}
        )
    assert code == 0, payload
    assert (payload["state"], payload["next"]) == ("located", "run_resolve")
    assert payload["facts"] == {
        "locality": {
            "timezone": "Europe/Berlin",
            "lang": "de_DE.UTF-8",
            "google_reachable": True,
            "result": "outside_china",
        },
        "source": "ngc",
    }


def test_locate_in_china_asks_which_distributor() -> None:
    # the runner of run_cli points the check at a closed port: Google is not reachable
    code, payload = locate({"TZ": "Asia/Shanghai", "LANG": "zh_CN.UTF-8"})
    assert code == 0, payload
    assert payload["next"] == "ask_user"
    assert payload["facts"]["locality"]["result"] == "china"
    assert payload["facts"]["source"] is None


def test_locate_unknown_asks_for_the_source(tmp_path: Path) -> None:
    out = tmp_path / "locality.json"
    code, payload = run_cli(
        "nimselect.py",
        "locate",
        "--out",
        str(out),
        env={"TZ": "Europe/Berlin", "LC_ALL": "", "LANG": ""},
    )
    assert code == 0, payload
    assert (payload["next"], payload["facts"]["locality"]["result"]) == ("ask_user", "unknown")
    assert out.exists()


def help_text(command: list[str]) -> str:
    """The --help of nimselect.py (or a command of it), with whitespace and case folded."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "nimselect.py"), *command, "--help"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return " ".join(proc.stdout.split()).lower()


def locate_help_texts() -> list[str]:
    """The help of locate, and its entry in the help of nimselect.py (up to the next command:
    the country lookup, which does ask first)."""
    top = help_text([])
    entry = re.search(r"\blocate (hint whether .*?)\bcountry look up", top)
    assert entry is not None, top
    return [help_text(["locate"]), entry.group(1)]


def test_locate_help_says_no_question_comes_first() -> None:
    """Detect the location first, then confirm the source with the user: no help text asks for
    the user's agreement before locate."""
    texts = locate_help_texts()
    assert "with no question first" in texts[0]
    for text in texts:
        for asks_first in ("agree", "may i", "permission", "consent", "only after"):
            assert asks_first not in text, asks_first


def test_locate_help_maps_each_result_to_what_the_agent_does() -> None:
    text = help_text(["locate"])
    assert "outside_china: use ngc and tell the user so" in text
    # "outside_china: ..." must not stand in for the china mapping
    assert re.search(r"(?<![a-z_])china: ask whether they use tgc, turingcm or lichan", text)
    assert "unknown: ask ngc, tgc, turingcm or lichan" in text
    assert "only on the user's answer" in text


def test_locality_run_directly_prints_a_notice() -> None:
    code, payload = run_cli("locality.py")
    assert (code, payload["state"]) == (0, "library")


# --- live (opt-in) ------------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("NIM_SELECT_LIVE") != "1", reason="live test; set NIM_SELECT_LIVE=1"
)
def test_live_google_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real check answers either way within its budget."""
    use_built_in_endpoints(monkeypatch)
    start = time.monotonic()
    assert isinstance(locality.google_reachable(), bool)
    assert time.monotonic() - start < locality.PROBE_BUDGET + 1.0


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("NIM_SELECT_LIVE") != "1", reason="live test; set NIM_SELECT_LIVE=1"
)
def test_live_country_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real trace page names a country code within the budget."""
    use_built_in_endpoints(monkeypatch)
    start = time.monotonic()
    code, why = locality.lookup_country()
    assert time.monotonic() - start < locality.COUNTRY_BUDGET + 1.0
    assert code is not None and re.fullmatch(r"[A-Z]{2}", code), why
