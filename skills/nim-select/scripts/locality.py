#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The location hint of `nimselect.py locate`: `import locality`.

It helps choose where to get the NIM image from: NGC, or a China NIM distributor (CND). It reads
the local timezone (TZ, /etc/timezone, or the /etc/localtime link) and the language (LC_ALL, then
LANG), and checks whether https://www.google.com/generate_204 answers 204 within 3 seconds. The
result is `china` (Google not reachable, and a mainland China timezone or a zh_CN language),
`outside_china` (Google reachable and no China timezone) or `unknown`. It reads the machine the
agent runs on, which may not be where the GPUs are, and sends nothing about the user anywhere: the
check is one plain GET of PROBE_URL. TZ, LC_ALL and LANG are the only environment variables it
reads, as what it measures; none changes the URL.

It also holds the country lookup of `nimselect.py country`, which the agent runs only after the
user agrees: one GET of COUNTRY_URL, Cloudflare's trace page, which names the country of the
public IP address the request comes from (`loc=DE`). Only that code is kept; the page also shows
the address, which is never kept or printed. Stdlib only. Run directly, it prints a notice: it has
no command of its own.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import listing
import ns_common as ns
import registry_labels

PROBE_URL = "https://www.google.com/generate_204"
PROBE_BUDGET = 3.0
# The country lookup (`nimselect.py country`), run only after the user agrees.
COUNTRY_URL = "https://www.cloudflare.com/cdn-cgi/trace"
COUNTRY_BUDGET = 5.0
COUNTRY_MAX_ANSWER = 4096  # bytes; the trace page is about 300
# Mainland China zones, with the tzdata aliases of two of them (Asia/Chungking, Asia/Kashgar).
CHINA_ZONES = frozenset(
    {
        "Asia/Shanghai",
        "Asia/Chongqing",
        "Asia/Chungking",
        "Asia/Harbin",
        "Asia/Urumqi",
        "Asia/Kashgar",
        "PRC",
    }
)


def zone_name(value: str | None) -> str | None:
    """`:Asia/Shanghai`, `/usr/share/zoneinfo/Asia/Shanghai`, `posix/Asia/Shanghai` ->
    `Asia/Shanghai`. A path to a link (TZ=:/etc/localtime, glibc's form) is followed to the zone
    file it names. None if empty, or a path that names no zone (a copied zone file)."""
    text = (value or "").strip().lstrip(":")
    if not text:
        return None
    if text.startswith("/") and "zoneinfo/" not in text:
        text = os.path.realpath(text)
        if "zoneinfo/" not in text:
            return None
    if "zoneinfo/" in text:
        text = text.split("zoneinfo/", 1)[1]
    for prefix in ("posix/", "right/"):  # the same zones, with and without leap seconds
        text = text.removeprefix(prefix)
    return text or None


def local_timezone(environ: Mapping[str, str], etc: Path = Path("/etc")) -> str | None:
    """The timezone name from TZ, else /etc/timezone, else the /etc/localtime link target."""
    zone = zone_name(environ.get("TZ"))
    if zone:
        return zone
    try:
        lines = (etc / "timezone").read_text(encoding="utf-8").splitlines()
        zone = zone_name(lines[0] if lines else None)
    except (OSError, UnicodeDecodeError):
        zone = None
    if zone:
        return zone
    try:
        link = os.readlink(etc / "localtime")
    except OSError:
        return None
    # a link that names no zoneinfo/ path (one to another link) is followed to the zone file
    return zone_name(link if "zoneinfo/" in link else str(etc / "localtime"))


def local_lang(environ: Mapping[str, str]) -> str | None:
    """LC_ALL, else LANG (LC_ALL overrides LANG for every category)."""
    for key in ("LC_ALL", "LANG"):
        value = (environ.get(key) or "").strip()
        if value:
            return value
    return None


def is_china_zone(zone: str | None) -> bool:
    return zone in CHINA_ZONES


def is_china_lang(lang: str | None) -> bool:
    return (lang or "").lower().replace("-", "_").startswith("zh_cn")


def google_reachable(url: str | None = None, budget: float = PROBE_BUDGET) -> bool:
    """True when `url` (default PROBE_URL) answers HTTP 204 within `budget` seconds (a proxy
    page, a block page, a redirect or a timeout is False). One anonymous GET, never redirected
    (generate_204 sends no redirect: one is a proxy or a portal answering); it sends no data
    about the user."""
    target = url or PROBE_URL

    def probe() -> bool:
        request = urllib.request.Request(target, headers={"User-Agent": "nim-select/0.1"})
        try:
            with ns.open_url(request, budget) as response:
                return int(response.status) == 204
        except (urllib.error.URLError, OSError, ValueError):
            return False

    ns.log(f"checking whether {target} answers within {budget:g} s")
    return registry_labels.within(budget, probe, lambda _why: False)


def trace_country(text: str) -> str | None:
    """The `loc=` value of a trace page, as an upper-case ISO 3166 code, or None when there is
    none or it is not a country code (listing.ISO_COUNTRIES: the assigned codes, and XK for
    Kosovo): Cloudflare's XX (not known) and T1 (a Tor exit) name no country. Every other line,
    the `ip=` line included, is skipped."""
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "loc":
            code = value.strip().upper()
            if re.fullmatch(r"[A-Z]{2}", code) and code.lower() in listing.ISO_COUNTRIES:
                return code
            return None
    return None


def lookup_country(
    url: str | None = None, budget: float = COUNTRY_BUDGET
) -> tuple[str | None, str | None]:
    """(country, problem): the ISO 3166 code of the country of this machine's public IP address,
    as the trace page at `url` (default COUNTRY_URL) names it, or (None, why) when it cannot be
    read within `budget` seconds or names no country. One anonymous GET, never redirected; it
    sends nothing but the request itself, and keeps nothing of the answer but the code."""
    target = url or COUNTRY_URL

    def read() -> tuple[str | None, str | None]:
        request = urllib.request.Request(target, headers={"User-Agent": "nim-select/0.1"})
        try:
            with ns.open_url(request, budget) as response:
                status = int(response.status)
                body = response.read(COUNTRY_MAX_ANSWER + 1)
        except urllib.error.HTTPError as exc:  # a redirect too: open_url follows none
            return None, f"{target} answered HTTP {exc.code}"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            reason = getattr(exc, "reason", None) or exc
            return None, f"{target} did not answer ({type(exc).__name__}: {reason})"
        if status != 200:
            return None, f"{target} answered HTTP {status}"
        if len(body) > COUNTRY_MAX_ANSWER:
            return None, f"{target} answered with more than {COUNTRY_MAX_ANSWER} bytes"
        code = trace_country(body.decode("utf-8", "replace"))
        if code is None:
            return None, f"{target} named no country for this address"
        return code, None

    ns.log(f"looking up the country of this machine's public IP address at {target}")
    return registry_labels.within(budget, read, lambda why: (None, f"{target} {why}"))


def result(zone: str | None, lang: str | None, reachable: bool) -> str:
    """`china`, `outside_china` or `unknown` (see the module docstring)."""
    if not reachable and (is_china_zone(zone) or is_china_lang(lang)):
        return "china"
    if reachable and not is_china_zone(zone):
        return "outside_china"
    return "unknown"


def locate(environ: Mapping[str, str] | None = None, etc: Path = Path("/etc")) -> dict[str, Any]:
    """facts.locality: {timezone, lang, google_reachable, result}."""
    env = os.environ if environ is None else environ
    zone, lang = local_timezone(env, etc), local_lang(env)
    reachable = google_reachable()
    return {
        "timezone": zone,
        "lang": lang,
        "google_reachable": reachable,
        "result": result(zone, lang, reachable),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Location hint for the nim-select scripts; import it, or run "
        "nimselect.py locate.",
        epilog='Example: python3 -I "<SKILL_DIR>/scripts/nimselect.py" locate',
    )
    parser.parse_args(argv)
    ns.emit(True, "library", "done", {"note": "import this module; run nimselect.py locate"})


if __name__ == "__main__":
    main()
