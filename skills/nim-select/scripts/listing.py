#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The list of models compatible with the user's GPUs and region (`nimselect.py list`):
`import listing`.

nimselect.py runs the list: it reads the catalog of the source (NGC, or one China NIM distributor)
and checks the images select checks for every model (on NGC the newest; when it does not fit, the
hardware variants and older versions; and every image of the repository select checks when the
tests they record for the same profile ids could change the row, as select counts them) with
select's own tag, fit and ranking code. This module holds the parts that send no request:
- which catalog entries are models to list: the skill's own name lists (NON_LLM, MODEL_FREE,
  NOT_A_NIM, written from the NGC, TGC, TuringCM and Lichan catalogs of 2026-11-11), the catalog's
  or the labels' category, and production branches; after the metadata read, the profiles'
  inference_backend confirms an LLM; and which names a --query matches;
- hardware variants (`-dgx-spark`, `-rtx`) folded under their base model;
- the card's "Deployment Geography" and whether it excludes the user's region;
- the cache of NGC answers (NgcCache), the pool that checks models with a deadline (run_tasks),
  and the rows, their order, counts and the brief form the list prints.
Stdlib only. Run directly, it prints a notice: `nimselect.py list` is the command.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import choose
import ngc_api
import ns_common as ns

LIST_SCHEMA = "nim-select/list/v1"
DEADLINE = 120.0  # seconds for the whole list: a hard stop, rows read by then are kept
REQUEST_TIMEOUT = 15.0  # seconds for one NGC request in the list
ATTEMPTS = 2  # attempts of one NGC request in the list
WORKERS = 8  # models checked at a time (at most 8 NGC requests: ngc_api.MAX_REQUESTS)
CACHE_NAME = "ngc-cache.json"
VERDICTS = ("fits", "unknown", "does_not_fit", "no_metadata", "ngc_build_unknown")
# Why a catalog entry is not a row: a production branch (`-pb6`), a container with no model of
# its own, not an LLM or VLM, not public on NGC, not a NIM (CND), or a CND catalog item that names
# no image.
SKIPPED = ("production_branch", "model_free", "non_llm", "not_public", "not_a_nim", "no_image")

# --- which catalog entries are models to list ---------------------------------------------------
#
# A marker is one name part, or several in a row joined by `-` (`page-elements`). Names are split
# into parts at `-`, `_`, `.`, `/` and spaces, in lower case; a marker word also matches the word
# followed by a version number (`embed` matches `embed1`, `wan` matches `wan2`). The lists were
# written by reading every name in the NGC NIM catalog (310 NIMs in org nim) and the TGC (91),
# TuringCM (76) and Lichan (49) catalogs on 2026-11-11, and checked against the inference_backend of
# each NGC NIM's newest metadata: none of the 135 NIMs with an LLM backend matches NON_LLM.

# NIMs that serve no LLM or VLM.
NON_LLM = (
    # speech and audio
    "asr", "tts", "nmt", "parakeet", "canary", "whisper", "magpie", "fastpitch", "hifigan",
    "chatterbox", "transcribe", "voicechat", "bnr", "audio2face", "studio-voice",
    "active-speaker", "eye-contact", "riva-translate-1",
    # retrieval: embedding, reranking, classifiers
    "embed", "embedqa", "embedding", "rerank", "rerankqa", "reranker", "bge", "nvclip", "gliner",
    "semantic-split", "jailbreak",
    # vision and documents, without a language model
    "ocr", "paddleocr", "yolox", "tesseract", "detection", "detect", "page-elements",
    "graphic-elements", "table-structure", "body-pose", "relighting", "harmonizer", "vsr",
    "vista3d", "maisi",
    # image, video and 3D generation
    "flux", "image", "stable-diffusion", "trellis", "wan", "generator", "predict", "transfer",
    # biology, chemistry, weather, physics and planning
    "evo", "alphafold", "openfold", "proteinmpnn", "proteina", "rfdiffusion", "boltz",
    "diffdock", "msa-search", "genmol", "molmim", "alchemi", "fourcastnet", "corrdiff",
    "domino", "kumo", "cuopt", "mace", "genos", "rdt",
)  # fmt: skip
# Containers that load a model given at run time: no model of their own to list.
MODEL_FREE = ("model-free", "basecontainer", "base-container")
MODEL_FREE_NAMES = frozenset({"llm-nim"})
# Images on a China NIM distributor that are not NIMs (Lichan's project cnd holds a few).
NOT_A_NIM = (
    "etcd", "minio", "milvus", "neo4j", "busybox", "ubuntu", "comfyui", "isaac", "blueprint",
    "vllm-openai",
)  # fmt: skip
PRODUCTION_BRANCH = re.compile(r"(?:^|-)pb[0-9][0-9a-z.]*$")
# What a catalog (TGC `model_type`) or a label (registry_labels.nim_category) calls a NIM.
LLM_CATEGORIES = frozenset({"llm", "vlm", "multimodal", "multi modal", "multi-modal"})
OTHER_CATEGORIES = frozenset(
    {
        "reranker", "reranking", "embedding", "retriever", "retrieval", "diffusion",
        "ai4science", "ai4s", "others", "other", "riva", "asr", "tts", "speech", "health",
        "triton", "forecasting",
    }
)  # fmt: skip
# inference_backend values of LLM and VLM profiles (`-` read as `_`).
LLM_BACKENDS = frozenset({"vllm", "vllm_omni", "sglang", "tensorrt_llm", "trtllm"})


def name_parts(text: str) -> list[str]:
    """`Qwen3.6-27B-FP8 Spark` -> ['qwen3', '6', '27b', 'fp8', 'spark']."""
    return [part for part in re.split(r"[-_./\s]+", (text or "").lower()) if part]


def _word_matches(word: str, part: str) -> bool:
    if part == word:
        return True
    return word.isalpha() and part.startswith(word) and part[len(word) :].isdigit()


def marker_in(text: str, markers: Iterable[str]) -> str | None:
    """The first marker the name holds (see the markers above), or None."""
    parts = name_parts(text)
    for marker in markers:
        words = marker.split("-")
        for start in range(len(parts) - len(words) + 1):
            if all(_word_matches(w, p) for w, p in zip(words, parts[start:])):
                return marker
    return None


def category_kind(value: object) -> str | None:
    """`llm` for an LLM or VLM category, `non_llm` for another one we know, None otherwise."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = " ".join(value.lower().split())
    if text in LLM_CATEGORIES:
        return "llm"
    return "non_llm" if text in OTHER_CATEGORIES else None


def classify(texts: Sequence[str], category: str | None = None) -> str | None:
    """Why a catalog entry is not listed: `not_a_nim`, `model_free` or `non_llm`; None to list it.
    `texts` are its names (repository name, display or catalog name); `category` what the catalog
    calls it. The name lists win over a catalog that calls the entry an LLM: TGC calls bge-m3 and
    a reranker `llm`."""
    names = [text for text in texts if text]
    if any(marker_in(text, NOT_A_NIM) for text in names):
        return "not_a_nim"
    if any(marker_in(text, MODEL_FREE) or text.lower() in MODEL_FREE_NAMES for text in names):
        return "model_free"
    if any(marker_in(text, NON_LLM) for text in names) or category_kind(category) == "non_llm":
        return "non_llm"
    return None


def is_production_branch(name: str) -> bool:
    """`llama-3.1-8b-instruct-pb25h2`, `gpt-oss-120b-pb6`."""
    return PRODUCTION_BRANCH.search((name or "").lower()) is not None


def backend_kind(documents: Iterable[dict[str, Any] | None]) -> str | None:
    """`llm` when a profile of the documents names an LLM backend, `non_llm` when profiles name
    only other backends (trt, onnx, pytorch), None when no profile names one."""
    named = {
        str(profile["inference_backend"]).strip().lower().replace("-", "_")
        for document in documents
        if isinstance(document, dict)
        for profile in document.get("profiles") or []
        if isinstance(profile, dict) and isinstance(profile.get("inference_backend"), str)
    }
    named.discard("")
    if named & LLM_BACKENDS:
        return "llm"
    return "non_llm" if named else None


def query_terms(query: str | None) -> list[str]:
    """The words of a --query, in lower case. A leading `nvcr.io/` and `nim/` are dropped: the
    names a query matches (`meta/llama-3.1-8b-instruct`) have neither."""
    text = (query or "").strip().lower()
    for prefix in ("nvcr.io/", "nim/"):
        text = text.removeprefix(prefix)
    return text.split()


def _term_in(term: str, text: str) -> bool:
    """A word of letters is found anywhere in the text (`llama` in `codellama`). A word that holds
    a digit is found only where no letter or digit follows it, and, when it starts with a digit,
    where it starts no number part: neither a digit nor `<digit>.` comes before it. So `4` is in
    `llama-4-scout`, not in `49b`, `nano-4b` or `405b`; `8b` is not in `128b`; `2` is in
    `llama-2-7b`, not in `llama-3.2-1b`; and `3` and `3.1` are in `llama3.1-nemotron` and
    `llama-3.1-8b` (a letter may come before it)."""
    text = text.lower()
    if not any(char.isdigit() for char in term):
        return term in text
    before = r"(?<![0-9])(?<![0-9]\.)" if term[0].isdigit() else ""
    return re.search(before + re.escape(term) + r"(?![a-z0-9])", text) is not None


def query_matches(query: str | None, texts: Sequence[str]) -> bool:
    """ns.every_word_in on query_terms(query), each word tested by _term_in (stricter than
    ns.in_text for a word with a digit); no query: True."""
    terms = query_terms(query)
    return not terms or ns.every_word_in(terms, texts, _term_in)


# --- hardware variants ------------------------------------------------------------------------


def fold_variants(resource_ids: Sequence[str]) -> dict[str, list[str]]:
    """Each model with the hardware variants found beside it (`<repo>-dgx-spark`, `<repo>-rtx` in
    the same team, choose.split_variants): {base: [variants]}, bases in catalog order. A variant
    whose base is not among `resource_ids` is a model of its own."""
    variants = {rid: choose.split_variants(rid, resource_ids).variants for rid in resource_ids}
    folded = {variant for found in variants.values() for variant in found}
    return {rid: found for rid, found in variants.items() if rid not in folded}


# --- region -----------------------------------------------------------------------------------

_GEOGRAPHY = re.compile(r"deployment\s+geography", re.IGNORECASE)
_HTML = re.compile(r"<[^>]*>")
EU_WORDS = frozenset({"eu", "european union"})
# The 27 EU member states, by English name and ISO 3166 code.
EU_COUNTRIES: dict[str, str] = {
    "austria": "at", "belgium": "be", "bulgaria": "bg", "croatia": "hr", "cyprus": "cy",
    "czechia": "cz", "czech republic": "cz", "denmark": "dk", "estonia": "ee", "finland": "fi",
    "france": "fr", "germany": "de", "greece": "gr", "hungary": "hu", "ireland": "ie",
    "italy": "it", "latvia": "lv", "lithuania": "lt", "luxembourg": "lu", "malta": "mt",
    "netherlands": "nl", "holland": "nl", "poland": "pl", "portugal": "pt", "romania": "ro",
    "slovakia": "sk", "slovenia": "si", "spain": "es", "sweden": "se",
}  # fmt: skip
# Places outside the EU that users name often; a place in neither table is not judged.
OUTSIDE_EU = frozenset(
    {
        "us", "usa", "united states", "united states of america", "america", "uk",
        "united kingdom", "great britain", "britain", "england", "scotland", "wales",
        "northern ireland", "gb", "switzerland", "ch", "norway", "no", "iceland", "is",
        "liechtenstein", "li", "india", "in", "china", "cn", "prc", "mainland china", "japan",
        "jp", "korea", "south korea", "kr", "taiwan", "tw", "hong kong", "hk", "singapore",
        "sg", "australia", "au", "new zealand", "nz", "canada", "ca", "mexico", "mx", "brazil",
        "br", "argentina", "ar", "chile", "cl", "colombia", "co", "israel", "il", "turkey",
        "türkiye", "tr", "uae", "united arab emirates", "ae", "saudi arabia", "sa", "qatar",
        "qa", "south africa", "za", "egypt", "eg", "nigeria", "ng", "kenya", "ke", "indonesia",
        "id", "malaysia", "my", "thailand", "th", "vietnam", "vn", "philippines", "ph",
        "ukraine", "ua", "serbia", "rs", "russia", "ru", "north america", "south america",
        "latin america", "latam", "asia", "apac", "asia pacific", "africa", "middle east",
        "oceania",
    }
)  # fmt: skip

# Every ISO 3166-1 alpha-2 country code (current as of 2024), and XK (Kosovo: a user-assigned
# code in common use), as `nimselect.py country` gives one: a code that is not an EU member's
# names a place outside the EU.
ISO_COUNTRIES = frozenset(
    (
        "ad ae af ag ai al am ao aq ar as at au aw ax az ba bb bd be bf bg bh bi bj bl bm bn bo "
        "bq br bs bt bv bw by bz ca cc cd cf cg ch ci ck cl cm cn co cr cu cv cw cx cy cz de dj "
        "dk dm do dz ec ee eg eh er es et fi fj fk fm fo fr ga gb gd ge gf gg gh gi gl gm gn gp "
        "gq gr gs gt gu gw gy hk hm hn hr ht hu id ie il im in io iq ir is it je jm jo jp ke kg "
        "kh ki km kn kp kr kw ky kz la lb lc li lk lr ls lt lu lv ly ma mc md me mf mg mh mk ml "
        "mm mn mo mp mq mr ms mt mu mv mw mx my mz na nc ne nf ng ni nl no np nr nu nz om pa pe "
        "pf pg ph pk pl pm pn pr ps pt pw py qa re ro rs ru rw sa sb sc sd se sg sh si sj sk sl "
        "sm sn so sr ss st sv sx sy sz tc td tf tg th tj tk tl tm tn to tr tt tv tw tz ua ug um "
        "us uy uz va vc ve vg vi vn vu wf ws xk ye yt za zm zw"
    ).split()
)

GEOGRAPHY_MAX = 80  # characters of a geography text kept: it is the card's text, shown as data


def geography(description: object) -> str | None:
    """The card's "Deployment Geography" text (`Global`, `Global, except EU`), or None when the
    card has none. HTML tags and Markdown emphasis are dropped; the text ends at a blank line or a
    heading, and at GEOGRAPHY_MAX characters (a longer text is cut, with `...`)."""
    text = _geography(description)
    if text and len(text) > GEOGRAPHY_MAX:
        return text[: GEOGRAPHY_MAX - 3].rstrip() + "..."
    return text


def _geography(description: object) -> str | None:
    if not isinstance(description, str):
        return None
    found = _GEOGRAPHY.search(description)
    if found is None:
        return None
    lines: list[str] = []
    for raw in _HTML.sub("\n", description[found.end() : found.end() + 400]).splitlines():
        line = raw.strip().strip("*").strip().lstrip(":").strip().strip("*").strip()
        if not line:
            if lines:
                break
            continue
        if line.startswith("#"):
            break
        lines.append(line)
    text = " ".join(lines).strip()
    # `The container will be deployed in the following geography: Global.`
    head, colon, tail = text.rpartition(":")
    if colon and "geograph" in head.lower():
        text = tail
    text = text.strip().strip("*").strip().rstrip(".").strip()
    return text or None


def _place(text: str) -> str | None:
    """`eu`, an EU country code, `outside` for a known place outside the EU (any other ISO 3166
    code included), None for others."""
    word = " ".join(text.lower().replace(".", "").split()).removeprefix("the ").strip()
    if word in EU_WORDS:
        return "eu"
    if word in EU_COUNTRIES:
        return EU_COUNTRIES[word]
    if word in EU_COUNTRIES.values():
        return word
    return "outside" if word in OUTSIDE_EU or word in ISO_COUNTRIES else None


def exclusions(text: str | None) -> set[str] | None:
    """The places a geography excludes: set() for `Global`, {"eu"} for `Global, except EU`; None
    when it cannot be read (it is then shown as it is and not judged)."""
    if not text:
        return None
    lowered = " ".join(text.lower().replace(".", " ").split())
    if lowered in ("global", "worldwide"):
        return set()
    match = re.fullmatch(
        r"(?:global|worldwide)\s*,?\s*(?:except(?: for)?|excluding|but not|other than)\s+(.+)",
        lowered,
    )
    if match is None:
        return None
    places = {_place(part) for part in re.split(r"\s*(?:,|/|&|\band\b)\s*", match.group(1))}
    if None in places or "outside" in places or not places:
        return None
    return {place for place in places if place}


def user_places(region: str | None) -> set[str] | None:
    """What the user's region names (`eu`, EU country codes, `outside`), or None when no part of
    it is known (`Berlin, Germany` names `de`)."""
    if not region or not region.strip():
        return None
    places = {_place(part) for part in re.split(r"[,/()]+", region) if part.strip()}
    places.discard(None)
    return {place for place in places if place} or None


def region_excluded(text: str | None, region: str | None) -> bool | None:
    """True when the geography excludes the user's region, False when it does not, None when that
    cannot be told: no geography, one that cannot be read, a region not given or not known, or
    only `EU` given when the geography excludes some EU countries."""
    excluded = exclusions(text)
    if excluded is None:
        return None
    if not excluded:
        return False
    places = user_places(region)
    if places is None:
        return None
    countries = places & set(EU_COUNTRIES.values())
    in_eu = "eu" in places or bool(countries)
    if ("eu" in excluded and in_eu) or countries & excluded:
        return True
    if "eu" in places and not countries and excluded - {"eu"}:
        return None
    return False


def judged_global(text: str | None) -> bool:
    """A geography that excludes nothing (`Global`), or none on the card."""
    return text is None or exclusions(text) == set()


# --- the cache of NGC answers -------------------------------------------------------------------

HOUR = 3600.0
CACHE_VERSION = 1
# Fields of a repository record the scripts read (description: only its geography part).
REPO_FIELDS = (
    "name", "displayName", "tags", "latestTag", "latestArchitectureVariants", "latestImageSize",
    "canGuestPull", "isPublic", "labels",
)  # fmt: skip
CATALOG_FIELDS = ("resourceId", "orgName", "teamName", "name", "displayName")


def cache_kind(key: str) -> str | None:
    """`catalog`, `images`, `repo` or `metadata` for an NGC path and query; None: not kept."""
    path = key.split("?", 1)[0]
    if path.startswith("/search/catalog/"):
        return "catalog"
    if path.startswith("/repos/"):
        return "images" if path.endswith("/images") else "repo"
    if path == "/nim/metadata":
        return "metadata"
    return None


def lifetime(key: str, status: int) -> float | None:
    """Seconds an answer is kept: metadata 24 h; repository records, image lists and the catalog
    1 h; a 404 (none published) or a 401/403 (not public) 1 h; anything else (5xx, 429) never."""
    kind = cache_kind(key)
    if kind is None:
        return None
    if status == 200:
        return 24 * HOUR if kind == "metadata" else HOUR
    return HOUR if status in (401, 403, 404) else None


def cacheable(key: str, status: int, body: object) -> bool:
    """Whether an answer may be kept (and, read back, used): a kind and status the cache keeps
    (lifetime), in the shape its reader accepts, so an answer the reader refuses is asked for again
    the next time, never taken from the cache. A 200: a catalog page with its results list
    (ngc_api.catalog_resources), a repository record (a name or a tags list), an image list with
    its images list, a metadata document with its profiles list. A 401, 403 or 404 only with NGC's
    error envelope: a proxy's or a firewall's page says nothing about the NIM."""
    if lifetime(key, status) is None or not isinstance(body, dict):
        return False
    if status != 200:
        return ngc_api.has_envelope(body)
    kind = cache_kind(key)
    if kind == "catalog":
        return ngc_api.catalog_resources(body) is not None
    if kind == "repo":
        return ngc_api.is_repo_record(body)
    if kind == "images":
        return isinstance(body.get("images"), list)
    return isinstance(body.get("profiles"), list)  # metadata


def trimmed(key: str, status: int, body: dict[str, Any]) -> dict[str, Any]:
    """What the cache keeps of an answer `cacheable` accepts: an error's requestStatus; of a
    repository record, the fields the scripts read and the geography part of its card; of a
    catalog page, the names of its CONTAINER resources; a metadata document or an image list as it
    is."""
    if status != 200:
        return {"requestStatus": body.get("requestStatus")}
    kind = cache_kind(key)
    if kind == "repo":
        kept = {name: body[name] for name in REPO_FIELDS if name in body}
        description = body.get("description")
        found = _GEOGRAPHY.search(description) if isinstance(description, str) else None
        if description is not None:
            kept["description"] = description[found.start() : found.end() + 400] if found else ""
        return kept
    if kind == "catalog":
        resources = [
            {name: resource.get(name) for name in CATALOG_FIELDS}
            for resource in ngc_api.catalog_resources(body) or []
        ]
        counts = {name: body[name] for name in ("resultTotal", "resultPageTotal") if name in body}
        return {"results": [{"groupValue": "CONTAINER", "resources": resources}], **counts}
    return body


class NgcCache:
    """NGC answers kept between list runs (ngc_api.AnswerCache), in ~/.cache/nim-select/
    ngc-cache.json: read only when ns.read_private_json accepts the file, written with
    ns.write_json_file. Only answers `cacheable` accepts are kept, and an entry `_valid` refuses
    is dropped (the policy: references/list-mode.md, The cache). Thread-safe."""

    def __init__(self, path: Path | None, now: Callable[[], float] = time.time) -> None:
        self.path = path
        self.now = now
        self.entries: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.dropped = 0
        self.problem: str | None = None
        self.saved = False

    def load(self) -> None:
        """Read the file, keeping only entries of the right shape that are still valid."""
        if self.path is None:
            return
        data, problem = ns.read_private_json(self.path)
        if problem:
            self.problem = f"the cache was not used: {problem}"
            ns.log(self.problem)
            return
        if data is None:
            return
        if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
            self.problem = f"the cache was not used: {self.path} is not a nim-select cache"
            return
        entries = data.get("entries")
        now = self.now()
        for key, entry in (entries if isinstance(entries, dict) else {}).items():
            if self._valid(key, entry, now):
                self.entries[key] = entry
            else:
                self.dropped += 1

    def _valid(self, key: object, entry: object, now: float) -> bool:
        if not isinstance(key, str) or not key.startswith("/") or not isinstance(entry, dict):
            return False
        at, status, body = entry.get("at"), entry.get("status"), entry.get("body")
        if isinstance(at, bool) or not isinstance(at, (int, float)) or not math.isfinite(at):
            return False
        if (
            isinstance(status, bool)
            or not isinstance(status, int)
            or not cacheable(key, status, body)
        ):
            return False
        keep = lifetime(key, status)
        return keep is not None and at <= now + 60 and now - at < keep

    def get(self, key: str) -> tuple[int, dict[str, Any]] | None:
        with self.lock:
            entry = self.entries.get(key)
            if entry is not None and not self._valid(key, entry, self.now()):
                del self.entries[key]
                entry = None
            if entry is None:
                self.misses += 1
                return None
            self.hits += 1
            return int(entry["status"]), entry["body"]

    def put(self, key: str, status: int, body: dict[str, Any]) -> None:
        if not cacheable(key, status, body):
            return
        entry = {"at": self.now(), "status": status, "body": trimmed(key, status, body)}
        with self.lock:
            self.entries[key] = entry

    def save(self) -> None:
        """Write the file (nothing for a cache without a path). A failure is logged, never
        raised: the list stands without its cache."""
        if self.path is None:
            return
        with self.lock:
            data = {"version": CACHE_VERSION, "entries": dict(self.entries)}
        try:
            ns.write_json_file(self.path, data, compact=True)
            self.saved = True
        except (OSError, ns.ScriptError) as exc:
            reason = exc.message if isinstance(exc, ns.ScriptError) else exc.strerror
            self.problem = f"the cache was not written: {reason}"
            ns.log(self.problem)

    def facts(self) -> dict[str, Any]:
        with self.lock:
            return {
                "path": str(self.path) if self.path else None,
                "used": self.path is not None,
                "hits": self.hits,
                "misses": self.misses,
                "dropped": self.dropped,
                "saved": self.saved,
                "problem": self.problem,
            }


# --- checking models with a deadline ----------------------------------------------------------


@dataclass
class Outcome:
    """What run_tasks did: each task's result or error (by index), the tasks the deadline cut
    while they ran, the tasks never started, and the error that stopped the run, if any."""

    results: dict[int, Any] = field(default_factory=dict)
    errors: dict[int, BaseException] = field(default_factory=dict)
    cut: list[int] = field(default_factory=list)
    not_started: list[int] = field(default_factory=list)
    stopped_by: ns.ScriptError | None = None
    late: bool = False


def run_tasks(
    tasks: Sequence[Callable[[], Any]],
    workers: int,
    deadline_at: float,
    stop_codes: Mapping[str, int] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Outcome:
    """Run the tasks, at most `workers` at a time, in daemon threads, until all have ended or the
    clock passes `deadline_at` (a hard stop: tasks still running are left behind and end with the
    process). `stop_codes` maps a ScriptError code to how many tasks in a row, as they end, may
    fail with such codes before the run stops: 1 stops it at once; 3 lets one model's failure (a
    reset, a timeout) pass while a source that stopped answering stops the run. A task that ends
    any other way starts the count again. No task starts after a stop; the ones running may end."""
    stop_on = dict(stop_codes or {})
    lock = threading.Condition()
    results: dict[int, Any] = {}
    errors: dict[int, BaseException] = {}
    stopped: list[ns.ScriptError] = []
    state = {"next": 0, "stop": False, "failed": 0}
    running: set[int] = set()

    def worker() -> None:
        while True:
            with lock:
                if state["stop"] or state["next"] >= len(tasks) or clock() >= deadline_at:
                    return
                index = state["next"]
                state["next"] += 1
                running.add(index)
            try:
                result = tasks[index]()
            except BaseException as exc:  # recorded, never raised in a thread
                # recorded and taken out of `running` under one lock: never both ended and cut
                with lock:
                    errors[index] = exc
                    limit = stop_on.get(exc.code) if isinstance(exc, ns.ScriptError) else None
                    state["failed"] = 0 if limit is None else state["failed"] + 1
                    if limit is not None and state["failed"] >= limit:
                        state["stop"] = True
                        stopped.append(exc)
                    running.discard(index)
                    lock.notify_all()
            else:
                with lock:  # the same: recorded and taken out of `running` under one lock
                    results[index] = result
                    state["failed"] = 0
                    running.discard(index)
                    lock.notify_all()

    count = min(workers, len(tasks))
    for thread in [threading.Thread(target=worker, daemon=True) for _ in range(count)]:
        thread.start()
    outcome = Outcome()
    with lock:
        while running or (not state["stop"] and state["next"] < len(tasks)):
            left = deadline_at - clock()
            if left <= 0:
                outcome.late = True
                break
            lock.wait(min(left, 1.0))
        state["stop"] = True
        # a copy: a task left behind changes nothing in the outcome
        outcome.results, outcome.errors = dict(results), dict(errors)
        outcome.cut = sorted(running)
        outcome.not_started = list(range(state["next"], len(tasks)))
        outcome.stopped_by = stopped[0] if stopped else None
    return outcome


# --- rows ---------------------------------------------------------------------------------------


def all_gpus(env: dict[str, Any]) -> dict[str, Any]:
    """The environment with every whole GPU of each type idle: the list judges what the host can
    run, and marks the rows that need GPUs busy now (needs_busy_gpus). On Kubernetes nothing
    changes: fit counts the allocatable GPUs of one node."""
    if env.get("target") == "k8s":
        return env
    groups = []
    for group in env.get("groups") or []:
        if isinstance(group, dict):
            group = {**group, "idle": group.get("count")}
            if isinstance(group.get("indices"), list):
                group["idle_indices"] = list(group["indices"])
        groups.append(group)
    return {**env, "groups": groups}


def idle_of(env: dict[str, Any], gpu: dict[str, Any] | None) -> int | None:
    """The idle GPUs, now, of the GPU type `gpu` (a fit result's group) on a host; None on
    Kubernetes or when not recorded."""
    if env.get("target") == "k8s" or not gpu:
        return None
    for group in env.get("groups") or []:
        if not isinstance(group, dict):
            continue
        if (group.get("name"), group.get("pci_device_id")) == (
            gpu.get("name"),
            gpu.get("pci_device_id"),
        ):
            idle = group.get("idle")
            return idle if isinstance(idle, int) and not isinstance(idle, bool) else None
    return None


def smallest(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The profile the row shows: the first (best, by select's ranking) of those that use the
    fewest GPUs."""
    if not records:
        return None
    fewest = min(int(record.get("gpus_required") or 1) for record in records)
    return next(r for r in records if int(r.get("gpus_required") or 1) == fewest)


def top_codes(counts: dict[str, int]) -> list[str]:
    """Rejection codes, the most frequent first."""
    return [code for code, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))]


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """fits first (fewest GPUs first), then unknown (grouped by reasons), then does_not_fit,
    no_metadata and ngc_build_unknown; by model within a group."""

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        verdict = row.get("verdict")
        rank = VERDICTS.index(verdict) if verdict in VERDICTS else len(VERDICTS)
        gpus = row.get("gpus_min") if verdict == "fits" else 0
        reasons = tuple(row.get("reasons") or ()) if verdict == "unknown" else ()
        return (rank, gpus or 0, reasons, str(row.get("model")), str(row.get("image")))

    return sorted(rows, key=key)


SHOWN = ("fits", "unknown")  # the verdicts whose rows the agent shows
# The fields of a row the list prints (brief): these, and the marks that say something.
BRIEF_FIELDS = ("model", "image", "verdict", "reasons", "gpus_min", "gpu_type")
FITS_FIELDS = ("precision", "backend")


def brief(row: dict[str, Any]) -> dict[str, Any]:
    """A row as the list prints it: model, image, verdict, reasons, gpus_min and gpu_type (fits:
    also precision and backend), and only the marks that say something: needs_ngc_key,
    needs_busy_gpus, region_excluded and records_unread when true; kind when it is not newest;
    variant_of; pinned when false; geography when it is not Global; on a CND, ngc_image and
    registry_gpus. The --out file has every field."""
    kept = {name: row.get(name) for name in BRIEF_FIELDS}
    if row.get("verdict") == "fits":
        kept.update({name: row.get(name) for name in FITS_FIELDS})
    for name in ("needs_ngc_key", "needs_busy_gpus", "region_excluded", "records_unread"):
        if row.get(name) is True:
            kept[name] = True
    if row.get("kind") not in (None, "newest"):
        kept["kind"] = row["kind"]
    if row.get("pinned") is False:
        kept["pinned"] = False
    if not judged_global(row.get("geography")):
        kept["geography"] = row["geography"]
    for name in ("variant_of", "ngc_image", "registry_gpus"):
        if row.get(name):
            kept[name] = row[name]
    return kept


def summary_rows(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """(the rows the agent shows: fits and unknown, brief; the models of the other rows, by
    verdict), in the order of `rows`."""
    shown: list[dict[str, Any]] = []
    others: dict[str, list[str]] = {}
    for row in rows:
        if row.get("verdict") in SHOWN:
            shown.append(brief(row))
        else:
            others.setdefault(str(row.get("verdict")), []).append(str(row.get("model")))
    return shown, others


def counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    found = {verdict: 0 for verdict in VERDICTS}
    for row in rows:
        if row.get("verdict") in found:
            found[row["verdict"]] += 1
    return found


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="The list of compatible models for nim-select (a library); "
        "`nimselect.py list` runs it.",
        epilog='Example: python3 -I "<SKILL_DIR>/scripts/nimselect.py" list '
        "--env ~/.cache/nim-select/env.json --source ngc",
    )
    parser.parse_args(argv)
    ns.emit(True, "library", "done", {"note": "import this module; run nimselect.py list instead"})


if __name__ == "__main__":
    main()
