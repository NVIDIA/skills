#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NGC access for nim-select: `import ngc_api`.

Parses what the user calls a model (an image reference, `team/repo`, a registry mirror reference or
free text), searches the NGC catalog for NIM containers (or lists them all, page by page, for the
list command), reads a repository's tags and its public image list (each tag's digest, CPU
architectures and size), picks the tag to evaluate and fetches the NIM Metadata API document for
that tag. Read-only GET requests, stdlib only. Every request goes to https://api.ngc.nvidia.com/v2:
no environment variable changes it, and no redirect is followed (a redirect is ngc_bad_response).
At most MAX_REQUESTS are open at a time, whatever thread sends them. The first retry delay comes
from NIM_SELECT_RETRY_DELAY (seconds, default 1.0, at most MAX_RETRY_DELAY), read at call time.
SETTINGS holds the timeout and attempts of a request, and a cache of answers that only the list
command sets. Run directly, it prints a notice: it has no command of its own.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_common as ns

DEFAULT_BASE = "https://api.ngc.nvidia.com/v2"
USER_AGENT = "nim-select/0.1"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_DELAY = 10.0  # seconds; the second retry waits twice as long
MAX_REQUESTS = 8  # NGC requests open at a time, in every thread
NGC_HOST = "nvcr.io"
MIRROR_PREFIXES = ("public/", "nim/")
NIM_FILTER = {"field": "nvidia_nim", "value": "nimmcro_nvidia_nim"}
CATALOG_PAGE_SIZE = 1000  # NIMs per catalog page (one page held all 311 on 2026-11-11)
CATALOG_PAGES = 5  # pages read at most
_SLOTS = threading.BoundedSemaphore(MAX_REQUESTS)

Mapping = Literal["exact", "by_name", "search"]


class AnswerCache(Protocol):
    """What get_json needs of a cache of NGC answers (the list command's, listing.NgcCache)."""

    def get(self, key: str) -> tuple[int, dict[str, Any]] | None: ...

    def put(self, key: str, status: int, body: dict[str, Any]) -> None: ...


@dataclass
class Settings:
    """How get_json sends a request: its timeout in seconds and attempts (when the caller names
    none), and a cache of answers. select and fit keep the defaults and no cache; the list command
    sets a shorter timeout, fewer attempts and its cache."""

    timeout: float = 30.0
    attempts: int = 3
    cache: AnswerCache | None = None


SETTINGS = Settings()


def base_url() -> str:
    """The fixed NGC API root; no environment variable can redirect the requests."""
    return DEFAULT_BASE


def retry_delay() -> float:
    """First back-off delay in seconds (doubles per retry); NIM_SELECT_RETRY_DELAY overrides it
    with a number from 0 to MAX_RETRY_DELAY (a larger one is cut to it; a value that is not a
    finite number is the default, 1)."""
    raw = os.environ.get("NIM_SELECT_RETRY_DELAY", "")
    try:
        value = float(raw) if raw.strip() else 1.0
    except ValueError:
        return 1.0
    return min(max(value, 0.0), MAX_RETRY_DELAY) if math.isfinite(value) else 1.0


def request_id(body: object) -> str | None:
    """`requestStatus.requestId` of an NGC error envelope, when there is one."""
    if isinstance(body, dict):
        status = body.get("requestStatus")
        if isinstance(status, dict) and isinstance(status.get("requestId"), str):
            return str(status["requestId"])
    return None


def _with_request_id(message: str, body: object) -> str:
    rid = request_id(body)
    return f"{message} (requestId {rid})" if rid else message


BAD_RESPONSE_FIX = (
    "A proxy or captive portal may be answering instead of NGC: check HTTPS_PROXY, then run "
    "the command again; if it keeps failing, report it"
)


def _parse_body(raw: bytes) -> dict[str, Any]:
    """The JSON body of an error answer, or {}: only its status matters."""
    try:
        data = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {"items": data}


def _parse_ok(raw: bytes, url: str) -> dict[str, Any]:
    """The JSON body of a 2xx answer. A body that is not JSON (an HTML proxy page) is an error,
    never an empty answer."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        start = " ".join(raw[:60].decode("utf-8", errors="replace").split())
        raise ns.ScriptError(
            "ngc_bad_response",
            f"{url} answered with a body that is not JSON (it starts {start!r})",
            BAD_RESPONSE_FIX,
            ns.EXIT_EXTERNAL,
        ) from None
    return data if isinstance(data, dict) else {"items": data}


def _redirected(url: str, status: int, location: str | None) -> ns.ScriptError:
    """ngc_bad_response (exit 4) for a redirect: NGC sends none, so a proxy or a portal answered,
    and the request is not sent on to the host it names."""
    host = urllib.parse.urlsplit(location or "").hostname
    where = f" to {host}" if host else ""
    return ns.ScriptError(
        "ngc_bad_response",
        f"{url} answered HTTP {status}, a redirect{where}, which nim-select does not follow",
        BAD_RESPONSE_FIX,
        ns.EXIT_EXTERNAL,
    )


def get_json(
    path_and_query: str,
    *,
    timeout: float | None = None,
    attempts: int | None = None,
    no_retry: frozenset[int] = frozenset(),
) -> tuple[int, dict[str, Any]]:
    """GET base_url() + path_and_query; return (HTTP status, parsed JSON body).

    Retries with back-off (delay, 2*delay) on 429, 5xx (except statuses in `no_retry`), timeouts
    and connection errors; never on other 4xx. After the last attempt an HTTP answer is returned
    as is; no HTTP answer at all raises ScriptError `ngc_unreachable` (exit 4). An error answer
    whose body cannot be read still counts by its status (its body is then {}). A success whose
    body is not JSON, or a redirect (never followed), raises `ngc_bad_response` (exit 4), without
    a retry. `timeout` and `attempts` default to SETTINGS'. With SETTINGS.cache, an answer it
    holds is returned without a request, and each answer is offered to it (it keeps what it may).
    """
    cache = SETTINGS.cache
    if cache is not None:
        held = cache.get(path_and_query)
        if held is not None:
            return held
    status, body = _request(
        path_and_query,
        SETTINGS.timeout if timeout is None else timeout,
        SETTINGS.attempts if attempts is None else attempts,
        no_retry,
    )
    if cache is not None:
        cache.put(path_and_query, status, body)
    return status, body


def _request(
    path_and_query: str, timeout: float, attempts: int, no_retry: frozenset[int]
) -> tuple[int, dict[str, Any]]:
    """The requests of get_json, at most MAX_REQUESTS open at a time."""
    url = base_url() + path_and_query
    headers = {"accept": "application/json", "User-Agent": USER_AGENT}
    last_problem = "no attempt made"
    for attempt in range(max(1, attempts)):
        if attempt:
            delay = retry_delay() * (2 ** (attempt - 1))
            ns.log(f"retrying {url} in {delay:g}s ({last_problem})")
            time.sleep(delay)
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            status, raw, location = _send(request, timeout)
            if 200 <= status < 300:
                return status, _parse_ok(raw, url)
            body = _parse_body(raw)
            if 300 <= status < 400:
                raise _redirected(url, status, location)
            if status not in RETRY_STATUSES or status in no_retry or attempt == attempts - 1:
                return status, body
            last_problem = _with_request_id(f"HTTP {status}", body)
        except TimeoutError as exc:
            last_problem = f"timed out after {timeout:g}s ({exc})"
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, TimeoutError):
                last_problem = f"timed out after {timeout:g}s"
            else:
                last_problem = f"connection failed: {reason}"
        except (OSError, http.client.HTTPException) as exc:
            last_problem = f"connection failed: {exc}"
    raise ns.ScriptError(
        "ngc_unreachable",
        f"cannot reach {base_url()} after {attempts} attempts: {last_problem}",
        "Check network access to api.ngc.nvidia.com (proxy settings such as HTTPS_PROXY), "
        "then run the command again",
        ns.EXIT_EXTERNAL,
    )


def _send(request: urllib.request.Request, timeout: float) -> tuple[int, bytes, str | None]:
    """(HTTP status, body, Location header) of one request, which keeps its slot until its answer
    is read and closed. An error answer (urllib's HTTPError) is returned too. Its body is read only
    for its requestId, so a body that cannot be read (cut short, or timed out after the status
    line) is b"" and the status still counts."""
    with _SLOTS:
        try:
            with ns.open_url(request, timeout) as response:
                return int(response.status), response.read(), None
        except urllib.error.HTTPError as exc:
            location = exc.headers.get("Location") if exc.headers else None
            try:
                raw = exc.read()
            except (OSError, http.client.HTTPException):
                raw = b""
            finally:
                exc.close()
            return int(exc.code), raw, location


RATE_LIMITED_FIX = (
    "NGC asks for fewer requests: wait a few minutes before running the command again, and never "
    "retry at once"
)


def _unexpected(status: int, body: object, what: str) -> ns.ScriptError:
    """ngc_http_error (exit 4); facts.http_status tells a 403 from a 503, and facts.not_public is
    true for a 401 or 403 that carries NGC's error envelope (NGC says it is not public; a proxy's
    or a firewall's 403 page says nothing about the NIM). A 429 that stays after the retries is
    ngc_rate_limited: NGC asks for fewer requests, so the command stops sending them."""
    if status == 429:
        return ns.ScriptError(
            "ngc_rate_limited",
            _with_request_id(
                f"NGC answered HTTP 429 (too many requests) for {what}, after its retries", body
            ),
            RATE_LIMITED_FIX,
            ns.EXIT_EXTERNAL,
            {"http_status": status},
        )
    facts: dict[str, Any] = {"http_status": status}
    if status in (401, 403) and has_envelope(body):
        facts["not_public"] = True
    return ns.ScriptError(
        "ngc_http_error",
        _with_request_id(f"NGC answered HTTP {status} for {what}", body),
        "Run the command again later; if it keeps failing, report it with the requestId",
        ns.EXIT_EXTERNAL,
        facts,
    )


# ---------------------------------------------------------------------------------------------
# Model references


@dataclass
class ModelRef:
    """What the user named, mapped to an NGC repository or a catalog search."""

    resource_id: str | None
    tag: str | None
    query: str | None
    mapping: Mapping
    source_registry: str | None = None
    warnings: list[str] = field(default_factory=list)


def _looks_like_host(segment: str) -> bool:
    return "." in segment or ":" in segment or segment == "localhost"


def _split_tag(path: str) -> tuple[str, str | None, str | None]:
    """Split `a/b/repo:tag@sha256:...` into (path, tag, digest)."""
    digest = None
    if "@" in path:
        path, digest = path.split("@", 1)
    head, _, last = path.rpartition("/")
    tag = None
    if ":" in last:
        last, tag = last.split(":", 1)
    path = f"{head}/{last}" if head else last
    return path, (tag or None), digest


def mirror_warning(registry: str, same_tag: bool, found: str | None = None) -> str:
    """The warning for an image named on another registry: it is checked as an nvcr.io image.
    `same_tag`: the tag named is the one checked; `found`: the NGC repository a search for the
    name found, when the path names no NGC repository."""
    if found:
        tag = "), with the same tag" if same_tag else "; the tag is chosen on NGC)"
        image = f"nvcr.io/{found} (found by a search for its name{tag}"
    else:
        same = "same name and tag" if same_tag else "same name (the tag is chosen on NGC)"
        image = f"the nvcr.io image with the {same}"
    return (
        f"This image comes from {registry}; it is checked, and handed off, as {image}, which "
        "assumes the two images are the same."
    )


def parse_model_ref(text: str) -> ModelRef:
    """Map a model reference to an NGC resource id (`nim/TEAM/REPO`), or to a search query."""
    raw = (text or "").strip()
    if not raw:
        raise ns.ScriptError(
            "bad_model_ref",
            "the model reference is empty",
            "Name the model, e.g. meta/llama-3.1-8b-instruct or "
            "nvcr.io/nim/meta/llama-3.1-8b-instruct",
        )
    if any(ch.isspace() for ch in raw) or "/" not in raw:
        name, tag, _ = _split_tag(raw) if not any(ch.isspace() for ch in raw) else (raw, None, None)
        return ModelRef(None, tag, name, "search")

    first, _, rest = raw.partition("/")
    warnings: list[str] = []
    if _looks_like_host(first):
        registry = first.lower()
        path, tag, digest = _split_tag(rest)
        if digest:
            warnings.append(
                f"The digest {digest} is ignored; the fit uses the tag or the newest one."
            )
        segments = [s for s in path.lower().split("/") if s]
        if registry == NGC_HOST:
            if not segments:
                raise ns.ScriptError(
                    "bad_model_ref",
                    f"{raw!r} names no repository",
                    "Pass a full image reference, e.g. nvcr.io/nim/meta/llama-3.1-8b-instruct:TAG",
                )
            if segments[0] != "nim":
                warnings.append(f"{raw} is not under nvcr.io/nim; it may not be a NIM.")
            return ModelRef("/".join(segments), tag, None, "exact", None, warnings)
        stripped = "/".join(segments)
        changed = True
        while changed:
            changed = False
            for prefix in MIRROR_PREFIXES:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix) :]
                    changed = True
        parts = [s for s in stripped.split("/") if s]
        warnings.append(mirror_warning(registry, bool(tag and tag != "latest")))
        if not parts:
            raise ns.ScriptError(
                "bad_model_ref",
                f"{raw!r} names no repository",
                "Pass the image reference with its repository name",
            )
        if len(parts) == 1:
            return ModelRef(None, tag, parts[0], "search", registry, warnings)
        if len(parts) > 2:
            warnings.append(
                f"Only the last two path parts of {raw} were used: {'/'.join(parts[-2:])}."
            )
        return ModelRef(f"nim/{'/'.join(parts[-2:])}", tag, None, "by_name", registry, warnings)

    path, tag, digest = _split_tag(raw)
    if digest:
        warnings.append(f"The digest {digest} is ignored; the fit uses the tag or the newest one.")
    segments = [s for s in path.lower().split("/") if s]
    if len(segments) == 3 and segments[0] == "nim":
        return ModelRef("/".join(segments), tag, None, "exact", None, warnings)
    if len(segments) == 2:
        resource = "/".join(segments) if segments[0] == "nim" else "nim/" + "/".join(segments)
        return ModelRef(resource, tag, None, "exact", None, warnings)
    leaf = segments[-1] if segments else raw
    return ModelRef(None, tag, leaf, "search", None, warnings)


# ---------------------------------------------------------------------------------------------
# Catalog search and repository


def _label_values(resource: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for label in resource.get("labels") or []:
        if isinstance(label, dict):
            for key in ("values", "unresolvedValues"):
                values.update(v for v in label.get(key) or [] if isinstance(v, str))
    return values


def _attribute(resource: dict[str, Any], key: str) -> str | None:
    for item in resource.get("attributes") or []:
        if isinstance(item, dict) and item.get("key") == key and isinstance(item.get("value"), str):
            return str(item["value"])
    return None


def parse_search(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Candidates from a catalog search answer: CONTAINER groups only, one per resource id."""
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    results = body.get("results")
    for group in results if isinstance(results, list) else []:
        if not isinstance(group, dict) or group.get("groupValue") != "CONTAINER":
            continue
        resources = group.get("resources")
        for resource in resources if isinstance(resources, list) else []:
            rid = resource.get("resourceId") if isinstance(resource, dict) else None
            if not isinstance(rid, str) or rid in seen:
                continue
            seen.add(rid)
            labels = _label_values(resource)
            candidates.append(
                {
                    "resource_id": rid,
                    "name": resource.get("name"),
                    "display_name": resource.get("displayName"),
                    "latest_tag": _attribute(resource, "latestTag"),
                    "nvaie_supported": bool(
                        {"nvaie_supported", "NVIDIA AI Enterprise Supported"} & labels
                    ),
                    "experimental": "Experimental" in labels,
                }
            )
    return candidates


def search_nims(query: str, page_size: int = 25) -> list[dict[str, Any]]:
    """Search the NGC catalog for NIM containers matching free text."""
    q = {
        "query": query,
        "filters": [NIM_FILTER],
        "orderBy": [{"field": "score", "value": "DESC"}],
        "page": 0,
        "pageSize": page_size,
        "queryFields": ["all", "description", "displayName", "name", "resourceId"],
        "fields": ["name", "displayName", "labels", "attributes", "resourceId"],
    }
    encoded = urllib.parse.quote(json.dumps(q, separators=(",", ":")), safe="")
    status, body = get_json(f"/search/catalog/resources/CONTAINER?q={encoded}")
    if status != 200:
        raise _unexpected(status, body, f"the catalog search {query!r}")
    return parse_search(body)


def catalog_query(page: int, page_size: int = CATALOG_PAGE_SIZE) -> str:
    """The path and query of one page of the NIM catalog: every container with the NIM label."""
    q = {"query": "", "filters": [NIM_FILTER], "page": page, "pageSize": page_size}
    encoded = urllib.parse.quote(json.dumps(q, separators=(",", ":")), safe="")
    return f"/search/catalog/resources/CONTAINER?q={encoded}"


def _count(body: dict[str, Any], key: str) -> int | None:
    value = body.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass
class Catalog:
    """The NIM containers of the catalog (CONTAINER groups only, one per resource id: resourceId,
    orgName, teamName, name, displayName), how many the catalog says it has, the pages it has that
    were not read (more than CATALOG_PAGES, or a later page that failed), how many NIMs it
    counts but did not give (`missing`), and how many rows it gave without a resource id, or with
    a blank one, which cannot be checked (`unusable`)."""

    resources: list[dict[str, Any]]
    total: int | None
    unread_pages: int = 0
    missing: int = 0
    unusable: int = 0

    @property
    def complete(self) -> bool:
        return not self.unread_pages and not self.missing and not self.unusable


def catalog_resources(body: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The resources of the CONTAINER groups of one catalog page; None when the page has no
    results list, or a group's resources are not a list (a shape the scripts do not read)."""
    results = body.get("results")
    if not isinstance(results, list):
        return None
    found: list[dict[str, Any]] = []
    for group in results:
        if not isinstance(group, dict) or group.get("groupValue") != "CONTAINER":
            continue
        resources = group.get("resources", [])
        if not isinstance(resources, list):
            return None
        found.extend(resource for resource in resources if isinstance(resource, dict))
    return found


def list_catalog(page_size: int = CATALOG_PAGE_SIZE, max_pages: int = CATALOG_PAGES) -> Catalog:
    """Every NIM container of the catalog: page 0, then the next pages while its resultPageTotal,
    or else its resultTotal against the resources the pages gave so far (a resource given twice
    counts twice: resultTotal counts what the pages give), says more remain (an empty page does not
    end that), at most `max_pages` in all. Page 0 not answered with 200 is ngc_http_error (exit 4),
    in a shape the scripts do not read ngc_bad_response, and no answer is ngc_unreachable; a later
    page that fails ends the reading, and what was not read is counted (unread_pages, missing), as
    are the rows without a resourceId or with a blank one (unusable)."""
    seen: set[str] = set()
    resources: list[dict[str, Any]] = []
    given = 0  # resources the pages gave, before duplicates are dropped
    unusable = 0  # rows without a resourceId, or a blank one: counted in given, never checked
    total: int | None = None
    page_total: int | None = None
    page = 0
    while page < max_pages:
        try:
            status, body = get_json(catalog_query(page, page_size))
            if status != 200:
                raise _unexpected(status, body, f"page {page} of the NIM catalog")
            found = catalog_resources(body)
            if found is None:
                raise ns.ScriptError(
                    "ngc_bad_response",
                    _with_request_id(
                        f"page {page} of the NIM catalog has no results list of the CONTAINER "
                        "resources",
                        body,
                    ),
                    BAD_RESPONSE_FIX,
                    ns.EXIT_EXTERNAL,
                )
        except ns.ScriptError as exc:
            if page == 0:
                raise
            ns.log(f"page {page} of the NIM catalog was not read: {exc.message}")
            break
        given += len(found)
        for resource in found:
            rid = resource.get("resourceId")
            if not isinstance(rid, str) or not rid.strip():
                unusable += 1
            elif rid not in seen:
                seen.add(rid)
                resources.append(resource)
        total = _count(body, "resultTotal") if _count(body, "resultTotal") is not None else total
        page_total = _count(body, "resultPageTotal") or page_total
        page += 1
        more = page < page_total if page_total else total is not None and given < total
        if not more:
            break
    unread = max(page_total - page, 0) if page_total else 0
    missing = max(total - given, 0) if total is not None else 0
    return Catalog(resources, total, unread, missing, unusable)


def get_repo(resource_id: str) -> dict[str, Any] | None:
    """The public repository object, or None when NGC has no such repository (404). A 200 with
    neither a name nor a tags list, or with tags that are not a list, is `ngc_bad_response`, never
    a repository with no tags."""
    status, body = get_json("/repos/" + urllib.parse.quote(resource_id, safe="/"))
    if status == 404:
        return None
    if status != 200:
        raise _unexpected(status, body, f"repository {resource_id}")
    if not is_repo_record(body):
        raise ns.ScriptError(
            "ngc_bad_response",
            _with_request_id(
                f"NGC answered HTTP 200 for repository {resource_id} without its name or tags, "
                "or with tags that are not a list",
                body,
            ),
            BAD_RESPONSE_FIX,
            ns.EXIT_EXTERNAL,
        )
    return body


def is_repo_record(body: object) -> bool:
    """A repository record has a name or a tags list, and tags only as a list: a number cannot be
    read, and a string would be read one character at a time. With no tags (none, or null),
    choose_tag uses its latestTag."""
    if not isinstance(body, dict):
        return False
    tags = body.get("tags")
    if tags is not None and not isinstance(tags, list):
        return False
    return isinstance(body.get("name"), str) or isinstance(tags, list)


def has_envelope(body: object) -> bool:
    """An NGC error answer carries its envelope, {"requestStatus": {...}}; a proxy's or a
    firewall's page (HTML, or other JSON) does not."""
    return isinstance(body, dict) and isinstance(body.get("requestStatus"), dict)


@dataclass(frozen=True)
class Image:
    """One tag of a repository's image list: the image digest, its CPU architectures and its size
    in bytes (NGC's `size`: the largest of its architecture variants)."""

    digest: str | None
    architectures: tuple[str, ...]
    size: int | None


def real_architectures(variants: object) -> list[str]:
    """The CPU architectures of NGC `architectureVariants`, sorted. Entries whose os or
    architecture is `unknown` (signature and attestation manifests) are no CPU architecture."""
    found: set[str] = set()
    for variant in variants if isinstance(variants, list) else []:
        if not isinstance(variant, dict):
            continue
        arch, os_name = variant.get("architecture"), variant.get("os")
        if not isinstance(arch, str) or not arch.strip() or arch.strip().lower() == "unknown":
            continue
        if isinstance(os_name, str) and os_name.strip().lower() == "unknown":
            continue
        found.add(arch.strip())
    return sorted(found)


def get_images(resource_id: str) -> dict[str, Image] | None:
    """Tag -> Image for every tag of the repository (`GET /repos/{id}/images`); None when NGC
    lists no images for it (404). A NIM that is not public answers 403: `ngc_http_error`. A 200
    without an `images` list is `ngc_bad_response`, never a repository with no images.

    The repository object describes only the image tagged latestTag; this list has every tag.
    """
    status, body = get_json("/repos/" + urllib.parse.quote(resource_id, safe="/") + "/images")
    if status == 404:
        return None
    if status != 200:
        raise _unexpected(status, body, f"the images of {resource_id}")
    if not isinstance(body.get("images"), list):
        raise ns.ScriptError(
            "ngc_bad_response",
            _with_request_id(
                f"NGC answered HTTP 200 for the images of {resource_id} without an images list",
                body,
            ),
            BAD_RESPONSE_FIX,
            ns.EXIT_EXTERNAL,
        )
    return parse_images(body)


def parse_images(body: object) -> dict[str, Image]:
    found: dict[str, Image] = {}
    images = body.get("images") if isinstance(body, dict) else None
    for image in images if isinstance(images, list) else []:
        if not isinstance(image, dict) or not isinstance(image.get("tag"), str):
            continue
        digest = image.get("digest")
        size = image.get("size")
        found[image["tag"]] = Image(
            digest=digest if isinstance(digest, str) and digest else None,
            architectures=tuple(real_architectures(image.get("architectureVariants"))),
            size=int(size) if isinstance(size, int) and not isinstance(size, bool) else None,
        )
    return found


def digests_of(images: abc.Mapping[str, Image]) -> dict[str, str]:
    """Tag -> image digest, for the tags whose digest is listed."""
    return {tag: image.digest for tag, image in images.items() if image.digest}


# ---------------------------------------------------------------------------------------------
# Tag choice

_TAG = re.compile(r"^v?(\d+(?:\.\d+)*)(?:-([0-9A-Za-z][0-9A-Za-z.\-]*))?$")
_PRERELEASE = re.compile(r"^(?:rc|alpha|beta|dev|pre)(?![a-z])", re.IGNORECASE)
# Suffixes that name hardware, of a repository (`-dgx-spark`) or a tag (`1.8.0-RTX`). A suffix
# counts when it is one of these or starts with one followed by `-` (`rtx-pro-6000`).
HARDWARE_SUFFIXES = frozenset(
    {
        "dgx-spark",
        "spark",
        "rtx",
        "rtx-pro",
        "gh200",
        "gb200",
        "gb300",
        "gb10",
        "jetson",
        "igx",
        "thor",
    }
)

Digests = abc.Mapping[str, str]


def parse_tag(tag: str) -> tuple[tuple[int, ...], str] | None:
    """`1.8.0-RTX` -> ((1, 8, 0), "RTX"); `2.1.4` -> ((2, 1, 4), ""); None for `latest`."""
    match = _TAG.match(tag)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split(".")), match.group(2) or ""


def _natural(suffix: str) -> tuple[tuple[int, int, str], ...]:
    """Sort key of a tag suffix: number parts compare as numbers, so `pb6.10` > `pb6.9`."""
    return tuple(
        (0, int(part), "") if part.isdigit() else (1, 0, part.lower())
        for part in re.findall(r"[0-9]+|[A-Za-z]+", suffix)
    )


def version_key(tag: str) -> tuple[tuple[int, ...], int, tuple[tuple[int, int, str], ...]] | None:
    """Sort key of a version tag; None for tags like `latest`. `1.2.0-rc1` sorts below `1.2.0`,
    and suffixes compare part by part, numbers as numbers (`2.0.4-pb6.10` > `2.0.4-pb6.9`)."""
    parsed = parse_tag(tag)
    if parsed is None:
        return None
    numbers, suffix = parsed
    return numbers, 0 if suffix else 1, _natural(suffix)


def version_tags(tags: list[str]) -> list[str]:
    """Version tags, newest first."""
    keyed = [(key, tag) for tag in tags if (key := version_key(tag)) is not None]
    return [tag for _, tag in sorted(keyed, reverse=True)]


def is_build(tag: str) -> bool:
    """A version tag with three or more numbers (`1.7.0-variant`), which names one build; one or
    two numbers (`1`, `1.7`) is a moving alias."""
    parsed = parse_tag(tag)
    return parsed is not None and len(parsed[0]) >= 3


def is_prerelease(tag: str) -> bool:
    """`1.2.0-rc1`, `-alpha`, `-beta`, `-dev`, `-pre`: a test build, never chosen on its own."""
    parsed = parse_tag(tag)
    return parsed is not None and _PRERELEASE.match(parsed[1]) is not None


def is_hardware_suffix(suffix: str) -> bool:
    """`dgx-spark`, `rtx`, `rtx-pro-6000` -> True; `0528`, `distill-llama-70b`, `turbo` -> False."""
    lowered = suffix.lower()
    return any(lowered == hw or lowered.startswith(hw + "-") for hw in HARDWARE_SUFFIXES)


def is_general(tag: str) -> bool:
    """A version tag for any GPU: plain (`2.1.4`) or with a suffix such as `-variant`, but not a
    prerelease (`-rc1`) or a build for named hardware (`1.8.0-RTX`)."""
    parsed = parse_tag(tag)
    if parsed is None or not parsed[1]:
        return parsed is not None
    return not (is_prerelease(tag) or is_hardware_suffix(parsed[1]))


def is_alias_of(alias: str, tag: str) -> bool:
    """True when `alias` is a moving alias of the version tag `tag`: one or two numbers that start
    `tag`'s longer numbers, with no suffix or the same one (`1`, `1.7` and `1.7-variant` for
    `1.7.0-variant`; never `1.7.0`, another build). Such a tag can move to a newer build."""
    short, full = parse_tag(alias), parse_tag(tag)
    if short is None or full is None or len(short[0]) > 2 or len(short[0]) >= len(full[0]):
        return False
    return full[0][: len(short[0])] == short[0] and short[1].lower() in ("", full[1].lower())


def alias_target(alias: str, tags: list[str], digests: Digests | None = None) -> str | None:
    """The build a moving alias (`1`, `1.7`, `1.7-variant`) stands for, else None.

    With the image digests of the repository's public image list, it is the newest build with the
    alias's digest (`1.1` can be the same image as `1.10.0`); a build whose digest differs is not
    it. Without them, it is judged by the version numbers: the newest build the alias starts
    (is_alias_of), prereleases left out, and for an alias without a suffix (`1`) builds for named
    hardware too: `1` on deepseek-r1-distill-llama-8b is 1.5.2, not 1.8.0-RTX.
    """
    known = digests or {}
    builds = [tag for tag in tags if is_build(tag)]
    digest = known.get(alias)
    if digest:
        same = [tag for tag in builds if known.get(tag) == digest]
        if same:
            return version_tags(same)[0]
        builds = [tag for tag in builds if not known.get(tag)]
    parsed = parse_tag(alias)
    bare = parsed is not None and not parsed[1]
    started = [
        tag
        for tag in builds
        if is_alias_of(alias, tag) and not is_prerelease(tag) and (is_general(tag) or not bare)
    ]
    return version_tags(started)[0] if started else None


def newest_version(tags: list[str], digests: Digests | None = None) -> str | None:
    """The newest version tag, preferring tags without a suffix (`-rc1`, `-RTX`) when any exist,
    then tags for any GPU (`1.7.0-variant`, not a prerelease or a build for named hardware).

    A moving alias (`1`, `1.7`) gives way to the build it stands for (`1.7.0-variant`): the alias
    can move to another build, and a profile id belongs to one build. When a major.minor alias of
    the chosen tag's line stands for a newer build for any GPU (is_general), that build is taken:
    `1.8` points to `1.8.3-variant`, so it wins over the plain `1.8.1`. Only major.minor aliases
    count here: `1` can point to a build for one GPU (`1.8.0-RTX`). `digests` (tag -> image digest,
    from the public image list) tell what an alias stands for; without them the version numbers do.
    An alias with a prerelease's digest (`2` is the image of 2.0.0-rc1) counts as that prerelease,
    so it is chosen only when no plain tag and no tag for any GPU is left.
    """
    ordered = version_tags(tags)
    prerelease_aliases = {
        tag
        for tag in ordered
        if not is_build(tag) and is_prerelease(alias_target(tag, tags, digests) or tag)
    }
    plain = [
        tag
        for tag in ordered
        if (key := version_key(tag)) is not None and key[1] == 1 and tag not in prerelease_aliases
    ]
    general = [tag for tag in ordered if is_general(tag) and tag not in prerelease_aliases]
    candidates = plain or general or ordered
    if not candidates:
        return None
    chosen = candidates[0]
    if not is_build(chosen):
        chosen = alias_target(chosen, tags, digests) or chosen
    line = _numbers(chosen)[:2]
    for alias in tags:
        if len(_numbers(alias)) != 2 or _numbers(alias) != line:
            continue
        target = alias_target(alias, tags, digests)
        if target is not None and _numbers(target)[:2] == line and is_general(target):
            if _numbers(target) > _numbers(chosen):
                chosen = target
    return chosen


def _numbers(tag: str) -> tuple[int, ...]:
    """The version numbers of a tag; () for tags like `latest`."""
    parsed = parse_tag(tag)
    return parsed[0] if parsed else ()


def choose_tag(
    repo: dict[str, Any], requested: str | None, digests: Digests | None = None
) -> tuple[str, str, list[str]]:
    """Pick the tag to evaluate: (tag, tag_source, notes).

    tag_source is `requested`, `alias_replaced` (a moving alias such as `1.7` was asked for; the
    build it stands for is used), `newest_version` (no tag asked for), `latest_replaced` (`latest`
    asked for and a version tag exists) or `latest_tag_field` (no version tag; the repo's
    latestTag is used). `digests` (tag -> image digest, from the public image list) tell what an
    alias stands for; without them the version numbers do.
    """
    tags = [t for t in repo.get("tags") or [] if isinstance(t, str) and t]
    newest = newest_version(tags, digests)
    latest_field = repo.get("latestTag") if isinstance(repo.get("latestTag"), str) else None
    name = repo.get("name") or "this repository"
    if requested == "latest" and newest:
        return (
            newest,
            "latest_replaced",
            [
                f"`latest` replaced by the newest version tag {newest}, "
                "so the result names a fixed image"
            ],
        )
    if requested:
        if requested in tags:
            target = None if is_build(requested) else alias_target(requested, tags, digests)
            if target is None or target == requested:
                return requested, "requested", []
            digest = (digests or {}).get(requested)
            same = digest is not None and digest == (digests or {}).get(target)
            how = "the same image digest" if same else "judged by its version numbers"
            return (
                target,
                "alias_replaced",
                [
                    f"`{requested}` is a moving tag; it was replaced by {target}, the build it "
                    f"stands for ({how}), so the result names a fixed image"
                ],
            )
        shown = (version_tags(tags) or tags)[:10]
        raise ns.ScriptError(
            "tag_not_found",
            f"{name} has no tag {requested!r}; tags: {', '.join(shown) or 'none listed'}",
            "Pick one of the listed tags, or leave the tag out to use the newest version",
            ns.EXIT_PRECONDITION,
            {"tags": shown},
        )
    if newest:
        return newest, "newest_version", []
    if latest_field:
        return (
            latest_field,
            "latest_tag_field",
            [f"No version tag is listed; using the repository's latestTag {latest_field}"],
        )
    raise ns.ScriptError(
        "tag_not_found",
        f"{name} lists no tags",
        "Name the tag explicitly, or check the repository on the NGC catalog",
        ns.EXIT_PRECONDITION,
        {"tags": []},
    )


# ---------------------------------------------------------------------------------------------
# Metadata API


def get_metadata(resource_id: str, tag: str) -> dict[str, Any] | None:
    """The NIM Metadata API document for resource_id:tag; None when none is published (404).

    A 400 or 500 is a service-side problem (bad stored data); it is reported, not retried with
    other parameters. 500 is not retried at all; 502/503/504 and 429 are. A 200 without a
    `profiles` list is `ngc_bad_response`, never an image with no profiles.
    """
    query = urllib.parse.urlencode({"resourceId": resource_id, "tag": tag}, safe="/")
    status, body = get_json(f"/nim/metadata?{query}", no_retry=frozenset({500}))
    if status == 200:
        if not isinstance(body.get("profiles"), list):
            raise ns.ScriptError(
                "ngc_bad_response",
                _with_request_id(
                    f"the NIM Metadata API answered HTTP 200 for {resource_id}:{tag} without a "
                    "profiles list",
                    body,
                ),
                BAD_RESPONSE_FIX,
                ns.EXIT_EXTERNAL,
            )
        return body
    if status == 404:
        return None
    if status == 400 or status >= 500:
        raise ns.ScriptError(
            "metadata_service_error",
            _with_request_id(
                f"the NIM Metadata API answered HTTP {status} for {resource_id}:{tag}", body
            ),
            "Report it with the requestId; do not retry with other parameters.",
            ns.EXIT_EXTERNAL,
        )
    raise _unexpected(status, body, f"metadata of {resource_id}:{tag}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="NGC client for the nim-select scripts; import it, do not run it.",
        epilog='Example: python3 -I "<SKILL_DIR>/scripts/ngc_api.py" --help',
    )
    parser.parse_args(argv)
    ns.emit(True, "library", "done", {"note": "import this module; it has no command of its own"})


if __name__ == "__main__":
    main()
