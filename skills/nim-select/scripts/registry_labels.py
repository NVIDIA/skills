#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image labels on the China NIM distributors (CNDs) for nim-select: `import registry_labels`.

An image on a CND (tgcr, turingcm, lichan) is a NIM published to NGC first, but the CND gives it
its own tags (io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0 is NGC's 1.7.0-variant). The image
labels say which NGC build it is: `com.nvidia.nim.ngc_url` (an nvcr.io reference, public only under
nvcr.io/nim/) and `com.nvidia.nim.version` (the NIM version, which is the NGC tag).

The labels are read from the CND's Harbor v2 API, read-only and without credentials:
GET https://HOST/api/v2.0/projects/PROJECT/repositories/REPO/artifacts/TAG?with_label=true, where
REPO has each `/` written as `%252F` (Harbor's double encoding); they sit in
`extra_attrs.config.Labels`. A multi-arch index has no config: the image for the host architecture
(else the first platform) is read instead. read_repository lists a repository's tagged images,
newest first (one GET of .../artifacts?with_tag=true&with_label=true&q=tags%3D*&sort=-push_time),
with their labels. list_repositories lists a project's repositories. A missing repository is
`no_repository`: Harbor says so in the 404 of a tag, but lists no tags (200 []) for it, so an empty
list is checked against the project's repository list (?q=name=PROJECT/REPO); when that list cannot
be read, the listing is `unreachable`.
HOST is always a built-in Harbor host (REGISTRIES): no environment variable changes it, and no
redirect is followed (http_get_json, which the catalog reads of cnd.py use too). lichan (project
cnd) and turingcm answer without a login; tgcr answers 401, so its labels need a tgcr login.

Each read has one overall deadline (a host with many addresses retries each one, so a per-request
timeout alone can add up); nothing is raised: a 401, a 404, a timeout or a bad answer becomes a
status with the problem in words. At most MAX_REQUESTS requests are open at a time, whatever thread
sends them, and no more than MAX_ANSWER bytes of an answer are read. Stdlib only. Run directly, it
prints a notice: it has no command of its own.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_common as ns

USER_AGENT = "nim-select/0.1"
TIMEOUT = 5.0  # one request
BUDGET = 8.0  # one image's labels, an index and its child image included
LIST_BUDGET = 12.0  # a repository's tagged images, or a project's repositories
LIST_SIZE = 10  # the most recent tagged images listed
REPO_PAGES = 5  # pages of 100 repositories read from one project
MAX_CHILDREN = 6  # multi-arch indexes whose child image is read while listing
MAX_REQUESTS = 4  # requests open at a time, in every thread
MAX_ANSWER = 4_000_000  # bytes read of one answer
# A request holds a slot until its answer is read or fails, even after a deadline gave up on it:
# a read left behind still counts against MAX_REQUESTS.
_SLOTS = threading.BoundedSemaphore(MAX_REQUESTS)
VERSION_LABEL = "com.nvidia.nim.version"
NGC_URL_LABEL = "com.nvidia.nim.ngc_url"
# What a NIM image's labels call it (`vlm`, `LLM`, `Multi Modal`; `reranking`, `Riva`): the first
# of these that is set.
CATEGORY_LABELS = ("com.nvidia.nim.category", "com.nvidia.nim.type")
# Image host -> (source, host that serves the Harbor API). The tgcr-gz mirror has no API.
REGISTRIES: dict[str, tuple[str, str]] = {
    "tgcr.turing-agi.com": ("tgcr", "tgcr.turing-agi.com"),
    "tgcr-gz.turing-agi.com": ("tgcr", "tgcr.turing-agi.com"),
    "resource.turingcm.com": ("turingcm", "resource.turingcm.com"),
    "io.chancloud.com": ("lichan", "io.chancloud.com"),
}
_PUBLIC_NGC = re.compile(r"^nvcr\.io/(nim/[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*):([^:@/]+)$")
_VERSION_TAG = re.compile(r"^v?\d+(?:\.\d+)*(?:-[0-9A-Za-z][0-9A-Za-z.\-]*)?$")
# The Docker reference grammar: one path component (lowercase), and a tag.
_COMPONENT = re.compile(r"^[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*$")
_TAG = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
# A com.nvidia.nim.version value usable as an NGC tag: a tag that starts with a version number.
_LABEL_VERSION = re.compile(r"^v?\d[0-9A-Za-z_.\-]{0,127}$")
# The message of Harbor's 404 when the repository itself is missing (not just the tag).
_NO_REPOSITORY = re.compile(r"^repository \S+ not found$", re.IGNORECASE)


@dataclass(frozen=True)
class ImageRef:
    """An image on a CND, split the way the Harbor API needs it."""

    host: str
    source: str
    api_host: str
    project: str
    repository: str
    tag: str | None

    @property
    def image(self) -> str:
        return f"{self.repository_ref}:{self.tag}" if self.tag else self.repository_ref

    @property
    def repository_ref(self) -> str:
        """HOST/PROJECT/REPO, without a tag."""
        return f"{self.host}/{self.project}/{self.repository}"

    def with_tag(self, tag: str | None) -> ImageRef:
        return ImageRef(self.host, self.source, self.api_host, self.project, self.repository, tag)


@dataclass(frozen=True)
class Labels:
    """What the CND said about one image. `status` is `read`, `no_tag`, `unauthorized`,
    `not_found` (no such tag), `no_repository` or `unreachable`; `problem` says why the labels
    were not read. `architectures` are
    the CPU architectures of the image (every platform of a multi-arch index), `tags` all the tags
    of that image (the same digest), `digest` its digest and `size` its size in bytes."""

    image: str
    status: str
    version: str | None = None
    ngc_url: str | None = None
    problem: str | None = None
    architectures: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    digest: str | None = None
    size: int | None = None

    def as_facts(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "status": self.status,
            "version": self.version,
            "ngc_url": self.ngc_url,
            "problem": self.problem,
            "architectures": list(self.architectures) or None,
            "tags": list(self.tags),
            "digest": self.digest,
        }


def parse_image(text: str, tag: str | None = None) -> ImageRef | None:
    """`HOST/PROJECT/REPO[:TAG][@sha256:...]` on tgcr, turingcm or lichan; None for any other
    host, a reference without a repository, or one that breaks the Docker reference grammar (a
    catalog can hold any text). `tag` (from --tag) is used when the text has none; a digest is
    dropped, as resolve drops it."""
    path = (text or "").strip().split("@", 1)[0]
    slash, colon = path.rfind("/"), path.rfind(":")
    if colon > slash:
        path, tag = path[:colon], path[colon + 1 :] or tag
    parts = path.split("/")
    host = parts[0].lower()
    if host not in REGISTRIES or len(parts) < 3:
        return None
    names = [part.lower() for part in parts[1:]]
    if not all(_COMPONENT.match(name) for name in names) or (tag and not _TAG.match(tag)):
        return None
    source, api_host = REGISTRIES[host]
    return ImageRef(host, source, api_host, names[0], "/".join(names[1:]), tag or None)


def harbor_base(api_host: str) -> str:
    """The Harbor API root of a built-in registry host (REGISTRIES); no environment variable can
    redirect the requests."""
    return f"https://{api_host}"


def artifacts_url(ref: ImageRef) -> str:
    repo = "%252F".join(urllib.parse.quote(part, safe="") for part in ref.repository.split("/"))
    project = urllib.parse.quote(ref.project, safe="")
    return f"{harbor_base(ref.api_host)}/api/v2.0/projects/{project}/repositories/{repo}/artifacts"


def artifact_url(ref: ImageRef, reference: str) -> str:
    tag = urllib.parse.quote(reference, safe=":")
    return f"{artifacts_url(ref)}/{tag}?with_label=true"


def list_url(ref: ImageRef) -> str:
    return (
        f"{artifacts_url(ref)}?with_tag=true&with_label=true&q=tags%3D*&sort=-push_time"
        f"&page_size={LIST_SIZE}"
    )


# within(budget, work, late): work() run in a daemon thread, or late(why) when it takes longer
# than `budget` seconds or fails, so a read never raises (a thread still waiting is left behind
# and ends with the process): ns.within without reraise, which locality.py calls by this name.
within = ns.within


def http_get_json(url: str, timeout: float = TIMEOUT) -> tuple[int | None, Any, str | None]:
    """(HTTP status, JSON body, problem) of one anonymous GET: status None when nothing answered,
    body None when the answer is not JSON or larger than MAX_ANSWER; for an HTTP error, the
    problem is set and the body is the error's JSON body, if any. A redirect is such an error: it
    is never followed, so the request goes only to the host of `url`. No credentials are ever
    sent. At most MAX_REQUESTS are open at a time, whatever thread sends them."""
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}, method="GET"
    )
    ns.log(f"reading {url}")
    with _SLOTS:
        try:
            with ns.open_url(request, timeout) as response:
                status, raw = int(response.status), response.read(MAX_ANSWER + 1)
        except urllib.error.HTTPError as exc:
            try:
                error: Any = json.loads(exc.read(MAX_ANSWER + 1) or b"null")
            except (OSError, ValueError, http.client.HTTPException):
                error = None
            finally:
                exc.close()
            if 300 <= exc.code < 400:
                return int(exc.code), None, f"answered HTTP {exc.code}, a redirect (not followed)"
            return int(exc.code), error, f"answered HTTP {exc.code}"
        except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as exc:
            reason = getattr(exc, "reason", exc)
            return None, None, f"could not be read ({type(exc).__name__}: {reason})"
    if len(raw) > MAX_ANSWER:
        return status, None, f"answered with more than {MAX_ANSWER:,} bytes"
    try:
        return status, json.loads(raw), None
    except ValueError:
        return status, None, "answered with a body that is not JSON"


def fetch_json(ref: ImageRef, url: str, timeout: float = TIMEOUT) -> tuple[str, Any, str | None]:
    """(status, JSON body or None, problem) for one Harbor GET about `ref`."""
    code, body, problem = http_get_json(url, timeout)
    if code in (401, 403):
        return (
            "unauthorized",
            None,
            f"{ref.host} answered HTTP {code}: the project {ref.project!r} needs a login on "
            f"{ref.api_host}",
        )
    if code == 404:
        if any(_NO_REPOSITORY.match(message) for message in harbor_messages(body)):
            return "no_repository", None, f"{ref.host} has no repository {ref.repository_ref}"
        return "not_found", None, f"{ref.host} has no image {ref.image}"
    if code is None or problem:
        return "unreachable", None, f"{ref.host} {problem}"
    return "read", body, None


def harbor_messages(body: object) -> list[str]:
    """The messages of a Harbor error body {"errors": [{"code", "message"}]}."""
    errors = body.get("errors") if isinstance(body, dict) else None
    return [
        item["message"].strip()
        for item in errors or []
        if isinstance(item, dict) and isinstance(item.get("message"), str)
    ]


def repository_exists(ref: ImageRef) -> bool | None:
    """Whether the project of `ref` lists its repository (one GET of the repository list with
    ?q=name=PROJECT/REPO, which Harbor answers anonymously); None when that cannot be read."""
    wanted = f"{ref.project}/{ref.repository}"
    query = urllib.parse.quote(f"name={wanted}", safe="")
    url = (
        f"{harbor_base(ref.api_host)}/api/v2.0/projects/{urllib.parse.quote(ref.project, safe='')}"
        f"/repositories?q={query}&page_size=10"
    )
    status, body, _ = fetch_json(ref, url)
    if status != "read" or not isinstance(body, list):
        return None
    return any(isinstance(item, dict) and item.get("name") == wanted for item in body)


def fetch_artifact(
    ref: ImageRef, reference: str, timeout: float = TIMEOUT
) -> tuple[str, dict[str, Any] | None, str | None]:
    """(status, artifact, problem) for one tag or digest: `read`, `unauthorized`, `not_found` or
    `unreachable`."""
    status, body, problem = fetch_json(ref, artifact_url(ref, reference), timeout)
    if status == "read" and not isinstance(body, dict):
        return "unreachable", None, f"the answer of {ref.host} is not a Harbor artifact"
    return status, body, problem


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def normalise_arch(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().lower()
    return {"x86_64": "amd64", "x64": "amd64", "aarch64": "arm64", "armv8": "arm64"}.get(text, text)


def platforms(artifact: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(architecture, child digest) of each real platform of a multi-arch index; attestation
    entries (architecture `unknown`) are left out."""
    found = []
    for reference in artifact.get("references") or []:
        item = _mapping(reference)
        arch = normalise_arch(_mapping(item.get("platform")).get("architecture"))
        digest = item.get("child_digest")
        if arch and arch != "unknown" and isinstance(digest, str) and digest:
            found.append((arch, digest))
    return found


def child_digest(artifact: Mapping[str, Any], host_arch: str | None) -> str | None:
    """For a multi-arch index (no image config of its own): the digest of the image for the host
    architecture, else of the first real platform."""
    if _mapping(_mapping(artifact.get("extra_attrs")).get("config")):
        return None
    children = platforms(artifact)
    wanted = normalise_arch(host_arch)
    for arch, digest in children:
        if arch == wanted:
            return digest
    return children[0][1] if children else None


def config_labels(artifact: Mapping[str, Any]) -> dict[str, str]:
    """The image config labels, non-empty strings only."""
    config = _mapping(_mapping(artifact.get("extra_attrs")).get("config"))
    labels = _mapping(config.get("Labels"))
    return {str(k): v.strip() for k, v in labels.items() if isinstance(v, str) and v.strip()}


def nim_category(labels: Mapping[str, str]) -> str | None:
    """What the image's labels call it (CATEGORY_LABELS), or None."""
    return next((labels[key] for key in CATEGORY_LABELS if labels.get(key)), None)


def artifact_tags(artifact: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(tag["name"])
        for tag in artifact.get("tags") or []
        if isinstance(tag, dict) and isinstance(tag.get("name"), str) and tag["name"]
    )


def architectures(artifact: Mapping[str, Any]) -> tuple[str, ...]:
    """The CPU architectures of an image: its own, or every real platform of an index."""
    own = normalise_arch(_mapping(artifact.get("extra_attrs")).get("architecture"))
    if own and own != "unknown":
        return (own,)
    return tuple(sorted({arch for arch, _ in platforms(artifact)}))


def _size(artifact: Mapping[str, Any]) -> int | None:
    size = artifact.get("size")
    return size if isinstance(size, int) and not isinstance(size, bool) and size > 0 else None


def public_ngc_image(ngc_url: str | None) -> tuple[str, str] | None:
    """(resource id, tag) when an ngc_url names a public NGC NIM image with a version tag
    (nvcr.io/nim/TEAM/REPO:VERSION); None for any other path (another NGC org,
    which is not public), `latest` or anything else."""
    match = _PUBLIC_NGC.match((ngc_url or "").strip().lower())
    if match is None or not _VERSION_TAG.match(match.group(2)):
        return None
    # The tag keeps its case (`1.8.0-RTX`); the repository is lowercase on NGC.
    tag = (ngc_url or "").strip().rsplit(":", 1)[1]
    return match.group(1), tag


def label_version(value: str | None) -> tuple[str | None, bool]:
    """(NGC tag, corrected) for a com.nvidia.nim.version value: the value when it is a version tag;
    with `=` read as `-` (`1.10.0=variant`, a label typo seen on turingcm), corrected; else None
    (`latest`, empty, or anything that is not a version)."""
    text = (value or "").strip()
    if _LABEL_VERSION.match(text) and text.lower() != "latest":
        return text, False
    fixed = text.replace("=", "-")
    if fixed != text and _LABEL_VERSION.match(fixed):
        return fixed, True
    return None, False


def read_labels(ref: ImageRef, host_arch: str | None = None) -> Labels:
    """The NIM version and NGC reference labels of `ref`, or why they could not be read, within
    BUDGET seconds."""
    if not ref.tag:
        return Labels(ref.image, "no_tag", problem=f"{ref.image} names no tag")
    return within(
        BUDGET,
        lambda: _read_labels(ref, ref.tag or "", host_arch),
        lambda why: Labels(ref.image, "unreachable", problem=f"{ref.host} {why}"),
    )


def _read_labels(ref: ImageRef, tag: str, host_arch: str | None) -> Labels:
    status, artifact, problem = fetch_artifact(ref, tag)
    if artifact is None:
        return Labels(ref.image, status, problem=problem)
    tags, digest, size = artifact_tags(artifact), artifact.get("digest"), _size(artifact)
    archs = architectures(artifact)
    child = child_digest(artifact, host_arch)
    if child is not None:
        child_status, image, child_problem = fetch_artifact(ref, child)
        where = f"the image {child} of the multi-arch index {ref.image}"
        if image is None and child_status in ("not_found", "no_repository"):
            # The tag is there; only the labels of its image are not: never tag_not_found.
            missing = f"the labels of {ref.image} could not be read: {where} is missing (HTTP 404)"
            digest = digest if isinstance(digest, str) and digest else None
            return Labels(ref.image, "read", None, None, missing, archs, tags, digest, size)
        if image is None:
            return Labels(ref.image, child_status, problem=f"{where}: {child_problem}")
        artifact = image
        size = size or _size(image)
    labels = config_labels(artifact)
    version, ngc_url = labels.get(VERSION_LABEL), labels.get(NGC_URL_LABEL)
    missing = None
    if version is None and ngc_url is None:
        missing = f"{ref.image} has no {VERSION_LABEL} or {NGC_URL_LABEL} label"
    return Labels(
        ref.image,
        "read",
        version,
        ngc_url,
        missing,
        archs,
        tags,
        digest if isinstance(digest, str) and digest else None,
        size,
    )


@dataclass(frozen=True)
class TaggedImage:
    """One tagged image of a CND repository and its NIM labels (None: not set or not read).
    `unread` says why the labels were not read: `index limit`, or `its image could not be
    read`."""

    tags: tuple[str, ...]
    version: str | None = None
    ngc_url: str | None = None
    architectures: tuple[str, ...] = ()
    digest: str | None = None
    labels_read: bool = True
    category: str | None = None
    unread: str | None = None


def read_repository(
    ref: ImageRef, host_arch: str | None = None, max_children: int | None = None
) -> tuple[str, list[TaggedImage], str | None]:
    """(status, tagged images newest first, problem) of the repository of `ref` (its tag is
    ignored), within LIST_BUDGET seconds. The child image of at most `max_children` (default
    MAX_CHILDREN) multi-arch indexes is read; other indexes, and one whose child image could not
    be read, are listed without labels (labels_read False, and `unread` says which)."""
    limit = MAX_CHILDREN if max_children is None else max_children
    return within(
        LIST_BUDGET,
        lambda: _read_repository(ref, host_arch, limit),
        lambda why: ("unreachable", [], f"{ref.host} {why}"),
    )


def _read_repository(
    ref: ImageRef, host_arch: str | None, max_children: int
) -> tuple[str, list[TaggedImage], str | None]:
    status, body, problem = fetch_json(ref, list_url(ref))
    if status != "read":
        return status, [], problem
    if not isinstance(body, list):
        return "unreachable", [], f"the answer of {ref.host} is not a Harbor artifact list"
    if not body:  # Harbor lists no tags of a missing repository
        exists = repository_exists(ref)
        if exists is None:
            problem = (
                f"{ref.host} lists no tagged image of {ref.repository_ref}, and whether it has "
                "that repository could not be read"
            )
            return "unreachable", [], problem
        if not exists:
            return "no_repository", [], f"{ref.host} has no repository {ref.repository_ref}"
    tagged = [(artifact, artifact_tags(artifact)) for artifact in map(_mapping, body)]
    tagged = [(artifact, tags) for artifact, tags in tagged if tags]
    wanted = [child for artifact, _ in tagged if (child := child_digest(artifact, host_arch))]
    asked = list(dict.fromkeys(wanted))[:max_children]
    children = read_children(ref, asked)
    images: list[TaggedImage] = []
    for artifact, tags in tagged:
        archs, digest = architectures(artifact), artifact.get("digest")
        digest = digest if isinstance(digest, str) and digest else None
        child = child_digest(artifact, host_arch)
        if child is not None and child not in children:
            unread = "its image could not be read" if child in asked else "index limit"
            images.append(TaggedImage(tags, None, None, archs, digest, False, unread=unread))
            continue
        labels = config_labels(children[child] if child is not None else artifact)
        images.append(
            TaggedImage(
                tags,
                labels.get(VERSION_LABEL),
                labels.get(NGC_URL_LABEL),
                archs,
                digest,
                category=nim_category(labels),
            )
        )
    return "read", images, None


def read_children(ref: ImageRef, digests: list[str]) -> dict[str, dict[str, Any]]:
    """The child images of multi-arch indexes that could be read (one that could not is left out,
    so its index is listed without labels), read at the same time: one after the other, a slow
    registry would spend the listing's budget on them."""
    found: dict[str, dict[str, Any]] = {}

    def read(digest: str) -> None:
        image = fetch_artifact(ref, digest)[1]
        if image is not None:
            found[digest] = image

    threads = [threading.Thread(target=read, args=(digest,), daemon=True) for digest in digests]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return {digest: found[digest] for digest in digests if digest in found}


def list_repositories(
    host: str, project: str, budget: float = LIST_BUDGET
) -> tuple[str, list[str], str | None]:
    """(status, repository paths such as `cnd/qwen/qwen3.5-35b-a3b`, problem) of one Harbor
    project on the CND image host `host`, within `budget` seconds. At most REPO_PAGES pages of 100
    are read: when every one is full, the status is still `read`, and the problem says that the
    project may list more."""
    source, api_host = REGISTRIES[host]
    ref = ImageRef(host, source, api_host, project, "", None)

    def work() -> tuple[str, list[str], str | None]:
        names: list[str] = []
        base = harbor_base(api_host)
        for page in range(1, REPO_PAGES + 1):
            url = (
                f"{base}/api/v2.0/projects/{urllib.parse.quote(project, safe='')}/repositories"
                f"?page_size=100&page={page}"
            )
            status, body, problem = fetch_json(ref, url)
            if status != "read":
                if status == "not_found":
                    problem = f"{host} has no project {project!r}"
                return status, [], problem
            if not isinstance(body, list):
                return "unreachable", [], f"the answer of {host} is not a Harbor repository list"
            names.extend(
                str(item["name"])
                for item in body
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            )
            if len(body) < 100:
                return "read", names, None
        problem = (
            f"only the first {REPO_PAGES * 100} repositories of project {project!r} on {host} "
            "were read: it may list more"
        )
        return "read", names, problem

    return within(budget, work, lambda why: ("unreachable", [], f"{host} {why}"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Image label reader of the China NIM distributors for the nim-select "
        "scripts; import it, do not run it.",
        epilog='Example: python3 -I "<SKILL_DIR>/scripts/registry_labels.py" --help',
    )
    parser.parse_args(argv)
    ns.emit(True, "library", "done", {"note": "import this module; it has no command of its own"})


if __name__ == "__main__":
    main()
