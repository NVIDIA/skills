#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""China NIM distributors (CNDs) for nim-select: `import cnd`.

The source of a NIM image is `ngc` or one CND:
  tgcr      images on tgcr.turing-agi.com and its public mirror tgcr-gz.turing-agi.com; catalog TGC
  turingcm  images on resource.turingcm.com; catalog TuringCM
  lichan    images on io.chancloud.com; catalog: its Harbor (project cnd)

The three CNDs have different APIs and list different NIMs, so each has its own small reader behind
one interface (Reader): `search` finds a model name in that CND's catalog, `list_all` lists the
whole catalog (for the list command), `image` reads what the CND says about one tagged image (its
labels), and `images` lists the tagged images of a repository, newest first. Only the reader of the
CND the user picked is used: nothing here searches another CND or NGC, and a name that CND does not
list is model_not_found, with the other CNDs to offer.

This module also decides the source of a model reference, and names the NGC builds a CND image may
be: from its com.nvidia.nim.ngc_url label or the TGC catalog's ngc_url (a public nvcr.io/nim
reference with a version tag), from its com.nvidia.nim.version label (the repository then comes
from the CND path), or from a version the user gives. It never maps an image by its name and tag
alone: CND tags are not NGC tags. nimselect.py checks each NGC build on NGC. Public, anonymous GET
requests only; nothing logs in. The catalog and Harbor hosts are built in (TGC_BASE,
TURINGCM_BASE, registry_labels.REGISTRIES): no environment variable changes them, and no redirect
is followed (registry_labels.http_get_json). Stdlib only. Run directly, it prints a notice: it has
no command of its own.
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_common as ns
import registry_labels as rl

SOURCES = ("ngc", "tgcr", "turingcm", "lichan")
CNDS = ("tgcr", "turingcm", "lichan")
NGC_HOST = "nvcr.io"
TGC_BASE = "https://tgc.turing-agi.com"  # the TGC catalog
TURINGCM_BASE = "https://nim.turingcm.com"  # the TuringCM catalog
LICHAN_HOST, LICHAN_PROJECT = "io.chancloud.com", "cnd"
CATALOG_TIMEOUT = 10.0  # one catalog request
SEARCH_BUDGET = 30.0  # one catalog search, its detail requests included
MAX_MATCHES = 10  # matches whose image is looked up (one detail request or tag list each)
PARALLEL = rl.MAX_REQUESTS  # requests at a time (registry_labels holds every request to it)
TGC_PAGES = 3  # pages of 100 catalog items
# The note on a tgcr tag looked up when a request for the other TGC items failed or timed out.
NOT_ALL_READ = "the other catalog items could not all be read"
# Projects stripped from a CND path to find the NGC TEAM/REPO (then a leading `nim/` too).
CND_PROJECTS = ("cnd/", "cnd-enterprise/", "public/")
_TURINGCM_IMAGE = re.compile(r"resource\.turingcm\.com/[^\s<>\"'`\\]+")
T = TypeVar("T")
# (status, tagged images newest first, problem) of a Harbor repository, as registry_labels reads it.
Listing = tuple[str, list[rl.TaggedImage], str | None]

UNREACHABLE_FIX = (
    "Check network access to {host} (the China NIM distributors can be slow or blocked outside "
    "China; check HTTPS_PROXY), then run the command again once; never switch to NGC or another "
    "China NIM distributor on your own (the user can choose --source)"
)
LOGIN_FIX = (
    "Do not retry: {catalog} answers only after a login, which nim-select never does. Ask the "
    "user for the full image (HOST/PROJECT/REPO:TAG) and its NGC version (--nim-version), or "
    "offer to confirm the profile after the pull with the image's own list-model-profiles, which "
    "nim-deploy runs; never log in, and never switch to NGC or another China NIM distributor on "
    "your own"
)
# The fix of a list whose catalog needs a login: the user named no image, so none is asked for.
LIST_LOGIN_FIX = (
    "Do not retry: {catalog} answers only after a login, which nim-select never does, so its "
    "models cannot be listed. Tell the user so, and offer to check an image they name "
    "(HOST/PROJECT/REPO:TAG) with select; never log in, and never switch to NGC or another China "
    "NIM distributor on your own"
)


class LoginNeeded(ns.ScriptError):
    """cnd_unreachable for a catalog that answers only after a login: a retry cannot help."""


# --- the source -------------------------------------------------------------------------------


def reference_host(model: str) -> str | None:
    """The registry host a model reference starts with, lowercased; None when it names none."""
    raw = (model or "").strip()
    if "/" not in raw or any(ch.isspace() for ch in raw):
        return None
    first = raw.split("/", 1)[0]
    if "." in first or ":" in first or first == "localhost":
        return first.lower()
    return None


def host_source(host: str | None) -> str | None:
    """`ngc` for nvcr.io, the CND of a CND image host, `mirror` for another registry (checked as
    the NGC image), None for no host."""
    if host is None:
        return None
    name = host.split(":", 1)[0]
    if name == NGC_HOST:
        return "ngc"
    if name in rl.REGISTRIES:
        return rl.REGISTRIES[name][0]
    return "mirror"


def describe(source: str) -> str:
    if source == "ngc":
        return "NGC"
    if source == "mirror":
        return "a registry checked as NGC"
    return f"the China NIM distributor {source} ({CATALOGS[source]})"


def choose_source(model: str, flag: str | None) -> str:
    """The source of MODEL: its host's, else --source, else ngc. A host that contradicts --source
    is bad input (exit 2): the reference already says where the image comes from."""
    host = reference_host(model)
    named = host_source(host)
    if named is None:
        return flag or "ngc"
    implied = "ngc" if named == "mirror" else named
    if flag and flag != implied:
        raise ns.ScriptError(
            "source_conflict",
            f"{model} is on {host}, {describe(named)}, but --source is {flag}",
            "Drop --source (the host names the source), or ask the user which one they mean and "
            "pass the model without a registry host",
        )
    return implied


def other_sources(source: str) -> list[str]:
    """The CNDs other than `source`: the ones to offer when `source` lacks a model."""
    return [name for name in CNDS if name != source]


def not_on_cnd(source: str, message: str, first: str, facts: dict[str, Any]) -> ns.ScriptError:
    """model_not_found (exit 3) on one CND. Each CND lists other NIMs, so the user is offered the
    other two; nothing searches them, or NGC, on its own."""
    others = other_sources(source)
    offer = " or ".join(f"{CATALOGS[name]} (--source {name})" for name in others)
    return ns.ScriptError(
        "model_not_found",
        message,
        f"{first}. Tell the user {CATALOGS[source]} lists no such NIM and offer to search the "
        f"other China NIM distributors, {offer} (facts.other_sources), with the model name "
        "facts.query, never the image: search one only when the user says so, never on your own, "
        "and never NGC unless the user asks for it",
        ns.EXIT_PRECONDITION,
        {**facts, "source": source, "other_sources": others},
    )


def repository_missing(source: str, ref: rl.ImageRef, message: str) -> ns.ScriptError:
    """model_not_found for an image whose repository the CND `source` lacks: the other CNDs are
    searched by the repository name (facts.query), as an image names its own host."""
    return not_on_cnd(
        source,
        message,
        "Check the image name with the user",
        {"image": ref.image, "query": ref.repository.rsplit("/", 1)[-1]},
    )


# --- what a CND says --------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogTag:
    """One image tag of a TGC catalog item: its image reference, NGC reference, CPU architecture,
    digest and creation time, as the catalog gives them."""

    tag: str
    image: str
    ngc_url: str | None = None
    arch: str | None = None
    digest: str | None = None
    created: str = ""

    def as_facts(self) -> dict[str, Any]:
        return {
            "catalog": "TGC",
            "image": self.image,
            "ngc_url": self.ngc_url,
            "arch": self.arch,
            "digest": self.digest,
        }


@dataclass
class Found:
    """One catalog match: the image the catalog gives for it (HOST/PROJECT/REPO:TAG) and what
    the catalog says (`arch`, `model_type`, and `devices`: the GPUs it lists the item for, which
    is information, never a fit). `tags` (TGC) lists its image tags newest first; `listing`
    (lichan) is the repository's tagged images read during the search. `tag_listed` is False when
    the image's tag is not from a tag list (a TuringCM deploy text names `latest`); `also` are the
    other images the same catalog item names (a TuringCM deploy text can name a DGX Spark build
    too). `hits` are the matches of the search that found it (pick): a tgcr tag it does not list
    is looked up in their tags, never by the same search again."""

    source: str
    name: str
    image: str | None
    catalog_ngc_url: str | None = None
    architectures: tuple[str, ...] = ()
    model_type: str | None = None
    item_id: str | None = None
    tags: list[CatalogTag] = field(default_factory=list)
    listing: Listing | None = None
    texts: tuple[str, ...] = ()
    tag_listed: bool = True
    also: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    hits: tuple[Found, ...] = ()

    def candidate(self) -> dict[str, Any]:
        image = rl.parse_image(self.image or "")
        return {
            "source": self.source,
            "catalog_name": self.name,
            "image": self.image,
            # what to rerun with: a tag is added only when the user names one
            "repository": image.repository_ref if image else None,
            "newest_tag": image.tag if image and self.tag_listed else None,
            "architectures": list(self.architectures) or None,
            "model_type": self.model_type,
        }


@dataclass
class CndImage:
    """One image on a CND and what is known about it: from its labels (Harbor) or from the TGC
    catalog. `tags` are all the tags of the same image, when known; `problems` say why it names
    no NGC build by itself; `category` is what its labels call it (registry_labels.nim_category),
    when they were read."""

    ref: rl.ImageRef
    version: str | None = None
    ngc_url: str | None = None
    catalog_ngc_url: str | None = None
    architectures: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    digest: str | None = None
    size: int | None = None
    labels: dict[str, Any] | None = None
    catalog: dict[str, Any] | None = None
    problems: list[str] = field(default_factory=list)
    category: str | None = None

    @property
    def pinned(self) -> str | None:
        """The most specific tag of this image that is not `latest`."""
        return pinned_tag(self.tags or ((self.ref.tag,) if self.ref.tag else ()))


def listed_image(ref: rl.ImageRef, item: rl.TaggedImage) -> CndImage:
    """A tagged image of the repository of `ref`, as its Harbor lists it."""
    shown = ref.with_tag(pinned_tag(item.tags) or item.tags[0])
    problems = []
    if not item.labels_read:
        why = f" ({item.unread})" if item.unread else ""
        problems.append(f"the labels of {shown.image} were not read{why}")
    elif item.version is None and item.ngc_url is None:
        problems.append(f"{shown.image} has no {rl.VERSION_LABEL} or {rl.NGC_URL_LABEL} label")
    return CndImage(
        shown,
        item.version,
        item.ngc_url,
        None,
        item.architectures,
        item.tags,
        item.digest,
        labels={
            "status": "read" if item.labels_read else "not_read",
            "version": item.version,
            "ngc_url": item.ngc_url,
            "architectures": list(item.architectures) or None,
            "tags": list(item.tags),
            "digest": item.digest,
        },
        problems=problems,
        category=item.category,
    )


# --- helpers for the readers ------------------------------------------------------------------


def query_words(query: str) -> list[str]:
    """The words of a model name, lowercased. TEAM/REPO is searched as REPO; a tag is dropped."""
    text = (query or "").strip().lower()
    if not any(ch.isspace() for ch in text):
        text = text.rsplit("/", 1)[-1].split(":", 1)[0]
    return [word for word in text.split() if word]


def _norm(text: str) -> str:
    return "-".join(text.strip().lower().split())


def matches(words: Sequence[str], texts: Sequence[str]) -> bool:
    """ns.every_word_in with its default word test, ns.in_text; no words match nothing."""
    return bool(words) and ns.every_word_in(words, texts)


def exact(words: Sequence[str], texts: Sequence[str]) -> bool:
    """The words, joined by `-`, are one of the texts (or a text's last path part)."""
    wanted = "-".join(words)
    for text in texts:
        if _norm(text) == wanted or _norm(text.rsplit("/", 1)[-1].split(":", 1)[0]) == wanted:
            return True
    return False


def deadline(budget: float, work: Callable[[], T], late: Callable[[], T]) -> T:
    """ns.within with `reraise`, and a late() that takes no reason."""
    return ns.within(budget, work, lambda _why: late(), reraise=True)


def run_parallel(tasks: Sequence[Callable[[], T]], workers: int = PARALLEL) -> list[T]:
    """Run the tasks, at most `workers` at a time, in daemon threads; results in task order. The
    first exception is raised again once all have ended (each task has its own deadline)."""
    results: list[Any] = [None] * len(tasks)
    errors: list[tuple[int, BaseException]] = []
    lock = threading.Lock()
    pending = iter(range(len(tasks)))

    def worker() -> None:
        while True:
            with lock:
                index = next(pending, None)
            if index is None:
                return
            try:
                results[index] = tasks[index]()
            except BaseException as exc:
                with lock:
                    errors.append((index, exc))

    threads = [
        threading.Thread(target=worker, daemon=True) for _ in range(min(workers, len(tasks)))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise min(errors, key=lambda pair: pair[0])[1]
    return results


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _records(body: Any, *path: str) -> list[dict[str, Any]] | None:
    """body[path...] when it is a list of objects (a dict `records` list is accepted too)."""
    value = body
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    if isinstance(value, dict):
        value = value.get("records")
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, dict)]


def _archs(*values: object) -> tuple[str, ...]:
    """The first catalog `arch` value that names one, as a tuple."""
    for value in values:
        arch = rl.normalise_arch(value)
        if arch and arch != "unknown":
            return (arch,)
    return ()


def _total(body: Any) -> int | None:
    """data.total of a catalog page, when it is a count."""
    data = body.get("data") if isinstance(body, dict) else None
    total = data.get("total") if isinstance(data, dict) else None
    return total if isinstance(total, int) and not isinstance(total, bool) else None


def exact_first(words: Sequence[str], items: list[T], text: Callable[[T], str]) -> list[T]:
    """The items whose text the words name exactly first, the others after, each in its order:
    an exact name is looked up even past MAX_MATCHES other names."""
    return sorted(items, key=lambda item: not exact(words, (text(item),)))


# One GPU as a catalog names it (`B200*4`, `H100-80GB-HBM3*1`, `Generic GPU*8`, `DGX Spark`,
# `DGX Spark (two-node)`, `通用`): at most three words of letters, digits and `*+.-()`, at most 32
# characters. The text is the catalog's, shown as data; anything longer or of other characters is
# left out.
_DEVICE = re.compile(r"^[\w*+.()\-]+(?: [\w*+.()\-]+){0,2}$")
DEVICE_MAX = 32


def _devices(value: object) -> tuple[str, ...]:
    """The GPUs a catalog lists an item for: a TGC `supported_devices` text (`B200*4,H200*8`) or
    a TuringCM `badges` list, as they are written (`通用` is TuringCM's word for any GPU). An
    entry that is no short GPU name (_DEVICE) is left out."""
    items = value.split(",") if isinstance(value, str) else value if isinstance(value, list) else []
    return tuple(
        dict.fromkeys(
            text
            for item in items
            if (text := _text(item)) and len(text) <= DEVICE_MAX and _DEVICE.match(text)
        )
    )


# Whether the list wants a catalog item, by its name and the catalog's model_type: an item it
# does not want gets no detail request.
Wanted = Callable[[str, "str | None"], bool]


@dataclass
class Catalog:
    """A CND's whole catalog, for the list command: one Found per image the catalog names, the
    wanted items whose detail names no image, the items not wanted, as (name, model_type), the
    wanted items whose detail could not be read, with why (they were not checked), and `unread`:
    None, or why the catalog was read only in part (it lists more than the pages a list reads, and
    the items past them are not checked; listing again reads the same pages)."""

    found: list[Found]
    no_image: list[str]
    left_out: list[tuple[str, str | None]] = field(default_factory=list)
    failed: list[tuple[str, ns.ScriptError]] = field(default_factory=list)
    unread: str | None = None


def _wanted(
    items: list[dict[str, Any]], wanted: Wanted | None
) -> tuple[list[dict[str, Any]], list[tuple[str, str | None]]]:
    """(the catalog items `wanted` keeps, the (name, model_type) of the others)."""
    kept, left_out = [], []
    for item in items:
        name = _text(item.get("name")) or str(item.get("id"))
        kind = _text(item.get("model_type")) or None
        if wanted is None or wanted(name, kind):
            kept.append(item)
        else:
            left_out.append((name, kind))
    return kept, left_out


# --- one reader per CND -----------------------------------------------------------------------


class Reader:
    """The interface every CND reader has. Each CND has its own API and lists its own NIMs, so
    each has its own small reader; only the reader of the CND the user picked is used.

    search(words): the catalog names that contain every word, best first, with their images, and
    how many names matched (only the first MAX_MATCHES have their image looked up). A catalog
    that lists more than a search reads (TGC, Lichan) is cnd_unreachable, with no image looked up
    (not_searched_in_full).
    list_all(wanted): every image of the catalog (the list command), as search gives a match;
    `wanted` (name, model_type) leaves an item out before any request about it.
    image(ref, found, host_arch): what the CND says about the tagged image `ref`.
    images(ref, found, host_arch): the tagged images of the repository of `ref`, newest first, and
    why the list is empty or incomplete. `found` is the catalog match `ref` came from, if any.
    """

    source: ClassVar[str]
    catalog: ClassVar[str]  # the name users know
    catalog_host: ClassVar[str]  # the host a search reads

    def search(self, words: list[str]) -> tuple[list[Found], int]:
        raise NotImplementedError

    def list_all(self, wanted: Wanted | None = None) -> Catalog:
        raise NotImplementedError

    def image(self, ref: rl.ImageRef, found: Found | None, host_arch: str | None) -> CndImage:
        raise NotImplementedError

    def images(
        self, ref: rl.ImageRef, found: Found | None, host_arch: str | None
    ) -> tuple[list[CndImage], str | None]:
        raise NotImplementedError

    def unreachable(self, problem: str, host: str | None = None) -> ns.ScriptError:
        """cnd_unreachable (exit 4): the catalog, or `host`, did not answer as expected."""
        where = host or self.catalog_host
        return ns.ScriptError(
            "cnd_unreachable",
            f"the {self.catalog} catalog ({where}) could not be read: {problem}",
            UNREACHABLE_FIX.format(host=where),
            ns.EXIT_EXTERNAL,
        )

    def needs_login(self, problem: str, host: str | None = None) -> ns.ScriptError:
        """cnd_unreachable (exit 4) for a catalog that answers 401 or 403: a retry cannot help."""
        where = host or self.catalog_host
        return LoginNeeded(
            "cnd_unreachable",
            f"the {self.catalog} catalog ({where}) needs a login, which nim-select never does: "
            f"{problem}",
            LOGIN_FIX.format(catalog=f"the {self.catalog} catalog"),
            ns.EXIT_EXTERNAL,
        )

    def not_searched_in_full(
        self, words: Sequence[str], problem: str, named: Sequence[str], pages: str, host: str
    ) -> ns.ScriptError:
        """cnd_unreachable (exit 4) for a search of a catalog that lists more than a search reads
        (`problem`): a name past the part read may be the one asked for, or the same as a name
        read (`named`: those the words name exactly), so neither model_not_found nor a match
        would be sure. A search reads the same `pages` again: the fix asks for the full image on
        `host`."""
        name, seen = "-".join(words), ", ".join(dict.fromkeys(named))
        read = (
            f"one past them may be named {name!r} too (read: {seen})"
            if named
            else f"none of those read is named {name!r}"
        )
        return ns.ScriptError(
            "cnd_unreachable",
            f"the {self.catalog} catalog ({self.catalog_host}) was not searched in full: "
            f"{problem}, and {read}",
            f"Do not retry: a search reads the same {pages}. Ask the user for the full image on "
            f"{host} (HOST/PROJECT/REPO:TAG) and run again with it; never switch to NGC or "
            "another China NIM distributor on your own (the user can choose --source)",
            ns.EXIT_EXTERNAL,
        )

    def catalog_json(self, url: str) -> Any:
        """The JSON body of one catalog GET; cnd_unreachable (exit 4) for anything else."""
        code, body, problem = rl.http_get_json(url, CATALOG_TIMEOUT)
        host = urllib.parse.urlsplit(url).netloc
        if code in (401, 403):
            raise self.needs_login(f"{host} {problem}", host)
        if code is None or problem or body is None:
            raise self.unreachable(problem or "no answer", host)
        return body

    def details(self, lookups: list[Callable[[], Found | list[Found] | None]]) -> list[Found]:
        """The matches whose detail could be read, at most PARALLEL requests at a time. One that
        fails, or answers in a shape the scripts do not read, is left out (logged); when every
        one fails, or none of the others names an image, the first error is raised: a failed
        lookup is never reported as a name with no image."""
        results = self.detail_results(lookups)
        found: list[Found] = []
        for item in results:
            if isinstance(item, Found):
                found.append(item)
            elif isinstance(item, list):
                found.extend(item)
        failed = [item for item in results if isinstance(item, ns.ScriptError)]
        if failed and not found:
            raise failed[0]
        return found

    def detail_results(
        self, lookups: list[Callable[[], Found | list[Found] | None]]
    ) -> list[Found | list[Found] | ns.ScriptError | None]:
        """What each lookup gave, in order: its matches, None (no image), or the ScriptError it
        failed with; at most PARALLEL requests at a time. When every one fails, the first error
        is raised."""

        def safe(
            lookup: Callable[[], Found | list[Found] | None],
        ) -> Found | list[Found] | ns.ScriptError | None:
            try:
                return lookup()
            except ns.ScriptError as exc:
                ns.log(f"a {self.catalog} catalog detail was not read: {exc.message}")
                return exc
            except Exception as exc:  # an odd answer must not fail the command
                ns.log(f"a {self.catalog} catalog detail was not read: {exc!r}")
                return self.unreachable("a detail answered in a shape the scripts do not read")

        results = run_parallel([lambda lookup=lookup: safe(lookup) for lookup in lookups])
        failed = [item for item in results if isinstance(item, ns.ScriptError)]
        if failed and len(failed) == len(results):
            raise failed[0]
        return results

    def catalog_of(
        self,
        items: list[dict[str, Any]],
        left_out: list[tuple[str, str | None]],
        detail: Callable[[dict[str, Any]], Found | list[Found] | None],
    ) -> Catalog:
        """The list command's Catalog of the catalog items it wants: one detail each (detail),
        at most PARALLEL at a time. An item whose detail names no image is `no_image`; one whose
        detail failed is `failed`, with why (it was not checked)."""
        results = self.detail_results([lambda item=item: detail(item) for item in items])
        found: list[Found] = []
        no_image: list[str] = []
        failed: list[tuple[str, ns.ScriptError]] = []
        for item, result in zip(items, results):
            name = _text(item.get("name")) or str(item.get("id"))
            if isinstance(result, ns.ScriptError):
                failed.append((name, result))
            elif isinstance(result, Found):
                found.append(result)
            elif result:
                found.extend(result)
            else:
                no_image.append(name)
        return Catalog(found, no_image, left_out, failed)


class HarborReader(Reader):
    """A CND whose images are read through its Harbor v2 API (registry_labels): an image's labels,
    and a repository's tagged images. When Harbor answers 401 (a login is needed), the image says
    nothing by itself; the TGC reader asks its catalog instead."""

    def image(self, ref: rl.ImageRef, found: Found | None, host_arch: str | None) -> CndImage:
        labels = rl.read_labels(ref, host_arch)
        if labels.status == "read":
            return CndImage(
                ref,
                labels.version,
                labels.ngc_url,
                None,
                labels.architectures,
                labels.tags,
                labels.digest,
                labels.size,
                labels.as_facts(),
                None,
                [labels.problem] if labels.problem else [],
            )
        if labels.status == "no_repository":
            raise repository_missing(self.source, ref, str(labels.problem))
        if labels.status == "not_found":
            raise ns.ScriptError(
                "tag_not_found",
                labels.problem or f"{ref.host} has no image {ref.image}",
                "Check the tag with the user, or leave it out to use the newest "
                f"{self.catalog} tag with a known NGC build",
                ns.EXIT_PRECONDITION,
                {"source": self.source, "image": ref.image},
            )
        if labels.status != "unauthorized":
            raise ns.ScriptError(
                "cnd_unreachable",
                f"the labels of {ref.image} could not be read: {labels.problem}",
                UNREACHABLE_FIX.format(host=ref.api_host),
                ns.EXIT_EXTERNAL,
            )
        return self.image_behind_login(ref, found, labels)

    def image_behind_login(
        self, ref: rl.ImageRef, found: Found | None, labels: rl.Labels
    ) -> CndImage:
        """The image when its labels need a login: only why nothing is known."""
        return CndImage(ref, labels=labels.as_facts(), problems=[str(labels.problem)])

    def listing(self, ref: rl.ImageRef, found: Found | None, host_arch: str | None) -> Listing:
        return rl.read_repository(ref, host_arch)

    def images(
        self, ref: rl.ImageRef, found: Found | None, host_arch: str | None
    ) -> tuple[list[CndImage], str | None]:
        status, tagged, problem = self.listing(ref, found, host_arch)
        if status == "unreachable":  # a reset, a 5xx or a slow answer: once more
            ns.log(f"the tags of {ref.repository_ref} were not read ({problem}): once more")
            status, tagged, problem = self.listing(ref, found, host_arch)
        if status == "read":
            images = [listed_image(ref, item) for item in tagged]
            return images, None if images else f"{ref.repository_ref} lists no tagged image"
        if status in ("no_repository", "not_found"):
            raise repository_missing(
                self.source, ref, f"{ref.host} has no repository {ref.repository_ref}"
            )
        if status != "unauthorized":
            raise ns.ScriptError(
                "cnd_unreachable",
                f"the tags of {ref.repository_ref} could not be read: {problem}",
                UNREACHABLE_FIX.format(host=ref.api_host),
                ns.EXIT_EXTERNAL,
            )
        return self.images_behind_login(ref, found, str(problem))

    def images_behind_login(
        self, ref: rl.ImageRef, found: Found | None, problem: str
    ) -> tuple[list[CndImage], str | None]:
        """The tagged images when the repository needs a login: none, and why."""
        return [], problem


class TgcReader(HarborReader):
    """tgcr. The TGC catalog (tgc.turing-agi.com) lists its NIMs: a search, then one detail per
    match for its image tags (docker_url, ngc_url, arch, digest). Its Harbor (tgcr.turing-agi.com)
    answers 401 without a login, so an image's tags and NGC reference come from the catalog."""

    source, catalog, catalog_host = "tgcr", "TGC", "tgc.turing-agi.com"

    def items(self, base: str, word: str | None) -> tuple[list[dict[str, Any]], bool]:
        """The catalog items whose name has `word` (TGC searches names only), or every item for
        None, and whether the catalog lists more than were read: pages of 100, read on while the
        catalog's total says more, at most TGC_PAGES."""
        items: list[dict[str, Any]] = []
        for page in range(1, TGC_PAGES + 1):
            query = urllib.parse.urlencode(
                {"page": page, "size": 100, **({"search": word} if word else {})}
            )
            body = self.catalog_json(f"{base}/tgc/api/resources?{query}")
            found = _records(body, "data", "items")
            if found is None:
                raise self.unreachable("no items list", urllib.parse.urlsplit(base).netloc)
            items.extend(found)
            total = _total(body)
            more = len(items) < total if total is not None else len(found) >= 100
            if not found or not more:
                return items, False
        ns.log(f"the TGC catalog lists more than its first {len(items)} items: the rest are unread")
        return items, True

    def search(self, words: list[str], in_part: bool = False) -> tuple[list[Found], int]:
        """Reader.search. When more items have the first word than TGC_PAGES pages hold, an item
        past them may be the one asked for, or have the name of one read: cnd_unreachable, with no
        detail asked for. `in_part` takes the items read instead: catalog_tags finds an image by
        its repository, not its name, and reads the catalog's first TGC_PAGES pages when those
        lack it (an image past them stays unknown)."""
        base = TGC_BASE
        items, more = self.items(base, words[0])
        # Names only: a catalog `model_name` can name another model (seen on TGC).
        hits = [item for item in items if matches(words, (_text(item.get("name")),))]
        hits = exact_first(words, hits, lambda item: _text(item.get("name")))
        if more and not in_part:
            names = [_text(item.get("name")) for item in hits]
            raise self.not_searched_in_full(
                words,
                f"it lists more items with {words[0]!r} in their name than the {len(items)} a "
                "search reads",
                [name for name in names if exact(words, (name,))],
                f"first {len(items)} items",
                "tgcr-gz.turing-agi.com or tgcr.turing-agi.com",
            )
        lookups = [lambda item=item: self.detail(base, item) for item in hits[:MAX_MATCHES]]
        return self.details(lookups), len(hits)

    def list_all(self, wanted: Wanted | None = None) -> Catalog:
        """Every catalog item (pages of 100, at most TGC_PAGES) with its image tags: one detail
        request for each item `wanted` keeps, at most PARALLEL at a time. When the catalog lists
        more, the items past those pages are not checked (Catalog.unread): the list is partial."""
        base = TGC_BASE
        listed, more = self.items(base, None)
        items, left_out = _wanted(listed, wanted)
        catalog = self.catalog_of(items, left_out, lambda item: self.detail(base, item))
        if more:
            catalog.unread = (
                f"it lists more than the {len(listed)} items of its first {TGC_PAGES} pages of 100"
            )
        return catalog

    def detail(self, base: str, item: dict[str, Any]) -> Found | None:
        item_id = item.get("id")
        body = self.catalog_json(f"{base}/tgc/api/resources/{urllib.parse.quote(str(item_id))}")
        detail = body.get("data") if isinstance(body, dict) else None
        if not isinstance(detail, dict):  # an error in a 200 answer: {"code": 500, "data": null}
            code = body.get("code") if isinstance(body, dict) else None
            raise self.unreachable(f"item {item_id} answered without its details (code {code})")
        tags = tgc_tags(detail)
        if not tags:
            return None
        pinned = [tag for tag in tags if tag.tag.lower() != "latest"]
        name = _text(detail.get("name")) or _text(item.get("name"))
        return Found(
            source=self.source,
            name=name,
            image=(pinned or tags)[0].image,
            catalog_ngc_url=_text(detail.get("ngc_url")) or None,
            architectures=_archs(detail.get("arch"), item.get("arch")),
            model_type=_text(detail.get("model_type")) or _text(item.get("model_type")) or None,
            item_id=str(item_id),
            tags=tags,
            texts=(name, *(tag.image for tag in tags)),
            devices=_devices(detail.get("supported_devices") or item.get("supported_devices")),
        )

    def catalog_tags(
        self, ref: rl.ImageRef, found: Found | None, tag: str | None = None
    ) -> tuple[list[CatalogTag], str]:
        """The catalog's image tags of the tgcr repository of `ref` (either tgcr host), newest
        first, and where they came from. First the items already read: the match `ref` was found
        as and the other matches of its search (Found.hits), else the items whose name has the
        repository name; then, when those list no tag of it (or not `tag`), every other item of
        the catalog, as TGC searches names only and a name need not hold the path (GLM-5.1 is
        z-ai/glm5_1). Only an item's detail names its images (the item list leaves image_tags
        empty), so no other item can be passed over. No search is repeated, and no item is read
        twice. A tag of another repository is left out, as one item can list several. The other
        items are read as far as they answer, within SEARCH_BUDGET: without a match the search
        shares it; with one, it follows the search that found the match, which had its own. With
        a match, a request for them that fails or times out (a catalog page, or the detail of one
        item or of all), or a read longer than SEARCH_BUDGET, keeps the tags read: a tag they
        lack is one the catalog does not list, and `where` notes that the other items were not
        all read. A catalog that lists more items than its first TGC_PAGES pages gives the same
        note when the items read lack the tag. The catalog answered for the match, so this is
        never cnd_unreachable. Without a match (`found` None) nothing is known yet: only a detail
        that fails among others gives the note, and a search, a catalog page or every detail that
        fails, or SEARCH_BUDGET running out, is cnd_unreachable. The catalog is the only
        anonymous source for tgcr."""
        base = TGC_BASE
        wanted = same_repository(ref)

        def of_repository(items: list[Found]) -> list[CatalogTag]:
            return [
                entry
                for item in items
                for entry in item.tags
                if (image := rl.parse_image(entry.image)) is not None
                and same_repository(image) == wanted
            ]

        def lists(tags: list[CatalogTag]) -> bool:
            return bool(tags) and (tag is None or any(entry.tag == tag for entry in tags))

        def newest_first(tags: list[CatalogTag]) -> list[CatalogTag]:
            # each item lists its own tags newest first, but not the items read together
            return merge_tags(sorted(tags, key=lambda entry: entry.created, reverse=True))

        def unread(tags: list[CatalogTag], where: str, note: str) -> tuple[list[CatalogTag], str]:
            ns.log(f"{note}: only the tags of {ref.repository_ref} read so far are used")
            return tags, f"{where} ({note})"

        read = [found, *found.hits] if found is not None else []
        own = of_repository(read)
        if found is not None and lists(own):
            return newest_first(own), f"the TGC catalog item {found.name!r}"

        def work() -> tuple[list[CatalogTag], str]:
            if found is not None:
                named = read
                ns.log(f"the TGC items read for {found.name!r} lack tag {tag}: reading the others")
            else:
                leaf = ref.repository.rsplit("/", 1)[-1]
                named = self.search(query_words(leaf), in_part=True)[0]
                if lists(tags := of_repository(named)):
                    return tags, "the TGC catalog"
                ns.log(f"no TGC item named like {leaf!r} lists it: reading every catalog item")
            done = {item.item_id for item in named}  # their details are read already
            try:
                listed, more = self.items(base, None)  # more: past its first TGC_PAGES pages
                items = [item for item in listed if str(item.get("id")) not in done]
                lookups = [lambda item=item: self.detail(base, item) for item in items]
                results = self.detail_results(lookups)  # raises when every detail fails
            except ns.ScriptError as exc:
                if found is None:
                    raise
                # A page or every detail failed or timed out: the catalog answered for the
                # match, so its tags stay and the tag it lacks is not listed, with a note.
                ns.log(f"the other TGC catalog items could not be read: {exc.message}")
                return unread(own, f"the TGC catalog item {found.name!r}", NOT_ALL_READ)
            others = [result for result in results if isinstance(result, Found)]
            tags = of_repository(named) + of_repository(others)
            failed = any(isinstance(result, ns.ScriptError) for result in results)
            # an item past the pages read, or one whose detail failed, may list it
            if not lists(tags) and (more or failed):
                return unread(tags, "the TGC catalog", NOT_ALL_READ)
            return tags, "the TGC catalog"

        def late() -> tuple[list[CatalogTag], str]:
            if found is None:
                raise self.unreachable(f"no answer within {SEARCH_BUDGET:g} s")
            note = f"the other catalog items were not all read within {SEARCH_BUDGET:g} s"
            return unread(own, f"the TGC catalog item {found.name!r}", note)

        try:
            tags, where = deadline(SEARCH_BUDGET, lambda: shaped(self, work), late)
        except ns.ScriptError as exc:
            raise ns.ScriptError(
                "cnd_unreachable",
                f"the labels of {ref.repository_ref} need a tgcr login, and the TGC catalog could "
                f"not be read: {exc.message}",
                exc.fix,
                ns.EXIT_EXTERNAL,
            ) from exc
        return newest_first(tags), where

    def image_behind_login(
        self, ref: rl.ImageRef, found: Found | None, labels: rl.Labels
    ) -> CndImage:
        image = super().image_behind_login(ref, found, labels)
        tags, where = self.catalog_tags(ref, found, ref.tag)
        entry = next((tag for tag in tags if tag.tag == ref.tag), None)
        if entry is None:
            image.problems.append(f"{where} lists no tag {ref.tag} of {ref.repository_ref}")
            return image
        image.catalog_ngc_url, image.catalog = entry.ngc_url, entry.as_facts()
        image.architectures = (entry.arch,) if entry.arch else ()
        image.digest = entry.digest
        return image

    def images_behind_login(
        self, ref: rl.ImageRef, found: Found | None, problem: str
    ) -> tuple[list[CndImage], str | None]:
        tags, where = self.catalog_tags(ref, found)
        images = [
            CndImage(
                ref.with_tag(tag.tag),
                catalog_ngc_url=tag.ngc_url,
                architectures=(tag.arch,) if tag.arch else (),
                tags=(tag.tag,),
                digest=tag.digest,
                catalog=tag.as_facts(),
                problems=[problem],
            )
            for tag in tags
        ]
        return images, None if images else f"{problem}, and {where} lists no image of it"


class TuringcmReader(HarborReader):
    """turingcm. The TuringCM catalog (nim.turingcm.com) lists its NIMs: one list, then one detail
    per match, whose deploy text names the image. Its Harbor (resource.turingcm.com) answers
    without a login."""

    source, catalog, catalog_host = "turingcm", "TuringCM", "nim.turingcm.com"

    def search(self, words: list[str]) -> tuple[list[Found], int]:
        base = TURINGCM_BASE
        body = self.catalog_json(f"{base}/server/resource/nimImages/simple-list")
        records = _records(body, "data")
        if records is None:
            raise self.unreachable("no records list", urllib.parse.urlsplit(base).netloc)
        hits = [item for item in records if matches(words, (_text(item.get("name")),))]
        hits = exact_first(words, hits, lambda item: _text(item.get("name")))
        lookups = [lambda item=item: self.detail(base, item) for item in hits[:MAX_MATCHES]]
        return self.details(lookups), len(hits)

    def list_all(self, wanted: Wanted | None = None) -> Catalog:
        """Every catalog record with the images its deploy text names: one detail request for
        each record `wanted` keeps, at most PARALLEL at a time."""
        base = TURINGCM_BASE
        body = self.catalog_json(f"{base}/server/resource/nimImages/simple-list")
        listed = _records(body, "data")
        if listed is None:
            raise self.unreachable("no records list", urllib.parse.urlsplit(base).netloc)
        records, left_out = _wanted(listed, wanted)
        return self.catalog_of(records, left_out, lambda item: self.detail(base, item))

    def detail(self, base: str, item: dict[str, Any]) -> list[Found]:
        """One match per image the item's deploy text names (it can name a DGX Spark build
        too), each with the others in `also`; none when it names no image."""
        item_id = _text(item.get("id")) or str(item.get("id"))
        query = urllib.parse.urlencode({"id": item_id})
        body = self.catalog_json(f"{base}/server/resource/nimImages/getNimImagesById?{query}")
        detail = body.get("data") if isinstance(body, dict) else None
        if not isinstance(detail, dict):  # an error in a 200 answer
            code = body.get("code") if isinstance(body, dict) else None
            raise self.unreachable(f"item {item_id} answered without its details (code {code})")
        images = turingcm_images(_text(detail.get("deploy")))
        name = _text(item.get("name")) or _text(detail.get("name"))
        return [
            Found(
                source=self.source,
                name=name,
                image=image,
                architectures=_archs(item.get("arch")),
                model_type=_text(item.get("model_type")) or None,
                item_id=item_id,
                texts=(name, image),
                # the deploy text's tag (often latest) is not the repository's newest tag
                tag_listed=False,
                also=tuple(other for other in images if other != image),
                devices=_devices(item.get("badges")),
            )
            for image in images
        ]


class LichanReader(HarborReader):
    """lichan. No catalog but its Harbor (io.chancloud.com, project cnd), which answers without a
    login: a name is searched in the repository paths, and the newest tagged image of each match
    is read."""

    source, catalog, catalog_host = "lichan", "Lichan", LICHAN_HOST

    def search(self, words: list[str]) -> tuple[list[Found], int]:
        status, names, problem = rl.list_repositories(LICHAN_HOST, LICHAN_PROJECT)
        if status == "unauthorized":
            raise self.needs_login(problem or status)
        if status != "read":
            raise self.unreachable(problem or status)
        prefix = f"{LICHAN_PROJECT}/"
        paths = [name.removeprefix(prefix) for name in names if name.startswith(prefix)]
        hits = exact_first(words, [path for path in paths if matches(words, (path,))], str)
        if problem:
            # Only the first REPO_PAGES pages were read, and the project may list more: a
            # repository past them may be the one asked for, or have the name of one read
            # (team-b/model past them, team-a/model read), so neither model_not_found nor a match
            # would be sure (list_all notes the rest as unread).
            raise self.not_searched_in_full(
                words,
                problem,
                [path for path in hits if exact(words, (path,))],
                f"first {rl.REPO_PAGES * 100} repositories",
                LICHAN_HOST,
            )

        def read(path: str) -> Found:
            ref = rl.ImageRef(LICHAN_HOST, self.source, LICHAN_HOST, LICHAN_PROJECT, path, None)
            # the newest tag is enough to show a match: one index child at most
            listing = rl.read_repository(ref, max_children=1)
            images = listing[1]
            tag = next((t for t in (pinned_tag(i.tags) for i in images) if t), None)
            tag = tag or (images[0].tags[0] if images else None)
            # A repository whose tags could not be listed keeps its untagged reference.
            return Found(
                self.source,
                path,
                ref.with_tag(tag).image,
                architectures=images[0].architectures if images else (),
                listing=listing,
                texts=(path,),
            )

        found = run_parallel([lambda path=path: read(path) for path in hits[:MAX_MATCHES]])
        return found, len(hits)

    def list_all(self, wanted: Wanted | None = None) -> Catalog:
        """Every repository of project cnd that `wanted` keeps, by its path (no request per
        repository: images() reads its tagged images when it is checked). When the project may
        list more than the REPO_PAGES pages read, the others are not checked (Catalog.unread):
        the list is partial."""
        status, names, problem = rl.list_repositories(LICHAN_HOST, LICHAN_PROJECT)
        if status == "unauthorized":
            raise self.needs_login(problem or status)
        if status != "read":
            raise self.unreachable(problem or status)
        prefix = f"{LICHAN_PROJECT}/"
        found, left_out = [], []
        for name in names:
            path = name.removeprefix(prefix)
            ref = rl.ImageRef(LICHAN_HOST, self.source, LICHAN_HOST, LICHAN_PROJECT, path, None)
            if not name.startswith(prefix) or rl.parse_image(ref.image) != ref:
                continue
            if wanted is None or wanted(path, None):
                found.append(Found(self.source, path, ref.image, texts=(path,)))
            else:
                left_out.append((path, None))
        catalog = Catalog(found, [], left_out)
        if problem:  # only the first REPO_PAGES pages were read: the project may list more
            catalog.unread = (
                f"it may list more than the {rl.REPO_PAGES * 100} repositories of its first "
                f"{rl.REPO_PAGES} pages of 100"
            )
        return catalog

    def listing(self, ref: rl.ImageRef, found: Found | None, host_arch: str | None) -> Listing:
        """The listing the search read, when it has every label and they are those of
        `host_arch`; else a new one. The search reads no architecture (the first image of a
        multi-arch index), so an index of several architectures is read again for `host_arch`."""
        listing = found.listing if found is not None else None
        if (
            listing is not None
            and listing[0] == "read"
            and all(i.labels_read for i in listing[1])
            and (host_arch is None or all(len(i.architectures) <= 1 for i in listing[1]))
        ):
            return listing
        return super().listing(ref, found, host_arch)


READERS: dict[str, Reader] = {
    reader.source: reader for reader in (TgcReader(), TuringcmReader(), LichanReader())
}
CATALOGS = {source: reader.catalog for source, reader in READERS.items()}


def reader(source: str) -> Reader:
    """The reader of one CND."""
    return READERS[source]


def tgc_tags(detail: dict[str, Any]) -> list[CatalogTag]:
    """The image tags of a TGC item, newest first. A tag without its own ngc_url takes the
    item's only when it is the item's one tag; a tag without an arch takes the item's. Only
    tagged images on tgcr are kept: another host, or a reference that is not one, is left out."""
    raw = detail.get("image_tags")
    items = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
    item_url = _text(detail.get("ngc_url")) or None
    tags = []
    for item in items:
        image = _text(item.get("docker_url"))
        parsed = rl.parse_image(image)
        if parsed is None or not parsed.tag or parsed.source != "tgcr":
            ns.log(f"a TGC image tag was left out: {image!r} is no tagged image on tgcr")
            continue
        tags.append(
            CatalogTag(
                tag=parsed.tag,
                image=image,
                ngc_url=_text(item.get("ngc_url")) or (item_url if len(items) == 1 else None),
                arch=rl.normalise_arch(item.get("arch")) or rl.normalise_arch(detail.get("arch")),
                digest=_text(item.get("digest")) or None,
                created=_text(item.get("create_time")),
            )
        )
    return sorted(tags, key=lambda tag: tag.created, reverse=True)


def merge_tags(tags: list[CatalogTag]) -> list[CatalogTag]:
    """The tags, once each, and each without an arch given that of another entry with the same
    digest: TGC lists one image under several items, not all with an arch. A tag kept without an
    ngc_url takes that of another entry of the same image and digest, never of another digest."""
    known = {entry.digest: entry.arch for entry in tags if entry.digest and entry.arch}
    merged: dict[tuple[str, str | None], CatalogTag] = {}
    for entry in tags:
        if not entry.arch and entry.digest in known:
            entry = replace(entry, arch=known[entry.digest])
        key = (entry.image, entry.digest)
        kept = merged.setdefault(key, entry)
        if entry.digest and entry.ngc_url and not kept.ngc_url:
            merged[key] = replace(kept, ngc_url=entry.ngc_url)
    return list(merged.values())


def turingcm_images(deploy: str) -> list[str]:
    """The resource.turingcm.com images in a TuringCM `deploy` text (HTML), one per repository,
    in their order: it can name a DGX Spark build besides the image itself. A reference that is
    not an image is left out."""
    images: list[str] = []
    repositories: set[str] = set()
    for match in _TURINGCM_IMAGE.finditer(deploy or ""):
        image = match.group(0).rstrip(".,;")
        ref = rl.parse_image(image)
        if ref is None:
            ns.log(f"a TuringCM image was left out: {image!r} is no image reference")
        elif ref.repository_ref not in repositories:
            repositories.add(ref.repository_ref)
            images.append(image)
    return images


def shaped(distributor: Reader, work: Callable[[], T]) -> T:
    """work(), with an error other than a ScriptError (a catalog answer in a shape the scripts do
    not read) reported as cnd_unreachable, never as a traceback."""
    try:
        return work()
    except ns.ScriptError:
        raise
    except Exception as exc:
        raise distributor.unreachable(
            f"it answered in a shape the scripts do not read ({type(exc).__name__}: {exc})"
        ) from exc


def same_repository(ref: rl.ImageRef) -> tuple[str, str, str]:
    """(source, project, repository): tgcr.turing-agi.com and tgcr-gz.turing-agi.com serve the
    same repositories."""
    return ref.source, ref.project, ref.repository


# --- catalog search on the CND the user picked --------------------------------------------------


@dataclass
class Search:
    """The catalog matches of a query, best first, and how many names matched (only the first
    MAX_MATCHES have their image looked up)."""

    found: list[Found]
    total: int


def search(source: str, query: str) -> Search:
    """The catalog matches of `query` on the CND `source` alone, best first: those the name names
    exactly, then the others. A match without an image is left out. cnd_unreachable (exit 4)
    when the catalog cannot be read within SEARCH_BUDGET seconds."""
    words = query_words(query)
    if not words:
        raise ns.ScriptError(
            "bad_model_ref", "the model name is empty", "Name the model, e.g. qwen3.6-27b"
        )
    distributor = reader(source)

    def late() -> tuple[list[Found], int]:
        raise distributor.unreachable(f"no answer within {SEARCH_BUDGET:g} s")

    found, total = deadline(
        SEARCH_BUDGET, lambda: shaped(distributor, lambda: distributor.search(words)), late
    )
    return Search(sorted(found, key=lambda item: not exact(words, item.texts)), total)


def one_image(items: list[Found]) -> Found:
    """The first of several matches that name one image, with the image tags of them all: TGC
    lists one image under several items, not all with an arch."""
    tags = [entry for item in items for entry in item.tags]
    return replace(items[0], tags=merge_tags(tags)) if len(items) > 1 else items[0]


def pick(source: str, query: str, result: Search) -> Found:
    """The one match the query names: the only match, or the only one it names exactly (catalog
    items that name the same image count as one; of the images one item names, the one whose
    repository is the name), with every match of the search as its `hits`; else model_ambiguous
    (exit 3, facts.candidates) or model_not_found (exit 3, facts.other_sources)."""
    catalog = CATALOGS[source]
    found = result.found
    if not found:
        listed = (
            f"; {result.total} name(s) match, but the catalog names no image for them"
            if result.total
            else ""
        )
        raise not_on_cnd(
            source,
            f"no NIM in the {catalog} catalog ({source}) matches {query!r}{listed}",
            f"Ask the user for the exact name, or the full image on {catalog}",
            {"query": query},
        )
    words = query_words(query)
    named = [item for item in found if exact(words, item.texts)]
    items = {item.item_id for item in named}
    if len({item.image for item in named}) > 1 and len(items) == 1 and None not in items:
        # one catalog item names several images (a TuringCM deploy text with a DGX Spark build)
        named = [item for item in named if exact(words, (item.image or "",))] or named
    if named and len({item.image for item in named}) == 1:
        return replace(one_image(named), hits=tuple(found))
    looked_up = {(item.item_id, item.name) for item in found}
    if len(looked_up) == result.total and len({item.image for item in found}) == 1:
        return replace(one_image(found), hits=tuple(found))
    shown = [item.candidate() for item in found]
    more = result.total - min(result.total, MAX_MATCHES)  # names whose image was not looked up
    raise ns.ScriptError(
        "model_ambiguous",
        f"{len(shown) + more} NIMs in the {catalog} catalog match {query!r}: "
        + ", ".join(str(item["image"]) for item in shown)
        + (f" and {more} more" if more > 0 else ""),
        "Show facts.candidates to the user and ask which; then rerun with the chosen candidate's "
        "repository (facts.candidates[].repository), adding --tag only if the user named a version "
        "or another candidate has the same repository (then --tag with the chosen one's "
        "newest_tag)",
        ns.EXIT_PRECONDITION,
        facts={"query": query, "source": source, "candidates": shown},
    )


# --- tags and NGC builds ----------------------------------------------------------------------


def pinned_tag(tags: Sequence[str]) -> str | None:
    """The most specific tag that is not `latest` (most dots and dashes, then the longest)."""
    usable = [tag for tag in tags if tag and tag.lower() != "latest"]
    if not usable:
        return None
    return max(usable, key=lambda tag: (tag.count(".") + tag.count("-"), len(tag)))


def ngc_repository(ref: rl.ImageRef) -> str | None:
    """The NGC resource id the CND path names: TEAM/REPO after the project (cnd/, cnd-enterprise/,
    public/) and a leading nim/ are stripped; None when that is not TEAM/REPO."""
    path = f"{ref.project}/{ref.repository}".lower()
    for prefix in CND_PROJECTS:
        if path.startswith(prefix):
            path = path[len(prefix) :]
            break
    path = path.removeprefix("nim/")
    parts = path.split("/")
    return f"nim/{path}" if len(parts) == 2 and all(parts) else None


@dataclass(frozen=True)
class Target:
    """One NGC build a CND image may be: how it was named (`mapping`), and where from."""

    mapping: str  # ngc_url | version_label | user_version
    resource_id: str
    tag: str
    origin: str

    @property
    def image(self) -> str:
        return f"nvcr.io/{self.resource_id}:{self.tag}"


def targets(
    ref: rl.ImageRef,
    version: str | None,
    label_ngc_url: str | None,
    catalog_ngc_url: str | None,
    nim_version: str | None,
) -> tuple[list[Target], list[dict[str, Any]]]:
    """The NGC builds to try for the CND image `ref`, in order (ngc_url, version_label,
    user_version), and the values that could not be used, with why."""
    found: list[Target] = []
    unused: list[dict[str, Any]] = []
    for origin, url in (
        (f"its {rl.NGC_URL_LABEL} label", label_ngc_url),
        ("the TGC catalog's ngc_url", catalog_ngc_url),
    ):
        if not url:
            continue
        public = rl.public_ngc_image(url)
        if public is None:
            unused.append(
                {
                    "mapping": "ngc_url",
                    "value": url,
                    "problem": f"{origin} {url} is not a public NGC NIM image with a version tag "
                    "(nvcr.io/nim/TEAM/REPO:VERSION)",
                }
            )
        elif all(t.image != f"nvcr.io/{public[0]}:{public[1]}" for t in found):
            found.append(Target("ngc_url", public[0], public[1], f"{origin} {url}"))
    repository = ngc_repository(ref)
    no_repository = (
        f"names only a version, and {ref.repository_ref} does not name an NGC TEAM/REPO once its "
        "project is stripped"
    )
    if version:
        tag, corrected = rl.label_version(version)
        origin = f"its {rl.VERSION_LABEL} label"
        if tag is None:
            unused.append(
                {
                    "mapping": "version_label",
                    "value": version,
                    "problem": f"{origin} {version!r} is not a version tag",
                }
            )
        elif repository is None:
            problem = f"{origin} {tag} {no_repository}"
            unused.append({"mapping": "version_label", "value": version, "problem": problem})
        else:
            said = f" {version!r}, read as {tag}" if corrected else f" {tag}"
            found.append(Target("version_label", repository, tag, origin + said))
    if nim_version:
        if repository is None:
            unused.append(
                {
                    "mapping": "user_version",
                    "value": nim_version,
                    "problem": f"--nim-version {nim_version} {no_repository}",
                }
            )
        else:
            found.append(
                Target("user_version", repository, nim_version, f"--nim-version {nim_version}")
            )
    return found, unused


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="China NIM distributor readers and NGC builds for the nim-select scripts; "
        "import it, do not run it.",
        epilog='Example: python3 -I "<SKILL_DIR>/scripts/cnd.py" --help',
    )
    parser.parse_args(argv)
    ns.emit(True, "library", "done", {"note": "import this module; it has no command of its own"})


if __name__ == "__main__":
    main()
