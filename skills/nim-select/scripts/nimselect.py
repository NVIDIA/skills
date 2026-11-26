#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nim-select: resolve a model to its NIM image, inspect where it would run, and check the fit.

  locate          hint whether this machine is in China, to choose where to get the image from
  country         look up the country of this machine's public IP address, for --region (run it
                  only after the user agrees)
  resolve MODEL   find the NIM: on NGC, its repository and version tag; on a China NIM
                  distributor (--source tgcr, turingcm or lichan, or an image on one), its image
                  and the NGC build its labels name
  inspect         read the GPUs, driver and runtime of this host or a Kubernetes namespace
  fit MODEL       match the NIM Metadata API profiles of that tag against a saved environment
  select MODEL    check the newest image, hardware variants and older versions; pick the image
                  and profile that fit a saved environment (on a China NIM distributor: the image
                  named or found)
  list            list the LLM and VLM NIMs of the source with select's verdict on a saved
                  environment; only its fits rows are compatible (schema nim-select/list/v1; it
                  hands nothing off: a pick goes through select)

Read-only: nothing is pulled, run or applied, and nothing logs in. Prints one JSON object
{"ok", "state", "next", "facts", "errors"}; `--out FILE` also saves the facts object as JSON.
Exit codes: 0 ok, 2 bad input, 3 precondition failed, 4 external tool or service failed, 5 refused.
Never prompts. Logs go to stderr. Stdlib only.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import re
import sys
import threading
import time
import urllib.parse
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from typing import Any, NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import choose
import cnd
import env_inspect
import fit
import listing
import locality
import ngc_api
import ns_common as ns
import registry_labels as rl

ENV_SCHEMA = "nim-select/environment/v1"
ASK_USER_CODES = frozenset(
    {
        "model_ambiguous",
        "model_not_found",
        "tag_not_found",
        "tag_conflict",
        "context_required",
        "context_not_found",
        "gpus_unknown",
        "ngc_build_unknown",
    }
)
PULL_FALLBACK = (
    "or, if the user decides to pull the image, offer nim-deploy to list its profiles with "
    "list-model-profiles (nim-select pulls and runs nothing)"
)
NO_METADATA_FIX = f"Rerun fit with a tag from facts.tags_to_try, {PULL_FALLBACK}"
SELECT_FIX = f"Rerun select with --tag set to one of facts.tags_to_try, {PULL_FALLBACK}"
NGC_BUILD_UNKNOWN_FIX = (
    "Tell the user the profile can only be confirmed after the pull, by the image's own "
    "list-model-profiles, which nim-deploy runs; or ask the user for the image's NGC version and "
    "rerun with the image in facts.image and --nim-version VERSION. Never use the NGC image with "
    "the same name: CND tags are not NGC tags"
)
# The same, for an image the CND lists only as `latest`: --nim-version needs the image's own tag.
NGC_BUILD_UNKNOWN_LATEST_FIX = (
    "Tell the user the profile can only be confirmed after the pull, by the image's own "
    "list-model-profiles, which nim-deploy runs; or ask the user for the image's own version tag "
    "(the CND lists it only as latest) and its NGC version, and rerun with facts.image:TAG and "
    "--nim-version VERSION. Never use the NGC image with the same name: CND tags are not NGC tags"
)
# The same, for an image whose tags the CND does not show (tgcr: a login, and TGC lists none).
NGC_BUILD_UNKNOWN_NO_TAG_FIX = (
    "Tell the user the profile can only be confirmed after the pull, by the image's own "
    "list-model-profiles, which nim-deploy runs; or ask the user for the image's own version tag "
    "(the CND shows no tag of it) and its NGC version, and rerun with facts.image:TAG and "
    "--nim-version VERSION. Never use the NGC image with the same name: CND tags are not NGC tags"
)
# When --nim-version cannot map the image: only the after-pull check is left.
NGC_BUILD_UNKNOWN_PULL_FIX = (
    "Tell the user the profile can only be confirmed after the pull, by the image's own "
    "list-model-profiles, which nim-deploy runs. Do not ask for the image's NGC version: "
    "--nim-version cannot map this image ({why}). Never use the NGC image with the same name: "
    "CND tags are not NGC tags"
)
CND_NO_METADATA_FIX = (
    "Tell the user NGC publishes no profiles for facts.ngc_image, the NGC build of facts.image; "
    "to check another version, select another tag of the image on its China NIM distributor "
    f"(never an NGC tag), {PULL_FALLBACK}"
)
PROFILE_NOT_FOUND_FIX = (
    "Pass --profile the full profile_id of facts.alternative with the --tag its note gives, or of "
    "a profile in facts.shortlist or facts.unconfirmed of a select of the same image (same --tag) "
    "without --profile, or drop --profile; never shorten or guess an id"
)
# A profile id as select --profile takes it and the note on facts.alternative puts it in a command
# line: nothing a shell reads as more than a word (NGC's ids are 64 hexadecimal digits). The note
# writes it as --profile=ID, so an id that starts with '-' is never read as an option.
PROFILE_ID = re.compile(r"[A-Za-z0-9._-]+")
# A repository or tag that note puts in the same command line: such words, joined by `/`
# (`nim/nvidia/nemotron-3.5-lightning-30b-a3b`, `2.1.5`).
COMMAND_WORD = re.compile(rf"{PROFILE_ID.pattern}(?:/{PROFILE_ID.pattern})*")
TAGS_TO_TRY = 5


class JsonArgumentParser(argparse.ArgumentParser):
    """Report usage errors as the standard JSON result (exit 2) instead of argparse text."""

    def error(self, message: str) -> NoReturn:
        fix = f"See: {self.prog} --help"
        ns.emit(False, "input", "ask_user", errors=[ns.error("bad_input", message, fix)])


# What a note says when a Deployment Geography needs the user's country.
COUNTRY_ASK = (
    "ask which country it will run in, or, only if the user agrees, look the country up with "
    "the country command"
)
COUNTRY_FAILED_FIX = "Tell the user the lookup failed and ask which country the NIM will run in"


def fail(state: str, exc: ns.ScriptError, facts: dict[str, Any] | None = None) -> NoReturn:
    """Report a library error: ask the user for precondition problems, else fix the input. A CND
    image with no known NGC build has its own state, ngc_build_unknown."""
    asks = exc.code in ASK_USER_CODES or exc.exit_code == ns.EXIT_PRECONDITION
    merged = {**(facts or {}), **exc.facts}
    if exc.code == "ngc_build_unknown":
        state = exc.code
    ns.emit(
        False, state, "ask_user" if asks else "fix_errors", merged, [exc.as_error()], exc.exit_code
    )


def save(out: str | None, facts: dict[str, Any]) -> None:
    if out:
        try:
            ns.write_json_file(Path(out), facts)
        except OSError as exc:
            raise ns.ScriptError(
                "out_not_writable",
                f"cannot write {out}: {exc.strerror}",
                "Pass --out a path in a folder you can write",
            ) from exc
        ns.log(f"wrote {out}")


# --- resolve --------------------------------------------------------------------------------


def labels_text(repo: dict[str, Any]) -> list[str]:
    return [label.lower() for label in ns.as_list(repo.get("labels"))]


def repo_tags(repo: dict[str, Any]) -> list[str]:
    return ns.as_list(repo.get("tags"))


def tags_to_try(
    repo: dict[str, Any] | None,
    current: str,
    digests: ngc_api.Digests | None = None,
    skip: Iterable[str] = (),
) -> list[str]:
    """Up to 5 other builds of the repo, newest first, none of them `current` or in `skip`.

    A moving alias is given as the build it stands for (`1.0` -> `1.0.0-variant`) and dropped
    when that is `current` (`1.7` for `1.7.0-variant`); an alias with no build listed stays
    (`1` alone does not). Suffixed builds stay (`1.0.0-variant`, `1.8.0-RTX`); prereleases
    (`-rc1`) do not.
    """
    if repo is None:
        return []
    tags = repo_tags(repo)
    left_out = {current, *skip}
    same = (digests or {}).get(current)
    found: list[str] = []
    for tag in ngc_api.version_tags(tags):
        if ngc_api.is_prerelease(tag):
            continue
        if not ngc_api.is_build(tag):
            if ngc_api.is_alias_of(tag, current):
                continue
            target = ngc_api.alias_target(tag, tags, digests)
            parsed = ngc_api.parse_tag(tag)
            if target is None and (parsed is None or len(parsed[0]) < 2):
                continue  # `1` alone says nothing
            tag = target or tag
            if ngc_api.is_prerelease(tag):
                continue  # an alias whose digest is a prerelease's
        if tag in left_out or tag in found:
            continue
        if same is not None and (digests or {}).get(tag) == same:
            continue  # another name of the current image
        found.append(tag)
    return found[:TAGS_TO_TRY]


def candidate_names(candidate: dict[str, Any]) -> list[str]:
    """The names a search result goes by: its name, its catalog display name, its resource id."""
    return [str(candidate.get(key) or "") for key in ("name", "display_name", "resource_id")]


def pick_candidate(
    query: str, candidates: list[dict[str, Any]], notes: list[str] | None = None
) -> str:
    """The resource id the query names, or a ScriptError asking the user to choose. NGC's search
    matches any word of the query, so the one result that has every word of it (in its name,
    display name or resource id) is the NIM meant; `notes` then says so."""
    if not candidates:
        raise ns.ScriptError(
            "model_not_found",
            f"no NIM on NGC matches {query!r}",
            "Ask the user for the exact NIM name (e.g. meta/llama-3.1-8b-instruct) or image",
            ns.EXIT_PRECONDITION,
        )
    wanted = " ".join(query.lower().split())
    exact = [
        c
        for c in candidates
        if wanted
        in (
            str(c.get("name") or "").lower(),
            " ".join(str(c.get("display_name") or "").lower().split()),
            str(c.get("resource_id") or "").lower().rsplit("/", 1)[-1],
        )
    ]
    if len(exact) == 1:
        return str(exact[0]["resource_id"])
    if len(candidates) == 1:
        return str(candidates[0]["resource_id"])
    every = [c for c in candidates if listing.query_matches(query, candidate_names(c))]
    if len(every) == 1:
        resource_id = str(every[0]["resource_id"])
        if notes is not None:
            notes.append(
                f"NGC's search for {query!r} returned {len(candidates)} NIMs; only {resource_id} "
                "has every word of it, so it was picked. Tell the user which NIM this is."
            )
        return resource_id
    matching = every or candidates
    shown = matching[:10]
    raise ns.ScriptError(
        "model_ambiguous",
        f"{len(matching)} NIMs match {query!r}; "
        + ", ".join(str(c.get("resource_id")) for c in shown),
        "Show the candidates to the user and rerun with the chosen resource id",
        ns.EXIT_PRECONDITION,
        facts={"query": query, "candidates": shown},
    )


def candidates_with_tag(candidates: list[dict[str, Any]], tag: str) -> list[dict[str, Any]]:
    """The candidates whose NGC repository lists `tag` (checks at most 10 repositories)."""
    found = []
    for candidate in candidates[:10]:
        if candidate.get("latest_tag") == tag:
            found.append(candidate)
            continue
        repo = ngc_api.get_repo(str(candidate["resource_id"]))
        if repo is not None and tag in repo_tags(repo):
            found.append(candidate)
    return found


def newest_tag(repo: dict[str, Any]) -> str | None:
    try:
        return ngc_api.choose_tag(repo, None)[0]
    except ns.ScriptError:
        return None


@dataclasses.dataclass
class ImageList:
    """NGC's public image list of one repository (tag -> digest, CPU architectures, size), or
    why it could not be read: a NIM that is not public answers 403."""

    resource_id: str
    images: dict[str, ngc_api.Image]
    problem: str | None = None

    @property
    def digests(self) -> dict[str, str]:
        return ngc_api.digests_of(self.images)

    def unread_note(self) -> str:
        return (
            f"The CPU architectures of the {self.resource_id} images were not read "
            f"({self.problem}); its profiles that record none are unknown (arch_not_recorded)"
        )


# NGC errors that stop a command at once: NGC does not answer, or asks for fewer requests.
NGC_STOPS = ("ngc_unreachable", "ngc_rate_limited")


def not_public_error(exc: ns.ScriptError) -> bool:
    """NGC says the repository is not public: a 401 or 403 with NGC's error envelope."""
    return exc.code == "ngc_http_error" and exc.facts.get("not_public") is True


def read_image_list(resource_id: str) -> ImageList:
    """The image list of `resource_id`; an empty one with the problem when NGC does not show it.
    It is read right after the repository, so even a timeout or reset here only leaves out the
    list (`ngc_unreachable` on the repository, search or metadata still stops the command)."""
    try:
        images = ngc_api.get_images(resource_id)
    except ns.ScriptError as exc:
        return ImageList(resource_id, {}, exc.message)
    if images is None:
        return ImageList(resource_id, {}, f"NGC lists no images for {resource_id} (HTTP 404)")
    return ImageList(resource_id, images)


def image_facts(
    repo: dict[str, Any], tag: str, listed: ImageList
) -> tuple[list[str] | None, int | None, str | None]:
    """(CPU architectures, size in bytes, note) of the image `tag`, where NGC records them.

    The image list gives each tag's own. Without it, the repository record does, but only for
    the image tagged latestTag. When the list is not public, the record is also used for the
    newest version tag when latestTag is `latest` or an alias of it (`1.7` for `1.7.0-variant`):
    the two are then taken to be the same image, which no digest can confirm, and the note says
    so. A latestTag that is another build (newest 1.5.2, latestTag 1.8.0-RTX) says nothing about
    the newest image.
    """
    own = listed.images.get(tag)
    if own is not None:
        return list(own.architectures) or None, own.size, None
    latest = repo.get("latestTag")
    if not isinstance(latest, str) or not latest:
        return None, None, None
    size = repo.get("latestImageSize")
    number = isinstance(size, (int, float)) and not isinstance(size, bool) and math.isfinite(size)
    record = (
        ngc_api.real_architectures(repo.get("latestArchitectureVariants")) or None,
        int(size) if number else None,
    )
    if tag == latest:
        return (*record, None)
    if listed.problem is None or tag != newest_tag(repo):
        return None, None, None
    if latest != "latest" and not ngc_api.is_alias_of(latest, tag):
        return None, None, None
    note = (
        f"{listed.problem}, so the CPU architectures and size of {tag} come from the repository "
        f"record of the image tagged {latest}, taken to be the same image (no digest confirms it)"
    )
    return (*record, note)


def search_resource(query: str, requested: str | None, notes: list[str]) -> str:
    """The resource id a catalog search for `query` names; a requested version tag may single
    out one of several matches."""
    ns.log(f"searching NGC for {query!r}")
    candidates = ngc_api.search_nims(query)
    try:
        return pick_candidate(query, candidates, notes)
    except ns.ScriptError as exc:
        # Several NIMs match the name: a requested version tag may single one out.
        if exc.code != "model_ambiguous" or not requested or requested == "latest":
            raise
        with_tag = candidates_with_tag(candidates, requested)
        if len(with_tag) != 1:
            raise
        resource_id = str(with_tag[0]["resource_id"])
        notes.append(
            f"{len(candidates)} NIMs match {query!r}; only {resource_id} has tag "
            f"{requested}, so it was picked. Tell the user which NIM this is."
        )
        return resource_id


def restate_mirror_warning(
    ref: ngc_api.ModelRef, warnings: list[str], same_tag: bool, found: str | None = None
) -> list[str]:
    """`warnings` with the mirror warning of parse_model_ref restated for the image checked: it
    knew neither whether the tag named is the one checked (--tag, an alias replaced by its build)
    nor the NGC repository a search for the name found."""
    if not ref.source_registry:
        return warnings
    parsed = ngc_api.mirror_warning(ref.source_registry, bool(ref.tag and ref.tag != "latest"))
    final = ngc_api.mirror_warning(ref.source_registry, same_tag, found)
    return [final if warning == parsed else warning for warning in warnings]


def resolve_model(
    model: str,
    tag_flag: str | None,
    source: str = "ngc",
    nim_version: str | None = None,
    host_arch: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], ImageList]:
    """Resolve MODEL (and --tag) to the resolve facts; also return the NGC repo object and its
    image list. On a China NIM distributor (`source`), MODEL is its image or a name for its
    catalog, and the facts name the NGC build it maps to (resolve_cnd)."""
    if source in cnd.CNDS:
        return resolve_cnd(model, source, tag_flag, nim_version, host_arch)
    if nim_version:
        raise ns.ScriptError(
            "bad_input",
            "--nim-version applies only to an image on a China NIM distributor (tgcr, turingcm, "
            "lichan)",
            "Drop --nim-version; on NGC the tag is the NIM version",
        )
    ref = ngc_api.parse_model_ref(model)
    warnings = list(ref.warnings)
    notes: list[str] = []
    if tag_flag and ref.tag and tag_flag != ref.tag:
        raise ns.ScriptError(
            "tag_conflict",
            f"the model reference has tag {ref.tag!r} but --tag is {tag_flag!r}",
            "Give the tag once: in the image reference or with --tag",
        )
    requested = tag_flag or ref.tag
    resource_id, mapping = ref.resource_id, ref.mapping
    if resource_id is None:
        resource_id = search_resource(ref.query or model, requested, notes)
    repo = ngc_api.get_repo(resource_id)
    if repo is None and mapping == "by_name":
        # A mirror path whose team is not an NGC team (registry.example.com/ai-models/REPO):
        # search NGC by the repository name, as for a one-part path.
        leaf = resource_id.rsplit("/", 1)[-1]
        found = search_resource(leaf, requested, notes)
        notes.append(
            f"NGC has no repository {resource_id}; a search for {leaf!r} found {found}. "
            "Tell the user which NIM this is."
        )
        resource_id, mapping = found, "search"
        repo = ngc_api.get_repo(resource_id)
    if repo is None:
        raise ns.ScriptError(
            "model_not_found",
            f"NGC has no public repository {resource_id}",
            "Check the name with the user; NIM repositories look like "
            "nim/meta/llama-3.1-8b-instruct",
            ns.EXIT_PRECONDITION,
        )
    listed = read_image_list(resource_id)
    tag, tag_source, tag_notes = ngc_api.choose_tag(repo, requested, listed.digests)
    notes.extend(tag_notes)
    found = resource_id if mapping == "search" else None
    warnings = restate_mirror_warning(ref, warnings, tag_source == "requested", found)
    architectures, size, image_note = image_facts(repo, tag, listed)
    if image_note:
        notes.append(image_note)
    elif architectures is None and listed.problem:
        notes.append(listed.unread_note())
    labels = labels_text(repo)
    model_name = resource_id.removeprefix("nim/")
    facts: dict[str, Any] = {
        "source": "ngc",
        "resource_id": resource_id,
        "model": model_name,
        "display_name": repo.get("displayName") or repo.get("name"),
        "tag": tag,
        "tag_source": tag_source,
        "image": f"nvcr.io/{resource_id}:{tag}",
        "latest_tag_field": repo.get("latestTag"),
        "tags": repo_tags(repo)[:15],
        "architectures": architectures,
        "image_size_gb": round(size / 1e9, 2) if size is not None else None,
        "nvaie_supported": any("nvaie_supported" in label for label in labels),
        "experimental": any("experimental" in label for label in labels),
        "can_guest_pull": repo.get("canGuestPull"),
        "mapping": mapping,
        "source_registry": ref.source_registry,
        "warnings": warnings,
        "notes": notes,
    }
    return facts, repo, listed


# --- China NIM distributors (CNDs) ------------------------------------------------------------


@dataclasses.dataclass
class NgcBuild:
    """The NGC build a CND image maps to, found on NGC."""

    target: cnd.Target
    repo: dict[str, Any]
    listed: ImageList
    tag: str
    tag_source: str
    notes: list[str]
    architectures: list[str] | None

    @property
    def resource_id(self) -> str:
        return self.target.resource_id

    @property
    def image(self) -> str:
        return f"nvcr.io/{self.resource_id}:{self.tag}"


class NgcChecker:
    """Checks the NGC builds a CND image may be: NGC has the repository and the tag, and the CPU
    architectures do not clash. Each repository and image list is read once, also when several
    threads ask for it at the same time (the list command)."""

    def __init__(self) -> None:
        self.repos: dict[str, dict[str, Any] | None] = {}
        self.missing: dict[str, str] = {}  # why NGC shows no repository
        self.lists: dict[str, ImageList] = {}
        self.locks: dict[str, threading.Lock] = {}
        self.lock = threading.Lock()

    def once(self, key: str) -> threading.Lock:
        """The lock of one read, so a second thread waits for the first one's answer."""
        with self.lock:
            return self.locks.setdefault(key, threading.Lock())

    def repository(self, rid: str) -> dict[str, Any] | None:
        """NGC's public repository `rid` (read once), or None. A repository that NGC says is not
        public (401, 403 with its error envelope) is none; NGC failing (5xx, or 429 after its
        retries), or something else answering (ngc_bad_response, a 401 or 403 without NGC's
        envelope) stops the command (exit 4), as on NGC: an outage says nothing about the image's
        build."""
        with self.once(f"repo {rid}"):
            if rid not in self.repos:
                try:
                    self.repos[rid] = ngc_api.get_repo(rid)
                except ns.ScriptError as exc:
                    if not not_public_error(exc):
                        raise
                    self.repos[rid], self.missing[rid] = None, exc.message
            return self.repos[rid]

    def check(
        self, target: cnd.Target, cnd_archs: Iterable[str]
    ) -> tuple[NgcBuild | None, str | None]:
        """(the build, None), or (None, why it cannot be the image's build)."""
        rid = target.resource_id
        repo = self.repository(rid)
        if repo is None:
            return None, self.missing.get(rid, f"NGC has no public repository {rid}")
        same = [tag for tag in repo_tags(repo) if tag.lower() == target.tag.lower()]
        if not same:
            return None, f"NGC {rid} has no tag {target.tag}"
        with self.once(f"images {rid}"):
            if rid not in self.lists:
                self.lists[rid] = read_image_list(rid)
            listed = self.lists[rid]
        tag, tag_source, notes = ngc_api.choose_tag(repo, same[0], listed.digests)
        archs, _, note = image_facts(repo, tag, listed)
        if note:
            notes = [*notes, note]
        elif archs is None and listed.problem:
            notes = [*notes, listed.unread_note()]
        mine = sorted(set(cnd_archs))
        if mine and archs and not set(mine) & set(archs):
            return None, (
                f"architecture clash: the {target.mapping} build nvcr.io/{rid}:{tag} has "
                f"{', '.join(archs)} only, the CND image {', '.join(mine)}"
            )
        return NgcBuild(target, repo, listed, tag, tag_source, notes, archs), None


def first_build(
    checker: NgcChecker, image: cnd.CndImage, nim_version: str | None
) -> tuple[NgcBuild | None, list[dict[str, Any]]]:
    """The first NGC build the image maps to (cnd.targets order), and every mapping tried."""
    found, unused = cnd.targets(
        image.ref, image.version, image.ngc_url, image.catalog_ngc_url, nim_version
    )
    tried = [{**item, "image": image.ref.image} for item in unused]
    for target in found:
        build, problem = checker.check(target, image.architectures)
        tried.append(
            {
                "mapping": target.mapping,
                "image": image.ref.image,
                "value": target.origin,
                "ngc_image": target.image,
                "problem": problem,
            }
        )
        if build is not None:
            return build, tried
    return None, tried


def split_name_tag(model: str) -> tuple[str, str | None]:
    """`qwen3.6-27b:1.0` -> (`qwen3.6-27b`, `1.0`); free text keeps its words and has no tag."""
    text = model.strip()
    if any(ch.isspace() for ch in text):
        return text, None
    head, _, last = text.rpartition("/")
    if ":" in last:
        last, tag = last.split(":", 1)
        return (f"{head}/{last}" if head else last), (tag or None)
    return text, None


def version_blocked(checker: NgcChecker, ref: rl.ImageRef, nim_version: str | None) -> str | None:
    """Why --nim-version cannot map `ref`, when what was read already says so. It takes the NGC
    repository from the CND path, so the path must name a TEAM/REPO that NGC shows; and a version
    the user gave that NGC has, yet that did not map (an architecture clash), will not map when
    asked for again."""
    rid = cnd.ngc_repository(ref)
    if rid is None:
        return (
            f"it takes the NGC repository from the path, and {ref.repository_ref} names no "
            "TEAM/REPO"
        )
    if rid in checker.repos and checker.repos[rid] is None:
        return f"it takes the NGC repository from the path, {rid}, which NGC does not show"
    repo = checker.repos.get(rid)
    if nim_version and repo and nim_version.lower() in {tag.lower() for tag in repo_tags(repo)}:
        return f"NGC has {rid}:{nim_version}, the version the user gave, and it still did not map"
    return None


def build_unknown(
    source: str,
    image: rl.ImageRef,
    cnd_image: cnd.CndImage | None,
    tried: list[dict[str, Any]],
    notes: list[str],
    why: str,
    checker: NgcChecker,
    nim_version: str | None = None,
    only_latest: bool = False,
) -> ns.ScriptError:
    """ngc_build_unknown (exit 3): no label, catalog entry or version given names an NGC build
    that NGC has, so no profile can be checked before the pull. The fix offers --nim-version only
    where it can map the image; `only_latest`: the CND lists the image only as latest."""
    # what each label, catalog entry or version named, and why it is no build NGC has
    reasons = []
    for item in tried:
        if not item.get("problem"):
            continue
        whose = "" if item.get("image") == image.image else f"{item.get('image')}: "
        if item.get("ngc_image"):
            reasons.append(
                f"{whose}{item['value']} names {item['ngc_image']}, which cannot be used: "
                f"{item['problem']}"
            )
        else:
            reasons.append(f"{whose}{item['problem']}")
    problems = list(cnd_image.problems) if cnd_image is not None else []
    message = f"no NGC build is known for {image.image}: "
    message += "; ".join(dict.fromkeys(reasons)) if reasons else why
    if problems:
        message += " (" + "; ".join(dict.fromkeys(problems)) + ")"
    blocked = version_blocked(checker, image, nim_version)
    if blocked:
        fix = NGC_BUILD_UNKNOWN_PULL_FIX.format(why=blocked)
    elif image.tag:
        fix = NGC_BUILD_UNKNOWN_FIX
    else:
        fix = NGC_BUILD_UNKNOWN_LATEST_FIX if only_latest else NGC_BUILD_UNKNOWN_NO_TAG_FIX
    return ns.ScriptError(
        "ngc_build_unknown",
        message,
        fix,
        ns.EXIT_PRECONDITION,
        {
            "source": source,
            "image": image.image,
            "cnd_repository": image.repository_ref,
            "cnd_tag": image.tag,
            "cnd_labels": cnd_image.labels if cnd_image else None,
            "cnd_catalog": cnd_image.catalog if cnd_image else None,
            "mapping": None,
            "mappings_tried": tried,
            "source_registry": image.host,
            "notes": notes,
        },
    )


def check_nim_version(nim_version: str | None, tag: str | None) -> None:
    """--nim-version names the NGC version of one CND image: one named with its own tag."""
    if nim_version is None:
        return
    if not tag or tag.lower() == "latest":
        raise ns.ScriptError(
            "bad_input",
            "--nim-version needs the CND image named with its own tag (not latest): it is the "
            "NGC version of that one image",
            "Pass the image as HOST/PROJECT/REPO:TAG (facts.image of the last run), or --tag",
        )
    version, corrected = rl.label_version(nim_version)
    if version is None or corrected or not ngc_api.version_key(version):
        raise ns.ScriptError(
            "bad_input",
            f"--nim-version must be a NIM version such as 1.7.0-variant; got {nim_version!r}",
            "Pass the version the user gave, e.g. --nim-version 1.7.0-variant",
        )


def resolve_cnd(
    model: str,
    source: str,
    tag_flag: str | None,
    nim_version: str | None,
    host_arch: str | None,
) -> tuple[dict[str, Any], dict[str, Any], ImageList]:
    """Resolve MODEL on a China NIM distributor: its image (named, or found in the CND's catalog)
    and the NGC build it maps to, through labels, the TGC catalog or --nim-version; never by its
    name. No tag or `latest`: the newest tag of the repository with a known build. Facts as
    resolve_model, with `image` the CND image (never `latest`; `pinned` false when no other tag
    is known) and `ngc_image` the NGC build whose profiles are used."""
    reader = cnd.reader(source)
    notes: list[str] = []
    found: cnd.Found | None = None
    if cnd.reference_host(model):
        ref = rl.parse_image(model, tag_flag)
        if ref is None:
            raise ns.ScriptError(
                "bad_model_ref",
                f"{model!r} (--tag {tag_flag!r}) is no image reference HOST/PROJECT/REPO[:TAG] "
                f"on {cnd.reference_host(model)}"
                if tag_flag
                else f"{model!r} is no image reference HOST/PROJECT/REPO[:TAG] on "
                f"{cnd.reference_host(model)}",
                "Pass the full image, e.g. io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0",
            )
        named = rl.parse_image(model)
        if tag_flag and named is not None and named.tag and named.tag != tag_flag:
            raise ns.ScriptError(
                "tag_conflict",
                f"the image has tag {named.tag!r} but --tag is {tag_flag!r}",
                "Give the tag once: in the image reference or with --tag",
            )
        asked = ref.tag
    else:
        query, query_tag = split_name_tag(model)
        if tag_flag and query_tag and tag_flag != query_tag:
            raise ns.ScriptError(
                "tag_conflict",
                f"the model reference has tag {query_tag!r} but --tag is {tag_flag!r}",
                "Give the tag once: in the model reference or with --tag",
            )
        asked = tag_flag or query_tag
        ns.log(f"searching the {reader.catalog} catalog for {query!r}")
        found = cnd.pick(source, query, cnd.search(source, query))
        ref = rl.parse_image(found.image or "")
        assert ref is not None
        kind = f" (model type {found.model_type})" if found.model_type else ""
        notes.append(
            f"The {reader.catalog} catalog names {found.name!r}{kind} for {query!r}: "
            f"{ref.repository_ref}."
        )
        if found.also:
            notes.append(
                f"The {reader.catalog} catalog item {found.name!r} also names "
                f"{', '.join(found.also)}, which was not checked; name it to check it."
            )
        ref = ref.with_tag(asked)
    check_nim_version(nim_version, asked)
    checker = NgcChecker()
    pinned = True
    if asked and asked.lower() != "latest":
        image = reader.image(ref, found, host_arch)
        build, tried = first_build(checker, image, nim_version)
        if build is None:
            raise build_unknown(
                source,
                ref,
                image,
                tried,
                notes,
                "no label, catalog entry or version names one",
                checker,
                nim_version,
            )
        hand_tag = asked
        if image.tags and asked in image.tags and image.pinned and image.pinned != asked:
            hand_tag = image.pinned
            notes.append(
                f"{ref.image} is the same image as {ref.with_tag(hand_tag).image} (one digest); "
                f"the result names {hand_tag}, which cannot move to another build."
            )
        cnd_tag_source = "requested"
    else:
        images, problem = reader.images(ref, found, host_arch)
        image, build, tried, hand_tag, pinned = newest_with_build(
            source, ref, images, problem, checker, notes, asked
        )
        cnd_tag_source = "newest_with_build"
    return cnd_facts(source, image, build, tried, hand_tag, pinned, cnd_tag_source, notes)


def newest_with_build(
    source: str,
    ref: rl.ImageRef,
    images: list[cnd.CndImage],
    problem: str | None,
    checker: NgcChecker,
    notes: list[str],
    asked: str | None = None,
) -> tuple[cnd.CndImage, NgcBuild, list[dict[str, Any]], str, bool]:
    """(image, NGC build, mappings tried, tag to hand off, pinned) for the newest of `images`
    (the tagged images of the repository of `ref`, newest first, and why that list is empty or
    incomplete: `problem`) with a version tag and a known NGC build; else the newest listed only as
    latest whose build is known (pinned False). ngc_build_unknown when none has one. A note says
    which image was taken; `asked` is `latest` when the user named it."""
    build, image, tried, unpinned = None, None, [], None
    pinned = True
    for item in images:
        if item.pinned is None:
            unpinned = unpinned or item
            continue
        build, item_tried = first_build(checker, item, None)
        tried.extend(item_tried)
        if build is not None:
            image = item
            break
    if build is None and unpinned is not None:
        build, item_tried = first_build(checker, unpinned, None)
        tried.extend(item_tried)
        image, pinned = unpinned, False
    if build is None or image is None:
        newest = next((item for item in images if item.pinned), None)
        # With no version tag known, the repository alone: the user names the tag.
        shown = newest.ref.with_tag(newest.pinned) if newest else ref.with_tag(None)
        why = problem or (
            f"no tag of {ref.repository_ref} has a label, catalog entry or version that names one"
        )
        only_latest = newest is None and unpinned is not None
        raise build_unknown(
            source, shown, newest, tried, notes, why, checker, only_latest=only_latest
        )
    hand_tag = image.pinned or str(image.ref.tag)
    if pinned:
        skipped = images.index(image)
        newer = (
            f" with a version tag and a known NGC build; {skipped} newer image(s) have only "
            "latest or no known build (facts.mappings_tried)"
            if skipped
            else ""
        )
        named = "The tag latest can move to another build" if asked else "No tag was named"
        notes.append(
            f"{named}: {image.ref.with_tag(hand_tag).image} is the newest image of "
            f"{ref.repository_ref}{newer} (tags {', '.join(image.tags) or hand_tag})."
        )
    return image, build, tried, hand_tag, pinned


def fit_architectures(cnd_archs: Iterable[str], ngc_archs: list[str] | None) -> list[str] | None:
    """The CPU architectures fit checks for a CND image: those of the CND image (what runs) that
    the NGC build has too (its profiles are that build's); the CND image's alone when the build's
    are unknown. When the CND image's are unknown, the build's only when it has one: a build for
    amd64 and arm64 says nothing about which the CND image is (a DGX Spark image is arm64)."""
    mine = sorted(set(cnd_archs))
    if mine and ngc_archs:
        return sorted(set(mine) & set(ngc_archs)) or None
    if mine:
        return mine
    return list(ngc_archs) if ngc_archs and len(ngc_archs) == 1 else None


def cnd_facts(
    source: str,
    image: cnd.CndImage,
    build: NgcBuild,
    tried: list[dict[str, Any]],
    hand_tag: str,
    pinned: bool,
    cnd_tag_source: str,
    notes: list[str],
) -> tuple[dict[str, Any], dict[str, Any], ImageList]:
    """The resolve facts of a CND image mapped to an NGC build, with its repo and image list."""
    ref = image.ref
    rid = build.resource_id
    shown = ref.with_tag(hand_tag).image
    warnings: list[str] = []
    if build.target.mapping in ("version_label", "user_version"):
        given = (
            f"its {rl.VERSION_LABEL} label"
            if build.target.mapping == "version_label"
            else "the version the user gave (--nim-version)"
        )
        warnings.append(
            f"The NGC repository {rid} was taken from the path of {ref.repository_ref}; "
            f"{given} names only the version ({build.target.tag})."
        )
    if build.tag_source == "alias_replaced":
        warnings.append(
            f"{build.target.origin} names the moving NGC tag {build.target.tag}; its profiles are "
            f"those of {build.tag}, the build it stands for today, which can be newer than the "
            f"build {shown} was made from."
        )
    labels_status = (image.labels or {}).get("status")
    if cnd_tag_source == "requested" and labels_status == "unauthorized" and not image.catalog:
        warnings.append(
            f"The tag {hand_tag} could not be checked: {ref.host} needs a login and the "
            f"{cnd.CATALOGS[source]} catalog does not list it. Confirm with the user that "
            f"{shown} exists before the hand-off."
        )
    if not pinned:
        where = (
            "the TGC catalog lists no other tag, and its tags on tgcr need a login"
            if source == "tgcr"
            else f"{cnd.CATALOGS[source]} lists no other tag of that image"
        )
        digest = f" (digest {image.digest})" if image.digest else ""
        path_rid = cnd.ngc_repository(ref)
        if source != "tgcr":
            again = ""
        elif path_rid == rid:
            again = (
                "; its labels need a tgcr login, so also pass --nim-version with the NGC version "
                f"the user gives ({build.tag} only if that tag is the same image as "
                f"latest{digest})"
            )
        else:
            # --nim-version takes the NGC repository from the path, which is not this build's
            again = (
                f"; its labels need a tgcr login, and --nim-version cannot map a tag TGC does not "
                f"list: it takes the NGC repository from the path ({path_rid or 'no TEAM/REPO'}), "
                f"not {rid}, so such a tag is confirmed only after the pull, with the image's own "
                "list-model-profiles through nim-deploy"
            )
        warnings.append(
            f"{ref.repository_ref} is listed only as latest ({where}), so the profile cannot be "
            f"pinned to one image: never hand off latest. Today latest{digest} maps to "
            f"{build.image}. Ask the user for the image's own version tag and select again with "
            f"it{again}."
        )
    said = [*notes]
    said.append(
        f"{shown} maps to {build.image} (mapping {build.target.mapping}: {build.target.origin}); "
        "its profiles come from the NGC metadata of that build."
    )
    said.extend(
        f"Not used: {item['problem']}."
        for item in tried
        if item.get("problem") and item.get("image") == ref.image
    )
    said.extend(problem for problem in image.problems if problem)
    archs = fit_architectures(image.architectures, build.architectures)
    build_notes = list(build.notes)
    if image.architectures:
        unread = build.listed.unread_note() if build.listed.problem else None
        if unread in build_notes:
            # fit checks the CND image's own architectures: the build's are not needed
            build_notes.remove(unread)
            build_notes.append(
                f"The CPU architectures of {build.image} were not read ({build.listed.problem}); "
                f"those of {shown} ({', '.join(image.architectures)}) are checked."
            )
    elif archs:
        said.append(
            f"The CPU architectures of {shown} are not known; those of {build.image} "
            f"({', '.join(archs)}) are checked instead."
        )
    elif build.architectures:
        said.append(
            f"The CPU architectures of {shown} are not known, and {build.image} is built for "
            f"several ({', '.join(build.architectures)}), so its profiles that record none are "
            "unknown (arch_not_recorded)."
        )
    else:
        said.append(f"The CPU architectures of {shown} and {build.image} are not known.")
    said.extend(build_notes)
    facts: dict[str, Any] = {
        "source": source,
        "resource_id": rid,
        "model": rid.removeprefix("nim/"),
        "display_name": build.repo.get("displayName") or build.repo.get("name"),
        "tag": build.tag,
        "tag_source": build.tag_source,
        "image": shown,
        "pinned": pinned,
        "ngc_image": build.image,
        "cnd_repository": ref.repository_ref,
        "cnd_tag": hand_tag,
        "cnd_tag_source": cnd_tag_source,
        "cnd_tags": list(image.tags),
        "cnd_digest": image.digest,
        "cnd_labels": image.labels,
        "cnd_catalog": image.catalog,
        "mapping": build.target.mapping,
        "mappings_tried": tried,
        "architectures": archs,
        "cnd_architectures": list(image.architectures) or None,
        "ngc_architectures": build.architectures,
        "image_size_gb": round(image.size / 1e9, 2) if image.size else None,
        "source_registry": ref.host,
        "warnings": warnings,
        "notes": said,
    }
    return facts, build.repo, build.listed


def registry_warning(resolved: dict[str, Any], resource_id: str, tag: str) -> str | None:
    """For an image named on another registry: check that it carries the image the result names,
    when that is not the one named (the tag was chosen on NGC, select chose another image, or a
    search for the name found the NGC repository)."""
    registry = resolved.get("source_registry")
    if not registry:
        return None
    image = f"nvcr.io/{resource_id}:{tag}"
    if image != resolved.get("image"):
        why = f"The chosen image {image} is not the one named"
    elif resolved.get("tag_source") != "requested":
        why = f"The tag {tag} was chosen on NGC, not named"
    elif resolved.get("mapping") == "search":
        why = f"The NGC repository {resource_id} was found by a search for the name, not named"
    else:
        return None
    return f"{why}; check that {registry} carries {resource_id}:{tag}."


def unpinned_warning(tag: str) -> str | None:
    """For a tag that can move to another build (`latest` when no version tag is listed, or an
    alias with no build listed): a profile id cannot be pinned to it (SKILL.md, the hard rule to
    pin a version tag)."""
    if ngc_api.is_build(tag):
        return None
    what = "is not a version" if ngc_api.version_key(tag) is None else "is an alias with no build"
    return (
        f"The image tag {tag} {what}, so it can move to another build: the profile cannot be "
        "pinned to one image. Ask the user before the hand-off."
    )


def image_warnings(resolved: dict[str, Any], resource_id: str, tag: str) -> list[str]:
    """The warnings about the image a result names (resource_id:tag). A CND image has its own
    (cnd_facts): the NGC build only lends its profiles."""
    if resolved.get("source") in cnd.CNDS:
        return []
    found = (registry_warning(resolved, resource_id, tag), unpinned_warning(tag))
    return [warning for warning in found if warning]


CND_KEYS = (
    "pinned",
    "ngc_image",
    "cnd_repository",
    "cnd_tag",
    "cnd_tag_source",
    "cnd_tags",
    "cnd_digest",
    "cnd_labels",
    "cnd_catalog",
    "mappings_tried",
    "cnd_architectures",
    "ngc_architectures",
)


def cnd_keys(resolved: dict[str, Any]) -> dict[str, Any]:
    """The CND facts of resolve, for fit and select (none on NGC)."""
    if resolved.get("source") not in cnd.CNDS:
        return {}
    return {key: resolved.get(key) for key in CND_KEYS}


def cmd_locate(args: argparse.Namespace) -> NoReturn:
    try:
        found = locality.locate()
        facts = {
            "locality": found,
            "source": "ngc" if found["result"] == "outside_china" else None,
        }
        save(args.out, facts)
    except ns.ScriptError as exc:
        fail("locate", exc)
    ns.emit(True, "located", "run_resolve" if facts["source"] else "ask_user", facts)


def cmd_country(args: argparse.Namespace) -> NoReturn:
    """The country of this machine's public IP address, for --region: run only after the user
    agrees to the lookup."""
    code, problem = locality.lookup_country()
    facts: dict[str, Any] = {
        "country": code,
        "checked": "the public IP address of the machine the agent runs on",
        "lookup_host": urllib.parse.urlsplit(locality.COUNTRY_URL).hostname,
    }
    if code is None:
        ns.emit(
            False,
            "country_unknown",
            "ask_user",
            facts,
            [ns.error("country_lookup_failed", str(problem), COUNTRY_FAILED_FIX)],
            ns.EXIT_EXTERNAL,
        )
    ns.emit(True, "country_found", "rerun_with_region", facts)


def cmd_resolve(args: argparse.Namespace) -> NoReturn:
    try:
        source = cnd.choose_source(args.model, args.source)
        facts, _, _ = resolve_model(args.model, args.tag, source, args.nim_version)
        facts["warnings"].extend(image_warnings(facts, facts["resource_id"], facts["tag"]))
        save(args.out, facts)
    except ns.ScriptError as exc:
        fail("resolve", exc)
    ns.emit(True, "resolved", "run_inspect", facts)


# --- inspect --------------------------------------------------------------------------------


def check_gpus(value: str) -> str:
    if value == "all" or re.fullmatch(r"\d+(,\d+)*", value):
        return value
    raise ns.ScriptError(
        "bad_gpus",
        f"--gpus must be 'all' or GPU indices like 0,1; got {value!r}",
        "Pass --gpus all",
    )


def cmd_inspect(args: argparse.Namespace) -> NoReturn:
    try:
        if args.target == "k8s":
            if args.gpus is not None:
                raise ns.ScriptError(
                    "bad_input",
                    "--gpus applies only to --target host; on Kubernetes the scheduler picks "
                    "the GPUs",
                    "Drop --gpus; to limit the GPU count, pass --max-gpus to select",
                )
            if args.cache_dir is not None:
                raise ns.ScriptError(
                    "bad_input",
                    "--cache-dir applies only to --target host; a cluster's disk is not checked",
                    "Drop --cache-dir",
                )
            if not args.context:
                raise ns.ScriptError(
                    "context_required",
                    "--target k8s needs both --context and --namespace (the current context is "
                    "never assumed)",
                    "Ask the user for the kube context and namespace, then pass --context and "
                    "--namespace",
                )
            # A missing namespace is reported after the named context is checked against the
            # kubeconfig, so a context it lacks and the namespace are asked for at once.
            env = env_inspect.inspect_k8s(args.context, args.namespace)
        else:
            if args.context or args.namespace:
                raise ns.ScriptError(
                    "bad_input",
                    "--context and --namespace apply only to --target k8s",
                    "Add --target k8s, or drop --context and --namespace",
                )
            gpus = check_gpus(args.gpus or "all")
            env = env_inspect.inspect_host(gpus=gpus, cache_dir=args.cache_dir)
        save(args.out, env)
    except ns.ScriptError as exc:
        fail("inspect", exc)
    ns.emit(True, "inspected", "run_fit" if env.get("gpus_known") else "ask_user", env)


# --- fit ------------------------------------------------------------------------------------


def load_env(path: str) -> dict[str, Any]:
    env = ns.load_json_file(Path(path), code="env_file_invalid")
    fix = "Pass the file written by: nimselect.py inspect --out env.json"
    if not isinstance(env, dict) or env.get("schema") != ENV_SCHEMA:
        raise ns.ScriptError(
            "env_file_invalid", f"{path} is not a nim-select environment ({ENV_SCHEMA})", fix
        )
    if not isinstance(env.get("groups"), list):
        raise ns.ScriptError("env_file_invalid", f"{path} has no 'groups' list", fix)
    return env


def load_metadata(path: str) -> dict[str, Any]:
    data = ns.load_json_file(Path(path), code="metadata_file_invalid")
    if not isinstance(data, dict) or not isinstance(data.get("profiles"), list):
        raise ns.ScriptError(
            "metadata_file_invalid",
            f"{path} is not a NIM metadata document (no 'profiles' list)",
            "Pass the JSON body of GET /v2/nim/metadata for this image",
        )
    return data


def environment_summary(env: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "name",
        "pci_device_id",
        "pci_device_ids",  # more than one (a product label two ids share): pci_device_id is None
        "count",
        "idle",
        "idle_indices",
        "memory_gb",
        "memory_mib",
        "unified_memory",
        "compute_capability",
        "placement",
    )
    return {
        "target": env.get("target"),
        "context": env.get("context"),
        "namespace": env.get("namespace"),
        "captured_at": env.get("captured_at"),
        "gpus_known": env.get("gpus_known"),
        "driver_version": env.get("driver_version"),
        "groups": [
            # a Kubernetes group has its own nodes' CPU architecture; a host's is the env's
            {**{key: group.get(key) for key in keys}, "arch": group.get("arch", env.get("arch"))}
            for group in env.get("groups") or []
            if isinstance(group, dict)
        ],
        "conditions": [
            str(c.get("code")) for c in env.get("conditions") or [] if isinstance(c, dict)
        ],
    }


def resolve_offline(model: str, tag_flag: str | None) -> dict[str, Any] | None:
    """With --metadata-file, the facts resolve would give from the reference alone, if it names
    a repository and tag (used when NGC cannot be reached)."""
    ref = ngc_api.parse_model_ref(model)
    tag = tag_flag or ref.tag
    if ref.resource_id is None or not tag or tag == "latest":
        return None
    return {
        "resource_id": ref.resource_id,
        "model": ref.resource_id.removeprefix("nim/"),
        "tag": tag,
        "tag_source": "requested",
        "image": f"nvcr.io/{ref.resource_id}:{tag}",
        "architectures": None,
        "mapping": ref.mapping,
        "source_registry": ref.source_registry,
        "warnings": restate_mirror_warning(ref, list(ref.warnings), True),
        "notes": ["NGC could not be reached; the tag was not checked against the repository"],
    }


def cmd_fit(args: argparse.Namespace) -> NoReturn:
    facts: dict[str, Any] = {}
    try:
        check_limits(args)
        source_of = cnd.choose_source(args.model, args.source)
        env = load_env(args.env)
        # As in select: env problems (MIG, no GPUs) stop here, before any NGC call.
        fit.evaluate({"profiles": []}, env, fit_options(args, None))
        metadata = load_metadata(args.metadata_file) if args.metadata_file else None
        repo: dict[str, Any] | None = None
        listed: ImageList | None = None
        digests: dict[str, str] = {}
        try:
            resolved, repo, listed = resolve_model(
                args.model, args.tag, source_of, args.nim_version, env.get("arch")
            )
            digests = listed.digests
        except ns.ScriptError as exc:
            # A CND image needs NGC to check the build its labels name: never offline.
            if exc.code != "ngc_unreachable" or metadata is None or source_of != "ngc":
                raise
            offline = resolve_offline(args.model, args.tag)
            if offline is None:
                raise
            ns.log(f"NGC unreachable ({exc.message}); using --metadata-file as given")
            resolved = offline
        # The tag's own architectures (image list), else the record's when it is this image.
        architectures = resolved.get("architectures")
        # Offline (repo None) the tag was not checked against the repository: no image warnings.
        rid, tag = str(resolved["resource_id"]), str(resolved["tag"])
        image_warned = image_warnings(resolved, rid, tag) if repo is not None else []
        on_cnd = source_of in cnd.CNDS
        facts = {
            "source": source_of,
            "resource_id": resolved["resource_id"],
            "model": resolved["model"],
            "tag": resolved["tag"],
            "image": resolved["image"],
            "mapping": resolved["mapping"],
            **cnd_keys(resolved),
            "warnings": [*resolved["warnings"], *image_warned],
        }
        source = "file"
        if metadata is None:
            metadata = ngc_api.get_metadata(resolved["resource_id"], resolved["tag"])
            source = "published"
        # A 404, or an answer that lists no profile: nothing can be checked before the pull.
        if metadata is None or not fit.usable_profiles(metadata):
            if not on_cnd:  # NGC tags are no tags of a CND image
                facts["tags_to_try"] = tags_to_try(repo, resolved["tag"], digests)
            where = (
                f"{args.metadata_file} lists" if source == "file" else "the NIM Metadata API has"
            )
            checked = resolved["ngc_image"] if on_cnd else resolved["image"]
            ns.emit(
                False,
                "no_metadata",
                "ask_user",
                {**facts, "notes": resolved.get("notes", [])} if on_cnd else facts,
                [
                    ns.error(
                        "no_metadata",
                        f"{where} no profiles for {checked}",
                        CND_NO_METADATA_FIX if on_cnd else NO_METADATA_FIX,
                    )
                ],
                ns.EXIT_PRECONDITION,
            )
        # As in a select of this image: the other images of its repository that a select without
        # a tag checks lend their test records (record_images). With --metadata-file (used instead
        # of the Metadata API), and on a CND, none does.
        records: dict[str, list[fit.OtherRecord]] | None = None
        unread: list[str] = []
        unlisted: str | None = None
        if source == "published" and repo is not None and listed is not None and not on_cnd:
            lenders, unlisted = record_images(
                resolved, repo, listed, env, choose.DEFAULT_MAX_IMAGES
            )
            lent: dict[str, dict[str, Any] | None] = {}
            unread = read_records(lenders, lent)
            records = records_for(choose.Candidate(rid, tag, "requested", None), lenders, lent)
        # As in select: an image whose CPU architectures NGC does not list is `unknown`
        # (arch_not_recorded) for profiles that record none either, never a fit by default.
        options = dataclasses.replace(
            fit_options(args, architectures), arch_required=True, other_records=records
        )
        result = fit.evaluate(metadata, env, options)
        unread_note = unread_records_note(unread)
        notes = [
            *resolved.get("notes", []),
            *([unread_note] if unread_note else []),
            *([unlisted] if unlisted else []),
            *result.get("notes", []),
        ]
        if source == "file" and resolved.get("tag_source") != "requested":
            notes.append(
                f"The profiles come from --metadata-file but the tag {resolved['tag']} was picked "
                "from the repository; check that the file belongs to that tag, or pass --tag"
            )
        add_fit_result(facts, result, env, source, notes)
        save(args.out, facts)
    except ns.ScriptError as exc:
        fail("fit", exc, facts)
    finish(facts)


def check_limits(args: argparse.Namespace) -> None:
    for flag, value, example in (
        ("--max-gpus", args.max_gpus, "1"),
        ("--top", args.top, "5"),
        ("--max-images", getattr(args, "max_images", None), "6"),
    ):
        if value is not None and value < 1:
            raise ns.ScriptError("bad_input", f"{flag} must be 1 or more", f"Pass {flag} {example}")


def fit_options(args: argparse.Namespace, architectures: list[str] | None) -> fit.FitOptions:
    return fit.FitOptions(
        max_gpus=args.max_gpus,
        need_lora=args.need_lora,
        backend=args.backend,
        prefer=args.prefer,
        top=args.top,
        image_architectures=architectures,
    )


def handoff_gpus(group: dict[str, Any], count: int, env: dict[str, Any]) -> dict[str, Any]:
    """The GPUs a hand-off names. Host: the first `count` idle GPUs of the chosen group, as
    `gpu_ids` and as `gpus` in Docker's device form (`device=3`: a bare `3` reads as a count).
    Kubernetes: the scheduler picks the GPUs, so `gpus` is the count and `gpu_ids` None."""
    if env.get("target") == "k8s":
        return {"gpus": count, "gpu_ids": None, "gpu_count": count,
                "gpu_memory_mib": group.get("memory_mib")}
    idle = group.get("idle_indices")
    if idle is None and group.get("idle") == group.get("count"):
        idle = group.get("indices")  # an environment saved without idle_indices, every GPU idle
    ids = [i for i in idle or [] if isinstance(i, int)][:count]
    if len(ids) < count:
        return {"gpus": None, "gpu_ids": None, "gpu_count": count}
    return {"gpus": "device=" + ",".join(map(str, ids)), "gpu_ids": ids, "gpu_count": count}


def handoff_resources(chosen: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    """Kubernetes only; a host hand-off has none of these. The CPU cores and host memory the
    profile records, in the units of a pod's resource requests (`cpu` in cores, `memory` in the
    metadata's decimal GB, such as `64G`), each None when not recorded; and `placement`, None
    unless a GPU node the pod can be placed on is not known to have them. A pod that requests only
    GPUs can be placed on any node with enough free GPUs, of the group checked or another, so the
    deployment must then request them or be pinned to a node that has them, and the user confirms
    how it is placed."""
    if env.get("target") != "k8s":
        return {}
    cpu = chosen.get("min_cpu_count")
    per_device = chosen.get("system_memory_per_device_gb")
    gpus = int(chosen["gpus_required"])
    gb = None if per_device is None else Decimal(str(per_device)) * gpus
    shown = None if gb is None else f"{gb.normalize():f}"
    # Every group whose nodes have the GPUs the pod requests, with its nodes' smallest values; a
    # file saved without them: those of the cluster's GPU nodes, which are no larger.
    groups = env.get("groups") or []
    can_take = [g for g in groups if isinstance(g, dict) and _at_least(g.get("count"), gpus)]
    every_node = all(
        (cpu is None or _at_least(g.get("smallest_cpu_count", env.get("cpu_count")), cpu))
        and (
            gb is None
            or _at_least(g.get("smallest_host_memory_gb", env.get("memory_gb")), float(gb))
        )
        for g in can_take or [{}]
    )
    placement = None
    if not every_node:
        needs = [f"{cpu} CPU cores"] if cpu is not None else []
        needs += [f"{shown} GB of host memory"] if shown is not None else []
        placement = (
            f"Not every GPU node the pod can be placed on is known to have {' and '.join(needs)}: "
            "a pod that requests only GPUs can be placed on any node with enough free GPUs. The "
            "deployment must request them or be pinned to a node that has them. Confirm with the "
            "user how it is placed before the hand-off."
        )
    return {"cpu": cpu, "memory": None if shown is None else f"{shown}G", "placement": placement}


def _at_least(have: object, need: float) -> bool:
    """`have` is a number (not a bool) of at least `need`; None or anything else is not."""
    return isinstance(have, int | float) and not isinstance(have, bool) and have >= need


def add_fit_result(
    facts: dict[str, Any],
    result: dict[str, Any],
    env: dict[str, Any],
    source: str,
    notes: list[str],
    kind: str | None = None,
) -> None:
    """Add what fit found for facts["image"] to facts, and the hand-off for the profile that fits
    (or, for an unknown verdict, the best unconfirmed one), naming the GPUs to use and, on
    Kubernetes, the CPU cores and memory it needs (handoff_resources). `backend` tells whether
    nim-tune can take it for tuning: it tunes vLLM NIMs only. An alternative of this image gets its
    `image` (null for a CND image listed only as latest) and `kind`, the image's kind in
    facts.images_considered (select; None for fit)."""
    facts.update({"metadata": source, "environment_summary": environment_summary(env), **result})
    alternative = facts.get("alternative")
    if alternative and not PROFILE_ID.fullmatch(str(alternative.get("profile_id") or "")):
        alternative = None  # select --profile could not take its id: never offered
    elif alternative and "image" not in alternative:  # fit's own: a profile of this image
        image = facts["image"] if facts.get("pinned", True) else None
        alternative = {"image": image, "kind": kind, **alternative}
    facts["alternative"] = alternative
    offered = alternative_note(facts)
    facts["alternative"] = alternative if offered else None  # never offered without its note
    facts["notes"] = [*notes, offered] if offered else notes
    chosen = result.get("recommended") or result.get("best_unconfirmed")
    if not chosen:
        return
    group = result["group"]
    facts["handoff"] = {
        "model": facts["model"],
        # Never a CND image tagged `latest`: facts.pinned is false when no other tag is known.
        "image": facts["image"] if facts.get("pinned", True) else None,
        "model_profile": chosen["profile_id"],
        "verdict": chosen["verdict"],
        "backend": chosen["backend"],
        **handoff_gpus(group, int(chosen["gpus_required"]), env),
        "gpu_name": group.get("name"),
        **handoff_resources(chosen, env),
        # The metadata screens profiles before the pull; the image confirms one after it.
        "confirm_with": "list-model-profiles",
    }
    if facts["handoff"]["gpus"] is None:
        why = (
            "Fewer GPUs are known to be idle than the profile needs"
            if "gpu_use_unknown" in chosen["reasons"]
            else "Which GPUs are idle is not recorded"
        )
        facts["warnings"] = [
            *facts.get("warnings", []),
            f"{why}; ask the user which GPUs of {group.get('name')} to use before the hand-off.",
        ]


# How the image of a confirmed facts.alternative stands to the image select chose
# (image_relation), in its offer and in its note.
OFFER_IMAGE = {
    "older": "the older image",
    "newer": "the newer image",
    "variant": "the hardware variant",
}
NOTE_IMAGE = {
    "older": "an older version",
    "newer": "a newer version",
    "variant": "a hardware variant",
}


def image_relation(chosen: dict[str, Any], other: dict[str, Any]) -> str:
    """How the image `other` (an images_considered entry) stands to `chosen` (its `resource_id`
    and `tag`): `older` or `newer` by the version numbers of their tags when both are of one
    repository, its tag for a GPU (`1.8.0-RTX`) included; else, and with the same numbers,
    `other`'s kind (`variant`; `older`: an older version of the repository asked for)."""
    if other["resource_id"] == chosen["resource_id"]:
        mine, theirs = ngc_api.parse_tag(str(chosen["tag"])), ngc_api.parse_tag(str(other["tag"]))
        if mine and theirs and mine[0] != theirs[0]:
            return "newer" if theirs[0] > mine[0] else "older"
    return str(other.get("kind"))


# Where the select of a taken alternative writes, so the first select's file stays as it was: the
# user may still go back to the recommended profile, whose hand-off reads that first file.
ALTERNATIVE_OUT = "~/.cache/nim-select/select-alternative.json"


def pinnable(entry: dict[str, Any]) -> bool:
    """Whether the note on facts.alternative can name the image of an images_considered entry in
    a select command: a fixed build (never a moving alias such as `1.7`, or latest: hard rule 5),
    with a repository and a tag a command line can carry as one word each (is_build alone also
    reads `2.1.4` followed by a line break as a build)."""
    tag, resource_id = str(entry.get("tag") or ""), str(entry.get("resource_id") or "")
    return (
        ngc_api.is_build(tag)
        and COMMAND_WORD.fullmatch(tag) is not None
        and COMMAND_WORD.fullmatch(resource_id) is not None
    )


def alternative_note(facts: dict[str, Any]) -> str | None:
    """The note on facts.alternative, a profile with a faster precision (more accurate with
    --prefer accuracy): offer it in the confirm question, and on a yes, select that profile alone
    (--tag, --profile) and hand off that result. `confirmed`: it fits in another image select
    checked, whose select gives fits; else it is a profile of this image that only lacks a test on
    this GPU, whose select gives unknown. None when there is no alternative, when a confirmed one
    names an image that facts.images_considered does not list (select never offers one) or that
    is not pinnable, when the profiles came from fit --metadata-file (the select reads what NGC
    publishes, which may lack the image or the profile), when an NGC image's tag is no fixed build
    (latest, an alias: it can move), or when a command line cannot carry the repository or tag."""
    alternative = facts.get("alternative")
    if not alternative or facts.get("metadata") == "file":
        return None
    pid = alternative["profile_id"]
    if alternative.get("confirmed"):
        image = alternative.get("image")
        checked = facts.get("images_considered") or []
        entry = next((item for item in checked if item["image"] == image), None)
        if entry is None or not pinnable(entry):
            return None
        chosen = {"resource_id": facts["resource_id"], "tag": facts["tag"]}
        where = NOTE_IMAGE.get(image_relation(chosen, entry), "another image")
        return (
            f"facts.alternative, the profile {pid} of {image}, is {alternative['offer']}: offer "
            "it in the confirm question beside the recommended profile, as a choice; it fits, but "
            f"in {where}, so ask whether that image is acceptable. On a yes, run select "
            f"{entry['resource_id']} --tag {entry['tag']} --profile={pid} with the other options "
            f"as before but --out {ALTERNATIVE_OUT} (the first select's file stays), and hand off "
            "that result and its file (verdict fits, with that image)."
        )
    offer = (
        f"facts.alternative, the profile {pid}, is {alternative['offer']}: offer it in the "
        "confirm question beside the recommended profile, as a choice that stays unconfirmed."
    )
    if not facts.get("pinned", True):  # a CND image listed only as latest: no tag to pin yet
        return (
            f"{offer} On a yes, once the user gives the image's own version tag, select again "
            f"with it, --profile={pid} and --out {ALTERNATIVE_OUT} (the first select's file "
            "stays), and hand off that result and its file (verdict unknown: say the profile is "
            "unconfirmed)."
        )
    on_cnd = facts.get("source") in cnd.CNDS
    model = facts["cnd_repository"] if on_cnd else facts["resource_id"]
    tag = facts["cnd_tag"] if on_cnd else facts["tag"]
    if not on_cnd and not ngc_api.is_build(str(tag)):
        return None  # latest, or an alias with no build: no profile is pinned to it (hard rule 5)
    if not (COMMAND_WORD.fullmatch(str(model)) and COMMAND_WORD.fullmatch(str(tag))):
        return None  # the command line could not carry them: never offered
    return (
        f"{offer} On a yes, run select {model} --tag {tag} --profile={pid} with the other options "
        f"as before but --out {ALTERNATIVE_OUT} (the first select's file stays), and hand off that "
        "result and its file (verdict unknown: say the profile is unconfirmed)."
    )


def finish(facts: dict[str, Any]) -> NoReturn:
    """Report a completed evaluation: ok, whatever the verdict. With no image to hand off (a CND
    image known only as `latest`), the user is asked for its tag first."""
    ready = facts.get("recommended") and (facts.get("handoff") or {}).get("image")
    ns.emit(True, str(facts["verdict"]), "confirm_with_user" if ready else "ask_user", facts)


# --- select ---------------------------------------------------------------------------------


@dataclasses.dataclass
class Search:
    """What the candidate search found besides the images to check."""

    variants: list[choose.Candidate] = dataclasses.field(default_factory=list)
    production: list[str] = dataclasses.field(default_factory=list)
    other: list[str] = dataclasses.field(default_factory=list)
    unchecked: list[str] = dataclasses.field(default_factory=list)
    notes: list[str] = dataclasses.field(default_factory=list)


def find_variants(base_id: str, slots: int) -> Search:
    """Hardware-variant repos of base_id, with their newest tag, and the related repos left out."""
    found = Search()
    leaf = base_id.rsplit("/", 1)[-1]
    try:
        hits = ngc_api.search_nims(leaf, page_size=50)
    except ns.ScriptError as exc:
        if exc.code in NGC_STOPS:
            raise
        found.notes.append(f"Hardware variants were not searched: {exc.message}")
        return found
    related = choose.split_variants(base_id, [str(h["resource_id"]) for h in hits])
    found.production, found.other = related.production, related.other
    found.unchecked = related.variants[max(slots, 0) :]
    for rid in related.variants[: max(slots, 0)]:
        try:
            repo = ngc_api.get_repo(rid)
            if repo is None:
                continue
            listed = read_image_list(rid)
            tag = ngc_api.choose_tag(repo, None, listed.digests)[0]
        except ns.ScriptError as exc:
            if exc.code in NGC_STOPS:
                raise
            found.notes.append(f"The variant {rid} was skipped: {exc.message}")
            continue
        archs, _, note = image_facts(repo, tag, listed)
        if note:
            found.notes.append(note)
        elif archs is None and listed.problem:
            found.notes.append(listed.unread_note())
        found.variants.append(
            choose.Candidate(rid, tag, "variant", tuple(archs) if archs else None)
        )
    return found


def same_repo_candidate(
    rid: str, repo: dict[str, Any], listed: ImageList, tag: str, kind: choose.Kind
) -> choose.Candidate:
    """A candidate of the requested repository, with the CPU architectures NGC records for it."""
    archs = image_facts(repo, tag, listed)[0]
    return choose.Candidate(rid, tag, kind, tuple(archs) if archs else None)


def later_candidates(
    rid: str,
    repo: dict[str, Any],
    listed: ImageList,
    newest: str,
    variants: list[choose.Candidate],
    env: dict[str, Any],
    limit: int,
) -> list[choose.Candidate]:
    """select's candidates after the newest image (tag `newest`) of `rid`, in order: the hardware
    variants given (other repositories), the repository's tag for a GPU found (`1.8.0-RTX`), then
    older versions: the earlier builds of the newest's major.minor line, newest first, as many as
    `limit` leaves room for before the newest build of the two older major.minor lines
    (choose.line_build_count), and those builds."""
    tags = repo_tags(repo)
    gpu_names = [str(g.get("name") or "") for g in env.get("groups") or [] if isinstance(g, dict)]
    gpu_tag = choose.gpu_variant_tag(tags, gpu_names)
    lines = choose.older_tags(tags, newest, digests=listed.digests)
    ahead = 1 + len(variants) + (1 if gpu_tag else 0)
    count = choose.line_build_count(limit, ahead, len(lines))
    older = [*choose.line_builds(tags, newest, listed.digests, count), *lines]
    return [
        *variants,
        *([same_repo_candidate(rid, repo, listed, gpu_tag, "variant")] if gpu_tag else []),
        *(same_repo_candidate(rid, repo, listed, tag, "older") for tag in older),
    ]


def build_candidates(
    resolved: dict[str, Any],
    repo: dict[str, Any],
    listed: ImageList,
    env: dict[str, Any],
    limit: int,
) -> tuple[list[choose.Candidate], list[str]]:
    """The images to check, in order, and notes about the search."""
    rid, tag = str(resolved["resource_id"]), str(resolved["tag"])
    archs = resolved.get("architectures")
    first = tuple(archs) if archs else None
    if one_image(resolved):
        return [choose.Candidate(rid, tag, "requested", first)], []
    search = find_variants(rid, limit - 1)
    ordered = [
        choose.Candidate(rid, tag, "newest", first),
        *later_candidates(rid, repo, listed, tag, search.variants, env, limit),
    ]
    kept, dropped = choose.plan(ordered, limit)
    notes = list(search.notes)
    resolve_notes = resolved.get("notes", [])
    unread = listed.unread_note() if listed.problem else None
    if unread and unread not in resolve_notes:
        if any(c.resource_id == rid and c.architectures is None for c in kept):
            if any(str(listed.problem) in note for note in resolve_notes):
                # resolve already named the problem (the newest image's architectures came
                # from the repository record): only say what it means for the others.
                unread = (
                    f"The CPU architectures of the other {rid} images were not read either; "
                    "their profiles that record none are unknown (arch_not_recorded)"
                )
            notes.append(unread)
    if search.production:
        notes.append(
            "Production-branch images were not considered automatically: "
            + ", ".join(search.production)
            + ". Name one with select or fit to check it."
        )
    if search.other:
        notes.append(
            "Related repositories that are other models or builds were not considered: "
            + ", ".join(search.other)
            + ". Name one with select or fit to check it."
        )
    skipped = [c.image for c in dropped] + [f"nvcr.io/{r}" for r in search.unchecked]
    if skipped:
        notes.append(
            f"Not checked (limit --max-images {limit}): "
            + ", ".join(skipped)
            + ". Raise --max-images, or name one with --tag."
        )
    return kept, notes


def one_image(resolved: dict[str, Any]) -> bool:
    """Whether select checks only the image named: a tag in MODEL or --tag (a moving alias is
    replaced by its build)."""
    return resolved.get("tag_source") in ("requested", "alias_replaced")


def record_images(
    resolved: dict[str, Any],
    repo: dict[str, Any],
    listed: ImageList,
    env: dict[str, Any],
    limit: int,
) -> tuple[list[choose.Candidate], str | None]:
    """The images whose test records a check of one image (select with a tag, fit) reads beside
    it, so that image gets the verdict a select without a tag gives it: the other images of its
    repository that such a select checks (build_candidates, within `limit`), and a note when they
    could not be found. Hardware variants are other repositories and lend no record. None when
    the repository lists no tag. `limit` counts the image checked too: at most `limit` - 1
    others, and none, with no search for them, for 1. NGC stopping (NGC_STOPS) while they are
    searched for leaves none: they are only evidence, so the image is still checked."""
    rid, tag = str(resolved["resource_id"]), str(resolved["tag"])
    if limit <= 1:
        return [], None
    try:
        newest, source, _ = ngc_api.choose_tag(repo, None, listed.digests)
    except ns.ScriptError:
        return [], None
    try:
        plan, _ = build_candidates(
            {"resource_id": rid, "tag": newest, "tag_source": source}, repo, listed, env, limit
        )
    except ns.ScriptError as exc:
        if exc.code not in NGC_STOPS:
            raise
        ns.log(f"the other images of {rid} were not searched for their records: {exc.message}")
        return [], (
            f"The other images of {rid} that a select without a tag checks were not read "
            f"({exc.message}): the tests recorded there for the same profile ids "
            "(tested_in_other_image) were not counted, so a select once NGC answers may "
            "recommend another profile or image."
        )
    return [c for c in plan if c.resource_id == rid and c.tag != tag][: limit - 1], None


def read_records(
    lenders: Iterable[choose.Candidate], documents: dict[str, dict[str, Any] | None]
) -> list[str]:
    """Read the metadata of `lenders` (record_images) into `documents`, by image, for their
    records alone; the images whose metadata could not be read. They are only evidence: when NGC
    stops (NGC_STOPS) no more is asked for, and the rest count as not read, so the image checked
    is still judged, with the records read. An error reading that image itself stops the
    command, as in a select of several images."""
    lenders = list(lenders)
    unread: list[str] = []
    for index, lender in enumerate(lenders):
        ns.log(f"reading the test records of {lender.image}")
        try:
            documents[lender.image] = ngc_api.get_metadata(lender.resource_id, lender.tag)
        except ns.ScriptError as exc:
            unread.append(lender.image)
            if exc.code in NGC_STOPS:
                ns.log(f"NGC stopped ({exc.message}): no other image is read for its records")
                unread.extend(other.image for other in lenders[index + 1 :])
                break
    return unread


def unread_records_note(images: list[str]) -> str | None:
    """A note naming the images of the repository whose metadata could not be read: the tests
    they record for the same profile ids (tested_in_other_image) were not counted."""
    if not images:
        return None
    return (
        f"Could not read the metadata of {', '.join(dict.fromkeys(images))}: the tests recorded "
        "there for the same profile ids (tested_in_other_image) were not counted, so a select "
        "once it can be read may recommend another profile or image."
    )


NAMED_TAGS = 10  # version tags a note names


def unchecked_newer_note(
    base_id: str,
    repo: dict[str, Any],
    listed: ImageList,
    chosen: choose.Candidate,
    considered: list[dict[str, Any]],
) -> str | None:
    """A note naming the version tags of the requested repository that are newer than the image
    chosen, an older version of it, and were not checked: image_choice.reason speaks only of the
    newer images checked. None when there are none, and for any other kind of choice: a hardware
    build (`1.8.0-RTX`) is taken for the GPU, not because the newer builds are not confirmed to
    fit. The older versions are checked newest first, so the --max-images note never names one
    newer than the chosen image."""
    if chosen.resource_id != base_id or chosen.kind != "older":
        return None
    checked = [item["tag"] for item in considered if item["resource_id"] == base_id]
    newer = choose.unchecked_newer(repo_tags(repo), chosen.tag, checked, listed.digests)
    if not newer:
        return None
    named = ", ".join(newer[:NAMED_TAGS])
    more = f" and {len(newer) - NAMED_TAGS} more" if len(newer) > NAMED_TAGS else ""
    return (
        f"Newer version tags of {base_id} that were not checked: {named}{more}. Whether they fit "
        "is not known; name one with --tag to check it."
    )


def read_image(
    candidate: choose.Candidate,
    documents: dict[str, dict[str, Any] | None] | None = None,
    profile: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Fetch one image's metadata: (its images_considered entry, with no verdict yet; the
    document to fit, or None when there is nothing to fit). `documents`, when given, receives the
    metadata document read (None: none published), by image. `profile` (select --profile): an
    image whose published profiles do not include that exact id gets no verdict and error_code
    profile_not_found."""
    entry: dict[str, Any] = {
        "image": candidate.image,
        "resource_id": candidate.resource_id,
        "tag": candidate.tag,
        "kind": candidate.kind,
        "metadata": "published",
        "verdict": None,
        "profile_id": None,
        "tier": None,
        "gpus": None,
        "reasons": [],
        "rejected_counts": {},
        "error": None,
        "error_code": None,
    }
    try:
        metadata = ngc_api.get_metadata(candidate.resource_id, candidate.tag)
    except ns.ScriptError as exc:
        if exc.code in NGC_STOPS:
            raise
        entry.update(metadata="error", error=exc.message, error_code=exc.code)
        return entry, None
    if documents is not None:
        documents[candidate.image] = metadata
    # A 404, or an answer that lists no profile: nothing to check here; the search goes on.
    if metadata is None or not fit.usable_profiles(metadata):
        entry["metadata"] = "not_published"
        return entry, None
    ids = [p.get("profile_id") for p in fit.usable_profiles(metadata)]
    if profile is not None and profile not in ids:
        entry.update(
            error=f"the metadata of {candidate.image} has no profile {profile}",
            error_code=choose.PROFILE_NOT_FOUND,
        )
        return entry, None
    return entry, metadata


def fit_image(
    candidate: choose.Candidate,
    metadata: dict[str, Any],
    env: dict[str, Any],
    options: fit.FitOptions,
    profile: str | None = None,
    records: dict[str, list[fit.OtherRecord]] | None = None,
) -> dict[str, Any]:
    """The fit of one image's published profiles, as select fits them: with the CPU
    architectures NGC records for the image, and the records of the other images of its
    repository it read (records_for)."""
    archs = list(candidate.architectures) if candidate.architectures else None
    # An image whose CPU architectures NGC does not list is `unknown` (arch_not_recorded) for
    # profiles that record none either, never a fit by default. With `profile`, fit reports that
    # profile alone, on the GPU type it chooses on every profile.
    return fit.evaluate(
        metadata,
        env,
        dataclasses.replace(
            options,
            image_architectures=archs,
            arch_required=True,
            profile=profile,
            other_records=records,
        ),
    )


def judge_image(
    candidate: choose.Candidate,
    entry: dict[str, Any],
    metadata: dict[str, Any],
    env: dict[str, Any],
    options: fit.FitOptions,
    profile: str | None = None,
    records: dict[str, list[fit.OtherRecord]] | None = None,
) -> dict[str, Any]:
    """Fit one image read (read_image, fit_image) and give its entry the verdict and the best
    profile: the fit result. Judged again with more records, the entry is updated in full."""
    result = fit_image(candidate, metadata, env, options, profile, records)
    best = result.get("recommended") or result.get("best_unconfirmed")
    entry.update(
        verdict=result["verdict"],
        profile_id=best["profile_id"] if best else None,
        tier=best["tier"] if best else None,
        gpus=best["gpus_required"] if best else None,
        # the best profile's reasons: for an unknown image, the checks it leaves open (choose.pick)
        reasons=list(best["reasons"]) if best else [],
        rejected_counts=result["rejected"]["counts"],
    )
    return result


def consider(
    candidate: choose.Candidate,
    env: dict[str, Any],
    options: fit.FitOptions,
    documents: dict[str, dict[str, Any] | None] | None = None,
    profile: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Fetch one image's metadata and fit it alone, with no record of another image:
    (images_considered entry, fit result or None). `documents` and `profile` as for read_image;
    `profile` fits that profile alone, on the GPU type chosen without it."""
    entry, metadata = read_image(candidate, documents, profile)
    if metadata is None:
        return entry, None
    return entry, judge_image(candidate, entry, metadata, env, options, profile)


def newest_first(tag: str) -> tuple[bool, Any]:
    """A sort key (reverse=True) that puts version tags newest first, other tags (`latest`) last."""
    key = ngc_api.version_key(tag)
    return key is not None, key or ()


def records_for(
    candidate: choose.Candidate,
    read: Iterable[choose.Candidate],
    documents: dict[str, dict[str, Any] | None],
) -> dict[str, list[fit.OtherRecord]]:
    """fit.FitOptions.other_records of one image: what the other images of its own repository
    among `read` record in their published `documents` (by image), newest version first. Another
    repository's images (a hardware variant) never count, nor does the image itself."""
    others = [
        other
        for other in dict.fromkeys(read)
        if other.resource_id == candidate.resource_id
        and other.tag != candidate.tag
        and documents.get(other.image)
    ]
    others.sort(key=lambda other: newest_first(other.tag), reverse=True)
    return fit.other_records((other.tag, documents[other.image] or {}) for other in others)


def judge_read(
    read: list[tuple[choose.Candidate, dict[str, Any], dict[str, Any] | None]],
    documents: dict[str, dict[str, Any] | None],
    env: dict[str, Any],
    options: fit.FitOptions,
    profile: str | None = None,
    lenders: Iterable[choose.Candidate] = (),
) -> list[dict[str, Any] | None]:
    """select's verdicts: each image read (candidate, its entry, the document to fit or None) is
    fitted with the records of the other images of its repository that were read, `lenders`
    (record_images) too (records_for), so a profile id one of them records as built for or
    tested on this GPU counts as tested in the image judged. The fit results, in order (None:
    nothing to fit)."""
    checked = [*(candidate for candidate, _, _ in read), *lenders]
    return [
        None
        if metadata is None
        else judge_image(
            candidate,
            entry,
            metadata,
            env,
            options,
            profile,
            records_for(candidate, checked, documents),
        )
        for candidate, entry, metadata in read
    ]


def other_image_alternative(
    index: int,
    considered: list[dict[str, Any]],
    results: list[dict[str, Any] | None],
    prefer: fit.Prefer,
) -> dict[str, Any] | None:
    """facts.alternative from another image select checked: a profile that fits there (built for
    or tested on this GPU by that image's own metadata: never by the record of a third image, as
    its offer says it was built for or tested on this GPU in that image), on the GPU type of the
    image chosen (considered[index]), whose precision is faster (more accurate with --prefer
    accuracy) than that of the recommended profile and of the other profiles that fit in the
    chosen image (fit.faster_fit: each result lists every profile that fits). The first such
    image in offer_order wins, and an image that is not pinnable is never named. The record has
    the image, its kind and `confirmed` true, and its offer says how that image stands to the
    chosen one (image_relation). None when the chosen image does not fit or no other image has
    one; fit's own alternative then stands, a profile of the chosen image not tested on this
    GPU."""
    chosen = results[index]
    if chosen is None or chosen["verdict"] != "fits":
        return None
    here = considered[index]
    base = str(considered[0]["resource_id"])
    for other in offer_order(index, considered):
        entry, result = considered[other], results[other]
        if not result or result["verdict"] != "fits" or not pinnable(entry):
            continue
        if result["group"] != chosen["group"]:  # its select would hand off another GPU type
            continue
        if not rerun_keeps_gpu_type(str(entry["resource_id"]), base, result):
            continue
        # a profile whose id select --profile could not take is never offered (add_fit_result)
        usable = [
            r for r in result["shortlist"] if PROFILE_ID.fullmatch(str(r["profile_id"] or ""))
        ]
        found = fit.faster_fit(chosen["shortlist"], usable, prefer)
        if found is None:
            continue
        what = OFFER_IMAGE.get(image_relation(here, entry), "the image")
        where = f"{what} {choose.label(entry, str(here['resource_id']))}"
        offer = fit.faster_fit_offer(found, chosen["recommended"], prefer, where)
        image = {"image": entry["image"], "kind": entry["kind"], "confirmed": True}
        return {**image, **found, "offer": offer}
    return None


def rerun_keeps_gpu_type(resource_id: str, base_id: str, result: dict[str, Any]) -> bool:
    """Whether the select a note on facts.alternative gives for an image of `resource_id` (--tag,
    --profile) judges it on the records this select judged it with (`result`), and so picks the
    same GPU type. Yes for an image of the repository asked for (`base_id`): that select reads
    the same other images of it (record_images). Not for a hardware variant in another
    repository: this select read none of that repository's other images, that select does, and
    with more than one GPU type their records could rank another type first. One GPU type: yes."""
    return resource_id == base_id or len(result.get("groups") or ()) <= 1


def offer_order(index: int, considered: list[dict[str, Any]]) -> list[int]:
    """The images other_image_alternative looks at, all checked but considered[index], in this
    order: the hardware variants in other repositories, in the order select checked them (their
    versions do not compare with the requested repository's); then the images of the requested
    repository (considered[0]'s), newest version first, its tag for a GPU (`1.8.0-RTX`) placed by
    its version numbers as well."""
    base = considered[0]["resource_id"] if considered else None
    others = [i for i in range(len(considered)) if i != index]
    own = [i for i in others if considered[i]["resource_id"] == base]
    variants = [i for i in others if considered[i]["resource_id"] != base]
    by_version = sorted(own, key=lambda i: newest_first(str(considered[i]["tag"])), reverse=True)
    return [*variants, *by_version]


def no_image_evaluated(
    facts: dict[str, Any], considered: list[dict[str, Any]], fix: str = SELECT_FIX
) -> NoReturn:
    """No candidate had readable metadata: exit 3 no_metadata, or exit 4 when every one failed."""
    errors = [item for item in considered if item["metadata"] == "error"]
    if errors and len(errors) == len(considered):
        first = errors[0]
        code = str(first["error_code"] or "metadata_service_error")
        if code == "metadata_service_error":
            fix = "Report it with the requestId; do not retry with other parameters."
        elif code == "ngc_bad_response":
            fix = ngc_api.BAD_RESPONSE_FIX
        else:
            fix = "Run the command again later; if it keeps failing, report it with the requestId"
        ns.emit(
            False,
            "select",
            "fix_errors",
            facts,
            [
                ns.error(
                    code, f"no image could be evaluated; {first['image']}: {first['error']}", fix
                )
            ],
            ns.EXIT_EXTERNAL,
        )
    shown = ", ".join(
        item.get("ngc_image") or item["image"]
        for item in considered
        if item["metadata"] == "not_published"
    )
    message = f"the NIM Metadata API has no profiles for {shown}"
    if errors:
        message += "; the metadata of " + ", ".join(e["image"] for e in errors) + " could not be "
        message += "read (see facts.images_considered[].error)"
    ns.emit(
        False,
        "no_metadata",
        "ask_user",
        facts,
        [ns.error("no_metadata", message, fix)],
        ns.EXIT_PRECONDITION,
    )


def profile_id(value: str) -> str:
    """select --profile: one profile id, as the metadata gives it (letters, digits, '.', '_' and
    '-' only)."""
    if not PROFILE_ID.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"a profile id is one word of letters, digits, '.', '_' and '-': {value!r}"
        )
    return value


def profile_not_found(profile: str, considered: list[dict[str, Any]]) -> ns.ScriptError:
    """select --profile named an id that no image checked has; an image whose metadata could not
    be read may have it, so the error names those too."""
    lacking = [item for item in considered if item["error_code"] == choose.PROFILE_NOT_FOUND]
    images = ", ".join(item.get("ngc_image") or item["image"] for item in lacking)
    message = f"the metadata of {images} has no profile {profile} (ids are matched in full)"
    fix = PROFILE_NOT_FOUND_FIX
    unread = [
        item.get("ngc_image") or item["image"] for item in considered if item["metadata"] == "error"
    ]
    if unread:
        message += (
            f"; the metadata of {', '.join(unread)} could not be read (see "
            "facts.images_considered[].error), so whether it has that profile is not known"
        )
        fix += "; or run the same select again later, when that metadata can be read"
    return ns.ScriptError(choose.PROFILE_NOT_FOUND, message, fix)


def select_region_notes(model: str, where: str | None, region: str | None) -> list[str]:
    """What to tell the user about the Deployment Geography of the model select checks, as the
    list does for its rows (region_notes)."""
    if listing.judged_global(where):
        return []
    if listing.exclusions(where) is None:
        return [
            f"The Deployment Geography of {model} could not be read: show it as the card says it "
            "(geography)."
        ]
    excluded = listing.region_excluded(where, region)
    if excluded:
        return [
            f"{model} is not offered in the user's region ({region}): its Deployment Geography "
            f"is {where}. Tell the user on its own line, after the verdict, which it does not "
            "change; the region the user gave stands, so do not ask for the country."
        ]
    if excluded is None and region is None:
        return [
            f"The Deployment Geography of {model} is not Global ({where}): {COUNTRY_ASK}; then "
            "select again with --region COUNTRY for the mark."
        ]
    if excluded is None:
        return [
            f"--region {region!r} could not be judged against the Deployment Geography of "
            f"{model} ({where}): {COUNTRY_ASK}; then select again with --region COUNTRY. Until "
            "then show that geography as it is."
        ]
    return []


def cnd_candidate(resolved: dict[str, Any]) -> tuple[list[choose.Candidate], list[str]]:
    """On a China NIM distributor select checks one image: the one named, or the newest with a
    known NGC build (its NGC build lends the profiles). Other CND tags are not checked."""
    archs = resolved.get("architectures")
    kind: choose.Kind = "requested" if resolved["cnd_tag_source"] == "requested" else "newest"
    candidate = choose.Candidate(
        str(resolved["resource_id"]), str(resolved["tag"]), kind, tuple(archs) if archs else None
    )
    catalog = cnd.CATALOGS[resolved["source"]]
    note = (
        f"On {catalog}, select checks only {resolved['image']} (as {resolved['ngc_image']}); "
        "other tags of it, older versions and hardware variants were not checked. Name another "
        "tag of the image to check it."
    )
    return [candidate], [note]


def cnd_reason(entry: dict[str, Any], catalog: str, pinned: bool = True) -> str:
    """Why select reports the one CND image it checked; for one listed only as latest, that it
    cannot be handed off before the user names its version tag."""
    what = (
        "the image you named"
        if entry["kind"] == "requested"
        else f"the newest {catalog} image with a known NGC build"
    )
    reason = {
        "fits": f"{what}; it fits",
        "unknown": f"{what}; its fit could not be confirmed",
    }.get(str(entry.get("verdict")), f"{what}; no profile of it runs here")
    if not pinned:
        reason += (
            f". {catalog} lists it only as latest, so the verdict is for what latest is today: "
            "ask the user for its version tag before any hand-off (facts.warnings)"
        )
    return reason


def cmd_select(args: argparse.Namespace) -> NoReturn:
    facts: dict[str, Any] = {}
    considered: list[dict[str, Any]] = []
    try:
        check_limits(args)
        source = cnd.choose_source(args.model, args.source)
        env = load_env(args.env)
        options = fit_options(args, None)
        # Env problems (MIG, no GPUs) stop here, before any NGC call: no image can fit them.
        fit.evaluate({"profiles": []}, env, options)
        resolved, repo, listed = resolve_model(
            args.model, args.tag, source, args.nim_version, env.get("arch")
        )
        on_cnd = source in cnd.CNDS
        base_id = str(resolved["resource_id"])
        facts = {
            "source": source,
            "resource_id": base_id,
            "model": resolved["model"],
            "mapping": resolved["mapping"],
            **cnd_keys(resolved),
            "warnings": list(resolved["warnings"]),
        }
        if on_cnd:
            candidates, search_notes = cnd_candidate(resolved)
        else:
            candidates, search_notes = build_candidates(
                resolved, repo, listed, env, args.max_images
            )
        notes = [*resolved.get("notes", []), *search_notes]
        where = listing.geography(repo.get("description"))
        facts.update(geography=where, region_excluded=listing.region_excluded(where, args.region))
        notes.extend(select_region_notes(str(resolved["model"]), where, args.region))
        # Kept in facts from here on, so an NGC failure mid-way still reports what was checked.
        facts.update(notes=notes, images_considered=considered)
        # Every profile that fits is kept, for other_image_alternative; facts get --top of them.
        every = dataclasses.replace(options, top=LIST_TOP)
        # Every image is read first, then each is fitted with the records of the other images of
        # its repository (judge_read); the images read keep their verdicts when NGC stops. With a
        # tag named, the other images a select without one checks are read for their records
        # alone (record_images), so the image gets the same verdict.
        documents: dict[str, dict[str, Any] | None] = {}
        read: list[tuple[choose.Candidate, dict[str, Any], dict[str, Any] | None]] = []
        lenders: list[choose.Candidate] = []
        unread: list[str] = []
        unlisted: str | None = None
        try:
            for candidate in candidates:
                ns.log(f"checking {candidate.image} ({candidate.kind})")
                entry, metadata = read_image(candidate, documents, args.profile)
                if on_cnd:  # the CND image, checked with the profiles of its NGC build
                    entry.update(image=resolved["image"], ngc_image=candidate.image)
                considered.append(entry)
                read.append((candidate, entry, metadata))
            if not on_cnd and one_image(resolved) and read[0][2] is not None:
                lenders, unlisted = record_images(resolved, repo, listed, env, args.max_images)
                unread = read_records(lenders, documents)
        except ns.ScriptError:
            judge_read(read, documents, env, every, args.profile, lenders)
            raise
        results = judge_read(read, documents, env, every, args.profile, lenders)
        if not on_cnd and any(results):  # images of the repository whose records were not counted
            unread += [
                item["image"]
                for item in considered
                if item["resource_id"] == base_id and item["metadata"] == "error"
            ]
            unread_note = unread_records_note(unread)
            if unread_note:
                notes.append(unread_note)
            if unlisted:
                notes.append(unlisted)
        index = choose.pick(considered)
        if index is None:
            # --profile names an id no image checked has: an input error, and --out keeps the
            # earlier result, whose hand-off still stands
            if any(i["error_code"] == choose.PROFILE_NOT_FOUND for i in considered):
                raise profile_not_found(str(args.profile), considered)
            if on_cnd:
                save(args.out, facts)
                no_image_evaluated(facts, considered, CND_NO_METADATA_FIX)
            seen = {item["tag"] for item in considered if item["resource_id"] == base_id}
            facts["tags_to_try"] = tags_to_try(
                repo, str(resolved["tag"]), listed.digests, skip=seen
            )
            save(args.out, facts)
            no_image_evaluated(facts, considered)
        chosen, result = candidates[index], results[index]
        assert result is not None
        # On a CND select checks one image, and with --profile one profile: no other to offer.
        if not on_cnd and args.profile is None:
            elsewhere = other_image_alternative(index, considered, results, args.prefer)
            if elsewhere:
                result = {**result, "alternative": elsewhere}
            elif not rerun_keeps_gpu_type(chosen.resource_id, base_id, result):
                result = {**result, "alternative": None}  # its select could take another type
        top = {key: result[key][: args.top] for key in ("shortlist", "unconfirmed")}
        result = {**result, **top}
        unchecked = (
            None if on_cnd else unchecked_newer_note(base_id, repo, listed, chosen, considered)
        )
        if unchecked:
            notes.append(unchecked)
        if args.profile is not None:
            mixed = len(result["groups"]) > 1
            on = f", on the GPU type select picks without it ({result['group'].get('name')})"
            notes.append(
                f"Only the profile {args.profile} was checked (--profile){on if mixed else ''}: "
                "this result names no other profile of the image."
            )
        facts.update(
            resource_id=chosen.resource_id,
            model=chosen.resource_id.removeprefix("nim/"),
            tag=chosen.tag,
            image=resolved["image"] if on_cnd else chosen.image,
        )
        facts["warnings"].extend(image_warnings(resolved, chosen.resource_id, chosen.tag))
        all_notes = [*notes, *result.get("notes", [])]
        add_fit_result(facts, result, env, "published", all_notes, chosen.kind)
        if on_cnd:
            reason = cnd_reason(
                considered[index], cnd.CATALOGS[source], bool(resolved.get("pinned", True))
            )
        else:
            # A requested moving alias (`1`) was replaced by its build: the reason names it.
            replaced = resolved.get("tag_source") == "alias_replaced"
            asked = (args.tag or ngc_api.parse_model_ref(args.model).tag) if replaced else None
            reason = choose.reason(considered, index, base_id, asked)
        facts["image_choice"] = {"image": facts["image"], "kind": chosen.kind, "reason": reason}
        save(args.out, facts)
    except ns.ScriptError as exc:
        fail("select", exc, facts)
    finish(facts)


# --- list -----------------------------------------------------------------------------------

# How many models in a row, as they end, may fail with a code before the list stops: one model's
# reset or timeout is that model's not_checked entry; a source that stopped answering stops the
# list; NGC asking for fewer requests (HTTP 429 after the retries) stops it at once.
LIST_STOP_CODES = {"ngc_unreachable": 3, "cnd_unreachable": 3, "ngc_rate_limited": 1}
LIST_TOP = 1_000_000  # every fitting and unconfirmed profile is kept, to find the fewest GPUs
NOT_PUBLIC_LIST = "NGC shows its image list only with an NGC key (canGuestPull is not true)"
NO_CANDIDATES_FIX = (
    "Tell the user the source lists no LLM or VLM NIM that matches (facts.skipped counts what was "
    "left out); ask for another --query, or list without one"
)
# no_candidates for a CND catalog longer than a list reads (ListScan.capped): only the part read
# was checked, and listing again reads the same part.
NO_CANDIDATES_CAPPED_FIX = (
    "Tell the user no LLM or VLM NIM matches in the part of the catalog a list reads "
    "(facts.skipped counts what was left out) and the rest was not checked (facts.notes), never "
    "that the source lists none. Listing again, with another --query or none, reads the same "
    "pages: to check a model past them, ask the user for its full image (HOST/PROJECT/REPO:TAG) "
    "and run select on it"
)
NAMED_AT_MOST = 5  # models a note names
# NGC refusing the record of every model a list checked, when there are at least this many, is
# NGC failing (a key now asked for, a network blocked), never "no LLM or VLM NIM to list".
NOT_PUBLIC_ALL = 3


@dataclasses.dataclass
class ListModel:
    """One model of the NGC list: its repository and the hardware variants folded under it."""

    resource_id: str
    variants: list[str]


@dataclasses.dataclass
class ListScan:
    """What a list checked: the model (or CND repository) of each check, in order, the catalog
    entries left out, by why, the outcome of the checks, the catalog entries that could not be
    checked (a CND catalog detail that failed), and a note when the catalog was not read in full;
    `capped` when only because a CND catalog lists more than the pages a list reads, which
    listing again does not change."""

    labels: list[str]
    skipped: dict[str, int]
    outcome: listing.Outcome
    not_checked: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    catalog_note: str | None = None
    capped: bool = False


@dataclasses.dataclass
class ListRun:
    """What one list run shares between the models it checks."""

    env: dict[str, Any]  # as inspected: idle GPUs and all
    judged: dict[str, Any]  # every whole GPU of each type idle (listing.all_gpus)
    options: fit.FitOptions
    region: str | None
    checker: NgcChecker = dataclasses.field(default_factory=lambda: NgcChecker())
    done: int = 0
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    def progress(self, total: int) -> None:
        with self.lock:
            self.done += 1
            if self.done % 20 == 0 or self.done == total:
                ns.log(f"list: {self.done} of {total} checked")


def list_image_list(resource_id: str, repo: dict[str, Any]) -> ImageList:
    """The image list select reads for `resource_id`. NGC answers 403 for the image list of every
    NIM that cannot be pulled without a key (all 213 of them on 2026-11-11, and none other), so for
    those the list takes that answer without the request: the CPU architectures then come from
    the repository record, as in select."""
    if repo.get("canGuestPull") is True:
        return read_image_list(resource_id)
    return ImageList(resource_id, {}, NOT_PUBLIC_LIST)


def newest_check(
    rid: str,
    repo: dict[str, Any],
    run: ListRun,
    kind: choose.Kind,
    documents: dict[str, dict[str, Any] | None],
) -> tuple[choose.Candidate, dict[str, Any], dict[str, Any] | None, ImageList]:
    """select's check of the newest image of one repository: its tag (select's tag rules), its
    CPU architectures and the fit of its published profiles (consider)."""
    listed = list_image_list(rid, repo)
    tag = ngc_api.choose_tag(repo, None, listed.digests)[0]
    archs = image_facts(repo, tag, listed)[0]
    candidate = choose.Candidate(rid, tag, kind, tuple(archs) if archs else None)
    entry, result = consider(candidate, run.judged, run.options, documents)
    return candidate, entry, result, listed


def row_profile(result: dict[str, Any]) -> dict[str, Any] | None:
    """The profile a list row shows of a fit result. fits: the best of the profiles that fit with
    the fewest GPUs. unknown: select's best unconfirmed profile, or one that leaves the same checks
    open on the same tier with fewer GPUs. None for does_not_fit."""
    if result["verdict"] == "fits":
        return listing.smallest(result["shortlist"])
    best = result.get("best_unconfirmed")
    if result["verdict"] != "unknown" or best is None:
        return None
    def open_checks(record: dict[str, Any]) -> frozenset[str]:
        # notes (pci_inferred, tested_in_other_image) leave no check open
        return frozenset(r for r in record["reasons"] if r not in choose.NOTE_CODES)

    return listing.smallest(
        [
            record
            for record in result["unconfirmed"]
            if record["tier"] == best["tier"] and open_checks(record) == open_checks(best)
        ]
    )


def row_outline(result: dict[str, Any]) -> tuple[Any, ...]:
    """What a list row and choose.pick take from a fit result: the GPU type, the verdict, and the
    id, tier, reasons and GPU count of the best profile and of the profile the row shows."""

    def brief(record: dict[str, Any] | None) -> tuple[Any, ...] | None:
        if record is None:
            return None
        return (
            record["profile_id"],
            record["tier"],
            tuple(record["reasons"]),
            record["gpus_required"],
        )

    group = result["group"]
    best = result.get("recommended") or result.get("best_unconfirmed")
    return (
        group.get("name"),
        group.get("pci_device_id"),
        tuple(group.get("pci_device_ids") or ()),
        result["verdict"],
        brief(best),
        brief(row_profile(result)),
    )


def fit_with_every_record(
    candidate: choose.Candidate, metadata: dict[str, Any], run: ListRun
) -> dict[str, Any]:
    """An image's fit as if another image of its repository recorded each of its profile ids, with
    the same workspace_hash, as tested on these GPUs (fit.recorded_everywhere): the most records
    can do."""
    everywhere = fit.recorded_everywhere(metadata, run.judged)
    return fit_image(candidate, metadata, run.judged, run.options, records=everywhere)


def records_matter(
    candidate: choose.Candidate,
    result: dict[str, Any] | None,
    documents: dict[str, dict[str, Any] | None],
    run: ListRun,
) -> bool:
    """Whether the records of other images of its repository could change what the list takes
    from this image's fit (row_outline): fitted with every record (fit_with_every_record), it
    gives another outline. When it gives the same, no record can change it: a record only makes a
    profile `tested`, so every record set gives an outline between the two. False for an image
    with nothing to fit."""
    metadata = documents.get(candidate.image)
    if result is None or not metadata:
        return False
    return row_outline(fit_with_every_record(candidate, metadata, run)) != row_outline(result)


def records_could_fit(
    candidate: choose.Candidate,
    result: dict[str, Any] | None,
    documents: dict[str, dict[str, Any] | None],
    run: ListRun,
) -> bool:
    """Whether the records of other images of its repository could make this image fit: fitted
    with every record (fit_with_every_record), it fits. A record only makes a profile `tested`:
    one that a check no record touches keeps from fitting, such as a VRAM need its metadata does
    not record (vram_not_recorded), never fits by one, though records can change its tier and
    reasons. False for an image with nothing to fit."""
    metadata = documents.get(candidate.image)
    if result is None or not metadata:
        return False
    return fit_with_every_record(candidate, metadata, run)["verdict"] == "fits"


def list_row(
    model: str,
    entry: dict[str, Any],
    result: dict[str, Any] | None,
    run: ListRun,
    geography: str | None,
) -> dict[str, Any]:
    """One row from select's check of an image (consider): the verdict and one profile
    (row_profile), with its reasons, tier, GPU count, precision and backend. No profile id: the
    list names none to deploy."""
    verdict = entry.get("verdict") or "no_metadata"
    shown = row_profile(result) if result is not None else None
    if shown is not None:
        reasons = list(shown["reasons"])
    elif verdict == "does_not_fit":
        reasons = listing.top_codes(entry.get("rejected_counts") or {})
    else:
        reasons = []
    group = result.get("group") if result else None
    gpus = int(shown["gpus_required"]) if shown else None
    idle = listing.idle_of(run.env, group)
    return {
        "model": model,
        "image": entry["image"],
        "verdict": verdict,
        "kind": "newest",
        "reasons": reasons,
        "tier": shown["tier"] if shown else None,
        "gpus_min": gpus,
        "gpu_type": group.get("name") if group else None,
        "precision": shown["precision"] if shown else None,
        "backend": shown["backend"] if shown else None,
        "pinned": True,
        "needs_ngc_key": None,
        "needs_busy_gpus": bool(gpus and idle is not None and gpus > idle),
        "geography": geography,
        "region_excluded": listing.region_excluded(geography, run.region),
        "variant_of": None,
        "next": "select",
    }


def not_public(rid: str) -> dict[str, Any]:
    return {"skip": "not_public", "model": rid.removeprefix("nim/")}


def public_repo(rid: str) -> dict[str, Any] | None:
    """NGC's repository record, or None when NGC says it is not public (401, or 403 "Not a
    catalog artifact!", with NGC's error envelope) or missing (404). A 401 or 403 without the
    envelope (a proxy's or a firewall's page) is an error, never "not public"."""
    try:
        return ngc_api.get_repo(rid)
    except ns.ScriptError as exc:
        if not_public_error(exc):
            return None
        raise


def variant_candidates(model: ListModel) -> list[tuple[choose.Candidate, dict[str, Any]]]:
    """select's hardware variants of a model (find_variants), from the repositories the catalog
    folds under it: each at its newest tag, with its repository record. One NGC does not show,
    or whose tag cannot be chosen, is left out, as select leaves it out."""
    found = []
    for rid in model.variants:
        try:
            repo = public_repo(rid)
            if repo is None:
                continue
            listed = list_image_list(rid, repo)
            tag = ngc_api.choose_tag(repo, None, listed.digests)[0]
        except ns.ScriptError as exc:
            if exc.code in NGC_STOPS:
                raise
            ns.log(f"list: the variant {rid} was skipped: {exc.message}")
            continue
        archs = image_facts(repo, tag, listed)[0]
        candidate = choose.Candidate(rid, tag, "variant", tuple(archs) if archs else None)
        found.append((candidate, repo))
    return found


# One image a list row checked: the candidate, its images_considered entry, its fit (None: nothing
# to fit) and its repository's record.
ListCheck = tuple[choose.Candidate, dict[str, Any], dict[str, Any] | None, dict[str, Any]]


def judge_with_records(
    own: list[choose.Candidate],
    checks: dict[str, ListCheck],
    documents: dict[str, dict[str, Any] | None],
    run: ListRun,
) -> None:
    """Fit each image of the requested repository a list row read again, with the records of
    the others read (records_for), as select's judge_read does once it has read them all, or
    when NGC stops before. An image not read (NGC stopped first) is left out."""
    for candidate in own:
        if candidate.image not in checks:
            continue
        _, entry, result, repo = checks[candidate.image]
        metadata = documents.get(candidate.image)
        if result is None or not metadata:
            continue
        records = records_for(candidate, own, documents)
        result = judge_image(candidate, entry, metadata, run.judged, run.options, records=records)
        checks[candidate.image] = (candidate, entry, result, repo)


def list_ngc_model(model: ListModel, run: ListRun) -> dict[str, Any]:
    """{"row": ...} for one NGC model, {"skip": why} when it is not listed (not public, or its
    profiles name only backends that serve no LLM), or {"not_checked": ...} when its metadata
    could not be read. The images select checks, in select's order (build_candidates): the newest;
    when it does not fit, the hardware variants, the repository's tag for this GPU and the older
    versions, until one fits. select fits each image with the records of the other images of its
    repository it checks (judge_read): when those could change what the list takes from an image
    of the repository it read (records_matter), the list also reads every other image of the
    repository select checks (within --max-images, through the same cache and request limit), and
    when it has read them all it fits each with those records, as select does. The row shows
    select's choice of them (choose.pick) and its kind: `newest`, `variant` (with variant_of) or
    `older`."""
    rid = model.resource_id
    repo = public_repo(rid)
    if repo is None:
        return not_public(rid)
    documents: dict[str, dict[str, Any] | None] = {}
    first, entry, result, listed = newest_check(rid, repo, run, "newest", documents)
    if listing.backend_kind(documents.values()) == "non_llm":
        return {"skip": "non_llm", "model": rid.removeprefix("nim/")}
    checks: dict[str, ListCheck] = {first.image: (first, entry, result, repo)}
    plan = [first]
    own: list[choose.Candidate] = []
    complete = False
    stopped: ns.ScriptError | None = None
    records = records_matter(first, result, documents, run)
    if entry.get("verdict") != "fits" or records:
        try:
            variants = variant_candidates(model)
            repos = {candidate.resource_id: found for candidate, found in variants}
            limit = choose.DEFAULT_MAX_IMAGES
            found = [c for c, _ in variants]
            later = later_candidates(rid, repo, listed, first.tag, found, run.env, limit)
            plan = choose.plan([first, *later], limit)[0]
            own = [candidate for candidate in plan if candidate.resource_id == rid]

            def check(candidate: choose.Candidate) -> ListCheck:
                later_entry, later_result = consider(candidate, run.judged, run.options, documents)
                repo_of = repos.get(candidate.resource_id, repo)
                done = (candidate, later_entry, later_result, repo_of)
                checks[candidate.image] = done
                return done

            def read_own() -> None:
                # Every image of the repository select checks, each fitted with the records of
                # the others, as select fits them. A record only adds fits, so an image that fit
                # still fits, and none that came before it in select's order can fit now unless
                # its own outline could change: the walk below stops where select's choice is.
                nonlocal complete
                for candidate in own:
                    if candidate.image not in checks:
                        check(candidate)
                judge_with_records(own, checks, documents, run)
                complete = True

            if records:
                read_own()
            # select's order, up to the first image that fits (select takes it)
            for candidate in plan:
                if candidate.image not in checks:
                    _, _, later_result, _ = check(candidate)
                    open_own = candidate.resource_id == rid and not complete
                    if open_own and records_matter(candidate, later_result, documents, run):
                        read_own()
                if checks[candidate.image][1].get("verdict") == "fits":
                    break
            if not complete and all(candidate.image in checks for candidate in own):
                judge_with_records(own, checks, documents, run)  # read in full: as select
        except ns.ScriptError as exc:
            # NGC stopped while the row read images only for the records they lend: the images
            # read are fitted with the records read, as select fits them when NGC stops, and
            # when they give select's pick, the row is kept, marked (RowKept)
            if exc.code not in NGC_STOPS:
                raise
            if not complete:
                judge_with_records(own, checks, documents, run)
            if not pick_on_stop(plan, checks, documents, run, rid, complete):
                raise
            stopped = exc
    checked = [checks[candidate.image] for candidate in plan if candidate.image in checks]
    index = choose.pick([check[1] for check in checked])
    # The newest image's metadata could not be read: only an image that fits makes a row; any
    # other verdict would be given without the newest image's.
    fits = index is not None and checked[index][1].get("verdict") == "fits"
    if entry.get("metadata") == "error" and not fits:
        return {
            "not_checked": {
                "model": rid.removeprefix("nim/"),
                "image": first.image,
                "why": entry.get("error_code"),
                "message": entry.get("error"),
            }
        }
    shown, shown_entry, shown_result, shown_repo = checked[index or 0]
    where = listing.geography(repo.get("description")) or listing.geography(
        shown_repo.get("description")
    )
    row = list_row(rid.removeprefix("nim/"), shown_entry, shown_result, run, where)
    row.update(
        kind=shown.kind,
        pinned=ngc_api.is_build(shown.tag),
        needs_ngc_key=shown_repo.get("canGuestPull") is not True,
        variant_of=first.image if shown.kind == "variant" else None,
    )
    if stopped is not None:
        row["records_unread"] = True
        raise RowKept(stopped, row)
    return {"row": row}


class RowKept(ns.ScriptError):
    """NGC stopped (NGC_STOPS) while a list row read other images of its repository only for
    the records they lend, after the images read gave select's pick (pick_on_stop): that stop,
    which counts for the list as any failed model does (listing.run_tasks), with the row they
    give (`row`, records_unread true), which list_rows keeps."""

    def __init__(self, stop: ns.ScriptError, row: dict[str, Any]) -> None:
        super().__init__(stop.code, stop.message, stop.fix, stop.exit_code, stop.facts)
        self.row = row


def pick_on_stop(
    plan: list[choose.Candidate],
    checks: dict[str, ListCheck],
    documents: dict[str, dict[str, Any] | None],
    run: ListRun,
    rid: str,
    complete: bool,
) -> bool:
    """Whether the images a list row read before NGC stopped, fitted with the records read
    (judge_with_records), give select's pick whatever the images not read record: in select's
    order (plan), one that fits, every image before it read, and none of them able to fit with
    more records (records_could_fit: an image of the repository before all were read; another
    repository's image lends and takes none). Select takes the first image that fits, so records
    that could change only the tier and reasons of an image before it leave its pick as it is."""
    for candidate in plan:
        if candidate.image not in checks:
            return False
        _, entry, result, _ = checks[candidate.image]
        if entry.get("verdict") == "fits":
            return True
        open_own = candidate.resource_id == rid and not complete
        if open_own and records_could_fit(candidate, result, documents, run):
            return False
    return False


def ngc_catalog(deadline_at: float) -> tuple[ngc_api.Catalog, str | None]:
    """The NIM catalog, read within the list's deadline, and a note when it was not read in full
    (the list is then partial)."""

    def late() -> ngc_api.Catalog:
        raise ns.ScriptError(
            "ngc_unreachable",
            f"the NIM catalog of {ngc_api.base_url()} was not read within {listing.DEADLINE:g} s",
            "Check network access to api.ngc.nvidia.com (proxy settings such as HTTPS_PROXY), "
            "then list again",
            ns.EXIT_EXTERNAL,
        )

    budget = max(deadline_at - time.monotonic(), 0.0)
    catalog = cnd.deadline(budget, ngc_api.list_catalog, late)
    if catalog.complete:
        return catalog, None
    read = len(catalog.resources)
    of = f" of the {catalog.total} it counts" if catalog.total is not None else ""
    pages = f"; {catalog.unread_pages} page(s) were not read" if catalog.unread_pages else ""
    note = (
        f"The NIM catalog was not read in full ({read} NIMs read{of}{pages}): the list is "
        "partial; say so, and list again later."
    )
    ns.log(note)
    return catalog, note


def list_ngc(query: str | None, run: ListRun, deadline_at: float) -> ListScan:
    """The NGC list: the catalog (one search, more pages when it says so, within the deadline),
    the models it keeps, and the outcome of checking them, WORKERS at a time, until the
    deadline."""
    skipped = dict.fromkeys(listing.SKIPPED, 0)
    catalog, catalog_note = ngc_catalog(deadline_at)
    kept: list[str] = []
    for resource in catalog.resources:
        rid, name = str(resource.get("resourceId")), str(resource.get("name") or "")
        if resource.get("orgName") != "nim" or not rid.startswith("nim/"):
            continue
        if not listing.query_matches(query, [rid.removeprefix("nim/")]):
            continue
        why = listing.classify([name, str(resource.get("displayName") or "")])
        if why is None and listing.is_production_branch(name):
            why = "production_branch"
        if why:
            skipped[why] += 1
        else:
            kept.append(rid)
    models = [ListModel(base, found) for base, found in listing.fold_variants(kept).items()]

    def task(model: ListModel) -> dict[str, Any]:
        try:
            return list_ngc_model(model, run)
        finally:
            run.progress(len(models))

    ns.log(f"list: checking {len(models)} models on NGC, {listing.WORKERS} at a time")
    outcome = listing.run_tasks(
        [lambda model=model: task(model) for model in models],
        listing.WORKERS,
        deadline_at,
        LIST_STOP_CODES,
    )
    labels = [model.resource_id.removeprefix("nim/") for model in models]
    return ListScan(labels, skipped, outcome, catalog_note=catalog_note)


def cnd_row(found: cnd.Found, source: str, run: ListRun) -> dict[str, Any]:
    """{"row": ...} for one image of a China NIM distributor's catalog, {"skip": "non_llm"}, or
    {"not_checked": ...}. As select on that repository without a tag: the newest image with a
    version tag and a known NGC build (labels or the TGC catalog, never the name), checked with
    that build's profiles. No known build: a row with ngc_build_unknown. registry_gpus are the GPUs
    the catalog lists the image for: information, never a verdict. `pinned` is false when the row
    shows the image with no tag: no tag but latest is known for it."""
    ref = rl.parse_image(found.image or "")
    assert ref is not None
    ref = ref.with_tag(None)
    images, problem = cnd.reader(source).images(ref, found, run.env.get("arch"))
    category = next((image.category for image in images if image.category), None)
    if listing.category_kind(category) == "non_llm":
        return {"skip": "non_llm", "model": ref.repository_ref}
    devices = list(found.devices) or None
    try:
        image, build, tried, hand_tag, pinned = newest_with_build(
            source, ref, images, problem, run.checker, []
        )
    except ns.ScriptError as exc:
        if exc.code != "ngc_build_unknown":
            raise
        entry = {"image": exc.facts.get("image") or ref.image, "verdict": "ngc_build_unknown"}
        row = list_row(ref.repository_ref, entry, None, run, None)
        row.update(
            reasons=["ngc_build_unknown"],
            pinned=exc.facts.get("cnd_tag") is not None,
            needs_ngc_key=False,
            ngc_image=None,
            registry_gpus=devices,
        )
        return {"row": row}
    resolved, repo, _ = cnd_facts(
        source, image, build, tried, hand_tag, pinned, "newest_with_build", []
    )
    candidate = cnd_candidate(resolved)[0][0]
    documents: dict[str, dict[str, Any] | None] = {}
    entry, result = consider(candidate, run.judged, run.options, documents)
    if listing.backend_kind(documents.values()) == "non_llm":
        return {"skip": "non_llm", "model": ref.repository_ref}
    if entry.get("metadata") == "error":
        return {
            "not_checked": {
                "model": ref.repository_ref,
                "image": resolved["image"],
                "why": entry.get("error_code"),
                "message": entry.get("error"),
            }
        }
    where = listing.geography(repo.get("description"))
    # An image the CND lists only as latest is shown without a tag: never IMAGE:latest.
    shown = resolved["image"] if resolved["pinned"] else resolved["cnd_repository"]
    row = list_row(ref.repository_ref, {**entry, "image": shown}, result, run, where)
    row.update(
        pinned=resolved["pinned"],
        needs_ngc_key=False,
        ngc_image=resolved["ngc_image"],
        registry_gpus=devices,
    )
    return {"row": row}


def list_cnd(source: str, query: str | None, run: ListRun, deadline_at: float) -> ListScan:
    """The list of one China NIM distributor: its whole catalog (Reader.list_all), the images it
    keeps (one per repository) and the outcome of checking them until the deadline. A catalog
    item whose detail could not be read is not checked (ListScan.not_checked). A catalog that
    lists more than the pages a list reads is not checked past them, with a note
    (ListScan.capped)."""
    reader = cnd.reader(source)

    def late() -> cnd.Catalog:
        raise reader.unreachable(f"the catalog was not read within {listing.DEADLINE:g} s")

    def wanted(name: str, kind: str | None) -> bool:
        # no detail request for an item the query or the name lists leave out
        return listing.query_matches(query, [name]) and listing.classify([name], kind) is None

    budget = max(deadline_at - time.monotonic(), 0.0)
    catalog = cnd.deadline(
        budget, lambda: cnd.shaped(reader, lambda: reader.list_all(wanted)), late
    )
    skipped = dict.fromkeys(listing.SKIPPED, 0)
    for name, kind in catalog.left_out:
        why = listing.classify([name], kind) if listing.query_matches(query, [name]) else None
        if why:
            skipped[why] += 1
    skipped["no_image"] += len(catalog.no_image)
    seen: set[str] = set()
    kept: list[cnd.Found] = []
    for found in catalog.found:
        ref = rl.parse_image(found.image or "")
        if ref is None or ref.repository_ref in seen:
            continue
        seen.add(ref.repository_ref)
        why = listing.classify([found.name, f"{ref.project}/{ref.repository}"], found.model_type)
        if why:
            skipped[why] += 1
        else:
            kept.append(found)

    def task(found: cnd.Found) -> dict[str, Any]:
        try:
            return cnd_row(found, source, run)
        finally:
            run.progress(len(kept))

    catalog_name = cnd.CATALOGS[source]
    ns.log(f"list: checking {len(kept)} images on {catalog_name}, {listing.WORKERS} at a time")
    outcome = listing.run_tasks(
        [lambda found=found: task(found) for found in kept],
        listing.WORKERS,
        deadline_at,
        LIST_STOP_CODES,
    )
    labels = []
    for found in kept:
        ref = rl.parse_image(found.image or "")
        labels.append(ref.repository_ref if ref else found.name)
    failed = [
        {
            "model": name,
            "why": exc.code,
            "message": f"its {catalog_name} catalog detail: {exc.message}",
        }
        for name, exc in catalog.failed
    ]
    note = None
    if catalog.unread:
        note = (
            f"The {catalog_name} catalog was not read in full ({catalog.unread}, and a list "
            "reads only those): the rest was not checked, and listing again reads the same "
            "pages; say so."
        )
        ns.log(note)
    return ListScan(labels, skipped, outcome, failed, note, capped=note is not None)


def gpus_judged(run: ListRun) -> list[dict[str, Any]]:
    """Each GPU type and how many of its GPUs the list judges (all whole GPUs, within --max-gpus
    and the namespace quota), with how many are idle now (host only)."""
    groups = fit.evaluate({"profiles": []}, run.judged, run.options)["groups"]
    return [
        {
            "name": group.get("name"),
            "pci_device_id": group.get("pci_device_id"),
            "count": group.get("count"),
            "judged": group.get("available_gpus"),
            "idle": listing.idle_of(run.env, group),
        }
        for group in groups
    ]


def list_run(args: argparse.Namespace) -> ListRun:
    """The options and environment of a list. MIG, shared or no GPUs stop it here (gpus_unknown,
    exit 3), before any request: no NIM can be judged on them."""
    if args.max_gpus is not None and args.max_gpus < 1:
        raise ns.ScriptError("bad_input", "--max-gpus must be 1 or more", "Pass --max-gpus 1")
    env = load_env(args.env)
    options = fit.FitOptions(
        max_gpus=args.max_gpus,
        need_lora=args.need_lora,
        backend=args.backend,
        prefer=args.prefer,
        top=LIST_TOP,
    )
    try:
        fit.evaluate({"profiles": []}, env, options)
    except ns.ScriptError as exc:
        fail("gpus_unknown", exc)
    return ListRun(env, listing.all_gpus(env), options, args.region)


def list_rows(scan: ListScan) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(rows, not checked) from what the checks returned; skips are counted in scan.skipped."""
    rows: list[dict[str, Any]] = []
    not_checked: list[dict[str, Any]] = list(scan.not_checked)
    outcome = scan.outcome
    for index, label in enumerate(scan.labels):
        answer, error = outcome.results.get(index), outcome.errors.get(index)
        if answer is not None and "row" in answer:
            rows.append(answer["row"])
        elif answer is not None and "skip" in answer:
            scan.skipped[answer["skip"]] += 1
        elif answer is not None:
            not_checked.append(answer["not_checked"])
        elif isinstance(error, RowKept):  # NGC stopped after the images read gave the row
            rows.append(error.row)
        elif isinstance(error, ns.ScriptError):
            not_checked.append({"model": label, "why": error.code, "message": error.message})
        elif error is not None:
            ns.log(f"list: {label} failed: {error!r}")
            not_checked.append({"model": label, "why": "error", "message": repr(error)})
        else:
            # cut while it ran, or never started: by the deadline, or after a failure
            not_checked.append({"model": label, "why": "deadline" if outcome.late else "stopped"})
    return rows, not_checked


def cmd_list(args: argparse.Namespace) -> NoReturn:
    started = time.monotonic()
    deadline_at = started + listing.DEADLINE
    facts: dict[str, Any] = {}
    saved = ngc_api.SETTINGS
    cache = listing.NgcCache(None if args.no_cache else ns.private_folder() / listing.CACHE_NAME)
    on_cnd = args.source in cnd.CNDS
    try:
        run = list_run(args)
        cache.load()
        ngc_api.SETTINGS = ngc_api.Settings(listing.REQUEST_TIMEOUT, listing.ATTEMPTS, cache)
        facts = {
            "schema": listing.LIST_SCHEMA,
            "source": args.source,
            "environment_summary": environment_summary(run.env),
            "gpus_judged": gpus_judged(run),
            "options": {
                "query": args.query,
                "region": args.region,
                "max_gpus": args.max_gpus,
                "need_lora": args.need_lora,
                "backend": args.backend,
                "prefer": args.prefer,
            },
        }
        try:
            if on_cnd:
                scan = list_cnd(args.source, args.query, run, deadline_at)
            else:
                scan = list_ngc(args.query, run, deadline_at)
        except ns.ScriptError as exc:
            if exc.exit_code != ns.EXIT_EXTERNAL:
                raise
            # The catalog could not be read: there is nothing to list.
            cache.save()
            facts.update(list_facts([], dict.fromkeys(listing.SKIPPED, 0), [], cache, started))
            facts.update(partial=True, notes=[])
            save(args.out, facts)
            state = "cnd_unreachable" if exc.code.startswith("cnd_") else "ngc_unreachable"
            error = exc.as_error()
            if isinstance(exc, cnd.LoginNeeded):
                catalog = f"the {cnd.CATALOGS[args.source]} catalog"
                error["fix"] = cnd.LIST_LOGIN_FIX.format(catalog=catalog)
            summary = list_summary(facts, args.out)
            ns.emit(False, state, "fix_errors", summary, [error], ns.EXIT_EXTERNAL)
        rows, not_checked = list_rows(scan)
        cache.save()
        facts.update(list_facts(rows, scan.skipped, not_checked, cache, started))
        facts["partial"] = facts["partial"] or scan.catalog_note is not None
        facts["notes"] = list_notes(facts, run, scan, on_cnd)
        state, failure = list_state(facts, scan.outcome, on_cnd, len(scan.labels), scan.capped)
        save(args.out, facts)
    except ns.ScriptError as exc:
        fail("list", exc, list_summary(facts, None) if "rows" in facts else facts)
    finally:
        ngc_api.SETTINGS = saved
    summary = list_summary(facts, args.out)
    if failure is not None:
        ns.emit(False, state, "fix_errors", summary, [failure.as_error()], ns.EXIT_EXTERNAL)
    if state == "no_candidates":
        where = " in the part of the catalog a list reads (facts.notes)" if scan.capped else ""
        fix = NO_CANDIDATES_CAPPED_FIX if scan.capped else NO_CANDIDATES_FIX
        error = ns.error("no_candidates", f"no LLM or VLM NIM to list{where}", fix)
        ns.emit(False, state, "ask_user", summary, [error], ns.EXIT_PRECONDITION)
    ns.emit(True, state, "ask_user", summary)


def list_facts(
    rows: list[dict[str, Any]],
    skipped: dict[str, int],
    not_checked: list[dict[str, Any]],
    cache: listing.NgcCache,
    started: float,
) -> dict[str, Any]:
    return {
        "counts": listing.counts(rows),
        "rows": listing.sort_rows(rows),
        "skipped": skipped,
        "not_checked": not_checked,
        "partial": bool(not_checked),
        "cache": cache.facts(),
        "elapsed_s": round(time.monotonic() - started, 1),
    }


def list_summary(facts: dict[str, Any], out: str | None) -> dict[str, Any]:
    """The facts as the list prints them, to keep the agent's context small: every key, but in
    `rows` only the fits and unknown rows, brief (listing.brief); in `other_rows` the models of
    the other rows, by verdict; and in `rows_file` the --out file, which holds every row with
    every field."""
    shown, others = listing.summary_rows(facts.get("rows") or [])
    rows_file = ns.full_path(out) if out else None
    return {**facts, "rows": shown, "other_rows": others, "rows_file": rows_file}


def named(rows: list[dict[str, Any]]) -> str:
    """The models of the rows, at most NAMED_AT_MOST of them."""
    models = [str(row.get("model")) for row in rows]
    more = len(models) - NAMED_AT_MOST
    return ", ".join(models[:NAMED_AT_MOST]) + (f" and {more} more" if more > 0 else "")


def region_notes(shown: list[dict[str, Any]], region: str | None) -> list[str]:
    """What to tell the user about the Deployment Geography of the rows shown (fits, unknown):
    a text the script cannot read is shown as it is; a geography that leaves out places needs the
    user's country, when the region is not given or could not be judged against it."""
    other = [row for row in shown if not listing.judged_global(row.get("geography"))]
    unread = [row for row in other if listing.exclusions(row.get("geography")) is None]
    open_ = [row for row in other if row not in unread and row.get("region_excluded") is None]
    notes = []
    if unread:
        notes.append(
            f"The Deployment Geography of {named(unread)} could not be read: show it as the card "
            "says it (geography)."
        )
    if open_:
        texts = ", ".join(dict.fromkeys(str(row.get("geography")) for row in open_))
        if region is None:
            notes.append(
                f"The Deployment Geography of {named(open_)} is not Global ({texts}): "
                f"{COUNTRY_ASK}; then list again with --region COUNTRY."
            )
        else:
            notes.append(
                f"--region {region!r} could not be judged against the Deployment Geography of "
                f"{named(open_)} ({texts}): {COUNTRY_ASK}; then list again with --region "
                "COUNTRY. Until then show that geography as it is."
            )
    return notes


def list_notes(facts: dict[str, Any], run: ListRun, scan: ListScan, on_cnd: bool) -> list[str]:
    """What the agent tells the user besides the rows; only the rows it shows (fits, unknown)
    count for the notes on marks and region."""
    shown = [row for row in facts["rows"] if row.get("verdict") in listing.SHOWN]
    notes = [scan.catalog_note] if scan.catalog_note else []
    if on_cnd:
        notes.append(
            "Each row checks the newest image of the repository with a version tag and a known "
            "NGC build, as select does on a China NIM distributor; an ngc_build_unknown row has "
            "no profile check before the pull, and registry_gpus is the catalog's own list, never "
            "a verdict."
        )
    if any(row.get("needs_busy_gpus") for row in shown):
        notes.append(
            "Rows marked needs_busy_gpus need more GPUs than are idle now: the list judges every "
            "whole GPU, select judges the idle ones."
        )
    unread = [row for row in shown if row.get("records_unread")]
    if unread:
        notes.append(
            f"NGC stopped answering while the list read other images of {named(unread)} only "
            "for the tests they record for the same profile ids (records_unread): the image of "
            "each such row is select's pick, but select at the pick reads them all and may "
            "recommend another of its profiles."
        )
    if run.env.get("target") == "k8s":
        notes.append(
            "Kubernetes: each row judges the allocatable GPUs of one node, within the namespace "
            "quota, as select does; GPU usage on the nodes is unknown, so say so."
        )
    notes.extend(region_notes(shown, facts["options"]["region"]))
    if scan.outcome.late:
        again = "the cache answers the NGC part" if on_cnd else "the cache makes it faster"
        notes.append(
            f"The {listing.DEADLINE:g} s limit stopped the list: facts.not_checked names the "
            f"models not checked. Run it again: {again}."
        )
    return notes


def list_state(
    facts: dict[str, Any],
    outcome: listing.Outcome,
    on_cnd: bool,
    checked: int,
    capped: bool = False,
) -> tuple[str, ns.ScriptError | None]:
    """The list's state, and the failure to report (exit 4) when there is one: NGC or the China
    NIM distributor stopped answering (outcome.stopped_by), no model could be checked at all
    (never for the deadline alone), or NGC refused the repository record of every one of at
    least NOT_PUBLIC_ALL models it lists (`checked`). Otherwise no_candidates (no row, nothing
    left unchecked but what a CND catalog lists past the pages a list reads: `capped`), partial
    (facts.partial: rows the deadline, an error or the catalog left out), listed (a row fits) or
    none_fit."""
    rows, not_checked = facts["rows"], facts["not_checked"]
    failure = outcome.stopped_by
    cut = any(item.get("why") == "deadline" for item in not_checked)
    refused = facts["skipped"].get("not_public", 0)
    if failure is None and not rows and refused >= NOT_PUBLIC_ALL and refused == checked:
        failure = ns.ScriptError(
            "ngc_http_error",
            f"NGC refused the repository record of all {refused} NIMs checked (HTTP 401 or 403), "
            "though its catalog lists them: NGC may now ask for a key, or block this network",
            "Run the list once more later; if NGC keeps refusing, tell the user the NIMs cannot be "
            "listed without an NGC key, which nim-select never uses or asks for",
            ns.EXIT_EXTERNAL,
        )
    if failure is None and not rows and not_checked and not cut:
        first = not_checked[0]
        ngc = not on_cnd or str(first.get("why")).startswith("ngc_")
        failure = ns.ScriptError(
            "ngc_unreachable" if ngc else "cnd_unreachable",
            f"no model could be checked; {first['model']}: {first.get('message') or first['why']}",
            "Check the network or proxy, then run the list once more; the cache keeps what NGC "
            "answered",
            ns.EXIT_EXTERNAL,
        )
    if failure is not None:
        facts["partial"] = True
        state = "cnd_unreachable" if failure.code.startswith("cnd_") else "ngc_unreachable"
        return state, failure
    if not rows and (not facts["partial"] or (capped and not not_checked)):
        return "no_candidates", None
    if facts["partial"]:
        return "partial", None
    return ("listed" if facts["counts"]["fits"] else "none_fit"), None


# --- parser ---------------------------------------------------------------------------------

# Help texts that name the default, so `SUBCOMMAND --help` answers what an option does unset.
PREFER_HELP = "throughput ranks lower precisions first, accuracy higher ones (default: %(default)s)"
TOP_HELP = "shortlist length (default: %(default)s)"
MAX_GPUS_HELP = (
    "use at most N GPUs (default: the idle GPUs on a host; on Kubernetes, the GPUs of one node "
    "within the namespace quota)"
)


def add_source_options(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--source",
        choices=cnd.SOURCES,
        help="where the image comes from: ngc, or a China NIM distributor (tgcr, turingcm, "
        "lichan). Default: the host of an image reference, else ngc",
    )
    command.add_argument(
        "--nim-version",
        metavar="VERSION",
        help="the NGC version of a China NIM distributor image named with its own tag, when its "
        "labels name none (e.g. tgcr, whose labels need a login)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python3 -I "<SKILL_DIR>/scripts/nimselect.py" resolve '
        "meta/llama-3.1-8b-instruct --out ~/.cache/nim-select/resolved.json",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    locate = sub.add_parser(
        "locate",
        help="hint whether this machine is in China, to choose where to get the image from (run "
        "it with no question first; the user confirms a China source)",
        description="Read the local timezone and language, and check whether "
        f"{locality.PROBE_URL} answers within {locality.PROBE_BUDGET:g} s, to suggest where to "
        "get the NIM image from. Run it with no question first when neither the image nor the "
        "user names the source or says whether they are in China, and no source was settled "
        "earlier. Result outside_china: use NGC and tell the user so; china: ask whether they use "
        "TGC, TuringCM or Lichan (NGC stays a valid answer); unknown: ask NGC, TGC, TuringCM or "
        "Lichan. Ask in the same message as any other question, and use a China NIM distributor "
        "only on the user's answer. It reads the machine the agent runs on, which may not be "
        "where the GPUs are (then say so when you report the source), and sends nothing about "
        "the user.",
        epilog='Example: python3 -I "<SKILL_DIR>/scripts/nimselect.py" locate',
    )
    locate.add_argument("--out", metavar="FILE", help="also save the facts object as JSON")
    locate.set_defaults(func=cmd_locate)

    country = sub.add_parser(
        "country",
        help="look up the country of this machine's public IP address, for --region (only "
        "after the user agrees)",
        description="Look up the country of this machine's public IP address, for the "
        "Deployment Geography marks of list and select (--region). Run it only after the user "
        "agrees to the lookup; otherwise ask which country the NIM will run in. It sends one "
        f"request to {locality.COUNTRY_URL} (Cloudflare), which sees the address; only the "
        "country code is kept. It checks the machine the agent runs on, which may not be where "
        "the GPUs are: say so when they are elsewhere. Pass facts.country as --region to list "
        "or select. It is not used to choose NGC or a China NIM distributor (that is locate).",
        epilog='Example: python3 -I "<SKILL_DIR>/scripts/nimselect.py" country',
    )
    country.set_defaults(func=cmd_country)

    resolve = sub.add_parser(
        "resolve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="find the NIM image on NGC or a China NIM distributor and choose the version tag",
        description="Find the NIM for MODEL on NGC and choose its version tag (never 'latest'). "
        "On a China NIM distributor (--source, or an image on one), find its image and the NGC "
        "build its labels name; it searches only that distributor.",
        epilog='Example: python3 -I "<SKILL_DIR>/scripts/nimselect.py" resolve '
        "nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.4 --out ~/.cache/nim-select/resolved.json",
    )
    resolve.add_argument(
        "model", metavar="MODEL", help="image ref, nim/TEAM/REPO, TEAM/REPO, or free text"
    )
    resolve.add_argument("--tag", help="image tag to use (default: the newest version tag)")
    add_source_options(resolve)
    resolve.add_argument("--out", metavar="FILE", help="also save the facts object as JSON")
    resolve.set_defaults(func=cmd_resolve)

    inspect = sub.add_parser(
        "inspect",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="read GPUs, driver and runtime of this host or a Kubernetes namespace",
        description="Inspect where the NIM would run (read-only).",
        epilog='Example: python3 -I "<SKILL_DIR>/scripts/nimselect.py" inspect --target k8s '
        "--context prod --namespace nim --out ~/.cache/nim-select/env.json",
    )
    inspect.add_argument(
        "--target",
        choices=("host", "k8s"),
        default="host",
        help="host: this machine; k8s: a Kubernetes context and namespace (default: %(default)s)",
    )
    inspect.add_argument("--context", help="kube context (required with --target k8s)")
    inspect.add_argument("--namespace", help="namespace (required with --target k8s)")
    inspect.add_argument(
        "--gpus", help="host GPUs to count: all (default), or indices like 0,1 (host only)"
    )
    inspect.add_argument("--cache-dir", help="NIM cache folder for the disk check (~/.cache/nim)")
    inspect.add_argument("--out", metavar="FILE", help="save the environment as JSON for fit")
    inspect.set_defaults(func=cmd_inspect)

    fit_ = sub.add_parser(
        "fit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="check which profiles of the NIM fit a saved environment",
        description="Match the NIM Metadata API profiles of MODEL against an environment file. "
        "As select does, it counts a profile as tested when another image of the repository that "
        "select checks records that profile id, with the same workspace hash, as built for or "
        "tested on the GPU (tested_in_other_image); with --metadata-file, and on a China NIM "
        "distributor, it reads none.",
        epilog='Example: python3 -I "<SKILL_DIR>/scripts/nimselect.py" fit '
        "meta/llama-3.1-8b-instruct --tag 2.1.4 --env ~/.cache/nim-select/env.json --max-gpus 2",
    )
    fit_.add_argument("model", metavar="MODEL", help="as for resolve")
    fit_.add_argument("--tag", help="image tag (default: the newest version tag)")
    add_source_options(fit_)
    fit_.add_argument("--env", required=True, metavar="FILE", help="file from inspect --out")
    fit_.add_argument("--max-gpus", type=int, metavar="N", help=MAX_GPUS_HELP)
    fit_.add_argument("--need-lora", action="store_true", help="only profiles that support LoRA")
    fit_.add_argument("--backend", metavar="NAME", help="inference backend, e.g. vllm")
    fit_.add_argument(
        "--prefer", choices=("throughput", "accuracy"), default="throughput", help=PREFER_HELP
    )
    fit_.add_argument(
        "--metadata-file",
        metavar="FILE",
        help="NIM metadata JSON to use instead of the API (NGC is still asked for the repository "
        "first; the file is used alone only when NGC cannot be reached)",
    )
    fit_.add_argument("--top", type=int, default=5, metavar="N", help=TOP_HELP)
    fit_.add_argument("--out", metavar="FILE", help="also save the facts object as JSON")
    fit_.set_defaults(func=cmd_fit)

    select = sub.add_parser(
        "select",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="choose the image and profile that fit a saved environment",
        description="Check the candidate images of MODEL against an environment file and pick "
        "one: the newest version, hardware variants (never production branches), then older "
        "versions: the earlier builds of the newest's major.minor line, newest first, in the "
        "places the limit leaves after the newest build of the two older major.minor lines, "
        "then those builds. A profile that an image's own metadata records neither as built for "
        "nor as tested on the GPU counts as tested when another image of the same repository it "
        "checks records that profile id, with the same workspace hash, as built for or tested on "
        "it (tested_in_other_image; a why line names that image). The first that fits wins, else "
        "the first unknown, or a later unknown whose profile was built for or tested on the GPU "
        "and leaves fewer checks open. A tag in MODEL or --tag checks only that image, and reads "
        "the others a select without it checks for the tests they record alone; --profile checks "
        "only that profile. On a China NIM distributor, it "
        "checks one image: the one named, or the newest with a known NGC build.",
        epilog='Example: python3 -I "<SKILL_DIR>/scripts/nimselect.py" select '
        "meta/llama-3.1-8b-instruct --env ~/.cache/nim-select/env.json --max-gpus 2 "
        "--out ~/.cache/nim-select/select.json",
    )
    select.add_argument("model", metavar="MODEL", help="as for resolve")
    select.add_argument("--tag", help="check only this tag (default: search the images)")
    select.add_argument(
        "--profile",
        type=profile_id,
        metavar="ID",
        help="check only this profile, by its full profile_id, in the images select checks (with "
        "--tag, in that image alone), on the GPU type select picks without it; the hand-off "
        "names it. Use it when the user takes "
        "facts.alternative (select again with --tag TAG --profile=ID, as its note says) or "
        "another profile from facts.shortlist or facts.unconfirmed. An id that no image checked "
        "has is profile_not_found (exit 2)",
    )
    add_source_options(select)
    select.add_argument("--env", required=True, metavar="FILE", help="file from inspect --out")
    select.add_argument("--max-gpus", type=int, metavar="N", help=MAX_GPUS_HELP)
    select.add_argument("--need-lora", action="store_true", help="only profiles that support LoRA")
    select.add_argument("--backend", metavar="NAME", help="inference backend, e.g. vllm")
    select.add_argument(
        "--prefer", choices=("throughput", "accuracy"), default="throughput", help=PREFER_HELP
    )
    select.add_argument(
        "--max-images",
        type=int,
        default=choose.DEFAULT_MAX_IMAGES,
        metavar="N",
        help=f"check at most N images (default {choose.DEFAULT_MAX_IMAGES})",
    )
    select.add_argument("--top", type=int, default=5, metavar="N", help=TOP_HELP)
    select.add_argument(
        "--region",
        metavar="REGION",
        help="where the NIM will run, in the user's words (germany, us, india) or as the country "
        "command gives it (DE): marks the NIM when its Deployment Geography excludes it",
    )
    select.add_argument(
        "--out",
        metavar="FILE",
        help="also save the facts object as JSON once every image is checked, on no_metadata "
        "too (an error that stops select earlier, and profile_not_found, leave the file as it "
        "was)",
    )
    select.set_defaults(func=cmd_select)

    list_ = sub.add_parser(
        "list",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="list the LLM and VLM NIMs of the source with select's verdict on a saved "
        "environment (only fits rows are compatible)",
        description="List the LLM and VLM NIMs of the source (NGC, or one China NIM distributor) "
        "with select's verdict on the GPUs of an environment file: the images select checks "
        "(the newest; when it does not fit, hardware variants and older versions; and every "
        "image of the repository select checks when the tests they record for the same profile "
        "ids could change the row), with select's own tag, fit and ranking rules. Each row: the "
        "verdict (fits, unknown, does_not_fit, "
        "no_metadata, ngc_build_unknown); only fits rows are compatible, and an unknown row is "
        "not confirmed. A row gives the smallest GPU count that fits, its precision and backend, "
        "and marks: needs_ngc_key, needs_busy_gpus, region_excluded. It prints the fits and "
        "unknown rows, brief, and the models of the others; --out FILE keeps every row. Output "
        "schema nim-select/list/v1. It never hands off: run select on the model the user picks. "
        "NGC answers are cached in ~/.cache/nim-select/ngc-cache.json (metadata 24 h, the rest "
        f"1 h); the list stops after {listing.DEADLINE:g} s and keeps the rows read.",
        epilog='Example: python3 -I "<SKILL_DIR>/scripts/nimselect.py" list '
        "--env ~/.cache/nim-select/env.json --source ngc --query llama "
        "--out ~/.cache/nim-select/list.json",
    )
    list_.add_argument("--env", required=True, metavar="FILE", help="file from inspect --out")
    list_.add_argument(
        "--source",
        required=True,
        choices=cnd.SOURCES,
        help="the source the Locality step chose: ngc, or a China NIM distributor (tgcr, "
        "turingcm, lichan)",
    )
    list_.add_argument(
        "--query",
        metavar="TEXT",
        help="only models whose name holds every word of TEXT (a word with a digit, such as 4 or "
        "8b, never inside a longer number or size: 4 is not in 49b or nano-4b)",
    )
    list_.add_argument(
        "--region",
        metavar="REGION",
        help="where the NIMs will run, in the user's words (eu, germany, us, india) or as the "
        "country command gives it (DE): marks the rows whose Deployment Geography excludes it",
    )
    list_.add_argument(
        "--max-gpus",
        type=int,
        metavar="N",
        help="use at most N GPUs (default: every whole GPU on a host, idle or not, a row that "
        "needs busy ones marked needs_busy_gpus; on Kubernetes, the GPUs of one node within the "
        "namespace quota)",
    )
    list_.add_argument("--need-lora", action="store_true", help="only profiles that support LoRA")
    list_.add_argument("--backend", metavar="NAME", help="inference backend, e.g. vllm")
    list_.add_argument(
        "--prefer", choices=("throughput", "accuracy"), default="throughput", help=PREFER_HELP
    )
    list_.add_argument(
        "--no-cache", action="store_true", help="neither read nor write the cache of NGC answers"
    )
    list_.add_argument(
        "--out", metavar="FILE", help="also save the facts object as JSON, with every row"
    )
    list_.set_defaults(func=cmd_list)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
