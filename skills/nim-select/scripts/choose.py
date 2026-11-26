#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image choice for `nimselect.py select`: which images of a model to check, and which one to pick.

Pure functions, no I/O. The candidate images of one model are, in this order: the newest version,
hardware variants (repositories `<repo>-<hardware>` in the same team, where the suffix names
hardware such as `dgx-spark` or `rtx`, and a tag of the repo whose suffix names the GPU, e.g.
`1.8.0-RTX`), then older versions: the earlier builds of the newest's major.minor line, newest
first (2.1.5 and 2.1.4 after 2.1.6), in the places the limit leaves after the newest build of the
next older major.minor lines, which follow them. Production-branch repositories (`-pb6`,
`-pb25h2`) and other repositories that only share the name prefix (`deepseek-r1-distill-llama-70b`,
`gpt-oss-120b-turbo`: other models) are never picked automatically. The first candidate that fits
wins; else the first `unknown`, or a later `unknown` whose best profile was built for or tested on
this GPU and leaves fewer checks open; else the first image with published profiles. Run
directly, this module prints a notice: `nimselect.py select` is the command.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ngc_api
import ns_common as ns

Kind = Literal["requested", "newest", "variant", "older"]

DEFAULT_MAX_IMAGES = 6
OLDER_LINES = 2
PRODUCTION_BRANCH = re.compile(r"^pb[0-9]")
BASE_FAMILY = re.compile(r"^base(?:-|$)")
# Repository suffixes that name hardware are ngc_api.HARDWARE_SUFFIXES. Anything else after the
# name (`-0528`, `-distill-*`, `-turbo`) is another model or build, never a hardware variant.
VERDICT_ORDER = ("fits", "unknown", "does_not_fit")
# GPU-match tiers that are evidence for this GPU (fit.py: built for it, tested on it).
MATCHED_TIERS = ("direct", "tested")
# Codes in a profile's `reasons` that are notes, not checks left open (fit.py: PCI_INFERRED_NOTE,
# TESTED_ELSEWHERE).
NOTE_CODES = frozenset({"pci_inferred", "tested_in_other_image"})
# The error_code of an image whose metadata lacks the profile `select --profile` names.
PROFILE_NOT_FOUND = "profile_not_found"


@dataclass(frozen=True)
class Candidate:
    """One image to check."""

    resource_id: str
    tag: str
    kind: Kind
    architectures: tuple[str, ...] | None = None

    @property
    def image(self) -> str:
        return f"nvcr.io/{self.resource_id}:{self.tag}"


# --- tags -------------------------------------------------------------------------------------


def tag_suffix(tag: str) -> str | None:
    """`1.8.0-RTX` -> `RTX`; None for a plain version or a non-version tag."""
    parsed = ngc_api.parse_tag(tag)
    if parsed is None or not parsed[1]:
        return None
    return parsed[1]


def older_tags(
    tags: Iterable[str],
    newest: str,
    count: int = OLDER_LINES,
    digests: ngc_api.Digests | None = None,
) -> list[str]:
    """The newest build of each of the next `count` major.minor lines below `newest`.

    A build is a version tag with three or more numbers for any GPU (is_general), so a line whose
    builds carry a suffix (`1.7.0-variant`) counts too. A major.minor alias (`1.7`) stands for the
    build it points to (`digests`, from the public image list, else its version numbers), and is
    used as it is only when no build is listed for it; `1` alone is skipped. A plain build wins
    over a suffixed one with the same numbers. After 2.1.4 come 2.0.13 and 1.15.5.
    """
    listed = list(tags)
    top = ngc_api.parse_tag(newest)
    if top is None:
        return []
    current = top[0][:2]
    best: dict[tuple[int, ...], tuple[Any, str]] = {}
    for tag in listed:
        parsed = ngc_api.parse_tag(tag)
        if parsed is None or len(parsed[0]) < 2 or not ngc_api.is_general(tag):
            continue
        if len(parsed[0]) == 2:
            tag = ngc_api.alias_target(tag, listed, digests) or tag
            if not ngc_api.is_general(tag):
                continue
        key = ngc_api.version_key(tag)
        if key is None:
            continue
        line = key[0][:2]
        if line >= current:
            continue
        if line not in best or key > best[line][0]:
            best[line] = (key, tag)
    return [best[line][1] for line in sorted(best, reverse=True)[:count]]


def _general_builds(tags: Iterable[str]) -> list[tuple[Any, str]]:
    """(version key, tag) of each build for any GPU: three or more numbers, no prerelease and no
    build for named hardware (`1.7.0-variant` counts; `1.8.0-RTX`, `2.0.0-rc1` and the moving
    aliases `2.1`, `1` do not)."""
    return [
        (key, tag)
        for tag in dict.fromkeys(tags)
        if ngc_api.is_build(tag)
        and ngc_api.is_general(tag)
        and (key := ngc_api.version_key(tag)) is not None
    ]


def same_image(first: str, second: str, digests: ngc_api.Digests | None = None) -> bool:
    """Whether two version tags name one image: the same digest in the public image list, or,
    without the digest of both, two builds for any GPU with the same version numbers (never a
    build for named hardware such as `1.8.0-RTX`). A plain build and its suffixed twin (`1.7.0`,
    `1.7.0-variant`) were the same image on every NIM that listed both on 2026-11-11."""
    known = digests or {}
    if known.get(first) and known.get(second):
        return known[first] == known[second]
    one, other = ngc_api.parse_tag(first), ngc_api.parse_tag(second)
    return (
        one is not None
        and other is not None
        and one[0] == other[0]
        and ngc_api.is_general(first)
        and ngc_api.is_general(second)
    )


def line_builds(
    tags: Iterable[str],
    newest: str,
    digests: ngc_api.Digests | None = None,
    count: int | None = None,
) -> list[str]:
    """The earlier builds of the newest's major.minor line, newest first, at most `count`: 2.1.5
    and 2.1.4 after 2.1.6, `1.7.0-variant` after the arm64-only `1.7.1-variant`. A newest build
    can lose a GPU from its tested list (on 2026-11-11, 2.1.6 of llama-3.1-8b-instruct lists no
    profile as tested on the H100 NVL; 2.1.5 lists four), so the builds just before it come first
    among the older versions. A build that is the same image as the newest or as one taken
    (same_image: `1.7.0-variant` beside `1.7.0`) is skipped."""
    top = ngc_api.version_key(newest)
    if top is None:
        return []
    taken = [newest]
    for key, tag in sorted(_general_builds(tags), reverse=True):
        if count is not None and len(taken) > count:
            break
        if key[0][:2] != top[0][:2] or key >= top:
            continue
        if not any(same_image(tag, other, digests) for other in taken):
            taken.append(tag)
    return taken[1:]


def line_build_count(limit: int, ahead: int, older_lines: int) -> int:
    """How many earlier builds of the newest's line select checks (line_builds): the places that
    `limit` leaves after the `ahead` images checked before them (the newest, the hardware variants
    and the repository's tag for the GPU) and the `older_lines` builds checked after them, and at
    least one. So the older lines keep their places: on an RTX 6000 Ada, llama-3.1-8b-instruct's
    2.1.6, its DGX Spark build and 1.8.0-RTX leave one place before 2.0.13 and 1.15.5 (2.1.5); on
    an H100, two (2.1.5, 2.1.4)."""
    return max(1, limit - ahead - older_lines)


def unchecked_newer(
    tags: Iterable[str],
    chosen: str,
    checked: Iterable[str],
    digests: ngc_api.Digests | None = None,
) -> list[str]:
    """The builds for any GPU of the repository with a higher version than `chosen` that are not
    among `checked` (the repository's tags select checked) or the same image as one of them
    (same_image), newest first: with 2.1.4, 2.0.13 and 1.15.5 checked and 1.15.5 chosen, 2.0.12
    down to 2.0.1. Whether they fit is not known."""
    floor = ngc_api.version_key(chosen)
    if floor is None:
        return []
    seen = list(checked)
    newer = [
        (key, tag)
        for key, tag in _general_builds(tags)
        if key[0] > floor[0] and not any(same_image(tag, other, digests) for other in seen)
    ]
    return [tag for _, tag in sorted(newer, reverse=True)]


def _mentions(name: str, word: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(word.lower())}(?![a-z0-9])"
    return re.search(pattern, name.lower()) is not None


def gpu_variant_tag(tags: Iterable[str], gpu_names: Iterable[str]) -> str | None:
    """The newest tag whose suffix names one of the GPUs (`1.8.0-RTX` for `NVIDIA RTX 6000 Ada`).

    The suffix must appear as a whole word in a GPU group name, ignoring case.
    """
    names = [name for name in gpu_names if name]
    matching = [
        tag
        for tag in tags
        if (suffix := tag_suffix(tag)) and any(_mentions(name, suffix) for name in names)
    ]
    ordered = ngc_api.version_tags(matching)
    return ordered[0] if ordered else None


# --- repositories -----------------------------------------------------------------------------


is_hardware_suffix = ngc_api.is_hardware_suffix


@dataclass(frozen=True)
class Related:
    """Catalog hits that share the requested repository's name prefix, sorted and unique."""

    variants: list[str]
    production: list[str]
    other: list[str]


def split_variants(base_id: str, resource_ids: Iterable[str]) -> Related:
    """Sort catalog search hits of `base_id` into hardware variants, production branches, others.

    A variant is `<team>/<base leaf>-<hardware suffix>` in the same team (`is_hardware_suffix`).
    Production branches (`-pb6`, `-pb25h2`) are listed apart; the `-base` model family is dropped;
    any other `<base leaf>-<suffix>` repo is another model (`-0528`, `-distill-*`, `-turbo`).
    """
    team, _, leaf = base_id.lower().rpartition("/")
    prefix = f"{team}/{leaf}-"
    groups: dict[str, set[str]] = {"variants": set(), "production": set(), "other": set()}
    for rid in resource_ids:
        lowered = rid.lower()
        if not lowered.startswith(prefix):
            continue
        suffix = lowered[len(prefix) :]
        if not suffix or "/" in suffix or BASE_FAMILY.match(suffix):
            continue
        if PRODUCTION_BRANCH.match(suffix):
            groups["production"].add(rid)
        elif is_hardware_suffix(suffix):
            groups["variants"].add(rid)
        else:
            groups["other"].add(rid)
    return Related(
        variants=sorted(groups["variants"]),
        production=sorted(groups["production"]),
        other=sorted(groups["other"]),
    )


def plan(
    candidates: Sequence[Candidate], limit: int = DEFAULT_MAX_IMAGES
) -> tuple[list[Candidate], list[Candidate]]:
    """(images to check, images left out by `limit`): repeats dropped, order kept.

    When the limit would leave out every older version, and it allows 3 or more images, the first
    older version takes the place of the last variant, so a newer image that does not fit can
    still fall back to an older one.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[Candidate] = []
    for candidate in candidates:
        key = (candidate.resource_id, candidate.tag)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    kept, dropped = unique[:limit], unique[limit:]
    older = next((c for c in dropped if c.kind == "older"), None)
    if (
        older is not None
        and limit >= 3
        and not any(c.kind == "older" for c in kept)
        and kept[-1].kind == "variant"
    ):
        kept, dropped = [*kept[:-1], older], [kept[-1], *(c for c in dropped if c is not older)]
    return kept, dropped


# --- the choice -------------------------------------------------------------------------------


def open_checks(item: dict[str, Any]) -> list[str]:
    """The checks an image's best profile leaves open: its `reasons`, notes aside."""
    return [str(code) for code in item.get("reasons") or () if code not in NOTE_CODES]


def pick(considered: Sequence[dict[str, Any]]) -> int | None:
    """Index of the image to report: the first `fits`; else the first `unknown`, unless a later
    `unknown` whose best profile was built for or tested on this GPU (`tier`) leaves no check open
    that the first one leaves closed, so it confirms more; else the first evaluated."""
    for verdict in VERDICT_ORDER:
        found = [index for index, item in enumerate(considered) if item.get("verdict") == verdict]
        if not found:
            continue
        first = considered[found[0]]
        if verdict == "unknown" and first.get("tier") not in MATCHED_TIERS:
            for index in found[1:]:
                item = considered[index]
                if item.get("tier") in MATCHED_TIERS and set(open_checks(item)) <= set(
                    open_checks(first)
                ):
                    return index
        return found[0]
    return None


def label(item: dict[str, Any], base_id: str | None = None) -> str:
    """The tag, or `repo-leaf:tag` when the image is in another repository than `base_id`."""
    rid = item.get("resource_id")
    if base_id and rid and rid != base_id:
        return f"{str(rid).rsplit('/', 1)[-1]}:{item['tag']}"
    return str(item["tag"])


def short_status(item: dict[str, Any]) -> str:
    """Why one image was not taken, in a few words."""
    if item.get("metadata") == "not_published":
        return "no published profiles"
    if item.get("metadata") == "error":
        return "metadata could not be read"
    if item.get("error_code") == PROFILE_NOT_FOUND:
        return "no profile with the id asked for (--profile)"
    if item.get("verdict") == "unknown":
        return "fit could not be confirmed" + _open_codes(item)
    return "no profile runs here" + _top_codes(item)


def _top_codes(item: dict[str, Any]) -> str:
    counts = item.get("rejected_counts") or {}
    top = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:2]
    return f" ({', '.join(code for code, _ in top)})" if top else ""


def _open_codes(item: dict[str, Any]) -> str:
    codes = open_checks(item)
    return f" ({', '.join(codes)})" if codes else ""


def _matched_as(item: dict[str, Any]) -> str:
    return "tested on" if item.get("tier") == "tested" else "built for"


def _newest_has(item: dict[str, Any]) -> str:
    """`the newest image has ...` for the newest image's status."""
    if item.get("metadata") == "not_published":
        return "no published profiles"
    if item.get("metadata") == "error":
        return "metadata that could not be read"
    if item.get("error_code") == PROFILE_NOT_FOUND:
        return "no profile with the id asked for (--profile)"
    if item.get("verdict") == "unknown":
        return "a fit that could not be confirmed" + _open_codes(item)
    return "no profile that runs here" + _top_codes(item)


def reason(
    considered: Sequence[dict[str, Any]],
    index: int,
    base_id: str | None = None,
    asked: str | None = None,
) -> str:
    """One plain sentence on why considered[index] was picked; `base_id` is the requested repo,
    `asked` the moving alias the user named when its build was checked instead (`1`)."""
    chosen = considered[index]
    verdict = chosen.get("verdict")
    kind = chosen.get("kind")
    earlier = list(considered[:index])
    status = "; ".join(f"{label(item, base_id)}: {short_status(item)}" for item in earlier)
    if kind == "requested":
        what = f"the build `{asked}` stands for" if asked else "the tag you asked for"
        return {
            "fits": f"{what}; it fits",
            "unknown": f"{what}; its fit could not be confirmed",
        }.get(str(verdict), f"{what}; no profile of it runs here")
    if verdict == "does_not_fit":
        return "no image fits here; this is the first checked image with published profiles"
    if verdict == "unknown":
        start = {"newest": "newest version", "variant": "hardware variant"}.get(
            str(kind), "older version"
        )
        tail = f" ({status})" if status else ""
        first = next((item for item in earlier if item.get("verdict") == "unknown"), None)
        if first is not None:
            # pick() passed over the first unknown image for this one (see its docstring)
            return (
                f"{start}: no image is confirmed to fit, and its profile was "
                f"{_matched_as(chosen)} this GPU and leaves fewer checks open than that of "
                f"{label(first, base_id)}{tail}"
            )
        return f"{start}: no image is confirmed to fit and this one comes first{tail}"
    if kind == "newest":
        return "newest version that fits"
    if kind == "variant":
        newest = next((item for item in earlier if item.get("kind") == "newest"), None)
        if newest is None:
            return "hardware variant for this GPU"
        return f"hardware variant for this GPU; the newest image has {_newest_has(newest)}"
    return f"older version: the newer images checked are not confirmed to fit ({status})"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Image choice for nim-select (a library); `nimselect.py select` runs it.",
        epilog='Example: python3 -I "<SKILL_DIR>/scripts/nimselect.py" select '
        "meta/llama-3.1-8b-instruct --env ~/.cache/nim-select/env.json",
    )
    parser.parse_args(argv)
    ns.emit(
        True, "library", "done", {"note": "import this module; run nimselect.py select instead"}
    )


if __name__ == "__main__":
    main()
