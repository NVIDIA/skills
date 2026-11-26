# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/choose.py: which images select checks, in which order, and which it picks."""

from __future__ import annotations

from typing import Any

import choose
import pytest
from ns_testkit import fixture_json, run_cli

# nim/meta/llama-3.1-8b-instruct on 2026-11-04, when 2.1.4 was the newest version
REAL_TAGS: list[str] = fixture_json("ngc", "repo-llama-3.1-8b-instruct.json")["tags"]
BASE = "nim/meta/llama-3.1-8b-instruct"


def test_running_it_prints_a_library_notice() -> None:
    code, payload = run_cli("choose.py")
    assert code == 0
    assert payload["state"] == "library"


@pytest.mark.parametrize(
    ("newest", "expected"),
    [
        ("2.1.4", ["2.0.13", "1.15.5"]),
        ("2.0.13", ["1.15.5", "1.13.1"]),
        ("1.3.3", ["1.2.2", "1.1.2"]),
        ("1.1.2", []),
        ("latest", []),
    ],
)
def test_older_tags_take_the_newest_patch_of_the_next_two_lines(
    newest: str, expected: list[str]
) -> None:
    assert choose.older_tags(REAL_TAGS, newest) == expected


def test_older_tags_skip_prereleases_hardware_builds_and_major_only_tags() -> None:
    tags = ["3.0.0", "2.5.0-rc1", "2", "2.4", "2.4.1", "1.0.0-RTX"]
    assert choose.older_tags(tags, "3.0.0") == ["2.4.1"]


# Real NGC tag sets on 2026-11-04. A line whose builds carry a suffix counts, and its
# major.minor alias stands for the build it points to: 1.7 is 1.7.0-variant, 1.0 is 1.0.0-variant.
MISTRAL_SMALL_4 = [
    "2.1.5", "2.1.4", "2.0.13", "2.0.12", "2.0.6-variant", "latest", "1", "1.7", "1.7.0-variant",
]  # fmt: skip
QWEN3_NEXT = ["1.5.0-variant", "1.5", "1.0", "1", "1.0.0-variant", "latest"]


@pytest.mark.parametrize(
    ("tags", "newest", "expected"),
    [
        (MISTRAL_SMALL_4, "2.1.5", ["2.0.13", "1.7.0-variant"]),
        (QWEN3_NEXT, "1.5.0-variant", ["1.0.0-variant"]),
        (["2.1.6", "2.0.13", "1.11.0-variant", "latest"], "2.1.6", ["2.0.13", "1.11.0-variant"]),
        (["1.2.0", "1.1", "1.0.0"], "1.2.0", ["1.1", "1.0.0"]),
        (["1.9.0", "1.8.6", "1.8", "1.8.0-RTX"], "1.9.0", ["1.8.6"]),
        (["2.1.5", "1.8.3-variant", "1.8", "1.8.1", "1.8.1-variant"], "2.1.5", ["1.8.3-variant"]),
    ],
    ids=[
        "mistral-small-4-119b-2603",
        "qwen3-next-80b-a3b-instruct",
        "variant-line-without-alias",
        "alias-without-build",
        "rtx-build-left-out",
        "nemotron-3-super-120b-a12b",
    ],
)
def test_older_tags_take_builds_not_moving_aliases(
    tags: list[str], newest: str, expected: list[str]
) -> None:
    assert choose.older_tags(tags, newest) == expected


def test_older_tags_follow_the_image_digests() -> None:
    """nim/nvidia/llama-nemotron-rerank-500m-v2: `1.1` is the 1.10.0 image, not an older one."""
    tags = ["1", "1.1", "latest", "1.10.0"]
    assert choose.older_tags(tags, "1.10.0") == ["1.1"]
    assert choose.older_tags(tags, "1.10.0", digests=dict.fromkeys(tags, "sha256:3e39")) == []


# nim/meta/llama-3.1-8b-instruct on 2026-11-11 (the list fixture's repository record)
LIVE_TAGS: list[str] = fixture_json("list", "repo-meta__llama-3.1-8b-instruct.json")["tags"]


@pytest.mark.parametrize(
    ("tags", "newest", "expected"),
    [
        (LIVE_TAGS, "2.1.6", ["2.1.5", "2.1.4"]),
        (LIVE_TAGS, "2.1.5", ["2.1.4"]),
        (LIVE_TAGS, "2.1.4", []),
        (LIVE_TAGS, "2.0.13", [f"2.0.{n}" for n in range(12, 0, -1)]),
        # qwen3.6-35b-a3b: the arm64-only 1.7.1-variant, then 1.7.0-variant (amd64 and arm64)
        (["1.7.1-variant", "1.7.0-variant", "latest"], "1.7.1-variant", ["1.7.0-variant"]),
        # a plain build, never its suffixed twin with the same numbers (one image), a prerelease,
        # a build for named hardware, a moving alias, another line or a newer build
        (
            ["2.0.3", "2.0.2", "2.0.1-variant", "2.0.1", "2.0.1-rc1", "2.0.0-RTX", "2.0", "1.9.0"],
            "2.0.2",
            ["2.0.1"],
        ),
        (["latest", "1"], "latest", []),
    ],
    ids=["2.1.6", "2.1.5", "first-of-line", "long-line", "one-platform", "rules", "no-version"],
)
def test_line_builds_are_the_earlier_builds_of_the_newest_line(
    tags: list[str], newest: str, expected: list[str]
) -> None:
    assert choose.line_builds(tags, newest) == expected


# Live NGC tags and image digests on 2026-11-11: a plain build and its `-variant` twin are one
# image.
MINIMAX_M25 = ["1.7.0", "1.7.0-variant", "1", "1.7.1-variant", "1.7.1", "latest"]
MINIMAX_DIGESTS = {
    "1.7.0": "sha256:c031ab1ebc8c",
    "1.7.0-variant": "sha256:c031ab1ebc8c",
    "1": "sha256:43942fa10760",
    "1.7.1-variant": "sha256:43942fa10760",
    "1.7.1": "sha256:43942fa10760",
    "latest": "sha256:43942fa10760",
}
KIMI_K25 = ["1.7.0-variant", "latest", "1", "1.7.0", "1.7"]
KIMI_DIGESTS = dict.fromkeys(KIMI_K25, "sha256:aba5f879b458")
NEMOTRON_SUPER_EXP = [
    "1.8.3-variant", "1.8.2-variant", "1.8", "1", "1.8.1-variant", "1.8.1", "latest",
]  # fmt: skip
NEMOTRON_SUPER_EXP_DIGESTS = {
    "latest": "sha256:e8355971fc7e",
    "1": "sha256:e8355971fc7e",
    "1.8": "sha256:e8355971fc7e",
    "1.8.3-variant": "sha256:e8355971fc7e",
    "1.8.2-variant": "sha256:90efe5fdb2ad",
    "1.8.1-variant": "sha256:a7e96e45d2ad",
    "1.8.1": "sha256:a7e96e45d2ad",
}


@pytest.mark.parametrize(
    ("tags", "digests", "newest", "expected"),
    [
        (KIMI_K25, KIMI_DIGESTS, "1.7.0", []),
        (MINIMAX_M25, MINIMAX_DIGESTS, "1.7.1", ["1.7.0"]),
        (NEMOTRON_SUPER_EXP, NEMOTRON_SUPER_EXP_DIGESTS, "1.8.3-variant",
         ["1.8.2-variant", "1.8.1"]),
        # no image list (it is not public): the same numbers make a twin
        (MINIMAX_M25, None, "1.7.1", ["1.7.0"]),
        # twins whose digests differ are two images
        (["1.7.1", "1.7.1-variant"], {"1.7.1": "sha256:a", "1.7.1-variant": "sha256:b"}, "1.7.1",
         ["1.7.1-variant"]),
    ],
    ids=["kimi-k2.5", "minimax-m25", "nemotron-3-super-120b-a12b-experimental", "no-digests",
         "other-image"],
)  # fmt: skip
def test_line_builds_check_each_image_once(
    tags: list[str], digests: dict[str, str] | None, newest: str, expected: list[str]
) -> None:
    """A suffixed twin of the newest or of a build taken (same digest, same image and metadata) is
    no other image to check: it would cost a metadata read and a place in --max-images."""
    assert choose.line_builds(tags, newest, digests) == expected


def test_same_image() -> None:
    assert choose.same_image("1.7.1-variant", "1.7.1", MINIMAX_DIGESTS)
    assert not choose.same_image("1.7.0", "1.7.1", MINIMAX_DIGESTS)
    assert choose.same_image("2.0.1-variant", "2.0.1")  # no digests: the same numbers
    assert not choose.same_image("2.0.1", "2.0.0")
    # a build for named hardware is never the twin of a plain build by its numbers alone
    assert not choose.same_image("1.8.0-RTX", "1.8.0")
    assert not choose.same_image("latest", "2.1.6") and not choose.same_image("2.1.6", "latest")


def test_line_builds_stop_at_the_count() -> None:
    assert choose.line_builds(LIVE_TAGS, "2.1.6", count=1) == ["2.1.5"]
    assert choose.line_builds(LIVE_TAGS, "2.0.13", count=3) == ["2.0.12", "2.0.11", "2.0.10"]
    assert choose.line_builds(MINIMAX_M25, "1.7.1", MINIMAX_DIGESTS, count=1) == ["1.7.0"]


@pytest.mark.parametrize(
    ("ahead", "older_lines", "expected"),
    [
        (2, 2, 2),  # H100 NVL: 2.1.6 and its DGX Spark build, then 2.1.5 and 2.1.4
        (3, 2, 1),  # RTX 6000 Ada: 1.8.0-RTX takes a place, so 2.1.5 alone
        (1, 0, 5),  # no variant and no older line
        (6, 2, 1),  # variants fill the limit: still one (choose.plan keeps an older version)
    ],
    ids=["h100", "rtx", "one-line", "full"],
)
def test_line_build_count_leaves_the_older_lines_their_places(
    ahead: int, older_lines: int, expected: int
) -> None:
    assert choose.line_build_count(choose.DEFAULT_MAX_IMAGES, ahead, older_lines) == expected


def test_unchecked_newer_names_the_builds_between_the_choice_and_the_newest() -> None:
    checked = ["2.1.6", "2.1.5", "2.1.4", "2.0.13", "1.15.5"]
    assert choose.unchecked_newer(LIVE_TAGS, "2.1.5", checked) == []
    assert choose.unchecked_newer(LIVE_TAGS, "2.0.13", checked) == []
    assert choose.unchecked_newer(LIVE_TAGS, "1.15.5", checked) == [
        f"2.0.{n}" for n in range(12, 0, -1)
    ]
    # a build for named hardware: the builds for any GPU with a higher version, never a moving
    # alias (1.8, 1.10), a prerelease or another build of the same version
    newer = choose.unchecked_newer(LIVE_TAGS, "1.8.0-RTX", checked)
    assert newer[:3] == ["2.0.12", "2.0.11", "2.0.10"] and newer[-3:] == ["1.8.5", "1.8.4", "1.8.3"]
    assert not {"1.8", "1.10", "1.15", "latest", "1", "1.8.0-RTX"} & set(newer)
    assert choose.unchecked_newer(["2.0.6", "2.0.6-variant"], "2.0.6-variant", []) == []
    assert choose.unchecked_newer(LIVE_TAGS, "latest", checked) == []
    # the twin of a build checked is that image: not a newer tag that was not checked
    assert choose.unchecked_newer(MINIMAX_M25, "1.7.0", ["1.7.1", "1.7.0"], MINIMAX_DIGESTS) == []
    assert choose.unchecked_newer(MINIMAX_M25, "1.7.0", ["1.7.1", "1.7.0"]) == []


def test_tag_suffix() -> None:
    assert choose.tag_suffix("1.8.0-RTX") == "RTX"
    assert choose.tag_suffix("2.1.4") is None
    assert choose.tag_suffix("latest") is None


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["NVIDIA RTX 6000 Ada Generation"], "1.8.0-RTX"),
        (["NVIDIA GeForce RTX 5090"], "1.8.0-RTX"),
        (["NVIDIA H100 NVL"], None),
        (["NVIDIA RTXPRO"], None),
        ([], None),
    ],
)
def test_gpu_variant_tag_matches_the_suffix_as_a_word(
    names: list[str], expected: str | None
) -> None:
    assert choose.gpu_variant_tag(REAL_TAGS, names) == expected


def test_gpu_variant_tag_takes_the_newest_match() -> None:
    tags = ["1.0.0-a100", "1.2.0-A100", "1.1.0-a100", "2.0.0"]
    assert choose.gpu_variant_tag(tags, ["NVIDIA A100-SXM4-80GB"]) == "1.2.0-A100"


def test_split_variants_keeps_hardware_variants_and_lists_production_branches() -> None:
    hits = [
        BASE,
        f"{BASE}-dgx-spark",
        f"{BASE}-pb6",
        f"{BASE}-pb25h2",
        f"{BASE}-pb24h2",
        "nim/meta/llama-3.1-8b-base",
        "nim/meta/llama-3.1-70b-instruct",
        "nim/other/llama-3.1-8b-instruct-dgx-spark",
        f"{BASE}-dgx-spark",
    ]
    related = choose.split_variants(BASE, hits)
    assert related.variants == [f"{BASE}-dgx-spark"]
    assert related.production == [f"{BASE}-pb24h2", f"{BASE}-pb25h2", f"{BASE}-pb6"]
    assert related.other == []


def test_split_variants_drops_the_base_family_and_dotted_names() -> None:
    base = "nim/nvidia/llama-3.3-nemotron-super-49b-v1"
    hits = [f"{base}.5", f"{base}.5-pb6", f"{base}-base", f"{base}-base-fp8", f"{base}-rtx"]
    assert choose.split_variants(base, hits) == choose.Related([f"{base}-rtx"], [], [])


def test_split_variants_keeps_other_models_with_the_same_prefix_apart() -> None:
    """deepseek-r1-distill-* and -0528 are other models, not hardware builds of deepseek-r1."""
    base = "nim/deepseek-ai/deepseek-r1"
    distills = [
        f"{base}-distill-llama-70b",
        f"{base}-distill-llama-8b",
        f"{base}-distill-qwen-14b",
        f"{base}-distill-qwen-32b",
        f"{base}-distill-qwen-7b",
    ]
    related = choose.split_variants(base, [base, f"{base}-0528", *distills])
    assert related.variants == [] and related.production == []
    assert related.other == [f"{base}-0528", *distills]
    gpt = "nim/openai/gpt-oss-120b"
    assert choose.split_variants(gpt, [f"{gpt}-turbo", f"{gpt}-dgx-spark"]) == choose.Related(
        [f"{gpt}-dgx-spark"], [], [f"{gpt}-turbo"]
    )


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        ("dgx-spark", True),
        ("RTX", True),
        ("rtx-pro-6000", True),
        ("gh200", True),
        ("jetson-thor", True),
        ("rtxpro", False),
        ("0528", False),
        ("distill-llama-70b", False),
        ("turbo", False),
        ("thoroughbred", False),
    ],
)
def test_is_hardware_suffix(suffix: str, expected: bool) -> None:
    assert choose.is_hardware_suffix(suffix) is expected


def test_plan_drops_repeats_and_stops_at_the_limit() -> None:
    items = [
        choose.Candidate(BASE, "2.1.4", "newest"),
        choose.Candidate(BASE, "2.1.4", "older"),
        choose.Candidate(f"{BASE}-dgx-spark", "1.0", "variant"),
        choose.Candidate(BASE, "2.0.13", "older"),
    ]
    kept, dropped = choose.plan(items, limit=2)
    assert [(c.resource_id, c.tag, c.kind) for c in kept] == [
        (BASE, "2.1.4", "newest"),
        (f"{BASE}-dgx-spark", "1.0", "variant"),
    ]
    assert [(c.tag, c.kind) for c in dropped] == [("2.0.13", "older")]
    assert kept[0].image == f"nvcr.io/{BASE}:2.1.4"
    assert choose.plan(items, limit=6) == (
        [items[0], items[2], items[3]],
        [],
    )


def test_plan_keeps_one_older_version_when_variants_fill_the_limit() -> None:
    items = [
        choose.Candidate(BASE, "2.1.4", "newest"),
        choose.Candidate(f"{BASE}-dgx-spark", "1.0", "variant"),
        choose.Candidate(f"{BASE}-rtx", "1.0", "variant"),
        choose.Candidate(f"{BASE}-gh200", "1.0", "variant"),
        choose.Candidate(BASE, "2.0.13", "older"),
        choose.Candidate(BASE, "1.15.5", "older"),
    ]
    kept, dropped = choose.plan(items, limit=3)
    assert [(c.resource_id, c.tag) for c in kept] == [
        (BASE, "2.1.4"),
        (f"{BASE}-dgx-spark", "1.0"),
        (BASE, "2.0.13"),
    ]
    assert [c.resource_id for c in dropped] == [f"{BASE}-rtx", f"{BASE}-gh200", BASE]
    assert dropped[-1].tag == "1.15.5"


def item(tag: str, kind: str, verdict: str | None, **extra: Any) -> dict[str, Any]:
    metadata = extra.pop("metadata", "published" if verdict else "not_published")
    return {"tag": tag, "kind": kind, "verdict": verdict, "metadata": metadata, **extra}


def test_pick_prefers_fits_then_unknown_then_the_first_evaluated() -> None:
    assert choose.pick([item("a", "newest", "unknown"), item("b", "older", "fits")]) == 1
    assert choose.pick([item("a", "newest", None), item("b", "older", "unknown")]) == 1
    assert choose.pick([item("a", "newest", None), item("b", "older", "does_not_fit")]) == 1
    assert choose.pick([item("a", "newest", None), item("b", "older", None)]) is None


def test_pick_prefers_an_unknown_built_for_or_tested_on_the_gpu() -> None:
    """One RTX 6000 Ada: the newest image's profiles are GPU-agnostic and untested there, and
    record no VRAM; the 1.8.0-RTX build's profile is built for it and records no VRAM either. Both
    are unknown, and the build for the GPU leaves fewer checks open."""
    untested = ["gpu_not_tested", "vram_not_recorded"]
    checked = [
        item("2.1.6", "newest", "unknown", tier="agnostic", reasons=untested),
        item("1.0.0-variant", "variant", "does_not_fit"),
        item("1.8.0-RTX", "variant", "unknown", tier="direct", reasons=["vram_not_recorded"]),
        item("2.0.13", "older", "unknown", tier="tested", reasons=[]),
    ]
    assert choose.pick(checked) == 2
    assert choose.reason(checked, 2) == (
        "hardware variant: no image is confirmed to fit, and its profile was built for this GPU "
        "and leaves fewer checks open than that of 2.1.6 (2.1.6: fit could not be confirmed "
        "(gpu_not_tested, vram_not_recorded); 1.0.0-variant: no profile runs here)"
    )
    # a note such as pci_inferred is no open check
    inferred = [checked[0], {**checked[2], "reasons": ["vram_not_recorded", "pci_inferred"]}]
    assert choose.pick(inferred) == 1
    # only the unknown images were passed over for their GPU match; one that does not fit was not
    rtx_too_small = [
        item("2.1.6", "newest", "unknown", tier="agnostic", reasons=untested),
        item("1.8.0-RTX", "variant", "does_not_fit", rejected_counts={"vram_too_small": 2}),
        item("2.0.13", "older", "unknown", tier="direct", reasons=["vram_not_recorded"]),
    ]
    assert choose.pick(rtx_too_small) == 2
    assert "fewer checks open than that of 2.1.6 (" in choose.reason(rtx_too_small, 2)
    tested_first = [item("2.1.6", "newest", "unknown", tier="tested"), *checked[1:]]
    assert choose.pick(tested_first) == 0
    assert choose.reason(tested_first, 0).endswith("this one comes first")
    agnostic_only = [
        item("2.1.6", "newest", "unknown", tier="agnostic", reasons=untested),
        item("2.0", "older", "unknown", reasons=untested),
    ]
    assert choose.pick(agnostic_only) == 0


def test_pick_keeps_the_first_unknown_when_the_gpu_build_leaves_other_checks_open() -> None:
    """llama-3.1-8b-instruct on one RTX 6000 Ada: 2.1.6's vLLM profile records the VRAM it needs
    and is unknown only for gpu_not_tested; 1.8.0-RTX, built for the GPU, records no VRAM. The
    older build confirms less, so the newest image stays (and so for an older image built for the
    GPU that records no CPU architecture)."""
    checked = [
        item("2.1.6", "newest", "unknown", tier="agnostic", reasons=["gpu_not_tested"]),
        item("1.8.0-RTX", "variant", "unknown", tier="direct", reasons=["vram_not_recorded"]),
        item(
            "1.8.6",
            "older",
            "unknown",
            tier="direct",
            reasons=["vram_not_recorded", "arch_not_recorded"],
        ),
    ]
    assert choose.pick(checked) == 0
    assert choose.reason(checked, 0) == (
        "newest version: no image is confirmed to fit and this one comes first"
    )
    nemotron = [
        item("1.10.1", "newest", "unknown", tier="agnostic", reasons=["gpu_not_tested", "x"]),
        item("1.8.6", "older", "unknown", tier="direct", reasons=["x", "arch_not_recorded"]),
    ]
    assert choose.pick(nemotron) == 0


def test_a_test_recorded_in_another_image_is_a_note_not_an_open_check() -> None:
    """tested_in_other_image (fit.TESTED_ELSEWHERE) says where the record of a `tested` profile
    comes from: like pci_inferred it leaves no check open, so an image whose profile counts as
    tested on this GPU by another image's record confirms as much as one tested by its own."""
    carried = ["tested_in_other_image", "vram_not_recorded"]
    assert choose.open_checks({"reasons": carried}) == ["vram_not_recorded"]
    checked = [
        item("2.1.6", "newest", "unknown", tier="agnostic", reasons=["gpu_not_tested", "x"]),
        item("2.1.5", "older", "unknown", tier="tested", reasons=["tested_in_other_image", "x"]),
    ]
    assert choose.pick(checked) == 1
    assert choose.reason(checked, 1) == (
        "older version: no image is confirmed to fit, and its profile was tested on this GPU and "
        "leaves fewer checks open than that of 2.1.6 (2.1.6: fit could not be confirmed "
        "(gpu_not_tested, x))"
    )
    newest = [item("2.1.6", "newest", "unknown", tier="tested", reasons=carried)]
    assert choose.reason(newest, 0) == (
        "newest version: no image is confirmed to fit and this one comes first"
    )
    assert choose.short_status(newest[0]) == "fit could not be confirmed (vram_not_recorded)"


def test_reason_sentences() -> None:
    newest_fits = [item("2.1.4", "newest", "fits")]
    assert choose.reason(newest_fits, 0) == "newest version that fits"
    variant = [
        item("2.1.4", "newest", "does_not_fit", rejected_counts={"gpu_mismatch": 3}),
        item("1.0", "variant", "fits"),
    ]
    assert choose.reason(variant, 1) == (
        "hardware variant for this GPU; the newest image has no profile that runs here "
        "(gpu_mismatch)"
    )
    # an unknown image is named with the checks its best profile leaves open
    untested = [item("2.1.4", "newest", "unknown", reasons=["gpu_not_tested"]), variant[1]]
    assert choose.reason(untested, 1) == (
        "hardware variant for this GPU; the newest image has a fit that could not be confirmed "
        "(gpu_not_tested)"
    )
    older = [
        item("2.1.4", "newest", "does_not_fit", rejected_counts={"not_enough_gpus": 4}),
        item("2.0.13", "older", None),
        item("1.15.5", "older", "fits"),
    ]
    # only the images checked are named: other newer builds were not checked (facts.notes)
    assert choose.reason(older, 2) == (
        "older version: the newer images checked are not confirmed to fit (2.1.4: no profile runs "
        "here (not_enough_gpus); 2.0.13: no published profiles)"
    )
    # a newer image that is unknown is not said to not fit
    untested_newest = [item("2.1.6", "newest", "unknown", reasons=["gpu_not_tested"]), older[2]]
    assert choose.reason(untested_newest, 1) == (
        "older version: the newer images checked are not confirmed to fit (2.1.6: fit could not "
        "be confirmed (gpu_not_tested))"
    )
    unknown = [item("2.1.4", "newest", None, metadata="error"), item("2.0.13", "older", "unknown")]
    assert choose.reason(unknown, 1).startswith("older version: no image is confirmed to fit")
    assert "metadata could not be read" in choose.reason(unknown, 1)
    none_fit = [item("2.1.4", "newest", "does_not_fit")]
    assert choose.reason(none_fit, 0).startswith("no image fits here")
    asked = [item("2.0.13", "requested", "fits")]
    assert choose.reason(asked, 0) == "the tag you asked for; it fits"
    # a moving alias the user named was replaced by its build: the reason says so
    alias = [item("1.5.2", "requested", "unknown")]
    assert choose.reason(alias, 0, asked="1") == (
        "the build `1` stands for; its fit could not be confirmed"
    )


def test_reason_names_the_repository_of_images_from_another_repo() -> None:
    spark = f"{BASE}-dgx-spark"
    checked = [
        item("2.1.4", "newest", "does_not_fit", resource_id=BASE, rejected_counts={"x": 1}),
        item("1.1.0", "variant", None, resource_id=spark),
        item("2.0.13", "older", "fits", resource_id=BASE),
    ]
    text = choose.reason(checked, 2, BASE)
    assert "2.1.4: no profile runs here (x)" in text
    assert "llama-3.1-8b-instruct-dgx-spark:1.1.0: no published profiles" in text
    assert choose.label(checked[1]) == "1.1.0"


def test_an_image_without_the_profile_asked_for_is_passed_over_and_named() -> None:
    """select --profile: an image whose metadata lacks that profile gets no verdict."""
    lacking = item("2.1.6", "newest", None, metadata="published", error_code="profile_not_found")
    variant = [lacking, item("1.0", "variant", "fits")]
    assert choose.pick(variant) == 1
    assert choose.reason(variant, 1) == (
        "hardware variant for this GPU; the newest image has no profile with the id asked for "
        "(--profile)"
    )
    older = [lacking, item("2.1.5", "older", "unknown", reasons=["gpu_not_tested"])]
    assert choose.pick(older) == 1
    assert choose.reason(older, 1) == (
        "older version: no image is confirmed to fit and this one comes first (2.1.6: no profile "
        "with the id asked for (--profile))"
    )
    assert choose.pick([lacking]) is None
