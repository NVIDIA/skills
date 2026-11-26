# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/ngc_api.py: model references, tag choice and the NGC HTTP client."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import ngc_api
import ns_common as ns
import pytest
from ns_testkit import SCRIPTS, fixture_json, run_cli, serve, set_ngc_base

REPO_ID = "nim/meta/llama-3.1-8b-instruct"
META_PATH = "/v2/nim/metadata"
MIRROR_HOSTS = ["registry.example.com", "mirror.example.org:5000", "localhost:5000"]


@pytest.fixture(autouse=True)
def _no_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIM_SELECT_RETRY_DELAY", "0")


def _envelope(code: str, rid: str) -> dict[str, Any]:
    return {"requestStatus": {"statusCode": code, "requestId": rid}}


# --- model references --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "resource_id", "tag", "query", "mapping"),
    [
        ("nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.4", REPO_ID, "2.1.4", None, "exact"),
        ("nvcr.io/nim/meta/llama-3.1-8b-instruct", REPO_ID, None, None, "exact"),
        ("nvcr.io/nim/openfold3:1.2.0", "nim/openfold3", "1.2.0", None, "exact"),
        ("NVCR.IO/nim/Meta/Llama-3.1-8B-Instruct:1.8.0-RTX", REPO_ID, "1.8.0-RTX", None, "exact"),
        ("nim/meta/llama-3.1-8b-instruct", REPO_ID, None, None, "exact"),
        ("meta/llama-3.1-8b-instruct", REPO_ID, None, None, "exact"),
        ("meta/llama-3.1-8b-instruct:latest", REPO_ID, "latest", None, "exact"),
        ("llama 3.1 8b", None, None, "llama 3.1 8b", "search"),
        ("llama-3.1-8b-instruct", None, None, "llama-3.1-8b-instruct", "search"),
        ("llama-3.1-8b-instruct:2.1.4", None, "2.1.4", "llama-3.1-8b-instruct", "search"),
    ],
)
def test_parse_model_ref(
    text: str, resource_id: str | None, tag: str | None, query: str | None, mapping: str
) -> None:
    ref = ngc_api.parse_model_ref(text)
    assert (ref.resource_id, ref.tag, ref.query, ref.mapping) == (resource_id, tag, query, mapping)
    assert ref.source_registry is None


@pytest.mark.parametrize("host", MIRROR_HOSTS)
@pytest.mark.parametrize(
    "path",
    [
        "nim/meta/llama-3.1-8b-instruct",
        "public/nim/meta/llama-3.1-8b-instruct",
        "library/nim/Meta/Llama-3.1-8B-Instruct",
        "meta/llama-3.1-8b-instruct",
    ],
)
def test_mirror_refs_map_by_name(host: str, path: str) -> None:
    ref = ngc_api.parse_model_ref(f"{host}/{path}:2.1.4")
    assert ref.resource_id == REPO_ID
    assert ref.tag == "2.1.4"
    assert ref.mapping == "by_name"
    assert ref.source_registry == host
    assert any(host in w and "same name and tag" in w for w in ref.warnings)
    assert any("handed off, as the nvcr.io image" in w for w in ref.warnings)


def test_mirror_ref_without_a_tag_says_the_tag_is_chosen_on_ngc() -> None:
    ref = ngc_api.parse_model_ref("registry.example.com/nim/meta/llama-3.1-8b-instruct")
    assert (ref.resource_id, ref.tag) == (REPO_ID, None)
    assert any("the tag is chosen on NGC" in w for w in ref.warnings)
    assert not any("same name and tag" in w for w in ref.warnings)


def test_mirror_single_leaf_searches() -> None:
    ref = ngc_api.parse_model_ref("registry.example.com/nim/llama-3.1-8b-instruct:2.1.4")
    assert (ref.resource_id, ref.query, ref.mapping) == (None, "llama-3.1-8b-instruct", "search")
    assert ref.source_registry == "registry.example.com" and ref.warnings


def test_other_registry_is_by_name() -> None:
    ref = ngc_api.parse_model_ref("registry.example.com:5000/nim/meta/llama-3.1-8b-instruct:1.15")
    assert (ref.resource_id, ref.tag, ref.mapping) == (REPO_ID, "1.15", "by_name")
    assert ref.source_registry == "registry.example.com:5000"


def test_digest_is_dropped_with_warning() -> None:
    ref = ngc_api.parse_model_ref("nvcr.io/nim/meta/llama-3.1-8b-instruct@sha256:abc")
    assert (ref.resource_id, ref.tag) == (REPO_ID, None)
    assert any("sha256:abc" in w for w in ref.warnings)


def test_empty_ref_is_bad_input() -> None:
    with pytest.raises(ns.ScriptError) as info:
        ngc_api.parse_model_ref("  ")
    assert (info.value.code, info.value.exit_code) == ("bad_model_ref", ns.EXIT_INPUT)


# --- tag choice ----------------------------------------------------------------------------------


def test_choose_newest_on_real_repo() -> None:
    repo = fixture_json("ngc", "repo-llama-3.1-8b-instruct.json")
    assert repo["latestTag"] == "latest"
    assert ngc_api.choose_tag(repo, None) == ("2.1.4", "newest_version", [])


def test_latest_is_replaced() -> None:
    repo = fixture_json("ngc", "repo-llama-3.1-8b-instruct.json")
    tag, source, notes = ngc_api.choose_tag(repo, "latest")
    assert (tag, source) == ("2.1.4", "latest_replaced")
    assert notes and "2.1.4" in notes[0]


@pytest.mark.parametrize("tag", ["1.8.0-RTX", "2.0.13", "1.15.5"])
def test_requested_tag_kept(tag: str) -> None:
    repo = fixture_json("ngc", "repo-llama-3.1-8b-instruct.json")
    assert ngc_api.choose_tag(repo, tag) == (tag, "requested", [])


QWEN35_TAGS = ["latest", "1.7.0-variant", "1.7", "1"]  # nim/qwen/qwen3.5-35b-a3b, one image
# nim/deepseek-ai/deepseek-r1-distill-llama-8b and nim/nv-mistralai/mistral-nemo-12b-instruct on
# 2026-11-04, image lists not public. The Metadata API gives deepseek's `1` the same 26 profiles as
# 1.5.2; 1.8.0-RTX has 5 others, for RTX GPUs only.
DEEPSEEK_8B_TAGS = ["1.8.0-RTX", "1.5.2", "1", "latest", "1.5"]
MISTRAL_NEMO_TAGS = ["1.8.0-rtx", "1.2", "1.2.2", "1", "latest"]


@pytest.mark.parametrize(
    ("tags", "requested", "expected", "how"),
    [
        (QWEN35_TAGS, "1.7", "1.7.0-variant", "judged by its version numbers"),
        (QWEN35_TAGS, "1", "1.7.0-variant", "judged by its version numbers"),
        (None, "1.15", "1.15.5", "judged by its version numbers"),
        (DEEPSEEK_8B_TAGS, "1", "1.5.2", "judged by its version numbers"),
        (MISTRAL_NEMO_TAGS, "1", "1.2.2", "judged by its version numbers"),
    ],
)
def test_a_requested_moving_alias_is_replaced_by_its_build(
    tags: list[str] | None, requested: str, expected: str, how: str
) -> None:
    """`1.7` can move to another build, and a profile id belongs to one build (the hard rule to
    pin a version tag). By its version numbers, `1` is never a build for named hardware
    (1.8.0-RTX) that is newer."""
    repo = {"tags": tags} if tags else fixture_json("ngc", "repo-llama-3.1-8b-instruct.json")
    tag, source, notes = ngc_api.choose_tag(repo, requested)
    assert (tag, source) == (expected, "alias_replaced")
    assert f"`{requested}` is a moving tag" in notes[0] and how in notes[0]


def test_a_requested_alias_follows_its_digest() -> None:
    """With the image list, the digest says which build an alias is: here `1.8` is still 1.8.1."""
    tags = ["1.8.3-variant", "1.8", "1.8.1"]
    digests = {"1.8.3-variant": "sha256:new", "1.8": "sha256:old", "1.8.1": "sha256:old"}
    tag, source, notes = ngc_api.choose_tag({"tags": tags}, "1.8", digests)
    assert (tag, source) == ("1.8.1", "alias_replaced")
    assert "the same image digest" in notes[0]
    # an alias with no build listed stays as named
    assert ngc_api.choose_tag({"tags": ["1.1", "1.2.0"]}, "1.1") == ("1.1", "requested", [])


def test_missing_tag_lists_tags() -> None:
    repo = fixture_json("ngc", "repo-llama-3.1-8b-instruct.json")
    with pytest.raises(ns.ScriptError) as info:
        ngc_api.choose_tag(repo, "9.9.9")
    err = info.value
    assert (err.code, err.exit_code) == ("tag_not_found", ns.EXIT_PRECONDITION)
    assert err.facts["tags"][:3] == ["2.1.4", "2.0.13", "2.0.12"]
    assert len(err.facts["tags"]) == 10
    assert "2.1.4" in err.message


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (["1.2.0-rc1", "1.2.0", "1.1.9"], "1.2.0"),
        (["1.1.9", "1.2.0-rc1", "1.2.0-rc2"], "1.1.9"),
        (["1.2.0-rc1", "1.2.0-rc2"], "1.2.0-rc2"),
        (["1.10.0", "1.9.9", "latest"], "1.10.0"),
        (["1.15", "1.15.5", "1"], "1.15.5"),
    ],
)
def test_version_ordering(tags: list[str], expected: str) -> None:
    assert ngc_api.choose_tag({"tags": tags}, None)[0] == expected


NEMOTRON_EXP_TAGS = [
    "1.8.3-variant",
    "1.8.2-variant",
    "1.8",
    "1",
    "1.8.1-variant",
    "1.8.1",
    "latest",
]
# GET /v2/repos/nim/nvidia/nemotron-3-super-120b-a12b-experimental/images on 2026-11-04
NEMOTRON_EXP_DIGESTS = {
    "latest": "sha256:e8355971fc7e",
    "1": "sha256:e8355971fc7e",
    "1.8": "sha256:e8355971fc7e",
    "1.8.3-variant": "sha256:e8355971fc7e",
    "1.8.2-variant": "sha256:90efe5fdb2ad",
    "1.8.1-variant": "sha256:a7e96e45d2ad",
    "1.8.1": "sha256:a7e96e45d2ad",
}


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (["latest", "1.7.0-variant", "1.7", "1"], "1.7.0-variant"),
        (["1.5.0-variant", "1", "1.5-variant", "latest"], "1.5.0-variant"),
        (["1.7", "1.6.0-variant"], "1.7"),
        (["2", "1.9.0-variant"], "2"),
        (NEMOTRON_EXP_TAGS, "1.8.3-variant"),
        (DEEPSEEK_8B_TAGS, "1.5.2"),
        (["2.1.4", "2.1", "2.1.5-rtx"], "2.1.4"),
        (["1.8.0-RTX", "1.7.0-variant", "latest"], "1.7.0-variant"),
        (["2.0.0-rc1", "1.9.0-variant", "latest"], "1.9.0-variant"),
        (["1", "1.8.0-RTX", "1.5.0-variant", "latest"], "1.5.0-variant"),
        (["1.0.0-RTX", "latest"], "1.0.0-RTX"),
    ],
    ids=[
        "qwen3.5-35b-a3b",
        "qwen3-coder-next",
        "other-minor",
        "other-major",
        "nemotron-3-super-experimental",
        "deepseek-r1-distill-llama-8b",
        "a-build-for-one-gpu-is-never-the-newest",
        "no-plain-tag-and-a-build-for-one-gpu",
        "no-plain-tag-and-a-prerelease",
        "a-bare-alias-and-a-build-for-one-gpu",
        "only-builds-for-one-gpu",
    ],
)
def test_a_moving_alias_gives_way_to_the_full_tag(tags: list[str], expected: str) -> None:
    """`1.7` and `1` can move to another build; the full tag they start names one build. A
    major.minor alias that stands for a newer build than the plain tag of its line (`1.8` for
    1.8.3-variant, not 1.8.1) wins; `1` alone never does (it can be an RTX build). The ones with
    a model id are real NGC tag sets."""
    assert ngc_api.newest_version(tags) == expected
    assert ngc_api.choose_tag({"tags": tags}, "latest")[0] == expected


def test_the_image_digests_say_what_an_alias_stands_for() -> None:
    assert ngc_api.newest_version(NEMOTRON_EXP_TAGS, NEMOTRON_EXP_DIGESTS) == "1.8.3-variant"
    # if `1.8` were still the 1.8.1 image, the plain 1.8.1 would stay the newest
    lagging = {**NEMOTRON_EXP_DIGESTS, "1.8": "sha256:a7e96e45d2ad"}
    assert ngc_api.newest_version(NEMOTRON_EXP_TAGS, lagging) == "1.8.1"
    # nim/nvidia/llama-nemotron-rerank-500m-v2: `1.1` is the same image as 1.10.0
    rerank = ["1", "1.1", "latest", "1.10.0"]
    same = dict.fromkeys(rerank, "sha256:3e39d44bdb3d")
    assert ngc_api.alias_target("1.1", rerank, same) == "1.10.0"
    assert ngc_api.alias_target("1.1", rerank) is None
    assert ngc_api.alias_target("1.5", ["1.5.2", "1.5.3-rc1"]) == "1.5.2"
    # `1.8` has a digest no listed build has: it stands for none of them, so the plain 1.8.1
    # stays the newest and a requested `1.8` is checked as named (it cannot be pinned)
    tags = ["1.8", "1.8.1", "1.8.3-variant"]
    moved = {"1.8": "sha256:x", "1.8.1": "sha256:a", "1.8.3-variant": "sha256:b"}
    assert ngc_api.alias_target("1.8", tags, moved) is None
    assert ngc_api.newest_version(tags, moved) == "1.8.1"
    assert ngc_api.choose_tag({"tags": tags}, "1.8", moved) == ("1.8", "requested", [])
    # a build whose digest is not listed can still be it, judged by its version numbers
    partial = {"1.8": "sha256:x", "1.8.1": "sha256:a"}
    assert ngc_api.alias_target("1.8", tags, partial) == "1.8.3-variant"
    # an alias with the hardware suffix of a build stands for it
    assert ngc_api.alias_target("1.8-RTX", ["1.8.0-RTX", "1.5.2"]) == "1.8.0-RTX"


def test_an_alias_with_the_digest_of_a_prerelease_is_not_chosen_on_its_own() -> None:
    """`2` with the digest of 1.8.2-rc1 is that prerelease, a test build: with no tag asked for,
    the stable 1.8.1 is chosen. Asked for by its exact tag, the prerelease is still used."""
    tags = ["2", "1.8.1", "1.8.2-rc1"]
    digests = {"2": "sha256:rc", "1.8.2-rc1": "sha256:rc", "1.8.1": "sha256:stable"}
    repo = {"tags": tags}
    assert ngc_api.newest_version(tags, digests) == "1.8.1"
    assert ngc_api.choose_tag(repo, None, digests) == ("1.8.1", "newest_version", [])
    assert ngc_api.choose_tag(repo, "1.8.2-rc1", digests) == ("1.8.2-rc1", "requested", [])
    # a build for any GPU wins over it too; with no other tag, the prerelease is still chosen
    other = {"2": "sha256:rc", "2.0.0-rc1": "sha256:rc"}
    assert ngc_api.newest_version(["2", "1.9.0-variant", "2.0.0-rc1"], other) == "1.9.0-variant"
    assert ngc_api.newest_version(["2", "2.0.0-rc1"], other) == "2.0.0-rc1"


# nim/meta/llama-3.1-8b-instruct-pb6 on 2026-11-04 (latestTag 2.0.4-pb6.10)
PB6_TAGS = [
    "2.0.4-pb6.10", "2.0.4-pb6.9", "2.0.4-pb6.8", "2.0.4-pb6.8.1", "latest", "2.0.4-pb6.7",
    "2.0.4-pb6.6", "2.0.4-pb6.5", "2.0.4-pb6.4", "2.0.4-pb6.3", "2.0.4-pb6.2", "2.0.4-pb6.1",
    "2.0.4-pb6.0",
]  # fmt: skip


def test_suffix_numbers_compare_as_numbers() -> None:
    """2.0.4-pb6.10 is newer than 2.0.4-pb6.9; a string compare put it last."""
    ordered = ngc_api.version_tags(PB6_TAGS)
    assert ordered[:4] == ["2.0.4-pb6.10", "2.0.4-pb6.9", "2.0.4-pb6.8.1", "2.0.4-pb6.8"]
    assert ordered[-1] == "2.0.4-pb6.0"
    assert ngc_api.choose_tag({"tags": PB6_TAGS}, None)[0] == "2.0.4-pb6.10"
    # nim/moonshotai/kimi-k2.6-turbo: 2.0.9-variant.1 is a rebuild of 2.0.9-variant
    rebuilt = ["2.0.9-variant", "2.0.9-variant.1", "1.0.0-rc10", "1.0.0-rc9"]
    expected = ["2.0.9-variant.1", "2.0.9-variant", "1.0.0-rc10", "1.0.0-rc9"]
    assert ngc_api.version_tags(rebuilt) == expected
    with pytest.raises(ns.ScriptError) as info:
        ngc_api.choose_tag({"tags": PB6_TAGS}, "2.0.4-pb6.11")
    assert info.value.facts["tags"][0] == "2.0.4-pb6.10"


@pytest.mark.parametrize(
    ("tag", "expected"),
    [("1.2.0-rc1", True), ("1.0.0-RC", True), ("1.0.0-beta.2", True), ("1.7.0-variant", False),
     ("1.8.0-RTX", False), ("2.1.4", False), ("1.0.0-device", False), ("latest", False)],
)  # fmt: skip
def test_is_prerelease(tag: str, expected: bool) -> None:
    assert ngc_api.is_prerelease(tag) is expected


@pytest.mark.parametrize(
    ("alias", "tag", "expected"),
    [
        ("1.7", "1.7.0-variant", True),
        ("1", "1.7.0-variant", True),
        ("1.7-variant", "1.7.0-variant", True),
        ("1.7.0", "1.7.0-variant", False),
        ("1.7-rtx", "1.7.0-variant", False),
        ("1.6", "1.7.0-variant", False),
        ("1.7.0", "1.7.0.1", False),
        ("latest", "1.7.0", False),
    ],
)
def test_is_alias_of(alias: str, tag: str, expected: bool) -> None:
    """Only `X` or `X.Y`, with no suffix or the same one; `1.7.0` is another build."""
    assert ngc_api.is_alias_of(alias, tag) is expected


def test_prerelease_sorts_below_release() -> None:
    assert ngc_api.version_tags(["1.2.0-rc1", "latest", "1.2.0", "1.1.0"]) == [
        "1.2.0",
        "1.2.0-rc1",
        "1.1.0",
    ]


def test_fallback_to_latest_tag_field() -> None:
    tag, source, notes = ngc_api.choose_tag({"tags": ["latest"], "latestTag": "latest"}, None)
    assert (tag, source) == ("latest", "latest_tag_field") and notes


def test_no_tags_at_all() -> None:
    with pytest.raises(ns.ScriptError) as info:
        ngc_api.choose_tag({"name": "x"}, None)
    assert info.value.code == "tag_not_found"


# --- HTTP client ---------------------------------------------------------------------------------


def _meta_query(fake_requests: list[str]) -> dict[str, list[str]]:
    assert fake_requests, "no request reached the server"
    path, _, query = fake_requests[-1].partition("?")
    assert path == META_PATH
    return urllib.parse.parse_qs(query)


def test_metadata_200(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = fixture_json("ngc", "metadata-llama-3.1-8b-instruct-2.1.4.json")
    with serve({META_PATH: (200, doc)}) as fake:
        set_ngc_base(monkeypatch, fake.base)
        assert ngc_api.get_metadata(REPO_ID, "2.1.4") == doc
    assert _meta_query(fake.requests) == {"resourceId": [REPO_ID], "tag": ["2.1.4"]}
    assert "resourceId=nim/meta/llama-3.1-8b-instruct" in fake.requests[0]


def test_metadata_404_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    with serve({META_PATH: (404, fixture_json("ngc", "metadata-not-found.json"))}) as fake:
        set_ngc_base(monkeypatch, fake.base)
        assert ngc_api.get_metadata(REPO_ID, "1.0.0") is None
    assert len(fake.requests) == 1


@pytest.mark.parametrize(("status", "calls"), [(400, 1), (500, 1)])
def test_metadata_service_errors(monkeypatch: pytest.MonkeyPatch, status: int, calls: int) -> None:
    with serve({META_PATH: (status, _envelope("INTERNAL_ERROR", "rid-42"))}) as fake:
        set_ngc_base(monkeypatch, fake.base)
        with pytest.raises(ns.ScriptError) as info:
            ngc_api.get_metadata(REPO_ID, "2.1.4")
    err = info.value
    assert (err.code, err.exit_code) == ("metadata_service_error", ns.EXIT_EXTERNAL)
    assert "rid-42" in err.message and str(status) in err.message
    assert err.fix == "Report it with the requestId; do not retry with other parameters."
    assert len(fake.requests) == calls


def test_metadata_retries_503_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = fixture_json("ngc", "metadata-llama-3.1-8b-instruct-2.1.4.json")
    answers: Any = [(503, _envelope("INTERNAL_ERROR", "r1")), (503, {}), (200, doc)]
    with serve({META_PATH: answers}) as fake:
        set_ngc_base(monkeypatch, fake.base)
        assert ngc_api.get_metadata(REPO_ID, "2.1.4") == doc
    assert len(fake.requests) == 3


def test_metadata_503_exhausted_is_service_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with serve({META_PATH: (503, _envelope("INTERNAL_ERROR", "r9"))}) as fake:
        set_ngc_base(monkeypatch, fake.base)
        with pytest.raises(ns.ScriptError) as info:
            ngc_api.get_metadata(REPO_ID, "2.1.4")
    assert info.value.code == "metadata_service_error"
    assert len(fake.requests) == 3


def test_a_success_that_is_not_json_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    page = "<!DOCTYPE html>\n<html><body>Proxy sign-in</body></html>"
    with serve({"/v2/a": (200, page), "/v2/empty": (200, "")}) as fake:
        set_ngc_base(monkeypatch, fake.base)
        with pytest.raises(ns.ScriptError) as info:
            ngc_api.get_json("/a")
        with pytest.raises(ns.ScriptError) as empty:
            ngc_api.get_json("/empty")
    err = info.value
    assert (err.code, err.exit_code) == ("ngc_bad_response", ns.EXIT_EXTERNAL)
    assert "not JSON" in err.message and "<!DOCTYPE html> <html>" in err.message
    assert "HTTPS_PROXY" in err.fix
    assert empty.value.code == "ngc_bad_response"
    assert fake.requests == ["/v2/a", "/v2/empty"]  # never retried


@pytest.mark.parametrize("body", [{}, {"profiles": None}, {"profiles": "none"}, [1, 2]])
def test_metadata_200_needs_a_profiles_list(monkeypatch: pytest.MonkeyPatch, body: Any) -> None:
    with serve({META_PATH: (200, body)}) as fake:
        set_ngc_base(monkeypatch, fake.base)
        with pytest.raises(ns.ScriptError) as info:
            ngc_api.get_metadata(REPO_ID, "2.1.4")
    assert (info.value.code, info.value.exit_code) == ("ngc_bad_response", ns.EXIT_EXTERNAL)
    assert "without a profiles list" in info.value.message


def test_metadata_with_no_profiles_is_still_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    with serve({META_PATH: (200, {"profiles": []})}) as fake:
        set_ngc_base(monkeypatch, fake.base)
        assert ngc_api.get_metadata(REPO_ID, "2.1.4") == {"profiles": []}


def test_get_json_retries_429_not_404(monkeypatch: pytest.MonkeyPatch) -> None:
    routes: dict[str, Any] = {
        "/v2/a": [(429, {}), (200, {"x": 1})],
        "/v2/b": (404, _envelope("NOT_FOUND", "n")),
    }
    with serve(routes) as fake:
        set_ngc_base(monkeypatch, fake.base)
        assert ngc_api.get_json("/a") == (200, {"x": 1})
        assert ngc_api.get_json("/b")[0] == 404
    assert fake.requests == ["/v2/a", "/v2/a", "/v2/b"]


def test_unreachable_raises_after_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    # the port stays bound, so no other program can take it, and nothing listens: each attempt
    # is refused
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        set_ngc_base(monkeypatch, f"http://127.0.0.1:{sock.getsockname()[1]}/v2")
        with pytest.raises(ns.ScriptError) as info:
            ngc_api.get_json("/repos/x", timeout=2)
    assert (info.value.code, info.value.exit_code) == ("ngc_unreachable", ns.EXIT_EXTERNAL)
    assert "3 attempts" in info.value.message


@pytest.mark.parametrize("stalls", [False, True], ids=["cut-short", "timed-out"])
def test_a_503_whose_body_cannot_be_read_is_still_retried(
    monkeypatch: pytest.MonkeyPatch, stalls: bool
) -> None:
    """NGC sends the status line and headers of a 503, then its body is cut short or stops coming
    (the read times out). The status still counts: the request is retried, and the next answer is
    used."""
    release = threading.Event()
    paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            paths.append(self.path)
            if len(paths) > 1:
                data = b'{"x": 1}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_response(503)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b'{"requestStatus"')  # 16 of the 100 bytes
            if stalls:
                release.wait(10)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        set_ngc_base(monkeypatch, f"http://127.0.0.1:{server.server_address[1]}/v2")
        assert ngc_api.get_json("/a", timeout=1) == (200, {"x": 1})
    finally:
        release.set()
        server.shutdown()
        server.server_close()
    assert paths == ["/v2/a", "/v2/a"]


def test_get_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = fixture_json("ngc", "repo-llama-3.1-8b-instruct.json")
    with serve({f"/v2/repos/{REPO_ID}": (200, repo)}) as fake:
        set_ngc_base(monkeypatch, fake.base)
        assert ngc_api.get_repo(REPO_ID) == repo
        assert ngc_api.get_repo("nim/meta/does-not-exist") is None


def test_get_repo_unexpected_status(monkeypatch: pytest.MonkeyPatch) -> None:
    with serve({f"/v2/repos/{REPO_ID}": (403, _envelope("FORBIDDEN", "f1"))}) as fake:
        set_ngc_base(monkeypatch, fake.base)
        with pytest.raises(ns.ScriptError) as info:
            ngc_api.get_repo(REPO_ID)
    assert info.value.code == "ngc_http_error" and "f1" in info.value.message


def test_a_429_after_the_retries_is_rate_limiting(monkeypatch: pytest.MonkeyPatch) -> None:
    with serve({f"/v2/repos/{REPO_ID}": (429, _envelope("TOO_MANY_REQUESTS", "r1"))}) as fake:
        set_ngc_base(monkeypatch, fake.base)
        with pytest.raises(ns.ScriptError) as info:
            ngc_api.get_repo(REPO_ID)
        requested = len(fake.requests)
    assert (info.value.code, info.value.exit_code) == ("ngc_rate_limited", ns.EXIT_EXTERNAL)
    assert "HTTP 429" in info.value.message and "r1" in info.value.message
    assert requested == 3  # the attempts of the request, then no more


def test_only_ngcs_own_403_says_a_repository_is_not_public(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = {
        f"/v2/repos/{REPO_ID}": (403, _envelope("FORBIDDEN", "f1")),
        "/v2/repos/nim/acme/model": (403, "<html><body>Access denied</body></html>"),
    }
    with serve(routes) as fake:
        set_ngc_base(monkeypatch, fake.base)
        for rid, said in ((REPO_ID, True), ("nim/acme/model", None)):
            with pytest.raises(ns.ScriptError) as info:
                ngc_api.get_repo(rid)
            assert (info.value.code, info.value.facts.get("not_public")) == ("ngc_http_error", said)


def test_a_repository_record_needs_a_name_or_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    with serve({f"/v2/repos/{REPO_ID}": (200, {"description": "a page, no record"})}) as fake:
        set_ngc_base(monkeypatch, fake.base)
        with pytest.raises(ns.ScriptError) as info:
            ngc_api.get_repo(REPO_ID)
    assert (info.value.code, info.value.exit_code) == ("ngc_bad_response", ns.EXIT_EXTERNAL)


def test_a_repository_record_with_tags_that_are_not_a_list_is_a_bad_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A number cannot be read as tags, and a string would be read one character at a time
    (`1.2.3` gave the tag `3`). A record with no tags, or null ones, still uses its latestTag."""
    routes = {
        "/v2/repos/nim/acme/number": (200, {"name": "number", "tags": 7}),
        "/v2/repos/nim/acme/text": (200, {"name": "text", "tags": "1.2.3"}),
        "/v2/repos/nim/acme/none": (200, {"name": "none", "latestTag": "1.0.0"}),
        "/v2/repos/nim/acme/null": (200, {"name": "null", "tags": None, "latestTag": "1.0.0"}),
    }
    with serve(routes) as fake:
        set_ngc_base(monkeypatch, fake.base)
        for rid in ("nim/acme/number", "nim/acme/text"):
            with pytest.raises(ns.ScriptError) as info:
                ngc_api.get_repo(rid)
            assert (info.value.code, info.value.exit_code) == ("ngc_bad_response", ns.EXIT_EXTERNAL)
        for rid in ("nim/acme/none", "nim/acme/null"):
            repo = ngc_api.get_repo(rid)
            assert repo is not None
            assert ngc_api.choose_tag(repo, None)[:2] == ("1.0.0", "latest_tag_field")


def test_get_images(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "images": [
            {
                "tag": "2.1.4",
                "digest": "sha256:aaa",
                "size": 13819791260,
                "architectureVariants": [{"architecture": "arm64"}, {"architecture": "amd64"}],
            },
            {
                "tag": "1.3.3",
                "digest": "sha256:bbb",
                "architectureVariants": [
                    {"os": "linux", "architecture": "amd64"},
                    {"os": "unknown", "architecture": "unknown"},
                ],
            },
            {"tag": "odd", "architectureVariants": []},
            {"architectureVariants": [{"architecture": "amd64"}]},
        ]
    }
    routes = {
        f"/v2/repos/{REPO_ID}/images": (200, body),
        "/v2/repos/nim/meta/locked/images": (403, _envelope("FORBIDDEN", "f2")),
    }
    with serve(routes) as fake:
        set_ngc_base(monkeypatch, fake.base)
        images = ngc_api.get_images(REPO_ID)
        assert ngc_api.get_images("nim/meta/does-not-exist") is None
        with pytest.raises(ns.ScriptError) as info:
            ngc_api.get_images("nim/meta/locked")
    assert info.value.code == "ngc_http_error" and "f2" in info.value.message
    assert images == {
        "2.1.4": ngc_api.Image("sha256:aaa", ("amd64", "arm64"), 13819791260),
        "1.3.3": ngc_api.Image("sha256:bbb", ("amd64",), None),
        "odd": ngc_api.Image(None, (), None),
    }
    assert ngc_api.digests_of(images) == {"2.1.4": "sha256:aaa", "1.3.3": "sha256:bbb"}


@pytest.mark.parametrize("body", [{}, {"images": None}, {"images": "none"}, [1, 2]])
def test_images_200_needs_an_images_list(monkeypatch: pytest.MonkeyPatch, body: Any) -> None:
    """A 200 that is not an image list is a bad answer, never a repository with no images."""
    with serve({f"/v2/repos/{REPO_ID}/images": (200, body)}) as fake:
        set_ngc_base(monkeypatch, fake.base)
        with pytest.raises(ns.ScriptError) as info:
            ngc_api.get_images(REPO_ID)
    assert (info.value.code, info.value.exit_code) == ("ngc_bad_response", ns.EXIT_EXTERNAL)
    with serve({f"/v2/repos/{REPO_ID}/images": (200, {"images": []})}) as fake:
        set_ngc_base(monkeypatch, fake.base)
        assert ngc_api.get_images(REPO_ID) == {}


def test_attestation_entries_are_no_cpu_architecture() -> None:
    """latestArchitectureVariants of nim/nvidia/llama-nemotron-rerank-500m-v2 on 2026-11-04."""
    variants = [
        {"os": "linux", "architecture": "amd64"},
        {"os": "linux", "architecture": "arm64"},
        {"os": "unknown", "architecture": "unknown"},
        {"os": "unknown", "architecture": "unknown"},
    ]
    assert ngc_api.real_architectures(variants) == ["amd64", "arm64"]
    assert ngc_api.real_architectures(None) == []


# --- catalog search ------------------------------------------------------------------------------


def test_parse_search_fixture() -> None:
    found = ngc_api.parse_search(fixture_json("ngc", "search-llama.json"))
    ids = [c["resource_id"] for c in found]
    assert len(ids) == len(set(ids)) == 8
    assert ids[0] == "nim/defog/llama-3-sqlcoder-8b"
    by_id = {c["resource_id"]: c for c in found}
    llama = by_id[REPO_ID]
    assert llama == {
        "resource_id": REPO_ID,
        "name": "llama-3.1-8b-instruct",
        "display_name": "Llama-3.1-8B-Instruct",
        "latest_tag": "latest",
        "nvaie_supported": True,
        "experimental": False,
    }
    assert by_id["nim/meta/llama-3.1-405b-instruct"]["latest_tag"] == "1.3.0"
    assert by_id["nim/defog/llama-3-sqlcoder-8b"]["experimental"] is True
    assert by_id["nim/defog/llama-3-sqlcoder-8b"]["nvaie_supported"] is False


def test_search_request(monkeypatch: pytest.MonkeyPatch) -> None:
    body = fixture_json("ngc", "search-llama.json")
    with serve({"/v2/search/catalog/resources/CONTAINER": (200, body)}) as fake:
        set_ngc_base(monkeypatch, fake.base)
        found = ngc_api.search_nims("llama 3.1 8b")
    assert len(found) == 8
    query = urllib.parse.parse_qs(fake.requests[0].partition("?")[2])
    sent = json.loads(query["q"][0])
    assert sent["query"] == "llama 3.1 8b"
    assert sent["filters"] == [{"field": "nvidia_nim", "value": "nimmcro_nvidia_nim"}]
    assert sent["pageSize"] == 25 and sent["page"] == 0


CATALOG_ROW = {"resourceId": REPO_ID, "orgName": "nim", "name": "llama-3.1-8b-instruct"}


@pytest.mark.parametrize(
    ("rows", "unusable"),
    [
        ([CATALOG_ROW, {"name": "no-id"}, {"resourceId": 7, "name": "id-not-text"}], 2),
        ([CATALOG_ROW, CATALOG_ROW], 0),
    ],
    ids=["rows-without-a-resource-id", "the-same-resource-twice"],
)
def test_a_catalog_row_without_a_resource_id_leaves_the_catalog_partial(
    monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]], unusable: int
) -> None:
    """NGC counts each row it gives. A row without a resourceId cannot be checked, so the catalog
    is not complete and the list says it is partial; a resource given twice is still read in
    full."""
    body = {"results": [{"groupValue": "CONTAINER", "resources": rows}], "resultTotal": len(rows)}
    with serve({"/v2/search/catalog/resources/CONTAINER": (200, body)}) as fake:
        set_ngc_base(monkeypatch, fake.base)
        catalog = ngc_api.list_catalog()
    assert catalog.resources == [CATALOG_ROW] and len(fake.requests) == 1
    assert catalog.complete is (unusable == 0)
    assert catalog.unusable == unusable


@pytest.mark.parametrize("rid", ["", "   "], ids=["empty", "spaces"])
def test_a_catalog_row_with_a_blank_resource_id_is_unusable(
    monkeypatch: pytest.MonkeyPatch, rid: str
) -> None:
    """The reported case: a one-row catalog whose resourceId was "" read as complete, with no
    unusable row, and the list then skipped the row. A blank id cannot be checked either."""
    row = {**CATALOG_ROW, "resourceId": rid}
    body = {"results": [{"groupValue": "CONTAINER", "resources": [row]}], "resultTotal": 1}
    with serve({"/v2/search/catalog/resources/CONTAINER": (200, body)}) as fake:
        set_ngc_base(monkeypatch, fake.base)
        catalog = ngc_api.list_catalog()
    assert (catalog.resources, catalog.unusable, catalog.complete) == ([], 1, False)


# --- script entry point --------------------------------------------------------------------------


def test_library_notice() -> None:
    code, payload = run_cli("ngc_api.py")
    assert code == 0 and payload["state"] == "library"


def test_help() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "ngc_api.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0 and "Example:" in proc.stdout


# --- live NGC (opt-in: NIM_SELECT_LIVE=1 pytest -m network) ---------------------------------------

live = pytest.mark.skipif(
    os.environ.get("NIM_SELECT_LIVE") != "1", reason="live NGC test; set NIM_SELECT_LIVE=1"
)


@pytest.mark.network
@live
def test_live_repo_tag_and_metadata() -> None:
    repo = ngc_api.get_repo(REPO_ID)
    assert repo is not None
    tag, source, _ = ngc_api.choose_tag(repo, None)
    assert source == "newest_version"
    meta = ngc_api.get_metadata(REPO_ID, tag)
    assert meta is not None and meta.get("profiles")
    assert ngc_api.get_metadata(REPO_ID, "1.0.0") is None


@pytest.mark.network
@live
def test_live_search() -> None:
    ids = [c["resource_id"] for c in ngc_api.search_nims("llama-3.1-8b-instruct")]
    assert REPO_ID in ids


@pytest.mark.network
@live
def test_live_image_architectures() -> None:
    images = ngc_api.get_images(REPO_ID)
    assert images is not None
    assert "amd64" in images["2.0.13"].architectures
    assert images["1.3.3"].architectures == ("amd64",)
    assert images["1.3.3"].digest and images["1.3.3"].size
