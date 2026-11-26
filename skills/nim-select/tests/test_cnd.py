# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/cnd.py: the source of a model reference, the catalog search on each China NIM
distributor, and the NGC builds a CND image may be.

The catalog answers in tests/fixtures/cnd are real, read anonymously on 2026-11-05 (trimmed to the
fields read). A local HTTP server stands in for the TGC and TuringCM catalogs and the Harbor
hosts; conftest.py points every other test at a closed port.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from typing import Any

import cnd
import ns_common as ns
import pytest
import registry_labels as rl
from ns_testkit import (
    CLOSE,
    GLM51,
    FakeNgc,
    fixture_json,
    glm_routes,
    run_cli,
    serve,
    set_harbor_base,
    set_tgc_base,
    set_turingcm_base,
    use_built_in_endpoints,
)

TGC_SEARCH = "/tgc/api/resources"
TURINGCM = "/server/resource/nimImages"
LICHAN_REPOS = "/api/v2.0/projects/cnd/repositories"
LICHAN_QWEN = f"{LICHAN_REPOS}/qwen%252Fqwen3.5-35b-a3b/artifacts"


def fx(name: str) -> Any:
    return fixture_json("cnd", name)


@pytest.fixture
def catalogs(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Routes for one fake server that answers as every catalog and Harbor host."""
    routes: dict[str, Any] = {}
    with serve(routes) as server:
        root = server.base.removesuffix("/v2")
        set_tgc_base(monkeypatch, root)
        set_turingcm_base(monkeypatch, root)
        set_harbor_base(monkeypatch, root)
        routes["_server"] = server
        yield routes


def requests_of(routes: dict[str, Any]) -> list[str]:
    server: FakeNgc = routes["_server"]
    return list(server.requests)


def tgc_routes() -> dict[str, Any]:
    return {
        TGC_SEARCH: (200, fx("tgc-search-qwen3.6-27b.json")),
        f"{TGC_SEARCH}/100341": (200, fx("tgc-detail-qwen3.6-27b.json")),
        f"{TGC_SEARCH}/100339": (200, fx("tgc-detail-qwen3.6-27b-fp8.json")),
        f"{TGC_SEARCH}/100340": (200, fx("tgc-detail-qwen3.6-27b-fp8-dgx-spark.json")),
    }


def turingcm_routes() -> dict[str, Any]:
    return {
        f"{TURINGCM}/simple-list": (200, fx("turingcm-simple-list.json")),
        f"{TURINGCM}/getNimImagesById?id=-20": (200, fx("turingcm-detail-qwen3.6-27b.json")),
    }


def lichan_routes() -> dict[str, Any]:
    return {
        LICHAN_REPOS: (200, fx("lichan-repositories.json")),
        LICHAN_QWEN: (200, fx("lichan-artifacts-qwen3.5-35b-a3b.json")),
    }


# --- the source -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "host", "source"),
    [
        ("nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.4", "nvcr.io", "ngc"),
        ("tgcr.turing-agi.com/cnd/z-ai/glm5:1.0.0", "tgcr.turing-agi.com", "tgcr"),
        ("tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b", "tgcr-gz.turing-agi.com", "tgcr"),
        ("resource.turingcm.com/qwen/qwen3.6-27b:1.0", "resource.turingcm.com", "turingcm"),
        ("IO.ChanCloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0", "io.chancloud.com", "lichan"),
        ("registry.example.com/nim/meta/llama-3.1-8b-instruct", "registry.example.com", "mirror"),
        ("localhost:5000/meta/llama", "localhost:5000", "mirror"),
        ("meta/llama-3.1-8b-instruct", None, None),
        ("nim/meta/llama-3.1-8b-instruct", None, None),
        ("qwen3.6-27b", None, None),
        ("llama 3.1 8b", None, None),
    ],
)
def test_host_to_source(model: str, host: str | None, source: str | None) -> None:
    assert cnd.reference_host(model) == host
    assert cnd.host_source(host) == source


@pytest.mark.parametrize(
    ("model", "flag", "source"),
    [
        ("io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0", None, "lichan"),
        ("io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0", "lichan", "lichan"),
        ("nvcr.io/nim/qwen/qwen3.5-35b-a3b", None, "ngc"),
        ("registry.example.com/nim/meta/llama", "ngc", "ngc"),
        ("qwen3.6-27b", "turingcm", "turingcm"),
        ("qwen3.6-27b", None, "ngc"),
    ],
)
def test_the_source_is_the_host_else_the_flag_else_ngc(
    model: str, flag: str | None, source: str
) -> None:
    assert cnd.choose_source(model, flag) == source


@pytest.mark.parametrize(
    ("model", "flag"),
    [
        ("io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0", "ngc"),
        ("io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0", "turingcm"),
        ("nvcr.io/nim/qwen/qwen3.5-35b-a3b", "tgcr"),
        ("registry.example.com/nim/meta/llama", "lichan"),
    ],
)
def test_a_host_that_contradicts_the_source_is_bad_input(model: str, flag: str) -> None:
    with pytest.raises(ns.ScriptError) as caught:
        cnd.choose_source(model, flag)
    assert (caught.value.code, caught.value.exit_code) == ("source_conflict", ns.EXIT_INPUT)
    assert flag in caught.value.message


# --- names, tags and repositories ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "words"),
    [
        ("qwen3.6-27b", ["qwen3.6-27b"]),
        ("Qwen3.6-27B", ["qwen3.6-27b"]),
        ("qwen/qwen3.6-27b", ["qwen3.6-27b"]),
        ("nim/qwen/qwen3.6-27b:1.0", ["qwen3.6-27b"]),
        ("qwen 3.6 27b", ["qwen", "3.6", "27b"]),
        ("  ", []),
    ],
)
def test_query_words(query: str, words: list[str]) -> None:
    assert cnd.query_words(query) == words


def test_names_match_by_their_words_and_one_matches_exactly() -> None:
    words = cnd.query_words("qwen3.6-27b")
    names = ["Qwen3.6-27B", "Qwen3.6-27B-FP8", "Qwen3.6-35B-A3B"]
    assert [name for name in names if cnd.matches(words, (name,))] == names[:2]
    assert [name for name in names if cnd.exact(words, (name,))] == ["Qwen3.6-27B"]
    assert cnd.matches(cnd.query_words("qwen 3.6 27b"), ("Qwen3.6-27B-FP8",))
    assert cnd.exact(["qwen3.6-27b"], ("tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b:latest",))
    assert not cnd.matches([], ("anything",))


@pytest.mark.parametrize(
    ("tags", "pinned"),
    [
        (("1.0.0", "1.0", "1", "latest"), "1.0.0"),
        (("1.0", "1", "latest"), "1.0"),
        (("1.0_sglang", "1_sglang", "latest"), "1.0_sglang"),
        (("2.1.2-variant-spark",), "2.1.2-variant-spark"),
        (("latest",), None),
        ((), None),
    ],
)
def test_the_pinned_tag_is_the_most_specific_one_but_latest(
    tags: tuple[str, ...], pinned: str | None
) -> None:
    assert cnd.pinned_tag(tags) == pinned


@pytest.mark.parametrize(
    ("image", "resource_id"),
    [
        ("io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0", "nim/qwen/qwen3.5-35b-a3b"),
        ("io.chancloud.com/cnd/nim/qwen/qwen-2.5-7b-instruct", "nim/qwen/qwen-2.5-7b-instruct"),
        ("tgcr.turing-agi.com/cnd-enterprise/qwen/qwen3.5-397b-a17b", "nim/qwen/qwen3.5-397b-a17b"),
        ("tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b:latest", "nim/qwen/qwen3.6-27b"),
        ("resource.turingcm.com/qwen/qwen3.6-27b:1.0", "nim/qwen/qwen3.6-27b"),
        ("resource.turingcm.com/nim/meta/llama-3.1-8b-instruct", "nim/meta/llama-3.1-8b-instruct"),
        ("io.chancloud.com/cnd/qwen3.5-35b-a3b-spark-nim:1.0", None),
        ("io.chancloud.com/cnd/nvidia/blueprint/vss-engine:2.4", None),
    ],
)
def test_the_ngc_repository_of_a_cnd_path(image: str, resource_id: str | None) -> None:
    ref = rl.parse_image(image)
    assert ref is not None
    assert cnd.ngc_repository(ref) == resource_id


# --- the NGC builds a CND image may be (the order of the mapping) ------------------------------


LICHAN = rl.parse_image("io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0")
PUBLIC = "nvcr.io/nim/qwen/qwen3.5-35b-a3b-rtx:1.7.0"


def mappings(found: list[cnd.Target]) -> list[tuple[str, str, str]]:
    return [(t.mapping, t.resource_id, t.tag) for t in found]


def test_a_public_ngc_url_comes_first_then_the_version_then_the_users_version() -> None:
    assert LICHAN is not None
    found, unused = cnd.targets(LICHAN, "1.7.0-variant", PUBLIC, None, "1.7.0")
    assert mappings(found) == [
        ("ngc_url", "nim/qwen/qwen3.5-35b-a3b-rtx", "1.7.0"),
        ("version_label", "nim/qwen/qwen3.5-35b-a3b", "1.7.0-variant"),
        ("user_version", "nim/qwen/qwen3.5-35b-a3b", "1.7.0"),
    ]
    assert unused == []
    assert "com.nvidia.nim.ngc_url label" in found[0].origin
    assert found[2].origin == "--nim-version 1.7.0"


@pytest.mark.parametrize(
    "ngc_url",
    [
        "nvcr.io/example-org/nim/qwen3.5-35b-a3b:modelscope_2",  # the real label: a non-public org
        "nvcr.io/nim/qwen/qwen3.5-35b-a3b:latest",
        "nvcr.io/0123456789/qwen/qwen3.5-35b-a3b:build_2",
        # a version tag does not make a staging or private-org path public
        "nvcr.io/example-org/nim/qwen3.5-35b-a3b:1.7.0-variant",
        "nvcr.io/0123456789/qwen/qwen3.5-35b-a3b:1.7.0",
    ],
)
def test_a_non_public_or_latest_ngc_url_is_not_used(ngc_url: str) -> None:
    assert LICHAN is not None
    found, unused = cnd.targets(LICHAN, "1.7.0-variant", ngc_url, None, None)
    assert mappings(found) == [("version_label", "nim/qwen/qwen3.5-35b-a3b", "1.7.0-variant")]
    [item] = unused
    assert item["mapping"] == "ngc_url" and "not a public NGC NIM image" in item["problem"]


def test_the_tgc_catalog_ngc_url_maps_a_tgcr_image() -> None:
    ref = rl.parse_image("tgcr-gz.turing-agi.com/public/z-ai/glm-5.2:latest")
    assert ref is not None
    found, _ = cnd.targets(ref, None, None, "nvcr.io/nim/zai-org/glm-5.2:1.11.1-variant", None)
    # the catalog names another NGC team than the path (z-ai): never the path's name
    assert mappings(found) == [("ngc_url", "nim/zai-org/glm-5.2", "1.11.1-variant")]
    assert "TGC catalog" in found[0].origin


def test_without_labels_or_a_version_there_is_no_build() -> None:
    ref = rl.parse_image("resource.turingcm.com/qwen/qwen3.6-27b:1.0")
    assert ref is not None
    assert cnd.targets(ref, None, None, None, None) == ([], [])


def test_a_version_label_needs_a_version_and_a_team_repo_path() -> None:
    assert LICHAN is not None
    found, unused = cnd.targets(LICHAN, "latest", None, None, None)
    assert found == [] and "not a version tag" in unused[0]["problem"]
    found, _ = cnd.targets(LICHAN, "1.7.0=variant", None, None, None)
    assert mappings(found) == [("version_label", "nim/qwen/qwen3.5-35b-a3b", "1.7.0-variant")]
    assert "read as 1.7.0-variant" in found[0].origin
    spark = rl.parse_image("io.chancloud.com/cnd/qwen3.5-35b-a3b-spark-nim:1.0")
    assert spark is not None
    found, unused = cnd.targets(spark, "1.7.0-variant", None, None, "1.7.0-variant")
    assert found == []
    assert [item["mapping"] for item in unused] == ["version_label", "user_version"]
    assert all("TEAM/REPO" in item["problem"] for item in unused)


# --- catalog search ---------------------------------------------------------------------------


def test_tgc_search_finds_one_by_its_exact_name(catalogs: dict[str, Any]) -> None:
    catalogs.update(tgc_routes())
    result = cnd.search("tgcr", "qwen3.6-27b")
    assert result.total == 3
    assert result.found[0].name == "Qwen3.6-27B"
    found = cnd.pick("tgcr", "qwen3.6-27b", result)
    assert found.image == "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b:latest"
    assert found.catalog_ngc_url == "nvcr.io/nim/qwen/qwen3.6-27b:1.7.0-variant"
    [tag] = found.tags
    assert (tag.tag, tag.ngc_url, tag.arch) == (
        "latest",
        "nvcr.io/nim/qwen/qwen3.6-27b:1.7.0-variant",
        None,
    )
    assert tag.digest and tag.digest.startswith("sha256:9bfb")
    sent = requests_of(catalogs)
    assert sent[0] == f"{TGC_SEARCH}?page=1&size=100&search=qwen3.6-27b"
    assert len(sent) == 4  # one search, one detail per match


def test_tgc_search_with_several_matches_asks(catalogs: dict[str, Any]) -> None:
    catalogs.update(tgc_routes())
    # two names match qwen3.6-27b-fp8, and one of them is that name exactly
    found = cnd.pick("tgcr", "qwen3.6-27b-fp8", cnd.search("tgcr", "qwen3.6-27b-fp8"))
    assert found.name == "Qwen3.6-27B-FP8"
    with pytest.raises(ns.ScriptError) as caught:
        cnd.pick("tgcr", "qwen3.6", cnd.search("tgcr", "qwen3.6"))
    error = caught.value
    assert (error.code, error.exit_code) == ("model_ambiguous", ns.EXIT_PRECONDITION)
    assert [item["catalog_name"] for item in error.facts["candidates"]] == [
        "Qwen3.6-27B",
        "Qwen3.6-27B-FP8",
        "Qwen3.6-27B-FP8-DGX-Spark",
    ]
    # the catalog's arch and model type help the user choose (the DGX Spark build is arm64)
    assert error.facts["candidates"][1] == {
        "source": "tgcr",
        "catalog_name": "Qwen3.6-27B-FP8",
        "image": "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b-fp8:latest",
        "repository": "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b-fp8",
        "newest_tag": "latest",
        "architectures": ["amd64"],
        "model_type": "vlm",
    }
    assert error.facts["candidates"][2]["architectures"] == ["arm64"]
    assert error.message.startswith("3 NIMs in the TGC catalog match 'qwen3.6'")
    # the rerun takes the chosen repository; a tag only when the user names one
    assert "facts.candidates[].repository" in error.fix and "--tag only if" in error.fix


def test_two_catalog_items_of_one_image_are_one_match(catalogs: dict[str, Any]) -> None:
    # TGC lists Qwen3.6-35B-A3B twice for DGX Spark, under two names, with one image
    image = "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-35b-a3b-spark:latest"
    tag = {
        "docker_url": image,
        "create_time": "2026-04-24 10:00:00",
        "digest": "sha256:" + "1" * 64,
    }
    names = {"1": "Qwen3.6-35B-A3B NIM Container for DGX Spark", "2": "Qwen3.6-35B-A3B-DGX-Spark"}
    catalogs[TGC_SEARCH] = (
        200,
        {"code": 200, "data": {"items": [{"id": i, "name": n} for i, n in names.items()]}},
    )
    for item_id, name in names.items():
        detail = {"id": item_id, "name": name, "image_tags": [tag]}
        catalogs[f"{TGC_SEARCH}/{item_id}"] = (200, {"code": 200, "data": detail})
    query = "qwen 3.6 35b spark"  # both names have every word; neither is named exactly
    assert cnd.pick("tgcr", query, cnd.search("tgcr", query)).image == image


def test_turingcm_search_reads_the_image_from_the_deploy_text(catalogs: dict[str, Any]) -> None:
    catalogs.update(turingcm_routes())
    result = cnd.search("turingcm", "qwen3.6-27b")
    found = cnd.pick("turingcm", "qwen3.6-27b", result)
    assert (found.name, found.image, found.item_id) == (
        "Qwen3.6-27B",
        "resource.turingcm.com/qwen/qwen3.6-27b:latest",
        "-20",
    )
    assert found.candidate() == {
        "source": "turingcm",
        "catalog_name": "Qwen3.6-27B",
        "image": "resource.turingcm.com/qwen/qwen3.6-27b:latest",
        "repository": "resource.turingcm.com/qwen/qwen3.6-27b",
        "newest_tag": None,  # the deploy text names latest, not the newest tag of the repository
        "architectures": None,  # TuringCM leaves arch and model_type empty for most items
        "model_type": None,
    }


def test_lichan_search_reads_the_newest_tag_of_each_match(catalogs: dict[str, Any]) -> None:
    catalogs.update(lichan_routes())
    found = cnd.pick("lichan", "qwen3.5-35b-a3b", cnd.search("lichan", "qwen3.5-35b-a3b"))
    assert found.image == "io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0"
    assert found.architectures == ("amd64",)
    assert found.listing is not None and found.listing[0] == "read"
    assert found.listing[1][0].version == "1.7.0-variant"


def test_lichan_search_with_two_repositories_of_one_name_asks(catalogs: dict[str, Any]) -> None:
    catalogs.update(lichan_routes())
    with pytest.raises(ns.ScriptError) as caught:
        cnd.pick("lichan", "qwen3-32b-dgx-spark-s3", cnd.search("lichan", "qwen3-32b-dgx-spark-s3"))
    error = caught.value
    assert (error.code, error.exit_code) == ("model_ambiguous", ns.EXIT_PRECONDITION)
    shown = {item["catalog_name"] for item in error.facts["candidates"]}
    assert shown == {"qwen/qwen3-32b-dgx-spark-s3", "nim/qwen/qwen3-32b-dgx-spark-s3"}
    assert error.facts["source"] == "lichan"


@pytest.mark.parametrize(
    ("source", "routes", "others"),
    [
        ("tgcr", tgc_routes, ["turingcm", "lichan"]),
        ("turingcm", turingcm_routes, ["tgcr", "lichan"]),
        ("lichan", lichan_routes, ["tgcr", "turingcm"]),
    ],
)
def test_a_name_no_catalog_lists_is_not_found_on_that_cnd_only(
    catalogs: dict[str, Any], source: str, routes: Any, others: list[str]
) -> None:
    catalogs.update(routes())
    if source == "tgcr":  # the TGC search is done by the server
        catalogs[TGC_SEARCH] = (200, {"code": 200, "data": {"items": [], "total": 0}})
    with pytest.raises(ns.ScriptError) as caught:
        cnd.pick(source, "no-such-model-xyz", cnd.search(source, "no-such-model-xyz"))
    error = caught.value
    assert (error.code, error.exit_code) == ("model_not_found", ns.EXIT_PRECONDITION)
    assert cnd.CATALOGS[source] in error.message
    # the other two CNDs are offered, never searched: each lists its own NIMs
    assert error.facts["other_sources"] == others
    for other in others:
        assert f"{cnd.CATALOGS[other]} (--source {other})" in error.fix
    assert "only when the user says so, never on your own" in error.fix
    assert "never NGC unless the user asks" in error.fix
    hosts = {"tgcr": TGC_SEARCH, "turingcm": TURINGCM, "lichan": LICHAN_REPOS}
    sent = requests_of(catalogs)
    assert sent and all(path.startswith(hosts[source]) for path in sent)


def test_names_with_no_image_are_not_found_and_say_so(catalogs: dict[str, Any]) -> None:
    catalogs.update(tgc_routes())
    for item_id in ("100341", "100339", "100340"):
        catalogs[f"{TGC_SEARCH}/{item_id}"] = (200, {"code": 200, "data": {"image_tags": []}})
    with pytest.raises(ns.ScriptError) as caught:
        cnd.pick("tgcr", "qwen3.6-27b", cnd.search("tgcr", "qwen3.6-27b"))
    assert caught.value.code == "model_not_found"
    assert "3 name(s) match, but the catalog names no image" in caught.value.message


@pytest.mark.parametrize("source", ["tgcr", "turingcm", "lichan"])
def test_a_catalog_that_does_not_answer_is_an_external_failure(source: str) -> None:
    # conftest points every catalog and Harbor host at a closed port
    with pytest.raises(ns.ScriptError) as caught:
        cnd.search(source, "qwen3.6-27b")
    assert (caught.value.code, caught.value.exit_code) == ("cnd_unreachable", ns.EXIT_EXTERNAL)


def test_a_catalog_that_never_answers_is_cut_off_by_the_search_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(8)
        set_turingcm_base(monkeypatch, f"http://127.0.0.1:{silent.getsockname()[1]}")
        monkeypatch.setattr(cnd, "SEARCH_BUDGET", 0.5)
        start = time.monotonic()
        with pytest.raises(ns.ScriptError) as caught:
            cnd.search("turingcm", "qwen")
        assert time.monotonic() - start < 2.5
    assert caught.value.code == "cnd_unreachable" and "within 0.5 s" in caught.value.message


def test_a_detail_that_fails_leaves_out_that_match_only(catalogs: dict[str, Any]) -> None:
    catalogs.update(tgc_routes())
    catalogs[f"{TGC_SEARCH}/100339"] = (500, {})
    result = cnd.search("tgcr", "qwen3.6-27b")
    assert sorted(item.name for item in result.found) == [
        "Qwen3.6-27B",
        "Qwen3.6-27B-FP8-DGX-Spark",
    ]
    for item_id in ("100341", "100340"):
        catalogs[f"{TGC_SEARCH}/{item_id}"] = (500, {})
    with pytest.raises(ns.ScriptError) as caught:
        cnd.search("tgcr", "qwen3.6-27b")
    assert caught.value.code == "cnd_unreachable"


def test_a_failed_detail_is_never_reported_as_a_name_with_no_image(
    catalogs: dict[str, Any],
) -> None:
    # one detail fails and the others name no image: the failed one may be the model, so the
    # search is cnd_unreachable, never model_not_found (which offers the other CNDs)
    catalogs.update(tgc_routes())
    catalogs[f"{TGC_SEARCH}/100339"] = (500, {})
    for item_id in ("100341", "100340"):
        catalogs[f"{TGC_SEARCH}/{item_id}"] = (200, {"code": 200, "data": {"image_tags": []}})
    with pytest.raises(ns.ScriptError) as caught:
        cnd.search("tgcr", "qwen3.6-27b")
    assert (caught.value.code, caught.value.exit_code) == ("cnd_unreachable", ns.EXIT_EXTERNAL)
    assert "HTTP 500" in caught.value.message


def test_the_tgc_catalog_names_the_tags_of_a_tgcr_repository(catalogs: dict[str, Any]) -> None:
    catalogs.update(tgc_routes())
    ref = rl.parse_image("tgcr.turing-agi.com/public/qwen/qwen3.6-27b:1.0")
    assert ref is not None
    tgc = cnd.reader("tgcr")
    assert isinstance(tgc, cnd.TgcReader)
    tags, where = tgc.catalog_tags(ref, None)  # the other tgcr host serves the same repository
    assert where == "the TGC catalog"
    assert [tag.image for tag in tags] == ["tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b:latest"]


@pytest.mark.parametrize(
    ("deploy", "images"),
    [
        ("docker run -it \\\nresource.turingcm.com/qwen/qwen3.6-27b:latest\n</code>",
         ["resource.turingcm.com/qwen/qwen3.6-27b:latest"]),
        ("<pre>resource.turingcm.com/megacube/qwen3-32b-dgx-spark:latest</pre>",
         ["resource.turingcm.com/megacube/qwen3-32b-dgx-spark:latest"]),
        # one image per repository, in order (DeepSeek-V4-Flash-0731 names its Spark build twice)
        (("<pre>resource.turingcm.com/d/x-spark:latest</pre> resource.turingcm.com/d/x-spark:latest"
          " <pre>resource.turingcm.com/d/x:latest</pre>"),
         ["resource.turingcm.com/d/x-spark:latest", "resource.turingcm.com/d/x:latest"]),
        ("no image here", []),
        ("resource.turingcm.com/ alone", []),
        ("resource.turingcm.com/qwen/qwen3.6-27b:1.0|sh", []),  # no image reference
    ],
)  # fmt: skip
def test_turingcm_images(deploy: str, images: list[str]) -> None:
    assert cnd.turingcm_images(deploy) == images


def test_tgc_tags_take_the_items_ngc_url_only_for_its_one_tag() -> None:
    detail = fx("tgc-detail-qwen3.6-27b.json")["data"]
    one = {**detail, "image_tags": [{**detail["image_tags"][0], "ngc_url": None}]}
    assert [tag.ngc_url for tag in cnd.tgc_tags(one)] == [detail["ngc_url"]]
    two = {**one, "image_tags": [*one["image_tags"], {**one["image_tags"][0], "tag_name": "1.0"}]}
    two["image_tags"][1]["docker_url"] = "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b:1.0"
    assert [tag.ngc_url for tag in cnd.tgc_tags(two)] == [None, None]


# --- one reader per CND -------------------------------------------------------------------------

TGCR_LATEST = "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b:latest"
TGCR_HARBOR = "/api/v2.0/projects/public/repositories/qwen%252Fqwen3.6-27b/artifacts"
TURINGCM_HARBOR = "/api/v2.0/projects/qwen/repositories/qwen3.6-27b/artifacts"
TURINGCM_CHILD = "sha256:efd6e93849fa23ee07f46e79bbb4d55a513d97d3537311b60b310805f97a3986"
LOGIN = (401, fx("tgcr-401.json"))


def test_each_cnd_has_its_own_reader_behind_one_interface() -> None:
    assert list(cnd.READERS) == list(cnd.CNDS)
    kinds = {source: type(reader) for source, reader in cnd.READERS.items()}
    assert kinds == {
        "tgcr": cnd.TgcReader,
        "turingcm": cnd.TuringcmReader,
        "lichan": cnd.LichanReader,
    }
    for source, reader in cnd.READERS.items():
        assert isinstance(reader, cnd.Reader) and reader.source == source
        # each searches its own catalog API; image and images read Harbor, or the TGC catalog
        assert type(reader).search is not cnd.Reader.search
        assert type(reader).image is not cnd.Reader.image
        assert type(reader).images is not cnd.Reader.images
    assert cnd.CATALOGS == {"tgcr": "TGC", "turingcm": "TuringCM", "lichan": "Lichan"}
    # each catalog has its own host: a search reads only the CND it was asked for
    assert {r.catalog_host for r in cnd.READERS.values()} == {
        "tgc.turing-agi.com",
        "nim.turingcm.com",
        "io.chancloud.com",
    }


def test_the_lichan_reader_reads_the_version_label(catalogs: dict[str, Any]) -> None:
    catalogs[f"{LICHAN_QWEN}/1.0.0"] = (200, fx("lichan-artifact-qwen3.5-35b-a3b-1.0.0.json"))
    ref = rl.parse_image("io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0")
    assert ref is not None
    image = cnd.reader("lichan").image(ref, None, "amd64")
    assert (image.version, image.architectures, image.pinned) == (
        "1.7.0-variant",
        ("amd64",),
        "1.0.0",
    )
    assert image.ngc_url and "example-org" in image.ngc_url and image.problems == []


def test_the_turingcm_reader_follows_the_index_to_an_image_without_labels(
    catalogs: dict[str, Any],
) -> None:
    catalogs.update(
        {
            f"{TURINGCM_HARBOR}/1.0": (200, fx("turingcm-artifacts-qwen3.6-27b.json")[0]),
            f"{TURINGCM_HARBOR}/{TURINGCM_CHILD}": (200, fx("turingcm-child-qwen3.6-27b.json")),
            TURINGCM_HARBOR: (200, fx("turingcm-artifacts-qwen3.6-27b.json")),
        }
    )
    ref = rl.parse_image("resource.turingcm.com/qwen/qwen3.6-27b:1.0")
    assert ref is not None
    reader = cnd.reader("turingcm")
    image = reader.image(ref, None, "amd64")
    assert (image.version, image.ngc_url, image.tags) == (None, None, ("1.0", "1", "latest"))
    assert image.problems and rl.VERSION_LABEL in image.problems[0]
    images, problem = reader.images(ref.with_tag(None), None, "amd64")
    assert problem is None
    assert [item.ref.image for item in images] == ["resource.turingcm.com/qwen/qwen3.6-27b:1.0"]


def test_the_tgc_reader_asks_its_catalog_when_harbor_needs_a_login(
    catalogs: dict[str, Any],
) -> None:
    catalogs.update(tgc_routes())
    # tgcr's Harbor answers 401 to every anonymous request
    catalogs.update({f"{TGCR_HARBOR}/{t}": LOGIN for t in ("latest", "1.0")})
    catalogs[TGCR_HARBOR] = LOGIN
    ref = rl.parse_image(TGCR_LATEST)
    assert ref is not None
    reader = cnd.reader("tgcr")
    image = reader.image(ref, None, "amd64")
    assert image.catalog_ngc_url == "nvcr.io/nim/qwen/qwen3.6-27b:1.7.0-variant"
    assert image.catalog is not None and image.catalog["catalog"] == "TGC"
    assert image.digest and image.digest.startswith("sha256:9bfb")
    assert "needs a login on tgcr.turing-agi.com" in image.problems[0]
    images, problem = reader.images(ref.with_tag(None), None, "amd64")
    assert problem is None
    assert [(item.ref.image, item.pinned) for item in images] == [(TGCR_LATEST, None)]
    # a tag the catalog does not list: nothing names its build
    other = reader.image(ref.with_tag("1.0"), None, "amd64")
    assert other.catalog_ngc_url is None
    assert any("lists no tag 1.0" in problem for problem in other.problems)


def test_a_tag_the_found_tgc_item_lacks_is_looked_up_in_the_other_items(
    catalogs: dict[str, Any],
) -> None:
    # TGC can list two tags of one repository as two items, and a name need not hold the path
    catalogs.update(tgc_routes())
    catalogs[f"{TGCR_HARBOR}/1.0"] = LOGIN
    other = {"id": 100350, "name": "Qwen3.6 27B (1.0)"}  # a search for qwen3.6-27b misses it
    tag = {
        "docker_url": "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b:1.0",
        "ngc_url": "nvcr.io/nim/qwen/qwen3.6-27b:1.6.0",
        "create_time": "2026-03-01 10:00:00",
        "digest": "sha256:" + "2" * 64,
    }
    every = fx("tgc-search-qwen3.6-27b.json")
    every["data"]["items"].append(other)
    every["data"]["total"] += 1
    catalogs[f"{TGC_SEARCH}?page=1&size=100"] = (200, every)
    catalogs[f"{TGC_SEARCH}/100350"] = (200, {"code": 200, "data": {**other, "image_tags": [tag]}})
    found = cnd.pick("tgcr", "qwen3.6-27b", cnd.search("tgcr", "qwen3.6-27b"))
    assert [entry.tag for entry in found.tags] == ["latest"]
    ref = rl.parse_image(found.image or "")
    assert ref is not None
    image = cnd.reader("tgcr").image(ref.with_tag("1.0"), found, "amd64")
    assert image.catalog_ngc_url == "nvcr.io/nim/qwen/qwen3.6-27b:1.6.0"


def test_a_tag_the_found_tgc_item_lacks_is_looked_up_without_searching_again(
    catalogs: dict[str, Any],
) -> None:
    # the search that found the item read the items named like it: the rest of the catalog is
    # read, and no search or detail twice
    catalogs.update(tgc_routes())
    catalogs[f"{TGCR_HARBOR}/9.9.9"] = LOGIN
    found = cnd.pick("tgcr", "qwen3.6-27b", cnd.search("tgcr", "qwen3.6-27b"))
    ref = rl.parse_image(found.image or "")
    assert ref is not None
    before = len(requests_of(catalogs))
    image = cnd.reader("tgcr").image(ref.with_tag("9.9.9"), found, "amd64")
    assert requests_of(catalogs)[before:] == [
        f"{TGCR_HARBOR}/9.9.9?with_label=true",
        f"{TGC_SEARCH}?page=1&size=100",  # every item: the search read all three
    ]
    assert image.problems[-1] == f"the TGC catalog lists no tag 9.9.9 of {ref.repository_ref}"


def test_a_tag_the_found_tgc_item_lacks_is_not_listed_when_the_rest_takes_too_long(
    catalogs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # the catalog answered for the item: the tag is one it does not list, with a note, never
    # cnd_unreachable
    catalogs.update(tgc_routes())
    catalogs[f"{TGCR_HARBOR}/9.9.9"] = LOGIN
    found = cnd.pick("tgcr", "qwen3.6-27b", cnd.search("tgcr", "qwen3.6-27b"))
    ref = rl.parse_image(found.image or "")
    assert ref is not None
    release, real = threading.Event(), cnd.TgcReader.items

    def every_item_never_answers(
        self: cnd.TgcReader, base: str, word: str | None
    ) -> tuple[list[dict[str, Any]], bool]:
        if word is None:
            release.wait(5)
            return [], False
        return real(self, base, word)

    monkeypatch.setattr(cnd.TgcReader, "items", every_item_never_answers)
    monkeypatch.setattr(cnd, "SEARCH_BUDGET", 0.3)
    try:
        image = cnd.reader("tgcr").image(ref.with_tag("9.9.9"), found, "amd64")
    finally:
        release.set()
    assert image.catalog_ngc_url is None
    assert image.problems[-1] == (
        "the TGC catalog item 'Qwen3.6-27B' (the other catalog items were not all read within "
        f"0.3 s) lists no tag 9.9.9 of {ref.repository_ref}"
    )


def test_a_tag_the_found_tgc_item_lacks_is_not_listed_when_a_request_for_the_rest_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the catalog answered for the item, so the tag is one it does not list, with a note, never
    # cnd_unreachable; with no item found first nothing is known, and the timeout is
    every_item = f"{TGC_SEARCH}?page=1&size=100"  # the catalog read without a search word
    routes = {**tgc_routes(), every_item: CLOSE, f"{TGCR_HARBOR}/9.9.9": LOGIN}
    with serve(routes, slow={every_item: 1.5}) as server:  # no answer within CATALOG_TIMEOUT
        root = server.base.removesuffix("/v2")
        set_tgc_base(monkeypatch, root)
        set_harbor_base(monkeypatch, root)
        found = cnd.pick("tgcr", "qwen3.6-27b", cnd.search("tgcr", "qwen3.6-27b"))
        ref = rl.parse_image(found.image or "")
        assert ref is not None
        ref = ref.with_tag("9.9.9")
        monkeypatch.setattr(cnd, "CATALOG_TIMEOUT", 0.5)
        image = cnd.reader("tgcr").image(ref, found, "amd64")
        assert image.catalog_ngc_url is None
        assert image.problems[-1] == (
            "the TGC catalog item 'Qwen3.6-27B' (the other catalog items could not all be read) "
            f"lists no tag 9.9.9 of {ref.repository_ref}"
        )
        with pytest.raises(ns.ScriptError) as caught:
            cnd.reader("tgcr").image(ref, None, "amd64")
    assert (caught.value.code, caught.value.exit_code) == ("cnd_unreachable", ns.EXIT_EXTERNAL)
    assert "timed out" in caught.value.message


OTHER_REPOSITORY_TAG = {
    "docker_url": "tgcr-gz.turing-agi.com/public/x/other:1.0",
    "create_time": "2026-05-01 10:00:00",
}


@pytest.mark.parametrize(
    ("details", "where"),
    [
        ([None], "the TGC catalog item 'Qwen3.6-27B'"),  # every other detail fails
        ([None, [OTHER_REPOSITORY_TAG]], "the TGC catalog"),  # one of two fails
    ],
)
def test_a_tag_the_found_tgc_item_lacks_is_not_listed_when_another_detail_fails(
    catalogs: dict[str, Any], details: list[list[dict[str, str]] | None], where: str
) -> None:
    # a failed detail (None; else the image tags it lists) may be the item that lists the tag:
    # not listed, with a note, never cnd_unreachable, and never a plain "lists no tag"
    catalogs.update(tgc_routes())
    catalogs[f"{TGCR_HARBOR}/9.9.9"] = LOGIN
    every = fx("tgc-search-qwen3.6-27b.json")
    for item_id, tags in enumerate(details, start=100350):
        item = {"id": item_id, "name": f"Other {item_id}"}
        every["data"]["items"].append(item)
        body = {"code": 200, "data": {**item, "image_tags": tags}}
        catalogs[f"{TGC_SEARCH}/{item_id}"] = (500, {}) if tags is None else (200, body)
    every["data"]["total"] = len(every["data"]["items"])
    catalogs[f"{TGC_SEARCH}?page=1&size=100"] = (200, every)
    found = cnd.pick("tgcr", "qwen3.6-27b", cnd.search("tgcr", "qwen3.6-27b"))
    ref = rl.parse_image(found.image or "")
    assert ref is not None
    image = cnd.reader("tgcr").image(ref.with_tag("9.9.9"), found, "amd64")
    assert image.problems[-1] == (
        f"{where} (the other catalog items could not all be read) lists no tag 9.9.9 of "
        f"{ref.repository_ref}"
    )


def test_a_tag_the_tgc_items_read_lack_is_not_listed_when_the_catalog_has_more_pages(
    catalogs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # the catalog counts more items than its pages read: an unread one may list the tag, so it
    # is not listed, with a note, never a plain "lists no tag"
    monkeypatch.setattr(cnd, "TGC_PAGES", 1)
    catalogs.update(tgc_routes())
    catalogs[f"{TGCR_HARBOR}/9.9.9"] = LOGIN
    every = fx("tgc-search-qwen3.6-27b.json")
    item = {"id": 100350, "name": "Other 100350"}
    every["data"]["items"].append(item)
    body = {"code": 200, "data": {**item, "image_tags": [OTHER_REPOSITORY_TAG]}}
    catalogs[f"{TGC_SEARCH}/100350"] = (200, body)
    every["data"]["total"] = len(every["data"]["items"]) + 1  # one item past the page read
    catalogs[f"{TGC_SEARCH}?page=1&size=100"] = (200, every)
    found = cnd.pick("tgcr", "qwen3.6-27b", cnd.search("tgcr", "qwen3.6-27b"))
    ref = rl.parse_image(found.image or "")
    assert ref is not None
    image = cnd.reader("tgcr").image(ref.with_tag("9.9.9"), found, "amd64")
    assert image.problems[-1] == (
        "the TGC catalog (the other catalog items could not all be read) lists no tag 9.9.9 of "
        f"{ref.repository_ref}"
    )


def test_tgc_tags_of_the_items_read_are_newest_first_whichever_item_lists_them() -> None:
    # the match lists 1.0, and another match of its search the newer 2.0 of the same repository
    repository = "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b"

    def item(name: str, tag: str, created: str, *hits: cnd.Found) -> cnd.Found:
        entry = cnd.CatalogTag(tag, f"{repository}:{tag}", created=created)
        return cnd.Found("tgcr", name, entry.image, tags=[entry], hits=hits)

    found = item("Qwen3.6-27B", "1.0", "2026-03-01", item("Qwen3.6-27B 2.0", "2.0", "2026-06-01"))
    ref = rl.parse_image(repository)
    assert ref is not None
    images, _ = cnd.TgcReader().images_behind_login(ref, found, "a login is needed")
    assert [image.ref.tag for image in images] == ["2.0", "1.0"]


def test_a_tgc_item_gives_an_image_only_the_tags_of_its_own_repository(
    catalogs: dict[str, Any],
) -> None:
    # an item may list the images of two repositories, such as a build and its DGX Spark build
    image = "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-35b-a3b"
    tags = [
        {
            "docker_url": f"{image}-spark:latest",
            "arch": "arm64",
            "create_time": "2026-07-15 20:03:49",
            "digest": "sha256:" + "1" * 64,
        },
        {
            "docker_url": f"{image}:1.0",
            "arch": "amd64",
            "create_time": "2026-04-18 22:59:29",
            "digest": "sha256:" + "2" * 64,
        },
    ]
    item = {"id": 100370, "name": "Qwen3.6-35B-A3B"}
    harbor = "/api/v2.0/projects/public/repositories/qwen%252Fqwen3.6-35b-a3b/artifacts"
    catalogs.update(
        {
            TGC_SEARCH: (200, {"code": 200, "data": {"items": [item], "total": 1}}),
            f"{TGC_SEARCH}/100370": (200, {"code": 200, "data": {**item, "image_tags": tags}}),
            harbor: LOGIN,
            f"{harbor}/latest": LOGIN,
        }
    )
    reader = cnd.reader("tgcr")
    found = cnd.pick("tgcr", "qwen3.6-35b-a3b", cnd.search("tgcr", "qwen3.6-35b-a3b"))
    ref = rl.parse_image(found.image or "")
    assert ref is not None and ref.image == f"{image}:1.0"
    images, _ = reader.images(ref.with_tag(None), found, "amd64")
    assert [listed.ref.image for listed in images] == [f"{image}:1.0"]
    # latest is a tag of the Spark build: its digest and arch are not this image's
    latest = reader.image(ref.with_tag("latest"), found, "amd64")
    assert (latest.digest, latest.architectures) == (None, ())
    assert "lists no tag latest" in latest.problems[-1]


def test_a_login_on_lichan_leaves_the_image_unknown(catalogs: dict[str, Any]) -> None:
    catalogs.update({f"{LICHAN_QWEN}/1.0.0": LOGIN, LICHAN_QWEN: LOGIN})
    ref = rl.parse_image("io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0")
    assert ref is not None
    reader = cnd.reader("lichan")
    image = reader.image(ref, None, None)
    assert (image.version, image.catalog_ngc_url) == (None, None)
    assert "needs a login" in image.problems[0]
    assert reader.images(ref, None, None) == ([], image.problems[0])


def test_the_lichan_reader_reuses_the_tags_its_search_read(catalogs: dict[str, Any]) -> None:
    catalogs.update(lichan_routes())
    reader = cnd.reader("lichan")
    found = cnd.pick("lichan", "qwen3.5-35b-a3b", cnd.search("lichan", "qwen3.5-35b-a3b"))
    ref = rl.parse_image(found.image or "")
    assert ref is not None
    before = len(requests_of(catalogs))
    images, problem = reader.images(ref, found, "amd64")
    assert problem is None and len(requests_of(catalogs)) == before
    assert images[0].version == "1.7.0-variant"


def test_the_lichan_reader_reads_a_multi_arch_index_again_for_the_host(
    catalogs: dict[str, Any],
) -> None:
    # the search reads the labels of an index's first image; the host's image may differ
    amd64, arm64 = "sha256:" + "a" * 64, "sha256:" + "b" * 64
    index = {
        "digest": "sha256:" + "c" * 64,
        "references": [
            {"child_digest": amd64, "platform": {"architecture": "amd64", "os": "linux"}},
            {"child_digest": arm64, "platform": {"architecture": "arm64", "os": "linux"}},
        ],
        "tags": [{"name": "1.0.0"}, {"name": "latest"}],
    }

    def image(arch: str, version: str) -> dict[str, Any]:
        labels = {rl.VERSION_LABEL: version}
        return {"extra_attrs": {"architecture": arch, "config": {"Labels": labels}}}

    catalogs.update(
        {
            LICHAN_REPOS: (200, fx("lichan-repositories.json")),
            LICHAN_QWEN: (200, [index]),
            f"{LICHAN_QWEN}/{amd64}": (200, image("amd64", "1.7.0-variant")),
            f"{LICHAN_QWEN}/{arm64}": (200, image("arm64", "1.7.1-variant")),
        }
    )
    found = cnd.pick("lichan", "qwen3.5-35b-a3b", cnd.search("lichan", "qwen3.5-35b-a3b"))
    assert found.listing is not None and found.listing[1][0].version == "1.7.0-variant"
    ref = rl.parse_image(found.image or "")
    assert ref is not None
    images, _ = cnd.reader("lichan").images(ref, found, "arm64")
    assert images[0].version == "1.7.1-variant"


def test_a_listed_image_says_why_its_labels_were_not_read(
    catalogs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    child = "sha256:" + "a" * 64
    index = {
        "digest": "sha256:" + "c" * 64,
        "references": [
            {"child_digest": child, "platform": {"architecture": "amd64", "os": "linux"}},
        ],
        "tags": [{"name": "1.0.0"}],
    }
    catalogs.update({LICHAN_QWEN: (200, [index]), f"{LICHAN_QWEN}/{child}": (503, {})})
    ref = rl.parse_image("io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b")
    assert ref is not None
    reader = cnd.reader("lichan")
    shown = ref.with_tag("1.0.0").image
    # the index's image answered 503: that is why, not the index limit
    images, _ = reader.images(ref, None, "amd64")
    assert images[0].problems == [
        f"the labels of {shown} were not read (its image could not be read)"
    ]
    monkeypatch.setattr(rl, "MAX_CHILDREN", 0)  # past the limit: its image is not asked for
    images, _ = reader.images(ref, None, "amd64")
    assert images[0].problems == [f"the labels of {shown} were not read (index limit)"]


@pytest.mark.parametrize("tag", [None, "1.0"])
def test_a_repository_a_cnd_lacks_offers_the_other_cnds(
    catalogs: dict[str, Any], tag: str | None
) -> None:
    # as Harbor answers: no tags (200 []) for the list, a 404 naming the repository for a tag
    missing = f"{LICHAN_REPOS}/qwen%252Fqwen3.6-27b/artifacts"
    catalogs.update(
        {
            missing: (200, []),
            f"{missing}/1.0": (404, fx("lichan-404-repository.json")),
            LICHAN_REPOS: (200, fx("lichan-repositories.json")),
        }
    )
    ref = rl.parse_image("io.chancloud.com/cnd/qwen/qwen3.6-27b", tag)
    assert ref is not None
    reader = cnd.reader("lichan")
    with pytest.raises(ns.ScriptError) as caught:
        if tag:
            reader.image(ref, None, None)
        else:
            reader.images(ref, None, None)
    error = caught.value
    assert (error.code, error.exit_code) == ("model_not_found", ns.EXIT_PRECONDITION)
    assert "has no repository io.chancloud.com/cnd/qwen/qwen3.6-27b" in error.message
    assert error.facts["other_sources"] == ["tgcr", "turingcm"]
    assert "TGC (--source tgcr) or TuringCM (--source turingcm)" in error.fix
    # another CND is searched by the name: the image names Lichan's host (source_conflict)
    assert error.facts["query"] == "qwen3.6-27b"
    assert "with the model name facts.query, never the image" in error.fix


def test_a_tag_a_cnd_lacks_is_tag_not_found(catalogs: dict[str, Any]) -> None:
    catalogs[f"{LICHAN_QWEN}/9.9.9"] = (404, fx("lichan-404.json"))
    ref = rl.parse_image("io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:9.9.9")
    assert ref is not None
    with pytest.raises(ns.ScriptError) as caught:
        cnd.reader("lichan").image(ref, None, None)
    assert (caught.value.code, caught.value.exit_code) == ("tag_not_found", ns.EXIT_PRECONDITION)


def test_a_tgcr_image_is_found_in_the_catalog_by_its_repository_not_its_name(
    catalogs: dict[str, Any],
) -> None:
    catalogs.update(glm_routes("1.0", "9.9"))
    ref = rl.parse_image(GLM51)
    assert ref is not None
    reader = cnd.reader("tgcr")
    image = reader.image(ref, None, "amd64")
    assert (image.catalog_ngc_url, image.architectures) == (
        "nvcr.io/nim/zai-org/glm-51:1.8.0-turbo",
        ("amd64",),
    )
    sent = requests_of(catalogs)
    assert f"{TGC_SEARCH}?page=1&size=100&search=glm5_1" in sent
    assert f"{TGC_SEARCH}?page=1&size=100" in sent  # every item: no name holds glm5_1
    # the repository alone (select after resolve): its one tag, from the catalog
    images, problem = reader.images(ref.with_tag(None), None, "amd64")
    assert problem is None and [item.ref.image for item in images] == [GLM51]
    # a tag, or a repository, no item lists: the whole catalog was read, so it says so
    other = reader.image(ref.with_tag("9.9"), None, "amd64")
    assert "the TGC catalog lists no tag 9.9" in other.problems[-1]
    missing = rl.parse_image("tgcr-gz.turing-agi.com/public/z-ai/glm9")
    assert missing is not None
    images, problem = reader.images(missing, None, "amd64")
    assert images == [] and problem and problem.endswith("the TGC catalog lists no image of it")


def test_a_tgc_entry_without_an_arch_takes_that_of_the_same_image(
    catalogs: dict[str, Any],
) -> None:
    # TGC lists the Qwen3.6-35B-A3B DGX Spark image under two items: one gives no arch, one arm64
    spark = "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-35b-a3b-spark"
    harbor = "/api/v2.0/projects/public/repositories/qwen%252Fqwen3.6-35b-a3b-spark/artifacts"
    items = [
        {"id": 100336, "name": "Qwen3.6-35B-A3B-DGX-Spark"},
        {"id": 100360, "name": "Qwen3.6-35B-A3B NIM Container for DGX Spark"},
    ]
    catalogs.update(
        {
            TGC_SEARCH: (200, fx("tgc-search-glm5_1.json")),  # no name holds the path
            f"{TGC_SEARCH}?page=1&size=100": (200, {"code": 200, "data": {"items": items}}),
            f"{TGC_SEARCH}/100336": (200, fx("tgc-detail-qwen3.6-35b-a3b-dgx-spark.json")),
            f"{TGC_SEARCH}/100360": (200, fx("tgc-detail-qwen3.6-35b-a3b-spark-container.json")),
            harbor: LOGIN,
        }
    )
    [entry] = cnd.tgc_tags(fx("tgc-detail-qwen3.6-35b-a3b-dgx-spark.json")["data"])
    assert entry.arch is None
    [other] = cnd.tgc_tags(fx("tgc-detail-qwen3.6-35b-a3b-spark-container.json")["data"])
    [merged] = cnd.merge_tags([entry, other])  # one entry per image, with the arch it has
    assert (merged.image, merged.arch) == (entry.image, "arm64")
    ref = rl.parse_image(spark)
    assert ref is not None
    images, _ = cnd.reader("tgcr").images(ref, None, "amd64")
    [image] = images  # the two entries of one image (one digest) are one
    assert (image.ref.tag, image.architectures) == ("latest", ("arm64",))


def test_a_tgc_entry_without_an_ngc_url_takes_that_of_the_same_image_and_digest() -> None:
    image = "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-35b-a3b-spark:latest"
    url, digest = "nvcr.io/nim/qwen/qwen3.6-35b-a3b:1.7.0-variant", "sha256:" + "1" * 64
    newer = cnd.CatalogTag("latest", image, None, "arm64", digest, "2026-07-15 20:03:49")
    older = cnd.CatalogTag("latest", image, url, "arm64", digest, "2026-04-18 22:59:29")
    [merged] = cnd.merge_tags([newer, older])  # the newer entry is kept, with the mapping
    assert (merged.ngc_url, merged.created) == (url, newer.created)
    # another digest is another build: it lends no mapping
    other = cnd.CatalogTag("latest", image, url, "arm64", "sha256:" + "2" * 64, older.created)
    assert [entry.ngc_url for entry in cnd.merge_tags([newer, other])] == [None, url]


def test_a_turingcm_item_with_a_dgx_spark_build_first_takes_the_image_named(
    catalogs: dict[str, Any],
) -> None:
    # llama-3.1-8b-instruct: its deploy text names the arm64 DGX Spark build first (real)
    catalogs.update(
        {
            f"{TURINGCM}/simple-list": (200, fx("turingcm-simple-list.json")),
            f"{TURINGCM}/getNimImagesById?id=5": (
                200,
                fx("turingcm-detail-llama-3.1-8b-instruct.json"),
            ),
        }
    )
    spark = "resource.turingcm.com/megacube/llama-3.1-8b-instruct-dgx-spark:latest"
    result = cnd.search("turingcm", "llama-3.1-8b-instruct")
    assert [item.image for item in result.found] == [
        spark,
        "resource.turingcm.com/nim/llama-3.1-8b-instruct:latest",
    ]
    found = cnd.pick("turingcm", "llama-3.1-8b-instruct", result)
    # of the images one item names, the one whose repository is the name; the other is named
    assert (found.image, found.also) == (
        "resource.turingcm.com/nim/llama-3.1-8b-instruct:latest",
        (spark,),
    )


def test_a_turingcm_item_naming_several_images_none_by_its_name_asks(
    catalogs: dict[str, Any],
) -> None:
    # gemma-4-26b names gemma-4-26b-a4b and its Spark build: neither repository is the name
    deploy = (
        "<pre>resource.turingcm.com/google/gemma-4-26b-a4b:latest</pre>\n"
        "<pre>resource.turingcm.com/google/gemma-4-26b-a4b-spark:latest</pre>"
    )
    records = [{"id": "-27", "name": "gemma-4-26b"}]
    catalogs.update(
        {
            f"{TURINGCM}/simple-list": (200, {"code": 200, "data": {"records": records}}),
            f"{TURINGCM}/getNimImagesById?id=-27": (200, {"code": 200, "data": {"deploy": deploy}}),
        }
    )
    with pytest.raises(ns.ScriptError) as caught:
        cnd.pick("turingcm", "gemma-4-26b", cnd.search("turingcm", "gemma-4-26b"))
    error = caught.value
    assert error.code == "model_ambiguous"
    assert error.message.startswith("2 NIMs in the TuringCM catalog match 'gemma-4-26b'")
    assert [item["repository"] for item in error.facts["candidates"]] == [
        "resource.turingcm.com/google/gemma-4-26b-a4b",
        "resource.turingcm.com/google/gemma-4-26b-a4b-spark",
    ]


def test_an_exact_name_is_looked_up_past_max_matches(catalogs: dict[str, Any]) -> None:
    names = [f"qwen3-variant-{c}" for c in "abcdefghijk"] + ["Qwen3"]
    records = [{"id": str(i), "name": name} for i, name in enumerate(names)]
    catalogs[f"{TURINGCM}/simple-list"] = (200, {"code": 200, "data": {"records": records}})
    for i, name in enumerate(names):
        deploy = f"<pre>resource.turingcm.com/qwen/{name.lower()}:1.0</pre>"
        route = f"{TURINGCM}/getNimImagesById?id={i}"
        catalogs[route] = (200, {"code": 200, "data": {"deploy": deploy}})
    result = cnd.search("turingcm", "qwen3")
    assert (result.total, len(result.found)) == (12, cnd.MAX_MATCHES)
    assert cnd.pick("turingcm", "qwen3", result).image == "resource.turingcm.com/qwen/qwen3:1.0"


def test_tgc_pages_are_read_while_the_total_says_more(catalogs: dict[str, Any]) -> None:
    # a catalog that answers 20 items a page: the match is on page 2
    filler = [{"id": 1000 + i, "name": f"Qwen-other-{i}"} for i in range(20)]
    match = fx("tgc-search-qwen3.6-27b.json")["data"]["items"][:1]
    page = f"{TGC_SEARCH}?page={{}}&size=100&search=qwen"
    catalogs.update(
        {
            page.format(1): (200, {"code": 200, "data": {"items": filler, "total": 21}}),
            page.format(2): (200, {"code": 200, "data": {"items": match, "total": 21}}),
            f"{TGC_SEARCH}/100341": (200, fx("tgc-detail-qwen3.6-27b.json")),
        }
    )
    assert (
        cnd.pick("tgcr", "qwen 3.6 27b", cnd.search("tgcr", "qwen 3.6 27b")).name == "Qwen3.6-27B"
    )
    # without a total, a full page of 100 asks for the next one
    catalogs[page.format(1)] = (200, {"code": 200, "data": {"items": filler * 5}})
    catalogs[page.format(2)] = (200, {"code": 200, "data": {"items": match}})
    assert cnd.search("tgcr", "qwen 3.6 27b").found[0].name == "Qwen3.6-27B"


def test_a_tgc_search_past_its_pages_asks_for_the_full_image(catalogs: dict[str, Any]) -> None:
    # 300 items have the first word, and the total says more: an item past them may be the one
    # asked for, or have the name of one read, so the search is neither model_not_found nor the
    # one exact name read, and no detail is asked for. The full image is found by its repository,
    # in the items read.
    catalogs.update(tgc_routes())
    match = fx("tgc-search-qwen3.6-27b.json")["data"]["items"][:1]  # Qwen3.6-27B
    for word in ("qwen", "qwen3.6-27b"):
        for number in range(1, cnd.TGC_PAGES + 1):
            filler = [
                {"id": number * 1000 + i, "name": f"Qwen-other-{number}-{i}"} for i in range(100)
            ]
            items = (match + filler)[:100] if number == 1 else filler
            body = {"code": 200, "data": {"items": items, "total": 301}}
            catalogs[f"{TGC_SEARCH}?page={number}&size=100&search={word}"] = (200, body)
    for query, read in (
        ("qwen xyz", "none of those read is named 'qwen-xyz'"),
        ("qwen3.6-27b", "one past them may be named 'qwen3.6-27b' too (read: Qwen3.6-27B)"),
    ):
        with pytest.raises(ns.ScriptError) as caught:
            cnd.search("tgcr", query)
        error = caught.value
        assert (error.code, error.exit_code) == ("cnd_unreachable", ns.EXIT_EXTERNAL)
        assert error.message.startswith(
            "the TGC catalog (tgc.turing-agi.com) was not searched in full: it lists more items "
            f"with {query.split()[0]!r} in their name than the 300 a search reads"
        )
        assert error.message.endswith(read)
        assert error.fix.startswith("Do not retry: a search reads the same first 300 items")
        assert "full image" in error.fix
    assert not any(f"{TGC_SEARCH}/" in path for path in requests_of(catalogs))
    catalogs.update({TGCR_HARBOR: LOGIN, f"{TGCR_HARBOR}/latest": LOGIN})
    ref = rl.parse_image(TGCR_LATEST)
    assert ref is not None
    image = cnd.reader("tgcr").image(ref, None, "amd64")
    assert image.catalog_ngc_url == "nvcr.io/nim/qwen/qwen3.6-27b:1.7.0-variant"


def test_a_tgc_catalog_longer_than_its_pages_is_listed_in_part(
    catalogs: dict[str, Any],
) -> None:
    filler = [{"id": 1000 + i, "name": f"Other-{i}"} for i in range(100)]
    page = f"{TGC_SEARCH}?page={{}}&size=100"

    def listed(total: int) -> cnd.Catalog:
        for number in range(1, cnd.TGC_PAGES + 1):
            body = {"code": 200, "data": {"items": filler, "total": total}}
            catalogs[page.format(number)] = (200, body)
        return cnd.reader("tgcr").list_all(lambda name, kind: False)  # no detail request

    # the catalog counts 301 items: the last one is not checked, so the list is partial; it is
    # no item that could not be read (Catalog.failed), which would make the list exit 4
    catalog = listed(301)
    assert catalog.unread == "it lists more than the 300 items of its first 3 pages of 100"
    assert catalog.failed == []
    assert len(requests_of(catalogs)) == cnd.TGC_PAGES
    # 300 items: the pages hold them all
    assert listed(300).unread is None


def full_lichan_pages(catalogs: dict[str, Any]) -> None:
    """5 full pages of 100 repositories: the project may list more, which are not read."""
    for page in range(1, rl.REPO_PAGES + 1):
        names = [{"name": f"cnd/team/repo-{page}-{i}"} for i in range(100)]
        catalogs[f"{LICHAN_REPOS}?page_size=100&page={page}"] = (200, names)


def test_a_lichan_project_past_its_500_repositories_is_listed_in_part(
    catalogs: dict[str, Any],
) -> None:
    full_lichan_pages(catalogs)
    catalog = cnd.reader("lichan").list_all(lambda name, kind: False)
    assert catalog.unread == (
        "it may list more than the 500 repositories of its first 5 pages of 100"
    )
    assert catalog.failed == [] and len(catalog.left_out) == 500


def test_a_lichan_search_past_its_500_repositories_asks_for_the_full_image(
    catalogs: dict[str, Any],
) -> None:
    # a repository past them may be the one asked for, or have the name of one read: never
    # model_not_found (which offers the other CNDs), nor a match in part (o-5-99 is part of
    # team/repo-5-99), nor the one exact name read (team-b/repo-5-99, past the pages read, makes
    # repo-5-99 ambiguous)
    full_lichan_pages(catalogs)
    catalogs[f"{LICHAN_REPOS}?page_size=100&page=6"] = (200, [{"name": "cnd/team-b/repo-5-99"}])
    for query, read in (
        ("no-such-model", "none of those read is named 'no-such-model'"),
        ("o-5-99", "none of those read is named 'o-5-99'"),
        ("repo-5-99", "one past them may be named 'repo-5-99' too (read: team/repo-5-99)"),
    ):
        with pytest.raises(ns.ScriptError) as caught:
            cnd.pick("lichan", query, cnd.search("lichan", query))
        error = caught.value
        assert (error.code, error.exit_code) == ("cnd_unreachable", ns.EXIT_EXTERNAL)
        assert "only the first 500 repositories of project 'cnd'" in error.message
        assert error.message.endswith(read)
        # "Do not retry" is the mark SKILL.md reads as no retry for cnd_unreachable
        assert error.fix.startswith("Do not retry: a search reads the same first 500")
        assert "full image" in error.fix
    sent = requests_of(catalogs)
    assert not any("/artifacts" in path or "page=6" in path for path in sent)  # nothing more


@pytest.mark.parametrize("value", [5, True, "x", {"docker_url": "a"}, None])
def test_tgc_image_tags_that_are_no_list_name_no_image(value: object) -> None:
    assert cnd.tgc_tags({"image_tags": value}) == []


def test_tgc_tags_keep_only_tagged_images_on_tgcr() -> None:
    detail = fx("tgc-detail-qwen3.6-27b.json")["data"]
    good = detail["image_tags"][0]
    left_out = [
        {**good, "docker_url": "io.chancloud.com/cnd/qwen/qwen3.6-27b:9.9"},  # another CND
        {**good, "docker_url": "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b:1.0;curl x|sh"},
        {**good, "docker_url": "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b"},  # no tag
    ]
    tags = cnd.tgc_tags({**detail, "image_tags": [*left_out, good]})
    assert [tag.image for tag in tags] == [good["docker_url"]]


def test_an_odd_catalog_answer_is_cnd_unreachable_never_a_traceback(
    catalogs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    catalogs.update(tgc_routes())
    real = cnd.tgc_tags

    def odd_once(detail: dict[str, Any]) -> list[cnd.CatalogTag]:
        if detail.get("id") == 100339:
            raise TypeError("odd")
        return real(detail)

    monkeypatch.setattr(cnd, "tgc_tags", odd_once)
    found = cnd.search("tgcr", "qwen3.6-27b").found  # that match alone is left out
    assert sorted(item.name for item in found) == ["Qwen3.6-27B", "Qwen3.6-27B-FP8-DGX-Spark"]
    monkeypatch.setattr(cnd, "tgc_tags", lambda detail: 1 / 0)
    with pytest.raises(ns.ScriptError) as caught:
        cnd.search("tgcr", "qwen3.6-27b")
    assert caught.value.code == "cnd_unreachable"
    monkeypatch.setattr(cnd.TgcReader, "search", lambda self, words: {}["items"])
    with pytest.raises(ns.ScriptError) as caught:
        cnd.search("tgcr", "qwen3.6-27b")
    assert caught.value.code == "cnd_unreachable" and "KeyError" in caught.value.message


def test_a_detail_answered_without_data_is_a_failed_lookup(catalogs: dict[str, Any]) -> None:
    catalogs.update(tgc_routes())
    for item_id in ("100341", "100339", "100340"):
        catalogs[f"{TGC_SEARCH}/{item_id}"] = (200, {"code": 500, "data": None})
    with pytest.raises(ns.ScriptError) as caught:
        cnd.search("tgcr", "qwen3.6-27b")
    # never model_not_found: the catalog did not say what the items are
    assert caught.value.code == "cnd_unreachable"
    assert "without its details (code 500)" in caught.value.message


@pytest.mark.parametrize(
    ("source", "route"),
    [
        ("tgcr", TGC_SEARCH),
        ("turingcm", f"{TURINGCM}/simple-list"),
        ("lichan", LICHAN_REPOS),
    ],
)
def test_a_catalog_that_needs_a_login_is_not_retried(
    catalogs: dict[str, Any], source: str, route: str
) -> None:
    catalogs[route] = LOGIN
    with pytest.raises(ns.ScriptError) as caught:
        cnd.search(source, "qwen3.6-27b")
    error = caught.value
    assert (error.code, error.exit_code) == ("cnd_unreachable", ns.EXIT_EXTERNAL)
    assert "needs a login, which nim-select never does" in error.message
    assert error.fix.startswith("Do not retry") and "--nim-version" in error.fix
    assert "list-model-profiles" in error.fix
    # one request: the answer to a second one would be the same 401
    assert [path.split("?", 1)[0] for path in requests_of(catalogs)] == [route]


def test_run_directly_prints_a_notice() -> None:
    code, payload = run_cli("cnd.py")
    assert (code, payload["state"]) == (0, "library")


# --- live (opt-in: NIM_SELECT_LIVE=1 pytest -m network) -----------------------------------------

live = pytest.mark.skipif(
    os.environ.get("NIM_SELECT_LIVE") != "1", reason="live test; set NIM_SELECT_LIVE=1"
)


@pytest.fixture
def real_catalogs(monkeypatch: pytest.MonkeyPatch) -> None:
    use_built_in_endpoints(monkeypatch)
    monkeypatch.setattr(cnd, "SEARCH_BUDGET", 90.0)
    monkeypatch.setattr(rl, "LIST_BUDGET", 60.0)


@pytest.mark.network
@live
@pytest.mark.usefixtures("real_catalogs")
@pytest.mark.parametrize(
    ("source", "query", "repository"),
    [
        ("tgcr", "qwen3.6-27b", "public/qwen/qwen3.6-27b"),
        ("turingcm", "qwen3.6-27b", "qwen/qwen3.6-27b"),
        ("lichan", "qwen3.5-35b-a3b", "cnd/qwen/qwen3.5-35b-a3b"),
    ],
)
def test_live_catalog_search(source: str, query: str, repository: str) -> None:
    """One anonymous catalog search on each China NIM distributor."""
    found = cnd.pick(source, query, cnd.search(source, query))
    ref = rl.parse_image(found.image or "")
    assert ref is not None and f"{ref.project}/{ref.repository}" == repository


@pytest.mark.network
@live
@pytest.mark.usefixtures("real_catalogs")
def test_live_tgcr_image_found_by_its_repository() -> None:
    """GLM-5.1 on TGC: no catalog name holds its path, z-ai/glm5_1, so every item is read."""
    ref = rl.parse_image("tgcr-gz.turing-agi.com/public/z-ai/glm5_1:1.0")
    assert ref is not None
    image = cnd.reader("tgcr").image(ref, None, "amd64")
    assert image.catalog_ngc_url
    assert image.catalog_ngc_url.startswith("nvcr.io/nim/zai-org/glm-51:")
