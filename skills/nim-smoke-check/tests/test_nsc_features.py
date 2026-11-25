# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/nsc_features.py and scripts/model_features.json: table, matching, plans."""

from __future__ import annotations

import copy
import json
import re
import shlex
import subprocess
import sys
import urllib.parse
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import nsc_common as nsc  # noqa: E402
import nsc_features as nf  # noqa: E402

SKILL = SCRIPTS.parent
TABLE = nf.load_table()
ALLOWED_HOSTS = {
    "docs.nvidia.com", "docs.vllm.ai", "docs.sglang.io", "docs.sglang.ai", "huggingface.co",
    "build.nvidia.com", "docs.api.nvidia.com", "catalog.ngc.nvidia.com", "api.ngc.nvidia.com",
}
GITHUB_ORGS = {"vllm-project", "sgl-project", "NVIDIA"}
# Built from parts so that this file does not name them itself.
PRIVATE = [
    "git" + "lab", "nv" + "staging", "nim-" + "internal", "nim" + "lib", "nim-" + "runtime",
    "nim-inference-" + "sdk", "git" + "lab-master",
]
# The private nim-llm repository; "about-nim-llm" is a path of the public docs site.
PRIVATE_REPO = re.compile(r"(?<![a-z-])" + "nim-" + r"llm(?![a-z-])")
PHASE1 = [
    "nim " + "cli", "nim" + "cli", "lean" + "tune", "infer" + "fit", "write-" + "skill",
    "infer" + "fit-model-ops", "nim" + "cli-model-ops", "nim-deploy-" + "cli",
    "nim-select-" + "cli",
]
REQUIRED_FAMILIES = {
    "llama-3.1-instruct", "llama-3.2-instruct", "llama-3.3-70b-instruct", "llama-3.2-vision",
    "llama-nemotron-v1", "llama-3.3-nemotron-super-49b-v1.5", "nemotron-nano-9b-v2",
    "nemotron-nano-12b-v2-vl", "nemotron-3", "nemotron-3.5-super-vl", "qwen3",
    "qwen3-next-thinking", "qwen3.5-vlm", "qwen2.5", "gemma-3-1b", "gemma-4", "gpt-oss",
    "mistral-7b-v0.3", "mistral-small-24b-2501", "mistral-nemo-12b",
    "mistral-small-3.2-24b-2506", "mistral-small-4", "deepseek-r1", "deepseek-v3.1-v3.2",
    "deepseek-v4-pro", "deepseek-v4.1-flash", "glm-5.2", "glm-5.3-flash", "kimi-k2",
    "kimi-k3.1-pro", "inkling", "phi-4-mini", "cosmos-reason2", "nemotron-parse",
    "safety-classifiers", "muse-glimmer", "nemotron-3.5-bolt-e4b", "step-3.7-flash",
    "ministral-3-14b",
}


def fake_key() -> str:
    return "nvapi" + "-" + "q" * 30


# --- helpers ----------------------------------------------------------------------------


def all_sources(table: dict):
    """(where, source) for every cited source, overrides and conflicts included."""
    for key, entry in table["engine"].items():
        for source in entry.get("sources", []):
            yield f"engine.{key}", source
    for family in table["families"]:
        objects = [(family["id"], family)]
        objects += [(f"{family['id']}/{o.get('name')}", o) for o in family.get("overrides", [])]
        for label, obj in objects:
            for feature in nf.FEATURE_KEYS:
                for list_name in ("sources", "conflicts"):
                    for source in (obj.get(feature) or {}).get(list_name, []) or []:
                        yield f"{label}.{feature}.{list_name}", source
            for source in obj.get("out_of_scope_sources", []) or []:
                yield f"{label}.out_of_scope", source


def all_urls(table: dict):
    for where, source in all_sources(table):
        yield where, source["url"]
    for family in table["families"]:
        issues = list(family.get("known_issues", []))
        for override in family.get("overrides", []):
            issues += override.get("known_issues", [])
        for issue in issues:
            yield f"{family['id']}.known_issues", issue["url"]


def plan_for(served_id=None, image=None, family=None, **kwargs):
    record, how = nf.match(TABLE, image=image, served_id=served_id, family=family)
    plan = nf.decide(record, table=TABLE, image=image, **kwargs)
    return plan, how


def entry(plan: dict, check: str) -> dict:
    found = nf.plan_entry(plan, check)
    assert found is not None, f"{check} not in the plan"
    return found


def write_table(tmp_path: Path, table: dict) -> Path:
    path = tmp_path / "model_features.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    return path


def family(table: dict, fid: str) -> dict:
    return next(f for f in table["families"] if f["id"] == fid)


def override_candidates(fam: dict, override: dict) -> list[str]:
    """Normalized ids that select one override of a family: its ids, or for an override with
    patterns only, the served ids of MATCHES for the family that its patterns match. Such an
    override with no matching served id fails the test, so it cannot go unchecked."""
    ids = [nf.normalize_id(item) for item in override.get("ids", [])]
    if not ids:
        ids = [nf.normalize_id(served) for served, fid in MATCHES
               if fid == fam["id"] and nf._override_matches(override, [nf.normalize_id(served)])]
    if not ids:
        pytest.fail(f"{fam['id']}: add a served id that the override {override.get('name')!r} "
                    "matches to MATCHES")
    return ids


def resolve_override(fam: dict, override: dict, candidates: list[str]) -> dict:
    """The family record resolved for `candidates`, checked to have this override applied."""
    record = nf.resolve(fam, candidates)
    label = override.get("name") or override.get("ids", ["?"])[0]
    assert label in (record["override"] or ""), (fam["id"], label, record["override"])
    return record


# --- the table --------------------------------------------------------------------------


def test_nsc_features_table_loads_and_has_required_families():
    assert TABLE["schema_version"] == 1
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", TABLE["reviewed"])
    ids = [f["id"] for f in TABLE["families"]]
    assert len(ids) == len(set(ids))
    assert set(ids) >= REQUIRED_FAMILIES
    for fam in TABLE["families"]:
        assert {"id", "match", "status", "kind"} <= set(fam)
        assert fam["status"] in nf.STATUSES and fam["kind"] in nf.KINDS
    assert {f["id"] for f in TABLE["families"] if f["status"] == "out_of_scope"} == {
        "nemotron-parse", "safety-classifiers"
    }
    for key in ("chat", "stream", "json_schema", "health"):
        assert TABLE["engine"][key]["sources"], key


def test_nsc_features_sources_are_public_https():
    count = 0
    for where, url in all_urls(TABLE):
        count += 1
        parsed = urllib.parse.urlsplit(url)
        assert parsed.scheme == "https", (where, url)
        assert not parsed.username and not parsed.password, (where, url)
        host = parsed.hostname
        if host == "github.com":
            assert parsed.path.strip("/").split("/")[0] in GITHUB_ORGS, (where, url)
        else:
            assert host in ALLOWED_HOSTS, (where, url)
        lowered = url.lower()
        for word in [*PRIVATE, "localhost", "file:", "/home/", "/tmp/"]:
            assert word not in lowered, (where, url)
        assert not PRIVATE_REPO.search(lowered), (where, url)
    assert count > 150


def test_nsc_features_every_quote_is_short_and_present():
    for where, source in all_sources(TABLE):
        assert isinstance(source.get("quote"), str) and source["quote"].strip(), where
        assert len(source["quote"]) <= 300, where
        assert source.get("doc_version"), where
        assert source.get("tier") in nf.TIERS, where


def test_nsc_features_every_documented_yes_has_a_quoted_source():
    def quoted(obj, tiers=nf.TIERS):
        return any(s.get("quote") and s.get("tier") in tiers for s in obj.get("sources", []))

    for fam in TABLE["families"]:
        for override in [None, *fam.get("overrides", [])]:
            record = nf.resolve(fam, []) if override is None else resolve_override(
                fam, override, override_candidates(fam, override)[:1]
            )
            tools, image, reasoning = record["tools"], record["image"], record["reasoning"]
            if tools.get("documented") in ("yes", "conflict"):
                assert quoted(tools, ("A", "B")), fam["id"]
            if "yes" in (tools.get("parallel"), tools.get("stream")):
                assert quoted(tools), fam["id"]
            if image.get("documented") == "yes":
                assert quoted(image, ("A", "B")), fam["id"]
            if reasoning.get("default") not in ("none", None):
                assert quoted(reasoning), fam["id"]
            if record["status"] == "out_of_scope":
                assert record.get("out_of_scope_reason") and record.get("out_of_scope_sources")


def test_nsc_features_patterns_compile_anchored_and_ids_unique():
    owners: dict[str, str] = {}
    for fam in TABLE["families"]:
        patterns = list(fam["match"].get("patterns", []))
        for override in fam.get("overrides", []):
            patterns += override.get("patterns", [])
        for pattern in patterns:
            re.compile(pattern)
            assert pattern.startswith("^") and pattern.endswith("$"), pattern
        for exact in nf._exact_ids(fam):
            assert owners.setdefault(exact, fam["id"]) == fam["id"], exact


def test_nsc_features_no_pattern_matches_another_familys_ids():
    for fam in TABLE["families"]:
        for exact in nf._exact_ids(fam):
            for other in TABLE["families"]:
                if other is fam:
                    continue
                for pattern in other["match"].get("patterns", []):
                    assert not re.fullmatch(pattern, exact), (other["id"], pattern, exact)


def test_nsc_features_phase1_words_absent_from_own_files():
    files = [SCRIPTS / "nsc_features.py", SCRIPTS / "model_features.json"]
    reference = SKILL / "references" / "model-features.md"
    if reference.exists():
        files.append(reference)
    for path in files:
        text = path.read_text(encoding="utf-8").lower()
        for word in PHASE1 + PRIVATE:
            assert word not in text, (path.name, word)
        assert not PRIVATE_REPO.search(text), path.name


def test_nsc_features_table_lines_are_valid_json_and_reviewed_docs():
    raw = json.loads((SCRIPTS / "model_features.json").read_text(encoding="utf-8"))
    assert raw["docs"]["nim_llm"] == "2.1.5"
    assert raw["docs"]["vllm"] == "0.31.0"


@pytest.mark.parametrize(
    "change, needle",
    [
        (lambda t: t["families"][0]["tools"]["sources"][0].update(
            url="https://git" + "lab.example.com/nim/x"), "url"),
        (lambda t: t["families"][0]["tools"]["sources"][0].update(url="http://docs.nvidia.com/x"),
         "https"),
        (lambda t: t["families"][0]["tools"]["sources"][0].update(url="https://github.com/acme/x"),
         "github"),
        (lambda t: t["families"][0]["tools"]["sources"][0].update(quote=""), "quote"),
        (lambda t: t["families"][0]["tools"]["sources"][0].update(quote="x" * 301), "quote"),
        (lambda t: t["families"][0]["match"]["patterns"].append("meta/llama"), "anchored"),
        (lambda t: t["families"][0]["match"]["patterns"].append("^meta/llama-3\\.1$"), "lower"),
        (lambda t: t["families"][0]["match"]["patterns"].append("^(meta$"), "compile"),
        (lambda t: t["families"][1].update(id=t["families"][0]["id"]), "twice"),
        (lambda t: t["families"][0].update(status="retired"), "status"),
        (lambda t: t["families"][0].update(kind="audio"), "kind"),
        (lambda t: t["families"][0]["tools"].update(sources=[]), "tier A"),
        (lambda t: t["families"][0]["tools"].update(documented="maybe"), "documented"),
        (lambda t: t["families"][0]["reasoning"].update(default="sometimes"), "default"),
        (lambda t: t["families"][0]["match"]["patterns"].append("^openai/gpt-oss-20b$"),
         "also matches"),
        (lambda t: t.update(schema_version=2), "schema_version"),
        (lambda t: t["families"][0]["tools"].update(mode=[{"method": "kwargs",
                                                           "key": "chat_template",
                                                           "value": "x"}]), "chat_template"),
        # urlsplit raises on an unmatched bracket in the host: a table problem, not a crash
        (lambda t: t["families"][0]["tools"]["sources"][0].update(
            url="https://[docs.nvidia.com/x"), "not a valid url"),
        (lambda t: t["families"][0]["known_issues"][0].update(url="https://docs.nvidia.com]/x"),
         "not a valid url"),
        # resolve() passes each override id to normalize_id and joins the names as text
        (lambda t: t["families"][0]["overrides"].append({"ids": [42], "status": "eos"}),
         "ids must be"),
        (lambda t: t["families"][0]["overrides"].append(
            {"ids": "meta/llama-3.1-8b-instruct", "status": "eos"}), "ids must be"),
        (lambda t: t["families"][0]["overrides"].append({"ids": [" "], "status": "eos"}),
         "ids must be"),
        (lambda t: t["families"][0]["overrides"].append(
            {"ids": [], "patterns": ["^meta/llama-3-1-8b-instruct$"]}), "ids must be"),
        (lambda t: t["families"][0]["overrides"].append(
            {"ids": None, "patterns": ["^meta/llama-3-1-8b-instruct$"]}), "ids must be"),
        (lambda t: t["families"][0]["overrides"].append(
            {"name": 42, "ids": ["meta/llama-3.1-8b-instruct"]}), "name must be text"),
    ],
)
def test_nsc_features_invalid_table_is_refused(tmp_path, change, needle):
    table = copy.deepcopy(TABLE)
    change(table)
    with pytest.raises(nsc.ScriptError) as caught:
        nf.load_table(write_table(tmp_path, table))
    assert caught.value.code == "features_table_invalid"
    assert caught.value.exit_code == nsc.EXIT_PRECONDITION
    assert needle.lower() in caught.value.message.lower()


def test_nsc_features_check_source_url_reports_a_malformed_url():
    """A "[" or "]" in the host part is never a documentation host: a bracketed IP address
    parses on every Python, and is refused as well. A bracket after the host is fine."""
    for url in ("https://[docs.nvidia.com/x", "https://docs.nvidia.com]/x",
                "https://[docs.nvidia.com]/x", "https://[::ffff:1.2.3.4]/x", "https://[v1.fe]/x"):
        assert nf.check_source_url(url) == "is not a valid URL", url
    for url in ("https://docs.nvidia.com/x", "https://docs.nvidia.com/a[1]?b=[2]#c[3]"):
        assert nf.check_source_url(url) is None, url


def test_nsc_features_a_bracketed_host_name_is_refused_on_every_python(monkeypatch):
    """Python 3.10 before 3.10.16 and 3.11 before 3.11.4 read https://[docs.nvidia.com]/x as the
    host docs.nvidia.com, where newer releases raise ValueError. With urlsplit standing in for
    such a release, the link is still refused, by --documented as well."""
    def older_urlsplit(url, *args, **kwargs):
        scheme, _, rest = url.partition("://")
        netloc, _, path = rest.partition("/")
        return urllib.parse.SplitResult(scheme, netloc, "/" + path, "", "")

    monkeypatch.setattr(urllib.parse, "urlsplit", older_urlsplit)
    assert urllib.parse.urlsplit("https://[docs.nvidia.com]/x").hostname == "docs.nvidia.com"
    assert nf.check_source_url("https://[docs.nvidia.com]/x") == "is not a valid URL"
    with pytest.raises(nsc.ScriptError) as caught:
        nf.parse_documented("tools=https://[docs.nvidia.com]/x")
    assert (caught.value.code, caught.value.exit_code) == ("bad_input", nsc.EXIT_INPUT)


def test_nsc_features_pattern_only_override_is_accepted_and_applied(tmp_path):
    """An override may have patterns and no ids; it applies to the ids its patterns match."""
    table = copy.deepcopy(TABLE)
    family(table, "gpt-oss").setdefault("overrides", []).append(
        {"name": "120B only", "patterns": ["^openai/gpt-oss-120b$"], "status": "eos"})
    loaded = nf.load_table(write_table(tmp_path, table))
    record, _ = nf.match(loaded, served_id="openai/gpt-oss-120b")
    assert (record["override"], record["status"]) == ("120B only", "eos")
    record, _ = nf.match(loaded, served_id="openai/gpt-oss-20b")
    assert (record["override"], record["status"]) == (None, "supported")


def test_nsc_features_override_null_removes_sources_and_conflicts(tmp_path):
    """_merge: a null in an override removes the key, sources and conflicts too. The table check
    accepts {"sources": null}, so resolving it must not stop the plan; a list still adds to the
    family's sources, also when the family's own value is null."""
    table = copy.deepcopy(TABLE)
    terminus = family(table, "deepseek-v3.1-v3.2")
    source = copy.deepcopy(terminus["tools"]["sources"][0])
    terminus.setdefault("overrides", []).append(
        {"name": "null lists", "ids": ["deepseek-ai/deepseek-v3.1-terminus"],
         "tools": {"sources": None, "conflicts": None}})
    cosmos = family(table, "cosmos-reason2")
    cosmos["tools"]["sources"] = None
    cosmos.setdefault("overrides", []).append(
        {"name": "one source", "ids": ["nvidia/cosmos-reason2-8b"], "tools": {"sources": [source]}})
    loaded = nf.load_table(write_table(tmp_path, table))
    record, _ = nf.match(loaded, served_id="deepseek-ai/deepseek-v3.1-terminus")
    assert record["override"] == "null lists"
    assert "sources" not in record["tools"] and "conflicts" not in record["tools"]
    assert record["tools"]["documented"] == "conflict"  # only the lists were removed
    plan = nf.decide(record, table=loaded)
    assert entry(plan, "tools")["conflicts"] == []
    record, _ = nf.match(loaded, served_id="nvidia/cosmos-reason2-8b")
    assert record["tools"]["sources"] == [source]


def test_nsc_features_unreadable_or_bad_json_table(tmp_path):
    with pytest.raises(nsc.ScriptError) as caught:
        nf.load_table(tmp_path / "missing.json")
    assert caught.value.code == "features_table_invalid"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(nsc.ScriptError):
        nf.load_table(bad)


# --- ids and matching -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "image, expected",
    [
        ("nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.6", "nim/meta/llama-3.1-8b-instruct"),
        ("nvcr.io/nim/openai/gpt-oss-20b@sha256:" + "a" * 64, "nim/openai/gpt-oss-20b"),
        ("registry.example.com:5000/nim/qwen/qwen3.5-122b-a10b:2.1.6",
         "nim/qwen/qwen3.5-122b-a10b"),
        ("NVCR.IO/NIM/Meta/Llama-3.3-70B-Instruct", "nim/meta/llama-3.3-70b-instruct"),
        ("nim/nvidia/nemotron-3-nano:latest", "nim/nvidia/nemotron-3-nano"),
        ("nvcr.io/nvidia/pytorch:24.01", None),
        ("vllm/vllm-openai:latest", None),
        ("", None),
        (None, None),
    ],
)
def test_nsc_features_resource_id(image, expected):
    assert nf.resource_id(image) == expected


def test_nsc_features_image_tag_generation_and_normalize():
    assert nf.image_tag("nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.6") == "2.1.6"
    assert nf.image_tag("localhost:5000/nim/a/b") is None
    assert nf.generation_of("2.1.5") == "2.x"
    assert nf.generation_of("1.15.0") == "1.x"
    assert nf.generation_of("2.1.4-variant") == "2.x"
    assert nf.generation_of("1.x") == "1.x"
    assert nf.generation_of("latest") is None
    assert nf.generation_of("10.1") is None
    assert nf.normalize_id("nvidia/Llama-3_3-Nemotron-Super-49B-v1_5") == (
        "nvidia/llama-3-3-nemotron-super-49b-v1-5"
    )


MATCHES = [
    ("meta/llama-3.1-8b-instruct", "llama-3.1-instruct"),
    ("meta/llama-3.1-70b-instruct", "llama-3.1-instruct"),
    ("meta-llama/Llama-3.1-405B-Instruct", "llama-3.1-instruct"),
    ("meta/llama-3.2-1b-instruct", "llama-3.2-instruct"),
    ("meta/llama-3.3-70b-instruct", "llama-3.3-70b-instruct"),
    ("meta/llama-3.2-90b-vision-instruct", "llama-3.2-vision"),
    ("nvidia/llama-3.3-nemotron-super-49b-v1", "llama-nemotron-v1"),
    ("nvidia/llama-3.1-nemotron-ultra-253b-v1", "llama-nemotron-v1"),
    ("nvidia/llama3.1-nemotron-nano-4b-v1.1", "llama-nemotron-v1"),
    ("nvidia/llama-3.3-nemotron-super-49b-v1.5", "llama-3.3-nemotron-super-49b-v1.5"),
    ("nvidia/Llama-3_3-Nemotron-Super-49B-v1_5", "llama-3.3-nemotron-super-49b-v1.5"),
    ("nvidia/nvidia-nemotron-nano-9b-v2", "nemotron-nano-9b-v2"),
    ("nvidia/nemotron-nano-12b-v2-vl", "nemotron-nano-12b-v2-vl"),
    ("nvidia/nemotron-3-nano", "nemotron-3"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron-3"),
    ("nvidia/nemotron-3-super-120b-a12b", "nemotron-3"),
    ("nvidia/nemotron-3-ultra-550b-a55b", "nemotron-3"),
    ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "nemotron-3"),
    ("nvidia/nemotron-3.5-super-vl-120b-a12b", "nemotron-3.5-super-vl"),
    ("qwen/qwen3-32b", "qwen3"),
    ("Qwen/Qwen3-30B-A3B", "qwen3"),
    ("qwen/qwen3-next-80b-a3b-thinking", "qwen3-next-thinking"),
    ("qwen/qwen3.5-122b-a10b", "qwen3.5-vlm"),
    ("qwen/qwen3.5-397b-a17b", "qwen3.5-vlm"),
    ("qwen/qwen3.6-35b-a3b", "qwen3.5-vlm"),
    ("qwen/qwen3.8-27b", "qwen3.5-vlm"),
    ("qwen/qwen-2.5-7b-instruct", "qwen2.5"),
    ("qwen/qwen2.5-coder-32b-instruct", "qwen2.5"),
    ("google/gemma-3-1b-it", "gemma-3-1b"),
    ("google/gemma-4-31b-it", "gemma-4"),
    ("google/gemma-4-26b-a4b-it", "gemma-4"),
    ("openai/gpt-oss-20b", "gpt-oss"),
    ("openai/gpt-oss-120b", "gpt-oss"),
    ("mistralai/mistral-7b-instruct-v0.3", "mistral-7b-v0.3"),
    ("mistralai/mistral-small-24b-instruct-2501", "mistral-small-24b-2501"),
    ("nv-mistralai/mistral-nemo-12b-instruct", "mistral-nemo-12b"),
    ("mistralai/mistral-small-3.2-24b-instruct-2506", "mistral-small-3.2-24b-2506"),
    ("mistralai/mistral-small-4-119b-2603", "mistral-small-4"),
    ("deepseek-ai/deepseek-r1", "deepseek-r1"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "deepseek-r1"),
    ("deepseek-ai/deepseek-r1-distill-llama-8b", "deepseek-r1"),
    ("deepseek-ai/deepseek-v3.1-terminus", "deepseek-v3.1-v3.2"),
    ("deepseek-ai/DeepSeek-V4-Pro-0813", "deepseek-v4-pro"),
    ("deepseek-ai/deepseek-v4.1-flash", "deepseek-v4.1-flash"),
    ("zai-org/GLM-5.2", "glm-5.2"),
    ("zai-org/glm-5.3-flash", "glm-5.3-flash"),
    ("moonshotai/kimi-k2.6", "kimi-k2"),
    ("moonshotai/kimi-k2.5", "kimi-k2"),
    ("moonshotai/Kimi-K3.1-Pro", "kimi-k3.1-pro"),
    ("thinkingmachines/Inkling-NVFP4", "inkling"),
    ("thinkingmachines/inkling-small", "inkling"),
    ("microsoft/phi-4-mini-instruct", "phi-4-mini"),
    ("nvidia/cosmos-reason2-8b", "cosmos-reason2"),
    ("nvidia/nemotron-parse-v2.0", "nemotron-parse"),
    ("nvidia/nemoretriever-parse", "nemotron-parse"),
    ("nvidia/nemotron-3.5-content-safety", "safety-classifiers"),
    ("nvidia/llama-3.1-nemoguard-8b-content-safety", "safety-classifiers"),
    # 2.1.5 support-matrix VLMs and VLM 1.7.0 models (table-4)
    ("meta/muse-glimmer", "muse-glimmer"),
    ("nvidia/nemotron-3.5-bolt-e4b-4gb", "nemotron-3.5-bolt-e4b"),
    ("stepfun-ai/step-3.7-flash", "step-3.7-flash"),
    ("mistralai/ministral-3-14b-instruct-2512", "ministral-3-14b"),
    # ids of the hosted API catalog (integrate.api.nvidia.com)
    ("nvidia/nemotron-nano-3-30b-a3b", "nemotron-3"),
    ("nvidia/nemotron-parse-2.0", "nemotron-parse"),
    ("z-ai/glm-5.3-flash", "glm-5.3-flash"),
]


@pytest.mark.parametrize("served_id, expected", MATCHES)
def test_nsc_features_match_served_ids(served_id, expected):
    record, how = nf.match(TABLE, served_id=served_id)
    assert record is not None and record["id"] == expected, served_id
    assert how in ("served_id", "served_id_pattern")
    hits = [f["id"] for f in TABLE["families"]
            if nf._find([f], nf.normalize_id(served_id))[0] is not None]
    assert hits == [expected], (served_id, hits)


NO_MATCH = [
    "acme/custom-llm-7b",
    "qwen/qwen3-235b-a22b-instruct-2507-fp8",
    "qwen/qwen3-coder-next",
    "qwen/qwen3-next-80b-a3b-instruct",
    "deepseek-ai/deepseek-v3-0324",
    "deepseek-ai/deepseek-v4-flash-0731",
    "deepseek-ai/deepseek-v4.1-flash-1017",
    "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "google/gemma-3-27b-it",
    "meta/llama-4-scout-17b-16e-instruct",
    "mistralai/mistral-small-3.1-24b-instruct-2503",
    "zai-org/glm-5",
    "zai-org/glm-51",
    "nvidia/nemoguard-jailbreak-detect",
    "meta/llama-3.1-8b-instruct-dgx",
    "llama",
]


@pytest.mark.parametrize("served_id", NO_MATCH)
def test_nsc_features_other_models_match_nothing(served_id):
    assert nf.match(TABLE, served_id=served_id) == (None, "none")


def test_nsc_features_match_by_image_wins_over_served_id():
    record, how = nf.match(
        TABLE,
        image="nvcr.io/nim/meta/llama-3.1-8b-instruct-dgx-spark:1.0.0-variant",
        served_id="meta/llama-3.1-8b-instruct",
    )
    assert (record["id"], how) == ("llama-3.1-8b-instruct-dgx-spark", "image")
    record, how = nf.match(TABLE, image="nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.6",
                           served_id="my-renamed-model")
    assert (record["id"], how) == ("llama-3.1-instruct", "image")
    record, how = nf.match(TABLE, image="nvcr.io/nim/meta/llama-3.1-8b-instruct-pb26h1:2.2.0")
    assert (record["id"], how) == ("llama-3.1-instruct", "image_pattern")


def test_nsc_features_match_falls_back_to_served_id_and_root():
    record, how = nf.match(TABLE, image="nvcr.io/nim/nvidia/vllm-model-free-nim:2.1.5",
                           served_id="openai/gpt-oss-20b")
    assert (record["id"], how) == ("gpt-oss", "served_id")
    record, how = nf.match(TABLE, served_id="my-model", root="Qwen/Qwen3-32B")
    assert (record["id"], how) == ("qwen3", "root")
    record, how = nf.match(TABLE, served_id="my-model", root="Qwen/Qwen3-14B")
    assert (record["id"], how) == ("qwen3", "root_pattern")


def test_nsc_features_match_family_and_unknown_family():
    record, how = nf.match(TABLE, family="GPT-OSS")
    assert (record["id"], how) == ("gpt-oss", "family")
    with pytest.raises(nsc.ScriptError) as caught:
        nf.match(TABLE, family="no-such-family")
    assert caught.value.code == "unknown_family"
    assert caught.value.exit_code == nsc.EXIT_INPUT


def test_nsc_features_family_warning_names_a_family_that_matches_nothing():
    """docs-7: --family takes the record without a match, so a family whose ids and patterns
    match neither the image, the served id nor its root gets a warning; a renamed served id
    with a matching image or root, or nothing to compare, gets none."""
    text = nf.family_warning(TABLE, "llama-3.1-instruct", served_id="acme/custom-llm-7b")
    assert text and "acme/custom-llm-7b" in text and "NIM_SERVED_MODEL_NAME" in text
    assert "--documented FEATURE=URL" in text
    assert nf.family_warning(TABLE, "llama-3.1-instruct",
                             served_id="meta/llama-3.1-8b-instruct") is None
    assert nf.family_warning(TABLE, "llama-3.1-instruct", served_id="my-model",
                             image="nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.5") is None
    assert nf.family_warning(TABLE, "qwen3", served_id="my-model", root="Qwen/Qwen3-14B") is None
    assert nf.family_warning(TABLE, "gpt-oss") is None
    assert nf.family_warning(TABLE, "gpt-oss", image="nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.5")


def test_nsc_features_no_record_asks_for_the_docs_before_family():
    plan = nf.decide(None, checks=nf.CHECKS, generation="2.1.5", table=TABLE)
    text = next(w for w in plan["warnings"] if w.startswith("No capability record"))
    assert text.index("--documented FEATURE=URL") < text.index("--family ID")
    assert "only when the served id renames" in text


def test_nsc_features_status_overrides():
    assert nf.match(TABLE, served_id="meta/llama-3.1-405b-instruct")[0]["status"] == "eos"
    assert nf.match(TABLE, served_id="meta/llama-3.1-8b-instruct")[0]["status"] == "supported"
    record, _ = nf.match(TABLE, image="nvcr.io/nim/meta/llama-3.1-8b-instruct-pb25h2:1.14.0")
    assert record["status"] == "eos"
    assert nf.match(TABLE, served_id="moonshotai/kimi-k2.5")[0]["status"] == "eos"
    assert nf.match(TABLE, served_id="moonshotai/kimi-k2.6")[0]["status"] == "supported"
    plan, _ = plan_for("meta/llama-3.1-405b-instruct")
    assert any("End of Support" in w for w in plan["warnings"])


def test_nsc_features_overrides_resolve_per_model():
    omni, _ = nf.match(TABLE, served_id="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
    assert omni["kind"] == "vlm" and omni["image"]["documented"] == "yes"
    assert omni["override"] == "Nano Omni 30B-A3B Reasoning"
    assert all("#llms" not in s["url"] for s in omni["image"]["sources"])
    nano, _ = nf.match(TABLE, served_id="nvidia/nemotron-3-nano")
    assert nano["kind"] == "llm" and nano["override"] is None
    ultra, _ = nf.match(TABLE, served_id="nvidia/nemotron-3-ultra-550b-a55b")
    assert ultra["tools"]["default_on"] == {"1.x": None, "2.x": True}
    small, _ = nf.match(TABLE, served_id="thinkingmachines/inkling-small")
    assert small["reasoning"]["default"] == "on" and small["backend"] == "vllm"
    assert "off_reason" not in small["reasoning"]
    inkling, _ = nf.match(TABLE, served_id="thinkingmachines/inkling-nvfp4")
    assert inkling["reasoning"]["default"] == "on" and inkling["backend"] == "sglang"
    assert inkling["reasoning"]["off_reason"] == "no_documented_switch"
    super_v3, _ = nf.match(TABLE, served_id="nvidia/nemotron-3-super-120b-a12b")
    urls = [s["url"] for s in super_v3["tools"]["sources"]]
    assert any("function-calling.html" in u for u in urls)  # family sources kept
    assert super_v3["tools"]["named"] == "yes"
    assert "overrides" not in super_v3


# --- switches -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("system=/think", {"method": "system", "key": None, "value": "/think"}),
        ("system= detailed thinking off ", {"method": "system", "key": None,
                                            "value": "detailed thinking off"}),
        ("user_suffix= /no_think", {"method": "user_suffix", "key": None, "value": " /no_think"}),
        ("kwargs.enable_thinking=false", {"method": "kwargs", "key": "enable_thinking",
                                          "value": False}),
        ("kwargs.thinking=true", {"method": "kwargs", "key": "thinking", "value": True}),
        ("kwargs.reasoning_effort=60", {"method": "kwargs", "key": "reasoning_effort",
                                        "value": 60}),
        ("kwargs.mode=fast", {"method": "kwargs", "key": "mode", "value": "fast"}),
        ('kwargs.mode="fast"', {"method": "kwargs", "key": "mode", "value": "fast"}),
        ("effort=low", {"method": "effort", "key": None, "value": "low"}),
        ("effort=none", {"method": "effort", "key": None, "value": "none"}),
        ("default", {"method": "default", "key": None, "value": None}),
        (" default ", {"method": "default", "key": None, "value": None}),
    ],
)
def test_nsc_features_parse_switch(text, expected):
    assert nf.parse_switch(text) == expected


@pytest.mark.parametrize(
    "text",
    ["", "   ", "think", "system=", "user_suffix=", "kwargs.chat_template=x", "kwargs.=1",
     "kwargs.a b=1", "kwargs.x=[1]", 'kwargs.x={"a": 1}', "effort=LOW!", "effort=",
     "foo=bar", "system=" + "x" * 600],
)
def test_nsc_features_parse_switch_rejects(text):
    with pytest.raises(nsc.ScriptError) as caught:
        nf.parse_switch(text)
    assert caught.value.code == "bad_input" and caught.value.exit_code == nsc.EXIT_INPUT


def test_nsc_features_apply_switch_forms():
    body = {"messages": [{"role": "user", "content": "Hi"}]}
    nf.apply_switch(body, nf.parse_switch("system=/no_think"))
    assert body["messages"][0] == {"role": "system", "content": "/no_think"}
    nf.apply_switch(body, nf.parse_switch("system=/think"))
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == "/think"
    nf.apply_switch(body, nf.parse_switch("user_suffix= /no_think"))
    assert body["messages"][-1]["content"] == "Hi /no_think"
    nf.apply_switch(body, nf.parse_switch("kwargs.enable_thinking=false"))
    nf.apply_switch(body, {"method": "kwargs", "key": "force_nonempty_content", "value": True})
    assert body["chat_template_kwargs"] == {"enable_thinking": False,
                                            "force_nonempty_content": True}
    nf.apply_switch(body, nf.parse_switch("effort=low"))
    assert body["reasoning_effort"] == "low"
    before = copy.deepcopy(body)
    nf.apply_switch(body, nf.parse_switch("default"))
    nf.apply_switch(body, None)
    assert body == before
    image_body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Color?"}, {"type": "image_url", "image_url": {"url": "x"}}]}]}
    nf.apply_switch(image_body, {"method": "user_suffix", "value": " Think."})
    assert image_body["messages"][0]["content"][0]["text"] == "Color? Think."


def test_nsc_features_parse_documented():
    assert nf.parse_documented("tools=https://example.com/model/tools") == (
        "tools", "https://example.com/model/tools"
    )
    for bad in ["tools", "speech=https://example.com/x", "tools=http://example.com/x",
                "tools=https://user:pw@example.com/x", "image=https://git" + "lab.example.com/x",
                "reasoning=https://localhost/x", "tools=https://exa mple.com/x",
                # a bracket in the host part: bad_input (exit 2), not internal_error. urlsplit
                # raises ValueError on some, and the last two parse on every Python
                "tools=https://[example.com/x", "tools_parallel=https://example.com]/x",
                "image=https://[example.com]/x", "tools=https://[::ffff:1.2.3.4]/x",
                "tools=https://[v1.fe]/x"]:
        with pytest.raises(nsc.ScriptError) as caught:
            nf.parse_documented(bad)
        assert caught.value.code == "bad_input", bad
        assert caught.value.exit_code == nsc.EXIT_INPUT, bad


# --- tier B -----------------------------------------------------------------------------------

IMAGE_DEFAULT = (
    "--reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"
)


def test_nsc_features_tier_b_image_default_applies():
    signals = nf.tier_b(IMAGE_DEFAULT, None)
    assert signals["tools_on_by_default"] is True
    assert signals["tools_in_deployment"] is True
    assert signals["tools_dropped_by_override"] is False
    assert signals["reasoning_parser_by_default"] == "nemotron_v3"
    assert signals["overridden"] is False


def test_nsc_features_tier_b_override_drops_parsers():
    signals = nf.tier_b(IMAGE_DEFAULT, "--max-num-seqs 8")
    assert signals["overridden"] is True
    assert signals["tools_dropped_by_override"] is True
    assert signals["reasoning_parser_dropped_by_override"] is True
    assert signals["tools_in_deployment"] is None


def test_nsc_features_tier_b_container_args_add_parsers():
    signals = nf.tier_b(None, None, ["--tool-call-parser=kimi_k3", "--enable-auto-tool-choice",
                                     "--reasoning-parser", "kimi_k3"])
    assert signals["tools_in_deployment"] is True
    assert signals["reasoning_parser_in_deployment"] == "kimi_k3"
    assert signals["args_known"] is True


LLAMA_DEFAULT = "--enable-auto-tool-choice --tool-call-parser llama3_json"


def test_nsc_features_tier_b_override_that_drops_only_auto_tool_choice():
    """vLLM answers tool_choice auto only with both flags: an override that keeps the parser
    but drops --enable-auto-tool-choice dropped tool calling."""
    signals = nf.tier_b(LLAMA_DEFAULT, "--tool-call-parser llama3_json --max-num-seqs 64")
    assert signals["effective"]["tool_parser"] == "llama3_json"
    assert signals["tools_on_by_default"] is True
    assert signals["tools_in_deployment"] is None
    assert signals["tools_dropped_by_override"] is True
    plan, _ = plan_for("meta/llama-3.1-8b-instruct", generation="2.1.6", backend="vllm",
                       signals=signals)
    assert entry(plan, "tools")["if_parser_missing"] == {
        "status": "fail", "reason": "parser_dropped_by_override", "warning": None}
    # an image default with only one of the two flags does not turn tool calling on
    assert nf.tier_b("--tool-call-parser llama3_json", None)["tools_on_by_default"] is None


def test_nsc_features_tier_b_unknown_default_warns_on_override_without_auto_tool_choice():
    signals = nf.tier_b(None, "--tool-call-parser llama3_json --max-num-seqs 64")
    assert signals["tools_in_deployment"] is None and signals["tools_in_set_value"] is False
    plan, _ = plan_for("meta/llama-3.1-8b-instruct", generation="2.1.6", backend="vllm",
                       signals=signals)
    missing = entry(plan, "tools")["if_parser_missing"]
    assert (missing["status"], missing["reason"]) == ("not_applicable",
                                                      "not_enabled_in_deployment")
    assert "override dropped it" in missing["warning"]
    assert "--enable-auto-tool-choice" in missing["warning"]


def test_nsc_features_tier_b_container_args_turn_auto_tool_choice_off():
    """--no-enable-auto-tool-choice (vLLM's BooleanOptionalAction form) in the container
    arguments turns it off; it is not the NIM_PASSTHROUGH_ARGS override that dropped it."""
    off = ["--no-enable-auto-tool-choice"]
    signals = nf.tier_b(LLAMA_DEFAULT, None, off)
    assert signals["effective"]["auto_tool_choice"] is False
    assert signals["tools_in_deployment"] is None
    assert signals["tools_dropped_by_override"] is False
    plan, _ = plan_for("meta/llama-3.1-8b-instruct", generation="2.1.6", backend="vllm",
                       signals=signals)
    assert entry(plan, "tools")["if_parser_missing"]["reason"] == "parser_not_enabled"
    signals = nf.tier_b(None, LLAMA_DEFAULT, off)
    assert signals["tools_in_deployment"] is None and signals["tools_in_set_value"] is True
    plan, _ = plan_for("meta/llama-3.1-8b-instruct", generation="2.1.6", backend="vllm",
                       signals=signals)
    warning = entry(plan, "tools")["if_parser_missing"]["warning"]
    assert warning and "override dropped it" not in warning  # the override kept both flags
    assert nf.tier_b(None, None, [*off, "--enable-auto-tool-choice", "--tool-call-parser",
                                  "llama3_json"])["tools_in_deployment"] is True


def test_nsc_features_tier_b_keeps_no_raw_text():
    key = fake_key()
    signals = nf.tier_b(IMAGE_DEFAULT, f"--api-key {key} --tool-call-parser '{key}'",
                        ["--hf-token", key])
    text = json.dumps(signals)
    assert key not in text and "api-key" not in text and "--max" not in text
    assert signals["effective"]["tool_parser"] is None  # a key-shaped name is not kept
    assert nf.parser_flags("--structured-outputs-config '{\"backend\":\"guidance\"}'")[
        "structured_outputs"] is True
    assert nf.parser_flags("--tool-call-parser 'unclosed")["tool_parser"] is None


# --- plans ------------------------------------------------------------------------------------

ENTRY_KEYS = {"check", "decision", "reason", "sources", "conflicts", "request", "known_issues",
              "hint", "warnings"}
ALL_REASONS = set(nf.TEST_REASONS) | set(nf.NOT_APPLICABLE_REASONS)


def assert_plan_shape(plan: dict, checks=nf.DEFAULT_CHECKS) -> None:
    assert [e["check"] for e in plan["checks"]] == [c for c in nf.CHECKS if c in checks]
    for item in plan["checks"]:
        assert set(item) >= ENTRY_KEYS, item["check"]
        assert item["decision"] in ("test", "not_applicable")
        assert item["reason"] in ALL_REASONS, item["reason"]
        assert (item["reason"] in nf.TEST_REASONS) == (item["decision"] == "test")
        assert all(isinstance(u, str) and u.startswith("https://") for u in item["sources"])
        if item["decision"] == "test" and item["check"] in nf.TOOL_CHECKS:
            missing = item["if_parser_missing"]
            assert missing["reason"] in nf.PARSER_MISSING_REASONS
            assert item["hint"] or missing["status"] == "fail"
    json.dumps(plan)


def test_nsc_features_plan_llama_31_8b():
    plan, _ = plan_for("meta/llama-3.1-8b-instruct", generation="2.1.6", backend="vllm")
    assert_plan_shape(plan)
    tools = entry(plan, "tools")
    assert (tools["decision"], tools["reason"]) == ("test", "documented")
    assert tools["request"]["max_tokens"] == 1024 and tools["request"]["mode"] == []
    assert tools["if_parser_missing"]["status"] == "not_applicable"
    assert tools["if_parser_missing"]["reason"] == "not_enabled_in_deployment"
    assert tools["if_parser_missing"]["warning"]
    assert "--tool-call-parser llama3_json" in tools["hint"] and "nim-deploy" in tools["hint"]
    assert "NIM_TOOL_CALL_PARSER" not in tools["hint"]  # 2.x takes server arguments
    for check in ("reasoning_on", "reasoning_off", "reasoning_off_stream"):
        assert entry(plan, check)["reason"] == "not_a_reasoning_model"
        assert any("reasoning-model.html" in url for url in entry(plan, check)["sources"])
    image = entry(plan, "image")
    assert (image["decision"], image["reason"]) == ("not_applicable", "text_only_model")
    assert image["sources"]
    assert entry(plan, "chat")["request"]["max_tokens"] == 64
    assert entry(plan, "json_schema")["decision"] == "test"
    assert entry(plan, "health")["decision"] == "test"


def test_nsc_features_plan_llama_31_on_1x_fails_missing_parser_in_its_own_image():
    """NIM 1.x turns tool calling on by itself only in the model's own (LLM-specific) image."""
    own = "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.15.4"
    plan, how = plan_for("meta/llama-3.1-8b-instruct", image=own, generation="1.15.4")
    tools = entry(plan, "tools")
    assert how == "image"
    assert tools["if_parser_missing"] == {"status": "fail", "reason": "parser_not_enabled",
                                          "warning": None}
    assert "NIM 1.x" in tools["hint"]
    assert nf.IMAGE_UNKNOWN_1X not in plan["warnings"]


@pytest.mark.parametrize(
    ("served_id", "image", "version"),
    [
        ("meta/llama-3.1-8b-instruct", None, "1.15.4"),  # a URL run: the image is unknown
        ("meta/llama-3.1-8b-instruct", None, "1.0.0-variant"),  # also a DGX Spark variant
        ("meta/llama-3.1-8b-instruct", "nvcr.io/nim/nvidia/llm-nim:1.15.0", "1.15.0"),
        ("meta-llama/Llama-3.1-8B-Instruct", "nvcr.io/nim/nvidia/llm-nim:1.15.0", "1.15.0"),
        ("google/gemma-3-1b-it", None, None),  # a 1.x-only default, the generation unknown
    ],
)
def test_nsc_features_plan_1x_default_on_needs_the_models_own_image(
    served_id: str, image: str | None, version: str | None
) -> None:
    """table-1: the record's NIM 1.x default_on true rests on "If an LLM-specific NIM container
    supports tool calling, it is enabled automatically" (1.15.0); the multi-LLM NIM serves the
    same ids and takes NIM_ENABLE_AUTO_TOOL_CHOICE. With the image unknown or another image, a
    missing parser is not applicable with a warning (D5: unknown default), never a fail."""
    plan, _ = plan_for(served_id, image=image, generation=version)
    tools = entry(plan, "tools")
    assert tools["decision"] == "test"
    missing = tools["if_parser_missing"]
    assert (missing["status"], missing["reason"]) == ("not_applicable",
                                                      "not_enabled_in_deployment")
    assert "own NIM container" in missing["warning"] and "opt-in" not in missing["warning"]
    assert ("pass --image" in missing["warning"]) == (image is None)
    if image is not None:
        assert image in missing["warning"]
    assert (nf.IMAGE_UNKNOWN_1X in plan["warnings"]) == (image is None and version is not None)
    assert "NIM_ENABLE_AUTO_TOOL_CHOICE" in tools["hint"] or version is None


def test_nsc_features_plan_own_image_keeps_the_documented_defaults():
    """The model's own image keeps the record's 1.x default (a scalar or a 1.x-only one too),
    and 2.x defaults, which the docs give for the model's NIM, are unchanged."""
    plan, _ = plan_for("google/gemma-3-1b-it", image="nvcr.io/nim/google/gemma-3-1b-it:1.15.0")
    assert entry(plan, "tools")["if_parser_missing"]["reason"] == "parser_not_enabled"
    plan, _ = plan_for("meta/llama-3.1-70b-instruct", generation="1.15.0",
                       image="nvcr.io/nim/meta/llama-3.1-70b-instruct:1.15.0")
    assert entry(plan, "tools")["if_parser_missing"]["status"] == "fail"
    plan, _ = plan_for("meta/llama-3.1-8b-instruct", family="llama-3.1-instruct",
                       generation="1.15.0", image="nvcr.io/nim/meta/llama-3.1-8b-instruct:1.15")
    assert entry(plan, "tools")["if_parser_missing"]["status"] == "fail"
    for image in (None, "nvcr.io/nim/zai-org/glm-5.3-flash:2.1.4"):
        plan, _ = plan_for("zai-org/glm-5.3-flash", image=image, generation="2.1.4")
        assert entry(plan, "tools")["if_parser_missing"]["reason"] == "parser_not_enabled"
        assert nf.IMAGE_UNKNOWN_1X not in plan["warnings"]
    plan, _ = plan_for("meta/llama-3.1-8b-instruct", generation="1.15.0", hosted=True)
    assert entry(plan, "tools")["if_parser_missing"]["status"] == "fail"  # hosted: no image
    assert nf.IMAGE_UNKNOWN_1X not in plan["warnings"]
    assert nf.own_image(None, "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.15.0") is False
    record, _ = nf.match(TABLE, served_id="meta/llama-3.1-8b-instruct")
    assert nf.own_image(record, "registry.example.com/llama:1.15.0") is False
    assert nf.own_image(record, "mirror.example.com/nim/meta/llama-3.1-8b-instruct:1.15.0")


def test_nsc_features_plan_llama_31_parallel_is_information_only():
    plan, _ = plan_for("meta/llama-3.1-8b-instruct", checks=nf.CHECKS)
    parallel = entry(plan, "tools_parallel")
    assert (parallel["decision"], parallel["reason"]) == ("not_applicable", "information_only")
    assert parallel["conflicts"]
    assert entry(plan, "tools_none")["decision"] == "test"
    assert entry(plan, "tools_stream")["request"]["stream"] is True


def test_nsc_features_plan_gpt_oss():
    plan, _ = plan_for("openai/gpt-oss-20b", checks=nf.CHECKS)
    assert_plan_shape(plan, nf.CHECKS)
    on = entry(plan, "reasoning_on")
    assert on["decision"] == "test"
    assert on["request"]["switch"] == {"method": "effort", "key": None, "value": "low"}
    for check in ("reasoning_off", "reasoning_off_stream"):
        assert (entry(plan, check)["decision"], entry(plan, check)["reason"]) == (
            "not_applicable", "always_on")
    assert entry(plan, "tools")["request"]["named_diagnostic"] is False
    assert entry(plan, "tools")["request"]["max_tokens"] == 4096
    assert entry(plan, "tools_stream")["request"]["retry"] == 1
    assert entry(plan, "tools_parallel")["reason"] == "not_documented_for_model"
    assert entry(plan, "tools_none")["reason"] == "not_documented_for_model"
    assert entry(plan, "image")["reason"] == "text_only_model"
    assert entry(plan, "json_schema")["request"]["max_tokens"] == 4096


def test_nsc_features_plan_nemotron_3_nano_switches():
    plan, _ = plan_for("nvidia/nemotron-3-nano", generation="2.1.6", backend="vllm")
    assert_plan_shape(plan)
    on, off = entry(plan, "reasoning_on"), entry(plan, "reasoning_off")
    assert on["request"]["switch"] == {"method": "kwargs", "key": "enable_thinking", "value": True}
    assert off["request"]["switch"] == {"method": "kwargs", "key": "enable_thinking",
                                        "value": False}
    stream_twin = entry(plan, "reasoning_off_stream")
    assert stream_twin["decision"] == "test" and stream_twin["request"]["stream"] is True
    tools = entry(plan, "tools")
    keys = [(s["key"], s["value"]) for s in tools["request"]["mode"]]
    assert keys == [("enable_thinking", True), ("force_nonempty_content", True)]
    assert tools["request"]["max_tokens"] == 4096
    assert "--tool-call-parser qwen3_coder" in tools["hint"]
    assert entry(plan, "json_schema")["request"]["switch"] == off["request"]["switch"]
    assert entry(plan, "concurrency")["request"]["switch"] == off["request"]["switch"]
    assert entry(plan, "image")["reason"] == "text_only_model"


def test_nsc_features_plan_nemotron_3_variant_1x_is_opt_in():
    plan, _ = plan_for("nvidia/nemotron-3-nano", generation="1.7.0-variant")
    tools = entry(plan, "tools")
    assert tools["if_parser_missing"] == {"status": "not_applicable",
                                          "reason": "not_enabled_in_deployment",
                                          "warning": None}
    assert "TOOL_CALL_PARSER=1" in tools["hint"]


def test_nsc_features_plan_dropped_by_override():
    signals = nf.tier_b(IMAGE_DEFAULT, "--max-num-seqs 8")
    plan, _ = plan_for("nvidia/nemotron-3-super-120b-a12b", generation="2.1.6",
                       backend="vllm", signals=signals)
    tools = entry(plan, "tools")
    assert tools["if_parser_missing"]["status"] == "fail"
    assert tools["if_parser_missing"]["reason"] == "parser_dropped_by_override"
    assert "NIM_PASSTHROUGH_ARGS" in tools["hint"]
    plan, _ = plan_for("nvidia/nemotron-3-super-120b-a12b", generation="2.1.6",
                       signals=nf.tier_b(IMAGE_DEFAULT, None))
    assert entry(plan, "tools")["if_parser_missing"]["reason"] == "parser_not_enabled"


def test_nsc_features_plan_documented_default_on_without_tier_b():
    plan, _ = plan_for("deepseek-ai/deepseek-v4.1-flash", generation="2.1.5",
                       signals=nf.tier_b(None, "--max-num-seqs 8"))
    tools = entry(plan, "tools")
    assert tools["if_parser_missing"]["reason"] == "parser_not_enabled"
    assert "deepseek_v41" in tools["hint"] and "NIM_PASSTHROUGH_ARGS" in tools["hint"]


def test_nsc_features_plan_mistral_small_4_never_sends_kwargs():
    plan, _ = plan_for("mistralai/mistral-small-4-119b-2603", checks=nf.CHECKS)
    assert_plan_shape(plan, nf.CHECKS)
    assert entry(plan, "reasoning_on")["request"]["switch"]["value"] == "high"
    assert entry(plan, "reasoning_off")["request"]["switch"] == {"method": "effort", "key": None,
                                                                  "value": "none"}
    switches = []
    for item in plan["checks"]:
        request = item["request"]
        switches += [request.get("switch"), *(request.get("mode") or [])]
    assert all(s is None or s["method"] != "kwargs" for s in switches)
    assert entry(plan, "image")["decision"] == "test"
    assert entry(plan, "tools")["request"]["named_diagnostic"] is False
    assert entry(plan, "reasoning_off_stream")["reason"] == "no_prefilled_think_block"


def test_nsc_features_plan_deepseek_v41_flash_profiles():
    text_only = {"id": "9f34efd50f49c2ff891db2deb4d51a666580bbbf4053d59e0a3ca7a0c33c2cf6"}
    plan, _ = plan_for("deepseek-ai/deepseek-v4.1-flash", profile=text_only)
    image = entry(plan, "image")
    assert (image["decision"], image["reason"]) == ("not_applicable", "text_only_profile")
    plan, _ = plan_for("deepseek-ai/deepseek-v4.1-flash", profile={"id": "06fcf7c1"})
    assert entry(plan, "image")["reason"] == "text_only_profile"
    nvfp4 = {"id": "30d2c978b8a1a2271e38f636de7402172262ef3a30f79d1c09b2991512cde916",
             "name": "vllm-nvfp4-tp4-pp1"}
    plan, _ = plan_for("deepseek-ai/deepseek-v4.1-flash", profile=nvfp4)
    image = entry(plan, "image")
    assert (image["decision"], image["reason"]) == ("test", "documented")
    assert image["request"]["switch"] == {"method": "effort", "key": None, "value": "none"}
    plan, _ = plan_for("deepseek-ai/deepseek-v4.1-flash")
    image = entry(plan, "image")
    assert image["decision"] == "test" and image["warnings"]


def test_nsc_features_plan_unknown_profile_expects_the_text_only_reply():
    """With the active profile's id unknown (no /v1/metadata, the hosted API, or a profile name
    only, which a hashed id list cannot match), the image check runs, warns, and treats the
    documented 400 of the text-only profiles as not applicable."""
    for profile in (None, {"name": "vllm-fp8-tp8-pp1"}):
        plan, _ = plan_for("deepseek-ai/deepseek-v4.1-flash", profile=profile)
        image = entry(plan, "image")
        assert image["decision"] == "test" and image["warnings"], profile
        assert image["if_text_only_reply"] == {"status": "not_applicable",
                                               "reason": "text_only_profile",
                                               "reply": nf.TEXT_ONLY_REPLY}
    nvfp4 = "30d2c978b8a1a2271e38f636de7402172262ef3a30f79d1c09b2991512cde916"
    plan, _ = plan_for("deepseek-ai/deepseek-v4.1-flash", profile={"id": nvfp4})
    assert entry(plan, "image")["warnings"] == []  # a known id that takes images
    plan, _ = plan_for("qwen/qwen3.5-397b-a17b")
    assert "if_text_only_reply" not in entry(plan, "image")  # no text-only profile documented
    sources = " ".join(q["quote"] for q in family(TABLE, "deepseek-v4.1-flash")["image"]["sources"])
    assert nf.TEXT_ONLY_REPLY in sources  # the reply is the one the docs quote


def test_nsc_features_plan_keeps_the_reasoning_budget_when_reasoning_stays_on():
    """json_schema and image of a model whose reasoning cannot be turned off: at least the
    record's reasoning budget (GLM-5.3-Flash and Kimi K3.1 Pro 8192), and the record's on
    switch when it only lowers the effort (reasoning_effort low)."""
    low = {"method": "effort", "key": None, "value": "low"}
    plan, _ = plan_for("zai-org/glm-5.3-flash")
    assert entry(plan, "json_schema")["request"] == {"max_tokens": 8192, "switch": low}
    image = entry(plan, "image")["request"]
    assert (image["max_tokens"], image["switch"]) == (8192, low)
    plan, _ = plan_for("moonshotai/Kimi-K3.1-Pro")
    assert entry(plan, "json_schema")["request"] == {"max_tokens": 8192, "switch": None}
    assert entry(plan, "image")["request"]["max_tokens"] == 8192
    plan, _ = plan_for("openai/gpt-oss-20b")
    assert entry(plan, "json_schema")["request"] == {"max_tokens": 4096, "switch": low}
    image = entry(plan, "image")
    assert image["reason"] == "text_only_model"
    plan, _ = plan_for("thinkingmachines/inkling-nvfp4")
    image = entry(plan, "image")["request"]
    assert (image["max_tokens"], image["switch"]) == (4096, None)
    plan, _ = plan_for("deepseek-ai/deepseek-r1")
    assert entry(plan, "json_schema")["request"] == {"max_tokens": 4096, "switch": None}
    plan, _ = plan_for("nvidia/nemotron-3-nano")  # an off switch: the small budgets stay
    assert entry(plan, "json_schema")["request"]["max_tokens"] == nf.JSON_TOKENS
    plan, _ = plan_for("nvidia/cosmos-reason2-2b")  # reasons only when the prompt asks
    assert entry(plan, "image")["request"] == {"max_tokens": nf.IMAGE_TOKENS_REASONING,
                                               "switch": None, "max_images": 1}


def test_nsc_features_plan_unknown_model():
    plan, how = plan_for("acme/custom-llm-7b")
    assert how == "none" and plan["record"] is None
    assert_plan_shape(plan)
    for check in ("tools", "reasoning_on", "reasoning_off", "reasoning_off_stream", "image"):
        item = entry(plan, check)
        assert (item["decision"], item["reason"]) == ("not_applicable", "no_capability_record")
    for check in ("health", "models", "chat", "stream", "json_schema", "concurrency"):
        assert entry(plan, check)["decision"] == "test"
    assert entry(plan, "chat")["request"]["max_tokens"] == 1024
    assert plan["warnings"]


def test_nsc_features_plan_documented_by_agent():
    url = "https://example.com/acme/custom-llm-7b/tools"
    plan, _ = plan_for("acme/custom-llm-7b", documented={"tools": url}, checks=nf.CHECKS)
    tools = entry(plan, "tools")
    assert (tools["decision"], tools["reason"]) == ("test", "documented_by_agent")
    assert tools["sources"] == [url]
    assert tools["if_parser_missing"]["status"] == "not_applicable"
    assert tools["if_parser_missing"]["warning"]
    assert entry(plan, "tools_parallel")["reason"] == "not_documented"
    assert entry(plan, "reasoning_on")["reason"] == "no_capability_record"
    assert "--documented" in plan["warnings"][0]


def test_nsc_features_plan_documented_tools_parallel_alone():
    """--documented tools_parallel=URL alone makes tools_parallel testable for a model with no
    record and for a record whose tool calling is unknown (model-features.md, and the advice for
    no_capability_record). It does not make the other tool checks run, and a documented no
    still wins: tools no (DeepSeek-R1) or parallel no (gpt-oss)."""
    url = "https://example.com/acme/parallel-tool-calls"
    for served_id, other in (("acme/custom-llm-7b", "no_capability_record"),
                             ("nvidia/cosmos-reason2-8b", "not_documented")):
        plan, _ = plan_for(served_id, documented={"tools_parallel": url}, checks=nf.CHECKS)
        assert_plan_shape(plan, nf.CHECKS)
        parallel = entry(plan, "tools_parallel")
        assert (parallel["decision"], parallel["reason"]) == ("test", "documented_by_agent")
        assert parallel["sources"] == [url] and parallel["request"]["parallel"] is True
        for check in ("tools", "tools_stream", "tools_none"):
            assert entry(plan, check)["reason"] == other, (served_id, check)
    for served_id in ("deepseek-ai/deepseek-r1", "openai/gpt-oss-20b"):
        plan, _ = plan_for(served_id, documented={"tools_parallel": url}, checks=nf.CHECKS)
        parallel = entry(plan, "tools_parallel")
        assert (parallel["decision"], parallel["reason"]) == ("not_applicable",
                                                              "not_documented_for_model")


def test_nsc_features_plan_agent_reasoning_switches():
    url = "https://example.com/acme/reasoning"
    switches = {"on": nf.parse_switch("kwargs.enable_thinking=true"),
                "off": nf.parse_switch("system=/no_think")}
    plan, _ = plan_for("acme/custom-llm-7b", documented={"reasoning": url}, switches=switches)
    on, off = entry(plan, "reasoning_on"), entry(plan, "reasoning_off")
    assert (on["decision"], on["reason"]) == ("test", "documented_by_agent")
    assert on["request"]["switch"]["key"] == "enable_thinking" and on["sources"] == [url]
    assert off["request"]["switch"] == {"method": "system", "key": None, "value": "/no_think"}
    assert entry(plan, "reasoning_off_stream")["reason"] == "no_prefilled_think_block"
    plan, _ = plan_for("acme/custom-llm-7b", documented={"reasoning": url})
    assert entry(plan, "reasoning_on")["reason"] == "no_documented_switch"
    plan, _ = plan_for("acme/custom-llm-7b", switches=switches)
    assert entry(plan, "reasoning_on")["reason"] == "no_capability_record"
    assert any("--documented reasoning" in w for w in plan["warnings"])


def test_nsc_features_plan_agent_cannot_override_a_documented_no():
    url = "https://example.com/llama/tools"
    plan, _ = plan_for("meta/llama-3.2-90b-vision-instruct", documented={"tools": url,
                                                                         "image": url})
    tools = entry(plan, "tools")
    assert (tools["decision"], tools["reason"]) == ("not_applicable", "not_documented_for_model")
    assert entry(plan, "image")["decision"] == "test"
    assert any("--documented tools was not used" in w for w in plan["warnings"])
    plan, _ = plan_for("openai/gpt-oss-20b", documented={"image": url},
                       switches={"off": nf.parse_switch("effort=none")})
    assert entry(plan, "image")["reason"] == "text_only_model"
    assert entry(plan, "reasoning_off")["reason"] == "always_on"
    assert len(plan["warnings"]) == 2


def test_nsc_features_plan_agent_switch_overrides_record_switch():
    """--reasoning-off replaces the record's off switch (enable_thinking false). 0 == False in
    Python, so the type is checked too: the request must send the integer 0, not false."""
    plan, _ = plan_for("nvidia/nemotron-3-nano",
                       switches={"off": nf.parse_switch("kwargs.enable_thinking=0")})
    switch = entry(plan, "reasoning_off")["request"]["switch"]
    assert switch == {"method": "kwargs", "key": "enable_thinking", "value": 0}
    assert type(switch["value"]) is int


def test_nsc_features_plan_out_of_scope():
    plan, _ = plan_for("nvidia/nemotron-parse-v2.0")
    assert_plan_shape(plan)
    for item in plan["checks"]:
        if item["check"] in ("health", "models"):
            assert item["decision"] == "test"
        else:
            assert (item["decision"], item["reason"]) == ("not_applicable", "out_of_scope")
    assert any("out of scope" in w for w in plan["warnings"])


def test_nsc_features_plan_hosted():
    plan, _ = plan_for("nvidia/nemotron-3.5-super-vl-120b-a12b", hosted=True)
    health = entry(plan, "health")
    assert (health["decision"], health["reason"]) == ("not_applicable", "hosted_no_health_route")
    tools = entry(plan, "tools")
    assert tools["if_parser_missing"]["status"] == "fail" and tools["hint"] is None
    assert entry(plan, "image")["decision"] == "test"
    assert entry(plan, "json_schema")["decision"] == "test"
    assert entry(plan, "reasoning_on")["request"]["switch"]["value"] == "high"


def test_nsc_features_plan_kimi_k31_pro_needs_the_reasoning_parser():
    plan, _ = plan_for("moonshotai/Kimi-K3.1-Pro")
    on = entry(plan, "reasoning_on")
    assert (on["decision"], on["reason"]) == ("not_applicable", "not_enabled_in_deployment")
    assert on["warnings"] and "--reasoning-parser kimi_k3" in on["hint"]
    assert entry(plan, "reasoning_off")["reason"] == "no_documented_switch"
    assert entry(plan, "tools")["if_parser_missing"]["status"] == "not_applicable"
    signals = nf.tier_b(None, None, ["--reasoning-parser", "kimi_k3", "--tool-call-parser",
                                     "kimi_k3", "--enable-auto-tool-choice"])
    plan, _ = plan_for("moonshotai/Kimi-K3.1-Pro", signals=signals)
    on = entry(plan, "reasoning_on")
    assert on["decision"] == "test" and on["request"]["max_tokens"] == 8192


def test_nsc_features_plan_documented_no_and_conflicts():
    plan, _ = plan_for("qwen/qwen-2.5-7b-instruct")
    tools = entry(plan, "tools")
    assert tools["reason"] == "not_documented_for_model" and tools["conflicts"]
    plan, _ = plan_for("deepseek-ai/deepseek-v3.1-terminus")
    tools = entry(plan, "tools")
    assert (tools["decision"], tools["reason"]) == ("test", "documented_conflict")
    assert tools["conflicts"] and tools["warnings"]
    assert tools["request"]["mode"] == [{"method": "kwargs", "key": "thinking", "value": False}]
    plan, _ = plan_for("deepseek-ai/deepseek-r1")
    assert entry(plan, "tools")["reason"] == "not_documented_for_model"
    assert entry(plan, "reasoning_off")["reason"] == "always_on"
    assert entry(plan, "reasoning_on")["request"]["sampling"] == {"temperature": 0.6}


def test_nsc_features_plan_reasoning_edge_families():
    plan, _ = plan_for("microsoft/phi-4-mini-instruct")
    for check in ("reasoning_on", "reasoning_off"):
        assert entry(plan, check)["reason"] == "reasoning_not_detectable"
    plan, _ = plan_for("nvidia/cosmos-reason2-2b")
    on, off = entry(plan, "reasoning_on"), entry(plan, "reasoning_off")
    assert on["request"]["switch"]["method"] == "user_suffix"
    assert off["decision"] == "test" and off["request"]["switch"]["method"] == "default"
    assert entry(plan, "tools")["reason"] == "not_documented"
    assert entry(plan, "image")["decision"] == "test"
    plan, _ = plan_for("deepseek-ai/DeepSeek-V4-Pro-0813")
    assert entry(plan, "reasoning_off")["reason"] == "no_documented_switch"
    assert entry(plan, "json_schema")["request"] == {
        "max_tokens": 4096, "switch": {"method": "effort", "key": None, "value": "low"}}
    plan, _ = plan_for("deepseek-ai/deepseek-v4-pro")
    assert entry(plan, "reasoning_off")["request"]["switch"]["key"] == "thinking"
    assert entry(plan, "json_schema")["request"]["max_tokens"] == 256
    plan, _ = plan_for("zai-org/glm-5.3-flash")
    assert entry(plan, "reasoning_on")["request"]["max_tokens"] == 8192
    assert entry(plan, "reasoning_off")["reason"] == "always_on"


def test_nsc_features_plan_streamed_tools_not_documented():
    plan, _ = plan_for("nvidia/nvidia-nemotron-nano-9b-v2", checks=["tools_stream"])
    item = entry(plan, "tools_stream")
    assert (item["decision"], item["reason"]) == ("not_applicable", "not_documented_for_model")
    plan, _ = plan_for("nvidia/llama-3.3-nemotron-super-49b-v1.5", checks=nf.CHECKS)
    assert entry(plan, "tools_stream")["reason"] == "not_documented_for_model"
    assert entry(plan, "tools")["request"]["mode"] == [
        {"method": "system", "key": None, "value": "/no_think"}]
    assert entry(plan, "tools")["request"]["max_tokens"] == 1024
    assert entry(plan, "tools_parallel")["decision"] == "test"
    assert entry(plan, "reasoning_off_stream")["decision"] == "test"


def test_nsc_features_plan_pipeline_parallel_on_1x():
    profile = {"name": "tensorrt_llm-a10g-bf16-tp4-pp2-latency"}
    plan, _ = plan_for("meta/llama-3.1-70b-instruct", generation="1.15.5", profile=profile)
    assert entry(plan, "tools")["reason"] == "pipeline_parallel_unsupported"
    assert entry(plan, "json_schema")["reason"] == "pipeline_parallel_unsupported"
    plan, _ = plan_for("meta/llama-3.1-70b-instruct", generation="2.1.6", profile=profile)
    assert entry(plan, "tools")["decision"] == "test"


def test_nsc_features_plan_pipeline_parallel_unknown_on_1x_warns():
    """A 1.x /v1/metadata gives only a hashed profile id, so pp cannot be read: tools and
    json_schema run, and the plan warns once why a failure may be the profile's."""
    hashed = {"id": "a35dd74f" + "0" * 56}
    for profile in (hashed, None):
        plan, _ = plan_for("meta/llama-3.1-70b-instruct", generation="1.15.5", profile=profile,
                           checks=nf.CHECKS)
        assert entry(plan, "tools")["decision"] == "test"
        assert entry(plan, "json_schema")["decision"] == "test"
        assert plan["warnings"].count(nf.PP_UNKNOWN) == 1
    plan, _ = plan_for("meta/llama-3.1-70b-instruct", generation="1.15.5",
                       profile={"id": hashed["id"], "name": "vllm-bf16-tp4-pp1"})
    assert nf.PP_UNKNOWN not in plan["warnings"] and entry(plan, "tools")["decision"] == "test"
    plan, _ = plan_for("meta/llama-3.1-70b-instruct", generation="2.1.6", profile=hashed)
    assert nf.PP_UNKNOWN not in plan["warnings"]


def test_nsc_features_plan_metadata_features():
    md = {"metadata_features": {"tool_calling": False},
          "metadata_url": "https://api.ngc.nvidia.com/x"}
    plan, _ = plan_for("meta/llama-3.1-8b-instruct", signals=md)
    tools = entry(plan, "tools")
    assert tools["reason"] == "not_documented_for_model"
    assert tools["sources"][0] == "https://api.ngc.nvidia.com/x"
    md = {"metadata_features": {"tool_calling": True, "parallel_tool_calling": True}}
    plan, _ = plan_for("google/gemma-4-31b-it", signals=md, checks=nf.CHECKS)
    assert entry(plan, "tools_parallel")["decision"] == "test"


def test_nsc_features_plan_metadata_parallel_yes_keeps_a_documented_no():
    """A no wins, as for tool calling: the Metadata API's parallel_tool_calling true does not
    turn the record's parallel "no" into a yes, with tool calling documented (gpt-oss) or when
    only --documented tools_parallel=URL makes the check eligible (tool calling unknown). With
    parallel unknown, that URL still makes the check run."""
    url = "https://example.com/acme/parallel-tool-calls"
    signals = {"metadata_features": {"parallel_tool_calling": True}}
    gpt_oss, _ = nf.match(TABLE, served_id="openai/gpt-oss-20b")
    unknown = copy.deepcopy(gpt_oss)
    unknown["tools"]["documented"] = "unknown"
    assert gpt_oss["tools"]["parallel"] == unknown["tools"]["parallel"] == "no"
    for record, documented in ((gpt_oss, None), (gpt_oss, {"tools_parallel": url}),
                               (unknown, {"tools_parallel": url})):
        plan = nf.decide(record, checks=["tools_parallel"], documented=documented,
                         signals=signals, table=TABLE)
        parallel = entry(plan, "tools_parallel")
        assert (parallel["decision"], parallel["reason"]) == (
            "not_applicable", "not_documented_for_model"), (record["tools"]["documented"],
                                                            documented)
    unknown["tools"]["parallel"] = "unknown"
    plan = nf.decide(unknown, checks=["tools_parallel"], documented={"tools_parallel": url},
                     signals=signals, table=TABLE)
    parallel = entry(plan, "tools_parallel")
    assert (parallel["decision"], parallel["reason"]) == ("test", "documented_by_agent")
    assert parallel["request"]["parallel"] is True


def test_nsc_features_plan_known_issues_and_vlm_image():
    plan, _ = plan_for("qwen/qwen3.5-397b-a17b")
    assert entry(plan, "image")["decision"] == "test"
    assert any("397b" in url for url in entry(plan, "image")["known_issues"])
    assert any("forced-tool-choice" in url for url in entry(plan, "tools")["known_issues"])
    plan, _ = plan_for("qwen/qwen3.8-27b", checks=nf.CHECKS)
    assert any("qwen3-8-27b" in url for url in entry(plan, "tools_none")["known_issues"])
    plan, _ = plan_for("meta/llama-3.2-11b-vision-instruct")
    image = entry(plan, "image")
    assert image["decision"] == "test" and image["request"]["max_images"] == 1
    assert image["request"]["max_tokens"] == 32


def test_nsc_features_plan_for_more_families():
    expectations = {
        "google/gemma-4-31b-it": {"reasoning_on": "test", "reasoning_off": "test",
                                  "image": "test", "tools": "test"},
        "zai-org/glm-5.2": {"reasoning_on": "test", "image": "not_applicable", "tools": "test"},
        "moonshotai/kimi-k2.6": {"reasoning_off": "test", "image": "test"},
        "thinkingmachines/inkling-nvfp4": {"reasoning_off": "not_applicable", "image": "test"},
        "thinkingmachines/inkling-small": {"reasoning_off": "test"},
        "qwen/qwen3-32b": {"reasoning_off_stream": "test", "tools": "test", "image":
                           "not_applicable"},
        "qwen/qwen3-next-80b-a3b-thinking": {"reasoning_off": "not_applicable"},
        "google/gemma-3-1b-it": {"tools": "test"},
        "mistralai/mistral-small-24b-instruct-2501": {"tools": "not_applicable"},
        "nv-mistralai/mistral-nemo-12b-instruct": {"tools": "not_applicable"},
        "mistralai/mistral-small-3.2-24b-instruct-2506": {"tools": "not_applicable",
                                                          "image": "test"},
        "nvidia/nemotron-nano-12b-v2-vl": {"image": "test", "reasoning_on": "test"},
        "nvidia/llama-3.1-nemotron-nano-8b-v1": {"reasoning_on": "test", "tools": "test"},
        "nvidia/nemotron-3.5-content-safety": {"chat": "not_applicable", "models": "test"},
    }
    for served_id, wanted in expectations.items():
        plan, _ = plan_for(served_id)
        assert_plan_shape(plan)
        for check, decision in wanted.items():
            assert entry(plan, check)["decision"] == decision, (served_id, check)


def test_nsc_features_inkling_small_reasoning_off_uses_effort():
    plan, _ = plan_for("thinkingmachines/inkling-small")
    off = entry(plan, "reasoning_off")
    assert off["request"]["switch"] == {"method": "effort", "key": None, "value": "none"}
    assert all("thinking-mode-reasoning" not in url for url in off["sources"])


V114_VARIANTS = (
    "https://docs.nvidia.com/nim/large-language-models/1.14.0/nim-container-variants.html"
)


def test_nsc_features_plan_nano_9b_v2_dgx_spark_variant_is_opt_in():
    """table-2: the 1.14.0 variants section of the Nano 9B v2 DGX Spark image, which the
    1.15.0 page points to, turns tool calling on only with TOOL_CALL_PARSER=1 and lists guided
    decoding as not supported. The override is found through the image; the model's own
    1.12.2 image keeps the LLM-specific default (on by itself)."""
    spark = "nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2-dgx-spark:1.0.0-variant"
    section = V114_VARIANTS + "#nvidia-nemotron-nano-9b-v2-dgx-spark"
    for generation in ("1.0.0-variant", None):
        plan, how = plan_for("nvidia/nvidia-nemotron-nano-9b-v2", image=spark,
                             generation=generation, checks=nf.CHECKS)
        assert_plan_shape(plan, nf.CHECKS)
        assert (how, plan["override"]) == ("image", "DGX Spark 1.x variant image")
        tools = entry(plan, "tools")
        assert tools["if_parser_missing"] == {"status": "not_applicable",
                                              "reason": "not_enabled_in_deployment",
                                              "warning": None}
        assert "TOOL_CALL_PARSER=1" in tools["hint"] and "by themselves" not in tools["hint"]
        assert section in tools["sources"]
        assert not any("function-calling.html" in url for url in tools["sources"])
        json_schema = entry(plan, "json_schema")
        assert (json_schema["decision"], json_schema["reason"]) == (
            "not_applicable", "not_documented_for_model")
        assert json_schema["sources"] == [section]
        assert entry(plan, "tools_stream")["reason"] == "not_documented_for_model"
        assert entry(plan, "tools_parallel")["decision"] == "test"
    own = "nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2:1.12.2"
    plan, _ = plan_for("nvidia/nvidia-nemotron-nano-9b-v2", image=own, generation="1.12.2")
    assert plan["override"] is None
    assert entry(plan, "tools")["if_parser_missing"]["reason"] == "parser_not_enabled"
    assert entry(plan, "json_schema")["decision"] == "test"


def test_nsc_features_plan_qwen3_variant_images_follow_their_variant_sections():
    """table-5: Qwen3 Next 80B A3B Thinking (1.0.0-variant) parses tool calls and reasoning
    only with TOOL_CALL_PARSER=1 and REASONING_PARSER=1, and its 1.14.0 section lists guided
    decoding as not supported: a missing parser is opt-in (no 'the docs do not say' warning)
    and the hints name those variables, not container arguments. The Qwen3-32B DGX Spark
    image, found by its own repository, skips json_schema for the same reason; qwen/qwen3-32b
    elsewhere keeps it."""
    plan, _ = plan_for("qwen/qwen3-next-80b-a3b-thinking", generation="1.0.0-variant")
    tools = entry(plan, "tools")
    assert tools["if_parser_missing"] == {"status": "not_applicable",
                                          "reason": "not_enabled_in_deployment", "warning": None}
    assert "TOOL_CALL_PARSER=1" in tools["hint"] and "<parser>" not in tools["hint"]
    assert V114_VARIANTS + "#qwen3-next-80b-a3b-thinking" in tools["sources"]
    assert "REASONING_PARSER=1" in entry(plan, "reasoning_on")["hint"]
    assert entry(plan, "reasoning_off")["reason"] == "always_on"
    json_schema = entry(plan, "json_schema")
    assert (json_schema["decision"], json_schema["reason"]) == ("not_applicable",
                                                                 "not_documented_for_model")
    assert json_schema["sources"] == [V114_VARIANTS + "#qwen3-next-80b-a3b-thinking"]
    plan, _ = plan_for("qwen/qwen3-next-80b-a3b-thinking", generation="2.1.5")
    assert entry(plan, "tools")["if_parser_missing"]["warning"]  # 2.x: not documented
    spark = "nvcr.io/nim/qwen/qwen3-32b-dgx-spark:1.0.0-variant"
    plan, how = plan_for("qwen/qwen3-32b", image=spark, generation="1.0.0-variant")
    assert (how, plan["override"]) == ("image", "Qwen3-32B DGX Spark 1.x variant image")
    assert entry(plan, "json_schema")["sources"] == [V114_VARIANTS + "#qwen3-32b-nim-for-dgx-spark"]
    assert entry(plan, "tools")["if_parser_missing"]["warning"] is None
    for kwargs in ({"generation": "1.0.0", "image": "nvcr.io/nim/qwen/qwen3-32b:1.0.0"},
                   {"generation": "2.1.5"}):
        plan, _ = plan_for("qwen/qwen3-32b", **kwargs)
        assert plan["override"] is None and entry(plan, "json_schema")["decision"] == "test"


def test_nsc_features_plan_nano_omni_vlm_does_not_inherit_the_llm_variant_switch():
    """table-5: the Nano Omni VLM override inherits nothing from the LLM Nemotron 3 Nano 1.x
    variant section. No VLM page names a tool switch or says whether the image turns the parser
    on, so on VLM NIM 1.7 a missing parser is unknown (not applicable, with a warning), and no
    hint names TOOL_CALL_PARSER or REASONING_PARSER; the 2.x vLLM switches stay."""
    omni = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    record, _ = nf.match(TABLE, served_id=omni)
    assert "default_on" not in record["tools"]
    assert "1.x" not in record["server_switches"]["tools"]
    assert "1.x" not in record["server_switches"]["reasoning"]
    plan, _ = plan_for(omni, generation="1.7.0", checks=nf.CHECKS)
    missing = entry(plan, "tools")["if_parser_missing"]
    assert (missing["status"], missing["reason"]) == ("not_applicable",
                                                      "not_enabled_in_deployment")
    assert missing["warning"] and "opt-in" not in missing["warning"]
    hints = " ".join(item["hint"] or "" for item in plan["checks"])
    assert "TOOL_CALL_PARSER" not in hints and "REASONING_PARSER" not in hints
    plan, _ = plan_for(omni, generation="2.1.5")
    assert "--tool-call-parser qwen3_coder" in entry(plan, "tools")["hint"]
    assert "--reasoning-parser nemotron_v3" in entry(plan, "reasoning_on")["hint"]
    plan, _ = plan_for("nvidia/nemotron-3-nano", generation="1.7.0-variant")
    assert "TOOL_CALL_PARSER=1" in entry(plan, "tools")["hint"]  # the LLM variant keeps it


def test_nsc_features_plan_inkling_reasoning_off_has_no_documented_switch():
    """table-6: no cited source says inkling's reasoning cannot be turned off (the release note
    says only that enable_thinking and include_reasoning do not turn it off), so reasoning off
    is no_documented_switch, never always_on, and an off switch from the docs is used."""
    plan, _ = plan_for("thinkingmachines/inkling-nvfp4", generation="2.1.5")
    for check in ("reasoning_off", "reasoning_off_stream"):
        item = entry(plan, check)
        assert (item["decision"], item["reason"]) == ("not_applicable", "no_documented_switch")
        assert item["hint"] == "Pass --reasoning-off SWITCH from the docs."
    assert entry(plan, "reasoning_on")["decision"] == "test"
    image = entry(plan, "image")["request"]
    assert (image["max_tokens"], image["switch"]) == (4096, None)  # reasoning stays on
    plan, _ = plan_for("thinkingmachines/inkling-nvfp4",
                       switches={"off": nf.parse_switch("effort=none")})
    off = entry(plan, "reasoning_off")
    assert off["decision"] == "test"
    assert off["request"]["switch"] == {"method": "effort", "key": None, "value": "none"}
    assert not any("was not used" in text for text in plan["warnings"])


def test_nsc_features_plan_new_vlm_records_test_their_documented_features():
    """table-4: the 2.1.5 support-matrix VLMs muse-glimmer and nemotron-3.5-bolt-e4b, and the
    VLM 1.7.0 models Step 3.7 Flash and Ministral 3 14B, have records: their documented features
    are tested instead of being no_capability_record."""
    plan, _ = plan_for("meta/muse-glimmer", generation="2.1.5", checks=nf.CHECKS)
    assert_plan_shape(plan, nf.CHECKS)
    assert entry(plan, "reasoning_on")["decision"] == "test"
    for check in ("reasoning_off", "reasoning_off_stream"):
        assert entry(plan, check)["reason"] == "always_on"  # "Reasoning cannot be turned off."
    assert entry(plan, "tools")["request"]["max_tokens"] == 4096
    parallel = entry(plan, "tools_parallel")
    assert parallel["decision"] == "test" and parallel["known_issues"]
    image = entry(plan, "image")
    assert image["decision"] == "test"
    assert (image["request"]["max_tokens"], image["request"]["switch"]) == (4096, None)
    assert entry(plan, "chat")["request"]["max_tokens"] >= 1024
    plan, _ = plan_for("nvidia/nemotron-3.5-bolt-e4b-4gb", generation="2.1.4")
    assert_plan_shape(plan)
    tools = entry(plan, "tools")
    assert tools["if_parser_missing"]["reason"] == "parser_not_enabled"
    assert "--tool-call-parser gemma4" in tools["hint"]
    off = {"method": "kwargs", "key": "enable_thinking", "value": False}
    assert entry(plan, "reasoning_off")["request"]["switch"] == off
    assert entry(plan, "image")["decision"] == "test"
    assert entry(plan, "image")["request"]["switch"] == off
    plan, _ = plan_for("stepfun-ai/step-3.7-flash")
    assert entry(plan, "reasoning_off")["request"]["switch"] == {"method": "kwargs",
                                                                  "key": "thinking",
                                                                  "value": False}
    assert entry(plan, "image")["decision"] == entry(plan, "tools")["decision"] == "test"
    plan, _ = plan_for("mistralai/ministral-3-14b-instruct-2512")
    assert entry(plan, "image")["decision"] == entry(plan, "tools")["decision"] == "test"
    assert entry(plan, "reasoning_on")["reason"] == "not_a_reasoning_model"


def test_nsc_features_reference_lists_the_models_the_table_does_not_cover():
    """table-4: references/model-features.md lists the documented NIMs that have no record yet,
    so an agent asks for --documented, and each listed id really matches no record."""
    text = (SKILL / "references" / "model-features.md").read_text(encoding="utf-8")
    section = text.split("\n## Models without a record", 1)[1].split("\n## ", 1)[0]
    rows = [line.split("|")[2] for line in section.splitlines() if line.startswith("| ")]
    ids = [found for cell in rows for found in re.findall(r"`([^`/]+/[^`/]+)`", cell)]
    assert len(ids) == len(rows) - 1  # one id per row below the header
    for wanted in ("deepseek-ai/deepseek-v4-flash-0731", "nvidia/nemotron-3.5-lightning-30b-a3b",
                   "Qwen/Qwen3.8-Flash-Next", "stepfun-ai/Step-5-Preview",
                   "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
                   "mistralai/Mistral-Large-4-1T-A52B", "meta/llama-4-scout-17b-16e-instruct"):
        assert wanted in ids, wanted
    for served_id in ids:
        assert nf.match(TABLE, served_id=served_id) == (None, "none"), served_id


def test_nsc_features_sources_are_versioned_and_back_their_value():
    """table-7: no NIM docs source is a /latest/ page (it changes under the pinned doc_version);
    Mistral Small 3.2's image source quotes the row that names the model; and no tools source
    of qwen3.5-vlm is about a model the family does not match (Qwen3.8-Flash-Next)."""
    for where, url in all_urls(TABLE):
        if urllib.parse.urlsplit(url).hostname == "docs.nvidia.com":
            assert "/latest/" not in url, (where, url)
    mistral = family(TABLE, "mistral-small-3.2-24b-2506")
    assert any("Mistral Small 3.2 24B Instruct 2506" in source["quote"]
               for source in mistral["image"]["sources"])
    assert nf.match(TABLE, served_id="Qwen/Qwen3.8-Flash-Next") == (None, "none")
    qwen = family(TABLE, "qwen3.5-vlm")
    assert not any("Flash-Next" in source["quote"] for source in qwen["tools"]["sources"])


def test_nsc_features_reference_validation_commands_use_the_skill_folder(tmp_path):
    """SKILL.md: stay in the project root and run a script as "<SKILL_DIR>/scripts/X.py". The
    validation commands of model-features.md name the skill's files that way, and the --check
    command, run as written from another folder, validates the table."""
    text = (SKILL / "references" / "model-features.md").read_text(encoding="utf-8")
    commands = [shlex.split(found) for found in re.findall(r"`(python3 [^`]+)`", text)]
    assert len(commands) >= 2
    for argv in commands:
        paths = [word for word in argv if word.endswith(".py")]
        assert paths and all(path.startswith("<SKILL_DIR>/") for path in paths), argv
    check = next(argv for argv in commands if "--check" in argv)
    argv = [sys.executable, *(word.replace("<SKILL_DIR>", str(SKILL)) for word in check[1:])]
    done = subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=60, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(done.stdout)["ok"] is True


def test_nsc_features_reference_names_the_checks_that_send_the_off_switch():
    """docs-14: model-features.md names the checks that send the family's off switch, and the
    plan agrees: json_schema, image and concurrency send it; chat and stream send none."""
    plan, _ = plan_for("qwen/qwen3.5-122b-a10b", generation="2.1.5")
    off = entry(plan, "reasoning_off")["request"]["switch"]
    sending = {item["check"] for item in plan["checks"]
               if item["check"] not in nf.REASONING_CHECKS
               and item["request"].get("switch") == off}
    assert sending == {"json_schema", "image", "concurrency"}
    assert all("switch" not in entry(plan, check)["request"] for check in ("chat", "stream"))
    text = " ".join((SKILL / "references" / "model-features.md").read_text(
        encoding="utf-8").split())
    sentence = next(s for s in text.split(". ") if "send the family's off switch" in s)
    named, rest = sentence.split("send the family's off switch", 1)
    assert set(re.findall(r"`([a-z_]+)`", named)) == sending
    assert set(re.findall(r"`([a-z_]+)`", rest.split("send no switch", 1)[0])) == {"chat",
                                                                                   "stream"}


@pytest.mark.parametrize("fid", [f["id"] for f in TABLE["families"]])
def test_nsc_features_every_family_plans_cleanly(fid):
    base = family(TABLE, fid)
    records = [nf.resolve(base, [])]
    for override in base.get("overrides", []):
        records.append(resolve_override(base, override, override_candidates(base, override)))
    for record in records:
        for kwargs in ({}, {"generation": "1.15.5"}, {"generation": "2.1.6", "backend": "sglang"},
                       {"hosted": True}, {"signals": nf.tier_b(IMAGE_DEFAULT, "--max-num-seqs 8")}):
            plan = nf.decide(record, checks=nf.CHECKS, table=TABLE, **kwargs)
            assert_plan_shape(plan, nf.CHECKS)
            assert plan["record"] == fid
            for item in plan["checks"]:
                for switch in [item["request"].get("switch"), *item["request"].get("mode", [])]:
                    assert switch is None or set(switch) == {"method", "key", "value"}


def test_nsc_features_select_checks():
    assert nf.select_checks() == nf.DEFAULT_CHECKS
    assert nf.select_checks(extended=True) == nf.CHECKS
    assert nf.select_checks(skip=["concurrency", "image"]) == tuple(
        c for c in nf.DEFAULT_CHECKS if c not in ("concurrency", "image"))
    assert nf.select_checks(["chat", "health"]) == ("health", "chat")
    assert nf.select_checks(["chat"], extended=True) == ("chat", *nf.EXTENDED_CHECKS)
    with pytest.raises(nsc.ScriptError) as caught:
        nf.select_checks(["chat", "speech"])
    assert caught.value.code == "bad_input"
    with pytest.raises(nsc.ScriptError):
        nf.decide(None, checks=["tool"], table=TABLE)


def test_nsc_features_decide_uses_the_shipped_table_by_default():
    record, _ = nf.match(TABLE, served_id="openai/gpt-oss-20b")
    assert nf.decide(record)["reviewed"] == TABLE["reviewed"]


# --- the command ------------------------------------------------------------------------------


def run_main(capsys, argv):
    with pytest.raises(SystemExit) as caught:
        nf.main(argv)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert set(payload) == {"ok", "state", "next", "facts", "errors"}
    for item in payload["errors"]:
        assert set(item) == {"code", "message", "fix"}
        assert re.fullmatch(r"[a-z][a-z0-9_]*", item["code"])
    return caught.value.code, payload


def test_nsc_features_main_notice_check_and_bad_flag(capsys, monkeypatch, tmp_path):
    code, payload = run_main(capsys, [])
    assert (code, payload["ok"], payload["state"]) == (0, True, "library")
    code, payload = run_main(capsys, ["--check"])
    assert (code, payload["state"]) == (0, "valid")
    assert payload["facts"]["families"] == len(TABLE["families"])
    code, payload = run_main(capsys, ["--bogus"])
    assert code == nsc.EXIT_INPUT and payload["errors"][0]["code"] == "bad_input"
    broken = copy.deepcopy(TABLE)
    broken["families"][0]["status"] = "retired"
    monkeypatch.setattr(nf, "TABLE", write_table(tmp_path, broken))
    code, payload = run_main(capsys, ["--check"])
    assert code == nsc.EXIT_PRECONDITION
    assert payload["errors"][0]["code"] == "features_table_invalid"


def test_nsc_features_main_check_reports_a_malformed_source_url(capsys, monkeypatch, tmp_path):
    """--check prints one JSON object with exit 3 for a cited URL that urlsplit cannot parse,
    never a traceback."""
    broken = copy.deepcopy(TABLE)
    broken["families"][0]["tools"]["sources"][0]["url"] = "https://[docs.nvidia.com/x"
    monkeypatch.setattr(nf, "TABLE", write_table(tmp_path, broken))
    code, payload = run_main(capsys, ["--check"])
    assert (code, payload["ok"], payload["state"]) == (nsc.EXIT_PRECONDITION, False, "invalid")
    assert payload["errors"][0]["code"] == "features_table_invalid"
    assert "not a valid URL" in payload["errors"][0]["message"]


def test_nsc_features_plan_named_diagnostic_only_for_yes_or_absent():
    """The named diagnostic is optional: only `named` yes or absent sends it; unknown (Mistral
    Small 4, whose forced tool calls can answer in text) and no skip it. tools_none differs:
    it is a check, so only `none: no` makes it not applicable."""
    plan, _ = plan_for("mistralai/mistral-small-4-119b-2603", checks=nf.CHECKS)
    assert family(TABLE, "mistral-small-4")["tools"]["named"] == "unknown"
    assert entry(plan, "tools")["request"]["named_diagnostic"] is False
    assert entry(plan, "tools_none")["decision"] == "test"
    plan, _ = plan_for("meta/llama-3.1-70b-instruct", checks=nf.CHECKS)
    assert "named" not in family(TABLE, "llama-3.1-instruct")["tools"]
    assert entry(plan, "tools")["request"]["named_diagnostic"] is True


def test_nsc_features_plan_vlm_1x_tag_skips_llm_1x_rules():
    """A VLM NIM numbers its own releases (1.7.0): the NIM LLM 1.x pipeline parallelism rule
    and the 1.x image warnings do not apply to a vlm record; an LLM record keeps them."""
    pp2 = {"name": "vllm-bf16-tp4-pp2"}
    vlm = ("google/gemma-4-31b-it", "nvcr.io/nim/google/gemma-4-31b-it:1.7.0")
    for image in (vlm[1], None):
        for profile in (pp2, None):
            plan, _ = plan_for(vlm[0], image=image, generation="1.7.0", profile=profile,
                               checks=nf.CHECKS)
            assert plan["kind"] == "vlm" and plan["generation"] == "1.x"
            for check in ("tools", "json_schema"):
                assert entry(plan, check)["reason"] != "pipeline_parallel_unsupported"
            assert nf.PP_UNKNOWN not in plan["warnings"]
            assert nf.IMAGE_UNKNOWN_1X not in plan["warnings"]
    plan, _ = plan_for("meta/llama-3.1-70b-instruct", generation="1.15.5", profile=pp2)
    assert entry(plan, "tools")["reason"] == "pipeline_parallel_unsupported"
    plan, _ = plan_for("meta/llama-3.1-70b-instruct", generation="1.15.5", checks=nf.CHECKS)
    assert nf.PP_UNKNOWN in plan["warnings"] and nf.IMAGE_UNKNOWN_1X in plan["warnings"]


def test_nsc_features_plan_nano_9b_v2_stream_no_cites_the_model_card_parser():
    """Nano 9B v2's family-wide `stream: no` rests on the model card's parser, which raises on
    streamed tool calls, not only on the DGX Spark release note (which the variant cites)."""
    parser = ("https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/blob/"
              "6533e8de2c68e4536bf7c411d7a3ce5734111476/nemotron_toolcall_parser_no_streaming.py")
    notes = ("https://docs.nvidia.com/nim/large-language-models/1.15.0/release-notes.html"
             "#all-current-known-issues-for-specific-models")
    quotes = {s["url"]: s["quote"]
              for s in family(TABLE, "nemotron-nano-9b-v2")["tools"]["sources"]}
    assert "not supported in streaming mode" in quotes[parser]
    for tag, cited in (("nvidia-nemotron-nano-9b-v2:1.12.2", parser),
                       ("nvidia-nemotron-nano-9b-v2-dgx-spark:1.0.0-variant", notes)):
        plan, _ = plan_for("nvidia/nvidia-nemotron-nano-9b-v2", image="nvcr.io/nim/nvidia/" + tag,
                           generation=tag.split(":")[1], checks=["tools_stream"])
        item = entry(plan, "tools_stream")
        assert (item["decision"], item["reason"]) == ("not_applicable", "not_documented_for_model")
        assert cited in item["sources"]
