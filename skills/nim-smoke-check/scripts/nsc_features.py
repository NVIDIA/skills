#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The model-feature table and the test plan of nim-smoke-check: `import nsc_features as nf`.

model_features.json, next to this file, holds one record per model family. Every value that
decides whether a check runs cites a public source (url, doc_version, verbatim quote). This
module:
  - loads and validates the table (load_table; ScriptError features_table_invalid, exit 3);
  - finds the record of a target (match): by --family, by the image's NGC repository
    (nvcr.io/nim/TEAM/REPO), then by the served model id or root, exactly or by pattern;
    case is ignored and ".", "-" and "_" count as the same character;
  - reads the parser flags of the image default and of the deployment (tier_b), never keeping
    the raw NIM_PASSTHROUGH_ARGS text;
  - decides, per check, "test" or "not_applicable" with a reason code and the sources (decide).

A plan (decide) is {"record", "name", "status", "kind", "override", "generation", "backend",
"reviewed", "checks": [entry, ...] in CHECKS order, "warnings"}. An entry is {"check",
"decision", "reason", "sources": [url], "conflicts": [url], "request", "known_issues": [url],
"hint", "warnings"}; a tools entry also has "if_parser_missing": {"status", "reason",
"warning"}, what the check reports when vLLM answers 400 "auto" tool choice requires ...
(no tool parser in this deployment); the image entry of a model with text-only profiles has
"if_text_only_reply": {"status", "reason", "reply"}, the documented 400 of those profiles.
request keys: max_tokens, sampling (temperature, top_p, top_k), switch (one reasoning switch),
mode (a list of switches for tool calls), stream, retry, named_diagnostic, expect_where.
A switch is {"method", "key", "value"}: system (a first system message), user_suffix (text
added to the last user message), kwargs (chat_template_kwargs[key] = value), effort
(reasoning_effort = value) or default (send nothing); apply_switch adds one to a request.

Stdlib only. Run directly, it prints a notice; --check validates the table and prints a summary.
Exit codes: 0 ok, 2 bad input, 3 precondition failed (the table is invalid).
"""

from __future__ import annotations

import copy
import json
import re
import shlex
import sys
import urllib.parse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nsc_common as nsc

TABLE = Path(__file__).resolve().with_name("model_features.json")
SCHEMA_VERSION = 1
MAX_TABLE_BYTES = 4_000_000

CHECKS = (
    "health", "models", "chat", "stream", "tools", "tools_stream", "tools_parallel", "tools_none",
    "reasoning_on", "reasoning_off", "reasoning_off_stream", "json_schema", "image", "concurrency",
)
EXTENDED_CHECKS = ("tools_stream", "tools_parallel", "tools_none")
DEFAULT_CHECKS = tuple(check for check in CHECKS if check not in EXTENDED_CHECKS)
TOOL_CHECKS = ("tools", "tools_stream", "tools_parallel", "tools_none")
REASONING_CHECKS = ("reasoning_on", "reasoning_off", "reasoning_off_stream")
FEATURES = ("tools", "tools_parallel", "reasoning", "image")  # what --documented FEATURE=URL names

# The reason codes of a plan entry (references/checks.md lists them with the check reasons).
TEST_REASONS = ("core_check", "engine_feature", "documented", "documented_conflict",
                "documented_by_agent")
NOT_APPLICABLE_REASONS = (
    "hosted_no_health_route", "out_of_scope", "no_capability_record", "not_documented",
    "not_documented_for_model", "not_a_reasoning_model", "always_on", "no_documented_switch",
    "no_prefilled_think_block", "reasoning_not_detectable", "text_only_model",
    "text_only_profile", "information_only", "pipeline_parallel_unsupported",
    "not_enabled_in_deployment",
)
# What the tools checks report on vLLM's 400 "no tool parser" answer.
PARSER_MISSING_REASONS = ("parser_not_enabled", "parser_dropped_by_override",
                          "not_enabled_in_deployment")

STATUSES = ("supported", "eos", "out_of_scope")
KINDS = ("llm", "vlm")
BACKENDS = ("vllm", "sglang", "either", None)
REASONING_DEFAULTS = ("none", "on", "off", "always", "prompt", "unknown")
DOCUMENTED = ("yes", "no", "unknown", "conflict")
SUB_VALUES = ("yes", "no", "unknown")
PARALLEL_VALUES = ("yes", "no", "unknown", "info")
ENGINE_VALUES = ("engine", "no")
OFF_REASONS = ("always_on", "no_documented_switch")
SWITCH_METHODS = ("system", "user_suffix", "kwargs", "effort", "default")
WHERE = ("reasoning", "reasoning_content", "inline", "inline_close_only")
SAMPLING_KEYS = ("temperature", "top_p", "top_k")
TIERS = ("A", "B", "C", "D")
GENERATIONS = ("1.x", "2.x")
SWITCH_KEYS = ("2.x-vllm", "2.x-sglang", "1.x")
FEATURE_KEYS = ("reasoning", "tools", "image", "json_schema", "stream")
OVERRIDE_KEYS = ("name", "ids", "patterns", "status", "kind", "backend", "server_switches",
                 "known_issues", *FEATURE_KEYS)

# Public documentation hosts a source may cite; github.com only for these organizations.
PUBLIC_HOSTS = (
    "docs.nvidia.com", "docs.vllm.ai", "docs.sglang.io", "docs.sglang.ai", "huggingface.co",
    "build.nvidia.com", "docs.api.nvidia.com", "catalog.ngc.nvidia.com", "api.ngc.nvidia.com",
)
GITHUB_ORGS = ("vllm-project", "sgl-project", "NVIDIA")
# Names of internal hosts, images and repositories a cited URL must not contain; assembled from
# parts so that the skill's own files do not name them.
PRIVATE_MARKERS = tuple(
    "".join(parts)
    for parts in (("git", "lab"), ("nv", "staging"), ("nim-", "internal"), ("nim", "lib"),
                  ("nim-", "runtime"), ("nim-inference-", "sdk"))
)

DEFAULT_REASONING_TOKENS = 4096
OFF_TOKENS = 256
TOOL_TOKENS = 1024
TOOL_TOKENS_REASONING = 4096
CHAT_TOKENS = 64
CHAT_TOKENS_REASONING = 1024
JSON_TOKENS = 256
JSON_TOKENS_REASONING = 4096
IMAGE_TOKENS = 32
IMAGE_TOKENS_REASONING = 1024
CONCURRENCY_TOKENS = 64
MAX_SWITCH_TEXT = 500
# reasoning_effort levels that only shorten the reasoning: json_schema and image send the
# record's on switch when it is one of these and reasoning cannot be turned off (gpt-oss,
# GLM-5.3-Flash: "To shorten reasoning, set reasoning_effort to low").
LIGHT_EFFORTS = ("minimal", "low")
# vLLM's 400 when an image goes to a profile that takes none; the docs quote it for
# deepseek-v4.1-flash's text-only profiles (model_features.json, its image sources).
TEXT_ONLY_REPLY = "At most 0 image(s) may be provided in one prompt"
PP_UNKNOWN = (
    "NIM 1.x does not support tool calling or structured generation with pipeline parallelism "
    "above 1, and the active profile's parallelism is not visible here (a 1.x /v1/metadata "
    "gives only the profile id): a tools or json_schema failure can come from such a profile."
)
# NIM 1.x turns tool calling on by itself only in the model's own (LLM-specific) container:
# "If an LLM-specific NIM container supports tool calling, it is enabled automatically"; the
# multi-LLM NIM (nvidia/llm-nim) needs NIM_ENABLE_AUTO_TOOL_CHOICE (1.15.0 function calling).
IMAGE_UNKNOWN_1X = (
    "The image this NIM 1.x runs is not known here, and on 1.x the image decides features: "
    "tool calling is on by itself only in the model's own NIM container (not in the multi-LLM "
    "NIM), and variants such as DGX Spark differ in tool calling and structured output. Pass "
    "--image with the image the NIM runs for an exact plan."
)
OWN_IMAGE_ONLY_1X = (
    "Tool calling is documented for this model, and NIM 1.x turns the tool parser on by itself "
    "only in the model's own NIM container (an LLM-specific NIM); in another image, such as the "
    "multi-LLM NIM, the deployment turns it on. {which} so whether it should be on here is "
    "unknown, and it is off in this deployment."
)

TABLE_FIX = (
    "Restore scripts/model_features.json from the skill release, or fix the entry the message "
    "names (references/model-features.md)"
)
SWITCH_SYNTAX = (
    "Use one of: system=TEXT, user_suffix=TEXT, kwargs.KEY=VALUE (VALUE as JSON: true, false, "
    "a number or a string), effort=LEVEL, default"
)
METADATA_DOC = "https://docs.nvidia.com/nim/metadata-api/1.0.0/reference/response-fields.md"
SWITCH_LABELS = {
    "2.x-vllm": "NIM 2.x (vLLM), as container arguments",
    "2.x-sglang": "NIM 2.x (SGLang), as container arguments",
    "1.x": "NIM 1.x",
}
PASSTHROUGH_NOTE = (
    " This deployment sets NIM_PASSTHROUGH_ARGS, which replaces the image's default value: keep "
    "the image's flags and add yours, or pass extra flags as container arguments."
)
DEPLOY_NOTE = " Changing the deployment is nim-deploy's job."


# --- ids ------------------------------------------------------------------------------------


def normalize_id(text: str) -> str:
    """Lower case, with "." and "_" read as "-": the form patterns and ids are compared in."""
    return text.strip().lower().replace("_", "-").replace(".", "-")


def resource_id(image: str | None) -> str | None:
    """nvcr.io/nim/TEAM/REPO[:TAG|@sha256:...] (or the same path on a mirror) -> nim/TEAM/REPO;
    None when the image is not a NIM repository path."""
    if not isinstance(image, str) or not image.strip():
        return None
    text = image.strip().lower().split("@", 1)[0]
    last = text.rsplit("/", 1)[-1]
    if ":" in last:
        text = text[: text.rfind(":")]
    parts = [part for part in text.split("/") if part]
    if len(parts) < 3 or parts[-3] != "nim":
        return None
    rid = "/".join(parts[-3:])
    return rid if re.fullmatch(r"nim/[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*", rid) else None


def image_tag(image: str | None) -> str | None:
    """The tag of an image reference (after the last ":" of its last path part), or None."""
    if not isinstance(image, str):
        return None
    last = image.strip().split("@", 1)[0].rsplit("/", 1)[-1]
    return last.rsplit(":", 1)[1] if ":" in last else None


def generation_of(version: str | None) -> str | None:
    """"2.1.5", "2", "2.x" or "2.1.4-variant" -> "2.x"; "1.15.0" -> "1.x"; else None."""
    if not isinstance(version, str):
        return None
    match = re.match(r"\s*v?(\d+)(?:[.x]|\b|-|$)", version)
    if match is None:
        return None
    major = int(match.group(1))
    return f"{major}.x" if major in (1, 2) else None


# --- loading and validation ---------------------------------------------------------------


def _invalid(where: str, problem: str) -> nsc.ScriptError:
    return nsc.ScriptError(
        "features_table_invalid", f"{where}: {problem}", TABLE_FIX, nsc.EXIT_PRECONDITION
    )


def _split_link(url: str) -> urllib.parse.SplitResult | None:
    """urlsplit(url), or None when `url` is not a valid link to a documentation page. Such a
    link names its host, so a "[" or "]" in the host part (a bracketed IP address) is refused
    here, the same on every Python: urlsplit refuses an unmatched one, but it refuses
    https://[docs.nvidia.com]/x only from 3.10.16, 3.11.4 and 3.12. Older releases read
    docs.nvidia.com as its host."""
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return None
    return None if "[" in parsed.netloc or "]" in parsed.netloc else parsed


def check_source_url(url: object) -> str | None:
    """Why a source URL is not allowed (None when it is): https only, a valid link
    (_split_link), a public documentation host, and no private-repository or internal-image
    name."""
    if not isinstance(url, str) or not url.startswith("https://") or any(c.isspace() for c in url):
        return "not an https URL"
    parsed = _split_link(url)
    if parsed is None:
        return "is not a valid URL"
    host = (parsed.hostname or "").lower()
    if parsed.username or parsed.password:
        return "has user information"
    lowered = url.lower()
    for marker in PRIVATE_MARKERS:
        if marker in lowered:
            return f"names {marker!r}"
    if host == "github.com":
        org = parsed.path.strip("/").split("/", 1)[0]
        return None if org in GITHUB_ORGS else f"github.com/{org} is not a public source org"
    return None if host in PUBLIC_HOSTS else f"host {host!r} is not a public documentation host"


def _check_sources(where: str, sources: object) -> None:
    if sources is None:
        return
    if not isinstance(sources, list):
        raise _invalid(where, "sources must be a list")
    for index, source in enumerate(sources):
        here = f"{where}[{index}]"
        if not isinstance(source, dict):
            raise _invalid(here, "a source must be an object")
        problem = check_source_url(source.get("url"))
        if problem:
            raise _invalid(here, f"url {problem}")
        quote = source.get("quote")
        if not isinstance(quote, str) or not quote.strip() or len(quote) > 300:
            raise _invalid(here, "quote must be text of 1 to 300 characters")
        if not isinstance(source.get("doc_version"), str) or not source["doc_version"]:
            raise _invalid(here, "doc_version is missing")
        if source.get("tier") not in TIERS:
            raise _invalid(here, f"tier must be one of {', '.join(TIERS)}")


def _check_switch(where: str, switch: object) -> None:
    if switch is None:
        return
    if not isinstance(switch, dict) or switch.get("method") not in SWITCH_METHODS:
        raise _invalid(where, f"a switch needs a method: {', '.join(SWITCH_METHODS)}")
    method, value = switch["method"], switch.get("value")
    if method == "kwargs":
        if not isinstance(switch.get("key"), str) or switch["key"] == "chat_template":
            raise _invalid(where, "a kwargs switch needs a key (never chat_template)")
        if not isinstance(value, (bool, int, float, str)) and value is not None:
            raise _invalid(where, "a kwargs value must be a JSON scalar")
    elif method in ("system", "user_suffix", "effort"):
        if not isinstance(value, str) or not value.strip() or len(value) > MAX_SWITCH_TEXT:
            raise _invalid(where, f"a {method} switch needs a text value")


def _check_sampling(where: str, sampling: object) -> None:
    if sampling is None:
        return
    if not isinstance(sampling, dict) or any(key not in SAMPLING_KEYS for key in sampling):
        raise _invalid(where, f"sampling keys are {', '.join(SAMPLING_KEYS)}")
    for key, value in sampling.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise _invalid(where, f"sampling {key} must be a number")


def _check_tokens(where: str, value: object) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)
                              or not 1 <= value <= 32768):
        raise _invalid(where, "max_tokens must be an integer from 1 to 32768")


def _check_default_on(where: str, value: object) -> None:
    if value is None or isinstance(value, bool):
        return
    if not isinstance(value, dict) or any(key not in GENERATIONS for key in value):
        raise _invalid(where, "default_on is true, false, null or {\"1.x\": ..., \"2.x\": ...}")
    if any(item is not None and not isinstance(item, bool) for item in value.values()):
        raise _invalid(where, "default_on values are true, false or null")


def _has_quote(obj: Mapping[str, Any], tiers: tuple[str, ...] = TIERS) -> bool:
    return any(
        isinstance(source, dict) and source.get("quote") and source.get("tier") in tiers
        for source in obj.get("sources") or []
    )


def _check_reasoning(where: str, obj: object, partial: bool) -> None:
    if not isinstance(obj, dict):
        raise _invalid(where, "reasoning must be an object")
    if (not partial or "default" in obj) and obj.get("default") not in REASONING_DEFAULTS:
        raise _invalid(where, f"default must be one of {', '.join(REASONING_DEFAULTS)}")
    for key in ("on", "off"):
        _check_switch(f"{where}.{key}", obj.get(key))
    for key in ("sampling_on", "sampling_off"):
        _check_sampling(f"{where}.{key}", obj.get(key))
    _check_tokens(f"{where}.max_tokens", obj.get("max_tokens"))
    where_list = obj.get("where", [])
    if not isinstance(where_list, list) or any(item not in WHERE for item in where_list):
        raise _invalid(where, f"where lists {', '.join(WHERE)}")
    if obj.get("off_reason") not in (*OFF_REASONS, None):
        raise _invalid(where, f"off_reason must be one of {', '.join(OFF_REASONS)}")
    for key in ("prefills_off", "detectable", "parser_default_on"):
        if obj.get(key) is not None and not isinstance(obj.get(key), bool):
            raise _invalid(where, f"{key} must be true or false")
    _check_sources(f"{where}.sources", obj.get("sources"))
    if not partial and obj.get("default") not in ("none", None) and not _has_quote(obj):
        raise _invalid(where, "a reasoning model needs a quoted source")


def _check_tools(where: str, obj: object, partial: bool) -> None:
    if not isinstance(obj, dict):
        raise _invalid(where, "tools must be an object")
    if (not partial or "documented" in obj) and obj.get("documented") not in DOCUMENTED:
        raise _invalid(where, f"documented must be one of {', '.join(DOCUMENTED)}")
    _check_default_on(f"{where}.default_on", obj.get("default_on"))
    for key in ("named", "none", "stream"):
        if obj.get(key) not in (*SUB_VALUES, None):
            raise _invalid(where, f"{key} must be one of {', '.join(SUB_VALUES)}")
    if obj.get("parallel") not in (*PARALLEL_VALUES, None):
        raise _invalid(where, f"parallel must be one of {', '.join(PARALLEL_VALUES)}")
    mode = obj.get("mode", [])
    if not isinstance(mode, list):
        raise _invalid(where, "mode must be a list of switches")
    for index, switch in enumerate(mode):
        _check_switch(f"{where}.mode[{index}]", switch)
    _check_sampling(f"{where}.sampling", obj.get("sampling"))
    _check_tokens(f"{where}.max_tokens", obj.get("max_tokens"))
    retry = obj.get("stream_retry")
    if retry is not None and (not isinstance(retry, int) or isinstance(retry, bool)
                              or not 0 <= retry <= 2):
        raise _invalid(where, "stream_retry must be 0, 1 or 2")
    _check_sources(f"{where}.sources", obj.get("sources"))
    _check_sources(f"{where}.conflicts", obj.get("conflicts"))
    if (not partial and obj.get("documented") in ("yes", "conflict")
            and not _has_quote(obj, ("A", "B"))):
        raise _invalid(where, "documented tool calling needs a quoted NIM source (tier A or B)")
    if not partial and "yes" in (obj.get("parallel"), obj.get("stream")) and not _has_quote(obj):
        raise _invalid(where, "a yes needs a quoted source")


def _check_image(where: str, obj: object, partial: bool) -> None:
    if not isinstance(obj, dict):
        raise _invalid(where, "image must be an object")
    if (not partial or "documented" in obj) and obj.get("documented") not in DOCUMENTED:
        raise _invalid(where, f"documented must be one of {', '.join(DOCUMENTED)}")
    profiles = obj.get("text_only_profiles", [])
    if not isinstance(profiles, list) or any(not isinstance(item, str) for item in profiles):
        raise _invalid(where, "text_only_profiles must be a list of profile ids or names")
    images = obj.get("max_images")
    if images is not None and (not isinstance(images, int) or isinstance(images, bool)
                               or images < 1):
        raise _invalid(where, "max_images must be a positive integer")
    _check_sources(f"{where}.sources", obj.get("sources"))
    if not partial and obj.get("documented") == "yes" and not _has_quote(obj, ("A", "B")):
        raise _invalid(where, "documented image input needs a quoted NIM source (tier A or B)")


def _check_engine_feature(where: str, obj: object, partial: bool) -> None:
    if not isinstance(obj, dict):
        raise _invalid(where, "must be an object")
    if (not partial or "documented" in obj) and obj.get("documented") not in ENGINE_VALUES:
        raise _invalid(where, f"documented must be one of {', '.join(ENGINE_VALUES)}")
    _check_switch(f"{where}.switch", obj.get("switch"))
    _check_sources(f"{where}.sources", obj.get("sources"))
    if not partial and obj.get("documented") == "no" and not _has_quote(obj):
        raise _invalid(where, "a documented no needs a quoted source")


def _check_switches(where: str, obj: object) -> None:
    if not isinstance(obj, dict):
        raise _invalid(where, "server_switches must be an object")
    for feature, table in obj.items():
        if feature not in ("tools", "reasoning") or not isinstance(table, dict):
            raise _invalid(where, "server_switches holds tools and reasoning objects")
        for key, text in table.items():
            if key not in SWITCH_KEYS or not isinstance(text, str) or not text.strip():
                raise _invalid(f"{where}.{feature}", f"keys are {', '.join(SWITCH_KEYS)}")


def _check_issues(where: str, issues: object) -> None:
    if not isinstance(issues, list):
        raise _invalid(where, "known_issues must be a list")
    for index, item in enumerate(issues):
        here = f"{where}[{index}]"
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise _invalid(here, "an issue needs url, affects and text")
        problem = check_source_url(item.get("url"))
        if problem:
            raise _invalid(here, f"url {problem}")
        affects = item.get("affects")
        if not isinstance(affects, list) or not affects or any(c not in CHECKS for c in affects):
            raise _invalid(here, "affects must list check ids")


def _check_patterns(where: str, patterns: object) -> None:
    if not isinstance(patterns, list):
        raise _invalid(where, "patterns must be a list")
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern.startswith("^") or not pattern.endswith("$"):
            raise _invalid(where, f"pattern {pattern!r} must be anchored with ^ and $")
        if "\\." in pattern or "_" in pattern or pattern != pattern.lower():
            raise _invalid(where, f"pattern {pattern!r}: write it lower case with - for . and _")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise _invalid(where, f"pattern {pattern!r} does not compile: {exc}") from exc


def _check_feature_objects(where: str, record: Mapping[str, Any], partial: bool) -> None:
    if "reasoning" in record:
        _check_reasoning(f"{where}.reasoning", record["reasoning"], partial)
    if "tools" in record:
        _check_tools(f"{where}.tools", record["tools"], partial)
    if "image" in record:
        _check_image(f"{where}.image", record["image"], partial)
    for key in ("json_schema", "stream"):
        if key in record:
            _check_engine_feature(f"{where}.{key}", record[key], partial)
    if "server_switches" in record:
        _check_switches(f"{where}.server_switches", record["server_switches"])
    if "known_issues" in record:
        _check_issues(f"{where}.known_issues", record["known_issues"])
    if record.get("status") not in (*STATUSES, None) or (not partial and "status" not in record):
        raise _invalid(where, f"status must be one of {', '.join(STATUSES)}")
    if record.get("kind") not in (*KINDS, None) or (not partial and "kind" not in record):
        raise _invalid(where, f"kind must be one of {', '.join(KINDS)}")
    if record.get("backend") not in BACKENDS:
        raise _invalid(where, "backend must be vllm, sglang, either or null")


def _check_family(index: int, record: object) -> str:
    if not isinstance(record, dict):
        raise _invalid(f"families[{index}]", "a family must be an object")
    fid = record.get("id")
    if not isinstance(fid, str) or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,63}", fid):
        raise _invalid(f"families[{index}]", "id must be lower-case letters, digits, . and -")
    where = f"family {fid}"
    match = record.get("match")
    if not isinstance(match, dict):
        raise _invalid(where, "match must be an object")
    for key in ("repos", "served_ids"):
        values = match.get(key, [])
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise _invalid(where, f"match.{key} must be a list of ids")
    if any(not repo.startswith("nim/") or repo.count("/") != 2 for repo in match.get("repos", [])):
        raise _invalid(where, "match.repos entries look like nim/TEAM/REPO")
    _check_patterns(f"{where}.match.patterns", match.get("patterns", []))
    if not (match.get("repos") or match.get("served_ids") or match.get("patterns")):
        raise _invalid(where, "match needs a repo, a served id or a pattern")
    for key in ("reasoning", "tools", "image"):
        if key not in record:
            raise _invalid(where, f"{key} is missing")
    _check_feature_objects(where, record, partial=False)
    if record["status"] == "out_of_scope" and not record.get("out_of_scope_reason"):
        raise _invalid(where, "an out_of_scope family needs out_of_scope_reason")
    _check_sources(f"{where}.out_of_scope_sources", record.get("out_of_scope_sources"))
    overrides = record.get("overrides", [])
    if not isinstance(overrides, list):
        raise _invalid(where, "overrides must be a list")
    for number, override in enumerate(overrides):
        here = f"{where}.overrides[{number}]"
        if not isinstance(override, dict) or any(key not in OVERRIDE_KEYS for key in override):
            raise _invalid(here, f"override keys are {', '.join(OVERRIDE_KEYS)}")
        # resolve() reads ids and name: normalize_id needs text, and the name is joined as text
        ids = override.get("ids")
        if "ids" in override and (not isinstance(ids, list) or not ids or any(
                not isinstance(item, str) or not item.strip() for item in ids)):
            raise _invalid(here, "ids must be a list of model ids; leave it out for an override "
                           "with patterns only")
        if override.get("name") is not None and not isinstance(override["name"], str):
            raise _invalid(here, "name must be text")
        if not (ids or override.get("patterns")):
            raise _invalid(here, "an override needs ids or patterns")
        _check_patterns(f"{here}.patterns", override.get("patterns", []))
        _check_feature_objects(here, override, partial=True)
    return fid


def _exact_ids(record: Mapping[str, Any]) -> set[str]:
    match = record.get("match") or {}
    ids = {normalize_id(repo) for repo in match.get("repos", [])}
    ids |= {normalize_id(repo)[4:] for repo in match.get("repos", [])}
    ids |= {normalize_id(served) for served in match.get("served_ids", [])}
    return ids


def validate_table(table: object) -> dict[str, Any]:
    """The table when it is valid; else ScriptError features_table_invalid (exit 3)."""
    if not isinstance(table, dict):
        raise _invalid("table", "the top level must be an object")
    if table.get("schema_version") != SCHEMA_VERSION:
        raise _invalid("table", f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(table.get("reviewed"), str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", table["reviewed"]
    ):
        raise _invalid("table", "reviewed must be a date, YYYY-MM-DD")
    engine = table.get("engine")
    if not isinstance(engine, dict):
        raise _invalid("table", "engine must be an object")
    for key, entry in engine.items():
        if not isinstance(entry, dict):
            raise _invalid(f"engine.{key}", "must be an object")
        _check_sources(f"engine.{key}.sources", entry.get("sources"))
    families = table.get("families")
    if not isinstance(families, list) or not families:
        raise _invalid("table", "families must be a non-empty list")
    seen: dict[str, str] = {}
    owners: dict[str, str] = {}
    for index, record in enumerate(families):
        fid = _check_family(index, record)
        if fid in seen:
            raise _invalid(f"family {fid}", "the id is used twice")
        seen[fid] = fid
        for exact in _exact_ids(record):
            if owners.setdefault(exact, fid) != fid:
                raise _invalid(f"family {fid}", f"{exact} also belongs to {owners[exact]}")
    for record in families:
        for exact in _exact_ids(record):
            for other in families:
                if other is record:
                    continue
                for pattern in (other.get("match") or {}).get("patterns", []):
                    if re.fullmatch(pattern, exact):
                        raise _invalid(
                            f"family {other['id']}",
                            f"pattern {pattern!r} also matches {exact} of {record['id']}",
                        )
    return table


def load_table(path: Path = TABLE) -> dict[str, Any]:
    """Read and validate model_features.json; ScriptError features_table_invalid (exit 3) when
    it cannot be read or is not valid."""
    try:
        with open(path, "rb") as stream:
            raw = stream.read(MAX_TABLE_BYTES + 1)
    except OSError as exc:
        raise _invalid(str(path), f"cannot read it: {exc.strerror}") from exc
    if len(raw) > MAX_TABLE_BYTES:
        raise _invalid(str(path), f"larger than {MAX_TABLE_BYTES:,} bytes")
    try:
        table = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _invalid(str(path), f"not valid JSON ({exc})") from exc
    return validate_table(table)


_DEFAULT_TABLE: dict[str, Any] | None = None


def default_table() -> dict[str, Any]:
    """The shipped table, loaded and validated once per process."""
    global _DEFAULT_TABLE
    if _DEFAULT_TABLE is None:
        _DEFAULT_TABLE = load_table(TABLE)
    return _DEFAULT_TABLE


# --- matching -------------------------------------------------------------------------------


MAIN_KEYS = {"reasoning": "default", "tools": "documented", "image": "documented",
             "json_schema": "documented", "stream": "documented"}


def _merge(record: dict[str, Any], override: Mapping[str, Any]) -> None:
    """Apply one override in place. Feature objects merge key by key: an override that sets the
    feature's main value (documented; default for reasoning) replaces its sources and
    conflicts, otherwise they are appended; a null value removes the key. known_issues are
    appended; other keys replace."""
    for key, value in override.items():
        if key in ("name", "ids", "patterns"):
            continue
        if key == "known_issues":
            record[key] = [*record.get(key, []), *value]
        elif isinstance(value, dict) and isinstance(record.get(key), dict):
            merged = dict(record[key])
            if MAIN_KEYS.get(key) in value:
                merged.pop("sources", None)
                merged.pop("conflicts", None)
            for inner, item in value.items():
                if item is None:  # before the lists: {"sources": null} removes them too
                    merged.pop(inner, None)
                elif inner in ("sources", "conflicts"):
                    merged[inner] = [*(merged.get(inner) or []), *item]
                else:
                    merged[inner] = copy.deepcopy(item)
            record[key] = merged
        else:
            record[key] = copy.deepcopy(value)


def _override_matches(override: Mapping[str, Any], candidates: list[str]) -> bool:
    ids = {normalize_id(item) for item in override.get("ids", [])}
    for candidate in candidates:
        if candidate in ids:
            return True
        if any(re.fullmatch(pattern, candidate) for pattern in override.get("patterns", [])):
            return True
    return False


def resolve(record: Mapping[str, Any], candidates: list[str]) -> dict[str, Any]:
    """A copy of a family record with the overrides that match a candidate id applied; the copy
    names them in "override" and has no "overrides" key."""
    resolved = copy.deepcopy(dict(record))
    overrides = resolved.pop("overrides", [])
    names = []
    for override in overrides:
        if _override_matches(override, candidates):
            _merge(resolved, override)
            names.append(override.get("name") or override.get("ids", ["?"])[0])
    resolved["override"] = ", ".join(names) or None
    return resolved


def _candidates(image: str | None, served_id: str | None, root: str | None) -> list[str]:
    out = []
    rid = resource_id(image)
    if rid:
        out.append(normalize_id(rid)[4:])
    for value in (served_id, root):
        if isinstance(value, str) and value.strip():
            out.append(normalize_id(value))
    return out


def _find(families: list[dict[str, Any]], value: str) -> tuple[dict[str, Any] | None, str]:
    """(record, "exact" or "pattern") for one normalized id, or (None, "")."""
    for record in families:
        if value in _exact_ids(record):
            return record, "exact"
    for record in families:
        if any(re.fullmatch(p, value) for p in (record.get("match") or {}).get("patterns", [])):
            return record, "pattern"
    return None, ""


def match(
    table: Mapping[str, Any],
    *,
    image: str | None = None,
    served_id: str | None = None,
    root: str | None = None,
    family: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """(record, how) for a target. how: "family", "image", "image_pattern", "served_id",
    "served_id_pattern", "root", "root_pattern" or "none" (record None). The image's NGC
    repository is tried before the served id, because NIM_SERVED_MODEL_NAME can rename the model.
    An unknown --family is ScriptError unknown_family (exit 2). The record is resolved: overrides
    that match the image, the served id or the root are applied."""
    families = table["families"]
    candidates = _candidates(image, served_id, root)
    if family is not None:
        wanted = family.strip().lower()
        for record in families:
            if record["id"] == wanted:
                return resolve(record, candidates), "family"
        raise nsc.ScriptError(
            "unknown_family",
            f"no family {family!r} in the model-feature table",
            "Pass a family id from scripts/model_features.json (nimsmoke.py features lists the "
            "match for a model), or leave --family out",
            nsc.EXIT_INPUT,
        )
    rid = resource_id(image)
    if rid:
        record, how = _find(families, normalize_id(rid)[4:])
        if record is not None:
            return resolve(record, candidates), "image" if how == "exact" else "image_pattern"
    for label, value in (("served_id", served_id), ("root", root)):
        if isinstance(value, str) and value.strip():
            record, how = _find(families, normalize_id(value))
            if record is not None:
                return resolve(record, candidates), label if how == "exact" else f"{label}_pattern"
    return None, "none"


def family_warning(
    table: Mapping[str, Any],
    family: str,
    *,
    image: str | None = None,
    served_id: str | None = None,
    root: str | None = None,
) -> str | None:
    """A warning when --family names a family whose ids and patterns match neither the image,
    the served id nor its root: its verdicts and sources then hold only if the NIM runs a model
    of that family under another name (NIM_SERVED_MODEL_NAME), which nothing here can confirm.
    None when one of them matches the family, or when nothing is known to compare."""
    wanted = family.strip().lower()
    record = next((item for item in table["families"] if item["id"] == wanted), None)
    candidates = _candidates(image, served_id, root)
    if record is None or not candidates:
        return None
    if any(_find([record], candidate)[0] is not None for candidate in candidates):
        return None
    shown = ", ".join(dict.fromkeys(value.strip() for value in (served_id, root, image)
                                    if isinstance(value, str) and value.strip()))
    return (
        f"--family {wanted} was applied by hand, and its ids and patterns match none of "
        f"{shown}: its verdicts and sources hold only if this NIM runs a {record.get('name')} "
        "model under another name (NIM_SERVED_MODEL_NAME). If it does not, rerun without "
        "--family and pass --documented FEATURE=URL from this model's own docs."
    )


# --- switches and requests ------------------------------------------------------------------


def _bad_switch(message: str) -> nsc.ScriptError:
    return nsc.ScriptError("bad_input", message, SWITCH_SYNTAX, nsc.EXIT_INPUT)


def parse_switch(text: str) -> dict[str, Any]:
    """--reasoning-on / --reasoning-off syntax -> {"method", "key", "value"}:
    "system=/think", "user_suffix= /no_think" (the value is kept as given, leading space too),
    "kwargs.enable_thinking=false" (the value is read as JSON when it parses, else as text),
    "effort=low" and "default" (send nothing). ScriptError bad_input otherwise."""
    if not isinstance(text, str) or not text.strip():
        raise _bad_switch("empty reasoning switch")
    if len(text) > MAX_SWITCH_TEXT:
        raise _bad_switch(f"the reasoning switch is longer than {MAX_SWITCH_TEXT} characters")
    if text.strip() == "default":
        return {"method": "default", "key": None, "value": None}
    if "=" not in text:
        raise _bad_switch(f"{text!r} is not a reasoning switch")
    head, value = text.split("=", 1)
    head = head.strip()
    if head == "system":
        if not value.strip():
            raise _bad_switch("system= needs the system prompt text")
        return {"method": "system", "key": None, "value": value.strip()}
    if head == "user_suffix":
        if not value.strip():
            raise _bad_switch("user_suffix= needs the text to add to the user message")
        return {"method": "user_suffix", "key": None, "value": value}
    if head == "effort":
        level = value.strip()
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,15}", level):
            raise _bad_switch(f"effort={level!r}: use a level word such as low, high or none")
        return {"method": "effort", "key": None, "value": level}
    if head.startswith("kwargs."):
        key = head[len("kwargs."):]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", key):
            raise _bad_switch(f"kwargs.{key}: the key must be a name")
        if key == "chat_template":
            raise _bad_switch("kwargs.chat_template is never sent (HTTP 400 on NIM)")
        raw = value.strip()
        try:
            parsed: Any = json.loads(raw)
        except ValueError:
            parsed = raw
        if isinstance(parsed, (list, dict)):
            raise _bad_switch(f"kwargs.{key}: the value must be one JSON scalar")
        return {"method": "kwargs", "key": key, "value": parsed}
    raise _bad_switch(f"{head!r} is not a switch method")


def normalize_switch(switch: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A table switch with all three keys ({"method", "key", "value"}), or None."""
    if switch is None:
        return None
    return {"method": switch["method"], "key": switch.get("key"), "value": switch.get("value")}


def apply_switch(body: dict[str, Any], switch: Mapping[str, Any] | None) -> dict[str, Any]:
    """Add one switch to a chat request body (in place; also returned). system: the first
    message becomes this system prompt; user_suffix: the text is added to the last user
    message (to its first text part when the content is a list); kwargs: chat_template_kwargs;
    effort: reasoning_effort; default or None: nothing."""
    if not switch or switch.get("method") in (None, "default"):
        return body
    method, value = switch["method"], switch.get("value")
    messages = body.setdefault("messages", [])
    if method == "system":
        if messages and messages[0].get("role") == "system":
            messages[0] = {**messages[0], "content": value}
        else:
            messages.insert(0, {"role": "system", "content": value})
    elif method == "user_suffix":
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = content + str(value)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        part["text"] = str(part.get("text", "")) + str(value)
                        break
            break
    elif method == "kwargs":
        kwargs = dict(body.get("chat_template_kwargs") or {})
        kwargs[switch["key"]] = value
        body["chat_template_kwargs"] = kwargs
    elif method == "effort":
        body["reasoning_effort"] = value
    return body


def _bad_documented(message: str) -> nsc.ScriptError:
    return nsc.ScriptError(
        "bad_input",
        message,
        f"Use --documented FEATURE=URL with FEATURE one of {', '.join(FEATURES)} and URL the "
        "https page that documents it for this model",
        nsc.EXIT_INPUT,
    )


def parse_documented(text: str) -> tuple[str, str]:
    """--documented FEATURE=URL -> (feature, url): a valid https link (_split_link) with a host
    name, no user information, no spaces, no private-repository or internal-image name."""
    if not isinstance(text, str) or "=" not in text:
        raise _bad_documented(f"{text!r} is not FEATURE=URL")
    feature, url = (part.strip() for part in text.split("=", 1))
    if feature not in FEATURES:
        raise _bad_documented(f"{feature!r} is not a feature --documented takes")
    parsed = _split_link(url)
    if (
        parsed is None
        or parsed.scheme != "https"
        or not parsed.hostname
        or "." not in parsed.hostname
        or parsed.username
        or parsed.password
        or any(char.isspace() for char in url)
        or len(url) > 500
    ):
        raise _bad_documented(f"--documented {feature}: the URL must be a public https link")
    if any(marker in url.lower() for marker in PRIVATE_MARKERS):
        raise _bad_documented(f"--documented {feature}: cite a public page, not an internal one")
    return feature, url


# --- tier B: the image default and the deployment --------------------------------------------


def _argv(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return shlex.split(value)
        except ValueError:
            return value.split()
    return [str(item) for item in value]


def _name(value: str | None) -> str | None:
    """A parser name as given, or None when it is not a plain name or looks like a key."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value):
        return None
    return None if nsc.looks_like_key(value) else value


def parser_flags(value: str | Iterable[str] | None) -> dict[str, Any]:
    """The parser flags in a NIM_PASSTHROUGH_ARGS string or an argument list: auto_tool_choice,
    tool_parser, reasoning_parser, structured_outputs, exclude_tools_none. No other value of the
    arguments is kept."""
    return _scan_flags(value)[0]


def _scan_flags(value: str | Iterable[str] | None) -> tuple[dict[str, Any], set[str]]:
    """parser_flags, and the keys the arguments name (--no-enable-auto-tool-choice names
    auto_tool_choice, so container arguments can turn it off)."""
    named: set[str] = set()
    flags: dict[str, Any] = {
        "auto_tool_choice": False,
        "tool_parser": None,
        "reasoning_parser": None,
        "structured_outputs": False,
        "exclude_tools_none": False,
    }
    argv = _argv(value)
    for index, item in enumerate(argv):
        name, _, inline = item.partition("=")
        following = argv[index + 1] if index + 1 < len(argv) else None
        given = inline if inline else following
        if name == "--enable-auto-tool-choice":
            flags["auto_tool_choice"] = True
            named.add("auto_tool_choice")
        elif name == "--no-enable-auto-tool-choice":
            flags["auto_tool_choice"] = False
            named.add("auto_tool_choice")
        elif name == "--tool-call-parser":
            flags["tool_parser"] = _name(given)
        elif name == "--reasoning-parser":
            flags["reasoning_parser"] = _name(given)
        elif name.startswith("--structured-outputs-config"):
            flags["structured_outputs"] = True
        elif name == "--exclude-tools-when-tool-choice-none":
            flags["exclude_tools_none"] = True
    return flags, named


def _tools_on(flags: Mapping[str, Any] | None) -> bool:
    """vLLM answers tool_choice "auto" only with both --enable-auto-tool-choice and
    --tool-call-parser; with one of them it returns the 400 the tools check reads."""
    return bool(flags and flags.get("tool_parser") and flags.get("auto_tool_choice"))


def tier_b(
    image_default: str | None,
    effective: str | None,
    args: Iterable[str] | None = None,
) -> dict[str, Any]:
    """What the image default and this deployment say about the parsers (tier B).

    image_default: the image's own NIM_PASSTHROUGH_ARGS (None: unknown or not set).
    effective: the value the deployment sets (None: not set, so the image default applies).
    args: the container arguments, which NIM adds at CLI priority.
    Only parser names and booleans are returned, never the raw strings."""
    default_flags = parser_flags(image_default) if image_default is not None else None
    set_in_deployment = effective is not None
    base = effective if set_in_deployment else image_default
    set_flags = parser_flags(base)
    effective_flags = dict(set_flags)
    extra, named = _scan_flags(args)
    for key, value in extra.items():
        if value or key in named:
            effective_flags[key] = value
    overridden = (
        set_in_deployment and image_default is not None and _argv(effective) != _argv(image_default)
    )
    default_on = _tools_on(default_flags)
    tool_on = _tools_on(effective_flags)
    default_reasoning = (default_flags or {}).get("reasoning_parser")
    return {
        "image_default_known": image_default is not None,
        "set_in_deployment": set_in_deployment,
        "args_known": args is not None,
        "overridden": overridden,
        "image_default": default_flags,
        "effective": effective_flags,
        "tools_on_by_default": True if default_on else None,
        "tools_dropped_by_override": bool(
            overridden and default_on and not _tools_on(set_flags) and not tool_on
        ),
        "tools_in_deployment": True if tool_on else None,
        "tools_in_set_value": _tools_on(set_flags) if set_in_deployment else None,
        "reasoning_parser_by_default": default_reasoning,
        "reasoning_parser_dropped_by_override": bool(
            overridden and default_reasoning and not effective_flags["reasoning_parser"]
        ),
        "reasoning_parser_in_deployment": effective_flags["reasoning_parser"],
    }


# --- the plan -------------------------------------------------------------------------------


def _urls(*groups: Iterable[Any] | None) -> list[str]:
    out: list[str] = []
    for group in groups:
        for item in group or []:
            url = item.get("url") if isinstance(item, dict) else item
            if isinstance(url, str) and url not in out:
                out.append(url)
    return out


def _entry(check: str, decision: str, reason: str, **fields: Any) -> dict[str, Any]:
    entry = {
        "check": check,
        "decision": decision,
        "reason": reason,
        "sources": _urls(fields.pop("sources", [])),
        "conflicts": _urls(fields.pop("conflicts", [])),
        "request": fields.pop("request", {}),
        "known_issues": fields.pop("known_issues", []),
        "hint": fields.pop("hint", None),
        "warnings": fields.pop("warnings", []),
    }
    entry.update(fields)
    return entry


def _issues(record: Mapping[str, Any] | None, check: str) -> list[str]:
    return _urls(
        item for item in (record or {}).get("known_issues", []) if check in item.get("affects", [])
    )


def _for_generation(value: Any, generation: str | None) -> bool | None:
    """default_on for a generation: a scalar applies to all; a dict picks one; with the
    generation unknown, a value both generations agree on."""
    if not isinstance(value, dict):
        return value if isinstance(value, bool) else None
    if generation in GENERATIONS:
        found = value.get(generation)
        return found if isinstance(found, bool) else None
    values = {value.get(key) for key in GENERATIONS if key in value}
    return values.pop() if len(values) == 1 and isinstance(next(iter(values)), bool) else None


def _profile_strings(profile: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(profile, Mapping):
        return []
    out = []
    for key in ("id", "profile_id", "selectedModelProfileId", "name", "profile_name",
                "description"):
        value = profile.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value.strip().lower())
    return out


def _pipeline_parallel(profile: Mapping[str, Any] | None) -> int | None:
    """The pp<N> tag of the active profile's name or id; None when no profile text shows it
    (a 1.x /v1/metadata gives only a hashed profile id)."""
    for text in _profile_strings(profile):
        found = re.search(r"(?:^|[-_])pp(\d+)(?:[-_]|$)", text)
        if found:
            return int(found.group(1))
    return None


def _profile_id(profile: Mapping[str, Any] | None) -> str | None:
    """The active profile's id (a hash in the support matrix), not its name."""
    if not isinstance(profile, Mapping):
        return None
    for key in ("id", "profile_id", "selectedModelProfileId"):
        value = profile.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _text_only_profile(image: Mapping[str, Any], profile: Mapping[str, Any] | None) -> bool:
    """Whether the active profile is listed as text only: the same id or name, or one id is a
    prefix (8 characters or more) of the other, as the support matrix shortens ids."""
    listed = [item.lower() for item in image.get("text_only_profiles", [])]
    for text in _profile_strings(profile):
        for item in listed:
            if text == item:
                return True
            short, full = sorted((text, item), key=len)
            if len(short) >= 8 and full.startswith(short):
                return True
    return False


def _switch_hint(
    record: Mapping[str, Any] | None,
    feature: str,
    generation: str | None,
    backend: str | None,
    signals: Mapping[str, Any],
    engine: Mapping[str, Any],
) -> str | None:
    table = ((record or {}).get("server_switches") or {}).get(feature) or {}
    if generation == "1.x":
        keys = ["1.x"]
    elif backend in ("vllm", "sglang"):
        keys = [f"2.x-{backend}"] if generation == "2.x" else [f"2.x-{backend}", "1.x"]
    else:
        keys = ["2.x-vllm", "2.x-sglang"] if generation == "2.x" else list(SWITCH_KEYS)
    parts = [f"{SWITCH_LABELS[key]}: {table[key]}" for key in keys if key in table]
    if not parts and feature == "tools":
        generic = (engine.get("tool_parser_flags") or {}).get("hint")
        if generic:
            parts = [generic.rstrip(".")]
    if not parts:
        return None
    text = "Documented server switch: " + "; ".join(parts) + "."
    if signals.get("set_in_deployment"):
        text += PASSTHROUGH_NOTE
    return text + DEPLOY_NOTE


def _reasons_by_default(record: Mapping[str, Any] | None) -> bool:
    if record is None:
        return True
    return ((record.get("reasoning") or {}).get("default") or "unknown") in (
        "on", "always", "unknown", "prompt"
    )


def _not_reasoning(record: Mapping[str, Any] | None) -> bool:
    return record is not None and (record.get("reasoning") or {}).get("default") == "none"


class _Context:
    """The inputs of one decide() call, with the record's feature objects at hand."""

    def __init__(
        self,
        record: Mapping[str, Any] | None,
        table: Mapping[str, Any],
        documented: Mapping[str, str] | None,
        switches: Mapping[str, Mapping[str, Any]] | None,
        generation: str | None,
        backend: str | None,
        profile: Mapping[str, Any] | None,
        signals: Mapping[str, Any] | None,
        hosted: bool,
        image: str | None = None,
    ) -> None:
        self.record = record
        self.target_image = image if isinstance(image, str) and image.strip() else None
        self.own_image = own_image(record, self.target_image)
        self.engine = table.get("engine") or {}
        self.documented = dict(documented or {})
        self.switches = dict(switches or {})
        self.generation = generation_of(generation) if generation else None
        # The NIM LLM 1.x rules (pipeline parallelism, the image deciding tool calling) are
        # for LLM NIMs only: a VLM NIM numbers its own releases (VLM NIM 1.7.0), so a 1.x tag
        # on a vlm record is not NIM LLM 1.x.
        self.llm_1x = self.generation == "1.x" and (record or {}).get("kind") != "vlm"
        record_backend = (record or {}).get("backend")
        self.backend = backend if backend in ("vllm", "sglang") else (
            record_backend if record_backend in ("vllm", "sglang") else None
        )
        self.profile = profile
        self.signals = dict(signals or {})
        self.hosted = bool(hosted)
        self.reasoning = dict((record or {}).get("reasoning") or {})
        self.tools = dict((record or {}).get("tools") or {})
        self.image = dict((record or {}).get("image") or {})
        self.warnings: list[str] = []

    def engine_sources(self, key: str) -> list[str]:
        return _urls((self.engine.get(key) or {}).get("sources"))

    def quiet_switch(self) -> dict[str, Any] | None:
        """The off switch the core checks send so they test the feature, not the reasoning."""
        if self.record is None or _not_reasoning(self.record):
            return None
        switch = normalize_switch(self.reasoning.get("off"))
        if switch and switch["method"] != "default":
            return switch
        return None

    def reasons_on(self) -> bool:
        """The record says the model reasons by default (on) or always."""
        return self.record is not None and self.reasoning.get("default") in ("on", "always")

    def reasoning_budget(self, floor: int) -> int:
        """The max_tokens of a core check that runs with the model's reasoning on: `floor`, or
        the record's reasoning budget when it is larger and the model reasons by default
        (GLM-5.3-Flash: "At the default effort, use a max_tokens of at least 8192")."""
        if not self.reasons_on():
            return floor
        return max(floor, int(self.reasoning.get("max_tokens") or 0))

    def light_switch(self) -> dict[str, Any] | None:
        """The record's on switch when it only lowers the reasoning effort (LIGHT_EFFORTS), for
        a core check of a model whose reasoning cannot be turned off; else None."""
        if not self.reasons_on():
            return None
        switch = normalize_switch(self.reasoning.get("on"))
        if switch and switch["method"] == "effort" and str(switch["value"]) in LIGHT_EFFORTS:
            return switch
        return None


def _pipeline_parallel_rule(ctx: _Context) -> bool:
    """NIM 1.x: tool calling and structured generation are not supported with pipeline
    parallelism above 1. True when the active profile shows pp above 1. When the parallelism
    is not visible (a 1.x /v1/metadata gives only the profile id), the check runs and the plan
    warns once (PP_UNKNOWN). Not for a VLM NIM's own 1.x releases (ctx.llm_1x)."""
    if not ctx.llm_1x:
        return False
    parallel = _pipeline_parallel(ctx.profile)
    if parallel is None:
        ctx.warnings.append(PP_UNKNOWN)
        return False
    return parallel > 1


def _tools_documented(ctx: _Context) -> tuple[str | None, list[Any], list[Any]]:
    """(value, sources, conflicts): value yes, conflict, agent, no, unknown or None (no record)."""
    value = ctx.tools.get("documented", "unknown") if ctx.record is not None else None
    sources: list[Any] = list(ctx.tools.get("sources") or [])
    conflicts: list[Any] = list(ctx.tools.get("conflicts") or [])
    features = ctx.signals.get("metadata_features") or {}
    if isinstance(features.get("tool_calling"), bool):
        url = ctx.signals.get("metadata_url") or METADATA_DOC
        if features["tool_calling"]:
            value = "yes" if value in (None, "unknown") else value
            sources.append(url)
        else:
            value, sources = "no", [url, *sources]
    agent = ctx.documented.get("tools")
    if agent:
        if value == "no":
            ctx.warnings.append(
                "--documented tools was not used: the table cites a source that says this "
                "model has no tool calling."
            )
        else:
            value = "agent" if value in (None, "unknown") else value
            sources.append(agent)
    return value, sources, conflicts


def own_image(record: Mapping[str, Any] | None, image: str | None) -> bool:
    """Whether `image` is one of the record's own NIM images: its NGC repository is one the
    record lists, or matches the record's patterns, as match() finds a record by its image."""
    rid = resource_id(image)
    if record is None or rid is None:
        return False
    found, _ = _find([dict(record)], normalize_id(rid)[4:])
    return found is not None


def _needs_own_image(ctx: _Context) -> bool:
    """Whether the record's default_on true is NIM 1.x's (the generation is 1.x, or unknown
    with a 1.x value of true), which holds only in the model's own NIM container: the 1.15.0
    docs enable tool calling automatically in LLM-specific NIM containers, and the multi-LLM
    NIM takes it as an option."""
    if ctx.generation == "2.x" or (ctx.generation == "1.x" and not ctx.llm_1x):
        return False
    return _for_generation(ctx.tools.get("default_on"), "1.x") is True


def _parser_missing(ctx: _Context) -> dict[str, Any]:
    """What the tools check reports when vLLM answers that no tool parser is configured: fail
    when the image turns the parser on by default (or an override dropped it), not applicable
    when it is opt-in or unknown (with a warning when unknown, which also names a deployment
    NIM_PASSTHROUGH_ARGS as the likely cause when the image default cannot be read). The
    record's NIM 1.x default_on true counts only for the model's own image (own_image): with
    the image unknown or another one (the multi-LLM NIM), the default is unknown."""
    if ctx.hosted:
        return {"status": "fail", "reason": "parser_not_enabled", "warning": None}
    default_on = _for_generation(ctx.tools.get("default_on"), ctx.generation)
    if ctx.signals.get("tools_dropped_by_override"):
        return {"status": "fail", "reason": "parser_dropped_by_override", "warning": None}
    if ctx.signals.get("tools_on_by_default") is True:
        return {"status": "fail", "reason": "parser_not_enabled", "warning": None}
    warning = None
    if default_on is True and not ctx.own_image and _needs_own_image(ctx):
        default_on = None
        if ctx.target_image is None:
            which = ("The image is not known here (pass --image with the image the NIM runs), "
                     "so")
        else:
            which = f"The image {nsc.clip(ctx.target_image, 120)} is not one of them, so"
        warning = OWN_IMAGE_ONLY_1X.format(which=which)
    if default_on is True:
        return {"status": "fail", "reason": "parser_not_enabled", "warning": None}
    if default_on is False:
        return {"status": "not_applicable", "reason": "not_enabled_in_deployment", "warning": None}
    warning = warning or (
        "Tool calling is documented for this model, but the docs do not say whether this image "
        "turns the tool parser on by default; it is off in this deployment."
    )
    signals = ctx.signals
    if (signals.get("set_in_deployment") and not signals.get("image_default_known")
            and not signals.get("tools_in_deployment")
            and signals.get("tools_in_set_value") is not True):
        warning += (
            " This deployment sets NIM_PASSTHROUGH_ARGS without both tool flags "
            "(--enable-auto-tool-choice and --tool-call-parser), and that replaces the image's "
            "default value: if the default turned the parser on, this override dropped it (the "
            "image default could not be read here)."
        )
    return {"status": "not_applicable", "reason": "not_enabled_in_deployment", "warning": warning}


def _tools_request(ctx: _Context) -> dict[str, Any]:
    mode = [normalize_switch(item) for item in ctx.tools.get("mode") or []]
    max_tokens = ctx.tools.get("max_tokens")
    if max_tokens is None:
        off = normalize_switch(ctx.reasoning.get("off"))
        reasoning_off = off is not None and off in mode
        max_tokens = (
            TOOL_TOKENS_REASONING
            if ctx.record is not None and _reasons_by_default(ctx.record) and not reasoning_off
            else TOOL_TOKENS
        )
    return {
        "max_tokens": max_tokens,
        "sampling": dict(ctx.tools.get("sampling") or {"temperature": 0}),
        "mode": mode,
    }


def _decide_tools(ctx: _Context, check: str) -> dict[str, Any]:
    issues = _issues(ctx.record, check)
    if ctx.record is not None and ctx.record.get("status") == "out_of_scope":
        return _entry(check, "not_applicable", "out_of_scope", known_issues=issues)
    value, sources, conflicts = _tools_documented(ctx)
    if (check == "tools_parallel" and value in (None, "unknown")
            and ctx.documented.get("tools_parallel")):
        # --documented tools_parallel=URL is enough for this check; the other tool checks
        # still need tool calling documented (the record, the Metadata API or --documented
        # tools). A documented no still wins: tools "no" is not changed here, and parallel
        # "no" makes the check not applicable below.
        value = "agent"
    if value is None:
        return _entry(check, "not_applicable", "no_capability_record")
    if value == "no":
        return _entry(check, "not_applicable", "not_documented_for_model", sources=sources,
                      conflicts=conflicts, known_issues=issues)
    if value == "unknown":
        return _entry(check, "not_applicable", "not_documented", sources=sources,
                      known_issues=issues)
    if _pipeline_parallel_rule(ctx):
        return _entry(check, "not_applicable", "pipeline_parallel_unsupported",
                      sources=ctx.engine_sources("pipeline_parallel_1x"))
    reason = {"yes": "documented", "conflict": "documented_conflict"}.get(value,
                                                                           "documented_by_agent")
    hint = None if ctx.hosted else _switch_hint(
        ctx.record, "tools", ctx.generation, ctx.backend, ctx.signals, ctx.engine
    )
    request = _tools_request(ctx)
    warnings: list[str] = []
    if conflicts and value == "conflict":
        warnings.append("The sources disagree on tool calling for this model; both are listed.")
    if check == "tools":
        request["named_diagnostic"] = ctx.tools.get("named", "yes") == "yes"
    elif check == "tools_stream":
        if ctx.tools.get("stream") == "no":
            return _entry(check, "not_applicable", "not_documented_for_model", sources=sources,
                          known_issues=issues)
        request.update({"stream": True, "retry": int(ctx.tools.get("stream_retry") or 0)})
    elif check == "tools_parallel":
        parallel = ctx.tools.get("parallel", "unknown") if ctx.record is not None else "unknown"
        metadata = (ctx.signals.get("metadata_features") or {}).get("parallel_tool_calling")
        # As for tool calling, a "no" in the record wins over a Metadata API yes.
        if isinstance(metadata, bool) and parallel != "no":
            parallel = "yes" if metadata else "no"
        agent = ctx.documented.get("tools_parallel")
        if agent and parallel != "no":
            parallel, reason = "yes", "documented_by_agent"
            sources = [*sources, agent]
        if parallel == "no":
            return _entry(check, "not_applicable", "not_documented_for_model", sources=sources,
                          conflicts=conflicts, known_issues=issues)
        if parallel == "info":
            return _entry(check, "not_applicable", "information_only", sources=sources,
                          conflicts=conflicts, known_issues=issues)
        if parallel != "yes":
            return _entry(check, "not_applicable", "not_documented", sources=sources,
                          known_issues=issues)
        request["parallel"] = True
    elif check == "tools_none" and ctx.tools.get("none") == "no":
        return _entry(check, "not_applicable", "not_documented_for_model", sources=sources,
                      known_issues=issues)
    return _entry(check, "test", reason, sources=sources, conflicts=conflicts, request=request,
                  known_issues=issues, hint=hint, warnings=warnings,
                  if_parser_missing=_parser_missing(ctx))


# (default, on switch, off switch, sources, a reason that makes both directions not applicable)
_Setup = tuple[str, dict[str, Any] | None, dict[str, Any] | None, list[Any], str | None]


def _reasoning_setup(ctx: _Context) -> _Setup:
    """(default, on, off, sources, na_reason): na_reason is set when both directions are n/a."""
    agent = ctx.documented.get("reasoning")
    agent_on, agent_off = ctx.switches.get("on"), ctx.switches.get("off")
    if ctx.record is None:
        if not agent:
            if agent_on or agent_off:
                ctx.warnings.append(
                    "--reasoning-on and --reasoning-off need --documented reasoning=URL for a "
                    "model with no capability record."
                )
            return "unknown", None, None, [], "no_capability_record"
        return "unknown", normalize_switch(agent_on), normalize_switch(agent_off), [agent], None
    default = ctx.reasoning.get("default", "unknown")
    sources: list[Any] = list(ctx.reasoning.get("sources") or [])
    if agent:
        sources.append(agent)
    if default == "none":
        if agent or agent_on or agent_off:
            ctx.warnings.append(
                "The reasoning switches were not used: the table records this model as not a "
                "reasoning model."
            )
        return default, None, None, sources, "not_a_reasoning_model"
    if ctx.reasoning.get("detectable") is False:
        return default, None, None, sources, "reasoning_not_detectable"
    on = normalize_switch(agent_on) or normalize_switch(ctx.reasoning.get("on"))
    if on is None and default in ("on", "always"):
        on = {"method": "default", "key": None, "value": None}
    off = normalize_switch(ctx.reasoning.get("off"))
    if agent_off:
        if default == "always":
            ctx.warnings.append(
                "--reasoning-off was not used: the sources say this model always reasons."
            )
        else:
            off = normalize_switch(agent_off)
    return default, on, off, sources, None


def _decide_reasoning(ctx: _Context, check: str, setup: _Setup) -> dict[str, Any]:
    default, on, off, sources, na_reason = setup
    issues = _issues(ctx.record, check)
    if ctx.record is not None and ctx.record.get("status") == "out_of_scope":
        return _entry(check, "not_applicable", "out_of_scope", known_issues=issues)
    if na_reason:
        return _entry(check, "not_applicable", na_reason, sources=sources, known_issues=issues)
    hint = None if ctx.hosted else _switch_hint(
        ctx.record, "reasoning", ctx.generation, ctx.backend, ctx.signals, ctx.engine
    )
    reason = "documented" if ctx.record is not None else "documented_by_agent"
    max_tokens = int(ctx.reasoning.get("max_tokens") or DEFAULT_REASONING_TOKENS)
    if check == "reasoning_on":
        if on is None:
            return _entry(check, "not_applicable", "no_documented_switch", sources=sources,
                          known_issues=issues, hint="Pass --reasoning-on SWITCH from the docs.")
        if ctx.reasoning.get("parser_default_on") is False and not ctx.signals.get(
            "reasoning_parser_in_deployment"
        ):
            seen = ctx.signals.get("set_in_deployment") or ctx.signals.get("args_known")
            warnings = [] if seen else [
                "The deployment's arguments are not visible here, so the reasoning parser "
                "cannot be confirmed."
            ]
            return _entry(check, "not_applicable", "not_enabled_in_deployment", sources=sources,
                          known_issues=issues, hint=hint, warnings=warnings)
        request = {
            "switch": on,
            "sampling": dict(ctx.reasoning.get("sampling_on") or {}),
            "max_tokens": max_tokens,
            "stream": False,
            "expect_where": list(ctx.reasoning.get("where") or []),
        }
        return _entry(check, "test", reason, sources=sources, request=request,
                      known_issues=issues, hint=hint)
    if default == "always":
        return _entry(check, "not_applicable", "always_on", sources=sources, known_issues=issues)
    if off is None:
        off_reason = ctx.reasoning.get("off_reason") or "no_documented_switch"
        return _entry(check, "not_applicable", off_reason, sources=sources, known_issues=issues,
                      hint=None if off_reason == "always_on" else
                      "Pass --reasoning-off SWITCH from the docs.")
    if check == "reasoning_off_stream" and not ctx.reasoning.get("prefills_off"):
        return _entry(check, "not_applicable", "no_prefilled_think_block", sources=sources,
                      known_issues=issues)
    request = {
        "switch": off,
        "sampling": dict(ctx.reasoning.get("sampling_off") or {"temperature": 0}),
        "max_tokens": OFF_TOKENS,
        "stream": check == "reasoning_off_stream",
    }
    return _entry(check, "test", reason, sources=sources, request=request, known_issues=issues,
                  hint=hint)


def _decide_image(ctx: _Context) -> dict[str, Any]:
    issues = _issues(ctx.record, "image")
    text_only = ctx.engine_sources("image_text_only")
    agent = ctx.documented.get("image")
    if ctx.record is None:
        if not agent:
            return _entry("image", "not_applicable", "no_capability_record")
        return _entry("image", "test", "documented_by_agent", sources=[agent],
                      request={"max_tokens": IMAGE_TOKENS_REASONING, "switch": None,
                               "max_images": 1})
    if ctx.record.get("status") == "out_of_scope":
        return _entry("image", "not_applicable", "out_of_scope", known_issues=issues)
    documented = ctx.image.get("documented", "unknown")
    sources = list(ctx.image.get("sources") or [])
    if ctx.record.get("kind") != "vlm" or documented == "no":
        if agent:
            ctx.warnings.append(
                "--documented image was not used: the table records this model as text only."
            )
        return _entry("image", "not_applicable", "text_only_model", sources=[*sources, *text_only],
                      known_issues=issues)
    if documented == "unknown":
        if not agent:
            return _entry("image", "not_applicable", "not_documented", sources=sources,
                          known_issues=issues)
        sources.append(agent)
    if _text_only_profile(ctx.image, ctx.profile):
        return _entry("image", "not_applicable", "text_only_profile", sources=sources,
                      known_issues=issues)
    warnings = []
    extra: dict[str, Any] = {}
    if ctx.image.get("text_only_profiles"):
        # the documented 400 of those profiles makes the check not applicable (nsc_checks)
        extra["if_text_only_reply"] = {"status": "not_applicable", "reason": "text_only_profile",
                                       "reply": TEXT_ONLY_REPLY}
        if not _profile_id(ctx.profile):
            warnings.append(
                "Some profiles of this model take text only, and the active profile's id is not "
                "known here; their documented HTTP 400 makes the image check not applicable."
            )
    quiet = ctx.quiet_switch()
    if quiet:
        max_tokens, switch = IMAGE_TOKENS, quiet
    elif _reasons_by_default(ctx.record):
        # reasoning stays on: the record's budget, and its lighter effort when it has one
        max_tokens = ctx.reasoning_budget(IMAGE_TOKENS_REASONING)
        switch = ctx.light_switch()
    else:
        max_tokens, switch = IMAGE_TOKENS, None
    request = {
        "max_tokens": max_tokens,
        "switch": switch,
        "max_images": int(ctx.image.get("max_images") or 1),
    }
    reason = "documented" if documented in ("yes", "conflict") else "documented_by_agent"
    return _entry("image", "test", reason, sources=sources, request=request, known_issues=issues,
                  warnings=warnings, **extra)


def _decide_core(ctx: _Context, check: str) -> dict[str, Any]:
    record = ctx.record
    issues = _issues(record, check)
    out_of_scope = record is not None and record.get("status") == "out_of_scope"
    if check == "health":
        if ctx.hosted:
            return _entry(check, "not_applicable", "hosted_no_health_route",
                          sources=ctx.engine_sources("health"))
        return _entry(check, "test", "core_check", sources=ctx.engine_sources("health"),
                      known_issues=issues)
    if check == "models":
        return _entry(check, "test", "core_check", sources=ctx.engine_sources("models"),
                      known_issues=issues)
    if out_of_scope:
        return _entry(check, "not_applicable", "out_of_scope",
                      sources=_urls((record or {}).get("out_of_scope_sources")),
                      known_issues=issues)
    chat_tokens = CHAT_TOKENS_REASONING if _reasons_by_default(record) else CHAT_TOKENS
    if check == "chat":
        return _entry(check, "test", "core_check", sources=ctx.engine_sources("chat"),
                      request={"max_tokens": chat_tokens}, known_issues=issues)
    if check in ("stream", "json_schema"):
        feature = dict((record or {}).get(check) or {})
        if feature.get("documented") == "no":
            return _entry(check, "not_applicable", "not_documented_for_model",
                          sources=feature.get("sources"), known_issues=issues)
        if check == "stream":
            return _entry(check, "test", "engine_feature", sources=ctx.engine_sources("stream"),
                          request={"max_tokens": chat_tokens}, known_issues=issues)
        if _pipeline_parallel_rule(ctx):
            return _entry(check, "not_applicable", "pipeline_parallel_unsupported",
                          sources=ctx.engine_sources("pipeline_parallel_1x"))
        quiet = ctx.quiet_switch()
        own = normalize_switch(feature.get("switch"))
        if quiet:
            request = {"max_tokens": JSON_TOKENS, "switch": quiet}
        elif own:
            request = {"max_tokens": ctx.reasoning_budget(JSON_TOKENS_REASONING), "switch": own}
        elif _reasons_by_default(record):
            # reasoning stays on: the record's budget, and its lighter effort when it has one
            request = {"max_tokens": ctx.reasoning_budget(JSON_TOKENS_REASONING),
                       "switch": ctx.light_switch()}
        else:
            request = {"max_tokens": JSON_TOKENS, "switch": None}
        return _entry(check, "test", "engine_feature",
                      sources=[*ctx.engine_sources("json_schema"), *_urls(feature.get("sources"))],
                      request=request, known_issues=issues)
    return _entry(check, "test", "core_check", sources=ctx.engine_sources("concurrency"),
                  request={"max_tokens": CONCURRENCY_TOKENS, "switch": ctx.quiet_switch()},
                  known_issues=issues)


def select_checks(
    checks: Iterable[str] | None = None,
    skip: Iterable[str] | None = None,
    extended: bool = False,
) -> tuple[str, ...]:
    """The checks to plan, in CHECKS order: `checks` (default DEFAULT_CHECKS, plus
    EXTENDED_CHECKS with `extended`) minus `skip`. ScriptError bad_input for an unknown id."""
    wanted = list(checks) if checks is not None else [
        *DEFAULT_CHECKS, *(EXTENDED_CHECKS if extended else ())
    ]
    if extended and checks is not None:
        wanted += list(EXTENDED_CHECKS)
    dropped = list(skip or [])
    unknown = sorted({name for name in [*wanted, *dropped] if name not in CHECKS})
    if unknown:
        raise nsc.ScriptError(
            "bad_input",
            f"unknown check id: {', '.join(unknown)}",
            f"Use check ids from: {', '.join(CHECKS)}",
            nsc.EXIT_INPUT,
        )
    return tuple(name for name in CHECKS if name in wanted and name not in dropped)


def decide(
    record: Mapping[str, Any] | None,
    *,
    checks: Iterable[str] = DEFAULT_CHECKS,
    hosted: bool = False,
    documented: Mapping[str, str] | None = None,
    switches: Mapping[str, Mapping[str, Any]] | None = None,
    generation: str | None = None,
    backend: str | None = None,
    profile: Mapping[str, Any] | None = None,
    signals: Mapping[str, Any] | None = None,
    table: Mapping[str, Any] | None = None,
    image: str | None = None,
) -> dict[str, Any]:
    """The test plan for a resolved record (from match) or None (no capability record).

    checks: the check ids to plan (any order; the plan follows CHECKS).
    hosted: the hosted API (no health route).
    documented: {"tools" | "tools_parallel" | "reasoning" | "image": url} from --documented.
    switches: {"on": switch, "off": switch} from --reasoning-on / --reasoning-off (parse_switch).
    generation: the NIM version or major ("2.1.5", "1.x"); backend: "vllm" or "sglang".
    profile: the active profile ({"id", "name"}, from /v1/metadata) or None.
    signals: tier_b(...) output, optionally with "metadata_features" (the NIM Metadata API
    features object: tool_calling, parallel_tool_calling) and "metadata_url".
    table: the table the record came from (default: the shipped table), for its engine sources.
    image: the image the NIM runs, when known. A NIM 1.x default_on true holds only for the
    record's own images (own_image); with the image unknown on 1.x, the plan warns.
    """
    selected = select_checks(checks)
    source_table = table if table is not None else default_table()
    ctx = _Context(record, source_table, documented, switches, generation, backend, profile,
                   signals, hosted, image)
    if record is None and ctx.documented:
        ctx.warnings.append(
            "No capability record matches this model: only the features passed with "
            "--documented are tested, on the agent's sources."
        )
    elif record is None:
        ctx.warnings.append(
            "No capability record matches this model: the model-specific checks are not "
            "applicable. Ask the user for the page that documents a feature for this model and "
            "pass --documented FEATURE=URL; pass --family ID only when the served id renames "
            "(NIM_SERVED_MODEL_NAME) a model of a family the table has."
        )
    elif record.get("status") == "out_of_scope":
        ctx.warnings.append(
            f"{record.get('name', record.get('id'))} is out of scope for the chat smoke check "
            f"({record.get('out_of_scope_reason')}); only health and models run."
        )
    elif record.get("status") == "eos":
        ctx.warnings.append("This NIM is End of Support; its sources are archived docs.")
    if (record is not None and record.get("status") != "out_of_scope" and not hosted
            and ctx.llm_1x and ctx.target_image is None):
        ctx.warnings.append(IMAGE_UNKNOWN_1X)
    setup = _reasoning_setup(ctx)
    entries = []
    for check in selected:
        if check in TOOL_CHECKS:
            entries.append(_decide_tools(ctx, check))
        elif check in REASONING_CHECKS:
            entries.append(_decide_reasoning(ctx, check, setup))
        elif check == "image":
            entries.append(_decide_image(ctx))
        else:
            entries.append(_decide_core(ctx, check))
    return {
        "record": (record or {}).get("id"),
        "name": (record or {}).get("name"),
        "status": (record or {}).get("status"),
        "kind": (record or {}).get("kind"),
        "override": (record or {}).get("override"),
        "generation": ctx.generation,
        "backend": ctx.backend,
        "reviewed": source_table.get("reviewed"),
        "checks": entries,
        "warnings": list(dict.fromkeys(ctx.warnings)),
    }


def plan_entry(plan: Mapping[str, Any], check: str) -> dict[str, Any] | None:
    """The entry of one check in a plan, or None when it was not planned."""
    for entry in plan.get("checks", []):
        if isinstance(entry, dict) and entry.get("check") == check:
            return entry
    return None


def summary(table: Mapping[str, Any]) -> dict[str, Any]:
    """Counts and ids of a validated table, for --check and the features command."""
    families = table["families"]
    return {
        "schema_version": table["schema_version"],
        "reviewed": table["reviewed"],
        "docs": table.get("docs", {}),
        "families": len(families),
        "ids": [record["id"] for record in families],
        "out_of_scope": [r["id"] for r in families if r["status"] == "out_of_scope"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = nsc.JsonArgumentParser(
        description="The model-feature table of nim-smoke-check; import it. --check validates "
        "scripts/model_features.json.",
        epilog="Example: python3 scripts/nsc_features.py --check",
    )
    parser.add_argument(
        "--check", action="store_true", help="validate the table and print a summary"
    )
    args = parser.parse_args(argv)
    if not args.check:
        nsc.emit(True, "library", "done", {
            "note": "import this module; run it with --check to validate model_features.json"
        })
    try:
        table = load_table(TABLE)
    except nsc.ScriptError as exc:
        nsc.emit(False, "invalid", "fix_errors", exc.facts, [exc.as_error()], exc.exit_code)
    nsc.emit(True, "valid", "done", summary(table))


if __name__ == "__main__":
    main()
