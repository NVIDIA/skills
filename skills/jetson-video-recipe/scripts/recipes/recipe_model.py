#!/usr/bin/env python3
"""Resolve and validate the sole surface-neutral NVCodec recipe contract."""

# This single domain owner keeps recipe parsing, resolution, projection, and
# validation on one schema contract.
# pylint: disable=too-many-lines

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

if not sys.flags.ignore_environment or not sys.flags.no_user_site:
    raise SystemExit("invoke this producer with isolated Python: python3 -I")

SCHEMA_VERSION = "2.0"
KIND = "nvcodec-recipe"
ENVIRONMENT_KIND = "nvcodec-environment"
ENVIRONMENT_SCHEMA_VERSION = "1.2"
# Python-surface readiness required by ``check-live``, read from the established
# schema-1.2 ``installation.python.packages`` map.  Torch applies to the GPU
# buffer mode only.
PYNVC_DEPENDENCIES = (
    ("numpy", "NumPy", {}),
    ("pycuda", "PyCUDA", {"version": "2026.1"}),
)
TORCH_GPU_DEPENDENCY = (
    "torch",
    "Torch GPU mode",
    {
        "version": "2.9.1+cu130",
        "cuda_build": "13.0",
        "cuda_available": True,
        "sample_readiness": "ready",
    },
)
# Artifact identity of the optional capability-owned PyNvVideoCodec
# GetEncoderCaps report.  Pinned in this one place; see ``_capability_report``.
CAPABILITY_REPORT_IDENTITY = {
    "kind": "nvcodec-encoder-capability-report",
    "schema_version": "1.0",
}
# A supplied report is bound to the exact environment artifact it was produced
# from.
CAPABILITY_BINDING_FIELDS = (
    "path",
    "size_bytes",
    "sha256",
    "canonical_sha256",
    "kind",
    "schema_version",
    "mode",
    "selected_gpu",
    "interpreter",
    "interpreter_identity",
    "pynvc_identity",
)
# Encoder capability evidence has two accepted producers, and each keeps its own
# "no encode was attempted" spelling.  The accepted value is therefore selected
# by source rather than merged, so neither producer can borrow the other's
# vocabulary and neither may ever assert that an operation ran.
REPORT_OPERATION_STATUS = "not_tested"
ENVIRONMENT_OPERATION_STATUS = "not_verified_by_probe"
CATALOG_PATH = Path(__file__).resolve().parent / "data" / "encoder-intent-catalog.json"
CATALOG_REFERENCE = "scripts/recipes/data/encoder-intent-catalog.json"
INT_MAX = (1 << 31) - 1
UINT32_MAX = (1 << 32) - 1

CORE_KEYS = {
    "use_case",
    "codec",
    "width",
    "height",
    "format",
    "fps",
    "gpu",
    "output_requirement",
    "frame_count",
    "frames",
}
CONTROL_KEYS = {
    "preset",
    "tuning_info",
    "rc",
    "gop",
    "bf",
    "multipass",
    "lookahead",
    "aq",
    "temporalaq",
    "cq",
    "constqp",
    "vbvbufsize",
    "profile",
    "bitrate",
    "maxbitrate",
}
ALLOWED_KEYS = CORE_KEYS | CONTROL_KEYS
FORMATS = {
    "NV12",
    "YUV420",
    "NV16",
    "YUV444",
    "P010",
    "P210",
    "YUV444_16BIT",
    "ARGB",
    "ABGR",
}
TEN_BIT_FORMATS = {"P010", "P210", "YUV444_16BIT"}
YUV422_FORMATS = {"NV16", "P210"}
YUV444_FORMATS = {"YUV444", "YUV444_16BIT"}
KNOWN_YUV_FORMATS = {
    "NV12",
    "YUV420",
    "NV16",
    "YUV444",
    "P010",
    "P210",
    "YUV444_16BIT",
}
CODECS = {"h264", "hevc", "av1"}
CODEC_PROFILES = {
    "h264": {"baseline", "main", "high", "high444"},
    "hevc": {"main", "main10", "frext"},
    "av1": {"main"},
}
PRESETS = {f"p{index}" for index in range(1, 8)}
TUNINGS = {
    "high_quality",
    "low_latency",
    "ultra_low_latency",
    "lossless",
    "ultra_high_quality",
}
RATE_CONTROLS = {"cbr", "vbr", "constqp"}
MULTIPASS = {"disabled", "qres", "fullres"}
PROFILES = set().union(*CODEC_PROFILES.values())

# These are exact AppEncCuda ``-if`` spellings from the released sample.  In
# particular, 4:2:2 and 4:4:4 are formats, not invented ``-422``/``-444``
# switches.  RGB spellings are included only where AppEncCuda advertises them.
NATIVE_FORMATS = {
    "NV12": "nv12",
    "YUV420": "iyuv",
    "NV16": "nv16",
    "YUV444": "yuv444",
    "P010": "p010",
    "P210": "p210",
    "YUV444_16BIT": "yuv444p16",
    "ARGB": "bgra",
    "ABGR": "abgr",
}
NATIVE_TUNINGS = {
    "high_quality": "hq",
    "low_latency": "lowlatency",
    "ultra_low_latency": "ultralowlatency",
    "lossless": "lossless",
    "ultra_high_quality": "uhq",
}
PY_CONFIG_KEYS = (
    "codec",
    "fps",
    "preset",
    "tuning_info",
    "rc",
    "gop",
    "bf",
    "multipass",
    "lookahead",
    "aq",
    "temporalaq",
    "cq",
    "constqp",
    "vbvbufsize",
    "bitrate",
    "maxbitrate",
)
NATIVE_OPTIONS = (
    ("preset", "-preset"),
    ("tuning_info", "-tuninginfo"),
    ("rc", "-rc"),
    ("profile", "-profile"),
    ("fps", "-fps"),
    ("gop", "-gop"),
    ("bf", "-bf"),
    ("multipass", "-multipass"),
    ("bitrate", "-bitrate"),
    ("maxbitrate", "-maxbitrate"),
    ("vbvbufsize", "-vbvbufsize"),
    ("lookahead", "-lookahead"),
    ("aq", "-aq"),
    ("constqp", "-constqp"),
    ("cq", "-cq"),
)


class RecipeError(ValueError):
    """An input cannot be represented by the exact recipe contract."""


def _validate_explicit_profile_format(codec: str, profile: str, raw_format: str) -> None:
    """Reject named profiles that cannot preserve the known YUV format."""
    if raw_format not in KNOWN_YUV_FORMATS:
        return
    allowed = {
        ("h264", "baseline"): {"NV12", "YUV420"},
        ("h264", "main"): {"NV12", "YUV420"},
        ("h264", "high"): {"NV12", "YUV420"},
        ("h264", "high444"): {"NV12", "YUV420", "YUV444"},
        ("hevc", "main"): {"NV12", "YUV420"},
        ("hevc", "main10"): {"NV12", "YUV420", "P010"},
        ("hevc", "frext"): {"NV16", "P210", "YUV444", "YUV444_16BIT"},
        ("av1", "main"): {"NV12", "YUV420", "P010"},
    }.get((codec, profile))
    if allowed is not None and raw_format not in allowed:
        raise RecipeError(
            f"profile {profile} cannot preserve format {raw_format} for {codec}; "
            f"allowed known YUV formats are {sorted(allowed)}"
        )


def _reject_constant(token: str) -> None:
    raise RecipeError(f"non-finite JSON constant is prohibited: {token}")


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecipeError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _strict_json_text(payload: str) -> Any:
    return json.loads(
        payload,
        object_pairs_hook=_strict_object_pairs,
        parse_constant=_reject_constant,
    )


def strict_json_load(path: Path) -> Any:
    """Read strict JSON, rejecting duplicate keys and non-finite numbers."""
    try:
        return _strict_json_text(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecipeError(f"cannot read strict JSON from {path}: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    """Hash one strict JSON value using its canonical representation."""
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RecipeError(f"value is not strict JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise RecipeError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def _enum(value: Any, name: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise RecipeError(f"{name} must be one of {sorted(choices)}")
    return value


def _enabled(value: Any) -> bool:
    return value is True or (isinstance(value, int) and not isinstance(value, bool) and value == 1)


def _validate_constqp(value: Any, codec: str) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        values = [value]
    elif isinstance(value, str):
        parts = value.split(",")
        if len(parts) not in {1, 3} or any(not part.isdigit() for part in parts):
            raise RecipeError("constqp must be one integer or three comma-separated integers")
        values = [int(part) for part in parts]
    else:
        raise RecipeError("constqp must be one integer or three comma-separated integers")
    maximum = 255 if codec == "av1" else 51
    if any(not 0 <= item <= maximum for item in values):
        raise RecipeError(f"constqp values must be in 0..{maximum} for {codec}")
    return ",".join(str(item) for item in values)


def _load_catalog() -> tuple[dict[str, Any], str]:  # pylint: disable=too-many-branches
    catalog = strict_json_load(CATALOG_PATH)
    if not isinstance(catalog, dict) or set(catalog) != {
        "schema_version",
        "kind",
        "version",
        "recipes",
    }:
        raise RecipeError("encoder intent catalog has an invalid exact shape")
    if catalog["schema_version"] != "1.0" or catalog["kind"] != "nvcodec-encoder-intent-catalog":
        raise RecipeError("encoder intent catalog identity is invalid")
    if not isinstance(catalog["version"], str) or not catalog["version"]:
        raise RecipeError("encoder intent catalog version is invalid")
    recipes = catalog["recipes"]
    if not isinstance(recipes, dict) or not recipes:
        raise RecipeError("encoder intent catalog recipes are invalid")
    for use_case, item in recipes.items():
        if not isinstance(use_case, str) or not isinstance(item, dict):
            raise RecipeError("encoder intent catalog recipe is invalid")
        required = {"encoder_intent_defaults", "projections", "assumptions", "rationale"}
        if set(item) != required:
            raise RecipeError(f"catalog recipe {use_case!r} has an invalid exact shape")
        defaults = item["encoder_intent_defaults"]
        if not isinstance(defaults, dict) or not defaults:
            raise RecipeError(f"catalog recipe {use_case!r} defaults are invalid")
        if set(defaults) - CONTROL_KEYS:
            raise RecipeError(f"catalog recipe {use_case!r} defaults contain core/unknown fields")
        projections = item["projections"]
        if not isinstance(projections, dict) or set(projections) != {"native", "pynvc"}:
            raise RecipeError(f"catalog recipe {use_case!r} projections are invalid")
        native = projections["native"]
        pynvc = projections["pynvc"]
        if native != {
            "sample_id": "native-appenc-cuda",
            "sample": "AppEncCuda",
            "release_family": "13.0",
        }:
            raise RecipeError(f"catalog recipe {use_case!r} native projection is unapproved")
        if (
            not isinstance(pynvc, dict)
            or pynvc.get("sample_id") != "pynvc-basic-encode"
            or pynvc.get("sample") != "samples/basic/encode.py"
            or pynvc.get("version") != "2.1.0"
        ):
            raise RecipeError(f"catalog recipe {use_case!r} Py projection is unapproved")
        if not isinstance(item["assumptions"], list) or not all(
            isinstance(value, str) and value for value in item["assumptions"]
        ):
            raise RecipeError(f"catalog recipe {use_case!r} assumptions are invalid")
        if not isinstance(item["rationale"], dict) or not all(
            isinstance(key, str) and isinstance(value, str) and value
            for key, value in item["rationale"].items()
        ):
            raise RecipeError(f"catalog recipe {use_case!r} rationale is invalid")
    return catalog, hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()


def _normalize(raw: Any, catalog: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(raw, dict):
        raise RecipeError("intent must be a JSON object")
    canonical_sha256(raw)
    unknown = sorted(set(raw) - ALLOWED_KEYS)
    if unknown:
        raise RecipeError(f"intent contains unsupported fields: {unknown}")
    missing = sorted({"use_case", "width", "height"} - set(raw))
    if missing:
        raise RecipeError(f"intent is missing required fields: {missing}")
    use_case = raw["use_case"]
    if not isinstance(use_case, str) or use_case not in catalog["recipes"]:
        raise RecipeError(f"use_case must be one of {sorted(catalog['recipes'])}")
    normalized = dict(raw)
    defaults: list[dict[str, Any]] = []
    for key, value, rule in (
        ("codec", "h264", "codec defaults to h264"),
        ("format", "NV12", "input format defaults to NV12"),
        ("fps", 30, "fps defaults to 30"),
        ("gpu", 0, "GPU ordinal defaults to 0"),
        ("output_requirement", "elementary_stream", "output defaults to an elementary stream"),
    ):
        if key not in normalized:
            normalized[key] = value
            defaults.append(
                {
                    "path": f"/encoder_intent/{key}",
                    "value": value,
                    "source": "resolver",
                    "rule": rule,
                }
            )
    normalized["codec"] = _enum(str(normalized["codec"]).lower(), "codec", CODECS)
    normalized["format"] = _enum(str(normalized["format"]).upper(), "format", FORMATS)
    normalized["width"] = _integer(normalized["width"], "width", 1, INT_MAX)
    normalized["height"] = _integer(normalized["height"], "height", 1, INT_MAX)
    normalized["fps"] = _integer(normalized["fps"], "fps", 1, INT_MAX)
    normalized["gpu"] = _integer(normalized["gpu"], "gpu", 0, INT_MAX)
    if normalized["format"] in {"NV12", "YUV420", "P010"} and (
        normalized["width"] % 2 or normalized["height"] % 2
    ):
        raise RecipeError(f"{normalized['format']} requires even width and height")
    if normalized["format"] in YUV422_FORMATS and normalized["width"] % 2:
        raise RecipeError(f"{normalized['format']} requires even width")
    if normalized["codec"] == "av1" and normalized["format"] in (
        YUV422_FORMATS | YUV444_FORMATS
    ):
        raise RecipeError(f"AV1 does not support {normalized['format']} encoder input")
    if normalized["output_requirement"] != "elementary_stream":
        raise RecipeError("output_requirement must be elementary_stream")
    frame_values = [
        _integer(normalized[key], key, 1, INT_MAX)
        for key in ("frame_count", "frames")
        if key in normalized
    ]
    if frame_values:
        frame_count = frame_values[0]
        if any(value != frame_count for value in frame_values[1:]):
            raise RecipeError("frame_count and frames must match")
        normalized["frame_count"] = frame_count
    normalized.pop("frames", None)
    return normalized, defaults


def _controls(  # pylint: disable=too-many-branches,too-many-statements
    raw: dict[str, Any],
    normalized: dict[str, Any],
    catalog_item: dict[str, Any],
    defaults: list[dict[str, Any]],
) -> dict[str, Any]:
    controls = dict(catalog_item["encoder_intent_defaults"])
    sources = {
        key: {
            "source": "catalog",
            "catalog_pointer": (f"/recipes/{normalized['use_case']}/encoder_intent_defaults/{key}"),
        }
        for key in controls
    }
    for key in CONTROL_KEYS:
        if key in normalized:
            controls[key] = normalized[key]
            sources.pop(key, None)
    controls["rc"] = _enum(controls.get("rc"), "rc", RATE_CONTROLS)
    rc = controls["rc"]
    if "cq" in raw:
        if "rc" in raw and rc != "vbr":
            raise RecipeError("cq requires vbr rate control")
        conflicting = sorted(set(raw) & {"bitrate", "maxbitrate"})
        if conflicting:
            raise RecipeError(f"cq conflicts with {conflicting}")
    if rc == "constqp":
        for key in ("bitrate", "maxbitrate", "vbvbufsize", "cq"):
            if key not in raw:
                controls.pop(key, None)
                sources.pop(key, None)
    else:
        if "constqp" not in raw:
            controls.pop("constqp", None)
            sources.pop("constqp", None)
        if "cq" in raw:
            if "rc" not in raw:
                controls["rc"] = "vbr"
                rc = "vbr"
                sources["rc"] = {
                    "source": "derived",
                    "derived_from": ["/intent/cq"],
                    "rule": "cq selects VBR when rc is omitted",
                }
            for key in ("bitrate", "maxbitrate"):
                if key not in raw:
                    controls.pop(key, None)
                    sources.pop(key, None)
    if "bitrate" in raw and rc != "constqp" and "cq" not in controls:
        # Validate the caller value before using it to derive VBV defaults.  A
        # quoted JSON number is an ordinary authoring error and must follow the
        # recipe error contract instead of escaping as a Python ``TypeError``.
        bitrate = _integer(controls["bitrate"], "bitrate", 1, UINT32_MAX)
        controls["bitrate"] = bitrate
        if "maxbitrate" not in raw:
            controls["maxbitrate"] = bitrate
            sources["maxbitrate"] = {
                "source": "derived",
                "derived_from": ["/intent/bitrate"],
                "rule": "maxbitrate follows caller bitrate when omitted",
            }
        if "vbvbufsize" not in raw:
            seconds = 0.1 if normalized["use_case"] == "conferencing" else 0.5
            if normalized["use_case"] not in {"conferencing", "live_streaming"}:
                seconds = 1.0
            controls["vbvbufsize"] = max(1, int(bitrate * seconds))
            sources["vbvbufsize"] = {
                "source": "derived",
                "derived_from": ["/intent/bitrate", "/intent/use_case"],
                "rule": f"VBV defaults to {seconds:g} seconds of caller bitrate",
            }
    controls["preset"] = _enum(controls.get("preset"), "preset", PRESETS)
    controls["tuning_info"] = _enum(controls.get("tuning_info"), "tuning_info", TUNINGS)
    controls["multipass"] = _enum(controls.get("multipass"), "multipass", MULTIPASS)
    for key, minimum, maximum in (
        ("gop", 1, INT_MAX),
        ("bf", 0, INT_MAX),
        ("lookahead", 0, 31),
        ("aq", 1, 15),
        ("cq", 0, 63 if normalized["codec"] == "av1" else 51),
        ("bitrate", 1, UINT32_MAX),
        ("maxbitrate", 1, UINT32_MAX),
        ("vbvbufsize", 1, UINT32_MAX),
    ):
        if key in controls:
            controls[key] = _integer(controls[key], key, minimum, maximum)
    if "temporalaq" in controls and not _enabled(controls["temporalaq"]):
        raise RecipeError("temporalaq must be true or integer 1")
    if "constqp" in controls:
        controls["constqp"] = _validate_constqp(controls["constqp"], normalized["codec"])
    if "profile" in controls:
        controls["profile"] = _enum(controls["profile"], "profile", PROFILES)
        if controls["profile"] not in CODEC_PROFILES[normalized["codec"]]:
            raise RecipeError(
                f"profile {controls['profile']} is invalid for codec {normalized['codec']}"
            )
        _validate_explicit_profile_format(
            normalized["codec"], controls["profile"], normalized["format"]
        )
    if (
        normalized["codec"] == "h264"
        and controls.get("profile") == "baseline"
        and controls["bf"] > 0
    ):
        raise RecipeError(
            f"H.264 Baseline profile requires bf=0; resolved bf={controls['bf']}"
        )
    if controls["gop"] < controls["bf"] + 1:
        raise RecipeError("gop must be at least bf + 1")
    if controls["lookahead"] > 31 - controls["bf"]:
        raise RecipeError("lookahead must be at most 31 - bf")
    if controls["tuning_info"] == "ultra_low_latency" and (controls["bf"] or controls["lookahead"]):
        raise RecipeError("ultra_low_latency requires bf=0 and lookahead=0")
    rc = controls["rc"]
    if rc == "constqp":
        if "constqp" not in controls:
            raise RecipeError("constqp rate control requires constqp")
        conflicting = sorted(set(controls) & {"bitrate", "maxbitrate", "vbvbufsize", "cq"})
        if conflicting:
            raise RecipeError(f"constqp conflicts with {conflicting}")
    elif "constqp" in controls:
        raise RecipeError("constqp values require constqp rate control")
    if rc in {"cbr", "vbr"} and "cq" not in controls and "bitrate" not in controls:
        raise RecipeError(f"{rc} requires bitrate")
    if rc == "cbr" and controls.get("maxbitrate", controls.get("bitrate")) != controls.get(
        "bitrate"
    ):
        raise RecipeError("CBR maxbitrate must equal bitrate")
    if (
        rc == "vbr"
        and "bitrate" in controls
        and controls.get("maxbitrate", controls["bitrate"]) < controls["bitrate"]
    ):
        raise RecipeError("VBR maxbitrate must be at least bitrate")
    if normalized["codec"] == "h264" and controls["tuning_info"] == "lossless":
        if rc != "constqp":
            raise RecipeError("H.264 lossless requires constqp rate control")
        if controls.get("constqp") not in {"0", "0,0,0"}:
            raise RecipeError("H.264 lossless requires every constqp component to be zero")
        if controls.get("profile") not in {None, "high444"}:
            raise RecipeError("H.264 lossless requires an omitted or high444 profile")
    for key, source in sources.items():
        if key in controls:
            defaults.append({"path": f"/encoder_intent/{key}", "value": controls[key], **source})
    return controls


def _loss(path: str, value: Any, sample: str, reason: str) -> dict[str, Any]:
    return {"path": path, "value": value, "sample": sample, "reason": reason}


def _projections(
    intent: dict[str, Any], sample_catalog: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    native_losses: list[dict[str, Any]] = []
    pynvc_losses: list[dict[str, Any]] = []
    native_options = [
        "-s",
        f"{intent['width']}x{intent['height']}",
        "-if",
        NATIVE_FORMATS[intent["format"]],
        "-gpu",
        str(intent["gpu"]),
        "-codec",
        intent["codec"],
    ]
    for key, option in NATIVE_OPTIONS:
        if key not in intent:
            continue
        value = intent[key]
        if key == "tuning_info":
            value = NATIVE_TUNINGS[value]
        native_options.extend([option, str(value).lower()])
    if _enabled(intent.get("temporalaq")):
        native_options.append("-temporalaq")
    pynvc_config = {key: intent[key] for key in PY_CONFIG_KEYS if key in intent}
    pynvc_config["gpu_id"] = intent["gpu"]
    if "preset" in pynvc_config:
        pynvc_config["preset"] = pynvc_config["preset"].upper()
    if pynvc_config.get("tuning_info") == "ultra_high_quality":
        if intent["codec"] in {"hevc", "av1"}:
            pynvc_config["tuning_info"] = "uhq"
        else:
            pynvc_config.pop("tuning_info")
            pynvc_losses.append(
                _loss(
                    "/encoder_intent/tuning_info",
                    intent["tuning_info"],
                    "pynvc-basic-encode",
                    "public PyNvVideoCodec 2.1 UHQ is restricted to HEVC and AV1",
                )
            )
    if "profile" in intent:
        pynvc_losses.append(
            _loss(
                "/encoder_intent/profile",
                intent["profile"],
                "pynvc-basic-encode",
                "the released PyNvVideoCodec 2.1 encode parser has no profile option",
            )
        )
    projections = {
        "native": {
            "status": "exact" if not native_losses else "unrepresentable",
            "sample": dict(sample_catalog["native"]),
            "cli_options": native_options,
            "frame_limit": {
                "mode": "exact_input" if "frame_count" in intent else "all_input",
                "frames": intent.get("frame_count"),
            },
        },
        "pynvc": {
            "status": "exact" if not pynvc_losses else "unrepresentable",
            "sample": {
                key: sample_catalog["pynvc"][key] for key in ("sample_id", "sample", "version")
            },
            "official_sample_baseline": sample_catalog["pynvc"]["official_sample_baseline"],
            "config": pynvc_config,
            "arguments": {
                "size": f"{intent['width']}x{intent['height']}",
                "format": intent["format"],
                "gpu": intent["gpu"],
                "codec": intent["codec"],
                "frame_count": intent.get("frame_count"),
            },
        },
    }
    return projections, {"native": native_losses, "pynvc": pynvc_losses}


def _rationale(catalog_rationale: dict[str, str], intent: dict[str, Any]) -> dict[str, str]:
    """Render request-bound rationale instead of retaining stale catalog prose."""
    result = {
        key: value
        for key, value in catalog_rationale.items()
        if key in intent
    }
    duration = intent["gop"] / intent["fps"]
    result["gop"] = (
        f"{intent['gop']} frames at {intent['fps']} fps spans approximately "
        f"{duration:.3f} seconds; validate random-access requirements"
    )
    result["bf"] = (
        "disables B-frame reordering"
        if intent["bf"] == 0
        else f"allows {intent['bf']} B frames; validate reorder latency"
    )
    result["lookahead"] = (
        "disables lookahead queueing"
        if intent["lookahead"] == 0
        else f"uses a {intent['lookahead']}-frame lookahead; validate latency and memory"
    )
    result["preset"] = (
        f"uses the resolved {intent['preset']} candidate; validate operation and"
        " measure throughput"
    )
    result["tuning_info"] = (
        f"uses the resolved {intent['tuning_info']} tuning path; validate it on the "
        "selected surface"
    )
    result["rc"] = (
        f"uses resolved {intent['rc']} rate control; validate its exact bitrate or "
        "constant-quality behavior"
    )
    result["multipass"] = (
        f"uses resolved multipass={intent['multipass']}; measure its throughput impact"
    )
    if "aq" in intent:
        result["aq"] = f"uses spatial AQ strength {intent['aq']}"
    if "vbvbufsize" in intent:
        result["vbvbufsize"] = (
            f"uses resolved VBV size {intent['vbvbufsize']} bits; validate buffering "
            "against the latency budget"
        )
    return result


def _assumptions(catalog_assumptions: list[str], intent: dict[str, Any]) -> list[str]:
    """Keep catalog context only when it still matches resolved controls."""
    result = []
    for value in catalog_assumptions:
        lowered = value.lower()
        if "b frame" in lowered or "b-frame" in lowered:
            continue
        if "lookahead consumes" in lowered and intent["lookahead"] == 0:
            continue
        if "strict cbr" in lowered and intent["rc"] != "cbr":
            continue
        result.append(value)
    if intent["bf"] > 0:
        result.append(f"{intent['bf']} B frames are acceptable only after latency validation")
    return result


def resolve_recipe(raw: Any) -> dict[str, Any]:
    """Build the one supported deterministic schema-2 recipe."""
    catalog, catalog_sha = _load_catalog()
    normalized, defaults = _normalize(raw, catalog)
    item = catalog["recipes"][normalized["use_case"]]
    controls = _controls(raw, normalized, item, defaults)
    intent = {
        key: normalized[key]
        for key in (
            "codec",
            "gpu",
            "width",
            "height",
            "format",
            "fps",
            "output_requirement",
            "frame_count",
        )
        if key in normalized
    }
    intent.update(controls)
    projections, losses = _projections(intent, item["projections"])
    provided = sorted(raw)
    body = {
        "intent": normalized,
        "encoder_intent": intent,
        "projections": projections,
        "projection_losses": losses,
        "catalog": {
            "path": CATALOG_REFERENCE,
            "sha256": catalog_sha,
            "version": catalog["version"],
            "use_case": normalized["use_case"],
        },
        "defaults": {
            "provided_intent_keys": provided,
            "entries": sorted(defaults, key=lambda item: item["path"]),
        },
        "assumptions": _assumptions(item["assumptions"], intent),
        "rationale": _rationale(item["rationale"], intent),
    }
    digests = {
        "intent_sha256": canonical_sha256(intent),
        "native_projection_sha256": canonical_sha256(
            {"projection": projections["native"], "losses": losses["native"]}
        ),
        "pynvc_projection_sha256": canonical_sha256(
            {"projection": projections["pynvc"], "losses": losses["pynvc"]}
        ),
        "contract_sha256": canonical_sha256(body),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": (
            "candidate"
            if any(item["status"] == "exact" for item in projections.values())
            else "blocked"
        ),
        **body,
        "digests": digests,
    }


def validate_recipe(recipe: Any) -> None:
    """Replay deterministic derivation, raising on legacy, hybrid, or drift."""
    if not isinstance(recipe, dict):
        raise RecipeError("recipe must be a JSON object")
    if recipe.get("schema_version") != SCHEMA_VERSION or recipe.get("kind") != KIND:
        raise RecipeError(f"recipe identity must be schema {SCHEMA_VERSION} kind {KIND}")
    if set(recipe) & {"surface", "pynvc_config", "official_sample_baseline"}:
        raise RecipeError(
            "legacy recipe fields are prohibited; migration adapters are not supported"
        )
    try:
        defaults = recipe.get("defaults", {})
        if not isinstance(defaults, dict):
            raise RecipeError("recipe defaults must be a JSON object")
        provided = defaults.get("provided_intent_keys")
        resolved = recipe.get("intent")
        if (
            not isinstance(provided, list)
            or not all(isinstance(key, str) for key in provided)
            or provided != sorted(provided)
            or len(provided) != len(set(provided))
            or not isinstance(resolved, dict)
        ):
            raise RecipeError("recipe defaults/provided keys are invalid")
        raw = {}
        for key in provided:
            source = "frame_count" if key == "frames" else key
            if source not in resolved:
                raise RecipeError(f"provided key {key!r} has no resolved value")
            raw[key] = resolved[source]
        expected = resolve_recipe(raw)
    except (OSError, RecipeError) as exc:
        raise RecipeError(str(exc)) from exc
    if recipe != expected:
        raise RecipeError("recipe differs from deterministic intent/catalog replay")


def _unknown(
    reasons: list[str], *, authenticated_dry_run_deferred: bool = True
) -> dict[str, Any]:
    return {
        "classification": "unknown",
        "authenticated_dry_run_deferred": authenticated_dry_run_deferred,
        "reasons": reasons,
    }


def _setup_skill_path() -> Path:
    """Return the setup marker in the lexical installed-skill catalog."""
    return Path(__file__).absolute().parents[3] / "jetson-video-setup" / "SKILL.md"


def _setup_dependency(*, surface: str) -> dict[str, Any]:
    """Describe whether the canonical setup sibling can resolve live readiness."""
    if surface not in {"native", "pynvc"}:
        raise RecipeError("setup dependency surface must be native or pynvc")
    candidate = _setup_skill_path()
    try:
        details = candidate.lstat()
        resolved_root = candidate.parent.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError:
        installed = False
    else:
        installed = bool(
            stat.S_ISREG(details.st_mode)
            and not stat.S_ISLNK(details.st_mode)
            and resolved_root.is_dir()
            and resolved == resolved_root / "SKILL.md"
        )
    action = (
        f"use jetson-video-setup to configure or repair the {surface} surface, "
        "then retry this live compatibility check"
        if installed
        else f"install jetson-video-setup as a sibling skill, use it to configure "
        f"or repair the {surface} surface, then retry this live compatibility check"
    )
    return {
        "skill": "jetson-video-setup",
        "installed": installed,
        "needed_for": f"{surface} live-compatibility prerequisites",
        "next_action": action,
    }


def _setup_remediation(
    result: dict[str, Any], *, surface: str, reason: str
) -> dict[str, Any]:
    result["remediation"] = {
        "route": "jetson-video-setup",
        "surface": surface,
        "mutation_performed": False,
        "reason": reason,
        "dependency": _setup_dependency(surface=surface),
    }
    return result


def _dependency_record_reasons(
    record: Any, label: str, exact: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Classify one schema-1.2 package entry into negatives and missing evidence."""
    if not isinstance(record, dict):
        return [], [f"{label} installation evidence is absent or malformed"]
    unsupported: list[str] = []
    unknown: list[str] = []
    status = record.get("status")
    if status is None:
        unknown.append(f"{label} installation status is unavailable")
    elif isinstance(status, str) and status in {"missing", "not_installed", "absent"}:
        unsupported.append(f"{label} is explicitly not installed")
    elif status != "installed":
        unknown.append(f"{label} installation status is {status!r}")
    satisfied = record.get("requirement_satisfied")
    if satisfied is False:
        unsupported.append(f"{label} requirement is explicitly unsatisfied")
    elif satisfied is not True:
        unknown.append(f"{label} requirement evidence is unavailable")
    for key, expected in exact.items():
        value = record.get(key)
        if value is None:
            unknown.append(f"{label} {key} evidence is unavailable")
        elif value != expected:
            unsupported.append(f"{label} {key}={value!r} does not match required {expected!r}")
    return unsupported, unknown


def _dependency_reasons(
    environment: dict[str, Any], buffer_mode: str
) -> tuple[list[str], list[str]]:
    """Separate explicit dependency negatives from missing or malformed evidence.

    Schema 1.2 carries Python readiness as ``installation.python.packages`` —
    ``{status, version, requirement_satisfied}`` per entry, plus ``cuda_build``,
    ``cuda_available``, and ``sample_readiness`` for torch.
    """
    node: Any = environment
    for key, absent in (
        ("installation", "Python installation evidence is absent or malformed"),
        ("python", "Python environment evidence is absent or malformed"),
        ("packages", "Python package evidence is absent or malformed"),
    ):
        node = node.get(key) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            return [], [absent]
    required = list(PYNVC_DEPENDENCIES)
    if buffer_mode == "gpu":
        required.append(TORCH_GPU_DEPENDENCY)
    unsupported: list[str] = []
    unknown: list[str] = []
    for name, label, exact in required:
        negatives, missing = _dependency_record_reasons(node.get(name), label, exact)
        unsupported.extend(negatives)
        unknown.extend(missing)
    return unsupported, unknown


def _capability_report(report: Any) -> dict[str, Any] | None:
    """Accept the capability-owned encoder report, gated by its artifact identity.

    Artifact identity is pinned in ``CAPABILITY_REPORT_IDENTITY`` and read here
    and only here, so adopting a future identity stays a one-line change.
    """
    if not isinstance(report, dict):
        return None
    if any(report.get(key) != value for key, value in CAPABILITY_REPORT_IDENTITY.items()):
        return None
    return report


def _encoder_authority(
    environment: dict[str, Any], capability_report: Any
) -> tuple[dict[str, Any] | None, str]:
    """Select the encoder capability authority and its producer's operation spelling.

    ``--capability-report`` is optional and additive. When one is supplied it is
    the authority for every encoder fact and must additionally prove it was
    produced from this exact environment artifact. When none is supplied the
    established schema-1.2 ``capabilities`` block of the environment is the
    authority; that is the baseline path, so an absent report is never a failure.
    Either way the evidence is still an API query and is classified as such.
    """
    if capability_report is None:
        caps = environment.get("capabilities")
        return (caps if isinstance(caps, dict) else None), ENVIRONMENT_OPERATION_STATUS
    return _capability_report(capability_report), REPORT_OPERATION_STATUS


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _absolute(value: Any) -> bool:
    return _text(value) and Path(str(value)).is_absolute()


def _environment_verdict(  # pylint: disable=too-many-locals,too-many-return-statements
    environment: Any, *, gpu: int, surface: str
) -> tuple[str, str] | None:
    """Validate the envelope and classify only explicit selected-surface negatives."""
    if (
        not isinstance(environment, dict)
        or environment.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION
        or environment.get("kind") != ENVIRONMENT_KIND
        or environment.get("mode") != "live"
    ):
        return "unknown", (
            "environment is not an exact live schema-"
            f"{ENVIRONMENT_SCHEMA_VERSION} {ENVIRONMENT_KIND}"
        )
    selected = environment.get("selected_gpu")
    if isinstance(selected, bool) or selected != gpu:
        return "unknown", "recipe GPU does not match environment selected_gpu"
    requested_runtime = environment.get("requested_runtime")
    if requested_runtime not in {surface, "both"}:
        return "unknown", (
            f"environment requested_runtime does not include the selected {surface} surface"
        )
    if surface == "native":
        installation = environment.get("installation")
        native = installation.get("native_sdk") if isinstance(installation, dict) else None
        package = native.get("package") if isinstance(native, dict) else None
        installed = isinstance(native, dict) and (
            native.get("installed") is True or native.get("status") == "installed"
        )
        if (
            not installed
            or not isinstance(package, dict)
            or package.get("name") != "nvidia-video-codec-sdk"
            or package.get("status") != "installed"
        ):
            reason = "environment does not report an installed native Video Codec SDK surface"
            explicit = (
                (
                    isinstance(native, dict)
                    and (
                        native.get("installed") is False
                        or native.get("status") in {"missing", "partial", "not_ready"}
                    )
                )
                or (
                    isinstance(package, dict)
                    and package.get("status") == "missing"
                )
            )
        else:
            readiness = environment.get("readiness")
            layers = readiness.get("layers") if isinstance(readiness, dict) else None
            native_readiness = layers.get("native_sdk") if isinstance(layers, dict) else None
            if (
                not isinstance(native_readiness, dict)
                or native_readiness.get("installation") != "ready"
            ):
                reason = (
                    "environment does not report native Video Codec SDK "
                    "installation readiness"
                )
                explicit = isinstance(native_readiness, dict) and (
                    native_readiness.get("installation") in {"missing", "not_ready"}
                )
            else:
                return None
        return ("unsupported" if explicit else "unknown"), reason
    pynvc = environment.get("pynvc")
    if not isinstance(pynvc, dict) or pynvc.get("imported") is not True:
        classification = (
            "unsupported"
            if isinstance(pynvc, dict) and pynvc.get("imported") is False
            else "unknown"
        )
        return classification, "environment does not report an imported PyNvVideoCodec surface"
    identity = pynvc.get("identity")
    if not isinstance(identity, dict) or identity.get("status") != "verified":
        classification = (
            "unsupported"
            if isinstance(identity, dict)
            and identity.get("status") in {"missing", "not_ready", "unverified"}
            else "unknown"
        )
        return classification, "environment does not report a verified PyNvVideoCodec identity"
    return None


def _declared_pynvc_identity(environment: dict[str, Any]) -> dict[str, Any]:
    """Read the schema-1.2 ``pynvc.identity`` block, or an empty required subset."""
    surface = environment.get("pynvc")
    identity = surface.get("identity") if isinstance(surface, dict) else None
    return identity if isinstance(identity, dict) else {}


def _declared_distribution_version(declared: dict[str, Any]) -> Any:
    """Read the manifest's PyNvVideoCodec distribution version by either spelling.

    The established 1.2 producer carries it as ``identity.distribution``; the
    flattened ``identity.version`` is an additive refinement. Both mean the
    installed distribution version, so either satisfies the comparison.
    """
    if "version" in declared:
        return declared.get("version")
    distribution = declared.get("distribution")
    if isinstance(distribution, dict):
        return distribution.get("version")
    return distribution


def _identity_subset_matches(recorded: Any, observed: Any) -> bool:
    """Compare one authenticated identity by required fields, permitting additions."""
    if not isinstance(recorded, dict) or not isinstance(observed, dict):
        return False
    return all(recorded.get(key) == observed.get(key) for key in ("path", "size_bytes", "sha256"))


def _pynvc_identity_matches(binding: dict[str, Any], declared: Any) -> bool:
    """Bind the report's authenticated live import to the environment Py identity.

    ``declared.version`` and ``declared.interpreter_identity`` are additive
    refinements, not requirements: the established producer spells the
    distribution version ``distribution`` and publishes no resolved interpreter
    identity. Either version spelling satisfies the comparison, and a resolved
    interpreter identity is compared only when the manifest carries one — but
    when carried it is held exactly, so a refinement can only tighten.
    """
    identity = binding.get("pynvc_identity")
    if not isinstance(identity, dict) or not isinstance(declared, dict):
        return False
    expected = (
        (identity.get("status"), "verified"),
        (identity.get("authentication"), "local_wheel_record_ownership"),
        (identity.get("interpreter"), declared.get("interpreter")),
        (identity.get("sys_prefix"), declared.get("sys_prefix")),
        (identity.get("version"), _declared_distribution_version(declared)),
    )
    observed_interpreter = declared.get("interpreter_identity")
    if any(recorded != observed for recorded, observed in expected) or (
        observed_interpreter is not None
        and (
            not _identity_subset_matches(
                identity.get("interpreter_identity"), observed_interpreter
            )
            or not _identity_subset_matches(
                binding.get("interpreter_identity"), observed_interpreter
            )
        )
    ):
        return False
    module, observed_module = identity.get("module"), declared.get("module")
    extension, observed_extension = identity.get("extension"), declared.get("extension")
    if not isinstance(module, dict) or not isinstance(observed_module, dict):
        return False
    if (
        module.get("path") != observed_module.get("path")
        or module.get("version") != observed_module.get("version")
    ):
        return False
    if not isinstance(extension, dict) or not isinstance(observed_extension, dict):
        return False
    return (
        extension.get("path") == observed_extension.get("path")
        and extension.get("sha256") == observed_extension.get("sha256")
        and _absolute(extension.get("loaded_path"))
        and _absolute(observed_extension.get("loaded_path"))
        and os.path.realpath(extension["loaded_path"])
        == os.path.realpath(observed_extension["loaded_path"])
    )


def _binding_reason(
    report: dict[str, Any], environment: dict[str, Any], environment_path: Path | None
) -> str | None:
    """Reject a report that was not produced from this exact environment artifact.

    The raw and canonical digests have different domains: ``sha256`` covers the
    exact file bytes, while ``canonical_sha256`` covers the parsed JSON value.
    The remaining fields bind schema/GPU and the authenticated live Py import.
    """
    if environment_path is None:
        raise RecipeError(
            "capability report binding requires the exact environment artifact path"
        )
    binding = report.get("environment")
    if not isinstance(binding, dict) or any(
        key not in binding for key in CAPABILITY_BINDING_FIELDS
    ):
        return "capability report does not record the environment it was produced from"
    try:
        raw_environment = environment_path.read_bytes()
    except OSError as exc:
        raise RecipeError(f"cannot read the environment artifact bytes: {exc}") from exc
    try:
        parsed_environment = _strict_json_text(raw_environment.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, RecipeError):
        return "capability report is bound to a malformed environment artifact"
    if parsed_environment != environment:
        return "capability report is bound to a different environment artifact"
    declared = _declared_pynvc_identity(environment)
    expected: list[tuple[Any, Any]] = [
        (binding["sha256"], hashlib.sha256(raw_environment).hexdigest()),
        (binding["canonical_sha256"], canonical_sha256(parsed_environment)),
        (binding["size_bytes"], len(raw_environment)),
        (binding["path"], os.path.realpath(environment_path)),
        (binding["kind"], environment.get("kind")),
        (binding["schema_version"], environment.get("schema_version")),
        (binding["mode"], environment.get("mode")),
        (binding["selected_gpu"], environment.get("selected_gpu")),
        (binding["interpreter"], declared.get("interpreter")),
    ]
    if any(recorded != observed for recorded, observed in expected) or not (
        _pynvc_identity_matches(binding, declared)
    ):
        return "capability report is bound to a different environment artifact"
    return None


def _documentation_verdict(record: dict[str, Any]) -> str | None:
    """Read the published documentation cross-check verdict for one codec."""
    crosscheck = record.get("documentation_crosscheck")
    if not isinstance(crosscheck, dict):
        return None
    support = crosscheck.get("support")
    return support.lower() if isinstance(support, str) else None


def _rc_supported(mask: Any, rc: str) -> bool | None:
    if isinstance(mask, bool) or not isinstance(mask, int) or mask <= 0:
        return None
    if rc == "constqp":
        return True
    return bool(mask & (0x1 if rc == "vbr" else 0x2))


def _api_version(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None
    fields = value.split(".")
    if len(fields) != 2 or any(not field.isdigit() for field in fields):
        return None
    return int(fields[0]), int(fields[1])


def _selected_gpu_name(environment: dict[str, Any], gpu: int) -> str | None:
    """Name the environment's selected GPU only when it is the exact requested one."""
    selected = environment.get("selected_gpu")
    if isinstance(selected, bool) or selected != gpu:
        return None
    inventory = environment.get("nvidia_smi")
    records = inventory.get("gpus") if isinstance(inventory, dict) else None
    if not isinstance(records, list):
        return None
    for record in records:
        if isinstance(record, dict) and record.get("index") == gpu:
            name = record.get("name")
            return name if isinstance(name, str) and name else None
    return None


def _uhq_reason(
    caps: dict[str, Any], environment: dict[str, Any], gpu: int
) -> dict[str, Any] | None:
    """Gate ultra-high-quality tuning on linked-NVENC API and generation evidence."""
    linked_api = caps.get("linked_nvenc_api", {})
    linked_version = _api_version(
        linked_api.get("value") if isinstance(linked_api, dict) else None
    )
    if (
        not isinstance(linked_api, dict)
        or linked_api.get("status") != "observed"
        or linked_version is None
    ):
        return _unknown(["PyNvVideoCodec UHQ linked-NVENC API evidence is unavailable"])
    if linked_version < (12, 2):
        return {
            "classification": "unsupported",
            "authenticated_dry_run_deferred": False,
            "reasons": ["public PyNvVideoCodec 2.1 UHQ requires linked NVENC API 12.2 or newer"],
        }
    gpu_name = _selected_gpu_name(environment, gpu)
    if gpu_name is None or "thor" not in gpu_name.lower():
        return _unknown(
            [
                "PyNvVideoCodec UHQ requires authenticated Turing-or-newer generation "
                "evidence; only Thor is established by this skill"
            ]
        )
    return None


def _bound_reasons(
    values: dict[str, Any], intent: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Compare requested geometry, rate control, and B frames against queried bounds."""
    unsupported: list[str] = []
    unknown: list[str] = []
    for key, requested, relation in (
        ("width_min", intent["width"], "min"),
        ("width_max", intent["width"], "max"),
        ("height_min", intent["height"], "min"),
        ("height_max", intent["height"], "max"),
    ):
        bound = values.get(key)
        if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
            unknown.append(f"{key} is unavailable")
        elif relation == "min" and requested < bound:
            unsupported.append(f"{key}={bound} excludes requested {requested}")
        elif relation == "max" and requested > bound:
            unsupported.append(f"{key}={bound} excludes requested {requested}")
    rc = _rc_supported(values.get("supported_ratecontrol_modes"), intent["rc"])
    if rc is False:
        unsupported.append(f"rate control {intent['rc']} is not reported")
    elif rc is None:
        unknown.append("supported_ratecontrol_modes is unavailable")
    if intent["bf"] > 0:
        maximum = values.get("num_max_bframes")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
            unknown.append("num_max_bframes is unavailable")
        elif intent["bf"] > maximum:
            unsupported.append(f"bf={intent['bf']} exceeds num_max_bframes={maximum}")
    return unsupported, unknown


def _gate_reasons(
    values: dict[str, Any], unavailable: Any, intent: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Require an exact positive capability flag for each requested optional control."""
    unsupported: list[str] = []
    unknown: list[str] = []
    gates = (
        (intent["lookahead"] > 0, "support_lookahead", "lookahead"),
        (_enabled(intent.get("temporalaq")), "support_temporal_aq", "temporal AQ"),
        ("vbvbufsize" in intent, "support_custom_vbv_buf_size", "custom VBV"),
        (intent["format"] in TEN_BIT_FORMATS, "support_10bit_encode", "10-bit"),
        (intent["format"] in YUV422_FORMATS, "support_yuv422_encode", "4:2:2"),
        (intent["format"] in YUV444_FORMATS, "support_yuv444_encode", "4:4:4"),
        (intent["tuning_info"] == "lossless", "support_lossless_encode", "lossless"),
    )
    for required, key, label in gates:
        if not required:
            continue
        value = values.get(key)
        if value == 0:
            unsupported.append(f"{label} requires {key}=1 but query reports 0")
        elif value != 1:
            reason = (
                unavailable.get(key, {}).get("reason") if isinstance(unavailable, dict) else None
            )
            unknown.append(f"{label} capability is unavailable" + (f": {reason}" if reason else ""))
    return unsupported, unknown


def _field_reasons(
    record: dict[str, Any], intent: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Compare requested controls against the exact queried capability fields."""
    values = record["values"]
    bounded, bounded_unknown = _bound_reasons(values, intent)
    gated, gated_unknown = _gate_reasons(values, record.get("unavailable_fields", {}), intent)
    return bounded + gated, bounded_unknown + gated_unknown


def _codec_record_verdict(record: Any, operation_status: str) -> dict[str, Any] | None:
    """Apply the evidence precedence to one codec record before reading its fields.

    Precedence is API query, then official operation, then documentation. An
    applicable documentation "No" is the final verdict for any codec and outranks
    a positive query, but the disagreement is never dropped. A positive query is
    ``capability_reported`` with ``supported: null`` and an ``operation_status``
    that explicitly denies an operation; it is never product support. Returning
    ``None`` means the record survived every gate and its fields may be read.
    """
    if not isinstance(record, dict):
        return _unknown(["exact codec capability record is absent"])
    if record.get("status") == "unsupported" or record.get("supported") is False:
        return {
            "classification": "unsupported",
            "authenticated_dry_run_deferred": False,
            "reasons": ["exact codec query reports unsupported"],
        }
    if _documentation_verdict(record) == "no":
        reasons = [
            "NVIDIA documentation reports this codec unsupported; documentation is the "
            "final product-support authority"
        ]
        if record.get("status") == "capability_reported":
            reasons.append(
                "recorded discrepancy: GetEncoderCaps reports capability_reported while "
                "documentation reports No"
            )
        return {
            "classification": "unsupported",
            "authenticated_dry_run_deferred": False,
            "reasons": reasons,
        }
    if not (
        record.get("status") == "capability_reported"
        and record.get("supported") is None
        and record.get("session_status") == "opened_by_GetEncoderCaps"
        and record.get("operation_status") == operation_status
        and isinstance(record.get("values"), dict)
    ):
        return _unknown(["exact codec GetEncoderCaps record is incomplete or malformed"])
    return None


def validate_live_capabilities(  # pylint: disable=too-many-branches,too-many-locals
    # pylint: disable=too-many-return-statements,too-many-arguments
    recipe: dict[str, Any],
    environment: Any | None,
    capability_report: Any,
    surface: str,
    *,
    environment_path: Path | None,
    buffer_mode: str = "gpu",
) -> dict[str, Any]:
    """Classify one projection from exact live environment and capability facts.

    ``environment`` is the schema-1.2 setup artifact, read for ``kind``,
    ``schema_version``, ``mode``, ``selected_gpu``, ``nvidia_smi``, ``pynvc`` and
    ``installation.python.packages``. Encoder facts come from the environment's
    own ``capabilities`` block, or from ``capability_report`` when one is
    supplied. Codec/API support and installation readiness are different
    questions and keep different sources.

    ``capability_report`` is optional and additive on every surface; pass ``None``
    when it was not supplied. A supplied report is accepted only when it is bound
    to this exact environment artifact, which is why ``environment_path``
    accompanies the parsed environment.
    """
    validate_recipe(recipe)
    if surface not in {"native", "pynvc"}:
        raise RecipeError("surface must be native or pynvc")
    buffer_mode = _enum(buffer_mode, "buffer_mode", {"cpu", "gpu"})
    if surface == "native" and capability_report is not None:
        raise RecipeError(
            "--capability-report is a PyNvVideoCodec-only refinement and is invalid "
            "for the native surface"
        )
    if recipe["projections"][surface]["status"] != "exact":
        return {
            "classification": "unsupported",
            "authenticated_dry_run_deferred": False,
            "reasons": ["recipe has blocking projection losses"],
        }
    intent = recipe["encoder_intent"]
    if environment is None:
        reason = (
            f"live {surface} environment evidence is absent; recipe planning and "
            "validation remain complete but live compatibility is unresolved"
        )
        if capability_report is not None:
            _capability_report(capability_report)
            reason += (
                "; a capability report alone does not authenticate selected-surface "
                "readiness or selected-GPU identity"
            )
        return _setup_remediation(
            _unknown([reason], authenticated_dry_run_deferred=False),
            surface=surface,
            reason="produce fresh selected-surface readiness evidence",
        )
    environment_verdict = _environment_verdict(
        environment, gpu=intent["gpu"], surface=surface
    )
    if environment_verdict is not None:
        classification, environment_reason = environment_verdict
        result = _unknown(
            [environment_reason], authenticated_dry_run_deferred=False
        )
        if classification == "unsupported":
            result["classification"] = classification
            result["reasons"][0] += (
                "; this is explicit live selected-surface SDK readiness, not a "
                "codec/product-support verdict"
            )
        return _setup_remediation(
            result,
            surface=surface,
            reason="refresh or repair selected-surface readiness evidence",
        )
    if surface == "native":
        return _unknown(
            [
                "native AppEncCuda has no equivalent authenticated GetEncoderCaps "
                "authority; require its authenticated dry run"
            ]
        )
    dependency_failures, dependency_unknown = _dependency_reasons(environment, buffer_mode)
    if dependency_failures:
        return _setup_remediation({
            "classification": "unsupported",
            "authenticated_dry_run_deferred": False,
            "reasons": dependency_failures + dependency_unknown,
        }, surface=surface, reason="repair selected-surface prerequisites")
    if dependency_unknown:
        return _setup_remediation(
            _unknown(dependency_unknown),
            surface=surface,
            reason="probe or repair selected-surface prerequisites",
        )
    caps, operation_status = _encoder_authority(environment, capability_report)
    expected_classification = {
        "evidence_source_type": "api_query_helper",
        "official_sample": False,
        "api_symbol": "PyNvVideoCodec.GetEncoderCaps",
    }
    evidence = caps.get("evidence_classification") if caps is not None else None
    if (
        caps is None
        or caps.get("authority") != "pynvc"
        or caps.get("gpu") != intent["gpu"]
        or not isinstance(evidence, dict)
        or evidence.get("encode") != expected_classification
    ):
        return _unknown(
            ["capability authority, exact GPU, or GetEncoderCaps classification is invalid"]
        )
    if capability_report is not None:
        binding_reason = _binding_reason(caps, environment, environment_path)
        if binding_reason is not None:
            return _unknown([binding_reason])
    if intent["tuning_info"] == "ultra_high_quality":
        uhq_reason = _uhq_reason(caps, environment, intent["gpu"])
        if uhq_reason is not None:
            return uhq_reason
    encode = caps.get("encode")
    record = encode.get(intent["codec"]) if isinstance(encode, dict) else None
    verdict = _codec_record_verdict(record, operation_status)
    if verdict is not None:
        return verdict
    unsupported, unknown = _field_reasons(record, intent)
    if unsupported:
        return {
            "classification": "unsupported",
            "authenticated_dry_run_deferred": False,
            "reasons": unsupported + unknown,
        }
    if unknown:
        return _unknown(unknown)
    return {
        "classification": "compatible",
        "authenticated_dry_run_deferred": True,
        "reasons": [
            "GetEncoderCaps reports all requested fields; an authenticated official "
            "encode is still required"
        ],
    }


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    destination = Path(os.path.abspath(os.path.expanduser(str(path))))
    if destination.parent.resolve(strict=True) != destination.parent or os.path.lexists(
        destination
    ):
        raise RecipeError("output parent must be resolved and output must be fresh")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Run the thin recipe CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="resolve intent into the sole schema-2 recipe")
    plan.add_argument("--intent", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="replay and validate one recipe")
    validate.add_argument("--recipe", type=Path, required=True)
    live = subparsers.add_parser(
        "check-live", help="classify one recipe projection from live facts"
    )
    live.add_argument("--recipe", type=Path, required=True)
    live.add_argument(
        "--environment",
        type=Path,
        help=(
            "optional live schema-1.2 setup environment; omission preserves the "
            "offline recipe and returns honest unresolved live compatibility"
        ),
    )
    live.add_argument(
        "--capability-report",
        type=Path,
        help=(
            "optional capability-owned PyNvVideoCodec GetEncoderCaps report bound to"
            " --environment; when omitted the environment's own capabilities block is"
            " the encoder authority"
        ),
    )
    live.add_argument("--surface", choices=("native", "pynvc"), required=True)
    live.add_argument("--buffer-mode", choices=("cpu", "gpu"), default="gpu")
    live.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = resolve_recipe(strict_json_load(args.intent))
            _write_exclusive(args.output, result)
        elif args.command == "validate":
            validate_recipe(strict_json_load(args.recipe))
            result = {"valid": True, "errors": []}
        else:
            result = validate_live_capabilities(
                strict_json_load(args.recipe),
                (
                    strict_json_load(args.environment)
                    if args.environment is not None
                    else None
                ),
                (
                    strict_json_load(args.capability_report)
                    if args.capability_report is not None
                    else None
                ),
                args.surface,
                environment_path=args.environment,
                buffer_mode=args.buffer_mode,
            )
            if args.output:
                _write_exclusive(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        classification = result.get("classification")
        if not result.get("valid", True) or classification == "unsupported":
            return 2
        if classification == "unknown" and (
            getattr(args, "surface", None) != "native"
            or result.get("authenticated_dry_run_deferred") is not True
        ):
            return 2
        return 0
    except (OSError, RecipeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
