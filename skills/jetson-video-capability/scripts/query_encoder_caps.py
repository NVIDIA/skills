#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish the capability-owned PyNvVideoCodec GPU-0 NVENC capability report."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).absolute().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# pylint: disable=wrong-import-position
from _capability_contract import (  # noqa: E402
    add_setup_remediation,
    assert_safe_python_import_environment,
    authenticated_environment,
    CapabilityAuthorityUnavailable,
    ENVIRONMENT_SCHEMA_VERSION,
    exclusive_write_text,
    linked_nvenc_api,
    local_pynvc_binding,
    reauthenticate_local_pynvc_binding,
    require_declared_pynvc_authority,
    require_authenticated_binding,
    scalar_attributes,
    strict_json_dumps,
)

SCHEMA_VERSION = "1.0"
REPORT_KIND = "nvcodec-encoder-capability-report"
AUTHORITY = "pynvc"
EVIDENCE_SOURCE_TYPE = "api_query_helper"
API_SYMBOL = "PyNvVideoCodec.GetEncoderCaps"

# GetEncoderCaps takes the codec label directly; there is no enum indirection.
CODECS = ("h264", "hevc", "av1")

# Capabilities that exist only in the NVENC 13.0 header. When the loaded
# extension links NVENC 12.1 their absence is unqueryable, never unsupported.
API_13_ONLY_CAPS = (
    "support_yuv422_encode",
    "support_mvhevc_encode",
    "support_temporal_filter",
    "support_lookahead_level",
    "support_unidirectional_b",
)

SESSION_STATUS = "opened_by_GetEncoderCaps"
OPERATION_STATUS = "not_tested"
INTERPRETATION = (
    "GetEncoderCaps returned capability fields but no explicit codec support result;"
    " an official encode operation is still required"
)

# Producer exit codes, frozen.
RC_COMPLETE = 0
RC_PARTIAL = 2
RC_FAILED = 3
GPU_ZERO_REASON = (
    "PyNvVideoCodec GetEncoderCaps discards the requested GPU and selects device 0;"
    " nonzero-GPU encoder capability support remains unknown"
)


def evidence_classification() -> dict[str, Any]:
    """Publish the closed per-family API-helper classification recipe requires."""
    return {
        "encode": {
            "evidence_source_type": EVIDENCE_SOURCE_TYPE,
            "official_sample": False,
            "api_symbol": API_SYMBOL,
        }
    }


def _unknown_capability(reason: str) -> dict[str, Any]:
    return {"status": "unknown", "supported": None, "reason": reason}


def _failed_capability(exc: Exception) -> dict[str, Any]:
    return {"status": "unknown", "query_status": "error", "supported": None, "error": str(exc)}


def _unavailable_fields(values: dict[str, Any], linked_api: dict[str, Any]) -> dict[str, Any]:
    if linked_api.get("value") != "12.1":
        return {}
    return {
        key: {
            "status": "unknown",
            "reason": "capability is absent from the linked NVENC 12.1 header",
        }
        for key in API_13_ONLY_CAPS
        if key not in values
    }


def _capability_record(values: dict[str, Any], linked_api: dict[str, Any]) -> dict[str, Any]:
    # A positive query is `capability_reported` with `supported: null`, never
    # `supported: true`. The API reports a capability, not product support.
    return {
        "status": "capability_reported",
        "supported": None,
        "session_status": SESSION_STATUS,
        "operation_status": OPERATION_STATUS,
        "interpretation": INTERPRETATION,
        "values": values,
        "unavailable_fields": _unavailable_fields(values, linked_api),
    }


def _summarize(result: dict[str, Any]) -> None:
    records = result["encode"].values()
    result["summary"] = {
        state: sum(1 for record in records if record.get("status") == state)
        for state in ("capability_reported", "unsupported", "unknown")
    }


def _query_encoder_caps_unchecked(
    module: Any,
    gpu: int,
    selected_codecs: tuple[str, ...],
    *,
    environment: dict[str, Any] | None = None,
    runtime_binding: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    linked_api = linked_nvenc_api(module)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "authority": AUTHORITY,
        "gpu": gpu,
        "evidence_classification": evidence_classification(),
        "linked_nvenc_api": linked_api,
        "codecs": list(selected_codecs),
        "encode": {},
    }
    if environment is not None:
        result["environment"] = environment
    else:
        result["runtime_binding"] = runtime_binding
    if gpu != 0:
        result.update({"status": "unknown", "reason": GPU_ZERO_REASON})
        result["encode"] = {
            codec: _unknown_capability(GPU_ZERO_REASON) for codec in selected_codecs
        }
        _summarize(result)
        return result, RC_PARTIAL

    get_encoder_caps = getattr(module, "GetEncoderCaps", None)
    for codec in selected_codecs:
        if not callable(get_encoder_caps):
            result["encode"][codec] = _unknown_capability("GetEncoderCaps is unavailable")
            continue
        try:
            values = scalar_attributes(get_encoder_caps(gpu, codec))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result["encode"][codec] = _failed_capability(exc)
            continue
        if not values:
            result["encode"][codec] = _unknown_capability("GetEncoderCaps returned no fields")
            continue
        result["encode"][codec] = _capability_record(values, linked_api)

    _summarize(result)
    if result["summary"]["unknown"]:
        result.update(
            {"status": "partial", "reason": "One or more requested codecs remain unknown"}
        )
        return result, RC_PARTIAL
    result["status"] = "complete"
    return result, RC_COMPLETE


def query_encoder_caps(
    module: Any,
    gpu: int,
    selected_codecs: tuple[str, ...] = CODECS,
    *,
    environment: dict[str, Any] | None = None,
    preimport_validation: object | None = None,
) -> tuple[dict[str, Any], int]:
    """Use supplied setup evidence or derive local authority inside this process."""
    runtime_binding = None
    if environment is not None:
        require_authenticated_binding(environment, gpu=gpu)
        _, initial_binding = authenticated_environment(
            Path(environment.get("path", "")),
            gpu=gpu,
            module=module,
            preimport_validation=preimport_validation,
        )
        if initial_binding != environment:
            raise ValueError(
                "encoder environment binding differs from current manifest authority"
            )
    else:
        runtime_binding = local_pynvc_binding(
            module, gpu=gpu, preimport_validation=preimport_validation
        )
    result, code = _query_encoder_caps_unchecked(
        module,
        gpu,
        selected_codecs,
        environment=environment,
        runtime_binding=runtime_binding,
    )
    if environment is not None:
        _, terminal_binding = authenticated_environment(
            Path(environment["path"]),
            gpu=gpu,
            module=module,
            preimport_validation=preimport_validation,
        )
        if terminal_binding != environment:
            raise ValueError(
                "encoder environment manifest or wheel identity changed during queries"
            )
    else:
        reauthenticate_local_pynvc_binding(
            runtime_binding,
            gpu=gpu,
            module=module,
            preimport_validation=preimport_validation,
        )
    return result, code


def unknown_result(
    reason: str, *, gpu: int, selected_codecs: tuple[str, ...], error: str | None = None
) -> dict[str, Any]:
    """Report an unqueried encoder surface without claiming absence."""
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "authority": AUTHORITY,
        "gpu": gpu,
        "evidence_classification": evidence_classification(),
        "linked_nvenc_api": {
            "status": "unknown",
            "value": None,
            "module_suffix": None,
            "extension_file": None,
            "provenance": "loaded PyNvVideoCodec extension filename/module_suffix",
        },
        "codecs": list(selected_codecs),
        "status": "unknown",
        "reason": reason,
        "encode": {codec: _unknown_capability(reason) for codec in selected_codecs},
    }
    if error is not None:
        result["error"] = error
    _summarize(result)
    return result


class _ExitThreeParser(argparse.ArgumentParser):
    """Report malformed input with the frozen producer exit code."""

    def error(self, message: str) -> Any:
        raise ValueError(f"{self.prog}: {message}")


def main() -> int:
    parser = _ExitThreeParser(description=__doc__)
    parser.add_argument(
        "--environment",
        type=Path,
        help=(
            f"Optional validated live schema-{ENVIRONMENT_SCHEMA_VERSION} "
            "nvcodec-environment manifest; when omitted authenticate the exact "
            "invoking interpreter and wheel locally"
        ),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--codec", action="append", choices=CODECS, dest="codecs")
    parser.add_argument("--output", type=Path)
    try:
        args = parser.parse_args()
        if args.gpu < 0:
            parser.error("--gpu must be non-negative")
    except ValueError as exc:
        print(strict_json_dumps({
            "schema_version": SCHEMA_VERSION,
            "kind": f"{REPORT_KIND}-error",
            "status": "error",
            "error": str(exc),
        }, sort_keys=True))
        return RC_FAILED
    selected_codecs = tuple(dict.fromkeys(args.codecs or CODECS))
    try:
        if args.environment is not None:
            require_declared_pynvc_authority(args.environment, gpu=args.gpu)
        preimport_validation = assert_safe_python_import_environment()
        import PyNvVideoCodec as nvc  # pylint: disable=import-error,import-outside-toplevel
        if args.environment is not None:
            _, binding = authenticated_environment(
                args.environment,
                gpu=args.gpu,
                module=nvc,
                preimport_validation=preimport_validation,
            )
            result, code = query_encoder_caps(
                nvc,
                args.gpu,
                selected_codecs,
                environment=binding,
                preimport_validation=preimport_validation,
            )
        else:
            result, code = query_encoder_caps(
                nvc,
                args.gpu,
                selected_codecs,
                preimport_validation=preimport_validation,
            )
    except CapabilityAuthorityUnavailable as exc:
        reason = "encoder capability authority is unavailable"
        result = unknown_result(
            reason, gpu=args.gpu, selected_codecs=selected_codecs, error=str(exc)
        )
        add_setup_remediation(result, surface="pynvc", reason=str(exc))
        code = RC_PARTIAL
    except ImportError as exc:
        reason = "encoder capability authority is unavailable"
        result = unknown_result(
            reason, gpu=args.gpu, selected_codecs=selected_codecs, error=str(exc)
        )
        add_setup_remediation(result, surface="pynvc", reason=str(exc))
        code = RC_PARTIAL
    except Exception as exc:  # pylint: disable=broad-exception-caught
        reason = "encoder capability authority authentication failed"
        result = unknown_result(
            reason, gpu=args.gpu, selected_codecs=selected_codecs, error=str(exc)
        )
        code = RC_FAILED
    rendered = strict_json_dumps(result, indent=2, sort_keys=True) + "\n"
    write_error = None
    if args.output:
        try:
            exclusive_write_text(args.output, rendered)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            write_error = str(exc)
    print(rendered, end="")
    if write_error is not None:
        print(f"encoder capability report write failed: {write_error}", file=sys.stderr)
        return RC_FAILED
    return code


if __name__ == "__main__":
    raise SystemExit(main())
