#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query the complete PyNvVideoCodec GPU-0 NVDEC capability matrix."""

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
    enum_member,
    exclusive_write_text,
    local_pynvc_binding,
    reauthenticate_local_pynvc_binding,
    require_declared_pynvc_authority,
    require_authenticated_binding,
    scalar_attributes,
    strict_json_dumps,
)

SCHEMA_VERSION = "1.2"
EVIDENCE_SOURCE_TYPE = "api_query_helper"
API_SYMBOL = "PyNvVideoCodec.GetDecoderCaps"

# Producer exit codes, frozen by the public capability contract.
RC_COMPLETE = 0
RC_PARTIAL = 2
RC_FAILED = 3

CODECS = {
    "mpeg1": ("MPEG1", "MPEG_1"),
    "mpeg2": ("MPEG2", "MPEG_2"),
    "mpeg4": ("MPEG4", "MPEG_4"),
    "vc1": ("VC1", "VC_1"),
    "h264": ("H264", "H_264"),
    "hevc": ("HEVC", "H265", "H_265"),
    "vp8": ("VP8", "VP_8"),
    "vp9": ("VP9", "VP_9"),
    "av1": ("AV1",),
    "jpeg": ("JPEG",),
}
CHROMAS = {
    "monochrome": ("Monochrome", "MONOCHROME"),
    "420": ("420", "YUV420"),
    "422": ("422", "YUV422"),
    "444": ("444", "YUV444"),
}
BIT_DEPTHS = (8, 10, 12)


def evidence_classification() -> dict[str, Any]:
    return {
        "evidence_source_type": EVIDENCE_SOURCE_TYPE,
        "official_sample": False,
        "api_symbol": API_SYMBOL,
    }


# Preserve the historical alias while using the capability-owned writer.
atomic_write = exclusive_write_text


def _unknown_records(reason: str, selected_codecs: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "codec": codec,
            "chroma": chroma,
            "bit_depth": bit_depth,
            "status": "unknown",
            "supported": None,
            "limits_applicable": None,
            "values": {},
            "reason": reason,
        }
        for codec in selected_codecs
        for chroma in CHROMAS
        for bit_depth in BIT_DEPTHS
    ]


def add_matrix_counts(result: dict[str, Any]) -> None:
    queries = result.get("queries", [])
    tuples = {
        (record.get("codec"), record.get("chroma"), record.get("bit_depth"))
        for record in queries
        if isinstance(record, dict)
    }
    expected = result.get("matrix", {}).get("expected_query_count")
    result["matrix"].update(
        {
            "observed_query_count": len(queries),
            "unique_tuple_count": len(tuples),
            "exact_matrix_complete": len(queries) == expected and len(tuples) == expected,
        }
    )


def _query_matrix_unchecked(  # pylint: disable=too-many-locals,too-many-branches
    module: Any,
    gpu: int,
    selected_codecs: tuple[str, ...] = tuple(CODECS),
    *,
    environment: dict[str, Any] | None = None,
    runtime_binding: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "nvcodec-decoder-capability-matrix",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "authority": API_SYMBOL,
        "evidence_classification": evidence_classification(),
        "gpu": gpu,
        "matrix": {
            "codecs": list(selected_codecs),
            "chromas": list(CHROMAS),
            "bit_depths": list(BIT_DEPTHS),
            "expected_query_count": len(selected_codecs) * len(CHROMAS) * len(BIT_DEPTHS),
        },
        "queries": [],
    }
    if environment is not None:
        result["environment"] = environment
    else:
        result["runtime_binding"] = runtime_binding
    if gpu != 0:
        reason = (
            "PyNvVideoCodec GetDecoderCaps routes capability queries to GPU 0; nonzero-GPU results"
            " are unknown"
        )
        result.update({"status": "unknown", "reason": reason})
        result["queries"] = _unknown_records(reason, selected_codecs)
        result["summary"] = {
            "capability_reported": 0,
            "unsupported": 0,
            "unknown": len(result["queries"]),
        }
        add_matrix_counts(result)
        return result, 2

    get_caps = getattr(module, "GetDecoderCaps", None)
    if not callable(get_caps):
        reason = "PyNvVideoCodec.GetDecoderCaps is unavailable"
        result.update({"status": "unknown", "reason": reason})
        result["queries"] = _unknown_records(reason, selected_codecs)
        result["summary"] = {
            "capability_reported": 0,
            "unsupported": 0,
            "unknown": len(result["queries"]),
        }
        add_matrix_counts(result)
        return result, 2

    codec_containers = ("cudaVideoCodec", "Codec", "VideoCodec")
    chroma_containers = ("cudaVideoChromaFormat", "ChromaFormat")
    queries: list[dict[str, Any]] = []
    for codec_name in selected_codecs:
        codec_value = enum_member(module, codec_containers, CODECS[codec_name])
        for chroma_name, chroma_aliases in CHROMAS.items():
            chroma_value = enum_member(module, chroma_containers, chroma_aliases)
            for bit_depth in BIT_DEPTHS:
                record: dict[str, Any] = {
                    "codec": codec_name,
                    "chroma": chroma_name,
                    "bit_depth": bit_depth,
                    "status": "unknown",
                    "supported": None,
                    "limits_applicable": None,
                    "values": {},
                }
                if codec_value is None or chroma_value is None:
                    missing = "codec" if codec_value is None else "chroma"
                    record["reason"] = (
                        f"Required {missing} enum is unavailable in the loaded module"
                    )
                    queries.append(record)
                    continue
                try:
                    values = scalar_attributes(get_caps(gpu, codec_value, chroma_value, bit_depth))
                    record["values"] = values
                    supported = (
                        values["supported"]
                        if "supported" in values
                        else values.get("bIsSupported")
                    )
                    if supported in (0, False):
                        record.update(
                            {
                                "status": "unsupported",
                                "supported": False,
                                "limits_applicable": False,
                                "interpretation": (
                                    "bIsSupported=0; remaining zeroed OUT fields are inapplicable"
                                ),
                            }
                        )
                    elif isinstance(supported, (int, bool)) and int(supported) > 0:
                        record.update(
                            {
                                "status": "capability_reported",
                                "supported": True,
                                "limits_applicable": True,
                                "interpretation": "bIsSupported=1; returned limits are applicable",
                            }
                        )
                    else:
                        record["reason"] = "bIsSupported was absent or not interpretable"
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    record.update({"reason": "GetDecoderCaps failed", "error": str(exc)})
                queries.append(record)

    result["queries"] = queries
    add_matrix_counts(result)
    summary = {
        state: sum(1 for record in queries if record["status"] == state)
        for state in ("capability_reported", "unsupported", "unknown")
    }
    result["summary"] = summary
    if summary["unknown"]:
        result.update(
            {"status": "partial", "reason": "One or more requested tuples remain unknown"}
        )
        return result, 2
    result["status"] = "complete"
    return result, 0


def query_matrix(
    module: Any,
    gpu: int,
    selected_codecs: tuple[str, ...] = tuple(CODECS),
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
                "decoder environment binding differs from current manifest authority"
            )
    else:
        runtime_binding = local_pynvc_binding(
            module, gpu=gpu, preimport_validation=preimport_validation
        )
    result, code = _query_matrix_unchecked(
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
                "decoder environment manifest or wheel identity changed during queries"
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
    queries = _unknown_records(reason, selected_codecs)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "nvcodec-decoder-capability-matrix",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "authority": API_SYMBOL,
        "evidence_classification": evidence_classification(),
        "gpu": gpu,
        "status": "unknown",
        "reason": reason,
        "queries": queries,
        "matrix": {
            "codecs": list(selected_codecs),
            "chromas": list(CHROMAS),
            "bit_depths": list(BIT_DEPTHS),
            "expected_query_count": len(queries),
        },
        "summary": {
            "capability_reported": 0,
            "unsupported": 0,
            "unknown": len(queries),
        },
    }
    if error is not None:
        result["error"] = error
    add_matrix_counts(result)
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
    parser.add_argument("--codec", action="append", choices=tuple(CODECS), dest="codecs")
    parser.add_argument("--output", type=Path)
    try:
        args = parser.parse_args()
        if args.gpu < 0:
            parser.error("--gpu must be non-negative")
    except ValueError as exc:
        print(strict_json_dumps({
            "schema_version": SCHEMA_VERSION,
            "kind": "nvcodec-decoder-capability-matrix-error",
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
            result, code = query_matrix(
                nvc,
                args.gpu,
                selected_codecs,
                environment=binding,
                preimport_validation=preimport_validation,
            )
        else:
            result, code = query_matrix(
                nvc,
                args.gpu,
                selected_codecs,
                preimport_validation=preimport_validation,
            )
    except (CapabilityAuthorityUnavailable, ImportError) as exc:
        reason = "decoder capability authority is unavailable"
        result = unknown_result(
            reason, gpu=args.gpu, selected_codecs=selected_codecs, error=str(exc)
        )
        add_setup_remediation(result, surface="pynvc", reason=str(exc))
        code = RC_PARTIAL
    except Exception as exc:  # pylint: disable=broad-exception-caught
        reason = "decoder capability authority authentication failed"
        result = unknown_result(
            reason, gpu=args.gpu, selected_codecs=selected_codecs, error=str(exc)
        )
        code = RC_FAILED
    rendered = strict_json_dumps(result, indent=2, sort_keys=True) + "\n"
    write_error = None
    if args.output:
        try:
            atomic_write(args.output, rendered)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            write_error = str(exc)
    print(rendered, end="")
    if write_error is not None:
        print(f"decoder capability report write failed: {write_error}", file=sys.stderr)
        return RC_FAILED
    return code


if __name__ == "__main__":
    raise SystemExit(main())
