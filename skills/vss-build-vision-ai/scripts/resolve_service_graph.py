#!/usr/bin/env -S uv run --quiet --script
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply host-CLI ownership rules and derive analytics readiness probes."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from dataclasses import dataclass

VSS_AGENT = "vss-agent"
VSS_VA_MCP = "vss-va-mcp"
PHOENIX = "phoenix"
# The Foundation LLM key is dynamic (llm_${LLM_MODE}_${LLM_NAME_SLUG}); these are the
# services that call it. With none of them enabled the NIM serves nothing.
LLM_PREFIX = "llm_"
LLM_CONSUMERS = frozenset({VSS_AGENT, "lvs-server"})


class UnexpectedHarnessDeltaError(ValueError):
    """A harness-only delta changed unrelated Foundation profiles."""


@dataclass(frozen=True)
class ReadinessTarget:
    service: str
    url: str


ANALYTICS_READINESS_TARGETS = (
    ReadinessTarget(
        "vss-video-analytics-api",
        "http://${HOST_IP}:${VIDEO_ANALYTICS_API_HOST_PORT:-8081}/livez",
    ),
    ReadinessTarget(
        "alert-bridge",
        "http://${HOST_IP}:${ALERT_BRIDGE_HOST_PORT:-9080}/health",
    ),
    ReadinessTarget(
        VSS_VA_MCP,
        "http://${HOST_IP}:${VSS_VA_MCP_HOST_PORT:-9901}/health",
    ),
)


def resolve_service_profiles(
    foundation_profiles: Iterable[str],
    requested_profiles: Iterable[str] = (),
    excluded_profiles: Iterable[str] = (),
    *,
    host_cli: bool,
) -> tuple[str, ...]:
    """Return an ordered profile set after applying explicit ownership rules."""
    requested = tuple(requested_profiles)
    profiles = dict.fromkeys((*foundation_profiles, *requested))

    if host_cli:
        explicitly_requested = set(requested)
        # The in-stack agent leaves with the two peers only it uses: phoenix, which
        # collects its traces and has no other client, and the llm_* key, once no
        # enabled service still calls the LLM (lvs-server does; alert-bridge does not).
        # An explicit request for any of them - a harness pointed at the build's own
        # NIM asks for the llm_* key - keeps it.
        for profile in (VSS_AGENT, VSS_VA_MCP, PHOENIX):
            if profile not in explicitly_requested:
                profiles.pop(profile, None)
        if not (LLM_CONSUMERS & set(profiles)):
            for profile in [p for p in profiles if p.startswith(LLM_PREFIX)]:
                if profile not in explicitly_requested:
                    profiles.pop(profile, None)

    for profile in excluded_profiles:
        profiles.pop(profile, None)

    return tuple(profiles)


def validate_harness_only_delta(
    foundation_profiles: Iterable[str],
    final_profiles: Iterable[str],
    requested_profiles: Iterable[str] = (),
) -> None:
    """Require a Q3-only delta to preserve every Foundation profile the agent did not own."""
    foundation = tuple(foundation_profiles)
    expected = resolve_service_profiles(
        foundation,
        requested_profiles=requested_profiles,
        host_cli=True,
    )
    actual = tuple(final_profiles)
    if set(actual) == set(expected):
        return

    expected_set = set(expected)
    actual_set = set(actual)
    unexpected_removed = sorted(expected_set - actual_set)
    unexpected_added = sorted(actual_set - expected_set)
    details = []
    if unexpected_removed:
        details.append(f"unexpected removals: {', '.join(unexpected_removed)}")
    if unexpected_added:
        details.append(f"unexpected additions: {', '.join(unexpected_added)}")
    raise UnexpectedHarnessDeltaError(
        "harness-only delta must preserve the Foundation except for the agent-owned "
        f"removals - vss-agent, its llm_* key and phoenix, an unrequested vss-va-mcp ({'; '.join(details)})"
    )


def _profile_list(value: str) -> tuple[str, ...]:
    return tuple(profile.strip() for profile in value.split(",") if profile.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reject a harness-only COMPOSE_PROFILES list that dropped or added "
            "unrelated Foundation profiles."
        )
    )
    parser.add_argument(
        "--foundation",
        required=True,
        help="comma-separated Foundation COMPOSE_PROFILES",
    )
    parser.add_argument(
        "--final",
        required=True,
        help="comma-separated effective COMPOSE_PROFILES after the Q3 delta",
    )
    parser.add_argument(
        "--requested",
        default="",
        help="comma-separated explicitly requested profiles, if any",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_harness_only_delta(
            _profile_list(args.foundation),
            _profile_list(args.final),
            requested_profiles=_profile_list(args.requested),
        )
    except UnexpectedHarnessDeltaError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "Validated harness-only delta: Foundation preserved except the agent-owned removals"
    )
    return 0


def analytics_readiness_targets(
    resolved_services: Iterable[str],
) -> tuple[ReadinessTarget, ...]:
    """Return only analytics probes whose owning service resolved."""
    selected = set(resolved_services)
    return tuple(
        target for target in ANALYTICS_READINESS_TARGETS if target.service in selected
    )


if __name__ == "__main__":
    raise SystemExit(main())
