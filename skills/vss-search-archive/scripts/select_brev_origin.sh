#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Prefer a validated public VSS origin, otherwise use the supplied host origin.
# JSON is the only stdout so callers can consume the decision deterministically.
set -u

if [[ "$#" -ne 2 ]]; then
  echo "usage: select_brev_origin.sh <public-https-origin-or-empty> <host-origin>" >&2
  exit 2
fi

PUBLIC_ORIGIN=${1%/}
HOST_ORIGIN=${2%/}
if [[ -z "${HOST_ORIGIN}" ]]; then
  echo "deployment-provided host origin is required" >&2
  exit 2
fi
if [[ -z "${PUBLIC_ORIGIN}" ]]; then
  echo "no public VSS origin published; media URLs will be host-local" >&2
  jq -cn --arg origin "${HOST_ORIGIN}" '{origin: $origin, media_scope: "host-local"}'
  exit 0
fi
if [[ ! "${PUBLIC_ORIGIN}" =~ ^https://[^/?#[:space:]]+$ ]]; then
  echo "public VSS origin must be an HTTPS origin; media URLs will be host-local" >&2
  jq -cn --arg origin "${HOST_ORIGIN}" '{origin: $origin, media_scope: "host-local"}'
  exit 0
fi
PROBE_BODY=$(mktemp /tmp/vss-public-vst.XXXXXX) || exit 1
trap 'rm -f -- "${PROBE_BODY}"' EXIT

if ! PROBE_STATUS=$(curl -sS --connect-timeout 5 --max-time 15 \
  --max-redirs 0 -o "${PROBE_BODY}" -w '%{http_code}' \
  "${PUBLIC_ORIGIN}/vst/api/v1/sensor/version" 2>/dev/null); then
  PROBE_STATUS=000
fi

if [[ "${PROBE_STATUS}" == 200 ]] &&
   jq -e '.type == "vst" and (.version | type == "string" and length > 0)' \
     "${PROBE_BODY}" >/dev/null 2>&1; then
  jq -cn --arg origin "${PUBLIC_ORIGIN}" '{origin: $origin, media_scope: "public"}'
  exit 0
fi

echo "public VST validation failed (HTTP ${PROBE_STATUS}); media URLs will be host-local" >&2
jq -cn --arg origin "${HOST_ORIGIN}" '{origin: $origin, media_scope: "host-local"}'
