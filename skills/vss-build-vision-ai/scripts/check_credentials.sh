#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Credential gate for vss-build-vision-ai. Checks the keys a deploy needs
# (NGC / NVIDIA_API_KEY / HF_TOKEN) so a bad key fails in seconds, not after a
# cold NIM start. Read-only: it reads env vars and curls — it does NOT write
# override.env (the skill writes the resolved key per credentials.md). The NGC
# key is probed and reported as validated, rejected by the service, not
# validated because the service never answered, or skipped; NVIDIA_API_KEY and
# HF_TOKEN are presence-only for the reasons given at their blocks. Which of
# them are required depends on the deployment mode, so it comes in via --require
# and the exit code is the verdict the caller branches on: 0 gate passed,
# 1 usage error, 2 gate failed. Any non-zero exit blocks.
set -u

usage() {
  cat <<'EOF'
Usage: check_credentials.sh [--require <name>[,<name>...]]...

Validate configured VSS deployment credentials without modifying them.

Options:
  --require ngc         NGC key is required (any local NIM image pull:
                        LLM_MODE / VLM_MODE of local or local_shared)
  --require nvidia-api  NVIDIA_API_KEY is required (a build.nvidia.com /
                        integrate.api.nvidia.com endpoint; any other remote
                        endpoint carries its own key — see credentials.md)
  --require hf          HF_TOKEN is required (a gated or private Hugging Face
                        checkpoint; the Cosmos-Embed defaults are public and
                        need no token)
  -h, --help            Print this help and exit without probing

Environment variables:
  NGC_CLI_API_KEY, NGC_API_KEY  NGC registry key for local NIM images
  NVIDIA_API_KEY                build.nvidia.com API key for remote NIMs
  HF_TOKEN                      Hugging Face token for gated checkpoints

A credential that is unset, rejected, or left unvalidated by an unreachable or
erroring service is a blocker only when its --require name was passed;
otherwise it is reported and does not gate. Conflicting NGC_CLI_API_KEY /
NGC_API_KEY values always gate. The probe is bounded at 5s to connect and 15s
in total, so the gate cannot hang on a host with no egress.

NVIDIA_API_KEY and HF_TOKEN are checked for presence only; neither is probed.
The build.nvidia.com model catalog answers 200 with no credential at all, and so
does a gated repository's Hugging Face metadata, so a 200 from either is not a
verdict on the key. A set key is reported unvalidated rather than claimed valid;
the selected endpoint and checkpoint are probed per credentials.md.

Exit codes:
  0  every required credential present, and validated where probed (NGC only)
  1  usage error — nothing was checked; fix the invocation and re-run
  2  gate failed (see the BLOCKER summary on stderr)

Any non-zero exit blocks.
EOF
  return 0
}

require_ngc=0
require_nvidia=0
require_hf=0

add_requirement() {
  local name names
  if [[ -z "$1" ]]; then
    echo "ERROR: --require needs a value" >&2
    usage >&2
    exit 1
  fi
  # Split on commas with `read`, not unquoted word splitting: the latter also
  # globs, so `--require '*'` run from a directory holding files named ngc or
  # hf would take its requirement set from the filesystem.
  IFS=',' read -ra names <<<"$1"
  for name in "${names[@]}"; do
    case "$name" in
      ngc) require_ngc=1 ;;
      nvidia-api) require_nvidia=1 ;;
      hf) require_hf=1 ;;
      *)
        echo "ERROR: unknown --require value: $name" >&2
        usage >&2
        exit 1
        ;;
    esac
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --require)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --require needs a value" >&2
        usage >&2
        exit 1
      fi
      add_requirement "$2"
      shift 2
      ;;
    --require=*)
      add_requirement "${1#--require=}"
      shift
      ;;
    *)
      echo "ERROR: unexpected argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

blockers=()

# Report a failed probe. It gates only when the caller declared the credential
# required for this build; an unrelated stale key must not block a deploy that
# never reads it.
report_failure() {
  local required="$1" message="$2"
  if [[ "$required" == 1 ]]; then
    echo "$message"
    blockers+=("$message")
  else
    echo "$message — not required by this build"
  fi
}

# HTTP status of a read-only probe, or 000 when the request never completed.
# Deliberately not `curl -f`: -f collapses a refused connection, a DNS failure
# and a real 401 into one non-zero exit, which reports a working key as invalid
# on a host with no egress. Timeouts are mandatory for a gate that promises to
# fail in seconds: without them, blackholed egress or a server that accepts and
# stalls leaves curl on the OS timeout, so the whole preflight hangs instead of
# reporting 000. Same bounds as
# skills/operations/vss-search-archive/scripts/select_brev_origin.sh.
http_status() {
  curl -s -o /dev/null -w '%{http_code}' --connect-timeout 5 --max-time 15 "$@" 2>/dev/null || true
}

# Turn a probe status into a verdict for one credential. Only 401/403 is a
# statement about the credential itself; 000 and 429/5xx mean the service never
# gave one, and reporting those as "rejected" sends someone to rotate a working
# key. Both still gate when the credential is required — an unvalidated
# requirement is not a pass — so this changes the message, not the exit code.
report_status() {
  local status="$1" required="$2" label="$3" host="$4"
  case "$status" in
    2??) echo "$label ok" ;;
    000) report_failure "$required" "$label not validated — $host did not answer within the probe timeout" ;;
    401|403) report_failure "$required" "$label rejected by $host (HTTP $status)" ;;
    429|5??) report_failure "$required" "$label not validated — $host returned HTTP $status (rate limit or service error), which is not a verdict on the credential; retry" ;;
    *) report_failure "$required" "$label not validated — unexpected HTTP $status from $host" ;;
  esac
}

# NGC — local NIM image pulls. NGC_CLI_API_KEY (NGC CLI / VSS env) and
# NGC_API_KEY (NIM / RT-VLM containers) are the SAME personal NGC key under two
# names; resolve to one. Refuse to proceed if both are set and differ.
if [[ -n "${NGC_CLI_API_KEY:-}" ]] && [[ -n "${NGC_API_KEY:-}" ]] && \
   [[ "$NGC_CLI_API_KEY" != "$NGC_API_KEY" ]]; then
  # Unconditional: credentials.md says stop and ask which key to use rather
  # than silently choosing one, whatever the mode needs.
  ngc_conflict="NGC: NGC_CLI_API_KEY and NGC_API_KEY differ — choose one NGC personal API key"
  echo "$ngc_conflict"
  blockers+=("$ngc_conflict")
elif [[ -n "${NGC_CLI_API_KEY:-${NGC_API_KEY:-}}" ]]; then
  ngc_resolved="${NGC_CLI_API_KEY:-${NGC_API_KEY:-}}"
  # Probe the registry pull scope (what image pulls actually use), not
  # service=ngc - a key scoped only for nvcr.io pulls is valid for a deploy
  # but is rejected by the ngc platform scope (false negative).
  ngc_status=$(http_status -u "\$oauthtoken:${ngc_resolved}" \
    "https://authn.nvidia.com/token?service=registry&scope=repository:nvidia/vss-core/vss-agent:pull")
  report_status "$ngc_status" "$require_ngc" "NGC key" "authn.nvidia.com"
elif [[ "$require_ngc" == 1 ]]; then
  ngc_missing="NGC: not set — required for any local NIM image pull"
  echo "$ngc_missing"
  blockers+=("$ngc_missing")
else
  echo "NGC: not set — skip (required for any local NIM)"
fi

# build.nvidia.com — remote NIM endpoints. Presence only, deliberately
# unprobed: /v1/models is the public model catalog and answers 200 with no
# Authorization header at all, so its 200 is not a verdict on the key and the
# 401/403 arm of report_status could never fire for this host. Claiming a key
# validated on a public 200 is worse than saying nothing. The inference route
# enforces auth but cannot be read as a verdict either: a bogus key and a valid
# key without access to the named model both answer 403, so only a 200 is
# informative and earning one needs a model the key is already entitled to,
# which this gate cannot know before the build resolves. probe_remote_models.sh
# clears the endpoint and the model, not the key; a bad key surfaces at the
# first authenticated inference request, per credentials.md.
if [[ -n "${NVIDIA_API_KEY:-}" ]]; then
  echo "NVIDIA_API_KEY: set — not validated here (the model catalog is public); probe the selected endpoint per credentials.md"
elif [[ "$require_nvidia" == 1 ]]; then
  nvidia_missing="NVIDIA_API_KEY: not set — required for a build.nvidia.com endpoint"
  echo "$nvidia_missing"
  blockers+=("$nvidia_missing")
else
  echo "NVIDIA_API_KEY: not set — skip (required only for a build.nvidia.com endpoint)"
fi

# HF — not needed by any in-tree edge path; kept for a gated or private
# checkpoint. Presence only, deliberately unprobed: this gate runs before the
# build resolves, so it has no selected repository to ask about, and no
# repository-free endpoint settles the question. Hugging Face keeps a gated
# repo's metadata and file list public, so /api/models answers 200 for a good
# token, a junk token and none alike. /api/whoami-v2 does separate a valid token
# from a junk one, but it could not be confirmed to accept an ordinary
# fine-grained read token, and a token it accepts can still lack access to the
# checkpoint. Only a repository-scoped probe decides that, which is why
# credentials.md's artifact probes own it once a repository is selected.
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN: set — not validated here; probe the selected checkpoint per credentials.md"
elif [[ "$require_hf" == 1 ]]; then
  hf_missing="HF_TOKEN: not set — required for a gated or private Hugging Face checkpoint"
  echo "$hf_missing"
  blockers+=("$hf_missing")
else
  echo "HF_TOKEN: not set — skip (no in-tree edge path needs it; used by gated or private HF checkpoints)"
fi

if [[ "${#blockers[@]}" -gt 0 ]]; then
  {
    echo "BLOCKER: credential gate failed:"
    for blocker in "${blockers[@]}"; do
      echo "  - $blocker"
    done
  } >&2
  exit 2
fi

exit 0
