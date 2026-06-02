# Credential Gate

Run this before mutating `generated.env` or starting any image pull. Validate credentials early: a bad key should fail in seconds, not after a cold NIM start.

## Required By Mode

- `NGC_CLI_API_KEY`: required for any local NIM image pull (`LLM_MODE` or `VLM_MODE` set to `local` / `local_shared`).
- `NVIDIA_API_KEY`: required for remote NIM endpoints.
- `HF_TOKEN`: required on edge targets that use the gated Edge 4B model.
- Customer LLM/VLM endpoint URL + model name: required for any selected
  remote endpoint. This includes build.nvidia.com / NVIDIA API catalog
  endpoints because their `/v1/models` response can list many models.

## Discovery

Surface discovered credentials to the user; do not auto-source them without confirmation.

- If `$NGC_CLI_API_KEY` is unset but `~/.ngc/config` exists, extract the account metadata and ask: `Use NGC account <org>/<team> for the deploy?`
- If `$HF_TOKEN` is unset but `~/.cache/huggingface/token` exists, ask before exporting it.

## Probes

Run each probe only when the corresponding key is set. An unset key prints `skip`; compare the result with the chosen deployment mode before continuing.

```bash
# NGC — local NIM image pulls
if [ -n "$NGC_CLI_API_KEY" ]; then
  curl -sf -u "\$oauthtoken:$NGC_CLI_API_KEY" \
    "https://authn.nvidia.com/token?service=ngc" >/dev/null \
    && echo "NGC_CLI_API_KEY ok" || echo "NGC_CLI_API_KEY invalid (401/403)"
else
  echo "NGC_CLI_API_KEY not set — skip (required for any local NIM)"
fi

# build.nvidia.com — remote NIM endpoints
if [ -n "$NVIDIA_API_KEY" ]; then
  curl -sf -H "Authorization: Bearer $NVIDIA_API_KEY" \
    "https://integrate.api.nvidia.com/v1/models" >/dev/null \
    && echo "NVIDIA_API_KEY ok" || echo "NVIDIA_API_KEY invalid (401/403)"
else
  echo "NVIDIA_API_KEY not set — skip (required only for remote NIM)"
fi

# HF — edge only (gated Edge 4B)
if [ -n "$HF_TOKEN" ]; then
  status=$(curl -sf -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-Edge-4B-v2.1-EA-020126_FP8")
  [ "$status" = "200" ] \
    && echo "HF_TOKEN ok" \
    || echo "HF_TOKEN invalid or no access to gated Edge 4B (HTTP $status)"
else
  echo "HF_TOKEN not set — skip (required only on edge with Edge 4B)"
fi
```

## Remote Endpoint Probes

For every selected remote LLM/VLM endpoint, probe the endpoint before writing
it into `generated.env`. Do this even when the endpoint is on localhost; it
catches wrong ports, stale tunnels, missing auth, and model-name mismatches
before the deploy flow spends time generating compose or warming containers.

Use the base URL without a trailing `/v1`; the script strips `/v1` and
`/v1/models` if the user supplied them. If the endpoint requires auth, set
`REMOTE_API_KEY` to the key that the agent will use for that endpoint.

Aggregate endpoints such as `https://integrate.api.nvidia.com` can advertise
many LLM and VLM models. Do not auto-select the first returned model from such
endpoints. If the endpoint lists multiple models and the user has not selected
an exact model id, stop and ask which model to use.

Run the skill script:

```bash
REMOTE_API_KEY="$NVIDIA_API_KEY" \
  skills/vss-deploy-profile/scripts/probe_remote_models.sh "$LLM_BASE_URL" "$LLM_NAME"

skills/vss-deploy-profile/scripts/probe_remote_models.sh \
  "http://localhost:30081" "nvidia/nvidia-nemotron-nano-9b-v2-dgx-spark"
```

If `/v1/models` fails or does not advertise the selected model, stop and ask
the user for the correct endpoint/model before mutating `generated.env`.

## Decision Rule

A key reported `invalid` that the chosen mode needs, a `skip` for a key the
mode requires, or a selected remote endpoint that fails `/v1/models` is a
blocker. Prompt the user, re-probe, and do not proceed to env mutation until it
resolves.

A `skip` for a key the mode does not use is fine.
