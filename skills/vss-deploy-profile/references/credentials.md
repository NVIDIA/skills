# Credential Gate

Run this before mutating `generated.env` or starting any image pull. Validate credentials early: a bad key should fail in seconds, not after a cold NIM start.

## Required By Mode

- `NGC_CLI_API_KEY`: required for any local NIM image pull (`LLM_MODE` or `VLM_MODE` set to `local` / `local_shared`).
- `NVIDIA_API_KEY`: required for remote NIM endpoints.
- `HF_TOKEN`: required on edge targets that use the gated Edge 4B model.

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

## Decision Rule

A key reported `invalid` that the chosen mode needs, or a `skip` for a key the mode requires, is a blocker. Prompt the user, re-probe, and do not proceed to env mutation until it resolves.

A `skip` for a key the mode does not use is fine.
