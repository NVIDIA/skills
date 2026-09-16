# RTVI-VLM Health Checks — Model Identity and Diagnostics

Continues the core health check in `SKILL.md`. All snippets assume the
variables established there:

```bash
cd "${VSS_DEPLOY_DIR}"
ALERTS_ENV=developer-profiles/dev-profile-alerts/generated.env
export RTVI_VLM_PORT=8018
```

and that `VLM_METHOD` and `ACTUAL_MODEL` have already been captured.

## Method A identity check (`openai-compat`)

For Method A only, verify that the configured deployment name appears in the
advertised `/v1/models` list:

```bash
test "${VLM_METHOD}" = "openai-compat"
EXPECTED_MODEL="$(docker compose --env-file "${ALERTS_ENV}" exec -T rtvi-vlm \
  printenv VIA_VLM_OPENAI_MODEL_DEPLOYMENT_NAME | tr -d '\r')"
test -n "${EXPECTED_MODEL}"

docker compose --env-file "${ALERTS_ENV}" exec -T rtvi-vlm \
  curl -fsS http://localhost:8000/v1/models \
  | jq -e --arg model "${EXPECTED_MODEL}" 'any(.data[]; .id == $model)' \
  || { echo "Endpoint does not advertise '${EXPECTED_MODEL}'" >&2; exit 1; }
```

## Method B identity inspection (`vllm-compatible`)

For Method B, do **not** compare `/v1/models[0].id` with
`VIA_VLM_OPENAI_MODEL_DEPLOYMENT_NAME` or `VLM_NAME`. The in-container server
derives its advertised ID from the basename of the resolved model directory,
while `VLM_NAME` is populated independently. For example:

```text
MODEL_PATH:     ngc:nim/nvidia/cosmos-reason2-8b:hf-1208
advertised ID:  nim_nvidia_cosmos-reason2-8b_hf-1208
```

Inspect and retain the live advertised ID:

```bash
test "${VLM_METHOD}" = "vllm-compatible"
MODEL_PATH="$(docker compose --env-file "${ALERTS_ENV}" exec -T rtvi-vlm \
  printenv MODEL_PATH | tr -d '\r')"
test -n "${MODEL_PATH}" && test -n "${ACTUAL_MODEL}"
printf 'Method B MODEL_PATH: %s\nMethod B advertised model ID: %s\n' \
  "${MODEL_PATH}" "${ACTUAL_MODEL}"
```

`MODEL_PATH` tells RTVI-VLM where to obtain the weights; it is not necessarily
the model identifier accepted by the OpenAI-compatible API. Configure
downstream consumers (`vlm-as-verifier`, `vss-agent`, and any other client)
with `VLM_NAME=${ACTUAL_MODEL}` from `/v1/models`, then regenerate the profile
environment and recreate those consumers. Do not use the original NGC or
Hugging Face path as the API model field.

## Warm-up diagnostics and consumer logs

```bash
# Bad: "model does not exist" -> wrong Method A deployment name or a downstream
# consumer that is not using the Method B advertised ID.
# Bad: connection error -> endpoint unreachable.
docker compose --env-file "${ALERTS_ENV}" logs --since 10m rtvi-vlm

# Verifier and interactive agent consumers.
docker compose --env-file "${ALERTS_ENV}" logs --since 10m alert-bridge
docker compose --env-file "${ALERTS_ENV}" logs --since 10m vss-agent
```

Read these logs for diagnosis only. `Warmup VlmProcess-<gpu-ids> done` is logged
in v3.2.1 even when warm-up raised an exception, so the conclusive line is
`Error during warmup` or `Error during model warmup` — and its absence, not the
completion line, is what a passing check requires. The completion line also names
one process at a time, so a `-0` grep ignores every other configured process.

A bad `MODEL_PATH` aborts startup during download, before the server ever binds
its port, so the container exits non-zero and no endpoint check will succeed
until the weights source is resolved. v3.2.1 raises a plain `Exception` with one
of `Failed to download model <name> from <url>`,
`Model download failed with status code <code>`,
`Could not authenticate with NGC.`, or `Could not find the model.`. Gate on the
container's exit status and treat these messages as the explanation.

## Inference check (both methods)

Model identity is a metadata check. Finish with one real request, sending the
advertised `${ACTUAL_MODEL}` rather than `MODEL_PATH` or `VLM_NAME`:

```bash
docker compose --env-file "${ALERTS_ENV}" exec -T rtvi-vlm \
  curl -fsS --max-time 120 -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "$(jq -nc --arg m "${ACTUAL_MODEL}" '{model: $m, max_tokens: 16, stream: false,
        messages: [{role: "user", content: "Reply with: ok"}]}')" \
  | jq -e '((.choices[0].message.content // "") | length) > 0'
```

The `jq` assertion is required: a pipeline that produced no output still returns
HTTP 200 with `content` set to `""`, so `curl -f` passes on its own.

## Querying a specific external endpoint

If multiple external inference servers use different ports, query the exact
endpoint from inside the RTVI container. For the Blueprint deployment, this
checks the container-native `VIA_VLM_ENDPOINT` derived from
`RTVI_VLM_ENDPOINT`:

```bash
docker compose --env-file "${ALERTS_ENV}" exec -T rtvi-vlm sh -lc \
  'set --
   if [ -n "${VIA_VLM_API_KEY:-}" ]; then
     set -- -H "Authorization: Bearer ${VIA_VLM_API_KEY}"
   fi
   curl -fsS "$@" "${VIA_VLM_ENDPOINT%/}/models"'
```

For the standalone Compose stack, run the equivalent check in `rtvi-server`:

```bash
docker compose exec -T rtvi-server sh -lc \
  'set --
   if [ -n "${VIA_VLM_API_KEY:-}" ]; then
     set -- -H "Authorization: Bearer ${VIA_VLM_API_KEY}"
   fi
   curl -fsS "$@" "${VIA_VLM_ENDPOINT%/}/models"'
```
