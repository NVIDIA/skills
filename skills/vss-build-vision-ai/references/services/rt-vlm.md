# RT-VLM Capability Owner

## Capabilities and service keys

| Capability | Canonical service profile key |
|---|---|
| Streaming and file VLM inference | `rtvi-vlm` |

## Required peers

- Integrated mode needs model credentials/cache access but no standalone VLM
  NIM service.
- OpenAI-compatible mode needs a reachable endpoint and matching `VLM_NAME`.
- Kafka is required when `RTVI_VLM_MESSAGE_BUS=kafka` (the current Compose default).
- Without the Agent tier, call VLM Q&A directly at
  `POST http://<host>:${RTVI_VLM_PORT:-8018}/v1/chat/completions`.
- Generated captions use `RTVI_VLM_MESSAGE_BUS_TOPIC`, which defaults to
  `mdx-vlm-captions`. Do not set the obsolete `RTVI_VLM_KAFKA_TOPIC`; it is not
  forwarded by the current Compose service. When the requested topic is the
  default, enable Kafka without repeating the topic in the delta. VSS Compose
  still forwards legacy `RTVI_VLM_KAFKA_ENABLED`, although RT-VLM 26.08.2 does
  not consume `KAFKA_ENABLED`; keep it aligned with `RTVI_VLM_MESSAGE_BUS`
  until that compatibility field is removed.
- Redis is required when `RTVI_VLM_MESSAGE_BUS=redis` or
  `RTVI_VLM_ERROR_BUS=redis`; `ENABLE_REDIS_ERROR_MESSAGES` is only a legacy
  alias for the latter.
- Do not add `vlm_${VLM_MODE}_${VLM_NAME_SLUG}` for an integrated RT-VLM path.

## Available integrated model variants

These are the Cosmos Reason3 checkpoints RT-VLM can load on the integrated path.
`RTVI_VLM_MODEL_PATH` selects the checkpoint; `VLM_NAME` is the id RT-VLM then
advertises at `GET /v1/models`, derived from the path by dropping the `ngc:`
prefix and replacing `/` and `:` with `_`. Take both values from the same row —
a mismatch makes consumers fail with `400 BadParameters: No such model`.

| Model | `VLM_NAME` | `RTVI_VLM_MODEL_PATH` |
| --- | --- | --- |
| CR3 Nano BF16 | `nim_nvidia_cosmos3-nano-reasoner_bf16-final` | `ngc:nim/nvidia/cosmos3-nano-reasoner:bf16-final` |
| CR3 Nano FP8 | `nim_nvidia_cosmos3-nano-reasoner_modelopt-fp8-final_format_fix` | `ngc:nim/nvidia/cosmos3-nano-reasoner:modelopt-fp8-final_format_fix` |
| CR3 Nano NVFP4 | `nim_nvidia_cosmos3-nano-reasoner_modelopt-nvfp4-full-quantize-final_format_fix` | `ngc:nim/nvidia/cosmos3-nano-reasoner:modelopt-nvfp4-full-quantize-final_format_fix` |
| CR3 Super BF16 | `nim_nvidia_cosmos3-super-reasoner_bf16-final` | `ngc:nim/nvidia/cosmos3-super-reasoner:bf16-final` |
| CR3 Super FP8 | `nim_nvidia_cosmos3-super-reasoner_modelopt-fp8-final_format_fix` | `ngc:nim/nvidia/cosmos3-super-reasoner:modelopt-fp8-final_format_fix` |
| CR3 Super NVFP4 | `nim_nvidia_cosmos3-super-reasoner_modelopt-nvfp4-full-quantize-final_format_fix` | `ngc:nim/nvidia/cosmos3-super-reasoner:modelopt-nvfp4-full-quantize-final_format_fix` |

Notes on choosing a row:

- **Default to the profile, not to this table.** When the user does not name a
  model or quantization, keep the variant the selected deployment profile already
  ships and leave `RTVI_VLM_MODEL_PATH` / `VLM_NAME` untouched (`alerts` and
  `lvs` ship Nano BF16, `search` ships Nano FP8). Consult this table only when
  the user asks for a specific variant, or when placement forces a change under
  [Singleton and variant convergence](#singleton-and-variant-convergence).
- Nano is the default family for every profile in this repo; select Super only on
  an explicit request.
- Within a family, quantization is a memory/placement decision, not a capability
  one: BF16 is heaviest, FP8 fits alongside another GPU service, NVFP4 is the
  lightest and requires FP4-capable (Blackwell-class) hardware.
- Super is supported only on H100 and RTX PRO 6000, and needs one GPU dedicated
  to the VLM — do not co-locate another GPU service on it. Treat both as hard
  constraints when you are the one choosing the variant.
- **Surface an unsupported-hardware Super request before acting on it.** If Super
  is requested and the detected GPUs are neither H100 nor RTX PRO 6000, or no GPU
  can be dedicated to RT-VLM, stop and tell the user which constraint their system
  fails and what the detected hardware and placement actually are. Then ask
  (`AskUserQuestion`) whether to fall back to the equivalent Nano variant or
  override the constraint anyway. Only proceed with Super after the user overrides
  it knowingly — never assume the request itself is the override, and never
  silently downgrade to Nano either.
- The BF16 tag is `bf16-final` for both Nano and Super. Copy tags verbatim from
  the table rather than deriving them.
- `RTVI_VLM_MODEL_TO_USE=cosmos-reason3` for all six rows, and the served
  endpoint stays `http://rtvi-vlm:8000`; neither changes with the variant.

## Singleton and variant convergence

RT-VLM is a singleton owner: one instance, one checkpoint, and one
variant/placement knob-set per build. When capabilities bring different
integrated Cosmos3 Nano variants, resolve the placement first, then converge on
one variant:

- a dedicated GPU selects the heavier BF16 variant;
- co-residence with another GPU service selects the lighter FP8 variant.

Resolve the variant/placement knobs as one set:
`RTVI_VLM_MODEL_PATH`, `VLM_NAME`, `RTVI_VLLM_GPU_MEMORY_UTILIZATION`,
`RTVI_VLM_MAX_MODEL_LEN`, and `RT_VLM_DEVICE_ID`. Take the checkpoint and model
name from the profile that ships the resolved variant; resolve maximum model
length, device ID, and utilization together for the selected hardware and
placement. Keep `VLM_NAME` aligned with the model id advertised by
`RTVI_VLM_MODEL_PATH` — one row of
[Available integrated model variants](#available-integrated-model-variants) —
and do not combine values from different variants.

Consumer wiring is not part of that set.
`RTVI_VLM_MESSAGE_BUS`, `RTVI_VLM_MESSAGE_BUS_TOPIC` (generated captions),
`RTVI_VLM_KAFKA_INCIDENT_TOPIC` (verification incidents), and verifier config
mounts follow the consuming capability and operating mode, never the profile that
supplied the variant. Realtime VLM alerting (`2d_vlm`) must set
`RTVI_VLM_MESSAGE_BUS=kafka`, `RTVI_VLM_KAFKA_ENABLED=true`, and
`RTVI_VLM_KAFKA_INCIDENT_TOPIC=mdx-vlm-incidents`; the explicit incident topic
avoids the legacy VSS Compose fallback. CV verification (`2d_cv`) sets
`RTVI_VLM_MESSAGE_BUS=` and `RTVI_VLM_KAFKA_ENABLED=false` — verified incidents
reach Elasticsearch through the alert bridge, not RT-VLM's generated-message
path. For the
integrated path, `RTVI_VLM_MODEL_TO_USE=cosmos-reason3` and
the `http://rtvi-vlm:8000` endpoint (a consumer's `VLM_BASE_URL`) are invariant
across BF16 and FP8; a consumer owns that URL but never inherits it from the
variant profile.

### One deployment, three uses

The single `rtvi-vlm` deployment covers three uses, only one of which makes it a
webhook receiver that runs the model ([`vios.md`](vios.md)):

- **VLM tagging** — stream-driven, and a receiver **only when the request asks
  for tag search**; deploying RT-VLM never implies it. The `camera_streaming`
  receiver carries the controlled JSON-tag prompt in `user_defined_metadata`
  (`response_format_type` `json_object`, `temperature` 0, 5s chunks), and RT-VLM
  starts inference on admission of any `stream/add` bearing a prompt, so no
  caller drives it. The output feeds BM25 tag search: RT-VLM publishes to its
  existing `mdx-vlm-captions` topic, the existing LVS Logstash pipeline writes
  each chunk to `default_<streamId>`, and the read side
  (`vss_core.search_core` `TagSearch`/fusion, exposed via `vss search tag` /
  `fusion`) queries it. RT-VLM, Kafka, Logstash, and Elasticsearch are unchanged
  by design; see
  [`docs/designs/vlm-tagging-search.md`](../../../../docs/designs/vlm-tagging-search.md).
- **Dense captioning** — the same leg with a free-form prompt for
  captions/incidents. No shipped notification config carries a dense-captioning
  receiver and a projection cannot add one, so it is driven by
  `POST /v1/generate_captions` per the dense-captioning deployment skill.
- **Critique and visual Q&A** — per request, not per stream:
  `POST /v1/chat/completions` against the deployed service, reached through the
  ingress route `/rtvi-vlm`. It needs RT-VLM deployed and routed, adds **no**
  receiver, and needs no agent tier.

A config can also make RT-VLM a receiver that does not run the model: a
`stream/add` receiver carrying no prompt admits each new stream and starts
nothing, answering `status: "added"`, `inference: false`. That is a receiver
under the same test — it acts on every new stream unasked — but it yields no
captions, so never count it as tagging
([`../profiles/lvs.md`](../profiles/lvs.md) ships the one example).

### Tearing down a hand-driven captioning session

A build never enables tagging and always-on alerting together
([`vios.md`](vios.md)), so a shared RT-VLM stream arises only where a caller
drives `POST /v1/generate_captions` itself. Such a caller is **not**
lifecycle-independent of other subscribers on that stream:
RT-VLM only isolates a subscriber's teardown when `DELETE
/v1/generate_captions/{stream_id}` carries that subscriber's `request_id`. That
`request_id` is **not** caller-selected: RT-VLM generates a UUID for every
`generate_captions` request and returns it as the top-level `id` in the JSON
response (VOD) or in each SSE `data:` event (live). The caller must:

1. parse the returned `id` from the admission response (VOD) or the first SSE
   `data:` event (live) — **do not** synthesize one;
2. persist it alongside its own session (e.g. keyed by the VIOS `sensorId`);
3. tear down with `DELETE /v1/generate_captions/{stream_id}?request_id={returned_id}`
   (then `DELETE /v1/streams/delete/{stream_id}`), leaving any other
   subscriber's stream intact.

Streaming-admission caveat: the server does not emit an immediate ID-only event
before captions begin, so on the live path the `id` is captured from the first
real SSE `data:` event. A caller that issues `DELETE` without the returned
`request_id` tears down every subscriber on the stream.

## Configuration knobs

| Environment variable | Use |
|---|---|
| `VSS_RT_VLM_TAG`, `RTVI_VLM_PORT`, `RT_VLM_DEVICE_ID` | Select image, host port, and GPU. |
| `RTVI_VLM_MODEL_TO_USE`, `RTVI_VLM_MODEL_PATH`, `VLM_NAME` | Select an integrated model and its advertised id (see [Available integrated model variants](#available-integrated-model-variants)). |
| `RTVI_VLM_ENDPOINT`, `RTVI_VLM_API_KEY`, `VLM_BASE_URL` | Configure an OpenAI-compatible backend. |
| `RTVI_VLLM_GPU_MEMORY_UTILIZATION`, `RTVI_VLM_MAX_MODEL_LEN`, `RTVI_VLLM_MAX_NUM_SEQS`, `RTVI_VLLM_MAX_NUM_BATCHED_TOKENS` | Bound vLLM memory and concurrency. |
| `RTVI_VLM_DEFAULT_NUM_FRAMES_PER_SECOND_OR_FIXED_FRAMES_CHUNK`, `RTVI_VLM_BATCH_SIZE` | Tune frame sampling and batching. |
| `VIA_EVS_SESSION`, `VLM_VIDEO_PRUNING_RATE`, `VLLM_EVS_SIMILARITY_THRESHOLD` | Enable and tune EVS++ video-session pruning. Set `VIA_EVS_SESSION=true`, choose a pruning rate greater than `0` and less than `1`, and tune the similarity threshold for the input streams. Keep all three unset to preserve the Foundation default (EVS++ disabled). |
| `VIA_EVS_TOKEN_BUDGET`, `VIA_EVS_MAX_SESSIONS` | Optionally tune EVS++ token packing and concurrent session capacity. |
| `RTVI_VLM_MESSAGE_BUS`, `RTVI_VLM_MESSAGE_BUS_TOPIC`, `RTVI_VLM_KAFKA_BOOTSTRAP_SERVERS` | Configure generated-message publication. Current defaults are `kafka`, `mdx-vlm-captions`, and `kafka:29092`. VSS Compose still forwards legacy `RTVI_VLM_KAFKA_ENABLED`, but RT-VLM 26.08.2 ignores `KAFKA_ENABLED`; keep it aligned with `RTVI_VLM_MESSAGE_BUS` during the transition. `RTVI_VLM_KAFKA_TOPIC` is obsolete. |
| `RTVI_VLM_KAFKA_INCIDENT_TOPIC`, `RTVI_VLM_ERROR_BUS`, `RTVI_VLM_ERROR_MESSAGE_TOPIC` | Configure incident and error publication. RT-VLM 26.08.2 defaults to `mdx-vlm-incidents` and `mdx-vlm-errors`. VSS Compose still falls back to `vision-llm-events-incidents` for incidents, so set `mdx-vlm-incidents` explicitly until that fallback is migrated. The error-topic fallback is `mdx-vlm-errors`. |
| `VLM_MODEL_SUPPORTS_AUDIO`, `VLM_TRUST_REMOTE_CODE`, `HF_TOKEN` | Enable supported audio or gated/custom HF models. |
| `INSTALL_PROPRIETARY_CODECS`, `FORCE_SW_AV1_DECODER` | Select runtime codec behavior. |

## Sources

- `deploy/docker/services/rtvi/rtvi-vlm/rtvi-vlm-docker-compose.yml`
- `skills/vss-build-vision-ai/references/composition.md`
- `skills/deployment/vss-deploy-dense-captioning/references/deploy-rt-vlm-service.md`
- `skills/deployment/vss-deploy-dense-captioning/references/integrate-rt-vlm.md`
