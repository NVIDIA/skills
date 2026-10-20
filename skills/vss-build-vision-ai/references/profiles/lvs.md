# LVS Developer Profile

## Capabilities and routing cues

- Long-video summarization, timestamped highlights, structured summaries, and
  report generation.
- Uses RT-VLM for VLM serving and ELK/Kafka for event flow.
- Choose for video summarization or LVS requests.
- RTX PRO 4500 Blackwell (`RTXPRO4500BW`) is not a valid hardware profile here;
  route that SKU to Alerts with a remote LLM.

## Profile Service Set

Authoritative source:
`deploy/docker/developer-profiles/dev-profile-lvs/overrides.env`.

```text
kibana-init-container-lvs,nvstreamer-lvs,vss-agent,phoenix,elasticsearch,elasticsearch-init-container,kafka,kafka-topic-init-container,redis,kibana,logstash,broker-health-check,vss-haproxy-ingress,init-dirs,render-config,wdm-env-from-config,wait-for-redis,sdr-controller,rtvi-vlm,vss-ui,lvs-server,centralizedb,vst-ingress,sensor-ms,streamprocessing-ms,llm_${LLM_MODE}_${LLM_NAME_SLUG}
```

## Capability owners present

| Owner | Service profile keys |
|---|---|
| LVS | `lvs-server` |
| RT-VLM | `rtvi-vlm` |
| ELK | `elasticsearch`, `elasticsearch-init-container`, `kafka`, `kafka-topic-init-container`, `redis`, `kibana`, `logstash`, `broker-health-check`, `kibana-init-container-lvs` |
| VIOS | `nvstreamer-lvs`, `init-dirs`, `render-config`, `wdm-env-from-config`, `wait-for-redis`, `sdr-controller`, `centralizedb`, `vst-ingress`, `sensor-ms`, `streamprocessing-ms` |
| Agent | `vss-agent`, `vss-ui`, `phoenix` |
| Ingress | `vss-haproxy-ingress` |
| LLM NIM | `llm_${LLM_MODE}_${LLM_NAME_SLUG}` |

## Headless fan-out (no-agent builds)

This profile pins `VST_NOTIFICATION_CONFIG_PATH` at a webhooks-enabled
`notification_config.json`, so a registered VIOS source reaches RT-VLM with no
caller: `stream/add` on `camera_add` and `camera_streaming`, `stream/remove` on
`camera_remove`. Registering the source is the whole caller-side job
(`vss-manage-video-io-storage` `provision-vios-source.md`).

The adds carry no prompt, so RT-VLM admits the stream and starts no inference —
summaries still come from a caller driving `/v1/summarize`, and no captions or
tags exist before that ([`../services/rt-vlm.md`](../services/rt-vlm.md)). The file
carries a single consumer and no item ids, which constrains what a delta can
project from it ([`../services/vios.md`](../services/vios.md)).

## Profile-specific environment knobs

| Knob | Purpose |
|---|---|
| `VST_NOTIFICATION_CONFIG_PATH` | Selects the VIOS webhook fan-out config; this profile points it at its own `notification_config.json`. A build whose stream-driven consumer set differs from what that file enables points this elsewhere — see [`../services/vios.md`](../services/vios.md). |
| `LVS_TAG`, `BACKEND_HOST_PORT`, `LVS_MCP_HOST_PORT`, `LVS_ENABLE_MCP` | Select the image and exposed LVS APIs. |
| `LVS_DATABASE_BACKEND`, `LVS_EMB_*` | Configure the supported summary database and optional text-embedding endpoint. |
| `KAFKA_ENABLED`, `KAFKA_STRUCTURED_SUMMARY_TOPIC`, `LVS_ENABLE_LLM_MERGING` | Configure summary event flow. |
| `ENABLE_AUDIO` | Enable audio-aware summarization when the selected VLM supports it. |
| `VLM_NAME`, `VLM_BASE_URL`, `RTVI_VLM_MODEL_PATH`, `RTVI_VLM_MODEL_TO_USE` | Keep the LVS-requested model id aligned with RT-VLM. |
| `RT_VLM_DEVICE_ID`, `RTVI_VLLM_GPU_MEMORY_UTILIZATION`, `RTVI_VLM_MAX_MODEL_LEN` | Place and size RT-VLM. |

## Stock readiness checks

```bash
curl -sf "http://${HOST_IP}:8000/health"
curl -sf "http://${HOST_IP}:38111/v1/ready"
curl -sf "http://${HOST_IP}:8018/v1/health/ready"
curl -sf "http://${HOST_IP}:${LLM_PORT:-30081}/v1/health/ready"
```

Skip a local model probe only when that model is explicitly remote, and probe
the selected remote `/v1/models` endpoint instead.

## LVS REST API surface

Surface this output contract in the architecture preview and final response for
summarization builds; it is authoritatively defined in
`skills/operations/vss-summarize-video/references/integrate-lvs-service.md`.

| Interface | Endpoint | Contract |
|---|---|---|
| Readiness | `GET /v1/ready` | Gate summarize calls on a ready LVS server. |
| Model discovery | `GET /models` | Returns the id sent as the request `model`. |
| Summarize (output path) | `POST /v1/summarize` | VIOS-uploaded or recorder-stored media enters via a VIOS-provided `url` (HTTP(S)/S3/VIOS-retrievable video URL) or `id` (file/clip id known to LVS), alongside required `model`, `scenario`, and `events`. |

The summary is returned in `choices[0].message.content`. `/v1/stream_summarize`
and `/v1/generate_captions` are not the output path for stored/uploaded media.

## Sources

- `deploy/docker/developer-profiles/dev-profile-lvs/.env`
- `deploy/docker/developer-profiles/dev-profile-lvs/overrides.env`
- `deploy/docker/developer-profiles/dev-profile-lvs/compose.yml`
- `deploy/docker/developer-profiles/dev-profile-lvs/vios/configs/notification_config.json`
- `deploy/docker/services/video-summarization/compose.yml`
- `deploy/docker/services/rtvi/rtvi-vlm/rtvi-vlm-docker-compose.yml`
- `skills/operations/vss-summarize-video/references/video-summarization-environment-variables.md`
- `skills/operations/vss-summarize-video/references/integrate-lvs-service.md`
