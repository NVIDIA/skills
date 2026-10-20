# Search Developer Profile

## Capabilities and routing cues

- Video ingest, RT-CV detection/tracking, RT-Embed video/text embeddings,
  Elasticsearch retrieval, RT-VLM critique / visual follow-up Q&A served to the
  agent and the retrieval CLI alike, and **VLM tagging** (controlled JSON-tag
  `generate_captions` → `mdx-vlm-captions` → Logstash → `default_<streamId>`,
  queried by `vss search tag`/`fusion`).
- Choose for natural-language video search or combined ingestion + detection +
  embedding requests.
- See `services/rt-cv.md` for detector model-family → Foundation mapping.

## Profile Service Set

Authoritative source:
`deploy/docker/developer-profiles/dev-profile-search/overrides.env`.

```text
kibana-init-container-search,vss-search-analytics-2d-fusion,vss-video-analytics-api,nvstreamer-2d-fusion,perception-2d-fusion,vss-agent,phoenix,elasticsearch,elasticsearch-init-container,kafka,kafka-topic-init-container,redis,kibana,logstash,broker-health-check,vss-haproxy-ingress,rtvi-embed,vss-ui,centralizedb,vst-ingress,sensor-ms,streamprocessing-ms,rtvi-vlm,llm_${LLM_MODE}_${LLM_NAME_SLUG}
```

Docker search runs VIOS in **direct** mode (`VST_USE_SDRC=false`, no SDRC
compose tokens). Helm search keeps SDRC enabled for live multi-worker scale.

## Capability owners present

| Owner | Service profile keys |
| --- | --- |
| Search | `vss-search-analytics-2d-fusion` |
| RT-CV | `perception-2d-fusion` |
| RT-Embed | `rtvi-embed` |
| RT-VLM | `rtvi-vlm` |
| ELK | `elasticsearch`, `elasticsearch-init-container`, `kafka`, `kafka-topic-init-container`, `redis`, `kibana`, `logstash`, `broker-health-check`, `kibana-init-container-search` |
| VIOS | `nvstreamer-2d-fusion`, `centralizedb`, `vst-ingress`, `sensor-ms`, `streamprocessing-ms` |
| Agent | `vss-agent`, `vss-ui`, `phoenix` |
| Ingress | `vss-haproxy-ingress` |
| LLM NIM | `llm_${LLM_MODE}_${LLM_NAME_SLUG}` |

## Source registration and fan-out

This profile pins `VST_NOTIFICATION_CONFIG_PATH` at a webhooks-enabled
`notification_config.json` — the pin is the profile's own, not a property of
Agent presence. Agent-backed and headless deployments both register one
source with `vss vios add`; configured receivers perform the whole fan-out
without an Agent `/complete` call. The caller calls no consumer
(`vss-manage-video-io-storage` `provision-vios-source.md`).

Helm selects its notification policy through `notificationConfigFile`. Inspect
the mounted policy on either deployment; unreadable policy means unknown
receiver state, not unavailable indexing.

That config ships RT-CV, RT-Embed and RT-VLM tagging enabled, plus the
Elasticsearch teardown cleanups, but the set is not inherited by default: it must
equal the build's own stream-driven consumer set, so a build never asked for
tagging projects those items off ([`../services/vios.md`](../services/vios.md)).
Tagging is what makes `vss search tag` and `fusion` return hits, so dropping it
drops that read path. Dense captioning is a separate, optional RT-VLM leg
governed by the Alert-Bridge carve-out, not a search requirement.

## Profile-specific environment knobs

| Knob | Purpose |
| --- | --- |
| `VST_USE_SDRC`, `VST_NGINX_MODE`, `STREAM_PROCESSOR_MODULE_ENDPOINT` | Pin VIOS to direct routing (`false` / `vst` / `http://vss-vios-streamprocessing:30001`). All three move together — see `services/vios.md`. A build that reintroduces SDRC must flip all three and re-add the SDRC compose tokens plus `SDR_CONTROLLER_CONFIG_PATH` / `SDRC_*_HOST_PORT`. |
| `VST_NOTIFICATION_CONFIG_PATH` | Selects the VIOS webhook fan-out config; this profile points it at its own `notification_config.json`. A build whose stream-driven consumer set differs from what that file enables points this at a projection of it instead — see [`../services/vios.md`](../services/vios.md). |
| `VST_ENABLE_NOTIFICATION` | Independent publish toggle for `vst.event` (Redis/Kafka) — not a routing knob. VIOS composes default it to `${VST_USE_SDRC:-false}`, so it follows the routing mode unless pinned; this profile pins it `false` explicitly since nothing here consumes `vst.event`. See `services/vios.md`. |
| `RT_CV_DEVICE_ID`, `RTVI_CV_HOST_PORT`, `DS_MODEL_FAMILY` | Configure the perception pipeline. |
| `VISION_ENCODER_MODEL`, `VISION_ENCODER_VERSION` | Select the vision encoder NGC artifact downloaded by ds-start phase 0; the checked-in RT-CV config uses the fixed RT-DETR warehouse artifact. |
| `RT_EMBED_DEVICE_ID`, `RTVI_EMBED_PORT`, `MODEL_PATH`, `HF_TOKEN` | Place and configure RT-Embed. |
| `MESSAGE_BUS`, `MESSAGE_BUS_TOPIC`, `ERROR_BUS` | RT-Embed's output and error buses, set to `kafka`/`mdx-embed`/`kafka` by this Foundation so the `mdx-embed` -> `mdx-embed-filtered` path indexes. Inherit them; see `../services/rt-embed.md` for builds on other Foundations. |
| `VLM_NAME`, `VLM_BASE_URL`, `VLM_MODEL_TYPE`, `RTVI_VLM_*` | Wire RT-VLM for result critique and visual follow-up Q&A. Include these whenever the build ships `rtvi-vlm`, with or without a `vss-agent` tier: the retrieval CLI invokes it too. Critique is a per-request option (`use_critic`, default on), not a build-time flag. |
| `COSMOS_EMBED_ENDPOINT`, `ELASTIC_SEARCH_ENDPOINT`, `ELASTIC_SEARCH_INDEX` | Wire the agent to embedding and retrieval services. |
| `ELASTICSEARCH_ENABLE_EMBEDDINGS`, `ELASTICSEARCH_RTVI_CV_EMBEDDINGS_DIM`, `ELASTICSEARCH_VISION_LLM_EMBEDDINGS_DIM` | Configure indexed vectors. |
| `LLM_DEVICE_ID`, `RT_VLM_DEVICE_ID`, `RESERVED_DEVICE_IDS`, `FIXED_SHARED_DEVICE_IDS` | Preserve the intended multi-GPU layout. |

## Stock readiness checks

```bash
curl -sf "http://${HOST_IP}:8000/health"
curl -sf "http://${HOST_IP}:8017/v1/ready"
curl -sf "http://${HOST_IP}:8018/v1/health/ready"
curl -sf "http://${HOST_IP}:${RTVI_CV_HOST_PORT:-9000}/ready"
curl -sf "http://${HOST_IP}:9200/_cluster/health"
curl -sf "http://${HOST_IP}:3000/"
```

Whenever the build ships `rtvi-vlm`, also probe RT-VLM's `/v1/models` endpoint.
This applies to headless builds too: the retrieval CLI probes the same endpoint
before verifying results. Skip the check only when the build omits `rtvi-vlm`.

## Sources

- `deploy/docker/developer-profiles/dev-profile-search/.env`
- `deploy/docker/developer-profiles/dev-profile-search/overrides.env`
- `deploy/docker/developer-profiles/dev-profile-search/compose.yml`
- `deploy/docker/developer-profiles/dev-profile-search/video-analytics-2d-app/compose.yml`
- `deploy/docker/services/rtvi/rtvi-cv/ds-start.sh`
- `deploy/docker/services/rtvi/rtvi-embed/rtvi-embed-docker-compose.yml`
- `deploy/docker/services/rtvi/rtvi-cv/compose.yaml`
- `deploy/docker/services/rtvi/rtvi-vlm/rtvi-vlm-docker-compose.yml`
- `skills/deployment/vss-deploy-video-embedding/references/environment.md`
- `skills/deployment/vss-deploy-detection-tracking-2d/references/environment.md`
