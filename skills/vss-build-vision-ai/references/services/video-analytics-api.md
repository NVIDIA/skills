# Video Analytics API Capability Owner

## Capabilities and service keys

| Capability | Canonical service profile keys |
|---|---|
| REST query and configuration API | `vss-video-analytics-api` |
| Host-side read-only incidents and analytics (`vss analytics`) | `vss-video-analytics-api` |
| Warehouse calibration import | `import-calibration-output-container-<mode>` |

`vss-video-analytics-api` is a shared Compose service: every Foundation uses the
same profile key and the same `vss-video-analytics-api` container. Include it
once when the requested capability needs its REST surface; never create a
Foundation-specific alias or a second API instance.

Read-only analytics requested through NemoClaw is owned here. Its skill declares
`vss-requires: "analytics"`; `vss configure` maps that command capability to
the `/video-analytics-api` route. The forward closure includes this API and its
Elasticsearch peers, and excludes `vss-va-mcp` and `vss-agent` unless the user
separately and explicitly requests those legacy/agent surfaces.

For a read-only analytics-only composition seeded from a Foundation that
contains those legacy keys, the deterministic delta is:

```text
included: vss-video-analytics-api
excluded: vss-va-mcp
excluded: vss-agent
```

Another explicitly selected capability may independently re-add a key it owns.

## Required peers

- Elasticsearch is required for the API's Elasticsearch-backed operations.
  Select `elasticsearch` and `elasticsearch-init-container`, including the
  `insertion-timestamp-pipeline`; a reachable Elasticsearch port alone does not
  establish that those operations are ready.
- With `STREAM_TYPE=kafka` and non-empty `kafka.brokers`, it also waits for
  `mdx-notification`, `mdx-amr`, and at least one `mdx-rtls*` topic. Include
  `kafka`, `kafka-topic-init-container`, and `broker-health-check` when the
  requested API features need dynamic configuration/calibration or RTLS/AMR.
  To intentionally omit Kafka startup work, use `STREAM_TYPE=redis`, or set
  `kafka.brokers` to `[]` or `null`; `redis` here skips Kafka work and does not
  configure a Redis client.
- The data-query routes need matching Elasticsearch indices and upstream
  producers. `/livez` returning `{ "isAlive": true }` is only liveness; empty
  query results mean no matching analytics data has been indexed, not a deployment
  failure.
- Dynamic configuration publishes on `mdx-notification` under the
  `behavior-analytics-config` key and needs Behavior Analytics to consume and
  acknowledge it. Dynamic calibration publishes on the same topic with the
  `calibration` key. These are API features, not a reason to add a second
  Behavior Analytics service.
- The default bind mount for `/web-api-app/files` needs
  `VSS_DATA_DIR/data_log/vss_video_analytics_api`. Retain it for calibration-image
  or other file upload endpoints; it may be removed when those files can be
  ephemeral.
- Warehouse extended variants pair the shared API key with
  `import-calibration-output-container-<mode>`; minimal variants do not. The
  import container posts to the API and is not a general dependency of the API.

## API and ingress surface

- The service publishes `${VIDEO_ANALYTICS_API_HOST_PORT:-8081}:8081`.
  `GET /livez` returns `{ "isAlive": true }` once the API registers its route.
  Registration follows the Elasticsearch ingest-pipeline check and, with configured
  Kafka, its topic check; it does not attest that upstream analytics data or the
  Elasticsearch indices queried by routes are populated.
- When HAProxy ingress is selected, expose the same service at
  `/video-analytics-api/...`; ingress strips that prefix before forwarding. Do
  not construct a distinct API URL for a profile.
- Its REST routes cover calibration/configuration, sensors, behavior, alerts,
  incidents, events, frames, metrics, tracking, and clustering. Route presence
  does not imply data availability: most query routes are Elasticsearch-backed.

## Configuration knobs

| Environment variable | Use |
|---|---|
| `VSS_VIDEO_ANALYTICS_API_IMAGE`, `VSS_VIDEO_ANALYTICS_API_TAG` | Select the API image. |
| `VIDEO_ANALYTICS_API_HOST_PORT` | Publish the API's host port (default `8081`). |
| `STREAM_TYPE` | Select `kafka` (default) or `redis`; any other value makes startup fail. |
| `VSS_APPS_DIR` | Resolve the service-shipped bootstrap JSON mount. |
| `VSS_DATA_DIR` | Resolve the optional persistent files mount. |

The API loads JSON through `--config`; the stock Compose command mounts
`services/analytics/video-analytics-api/configs/vss-video-analytics-api-config.json`.
That file controls `server.port`, Elasticsearch (`node`, index prefix/pattern,
retries), Kafka brokers/retries, and application settings such as body-size and
config-ack timeouts. Keep `server.configs[].value` values as strings. The
image-baked config disables Kafka with an empty broker list; the service-shipped
config enables localhost Kafka. Replace the mounted JSON when the requested
deployment needs a different endpoint or startup behavior rather than trying to
express those settings as Compose environment deltas.

## Sources

- `deploy/docker/services/analytics/video-analytics-api/compose.yml`
- `deploy/docker/services/analytics/video-analytics-api/configs/vss-video-analytics-api-config.json`
- `services/analytics/video-analytics-api/configs/default-configs/config.json`
- `deploy/docker/services/infra/compose.yml`
- `deploy/docker/services/infra/haproxy/haproxy.cfg.template`
- `skills/deployment/vss-setup-video-analytics-api/references/configuration.md`
- `skills/deployment/vss-setup-video-analytics-api/references/deploy-video-analytics-api-service.md`
