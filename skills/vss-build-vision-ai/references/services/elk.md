# ELK and Broker Capability Owner

## Capabilities and service keys

| Capability | Canonical service profile keys |
|---|---|
| Elasticsearch storage and initialization | `elasticsearch`, `elasticsearch-init-container` |
| Kafka broker and topics | `kafka`, `kafka-topic-init-container` |
| Redis | `redis` |
| Kibana and profile dashboards | `kibana`, `kibana-init-container-alerts`, `kibana-init-container-lvs`, `kibana-init-container-search` |
| Log ingestion and broker readiness | `logstash`, `broker-health-check` |

## Required peers

- Use `elasticsearch-init-container` with `elasticsearch`.
- **Probe these at the right paths — the wrong one reads as a dead service.**
  - **Kibana** serves under the `/kibana` base path, so bare `/` and
    `/api/status` return 404 while the container reports healthy. Use
    `/kibana/api/status`.
  - **Elasticsearch** on a single node reports `yellow`, never `green` —
    replicas stay unassigned with one node. `curl -sf` passes on yellow, so
    treat it as healthy.
- Use `kafka-topic-init-container` and `broker-health-check` with Kafka-backed
  capability owners.
- `logstash` is the **sole** bridge from Kafka topics to Elasticsearch. No other
  selected service writes Kafka events into ES indices. A build that publishes to
  Kafka and stores in Elasticsearch must include `logstash`; omitting it leaves
  the requested ES storage permanently empty.
- `logstash` requires the broker and the profile's selected `STREAM_TYPE`.
- `mdx-vlm-captions` is already in `kafka-topic-init-container`'s default
  `KAFKA_TOPICS`; do not override the topic list merely to add it again.
- When the selected Foundation ships `kibana` and a `kibana-init-container-*`
  key, retain both in any delta that stores data in Elasticsearch — they are the
  browse surface for that data and are not pruned by forward closure. They are
  **not** part of the Agent/UI tier, so a headless build (no agent/UI) still
  retains them; do not drop them as "UI". Do not add `kibana` to a Foundation
  that does not ship it.
- The initializer **key** is always the selected Foundation's `kibana-init-container-*`
  — never swap to another Foundation's key. Only the mounted **bundle** varies, chosen
  from the build's active ES write paths (not the Foundation name), to seed one data view
  per Elasticsearch index family this build writes:
  - Families of a **single** capability → mount that capability's bundle.
  - Families spanning **more than one** → patch the Foundation's initializer to bind-mount
    the shipped merged **union** bundle (Sources) **over the exact image-baked path its
    import script already reads**, not a new path; a mount at any other target is a no-op.
  Never add a second initializer or mount two bundles: the shared `mdx-raw-*`/
  `mdx-behavior-*` views would duplicate and collide on the default-view singleton.
- `redis` may be used without the full ELK/Kafka set when it is only a cache.

## Configuration knobs

| Environment variable | Use |
|---|---|
| `ELASTICSEARCH_HOST_PORT`, `ELASTICSEARCH_URL`, `ELASTICSEARCH_CONNECTION_MAX_ATTEMPTS` | Publish and initialize Elasticsearch. |
| `ELASTICSEARCH_ILM_MIN_AGE` | Set retention policy age. |
| `ELASTICSEARCH_ENABLE_EMBEDDINGS`, `ELASTICSEARCH_RTVI_CV_EMBEDDINGS_DIM`, `ELASTICSEARCH_VISION_LLM_EMBEDDINGS_DIM` | Configure indexed vector fields. |
| `KAFKA_HOST_PORT`, `KAFKA_BOOTSTRAP_HOST`, `KAFKA_INTERNAL_PORT` | Configure broker access. |
| `KAFKA_TOPICS`, `DEFAULT_PARTITIONS`, `DEFAULT_RETENTION_MS` | Configure topic initialization when overridden by a service definition. |
| `REDIS_HOST_PORT`, `KIBANA_HOST_PORT`, `PHOENIX_HOST_PORT` | Change shared host bindings. |
| `STREAM_TYPE`, `BROKER_BOOTSTRAP_HOST` | Select Kafka or Redis ingestion where supported. |

## Sources

- `deploy/docker/services/infra/compose.yml`
- `deploy/docker/services/infra/elk/`
- `deploy/docker/services/infra/elk/kibana/configs/search-and-alerts-kibana-objects.ndjson`
- `deploy/docker/developer-profiles/dev-profile-alerts/compose.yml`
- `deploy/docker/developer-profiles/dev-profile-lvs/compose.yml`
- `deploy/docker/developer-profiles/dev-profile-search/compose.yml`
