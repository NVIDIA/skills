# Behavior Analytics Capability Owner

## Capabilities and service keys

| Capability | Canonical service profile keys |
|---|---|
| Alerts behavior rules | `vss-behavior-analytics-alerts` |
| Search analytics | `vss-search-analytics-2d-fusion` |
| Warehouse behavior rules (ROI, tripwire, proximity) | `vss-behavior-analytics-<mode>` |

## Required peers

- Requires the matching profile-owned JSON config mounted by its extending
  service.
- On `warehouse` the key is mode-suffixed (`vss-behavior-analytics-2d`, `-3d`)
  and consumes the mode's perception output: `mdx-raw` for `2d`, `mdx-bev` for
  `3d`. It requires the blueprint configurator for its mounted config — see
  [`configurator.md`](configurator.md).
- **`vss-behavior-analytics` publishes no HTTP listener** and declares no ports;
  it is a broker consumer. The HAProxy `/behavior-analytics` route is defined but
  its backend never passes health check, so it always returns 503. Read behaviors
  from the `mdx-behavior` topic or the `mdx-behavior-*` Elasticsearch indices —
  never by probing that route, and never treat its 503 as a deployment fault.
- Kafka-backed configs require `kafka`, `kafka-topic-init-container`, and
  `broker-health-check`.
- Alerts mode consumes RT-CV events; Search mode consumes the Search perception
  pipeline. Do not activate both variants for a single capability.
- The object-class filter keys in the mounted config
  (`fovCountViolationIncidentObjectType`, `stateManagementFilter`) must match the
  class-label taxonomy the resolved RT-CV detector emits — label set and casing.
  In a combined build these follow the single converged detector, not the value
  a source profile's config happened to ship.
- This service's operating mode — the `numWorkersFor*` gates, `playbackLoop`,
  class scope — lives in the mounted analytics JSON, not env or `COMPOSE_PROFILES`,
  so env-delta resolution cannot touch it. A build that adds a capability or
  ingestion mode the source config did not assume **must replace** that JSON; env
  reconciliation alone silently inherits the source mode (Search ships
  `numWorkersForIncidentGeneration=0`, so it generates no incidents and yields no
  alerts).
- To serve more than one capability at once, run a single combined instance
  rather than two — under the selected Foundation's key for the one
  `vss-behavior-analytics` container, never both — mounting the shipped joint config
  `<repo-root>/services/analytics/behavior-analytics/configs/search_and_alerts_config.json`.
  This file is outside `VSS_APPS_DIR`, which points to
  `<repo-root>/deploy/docker`. Bind the shipped file directly, without copying
  it into `_builds/`, through
  `patches/vss-search-analytics-2d-fusion.yml` as
  `<repo-root>/services/analytics/behavior-analytics/configs/search_and_alerts_config.json:/resources/vss-search-analytics-config.json:ro`.
  No developer profile mounts it by default. Its `numWorkersFor*` knobs gate
  each processor independently, so enable one only for a requested capability:
  incident generation for detection-rule alerts, behavior creation for search
  analytics, embed filtering for search embeddings — leave the rest at zero. In
  particular, alerts that do not derive from this owner (see the Alerts owner)
  leave incident generation off.
- Verify the mounted config matches the request: incident workers non-zero only
  when alerts are requested, and multi-capability builds on the joint config.
  Fork the config only to change a knob with real runtime effect (`numWorkersFor*`,
  class scope) — never to alter a field the service does not read.
- A combined instance writes more than one Elasticsearch index family, so its
  Kibana initializer must seed all of them — see `elk.md` (Kibana seeding).

## Configuration knobs

| Environment variable | Use |
|---|---|
| `VSS_BEHAVIOR_ANALYTICS_IMAGE`, `VSS_BEHAVIOR_ANALYTICS_TAG` | Select the Behavior Analytics image. |
| `VSS_APPS_DIR` | Resolve profile-owned mounted JSON configs; it does not contain the repo-root combined config. |
| `STREAM_TYPE` | Select the checked-in Kafka or Redis Search config where supported. |

Incident rules, broker addresses, thresholds, and sensor settings are fields in
the mounted JSON config, not Compose environment knobs. A requested rule change
there is a config change outside this env-only contract.

## Sources

- `deploy/docker/services/analytics/behavior-analytics/compose.yml`
- `deploy/docker/developer-profiles/dev-profile-alerts/compose.yml`
- `deploy/docker/developer-profiles/dev-profile-search/video-analytics-2d-app/compose.yml`
- `services/analytics/behavior-analytics/configs/search_and_alerts_config.json`
- `skills/deployment/vss-setup-behavior-analytics/references/configuration.md`
- `skills/deployment/vss-setup-behavior-analytics/references/deploy-behavior-analytics-service.md`
