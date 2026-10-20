# VIOS Capability Owner

## Capabilities and service keys

| Capability | Canonical service profile keys |
|---|---|
| Video database and ingest | `centralizedb`, `vst-ingress` |
| Sensor and stream management | `sensor-ms`, `streamprocessing-ms`, `sensor-ms-<mode>`, `streamprocessing-ms-<mode>` |
| Profile stream sources | `nvstreamer-alerts`, `nvstreamer-lvs`, `nvstreamer-2d-fusion`, `nvstreamer-2d`, `nvstreamer-3d` |
| SDR controller and config rendering | `init-dirs`, `render-config`, `wdm-env-from-config`, `wait-for-redis`, `sdr-controller` |
| WebRTC relay for VST playback | `turnserver`, `turnserver-init` |

## Required peers

- `centralizedb`, `vst-ingress`, `sensor-ms`, and `streamprocessing-ms` form the
  normal developer VIOS core.
- SDR-controlled profiles require the full helper sequence shown above and
  `redis`. No capability names these helpers, so only their status as VIOS peers
  keeps them out of the forward-closure prune in
  [`../composition.md`](../composition.md).
- `turnserver` and `turnserver-init` are required wherever VST playback is
  served, alongside `sensor-bp-wait-bp-configurator`, which gates sensor
  registration on the configurator ([`configurator.md`](configurator.md)).
- NvStreamer variants require the matching developer profile's mounted configs
  and, where declared, `broker-health-check`.
- `vios-apt-cache-init` has no `profiles:` gate and is a `depends_on` of
  `streamprocessing-ms-*`; it resolves into every build and cannot be pruned.
- Add only the profile-specific NvStreamer key; do not activate multiple
  variants for one source.

## Configuration knobs

| Environment variable | Use |
|---|---|
| `VSS_APPS_DIR`, `VSS_DATA_DIR`, `VST_CONFIG_PATH` | Resolve checked-in configs and persistent data. |
| `VST_INGRESS_HOST_PORT`, `SENSOR_HTTP_HOST_PORT`, `STREAM_PROCESSOR_HTTP_HOST_PORT` | Publish VIOS APIs. |
| `RTSP_SERVER_HOST_PORT`, `RTSP_SERVER_HOST_PORT_END` | Publish RTSP playback ports. |
| `VST_BASE_URL`, `VST_INTERNAL_URL`, `VST_EXTERNAL_URL`, `VST_MCP_URL` | Configure internal and public routing. |
| `VST_NGINX_MODE` | Select direct or SDRC routing supported by the Foundation. |
| `VST_ENABLE_NOTIFICATION` | Publish VIOS lifecycle events (`camera_add` / `camera_streaming` / `camera_remove`) to Redis `vst.event`. The VIOS composes default it to `${VST_USE_SDRC:-false}`, so it flips with the routing mode unless the build pins it — pin it whenever a build changes `VST_USE_SDRC`. |
| `VST_NOTIFICATION_CONFIG_PATH` | Select the notification config bind-mounted over the fixed `configs/notification_config.json` in both `sensor-ms` and `streamprocessing-ms`. Defaults to the shared, webhooks-disabled `${VSS_APPS_DIR}/services/vios/configs/notification_config.json`. |
| `SDR_CONTROLLER_CONFIG_PATH`, `SDRC_*_HOST_PORT` | Select rendered SDR config and host ports. |
| `NVSTREAMER_HTTP_PORT`, `NVSTREAMER_HTTP_HOST_PORT`, `NVSTREAMER_INSTALL_ADDITIONAL_PACKAGES` | Configure a profile's NvStreamer source. |
| `NUM_SENSORS`, `STREAM_TYPE` | Configure source count and broker type where supported. |

When a build changes `VSS_APPS_DIR` or a public host primitive, put every
selected dependent path and URL in the build `override.env`; Compose does not
re-expand values already read from the Foundation env files.

## Notification config — a converged singleton

The mounted `notification_config.json` **is** the fan-out policy: VIOS posts
sensor lifecycle events (`camera_add`, `camera_streaming`, `camera_remove`) to
the webhook receivers it defines. It is a singleton config on a singleton
service, so every build decides it deliberately, exactly like the Behavior
Analytics joint config — though only a build whose decision differs from the
inherited default writes a file for it.

### Who belongs in it

**A service is a receiver if and only if it must act on every newly registered
stream without being asked.** Detection, embeddings and continuous tagging are
stream-driven: nothing else triggers them, so a stream they never saw is a
stream they never process. A capability invoked per request — a VLM asked to
critique a retrieved clip or answer a question about it — needs the service
deployed and routed, not a receiver; a receiver would start work no one asked
for. Deployed is not the same as receiving.

**The invariant: the enabled receiver set equals the build's stream-driven
consumer set.** Both directions cost real signal. A missing receiver leaves a
consumer unprovisioned, so the stream is never processed. An extra one fails
every delivery, and with no webhook introspection that failure is log-identical
to a genuine one.

### Items and the enable vector

A profile config carrying several stream-driven capabilities splits its fan-out
into items of one capability and one event, keyed `<capability>-<event>`.
Resolving such a config is choosing which ids are `enabled` — one decision per
row, both events together:

| Stream-driven capability | Item ids |
|---|---|
| Object detection and tracking (RT-CV) | `rtvi-cv-camera-streaming`, `rtvi-cv-camera-remove` |
| Chunk and video embeddings (RT-Embed) | `rtvi-embed-camera-streaming`, `rtvi-embed-camera-remove` |
| Continuous VLM tagging (RT-VLM) | `rtvi-vlm-tagging-camera-streaming`, `rtvi-vlm-tagging-camera-remove` |
| Always-on real-time alerting (Alert Bridge) | `alert-bridge-camera-streaming`, `alert-bridge-camera-remove` |

Teardown-only cleanup items follow the Elasticsearch index a build writes rather
than a service; enable one only when the build runs the path that fills it:
`es-raw-camera-remove` (`mdx-raw`, from DeepStream perception),
`es-behavior-camera-remove` (`mdx-behavior`, from behavior analytics),
`es-embed-filtered-camera-remove` (`mdx-embed-filtered`, from the search
analytics fusion of `mdx-embed`), and `es-vlm-tags-camera-remove` (the
per-video `default_<id>` tag indices, from RT-VLM tagging; file sensors only).
Enable the tag cleanup whenever `rtvi-vlm-tagging-camera-streaming` is enabled:
without it a deleted video's tags stay searchable.

### Resolving it

1. Derive the enable vector from the build's stream-driven consumer set.
2. Compare it with the config the build's inherited
   `VST_NOTIFICATION_CONFIG_PATH` already resolves to. **Equal — inherit:**
   change nothing and leave the variable out of the build `override.env`.
3. **Different — project:** copy that config to
   `_builds/<name>/patches/notification_config.json`, set each item's `enabled`
   to the derived vector, and point `VST_NOTIFICATION_CONFIG_PATH` at it with an
   absolute `${BUILD_DIR}/patches/` source. An **empty** vector needs no copy:
   point the variable at the shared webhooks-disabled default instead.

Project from a superset that carries every item the vector needs — normally the
one the inherited path resolves to; where it lacks an item, project from the
shipped superset that has it. Either way the mount points at the build-local
copy, never at another Foundation's directory, which would tie this build's
fan-out to that profile's future edits.

Not every profile config is decomposed. The summarization profile's carries no
ids and one consumer: a prompt-less RT-VLM `stream/add` that admits each new
stream without inferring ([`rt-vlm.md`](rt-vlm.md)). Inherit it where that
registration is the build's whole stream-driven set. Pairing it with detection,
embeddings or tagging has no shipped superset to project from — the tagging
items carry a prompt, a different capability — so take that combination to the
clarification gate in [`../composition.md`](../composition.md) rather than
assembling one.

A projection edits `enabled` and nothing else: URLs, `timeout_ms`, `retry`,
headers, `body`, and `user_defined_metadata` carry tested values, and a build
needing a different value there is changing a receiver rather than selecting one
(the single sanctioned case is the RT-CV port coupling below). Adding, deleting,
or reordering items is likewise out of bounds — a diff against the source config
that touches any line but an `enabled` or that port is an error.

### Mechanism and constraints

- **Selection is env-indirected, so a projection needs no `.yml` patch** — it is
  a payload in `patches/` that `VST_NOTIFICATION_CONFIG_PATH` points at
  ([`../composition.md`](../composition.md)). The file is bind-mounted into both
  `sensor-ms` and `streamprocessing-ms` at the fixed container path
  `configs/notification_config.json` and read once at process start — a change
  needs a container restart, and the two mount points must resolve the same
  file. The shared `${VSS_APPS_DIR}/services/vios/configs/notification_config.json`
  is the default and ships webhooks disabled: it is the right target for a build
  with no stream-driven consumer at all.
- **Tagging and always-on alerting are mutually exclusive.** One RT-VLM instance
  publishes every caption to a single `MESSAGE_BUS_TOPIC` (`mdx-vlm-captions`),
  so a static consumer cannot separate tagging output from alerting output.
  Enabling both also collides on the live stream itself: a second caption
  request with a different decode signature (`chunk_duration` and the rest) is
  rejected `400 BadParameters`. Enable at most one of the two.
- **Webhooks are independent of `message_broker.enable_notification`.** The
  factory gates only the Redis/Kafka publisher on that flag; the webhook
  notifier turns on the presence of enabled items alone.
  `VST_ENABLE_NOTIFICATION=false` with `webhooks.enabled: true` is valid and
  intended.
- **`VST_ENABLE_NOTIFICATION` overrides the JSON producer flag** — applied after
  the JSON load, so the file's `enable_notification` never wins; pin the env var
  in the build rather than trusting the file.
  `enable_notification_consumer` has **no** env override and is live: it gates
  the `LiveMetadataStore` Kafka consumer feeding the live bbox overlay off
  `mdx-raw`. A projection carries the whole `message_broker` block through
  unchanged, so this flag stays as the source config set it.
- **RT-CV port is a three-place coupling**: the webhook URL, `RTVI_CV_PORT`
  (container side of the `ports:` mapping), and `http-port` in the mounted
  DeepStream run config must agree. A build that remaps the port changes all
  three — the one receiver value a projection may edit, since the alternative is
  a webhook aimed at a port nothing listens on.

## Sources

- `deploy/docker/services/vios/compose.yml`
- `deploy/docker/services/vios/foundational/docker-compose.yaml`
- `deploy/docker/services/vios/initiator/docker-compose.yaml`
- `deploy/docker/services/vios/streamprocessing/docker-compose.yaml`
- `deploy/docker/services/vios/configs/notification_config.json` (shared default)
- `deploy/docker/developer-profiles/*/vios/configs/notification_config*.json` (the profile configs a build inherits or projects from)
- `deploy/docker/services/infra/sdrc/docker-compose.yaml`
- `skills/operations/vss-manage-video-io-storage/references/deploy-vios-service.md`
- `skills/operations/vss-manage-video-io-storage/references/integrate-vios-service.md`
