# Integration Contract

The contract a custom RTVI CV microservice must satisfy to plug into a VSS
deployment without any changes to the downstream Search / Alerts /
Behavior Analytics services.

## Data path

```text
[YOUR DS pipeline]──nvmsgconv──nvmsgbroker──┐
                                            │ Kafka topic: mdx-raw
                                            │ Bootstrap:   host-reachable listener
                                            ▼
   ┌──────────────────────────────┬────────────────────────────────┐
   │                              │                                │
   ▼                              ▼                                ▼
vss-search-analytics-*    vss-behavior-analytics-*       vss-video-analytics-api-*
(embed/attribute/        (dwell, direction, ROI,        (REST API for the
fusion search)            tripwire — emits mdx-incidents) Video Analytics index)
                                  │
                                  ▼
                              alert-bridge (VLM-verified incidents)
```

`vss-behavior-analytics` is deployed in **both** Search Profile and Alerts
Profile, so any frame that reaches `mdx-raw` triggers behavior derivation.

## Fixed broker contract

| Element                | Value                                                               | Source of truth                              |
|------------------------|---------------------------------------------------------------------|----------------------------------------------|
| Kafka topic            | `mdx-raw`                                                           | `topic=mdx-raw` in every VSS DS config       |
| Kafka connection string | `<broker-host>;<port>`; topic is configured separately              | DeepStream Kafka adapter contract             |
| Networking              | `network_mode: host`; `KAFKA_BOOTSTRAP=<host>:<port>`                 | Use Kafka's host-reachable advertised listener; default `localhost:9092` |
| `nvmsgbroker` proto    | `/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so`      | DeepStream-shipped Kafka adapter             |
| Payload type           | `msg-conv-payload-type=2` (`NVDS_PAYLOAD_DEEPSTREAM_PROTOBUF`); protobuf serialized by `msg-conv-msg2p-lib`, not by the Kafka adapter above | VSS deserializers expect type 2 |
| Startup ordering       | Verify `kafka-topic-init-container` completed before starting this separate Compose app | Foundational stack pre-creates `mdx-raw` |

## Payload schema

`msg-conv-payload-type=2` (`NVDS_PAYLOAD_DEEPSTREAM_PROTOBUF`) uses the msgconv library set by
`msg-conv-msg2p-lib` — `libnvds_msgconv_mega2d.so` for most VSS deployments
(`libnvds_msgconv.so` on DGX-SPARK/THOR). `libnvds_kafka_proto.so` is the
`msg-broker-proto-lib` Kafka protocol adapter, not the schema library. Each
Kafka message is one serialized `nv.Frame` protobuf produced by the msgconv
library. The fields the downstream VSS services rely on are:

| Field                | Type        | Used by                                                       | Notes                                  |
|----------------------|-------------|---------------------------------------------------------------|----------------------------------------|
| `Frame.sensorId`     | string      | search index, behavior windows, alerts                        | Customer-owned camera/stream id. The generated msgconv sensor config maps source 0 to runtime `SENSOR_ID`; verify by decoding a Kafka message. |
| `Frame.timestamp`    | uint64 (ns) | event ordering, dwell calculation                             | Must be monotonic per sensor           |
| `Frame.objects[].id` | int64       | track continuity, behavior derivation                         | Stable across frames                   |
| `Frame.objects[].type` | string    | class filter in search, alert rule matching                   | Comes from the labels file             |
| `Frame.objects[].bbox` | rect      | ROI / tripwire / direction estimation                         | `topX/topY/width/height` in pixels     |
| `Frame.objects[].confidence` | float | search ranking, alert thresholding                          | 0.0–1.0                                |
| `Frame.objects[].embedding`  | bytes | embed search (Search Profile only — optional)                | Optional; populated if a SGIE attaches it |

For a detection-only service, the customer does **not** need to construct this
protobuf manually. DeepStream's `nvmsgconv` builds it from `NvDsFrameMeta` /
`NvDsObjectMeta` when `msg-conv-payload-type=2` and
`msg-conv-msg2p-new-api=1` are set and a primary detector is producing object
meta. The new API flag is required for this `deepstream-app` scaffold because it
does not attach `NvDsEventMsgMeta` itself. The sink must also pass the generated
`msgconv_config.txt`; mega2d uses it to initialize sensor context and populate
`Frame.sensorId`. The pipeline plan in `pipeline/ds-app-config.txt`
already wires this. However, for segmentation-capable models, the customer
needs to append the wrapper pattern, explained below:
delegate base object serialization to the stock converter, then append the
frame-mask extension while preserving the base fields.


## Segmentation frame-mask payload contract

Use this section only for segmentation-capable CV models. Detection-only
services should follow the base payload schema above.

Segmentation does not replace the VSS object contract. A segmentation-capable
payload must preserve the normal `Frame.objects[]` data and add a frame-level
mask. Do not emit mask-only frames, and do not serialize only object-local
masks. Object-local or per-instance model masks are intermediate data that must
be projected or composited into one frame mask before output.

| Element | Requirement | Notes |
|---------|-------------|-------|
| Object data | Keep `Frame.objects[]` populated with `id`, `type`, `bbox`, and `confidence`. | Behavior Analytics and Search still depend on object metadata. |
| `Frame.sensorId` | Keep as the customer-owned stream/camera string. | Do not substitute numeric source IDs or add a conflicting top-level numeric field. |
| Frame mask | Emit one frame-level mask only for frames where `Frame.objects[]` is non-empty; omit the mask for empty-detection frames. | The mask must align with the same frame coordinate space used by object bboxes. |
| Frame-mask encoding | Match the model addendum contract: dimensions, dtype, value semantics, overlap policy, serialization. Empty-frame behavior in this integration contract is fixed to mask omission. | See `../../rtvi-cv-customize-model/references/segmentation-model-contract.md`. | 
| Msgconv wrapper | Declare the wrapper type/name, msgconv library path, payload type, schema version, and exact frame-mask field names. | This integration contract owns wrapper/schema details; the model addendum owns mask construction semantics. |
| Downstream compatibility | Confirm which VSS consumers are expected to consume or ignore the frame mask. | Do not assume existing detection consumers understand new mask fields. |

The stock detection path keeps `msg-conv-payload-type=2` so `nvmsgconv`
serializes object metadata into `nv.Frame`; `libnvds_kafka_proto.so` remains
the `nvmsgbroker` Kafka protocol adapter that publishes that payload. For a
segmentation service which needs a custom msgconv wrapper or extended converter to
carry the frame mask, document that wrapper here before implementation. At
minimum, record:

- the msgconv wrapper type or class name,
- the `msg-conv-msg2p-lib` path,
- the `msg-conv-payload-type`,
- the protobuf/schema message and field names for the frame mask,
- the frame-mask encoding used on the wire,
- how `sensorId`, timestamps, and `objects[]` remain identical to the base
  VSS contract.

### Msgconv wrapper implementation pattern

A segmentation wrapper should keep the DeepStream pipeline on the normal
VSS message path and extend the payload at the msgconv boundary:

- Keep `[sink]/msgconv` on `msg-conv-payload-type=2` and `msg-conv-msg2p-new-api=1` when serializing frame/object metadata directly.
- Keep the Kafka broker adapter wiring unchanged; `libnvds_kafka_proto.so`
  is the `nvmsgbroker` protocol adapter, not the msgconv wrapper library.
- Set `msg-conv-msg2p-lib` to a wrapper library that exports the standard
  DeepStream msgconv C ABI: `nvds_msg2p_ctx_create`,
  `nvds_msg2p_ctx_destroy`, `nvds_msg2p_generate`,
  `nvds_msg2p_generate_multiple`, `nvds_msg2p_generate_new`,
  `nvds_msg2p_generate_multiple_new`, and `nvds_msg2p_release`.
- Inside the wrapper, load and delegate base object serialization to the
  stock VSS converter, for example `libnvds_msgconv_mega2d.so` or the
  documented stock converter for that deployment. The wrapper should
  mutate or extend the serialized `nv.Frame` payload after the stock
  converter has populated `sensorId`, timestamp, and `objects[]`.
- Use original model tensor metadata from `NVDSINFER_TENSOR_OUTPUT_META`
  when constructing the frame mask, instead of tracker or object-meta mutation.
- If the wrapper reparses model tensors, compile or share the same parser
  core used by the DeepStream custom parser so bbox, class-id, threshold,
  and coordinate rules cannot diverge between OSD/object meta and Kafka
  frame-mask output.
- For models that emit detection, label, and mask tensors, derive class
  count and label mapping from tensor shapes and the labels file instead of
  a stale environment default.
- For empty detection frames with no objects, frame masks must be omitted downstream.
- Add wrapper-aware debug evidence when possible, such as
  `metadata_source=tensor_meta` or `metadata_source=object_mask_fallback`,
  so GPU test logs show which mask source reached Kafka.

### `Frame.sensorId` wrapper failure mode

Guard against segmentation wrappers that replace the customer-owned
`Frame.sensorId` with a numeric DeepStream source id such as `"0"`, or append a
conflicting top-level `sensorId` field. The canonical value is the stream/camera
id from frame metadata, for example `NvDsFrameMeta::sensorInfo_meta.sensor_id`
or the stream-add camera id.

If the wrapper mutates a serialized `nv.Frame`, replace the existing
`Frame.sensorId` field rather than appending a second value. Validate detection
and segmentation paths by consuming Kafka and asserting `Frame.sensorId` equals
the expected stream id.

Validation must decode at least one Kafka message from `mdx-raw` and assert:

- `Frame.sensorId` is the expected string stream ID,
- `Frame.objects[]` is present and populated when detections exist,
- every object has bbox, type, confidence, and stable ID,
- the frame-level mask exists when segmentation output is expected,
- the decoded frame mask matches the declared dimensions, dtype/value domain,
  coordinate space, empty-frame behavior, and overlap policy.

## Profile flags

| Profile             | Compose `--profile`              | What it brings up beyond the broker          |
|---------------------|----------------------------------|----------------------------------------------|
| Search Profile      | `bp_developer_search_2d`         | `vss-search-analytics`, search ingestion API, behavior consumer, embed/attribute/fusion query layer |
| Alerts Profile      | `bp_developer_alerts_2d_cv`      | `vss-behavior-analytics-alerts`, `alert-bridge` (VLM verifier), `vss-video-analytics-api-alerts` |
| Behavior Analytics  | (no separate flag)               | Runs as a consumer inside both of the above. Reads `mdx-raw`, emits `mdx-incidents`. |

The customer service should declare *both* profile names if it intends to
support both deployment shapes. Compose only spins it up when the matching
`--profile` flag is passed at deploy time, so listing both is free.

## Topic outputs from the rest of the stack

Useful for debugging downstream pickup — the customer's service does not
write to these:

| Topic              | Producer                          | Consumer                                |
|--------------------|-----------------------------------|-----------------------------------------|
| `mdx-incidents`    | `vss-behavior-analytics-*`        | `alert-bridge`                          |
| `mdx-vlm-incidents`| `alert-bridge`                    | Elasticsearch ingestion                 |
| `mdx-notification` | `vss-behavior-analytics-*`        | UI / notification dispatch              |

## Anti-patterns

- Publishing to a topic other than `mdx-raw`. The downstream behavior
  analytics consumers are pinned to `mdx-raw`. Renaming the topic
  silently produces an empty index.
- Using a Python `confluent_kafka` producer instead of `nvmsgbroker`.
  The protobuf payload is serialized by the `msg-conv-msg2p-lib` msgconv
  library and transported by `libnvds_kafka_proto.so` — it is not a
  documented external protobuf, and replicating it from Python is fragile
  and will drift across DeepStream versions.
- Pointing the host-networked service at Kafka without a host-reachable
  advertised listener. Kafka clients reconnect to the advertised broker
  endpoint, so its advertised listener must match the endpoint selected by
  `KAFKA_BOOTSTRAP`.
- Per-frame regenerated `track_id`. Track continuity is what behavior
  analytics keys on; without it dwell / direction / ROI logic produces
  garbage even though messages arrive.
- Mixing payload types across pipelines. If half the streams emit
  protobuf-2 and half emit minimal-deepstream, behavior analytics
  silently drops the latter.
- For segmentation services, emitting a frame mask without the base
  `Frame.objects[]` contract. Downstream consumers still require object
  metadata.
- Serializing object-local masks as the final service output when the
  integration contract requires one frame-level mask.
- Changing `Frame.sensorId` semantics in a segmentation wrapper. It must
  remain the customer-owned stream/camera string used by downstream VSS
  consumers.

## Validation evidence

Local (no GPU, no VSS):

- Unit tests in `tests/` validate the pipeline config wiring, compose
  profile gates, and the python event adapter.

End-to-end (GPU host with VSS deployed):

- `tools/kafka_smoketest.py` confirms at least one non-empty message arrived
  on `mdx-raw` within the timeout. It does not decode the protobuf payload
  or validate sensorId, objects, or bbox fields — it is a Kafka reachability
  check only. To validate the nv.Frame schema, decode the payload using the
  nv.Frame .proto and assert the required fields explicitly.
- For segmentation services, schema validation must cover both the base
  object contract and the decoded frame-mask contract.
- Health checks on `vss-search-analytics-*` and
  `vss-behavior-analytics-*` confirm the consumers are connected.
- A query against the Video Analytics API returns events with the
  expected `sensorId`.
