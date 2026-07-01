# § 16 — Kafka Message Schema Configuration & Custom Protobuf

> **Applies to**: `messager.py`, `ds_sop_process.py`, `nvds_action_detector/protos/`
> **Related sections**: § 6 (SOPProcessManager), § 11 (Environment Variables), § 15 (Latency Measurement)

---

## Message Schema Selection

The `SOP_MESSAGING_SCHEMA` environment variable controls how chunk results are
serialized before being published to Kafka.

| Value | Serialization | Description |
|-------|--------------|-------------|
| `JSON` (default) | Plain JSON | `chunk_info` dict is JSON-encoded directly. Simple, human-readable, easy to consume from any language. |
| `NvProtoSchema` | Protobuf (`VisionLLM`) | `chunk_info` is mapped into an `nv.VisionLLM` protobuf message and binary-serialized. Required for NVIDIA analytics pipelines that expect this schema. |

### Configuration

Set in `.env` or `compose.yaml`:

```bash
# .env
ENABLE_MESSAGING=1
SOP_MESSAGING_SCHEMA=JSON            # default — plain JSON
# SOP_MESSAGING_SCHEMA=NvProtoSchema # use protobuf VisionLLM schema
```

In `compose.yaml`, pass through to the container:

```yaml
environment:
  ENABLE_MESSAGING: "${ENABLE_MESSAGING:-false}"
  SOP_MESSAGING_SCHEMA: "${SOP_MESSAGING_SCHEMA:-JSON}"
  KAFKA_BROKER: "${KAFKA_BROKER:-localhost:9092}"
  DEFAULT_TOPIC: "${DEFAULT_TOPIC:-mdx-vlm-captions}"
```

### How it works in `messager.py`

```python
SOP_MESSAGING_SCHEMA = os.getenv("SOP_MESSAGING_SCHEMA", "JSON")

def create_producer(kafka_broker: str):
    schema = SOP_MESSAGING_SCHEMA
    if schema == "NvProtoSchema":
        return NvProtoMessageProducer(kafka_broker)
    elif schema == "JSON":
        return JSONMessageProducer(kafka_broker)
    else:
        logger.warning(f"Unknown SOP_MESSAGING_SCHEMA: {schema}, defaulting to JSON")
        return JSONMessageProducer(kafka_broker)

def create_consumer(kafka_broker: str, group_id: str):
    schema = SOP_MESSAGING_SCHEMA
    if schema == "NvProtoSchema":
        return NvProtoMessageConsumer(kafka_broker, group_id)
    elif schema == "JSON":
        return JSONMessageConsumer(kafka_broker, group_id)
    else:
        logger.warning(f"Unknown SOP_MESSAGING_SCHEMA: {schema}, defaulting to JSON")
        return JSONMessageConsumer(kafka_broker, group_id)
```

### Schema behavior differences

**JSON schema** — `JSONMessageProducer` serializes the `chunk_info` dict as-is:

```python
value = json.dumps(chunk_info).encode("utf-8")
key = str(request_id).encode("utf-8")
```

The consumer decodes with `json.loads()`. All fields from `chunk_info` are
preserved with their original types (floats stay floats, dicts stay dicts).

**NvProtoSchema** — `NvProtoMessageProducer` maps `chunk_info` through
`convert_to_vision_llm()`, which selectively copies fields into the
`nv.VisionLLM` protobuf structure:

```python
vision_llm = convert_to_vision_llm(chunk_info)
value = vision_llm.SerializeToString()
key = str(sensor_id).encode("utf-8")
```

The consumer decodes with `ParseFromString()` + `MessageToDict()`. Only fields
explicitly mapped in `convert_to_vision_llm()` are transmitted — any `chunk_info`
keys not handled by the mapping function are silently dropped.

### When to use which

| Use case | Schema |
|----------|--------|
| Standalone SOP deployment, debugging, custom consumers | `JSON` |
| Integration with NVIDIA Metropolis / VSS analytics pipelines | `NvProtoSchema` |
| Need all `chunk_info` fields without mapping boilerplate | `JSON` |
| Need typed protobuf for schema enforcement across services | `NvProtoSchema` |

---

## Extending Kafka Messages with Custom Data

When you need to transmit additional data (e.g., per-operator timestamps for
efficiency measurement) through Kafka, the approach depends on which schema
you are using.

### With JSON schema — add keys directly

With `SOP_MESSAGING_SCHEMA=JSON`, any key added to the `chunk_info` dict is
automatically included in the Kafka message. No mapping code needed.

**Step 1** — Add data in the pipeline (`ds_sop_process.py`):

```python
chunk_info = {
    "chunk_idx": chunk_idx,
    "start_time": start_time,
    "end_time": end_time,
    # ... existing fields ...
    # --- custom operator timestamps ---
    "op_preprocess_start_ts": preprocess_start,
    "op_preprocess_end_ts": preprocess_end,
    "op_inference_start_ts": inference_start,
    "op_inference_end_ts": inference_end,
}
```

**Step 2** — Consume directly:

```python
consumer = create_consumer(kafka_broker, group_id)
for key, value in consumer.consume():
    preprocess_latency = value["op_preprocess_end_ts"] - value["op_preprocess_start_ts"]
```

No changes to `messager.py` required.

### With NvProtoSchema — use the `info` field or recompile

With `SOP_MESSAGING_SCHEMA=NvProtoSchema`, only fields explicitly mapped in
`convert_to_vision_llm()` are transmitted. There are two sub-approaches:

#### Approach A — Use the `info` field (recommended)

Nearly every message type in `nv.proto` includes a generic extension point:

```protobuf
map<string, string> info = N;
```

The `VisionLLM.info` field (field 8) already carries pipeline metadata:

```protobuf
message VisionLLM {
  string version = 1;
  google.protobuf.Timestamp timestamp = 2;   // chunk start
  google.protobuf.Timestamp end = 3;         // chunk end
  string startFrameId = 4;
  string endFrameId = 5;
  Sensor sensor = 6;
  LLM llm = 7;
  map<string, string> info = 8;              // ← extension point
  repeated Frame frames = 9;
}
```

**Step 1** — Add data in the pipeline (`ds_sop_process.py`), same as JSON:

```python
chunk_info["op_preprocess_start_ts"] = preprocess_start
chunk_info["op_preprocess_end_ts"] = preprocess_end
```

**Step 2** — Propagate to the `info` field in `messager.py`:

Add your new keys to the info-propagation list in `convert_to_vision_llm()`:

```python
for key in [
    "chunk_idx", "cv_execute_time", "vlm_execute_time",
    "frame_number", "checker_execute_time", "first_timestamp",
    # --- custom operator timestamps ---
    "op_preprocess_start_ts", "op_preprocess_end_ts",
    "op_inference_start_ts", "op_inference_end_ts",
    "pipeline_starting_timestamp", "pipeline_cv_ready_timestamp",
    "pipeline_vlm_starting_timestamp", "pipeline_vlm_ready_timestamp",
]:
    if key in data:
        msg.info[key] = str(data[key])
```

All values must be converted to `str` because `map<string, string>` only
accepts string values. Consumers parse them back with `float()` or `json.loads()`.

For complex/nested data, JSON-encode it (same pattern used for `checker_result`):

```python
checker_result = data.get("checker_result")
if checker_result:
    try:
        msg.info["checker_result"] = json.dumps(checker_result)
    except Exception:
        pass
```

**Step 3** — Read the data on the consumer side:

```python
from google.protobuf.json_format import MessageToDict
from protos import nv_pb2

vision_llm = nv_pb2.VisionLLM()
vision_llm.ParseFromString(raw_bytes)

chunk_idx = int(vision_llm.info.get("chunk_idx", "-1"))
preprocess_start = float(vision_llm.info.get("op_preprocess_start_ts", "0"))

# Or via MessageToDict (used by NvProtoMessageConsumer)
msg_dict = MessageToDict(vision_llm)
info = msg_dict.get("info", {})
```

**Summary — info field data flow:**

```
ds_sop_process.py          messager.py                    Kafka Consumer
─────────────────          ───────────────                 ──────────────
chunk_info["my_key"] ──►   msg.info["my_key"] = str(v) ──► float(info["my_key"])
       (dict)                (VisionLLM protobuf)            (MessageToDict)
```

No `.proto` changes. No recompilation. No impact on other services.

#### Approach B — Modify and recompile `.proto`

> **Warning**: Modifying a `.proto` structure may require changes in **every
> service** that imports and deserializes these messages. Only proceed if you
> need typed fields, nested messages, or enum constraints that `info` cannot
> provide. Coordinate with all downstream consumers before merging.

**Source files (`.proto` is the source of truth):**

```
nvds_action_detector/protos/
├── nv.proto          # Source: Frame, Object, Sensor, VisionLLM, LLM, Query ...
├── ext.proto         # Source: Behavior, Incident, GeoLocation, SpaceUtilization
├── __init__.py       # Package init: imports nv_pb2, ext_pb2
├── nv_pb2.py         # ← Generated at Docker build time (do NOT copy or hand-edit)
└── ext_pb2.py        # ← Generated at Docker build time (do NOT copy or hand-edit)
```

The `.proto` files are shipped in the Docker image and compiled during
`docker build`. The `*_pb2.py` files are **not** version-controlled or
copied from references — they are always regenerated by `protoc`.

`ext.proto` imports `nv.proto` (`import "nv.proto";`), so changes to `nv.proto`
message types may cascade into `ext.proto` consumers.

**Step 1 — Edit the `.proto` file:**

```protobuf
message VisionLLM {
  string version = 1;
  google.protobuf.Timestamp timestamp = 2;
  google.protobuf.Timestamp end = 3;
  string startFrameId = 4;
  string endFrameId = 5;
  Sensor sensor = 6;
  LLM llm = 7;
  map<string, string> info = 8;
  repeated Frame frames = 9;

  // --- NEW: per-operator timing ---
  OperatorTimestamps operator_timestamps = 10;
}

// NEW message
message OperatorTimestamps {
  double preprocess_start = 1;
  double preprocess_end = 2;
  double inference_start = 3;
  double inference_end = 4;
  double postprocess_start = 5;
  double postprocess_end = 6;
}
```

Proto3 field number rules:
- New fields MUST use a **new, unused field number** (here `10`, since 1–9 are taken).
- NEVER reuse a deleted field number — old consumers may misinterpret the data.
- NEVER change the type or number of existing fields.

**Step 2 — Compilation happens at Docker build time:**

The Dockerfile installs `protobuf-compiler` and compiles after copying sources:

```dockerfile
# In apt install block:
RUN apt update && apt install -y ... protobuf-compiler

# After copy_sources.sh:
RUN cd /opt/nvidia/nvds_sop/nvds_action_detector/protos/ && \
    protoc -I. --python_out=. nv.proto ext.proto
```

This generates `nv_pb2.py` and `ext_pb2.py` inside the container.
No pre-compiled `*_pb2.py` files need to be shipped or version-controlled.

To compile locally (for testing outside Docker):

```bash
cd nvds_action_detector/protos/
protoc -I. --python_out=. nv.proto ext.proto
```

**Step 3 — Use the new fields in `messager.py`:**

```python
def convert_to_vision_llm(data: dict) -> nv_pb2.VisionLLM:
    msg = nv_pb2.VisionLLM()
    # ... existing field population ...

    if "op_preprocess_start_ts" in data:
        msg.operator_timestamps.preprocess_start = data["op_preprocess_start_ts"]
        msg.operator_timestamps.preprocess_end = data["op_preprocess_end_ts"]
        msg.operator_timestamps.inference_start = data["op_inference_start_ts"]
        msg.operator_timestamps.inference_end = data["op_inference_end_ts"]

    return msg
```

**Step 4 — Update ALL consumers:**

Every service that deserializes `VisionLLM` messages must compile from the
updated `.proto` files (or receive the updated `*_pb2.py`). Test backward
compatibility: old messages without the new field must still deserialize
correctly (proto3 defaults missing fields to zero/empty).

---

## Approach Comparison

| Dimension | JSON schema | NvProto + `info` field | NvProto + recompile |
|-----------|------------|----------------------|---------------------|
| **Files changed** | `ds_sop_process.py` only | `ds_sop_process.py`, `messager.py` | `.proto`, `messager.py`, all consumers |
| **Downstream impact** | None | None — unknown keys ignored | All services must recompile from updated `.proto` |
| **Type safety** | Python native types | All values are strings | Full proto3 type checking |
| **Nested structures** | Native Python dicts/lists | JSON-encode into string | Native protobuf nesting |
| **Mapping boilerplate** | None — all keys included | Must add key to propagation list | Must add field to mapping |
| **Recompile needed** | No | No | Yes — automatic at Docker build |
| **Deployment risk** | Low | Low | High — coordinated rollout |

---

## Common Patterns

### Pattern: Transmit pipeline timestamps for latency profiling

Add the four internal timestamps to the info-propagation list (NvProtoSchema)
or they are already present in `chunk_info` (JSON):

```python
# NvProtoSchema only — add to convert_to_vision_llm()
for key in [
    "pipeline_starting_timestamp",
    "pipeline_cv_ready_timestamp",
    "pipeline_vlm_starting_timestamp",
    "pipeline_vlm_ready_timestamp",
]:
    if key in data:
        msg.info[key] = str(data[key])
```

Consumer-side latency computation:

```python
# NvProtoSchema — values are strings
cv_latency = float(info["pipeline_cv_ready_timestamp"]) - float(info["pipeline_starting_timestamp"])

# JSON — values are already floats
cv_latency = msg["pipeline_cv_ready_timestamp"] - msg["pipeline_starting_timestamp"]
```

### Pattern: JSON-encode complex data into `info` (NvProtoSchema)

```python
# Producer
step_timestamps = [
    {"step": "decode", "start": 1.0, "end": 1.5},
    {"step": "preprocess", "start": 1.5, "end": 2.0},
]
msg.info["step_timestamps"] = json.dumps(step_timestamps)

# Consumer
steps = json.loads(info.get("step_timestamps", "[]"))
```

### Pattern: Use `Frame.info` for per-frame metadata (NvProtoSchema)

```python
frame = msg.frames.add()
frame.id = str(frame_id)
frame.timestamp.seconds = int(pts)
frame.timestamp.nanos = int((pts - int(pts)) * 1e9)
frame.info["decode_latency_ms"] = str(decode_ms)
```

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Kafka messages are empty / not sent | `ENABLE_MESSAGING` not set | Set `ENABLE_MESSAGING=1` in `.env` |
| Consumer receives binary gibberish | Producer uses `NvProtoSchema`, consumer expects JSON (or vice versa) | Ensure `SOP_MESSAGING_SCHEMA` matches on both sides |
| `KeyError` reading `info` on consumer | Key not present in older messages | Use `.get(key, default)` instead of `[key]` |
| Custom fields missing from Kafka (NvProtoSchema) | Key not added to `convert_to_vision_llm()` propagation list | Add key to the info-propagation loop, or switch to `JSON` schema |
| `float()` raises `ValueError` | Value is not a numeric string (e.g., JSON blob) | Use `json.loads()` for structured data |
| `DecodeError` after modifying proto | Consumer compiled from old `.proto` | Distribute updated `.proto` and recompile on all consumers |
| `protoc` not found in container | `protobuf-compiler` not in Dockerfile `apt install` | Add `protobuf-compiler` to the `apt install` block in `Docker.build` |
| `nv_pb2.py` missing at runtime | `.proto` files not compiled during build | Ensure Dockerfile has `protoc -I. --python_out=. nv.proto ext.proto` after `copy_sources.sh` |
| `ext_pb2.py` import error after compile | `nv.proto` not on protoc search path | Run `protoc -I. --python_out=. nv.proto ext.proto` from inside `protos/` directory |
