# RTVI VLM Kafka Workflows

### 3. Dense captions with alerts from an RTSP stream (Kafka incidents)

On VSS 3.1 the same `/v1/generate_captions_alerts` endpoint emits alerts — there is no
per-request alert flag. Alerts are driven by **prompt design + server-side phrase
detection**: the server lower-cases each chunk's VLM response and checks for the tokens
**`"yes"` or `"true"`**. If either appears, the server builds an incident protobuf
(`isAnomaly=True`, `info["triggerPhrase"]=<matched tokens>`, `info["verdict"]="confirmed"`)
and publishes it to `KAFKA_INCIDENT_TOPIC` in addition to the normal caption message on
`KAFKA_TOPIC`. Per <https://docs.nvidia.com/vss/latest/real-time-vlm.html>.

**Recommended prompt pattern** (from the docs):
```
Anomaly Detected: Yes/No
Reason: [Brief explanation]
```
Pair it with `system_prompt` that constrains the model to answer Yes/No.

```bash

### 4. HTTP response vs. Kafka message bus

When `KAFKA_ENABLED=true`, the same request produces both outputs: an HTTP
response to the caller and Kafka records for downstream message-bus consumers.

**HTTP response** from `POST /v1/generate_captions_alerts`:
- **`stream=true`** — Server-Sent Events. One SSE event per chunk containing the
  `VlmCaptionResponse` fields (`start_ts`, `end_ts`, `content`, `chunk_id` when
  supported). Terminated by `[DONE]` per OpenAI-style SSE convention.
- **`stream=false`** (default) — single JSON object wrapping all chunks:
  ```json
  {
    "id": "<request_id>",
    "object": "caption",
    "chunk_responses": [
      {"start_time": "...", "end_time": "...", "content": "..."}
    ],
    "usage": {...}
  }
  ```

**Kafka publish** (when `KAFKA_ENABLED=true`):
- Every caption → **`KAFKA_TOPIC`** (default `vision-llm-messages`) with header
  `message_type: vision_llm` and `info["incidentDetected"] = "true"|"false"`.
- Alert-positive chunks → **also** published to **`KAFKA_INCIDENT_TOPIC`** (default
  `vision-llm-events-incidents`) with header `message_type: incident`.
- Any upstream/VLM error → **`ERROR_MESSAGE_TOPIC`** (default `vision-llm-errors`)
  with header `message_type: error`.
- **Partition key:** `<request_id>:<chunk_idx>` — all messages for one (request, chunk)
  pair land on the same partition so a consumer can join the caption and the incident.
- **Value format:** NvSchema protobuf, not JSON. Use metadata-only consumers for
  quick verification; use the protobuf descriptors under
  `deployments/foundational/elk/pb_definitions/descriptors/` for structured decoding.

For deterministic validation, first check topic offsets:
```bash
for T in vision-llm-messages vision-llm-events-incidents vision-llm-errors; do
  docker exec mdx-kafka kafka-get-offsets \
    --bootstrap-server 127.0.0.1:9092 \
    --topic "$T"
done
```

Then consume bounded, metadata-only samples from all three topics. `--timeout-ms`
prevents a no-message topic from hanging indefinitely; `print.value=false` avoids
printing protobuf bytes:
```bash
for T in vision-llm-messages vision-llm-events-incidents vision-llm-errors; do
  docker exec mdx-kafka kafka-console-consumer \
    --bootstrap-server 127.0.0.1:9092 \
    --topic "$T" \
    --from-beginning \
    --timeout-ms 5000 \
    --max-messages 20 \
    --property print.timestamp=true \
    --property print.key=true \
    --property print.headers=true \
    --property print.value=false
done
```

Typical proof of an HTTP + Kafka alert pass:
```text
vision-llm-messages:0:8
vision-llm-events-incidents:0:1
vision-llm-errors:0:0

CreateTime:<ms> message_type:vision_llm <request_id>:5
CreateTime:<ms> message_type:incident   <request_id>:5
```

The incident key matching the caption key (`<request_id>:<chunk_idx>`) is the
join point between the normal caption message and the alert-positive incident.
On recent Confluent Kafka images, do not override the formatter with the older
`kafka.tools.DefaultMessageFormatter`; the default consumer formatter already
supports the `print.*` properties above.

**Docs reference:** <https://docs.nvidia.com/vss/latest/real-time-vlm.html>

---

