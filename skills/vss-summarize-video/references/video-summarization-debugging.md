# Video Summarization Debugging Reference

Use this for video summarization-specific troubleshooting after the `lvs` profile has been
deployed or partially deployed.

## Fast Status

```bash
curl -s -o /dev/null -w '%{http_code}\n' \
  "${LVS_BACKEND_URL:-http://localhost:38111}/v1/ready"

docker ps --filter name=vss-lvs --format '{{.Names}} {{.Status}}'
docker logs --tail 100 vss-lvs
```

HTTP 200 on `/v1/ready` means ready. HTTP 503 means the service is warming or a
dependency is unavailable.

## Video Summarization Service Not Ready

Check dependencies:

```bash
curl -sf "http://${HOST_IP}:8018/v1/models" | jq '.data[].id'
curl -sf "http://${HOST_IP}:9200/_cluster/health" | jq .
docker logs --tail 100 vss-rtvi-vlm
docker logs --tail 100 vss-lvs
```

Common causes:

| Symptom | Likely cause | Fix |
|---|---|---|
| `400 BadParameters: No such model` | The request used an id not advertised by the serving endpoint. | Query LVS `/models` or RT-VLM `/v1/models` before POST and use an advertised id. |
| `/v1/ready` returns 503 | LLM, RT-VLM, ES, or another dependency is warming/unreachable. | Check dependency logs and endpoint URLs. |
| `curl` to the video summarization service works on host but not in an agent sandbox | Network namespace or sandbox visibility differs. | Use host-visible shell/deployment context. |
| Summarize returns 503 | The video summarization service is busy processing another file. | In an interactive ops/debug session, wait and retry as a separate user-approved request. In the summarize workflow, do not retry automatically. |
| Empty or weak event output | Scenario/events too narrow or no matching content. | Report the exact LVS response and offer a separate rerun with broader events or scenario. Do not broaden events automatically. |

## Model Id Mismatch

The default `lvs` profile routes VLM calls through RT-VLM. Verify:

```bash
curl -sf "http://${HOST_IP}:8018/v1/models" | jq -r '.data[].id'
```

Treat the returned id as authoritative. Image tags and deployment modes can
change the advertised id, so do not rely on a hardcoded friendly or NIM profile
name.

## Kafka / Logstash Path

The 3.2 `lvs` profile uses Kafka and shared Logstash for streaming captions and
structured summaries.

Expected topics:

| Topic | Producer / consumer |
|---|---|
| `mdx-vlm-captions` | RT-VLM produces raw captions; Logstash consumes. |
| `mdx-structured-events-summary` | the video summarization service publishes structured summaries; Logstash consumes. |

Checks:

```bash
docker logs --tail 100 logstash
docker exec kafka kafka-topics --bootstrap-server localhost:29092 --list
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:29092 \
  --topic mdx-vlm-captions \
  --max-messages 1
```

If Logstash starts but does not index video summarization data, check that the shared infra
Logstash pipeline is loading the video summarization pipeline and that protobuf
definitions are mounted from `deploy/docker/services/infra/elk/logstash`.

## API Validation Failures

`422` usually means the request body violates the OpenAPI schema.

Rules:

- `model`, `scenario`, and `events` are required for `/v1/summarize`.
- `additionalProperties: false` means extra fields can fail validation.
- Omit frame sampling fields in the standard workflow so RT-VLM uses its
  model-specific deployment default. `num_frames_per_chunk` is deprecated.
- `schema` is a JSON schema serialized as a string, not a nested object.

## Logs

```bash
docker logs -f vss-lvs
docker logs -f vss-rtvi-vlm
docker logs -f logstash
docker logs -f kafka
```

Use bounded logs in automated checks:

```bash
docker logs --tail 200 --since 10m vss-lvs
```
