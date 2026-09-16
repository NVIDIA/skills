# Plugging the Microservice into a VSS Profile

The VSS deployment and the generated perception service are separate Compose
applications. The generated service integrates through Kafka rather than
through Compose model merging. It must:

1. Declare the matching `profiles:` tags in its own compose file.
2. Start only after the VSS `kafka-topic-init-container` has completed.
3. Use a network mode that can reach the address Kafka advertises after bootstrap.

## Generated `compose/service.compose.yml` shape

The scaffolder produces this skeleton (placeholders rendered at scaffold time):

```yaml
services:
  <service-name>:
    image: ${<SERVICE_IMAGE_VAR>:-<service-name>:dev}
    container_name: <service-name>
    network_mode: "host"
    runtime: nvidia
    profiles:
      - bp_developer_search_2d
      - bp_developer_alerts_2d_cv
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              device_ids: ["${RT_CV_DEVICE_ID:-0}"]
    volumes:
      - ${YOLO26_ONNX_PATH}:/opt/models/yolo26.onnx:ro
      - ${YOLO26_PARSER_LIB}:/opt/parser/libnvdsinfer_custom_yolo26.so:ro
      - ${YOLO26_LABELS_PATH:-../pipeline/configs/labels.txt}:/opt/configs/labels.txt:ro
      - <service-name>-storage:/opt/models
    environment:
      INPUT_URI: ${INPUT_URI}
      KAFKA_BOOTSTRAP: ${KAFKA_BOOTSTRAP:-localhost:9092}
      KAFKA_TOPIC: ${KAFKA_TOPIC:-mdx-raw}
      SENSOR_ID: ${SENSOR_ID:-<service-name>-source-0}
    healthcheck:
      test: ["CMD-SHELL", "ps -o etimes= -p 1 | awk '$$1 >= 30 { ok=1 } END { exit !ok }' && grep -qx deepstream-app /proc/1/comm"]
      interval: 5s
      timeout: 3s
      retries: 24
      start_period: 5s

volumes:
  <service-name>-storage:
```

`profiles:` lists both tags so the same service definition works for either
VSS workflow. The flag gates this service only; it does not merge this Compose
model with the VSS deployment. The service uses host networking and reaches
VSS Kafka through the host-reachable advertised listener selected by
`KAFKA_BOOTSTRAP` (default: `localhost:9092`).

## Bringing it up alongside the VSS stack

Clone and check out VSS once using the **VSS source location** section in
[`SKILL.md`](../SKILL.md). That sets `VSS_ROOT` / `VSS_DEPLOY_DIR`.

Start VSS with its documented Quickstart first. Then verify the one-shot Kafka
topic initializer before starting the generated service:

```bash
# Find all VSS topic initializers without assuming a project/container name.
docker ps -a \
  --filter label=com.docker.compose.service=kafka-topic-init-container \
  --format 'table {{.Names}}\t{{.Status}}'

# Confirm the selected deployment's initializer is Exited (0), then start
# the generated service with the matching profile gate.
cd <generated-service-root>
docker compose -f compose/service.compose.yml \
               --profile bp_developer_search_2d up -d

# Or for Alerts:
docker compose -f compose/service.compose.yml \
               --profile bp_developer_alerts_2d_cv up -d
```

The `--profile` option in this second command enables only the generated custom service; it does not start or configure VSS.

Do not add `depends_on: kafka-topic-init-container` to the generated compose
file. Compose resolves dependencies only within the Compose model loaded by a
single invocation; reusing a project name does not merge separately loaded
files. The readiness check supplies the required ordering.

Behavior Analytics belongs to the VSS deployment, not to the generated service Compose file. The selected VSS Search or Alerts deployment starts its own behavior consumer, which reads this service's messages from `mdx-raw`. Do not add a Behavior Analytics profile to the generated service.

## Replacing the default perception service

The custom YOLO26 service and the stock VSS perception service both publish to
`mdx-raw`. If this is a replacement, stop the stock CV container first.

Find it:

```bash
# Search Profile
docker ps \
  --filter label=com.docker.compose.service=perception-2d-fusion \
  --format 'table {{.Names}}\t{{.Status}}'

# Alerts Profile
docker ps \
  --filter label=com.docker.compose.service=perception-alerts \
  --format 'table {{.Names}}\t{{.Status}}'
```

Stop it:

```bash
docker stop <container-name>
```

Running both publishers together multiplexes detections into the same
downstream consumers and should only be done intentionally with distinct
sensor IDs.

## Required environment variables

Set these in the generated service shell or in
`<generated-service-root>/.env`:

| Variable | Purpose |
|----------|---------|
| `YOLO26_ONNX_PATH` | Host path to the YOLO26 ONNX export |
| `YOLO26_PARSER_LIB` | Host path to the compiled custom parser `.so` |
| `YOLO26_LABELS_PATH` | Optional labels path; defaults to the generated placeholder |
| `INPUT_URI` | RTSP URI or file path for the input stream |
| `RT_CV_DEVICE_ID` | GPU index (defaults to 0) |
| `KAFKA_BOOTSTRAP` | Kafka's host-reachable advertised listener; defaults to `localhost:9092` |
| `KAFKA_TOPIC` | Defaults to `mdx-raw`; do not override when integrating with VSS |
| `SENSOR_ID` | Optional stable sensor identifier for emitted metadata |

`VSS_DATA_DIR`, `HOST_IP`, and other blueprint variables remain in the VSS
checkout `generated.env`; the separate customer service does not consume them.
Do not assume a successful TCP connection to a published host port proves Kafka
will work: the client must also resolve and reach the advertised broker address.

## Verifying the integration

After both stacks are up:

```bash
# 1. From the generated service root, confirm the customer service is running.
cd <generated-service-root>
docker compose -f compose/service.compose.yml ps <service-name>

# 2. Confirm the topic exists, then consume metadata from mdx-raw.
python3 tools/kafka_smoketest.py --describe-only
python3 tools/kafka_smoketest.py --timeout 60

# If multiple Kafka containers are running, select the intended deployment.
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -i kafka
python3 tools/kafka_smoketest.py \
  --kafka-container <running-kafka-container-name> \
  --bootstrap-server localhost:9092 \
  --timeout 60

# 3. Confirm the messages are from THIS service (not another mdx-raw producer).
# Replace <service-name> with the generated service name (container_name /
# default SENSOR_ID prefix).
docker ps --filter name=<service-name> --format 'table {{.Names}}\t{{.Status}}'
docker logs <service-name> 2>&1 | grep -v '^[[:space:]]*$' | tail -20

docker exec <running-kafka-container-name> \
  /usr/bin/kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic mdx-raw --max-messages 30 --timeout-ms 20000 \
  2>/dev/null | strings | grep -o '<service-name>-source-[0-9]*' | sort | uniq -c

# Optional: Behavior Analytics is consuming your sensor
docker logs --tail 200 <running-behavior-analytics-container-name> 2>/dev/null \
  | grep <service-name>-source-0 | tail -5

# 4. Inspect downstream services with the same compose/env/override arguments
# used to deploy the selected VSS profile.
cd "${VSS_DEPLOY_DIR}"
docker compose <selected-vss-compose-arguments> ps

# 5. For Alerts, use the same selected Kafka container to watch incidents.
cd <generated-service-root>
python3 tools/kafka_smoketest.py \
  --kafka-container <running-kafka-container-name> \
  --bootstrap-server localhost:9092 \
  --topic mdx-incidents \
  --timeout 30
```

The smoke test runs Kafka's CLI inside the selected broker container, so its
bootstrap address is normally container-local `localhost:9092`. The generated
DeepStream service uses host networking and the endpoint selected by
`KAFKA_BOOTSTRAP`, which defaults to `localhost:9092`. Kafka must expose and
advertise that same host-reachable endpoint.

If step 2 fails, no metadata is reaching VSS — debug the perception
service's logs for `nvmsgbroker` errors. If step 2 succeeds but step 3
shows a different sensor id (or none), the smoke test likely passed on
another producer's message. If step 3 succeeds but 4/5 show no activity,
the issue is downstream consumer config, not the customer's service.
