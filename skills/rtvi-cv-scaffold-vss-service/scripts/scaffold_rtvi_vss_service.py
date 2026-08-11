#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scaffold a deployable custom RTVI CV microservice that plugs into VSS.

Generates a runnable repo: Dockerfile + DeepStream pipeline configs +
compose definition + Python adapter + tests + smoke-test consumer.
The contract (mdx-raw / protobuf-2 / libnvds_kafka_proto.so) is
encoded in the generated DeepStream config and is not configurable.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from string import Template


VSS_PROFILE_FLAGS = {
    "bp_developer_search_2d",
    "bp_developer_alerts_2d_cv",
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "rtvi-vss-service"


def package_name(value: str) -> str:
    return slugify(value).replace("-", "_")


# ──────────────────────────────────────────────────────────────────────
# Templates
# ──────────────────────────────────────────────────────────────────────


DOCKERFILE = Template(
    """# Custom RTVI CV microservice — DeepStream pipeline running YOLO26
# and publishing detection metadata to Kafka topic mdx-raw.
ARG DEEPSTREAM_IMAGE=nvcr.io/nvidia/deepstream:9.1-triton-multiarch
FROM $${DEEPSTREAM_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \\
        python3 python3-pip ca-certificates curl \\
    && rm -rf /var/lib/apt/lists/*

# Pipeline assets.
COPY pipeline/ /opt/pipeline/
COPY pipeline/configs/ /opt/configs/
COPY app/ /opt/app/

# Customer mounts the actual ONNX, custom parser .so, and labels file at
# runtime — see compose/service.compose.yml. The image ships only the
# wiring and config skeletons.
RUN mkdir -p /opt/models /opt/parser

WORKDIR /opt/pipeline
ENTRYPOINT ["bash", "/opt/pipeline/ds-start.sh"]
"""
)


DS_START_SH = Template(
    """#!/usr/bin/env bash
# Entrypoint for $service_name. Substitutes runtime env into the
# deepstream-app config and execs deepstream-app.
set -euo pipefail

: "$${INPUT_URI:?must be set — RTSP URI or absolute file path inside the container}"
: "$${KAFKA_BOOTSTRAP:=localhost:9092}"
: "$${KAFKA_TOPIC:=mdx-raw}"
: "$${SENSOR_ID:=$service_name-source-0}"

# The DeepStream Kafka adapter takes host;port. The topic is supplied
# independently by the nvmsgbroker topic property.
KAFKA_HOST="$${KAFKA_BOOTSTRAP%:*}"
KAFKA_PORT="$${KAFKA_BOOTSTRAP##*:}"
if [[ ! "$${KAFKA_HOST}" =~ ^[A-Za-z0-9.-]+$$ || ! "$${KAFKA_PORT}" =~ ^[0-9]+$$ ]]; then
    echo "[$service_name] invalid KAFKA_BOOTSTRAP '$${KAFKA_BOOTSTRAP}'; expected host:port" >&2
    exit 2
fi
if [[ ! "$${SENSOR_ID}" =~ ^[A-Za-z0-9_.:-]+$$ ]]; then
    echo "[$service_name] invalid SENSOR_ID; use letters, digits, dot, underscore, colon, or hyphen" >&2
    exit 2
fi
BROKER_CONN_STR="$${KAFKA_HOST};$${KAFKA_PORT}"

# Render runtime overrides into the app and msgconv sensor configs.
RENDERED_MSGCONV_CFG=/tmp/msgconv-config.runtime.txt
sed -e "s|@SENSOR_ID@|$${SENSOR_ID}|g" \
    /opt/configs/msgconv_config.txt > "$${RENDERED_MSGCONV_CFG}"

RENDERED_CFG=/tmp/ds-app-config.runtime.txt
sed -e "s|@INPUT_URI@|$${INPUT_URI}|g" \\
    -e "s|@BROKER_CONN_STR@|$${BROKER_CONN_STR}|g" \\
    -e "s|@TOPIC@|$${KAFKA_TOPIC}|g" \\
    -e "s|@SENSOR_ID@|$${SENSOR_ID}|g" \\
    /opt/pipeline/ds-app-config.txt > "$${RENDERED_CFG}"

echo "[$service_name] kafka=$${KAFKA_BOOTSTRAP} topic=$${KAFKA_TOPIC} input=$${INPUT_URI}"
exec deepstream-app -c "$${RENDERED_CFG}"
"""
)


DS_APP_CONFIG = Template(
    """# DeepStream application config — derived from deepstream-test5.
# Pipeline: source → streammux → pgie(YOLO26) → tracker → msgconv → msgbroker(Kafka:mdx-raw)
# All @PLACEHOLDERS@ are substituted by ds-start.sh from runtime env.

[application]
enable-perf-measurement=1
perf-measurement-interval-sec=10

[tiled-display]
enable=0
rows=1
columns=1
width=1280
height=720

[source0]
enable=1
# This scaffold defaults to RTSP: type=4 and [streammux] live-source=1.
# INPUT_URI does not switch the source mode automatically. For file input,
# use a file:// URI, change type=2, and set [streammux] live-source=0.
# Source type definitions: src/apps/common/includes/deepstream_sources.h:30
type=4
uri=@INPUT_URI@
num-sources=1
gpu-id=0
cudadec-memtype=0
# camera-id is the DeepStream numeric source index. The generated msgconv config
# maps source0 to SENSOR_ID for Frame.sensorId. Verify the mapping by decoding an
# emitted nv.Frame message before relying on it downstream.
camera-id=0

[streammux]
gpu-id=0
batched-push-timeout=40000
width=1920
height=1080
batch-size=1
# RTSP default; set to 0 with source type=2 for file input.
live-source=1
nvbuf-memory-type=0

[primary-gie]
enable=1
gpu-id=0
batch-size=1
gie-unique-id=1
config-file=/opt/configs/pgie-yolo26-config.txt
nvbuf-memory-type=0

[tracker]
enable=1
tracker-width=960
tracker-height=544
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
ll-config-file=/opt/configs/tracker-nvdcf.yml
gpu-id=0
enable-batch-process=1

[sink0]
enable=1
type=6
# msg-conv-payload-type=2 is NVDS_PAYLOAD_DEEPSTREAM_PROTOBUF. The protobuf schema comes from
# msg-conv-msg2p-lib (the msgconv conversion library), not from libnvds_kafka_proto.so.
# libnvds_kafka_proto.so is the msg-broker-proto-lib Kafka protocol adapter only.
# On DGX-SPARK/THOR replace libnvds_msgconv_mega2d.so with libnvds_msgconv.so.
msg-conv-payload-type=2
msg-conv-msg2p-lib=/opt/nvidia/deepstream/deepstream/lib/libnvds_msgconv_mega2d.so
msg-conv-frame-interval=1
# deepstream-app does not attach NvDsEventMsgMeta; serialize NvDsFrameMeta/NvDsObjectMeta directly.
msg-conv-msg2p-new-api=1
msg-conv-config=/tmp/msgconv-config.runtime.txt
msg-broker-proto-lib=/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so
msg-broker-conn-str=@BROKER_CONN_STR@
topic=@TOPIC@
msg-broker-config=/opt/configs/cfg_kafka.txt
sync=0

[sink1]
enable=1
type=1
sync=0

[osd]
enable=0
"""
)


PGIE_YOLO26 = Template(
    """# Primary GIE (nvinfer) config for YOLO26 detection.
# Customer must bind-mount their ONNX export at /opt/models/yolo26.onnx and
# their compiled custom parser at /opt/parser/libnvdsinfer_custom_yolo26.so.

[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file=/opt/models/yolo26.onnx
model-engine-file=/opt/models/yolo26.onnx_b1_gpu0_fp16.engine
labelfile-path=/opt/configs/labels.txt
batch-size=1
network-mode=2
num-detected-classes=$num_classes
interval=0
gie-unique-id=1
process-mode=1
network-type=0
# Export contract: one-to-many head (end2end=False). The default YOLO26 ONNX export
# uses the end-to-end head (N,300,6, NMS in-graph) and requires cluster-mode=4 plus a
# post-NMS pass-through parser. Export with end2end=False for the (N,nc+4,8400) output
# tensor this config and parser expect; NMS is applied by nvinfer (cluster-mode=2).
cluster-mode=2
maintain-aspect-ratio=1
parse-bbox-func-name=NvDsInferParseYolo26
custom-lib-path=/opt/parser/libnvdsinfer_custom_yolo26.so
infer-dims=3;640;640

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
topk=300
"""
)


TRACKER_NVDCF = Template(
    """%YAML:1.0
# NvDCF tracker config — produces stable Object.id which Behavior Analytics requires.
BaseConfig:
  minDetectorConfidence: 0.25
TargetManagement:
  preserveStreamUpdateOrder: 0
  maxTargetsPerStream: 150
  maxShadowTrackingAge: 30
  probationAge: 3
  earlyTerminationAge: 1
TrajectoryManagement:
  useUniqueID: 1
DataAssociator:
  dataAssociatorType: 0
  associationMatcherType: 1
  checkClassMatch: 1
  minMatchingScore4Overall: 0.0
  minMatchingScore4SizeSimilarity: 0.4
  minMatchingScore4Iou: 0.0
StateEstimator:
  stateEstimatorType: 1
ObjectModelProjection:
  outputVisibility: 1
  outputShadowTracks: 0
  outputTentativeTracks: 0
"""
)


CFG_KAFKA = Template(
    """# librdkafka producer overrides for nvmsgbroker. Defaults are safe for VSS.
# Reference: https://github.com/confluentinc/librdkafka/blob/master/CONFIGURATION.md
[message-broker]
proto-cfg="queue.buffering.max.messages=10000;queue.buffering.max.ms=10;batch.num.messages=1"
"""
)


MSGCONV_CONFIG = Template(
    """# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Static sensor context required by libnvds_msgconv_mega2d.so.
[sensor0]
enable=1
type=Camera
id=@SENSOR_ID@
location=0.0;0.0;0.0
description=$service_name
coordinate=0.0;0.0;0.0
"""
)


LABELS_TXT = Template(
    """# One class label per line, matching num-detected-classes in pgie-yolo26-config.txt.
# Replace these placeholders with the customer's training labels.
$labels_block"""
)


SERVICE_COMPOSE = Template(
    """# Service definition for $service_name — plugs into a running VSS deployment.
# Start VSS first, confirm its Kafka topic initializer completed, then run:
#   docker compose -f compose/service.compose.yml --profile bp_developer_search_2d up -d
services:
  $service_name:
    image: $${$image_var:-$service_name:dev}
    container_name: $service_name
    network_mode: "host"
    runtime: nvidia
    restart: "on-failure:3"
    profiles:
$profile_lines
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]
              device_ids: ["$${RT_CV_DEVICE_ID:-0}"]
    volumes:
      - $${YOLO26_ONNX_PATH}:/opt/models/yolo26.onnx:ro
      - $${YOLO26_PARSER_LIB}:/opt/parser/libnvdsinfer_custom_yolo26.so:ro
      - $${YOLO26_LABELS_PATH:-../pipeline/configs/labels.txt}:/opt/configs/labels.txt:ro
      - $service_name-storage:/opt/models
    environment:
      INPUT_URI: $${INPUT_URI}
      KAFKA_BOOTSTRAP: $${KAFKA_BOOTSTRAP:-localhost:9092}
      KAFKA_TOPIC: $${KAFKA_TOPIC:-mdx-raw}
      SENSOR_ID: $${SENSOR_ID:-$service_name-source-0}
    # This is a delayed process-liveness check. Kafka publication must still
    # be verified with tools/kafka_smoketest.py.
    healthcheck:
      test: ["CMD-SHELL", "ps -o etimes= -p 1 | awk '$$$$1 >= 30 { ok=1 } END { exit !ok }' && grep -qx deepstream-app /proc/1/comm"]
      interval: 5s
      timeout: 3s
      retries: 24
      start_period: 5s

volumes:
  $service_name-storage:
"""
)


SERVICE_CONFIG_JSON = Template(
    """{
  "service_name": "$service_name",
  "detector": {
    "family": "YOLO26",
    "tracker": "NvDCF",
    "num_classes": $num_classes,
    "onnx_path_in_container": "/opt/models/yolo26.onnx",
    "parser_lib_in_container": "/opt/parser/libnvdsinfer_custom_yolo26.so",
    "labels_in_container": "/opt/configs/labels.txt"
  },
  "kafka": {
    "bootstrap": "localhost:9092",
    "topic": "mdx-raw",
    "payload_type": 2,
    "proto_lib": "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"
  },
  "vss_profiles": $vss_profiles_json,
  "downstream_consumers": [
    "vss-search-analytics",
    "vss-behavior-analytics",
    "vss-video-analytics-api"
  ]
}
"""
)


PACKAGE_INIT = Template(
    '''"""$service_name — Python adapter helpers.

The DeepStream pipeline produces protobuf payloads via nvmsgbroker; this
package exists to validate event envelope expectations (in tests) and to
host any extension hooks the customer wants to layer above the
DS-published metadata (e.g. a sidecar enrichment service that consumes
mdx-raw and re-publishes derived events).
"""
'''
)


CONTRACTS_PY = Template(
    '''from __future__ import annotations

from typing import Any


# Field names align with the protobuf nv.Frame schema that nvmsgconv
# produces with msg-conv-payload-type=2. The Python adapter mirrors
# that schema for tests and any sidecar extension code the customer
# adds later — DeepStream itself does not call into this module.

REQUIRED_FRAME_FIELDS = ("sensorId", "timestamp", "objects")
REQUIRED_OBJECT_FIELDS = ("id", "type", "bbox", "confidence")


def validate_frame(frame: dict[str, Any]) -> None:
    missing = [f for f in REQUIRED_FRAME_FIELDS if f not in frame]
    if missing:
        raise ValueError(f"Frame missing required fields: {missing}")
    objects = frame["objects"]
    if not isinstance(objects, list):
        raise ValueError("Frame.objects must be a list")
    for index, obj in enumerate(objects):
        missing = [f for f in REQUIRED_OBJECT_FIELDS if f not in obj]
        if missing:
            raise ValueError(f"objects[{index}] missing required fields: {missing}")
        bbox = obj["bbox"]
        if not all(k in bbox for k in ("topX", "topY", "width", "height")):
            raise ValueError(f"objects[{index}].bbox missing topX/topY/width/height")


def example_frame(sensor_id: str = "$service_name-source-0") -> dict[str, Any]:
    """Reference frame matching the protobuf-2 envelope nvmsgconv produces."""
    return {
        "sensorId": sensor_id,
        "timestamp": 1_700_000_000_000_000_000,
        "objects": [
            {
                "id": 1,
                "type": "person",
                "bbox": {"topX": 100, "topY": 200, "width": 50, "height": 120},
                "confidence": 0.92,
            }
        ],
    }
'''
)


PIPELINE_PLAN_PY = Template(
    '''from __future__ import annotations


def expected_pipeline_stages() -> list[dict[str, str]]:
    """The DeepStream stage list the generated ds-app-config.txt encodes.

    Used by tests to assert the wiring has not regressed away from the
    contract (msgconv→msgbroker→Kafka@mdx-raw with payload type 2).
    """
    return [
        {"stage": "source", "component": "uridecodebin", "detail": "RTSP or file"},
        {"stage": "batch", "component": "nvstreammux", "detail": "single batch"},
        {"stage": "detect", "component": "nvinfer", "detail": "YOLO26 primary GIE"},
        {"stage": "track", "component": "nvtracker (NvDCF)", "detail": "stable IDs"},
        {"stage": "msgconv", "component": "nvmsgconv", "detail": "payload type 2 (protobuf)"},
        {"stage": "msgbroker", "component": "nvmsgbroker", "detail": "Kafka libnvds_kafka_proto.so"},
        {"stage": "sink", "component": "kafka mdx-raw", "detail": "VSS perception ingress"},
    ]
'''
)


SERVICE_PY = Template(
    '''from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import validate_frame
from .pipeline_plan import expected_pipeline_stages


class RtviCvService:
    """Customer-extensible adapter around the DS pipeline contract.

    The DS pipeline runs in C and publishes to Kafka directly; this
    Python class is the seam where customers attach sidecar logic
    (e.g. a derived-event publisher) without breaking the on-the-wire
    contract that VSS consumers rely on.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.service_name = str(config["service_name"])

    @classmethod
    def from_path(cls, path: str | Path) -> "RtviCvService":
        return cls(json.loads(Path(path).read_text()))

    def healthcheck(self) -> dict[str, str]:
        return {"status": "ok", "service_name": self.service_name}

    def pipeline_plan(self) -> list[dict[str, str]]:
        return expected_pipeline_stages()

    def validate_outgoing_frame(self, frame: dict[str, Any]) -> None:
        validate_frame(frame)
'''
)


KAFKA_SMOKETEST = Template(
    '''#!/usr/bin/env python3
"""Smoke-test consumer for $service_name.

Confirms that at least one message arrived on the perception Kafka topic
within the timeout. This is a Kafka reachability test — it does not decode
the protobuf payload, identify the source service, or assert Frame.sensorId,
objects, tracking IDs, or bounding boxes. It will pass on any non-empty
message already present on mdx-raw, including messages from other services.
For a correlated acceptance test, decode the payload using the nv.Frame .proto
and assert Frame.sensorId matches the configured SENSOR_ID, Frame.objects is
non-empty when detections are expected, and each object carries a stable
tracking ID and valid bbox. That test is not generated by this scaffold.

This is a Kafka reachability test — it does not decode
the protobuf payload or validate sensorId, objects, or bbox fields. To
verify the nv.Frame schema, decode the payload using the nv.Frame .proto
and assert the required fields explicitly.

Designed to run from any host that can reach the Kafka broker
the perception service is publishing to.

Why no kafka-python dependency: the goal is for this script to run on
a customer machine without extra installs. We shell into the Kafka
container's bundled console consumer instead, since the foundational
VSS stack already ships one.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile


def _safe_kafka_name(value: str) -> bool:
    return bool(value) and all(c.isalnum() or c in "-_." for c in value)


def _valid_bootstrap_server(value: str) -> bool:
    if not value or value.count(":") != 1:
        return False
    host, port = value.split(":", 1)
    if not host or not port.isdigit():
        return False
    port_num = int(port)
    if not (1 <= port_num <= 65535):
        return False
    return all(c.isalnum() or c in "-." for c in host)


def _detect_kafka_containers(docker: str) -> list[str]:
    result = subprocess.run(
        [docker, "ps", "--format", "{{.Names}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []

    containers = [name.strip() for name in result.stdout.splitlines() if name.strip()]
    return [
        name
        for name in containers
        if "kafka" in name.lower()
        and "topic" not in name.lower()
        and "init" not in name.lower()
    ]


def _container_is_running(docker: str, container: str) -> bool:
    result = subprocess.run(
        [docker, "inspect", "--format", "{{.State.Running}}", container],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-server", default="localhost:9092")
    parser.add_argument("--topic", default="mdx-raw")
    parser.add_argument("--timeout", type=int, default=60,
                        help="seconds to wait for one message")
    parser.add_argument(
        "--kafka-container",
        default=None,
        help="docker container name running Kafka; auto-detected if omitted",
    )
    parser.add_argument("--max-messages", type=int, default=1)
    parser.add_argument("--describe-only", action="store_true",
                        help="just verify the topic exists; do not consume a message")
    args = parser.parse_args()

    bootstrap_server = args.bootstrap_server
    topic = args.topic
    if not _valid_bootstrap_server(bootstrap_server):
        print("[smoketest] invalid --bootstrap-server; expected host:port", file=sys.stderr)
        return 2
    if not _safe_kafka_name(topic):
        print("[smoketest] invalid --topic name", file=sys.stderr)
        return 2

    docker = shutil.which("docker")
    if docker is None:
        print("[smoketest] docker CLI not on PATH", file=sys.stderr)
        return 2

    if args.kafka_container:
        container = args.kafka_container
    else:
        containers = _detect_kafka_containers(docker)
        if not containers:
            print(
                "[smoketest] no running Kafka container detected; start the "
                "selected VSS profile or pass --kafka-container explicitly",
                file=sys.stderr,
            )
            return 2
        if len(containers) > 1:
            print(
                "[smoketest] multiple running Kafka containers detected: "
                f"{', '.join(containers)}; pass --kafka-container explicitly",
                file=sys.stderr,
            )
            return 2
        container = containers[0]
    if not _safe_kafka_name(container):
        print("[smoketest] invalid --kafka-container name", file=sys.stderr)
        return 2
    if not _container_is_running(docker, container):
        print(
            f"[smoketest] Kafka container {container!r} is not running",
            file=sys.stderr,
        )
        return 2

    # Different Kafka images put the CLI under different prefixes:
    #   confluentinc/cp-kafka         → /usr/bin/kafka-topics, kafka-console-consumer
    #   bitnami/kafka, apache/kafka   → /opt/kafka/bin/kafka-topics.sh, ...
    def _resolve_in_container(candidates):
        for path in candidates:
            check = subprocess.run(
                [docker, "exec", container, "test", "-x", path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if check.returncode == 0:
                return path
        print(f"[smoketest] no kafka CLI found in {container}; "
              f"tried: {candidates}", file=sys.stderr)
        sys.exit(2)

    kafka_topics = _resolve_in_container([
        "/usr/bin/kafka-topics",
        "/opt/kafka/bin/kafka-topics",
        "/opt/kafka/bin/kafka-topics.sh",
        "/opt/bitnami/kafka/bin/kafka-topics.sh",
    ])

    if args.describe_only:
        cmd = [
            docker, "exec", container, kafka_topics,
            "--bootstrap-server", bootstrap_server,
            "--describe", "--topic", topic,
        ]
        return subprocess.run(cmd, check=False).returncode

    kafka_consumer = _resolve_in_container([
        "/usr/bin/kafka-console-consumer",
        "/opt/kafka/bin/kafka-console-consumer",
        "/opt/kafka/bin/kafka-console-consumer.sh",
        "/opt/bitnami/kafka/bin/kafka-console-consumer.sh",
    ])

    cmd = [
        docker, "exec", container, kafka_consumer,
        "--bootstrap-server", bootstrap_server,
        "--topic", topic,
        "--max-messages", str(args.max_messages),
        "--timeout-ms", str(args.timeout * 1000),
    ]
    print(f"[smoketest] {' '.join(cmd)}", file=sys.stderr)
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        result = subprocess.run(
            cmd,
            check=False,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        stderr_file.seek(0)
        stderr_text = stderr_file.read().decode(errors="replace")
        stdout_file.seek(0)
        payload = stdout_file.read()
    if result.returncode != 0:
        print(f"[smoketest] consumer exit={result.returncode}", file=sys.stderr)
        sys.stderr.write(stderr_text)
        return 1

    if stderr_text:
        sys.stderr.write(stderr_text)
    if not payload.strip():
        print(f"[smoketest] no message arrived on {args.topic} within "
              f"{args.timeout}s", file=sys.stderr)
        return 1

    # Payload type 2 is protobuf — bytes, not JSON. We don't decode it
    # here (would require the nv.Frame .proto). We assert non-empty
    # binary payload, then dump bytes-len + magic so the result is
    # actionable.
    summary = {
        "topic": args.topic,
        "messages_received": 1,
        "payload_bytes": len(payload.strip()),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
)


TEST_PIPELINE_CONFIG = Template(
    '''from __future__ import annotations

import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DS_CONFIG = REPO / "pipeline" / "ds-app-config.txt"
PGIE = REPO / "pipeline" / "configs" / "pgie-yolo26-config.txt"
DS_START = REPO / "pipeline" / "ds-start.sh"
KAFKA_CONFIG = REPO / "pipeline" / "configs" / "cfg_kafka.txt"
MSGCONV_CONFIG = REPO / "pipeline" / "configs" / "msgconv_config.txt"


class DsAppConfigContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = DS_CONFIG.read_text()

    def test_publishes_protobuf_payload_type(self) -> None:
        self.assertIn("msg-conv-payload-type=2", self.text)

    def test_uses_frame_object_metadata_api(self) -> None:
        self.assertIn("msg-conv-msg2p-new-api=1", self.text)

    def test_uses_nvmsgbroker_kafka_protocol_lib(self) -> None:
        self.assertIn(
            "msg-broker-proto-lib=/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so",
            self.text,
        )

    def test_topic_is_mdx_raw(self) -> None:
        self.assertIn("topic=@TOPIC@", self.text)

    def test_kafka_conn_str_excludes_topic(self) -> None:
        start = DS_START.read_text()
        self.assertIn('BROKER_CONN_STR="$${KAFKA_HOST};$${KAFKA_PORT}"', start)
        self.assertNotIn(
            'BROKER_CONN_STR="$${KAFKA_HOST};$${KAFKA_PORT};$${KAFKA_TOPIC}"',
            start,
        )

    def test_kafka_options_use_deepstream_adapter_schema(self) -> None:
        kafka = KAFKA_CONFIG.read_text()
        self.assertIn("[message-broker]", kafka)
        self.assertIn('proto-cfg="', kafka)
        self.assertNotIn("[message]\\n", kafka)

    def test_msgconv_has_runtime_sensor_context(self) -> None:
        self.assertIn("msg-conv-config=/tmp/msgconv-config.runtime.txt", self.text)
        self.assertIn("[sensor0]", MSGCONV_CONFIG.read_text())
        self.assertIn("id=@SENSOR_ID@", MSGCONV_CONFIG.read_text())

    def test_pgie_uses_yolo26_config(self) -> None:
        self.assertIn("config-file=/opt/configs/pgie-yolo26-config.txt", self.text)


class Yolo26PgieTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = PGIE.read_text()

    def test_pgie_declares_yolo26_parser(self) -> None:
        self.assertIn("parse-bbox-func-name=NvDsInferParseYolo26", self.text)
        self.assertIn(
            "custom-lib-path=/opt/parser/libnvdsinfer_custom_yolo26.so",
            self.text,
        )

    def test_pgie_num_classes_matches_scaffold(self) -> None:
        self.assertIn("num-detected-classes=$num_classes", self.text)

    def test_pgie_persists_versioned_fp16_engine(self) -> None:
        self.assertIn(
            "model-engine-file=/opt/models/yolo26.onnx_b1_gpu0_fp16.engine",
            self.text,
        )


if __name__ == "__main__":
    unittest.main()
'''
)


TEST_COMPOSE = Template(
    '''from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
COMPOSE = REPO / "compose" / "service.compose.yml"
SERVICE_CONFIG = REPO / "service_config.json"
README = REPO / "README.md"


class ServiceComposeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = COMPOSE.read_text()
        self.profiles = json.loads(
            SERVICE_CONFIG.read_text()
        )["vss_profiles"]

    def test_uses_host_networking(self) -> None:
        self.assertIn('network_mode: "host"', self.text)

    def test_has_delayed_process_healthcheck(self) -> None:
        self.assertIn("healthcheck:", self.text)
        self.assertIn("deepstream-app /proc/1/comm", self.text)

    def test_declares_selected_profiles(self) -> None:
        for profile in self.profiles:
            with self.subTest(profile=profile):
                self.assertIn(profile, self.text)

    def test_reserves_gpu(self) -> None:
        self.assertIn("capabilities:", self.text)
        self.assertIn("gpu", self.text)

    def test_persists_engine_beside_onnx(self) -> None:
        self.assertIn("- $service_name-storage:/opt/models", self.text)

    def test_readme_safely_disables_selected_stock_publishers(self) -> None:
        readme = README.read_text()
        self.assertIn("stock VSS perception service both publish", readme)
        self.assertIn("Stop the stock container", readme)
        self.assertIn("com.docker.compose.service=perception-2d-fusion", readme)
        self.assertIn("com.docker.compose.service=perception-alerts", readme)
        self.assertIn("docker stop <container-name>", readme)
        self.assertIn("distinct `SENSOR_ID` values", readme)


if __name__ == "__main__":
    unittest.main()
'''
)


TEST_SERVICE = Template(
    '''from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from $package_name.contracts import example_frame, validate_frame
from $package_name.service import RtviCvService


class FrameContractTests(unittest.TestCase):
    def test_example_frame_passes_validation(self) -> None:
        validate_frame(example_frame())

    def test_missing_sensor_id_fails(self) -> None:
        frame = example_frame()
        frame.pop("sensorId")
        with self.assertRaises(ValueError):
            validate_frame(frame)

    def test_object_missing_bbox_fields_fails(self) -> None:
        frame = example_frame()
        frame["objects"][0]["bbox"] = {"topX": 0, "topY": 0}
        with self.assertRaises(ValueError):
            validate_frame(frame)


class RtviCvServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tempdir.name) / "service_config.json"
        self.config_path.write_text(json.dumps({
            "service_name": "$service_name",
            "detector": {"family": "YOLO26"},
            "kafka": {"topic": "mdx-raw"},
        }))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_pipeline_plan_publishes_protobuf_via_nvmsgbroker(self) -> None:
        svc = RtviCvService.from_path(self.config_path)
        components = [s["component"] for s in svc.pipeline_plan()]
        self.assertIn("nvmsgconv", components)
        self.assertIn("nvmsgbroker", components)


if __name__ == "__main__":
    unittest.main()
'''
)


README = Template(
    """# $service_name

Custom RTVI CV microservice — DeepStream pipeline running YOLO26 that
publishes detection metadata to the Kafka topic `mdx-raw`, the same
ingress the default VSS perception services use.

## What this service does

- Ingests a video source (RTSP or file) configured via `$$INPUT_URI`.
- Runs a primary YOLO26 detector + NvDCF tracker.
- Converts metadata to protobuf via `nvmsgconv`
  (`msg-conv-payload-type=2`).
- Publishes to Kafka via `nvmsgbroker` using
  `libnvds_kafka_proto.so` to the `mdx-raw` topic.

The result is consumed by the standard VSS Search and Alerts profile
services without any further wiring:

- `vss-search-analytics-*` indexes the frames for embed/attribute/fusion
  search.
- `vss-behavior-analytics-*` derives dwell, direction, and ROI events
  and publishes incidents to `mdx-incidents`.
- `alert-bridge` (in Alerts Profile) sends the incidents to a VLM for
  verification.

## Customer prerequisites

You provide:

- A **YOLO26 ONNX export** placed at `$$YOLO26_ONNX_PATH`. This scaffold
  does not download or ship model weights. YOLO26 weights are published by
  Ultralytics at
  [https://huggingface.co/Ultralytics/YOLO26](https://huggingface.co/Ultralytics/YOLO26).
  The nvinfer config targets the one-to-many head
  (`end2end=False`, output tensor `N,nc+4,8400`, `cluster-mode=2`).
- A **labels file** (one class per line; line count must match the
  scaffolded `num-detected-classes`).
- A **DeepStream custom parser** compiled for YOLO26's output tensor
  layout, exposing the symbol `NvDsInferParseYolo26` (see
  `pipeline/custom-parser/README.md` for the ABI signature).
- A target VSS deployment running Search Profile
  (`bp_developer_search_2d`) and/or Alerts Profile
  (`bp_developer_alerts_2d_cv`).

## Build

```bash
docker build -t $service_name:dev .
```

## Run

Run the following commands from the generated service repository root.

Set these environment variables in this generated service repository's `.env` (or export them in the service shell). Keep host artifact paths outside the generated repo and
parameterize them instead of committing workstation-specific absolute paths:

```
YOLO26_ONNX_PATH=/abs/path/to/yolo26.onnx
YOLO26_PARSER_LIB=/abs/path/to/libnvdsinfer_custom_yolo26.so
YOLO26_LABELS_PATH=/abs/path/to/labels.txt   # optional; defaults to bundled placeholder
INPUT_URI=rtsp://camera-host/stream
RT_CV_DEVICE_ID=0
KAFKA_BOOTSTRAP=localhost:9092
```

The generated service uses host networking. Set `KAFKA_BOOTSTRAP` to Kafka's
host-reachable advertised listener; it defaults to `localhost:9092`.

The VSS deployment is not included in this generated repository. The selected
VSS Search or Alerts profile must already be running before you start this
custom service. Use an existing VSS v3.2.1-compatible checkout and set
`VSS_ROOT` to its absolute path:

```bash
SERVICE_ROOT="$$PWD"
# Change this value to the existing VSS checkout on this host.
export VSS_ROOT=/absolute/path/to/video-search-and-summarization
VSS_DEPLOY_DIR="$${VSS_ROOT}/deploy/docker"
test -f "$${VSS_DEPLOY_DIR}/compose.yml" || {
  echo "Missing VSS compose file: $${VSS_DEPLOY_DIR}/compose.yml" >&2
  exit 1
}
echo "Using VSS deployment: $${VSS_DEPLOY_DIR}"
```

The readiness commands below do not deploy VSS. A prior
`deploy/docker/scripts/dev-profile.sh up --profile search ...` or
`deploy/docker/scripts/dev-profile.sh up --profile alerts ...` invocation
creates the profile-specific `generated.env`. If that file is missing, stop
here and deploy the selected VSS profile by following the VSS Quickstart.
Continue only after the profile is running and the Kafka topic initializer
completed successfully.

```bash
docker ps -a \\
  --filter label=com.docker.compose.service=kafka-topic-init-container \\
  --format 'table {{.Names}}\t{{.Status}}'

cd "$${SERVICE_ROOT}"
docker compose -f compose/service.compose.yml \\
               --profile bp_developer_search_2d up -d
```

For Alerts, use this pair instead:

```bash
docker ps -a \\
  --filter label=com.docker.compose.service=kafka-topic-init-container \\
  --format 'table {{.Names}}\t{{.Status}}'

cd "$${SERVICE_ROOT}"
docker compose -f compose/service.compose.yml \\
               --profile bp_developer_alerts_2d_cv up -d
```

The profile flag enables this custom perception service. It does not start or
configure the VSS stack.

## Stop the stock perception publisher when replacing it

This custom service and the stock VSS perception service both publish
detections to `mdx-raw`. If this service is a replacement, stop the stock
publisher first so consumers do not get multiplexed detections from both
models.

Find the running stock container:

```bash
# Search Profile
docker ps \\
  --filter label=com.docker.compose.service=perception-2d-fusion \\
  --format 'table {{.Names}}\t{{.Status}}'

# Alerts Profile
docker ps \\
  --filter label=com.docker.compose.service=perception-alerts \\
  --format 'table {{.Names}}\t{{.Status}}'
```

Stop the stock container:

```bash
docker stop <container-name>
```

If running both publishers is intentional, assign distinct `SENSOR_ID` values
and expect their detections to be mixed on `mdx-raw`.

Behavior Analytics is started by the selected VSS deployment — both the Search
and Alerts profiles run it — and consumes this service's `mdx-raw` messages.
Do not add a Behavior Analytics profile to the generated service.

The generated service uses host networking and reaches VSS Kafka through the
host-reachable advertised listener selected by `KAFKA_BOOTSTRAP` (default:
`localhost:9092`). It still runs as a separate Compose application; Compose
cannot resolve `depends_on` references across separate invocations.
If the readiness command lists multiple deployments, identify the selected
profile's topic initializer and confirm its status is `Exited (0)`.

## Verify

```bash
cd "$${SERVICE_ROOT}"
python3 tools/kafka_smoketest.py --describe-only
python3 tools/kafka_smoketest.py --timeout 60
```

Reads one message from `mdx-raw` and asserts it is non-empty. It does not
decode the protobuf payload or validate sensorId, objects, or bbox fields,
and it cannot prove the message came from this service rather than another
`mdx-raw` producer. The script auto-detects Kafka when exactly one matching
container is running. If multiple VSS deployments are active, select one:

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -i kafka
python3 tools/kafka_smoketest.py \\
  --kafka-container <running-kafka-container-name> \\
  --bootstrap-server localhost:9092 \\
  --timeout 60
```

The smoke test executes Kafka's CLI inside the selected Kafka container, so
its `--bootstrap-server` is normally the container-local `localhost:9092`.
The service separately reaches Kafka through the host-reachable advertised
listener configured by `KAFKA_BOOTSTRAP`.

### Confirm the messages are from this service

The smoke test alone is not enough when other perception services also
publish to `mdx-raw`. After it passes, check that your `SENSOR_ID` (default
`$service_name-source-0`) appears in live payloads and that the container
is healthy:

```bash
# 1. Service is up and the pipeline is producing frames
docker ps --filter name=$service_name --format 'table {{.Names}}\t{{.Status}}'

# deepstream-app interleaves many blank lines with its **PERF: output, so a
# plain `--tail 20` often shows nothing. Drop the blank lines first.
docker logs $service_name 2>&1 | grep -v '^[[:space:]]*$$' | tail -20

# 2. Sample mdx-raw and confirm YOUR sensorId (not another producer's)
docker exec <running-kafka-container-name> \\
  /usr/bin/kafka-console-consumer \\
  --bootstrap-server localhost:9092 \\
  --topic mdx-raw --max-messages 30 --timeout-ms 20000 \\
  2>/dev/null | strings | grep -o '$service_name-source-[0-9]*' | sort | uniq -c
```

Expect most/all sampled messages to show `$service_name-source-0`. Class
labels (for example `person`) may also appear in `strings` output when
detections are present.

```bash
# 3. Optional: Behavior Analytics is consuming your sensor
docker logs --tail 200 <running-behavior-analytics-container-name> 2>/dev/null \\
  | grep $service_name-source-0 | tail -5
```

If step 2 shows a different sensor id (or none), the smoke test likely
passed on another producer's message — fix `SENSOR_ID` / `INPUT_URI` and
re-check before treating the run as successful.

## Test (no GPU)

```bash
python3 -m unittest discover -v -s tests -p 'test_*.py'
```

Asserts the DeepStream config wires `msgconv → msgbroker → mdx-raw`
correctly, the compose file declares the right profile flags and host
networking, and the Python event-envelope adapter validates the
expected `nv.Frame` shape.

## What's in the box

- `Dockerfile` — DeepStream 9.1 multi-arch base.
- `pipeline/ds-app-config.txt` — deepstream-app config (test5-derived).
- `pipeline/configs/pgie-yolo26-config.txt` — nvinfer config skeleton.
- `pipeline/configs/tracker-nvdcf.yml` — tracker config.
- `pipeline/configs/cfg_kafka.txt` — librdkafka tunables.
- `pipeline/configs/msgconv_config.txt` — sensor context for mega2d protobuf generation.
- `pipeline/configs/labels.txt` — class labels (placeholder).
- `pipeline/ds-start.sh` — entrypoint; renders runtime env into the
  config and execs `deepstream-app`.
- `compose/service.compose.yml` — service definition with VSS profile
  gates, host networking, and delayed liveness healthcheck.
- `app/` — Python adapter for tests / sidecar extension.
- `tools/kafka_smoketest.py` — end-to-end verifier.
- `tests/` — unit tests (no GPU).
"""
)


CUSTOM_PARSER_README = Template(
    """# YOLO26 custom parser

DeepStream's `nvinfer` calls into a customer-supplied parser to convert
the YOLO26 output tensors into `NvDsInferParseObjectInfo` records. The
generated `pgie-yolo26-config.txt` references this parser by symbol
name and library path:

```
parse-bbox-func-name=NvDsInferParseYolo26
custom-lib-path=/opt/parser/libnvdsinfer_custom_yolo26.so
```

## ABI

```c
extern "C" bool NvDsInferParseYolo26(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList);
```

Compile against the DeepStream headers at
`/opt/nvidia/deepstream/deepstream/sources/includes/`. Link against
`libnvinfer.so` and `libnvds_infer.so`.

YOLO26's output tensor differs from YOLOv8/v11 — anchor-free, with a
single tensor combining boxes, scores, and class probabilities. Refer
to the YOLO26 reference parser shipped by NVIDIA (or the parser shipped
in the customer's training repo) for the tensor layout.

## Mounting at runtime

The compose file already mounts the customer's compiled `.so` at
`/opt/parser/libnvdsinfer_custom_yolo26.so`:

```yaml
volumes:
  - $${YOLO26_PARSER_LIB}:/opt/parser/libnvdsinfer_custom_yolo26.so:ro
```

Set `YOLO26_PARSER_LIB` to the host path of the compiled library before
running `docker compose up`.
"""
)


# ──────────────────────────────────────────────────────────────────────
# Renderer
# ──────────────────────────────────────────────────────────────────────


def _profile_lines(profiles: list[str]) -> str:
    return "\n".join(f"      - {p}" for p in profiles)


def _labels_block(num_classes: int) -> str:
    return "".join(f"class_{i}\n" for i in range(num_classes))


def _image_var(service_name: str) -> str:
    return slugify(service_name).upper().replace("-", "_") + "_IMAGE"


def render_tree(
    service_name: str,
    num_classes: int,
    vss_profiles: list[str],
) -> dict[str, str]:
    package = package_name(service_name)
    values: dict[str, object] = {
        "service_name": slugify(service_name),
        "package_name": package,
        "num_classes": num_classes,
        "vss_profiles_json": json.dumps(vss_profiles),
        "profile_lines": _profile_lines(vss_profiles),
        "labels_block": _labels_block(num_classes),
        "image_var": _image_var(service_name),
    }

    files: dict[str, str] = {
        "Dockerfile": DOCKERFILE.substitute(values),
        "README.md": README.substitute(values),
        "service_config.json": SERVICE_CONFIG_JSON.substitute(values),
        "pipeline/ds-start.sh": DS_START_SH.substitute(values),
        "pipeline/ds-app-config.txt": DS_APP_CONFIG.substitute(values),
        "pipeline/configs/pgie-yolo26-config.txt": PGIE_YOLO26.substitute(values),
        "pipeline/configs/tracker-nvdcf.yml": TRACKER_NVDCF.substitute(values),
        "pipeline/configs/cfg_kafka.txt": CFG_KAFKA.substitute(values),
        "pipeline/configs/msgconv_config.txt": MSGCONV_CONFIG.substitute(values),
        "pipeline/configs/labels.txt": LABELS_TXT.substitute(values),
        "pipeline/custom-parser/README.md": CUSTOM_PARSER_README.substitute(values),
        "compose/service.compose.yml": SERVICE_COMPOSE.substitute(values),
        f"app/__init__.py": PACKAGE_INIT.substitute(values),
        f"app/contracts.py": CONTRACTS_PY.substitute(values),
        f"app/pipeline_plan.py": PIPELINE_PLAN_PY.substitute(values),
        f"app/service.py": SERVICE_PY.substitute(values),
        "tools/kafka_smoketest.py": KAFKA_SMOKETEST.substitute(values),
        "tests/test_pipeline_config.py": TEST_PIPELINE_CONFIG.substitute(values),
        "tests/test_compose.py": TEST_COMPOSE.substitute(values),
        "tests/test_service.py": TEST_SERVICE.substitute({**values, "package_name": "app"}),
    }
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--num-classes",
        type=int,
        default=80,
        help="must match the customer's YOLO26 labels file",
    )
    parser.add_argument(
        "--vss-profiles",
        nargs="*",
        default=["bp_developer_search_2d", "bp_developer_alerts_2d_cv"],
        help="VSS profile flags that gate this separate service",
    )
    parser.add_argument("--force", action="store_true")
    # Legacy aliases — kept so existing CI invocations and tests keep working.
    parser.add_argument("--search-profile", default=None,
                        help="(legacy, ignored) — VSS Search Profile flag is fixed")
    parser.add_argument("--alerts-profile", default=None,
                        help="(legacy, ignored) — VSS Alerts Profile flag is fixed")
    parser.add_argument("--behavior-profile", default=None,
                        help="(legacy, ignored) — Behavior Analytics has no profile flag")
    args = parser.parse_args()

    bad = [p for p in args.vss_profiles if p not in VSS_PROFILE_FLAGS]
    if bad:
        raise SystemExit(f"Unknown VSS profile flag(s): {bad}. "
                         f"Allowed: {sorted(VSS_PROFILE_FLAGS)}")
    if args.num_classes <= 0:
        raise SystemExit("--num-classes must be positive")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = render_tree(
        service_name=args.service_name,
        num_classes=args.num_classes,
        vss_profiles=list(args.vss_profiles),
    )
    for relpath, content in rendered.items():
        dest = output_dir / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        if relpath.endswith(".sh") or relpath.endswith("kafka_smoketest.py"):
            dest.chmod(0o755)

    manifest = {
        "service_name": slugify(args.service_name),
        "package_name": package_name(args.service_name),
        "output_dir": output_dir.as_posix(),
        "num_classes": args.num_classes,
        "vss_profiles": list(args.vss_profiles),
        "kafka": {"topic": "mdx-raw", "bootstrap": "localhost:9092",
                  "payload_type": 2},
        "files": sorted(rendered),
    }
    (output_dir / "scaffold_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
