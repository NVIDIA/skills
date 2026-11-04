#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the fresh LEAPP bundle's ROS interface and relative file references.

Numerical model validation is performed by upstream LEAPP during export.
This additional check prevents a malformed/configuration-mismatched bundle from
being handed to ROS Deploy. It requires only PyYAML from the export environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import yaml


def validate(path):
    validation_log = (path.parent / "log.txt").read_text()
    if "[PASS] DisplayPortTaskSpace passed validation" not in validation_log or "Passed: 1/1" not in validation_log:
        raise ValueError("LEAPP numerical validation did not report success")
    data = yaml.safe_load(path.read_text())
    model = data["models"]["DisplayPortTaskSpace"]
    expected = {
        "eef_pos": ([1, 3], "eef_pose_pos", "state/body/position"),
        "eef_rot_6d": ([1, 6], "eef_pose_rot6d", "state/body/rotation_6d"),
        "socket_kp_pos": ([1, 3], "socket_kp_pose_pos", "state/body/position"),
        "socket_kp_rot_6d": ([1, 6], "socket_kp_pose_rot6d", "state/body/rotation_6d"),
    }
    inputs = {v["name"]: v for v in model["inputs"]}
    for name, (shape, source, kind) in expected.items():
        actual = inputs[name]
        if (actual["shape"], actual["source"], actual["kind"]) != (shape, source, kind):
            raise ValueError(f"Incorrect ROS input contract: {name}")
    action = next(v for v in model["outputs"] if v["name"] == "arm_action")
    if action["shape"] != [1, 6] or action["kind"] != "target/body/pose_relative":
        raise ValueError("Incorrect relative Cartesian action contract")
    pipeline = data["pipeline"]
    if pipeline["configs"]["frequency"] != 30:
        raise ValueError("Expected 30 Hz exported policy")
    root = path.parent.resolve()

    def relative_file(value):
        file = (root / value).resolve()
        if not file.is_relative_to(root) or not file.is_file() or not file.stat().st_size:
            raise ValueError(f"Missing/unsafe model reference: {value}")
        return file

    model_path = relative_file(model["parameters"]["model_path"])
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != model["parameters"]["sha256sum"]:
        raise ValueError("ONNX file hash does not match LEAPP YAML")
    state = relative_file(pipeline["initial_values"])
    with state.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > 1024 * 1024:
            raise ValueError("Unexpected initial-state header length")
        header = json.loads(stream.read(length))
    feedback = pipeline["feedback_flow"]
    for index in (0, 1):
        output = f"DisplayPortTaskSpace/actor_state_{index}_out"
        input_name = f"DisplayPortTaskSpace/actor_state_{index}_in"
        if feedback.get(output) != [input_name]:
            raise ValueError(f"Missing recurrent feedback {output}")
        if not any(name in header for name in (input_name, output)):
            raise ValueError(f"Missing initial recurrent state {input_name}")
    return {"passed": True, "onnx_sha256": digest, "frequency_hz": 30,
            "numerical_validation": "Passed by upstream LEAPP compiler; see export log.txt"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    print(json.dumps(validate(parser.parse_args().bundle), indent=2))
