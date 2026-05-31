# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke tests for the NV-Reason-CXR skill wrapper.

These tests verify input handling, mock-mode output, and clean error paths.
They do not download or run the NV-Reason-CXR model.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
SCRIPT = SKILL_DIR / "scripts" / "run_nv_reason_cxr.py"
FIXTURE = SKILL_DIR / "fixtures" / "synthetic_cxr_input.json"


def _run(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(a) for a in args)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_json_fixture_mock_generates_valid_output(tmp_path: Path) -> None:
    proc = _run(FIXTURE, "--out-dir", tmp_path / "out")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["skill"] == "nv_reason_cxr"
    assert payload["runtime"]["mock"] is True
    assert payload["runtime"]["mode"] == "mock"
    assert payload["runtime"]["model"] == "nvidia/NV-Reason-CXR-3B"
    assert payload["runtime"]["generated_tokens"] == 0
    assert payload["runtime"]["truncated_by_max_new_tokens"] is False
    assert payload["input"]["image"]["source"] == "generated_fixture"
    assert payload["input"]["image"]["format"] == "png"
    assert payload["input"]["image"]["width"] > 0
    assert payload["input"]["image"]["height"] > 0
    assert payload["output"]["response_text"]


def test_check_setup_reports_json() -> None:
    proc = _run("--check-setup")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["skill"] == "nv_reason_cxr"
    assert "dependencies" in payload["setup"]
    assert "recommendation" in payload["setup"]


def test_direct_png_mock_path(tmp_path: Path) -> None:
    first = _run(FIXTURE, "--out-dir", tmp_path / "fixture_out")
    assert first.returncode == 0, first.stderr
    generated = Path(json.loads(first.stdout)["input"]["image"]["path"])

    proc = _run(
        generated,
        "--mock",
        "--prompt",
        "Describe the chest X-ray findings.",
        "--out-dir",
        tmp_path / "direct_out",
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["input"]["image"]["source"] == "file"
    assert payload["input"]["prompt"] == "Describe the chest X-ray findings."
    assert payload["runtime"]["mock"] is True


def test_prompt_preset_for_direct_image(tmp_path: Path) -> None:
    first = _run(FIXTURE, "--out-dir", tmp_path / "fixture_out")
    assert first.returncode == 0, first.stderr
    generated = Path(json.loads(first.stdout)["input"]["image"]["path"])

    proc = _run(
        generated,
        "--mock",
        "--prompt-preset",
        "structured",
        "--out-dir",
        tmp_path / "preset_out",
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "structured response" in payload["input"]["prompt"]


def test_rejects_non_image_file(tmp_path: Path) -> None:
    bad = tmp_path / "not_an_image.txt"
    bad.write_text("not an image")
    proc = _run(bad, "--mock", "--out-dir", tmp_path / "out")
    assert proc.returncode == 2
    assert "unsupported image format" in proc.stderr


def test_missing_input_fails_cleanly(tmp_path: Path) -> None:
    proc = _run(tmp_path / "missing.png", "--mock", "--out-dir", tmp_path / "out")
    assert proc.returncode == 2
    assert "input not found" in proc.stderr
