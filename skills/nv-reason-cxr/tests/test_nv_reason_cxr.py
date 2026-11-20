# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
SCRIPT = SKILL_DIR / "scripts" / "run_nv_reason_cxr.py"
FIXTURE = SKILL_DIR / "fixtures" / "synthetic_cxr_input.json"


def _run(*args: str | Path, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(a) for a in args)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )


# Import the runner module so tests can exercise internal stream parsing.
def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("nv_reason_cxr_runner", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_json_fixture_mock_generates_valid_output(tmp_path: Path) -> None:
    proc = _run(FIXTURE, "--out-dir", tmp_path / "out")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["skill"] == "nv_reason_cxr"
    assert "backend" not in payload["runtime"]
    assert payload["runtime"]["mock"] is True
    assert payload["runtime"]["mode"] == "mock"
    assert payload["runtime"]["model"] == "nvidia/NV-Reason-CXR-3B"
    assert payload["runtime"]["model_revision"] == ("056bd0383b35226554da9dc5866e095df174ae19")
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


def _generated_png(tmp_path: Path) -> Path:
    directory = tmp_path / "fixture_out"
    proc = _run(FIXTURE, "--out-dir", directory)
    assert proc.returncode == 0, proc.stderr
    generated = directory / "input_synthetic_chest_xray.png"
    assert Path(json.loads(proc.stdout)["input"]["image"]["path"]) == generated
    return generated


def test_direct_png_mock_path(tmp_path: Path) -> None:
    generated = _generated_png(tmp_path)

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


def test_direct_png_mock_without_out_dir_does_not_create_default_runs(tmp_path: Path) -> None:
    generated = _generated_png(tmp_path)

    proc = _run(generated, "--mock", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["input"]["image"]["source"] == "file"
    assert not (tmp_path / "runs").exists()


def test_generated_fixture_requires_out_dir() -> None:
    proc = _run(FIXTURE)
    assert proc.returncode == 2
    assert "--out-dir is required" in proc.stderr


def test_prompt_preset_for_direct_image(tmp_path: Path) -> None:
    generated = _generated_png(tmp_path)

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


# Verify API mock mode preserves the same public JSON contract.
def test_api_backend_mock_keeps_json_contract(tmp_path: Path) -> None:
    proc = _run(
        FIXTURE,
        "--backend",
        "hf-space-api",
        "--mock",
        "--out-dir",
        tmp_path / "api_mock_out",
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "backend" not in payload["runtime"]
    assert payload["runtime"]["mode"] == "mock"
    assert payload["runtime"]["mock"] is True
    assert payload["runtime"]["device"] == "none"
    assert payload["runtime"]["max_new_tokens"] is None
    assert "space" not in payload["runtime"]
    assert "api_name" not in payload["runtime"]
    assert "space_url" not in payload["runtime"]
    assert payload["output"]["response_text"]


# Reject local cache-only mode before attempting a remote API call.
def test_api_backend_rejects_local_files_only(tmp_path: Path) -> None:
    proc = _run(
        FIXTURE,
        "--backend",
        "hf-space-api",
        "--local-files-only",
        "--out-dir",
        tmp_path / "api_bad_out",
    )
    assert proc.returncode == 2
    assert "--local-files-only is only valid with --backend local" in proc.stderr


# Ensure streaming output keeps model-generated reasoning tags intact.
def test_hf_space_stream_preserves_reasoning_and_answer_tags() -> None:
    module = _script_module()
    generating = {
        "msg": "process_generating",
        "output": {
            "data": [
                [
                    ["append", None, "<think>first sentence."],
                    ["append", None, " second sentence.</think>"],
                    ["append", None, "\n<answer>Finding</answer>"],
                ]
            ]
        },
    }
    completed = {"msg": "process_completed", "success": True}
    stream = [
        f"data: {json.dumps(generating)}\n".encode(),
        b"\n",
        f"data: {json.dumps(completed)}\n".encode(),
        b"\n",
    ]

    assert (
        module._stream_hf_space_response(stream)
        == "<think>first sentence. second sentence.</think>\n<answer>Finding</answer>"
    )


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


def test_hosted_upload_contains_only_displayed_pixels(tmp_path: Path, monkeypatch) -> None:
    from PIL import Image, PngImagePlugin

    module = _script_module()
    path = tmp_path / "private-source-name.png"
    original = Image.new("L", (2, 3))
    original.putdata([10, 20, 30, 40, 50, 60])
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Description", "private-text-marker")
    exif = Image.Exif()
    exif[274] = 6  # Display rotated 90 degrees clockwise.
    original.save(path, pnginfo=metadata, exif=exif)
    with path.open("ab") as file:
        file.write(b"private-trailer-marker")

    def capture_upload(url, *, data, headers, timeout):
        assert url == module.HF_SPACE_URL + "/gradio_api/upload"
        assert path.name.encode() not in data
        assert b"private-text-marker" not in data and b"private-trailer-marker" not in data
        header, rest = data.split(b"\r\n\r\n", 1)
        assert b'filename="image.png"' in header and b"image/png" in header
        boundary = headers["Content-Type"].split("boundary=")[1].encode()
        png, ending = rest.split(b"\r\n--" + boundary + b"--", 1)
        assert ending == b"\r\n"
        with Image.open(io.BytesIO(png)) as decoded:
            assert decoded.info == {}
            assert decoded.size == (3, 2)
            assert list(decoded.getdata()) == [50, 30, 10, 60, 40, 20]
        return b'["/remote/image.png"]'

    monkeypatch.setattr(module, "_http_fetch", capture_upload)
    result = module._hf_space_image_payload(module.HF_SPACE_URL, path, timeout=1)
    assert result["orig_name"] == "image.png"
    assert result["mime_type"] == "image/png"
    assert result["size"] == len(module._hosted_png(path))


def test_hosted_upload_rejects_invalid_image_before_network(tmp_path: Path, monkeypatch) -> None:
    module = _script_module()
    path = tmp_path / "invalid.png"
    path.write_bytes(b"invalid image")
    monkeypatch.setattr(module, "_http_fetch", lambda *a, **kw: pytest.fail("unexpected upload"))
    with pytest.raises(module.SkillError, match="Could not prepare"):
        module._hf_space_image_payload(module.HF_SPACE_URL, path, timeout=1)


def test_model_and_processor_use_immutable_revisions(tmp_path, monkeypatch):
    module = _script_module()
    sha = "a" * 40
    hub = ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = lambda repo, name: str(tmp_path / "snapshots" / sha / name)
    hub.HfApi = lambda: SimpleNamespace(model_info=lambda repo, token: SimpleNamespace(sha=sha))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert (
        module._model_revision(module.DEFAULT_MODEL, False, None) == module.DEFAULT_MODEL_REVISION
    )
    assert module._model_revision(str(tmp_path), True, None) is None
    assert module._model_revision("example/model", False, None) == sha
    monkeypatch.setattr(hub, "HfApi", lambda: pytest.fail("unexpected network client"))
    assert module._model_revision("example/model", True, None) == sha
    monkeypatch.setattr(hub, "try_to_load_from_cache", lambda *args: None)
    with pytest.raises(module.SkillError, match="No cached"):
        module._model_revision("example/model", True, None)


def test_space_transport_accepts_only_its_https_origin(monkeypatch):
    module = _script_module()

    def open_fixed(opener, request, *, timeout):
        assert not any(hasattr(h, "redirect_request") for h in opener.handlers)
        assert request.full_url == module.HF_SPACE_URL + "/config"
        return io.BytesIO(b"public config")

    monkeypatch.setattr(module.OpenerDirector, "open", open_fixed)
    assert module._http_fetch(module.HF_SPACE_URL + "/config", timeout=1) == b"public config"
    with pytest.raises(module.SkillError, match="configured HTTPS Space"):
        module._http_fetch("https://example.org/config", timeout=1)
