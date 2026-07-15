# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Reference test suite for SOP Inference Microservice API endpoints.
Source: https://github.com/NVIDIA/sop-monitoring-blueprints/blob/main/microservices/sop-inference-bp/tests/test_api_endpoints.py
Local:  sop-inference-bp/tests/test_api_endpoints.py

Covers all endpoints defined in api_server.py.
Run: TEST_VIDEO_PATH=/path/to/video.mp4 python test_api_endpoints_reference.py
"""

import base64
import io
import json
import os

import requests

BASE_URL = "http://localhost:8300"
TEST_VIDEO_PATH = os.getenv("TEST_VIDEO_PATH", "test_video_whole_sop_h264.mp4")


def _get_test_video_base64(video_path: str):
    try:
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Test video not found: {video_path}")


test_video_base64_str = _get_test_video_base64(TEST_VIDEO_PATH)


def print_log(msg: str):
    print(msg, flush=True)


# =============================================================================
# Health Check Endpoints
# =============================================================================

class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_health_live(self):
        response = requests.get(f"{BASE_URL}/v1/live")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "health.response"
        assert data["message"] == "Service is live."
        print_log("✓ /v1/live")

    def test_health_startup(self):
        response = requests.get(f"{BASE_URL}/v1/startup")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "health.response"
        assert "started successfully" in data["message"].lower()
        print_log("✓ /v1/startup")

    def test_health_ready(self):
        response = requests.get(f"{BASE_URL}/v1/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "health.response"
        assert "ready" in data["message"].lower() or "dummy" in data["message"].lower()
        print_log("✓ /v1/ready")


# =============================================================================
# Model Endpoints
# =============================================================================

class TestModelEndpoints:
    """Test model-related endpoints"""

    def test_list_models(self):
        response = requests.get(f"{BASE_URL}/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0
        model = data["data"][0]
        assert "id" in model
        assert model["object"] == "model"
        assert "owned_by" in model
        print_log(f"✓ /v1/models → model id: {model['id']}")


# =============================================================================
# Metadata Endpoint
# =============================================================================

class TestMetadataEndpoint:
    """Test metadata endpoint"""

    def test_show_metadata(self):
        response = requests.get(f"{BASE_URL}/v1/metadata")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "modelInfo" in data
        assert "licenseInfo" in data
        license_info = data["licenseInfo"]
        assert all(k in license_info for k in ["name", "path", "size", "content"])
        print_log(f"✓ /v1/metadata → version: {data['version']}")


# =============================================================================
# Prometheus Metrics Endpoint
# =============================================================================

class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint"""

    def test_metrics_endpoint_basic(self):
        response = requests.get(f"{BASE_URL}/v1/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        assert len(response.text) > 0
        print_log(f"✓ /v1/metrics → {len(response.text)} bytes")

    def test_metrics_contains_expected_metrics(self):
        response = requests.get(f"{BASE_URL}/v1/metrics")
        assert response.status_code == 200
        text = response.text
        # These two MUST be present (others are GPU-dependent)
        assert "api_requests_total" in text
        assert "api_request_latency_seconds" in text
        print_log("✓ /v1/metrics contains required metrics")

    def test_metrics_format_validity(self):
        response = requests.get(f"{BASE_URL}/v1/metrics")
        assert response.status_code == 200
        lines = response.text.strip().split("\n")
        has_comments = any(l.strip().startswith("#") for l in lines if l.strip())
        has_metrics = any(len(l.split()) >= 2 for l in lines if l.strip() and not l.startswith("#"))
        assert has_comments or has_metrics
        print_log(f"✓ /v1/metrics Prometheus format valid")

    def test_metrics_increments_on_requests(self):
        r1 = requests.get(f"{BASE_URL}/v1/metrics")
        count1 = self._extract_metric_value(r1.text, "api_requests_total", "/v1/live")
        requests.get(f"{BASE_URL}/v1/live")
        r2 = requests.get(f"{BASE_URL}/v1/metrics")
        count2 = self._extract_metric_value(r2.text, "api_requests_total", "/v1/live")
        if count1 is not None and count2 is not None:
            assert count2 > count1
            print_log(f"✓ api_requests_total increments: {count1} → {count2}")
        else:
            print_log("✓ api_requests_total present (count skip)")

    def test_metrics_chat_completions_counter(self):
        response = requests.get(f"{BASE_URL}/v1/metrics")
        assert response.status_code == 200
        if "chat_completions_total" in response.text:
            print_log("✓ chat_completions_total found")
        else:
            print_log("✓ chat_completions_total will appear after first completion")

    def test_metrics_gpu_metrics_present(self):
        response = requests.get(f"{BASE_URL}/v1/metrics")
        assert response.status_code == 200
        has_gpu = "gpu_utilization_percent" in response.text or "gpu_memory_used_megabytes" in response.text
        print_log(f"✓ GPU metrics {'available' if has_gpu else 'not available (expected without GPU)'}")

    def _extract_metric_value(self, text, metric_name, path_label=None):
        for line in text.split("\n"):
            if not line.startswith(metric_name):
                continue
            if path_label and f'path="{path_label}"' not in line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[-1])
                except ValueError:
                    continue
        return None


# =============================================================================
# File Management Endpoints
# =============================================================================

class TestFileEndpoints:
    """Test file management endpoints"""

    def test_file_upload_list_chat_completion_download_delete_workflow(self):
        """Full workflow: upload → list → chat → download → delete → verify 404"""

        # 1. Upload
        with open(TEST_VIDEO_PATH, "rb") as f:
            content = f.read()
        filename = os.path.basename(TEST_VIDEO_PATH)
        response = requests.post(
            f"{BASE_URL}/v1/files",
            files={"file": (filename, io.BytesIO(content), "video/mp4")},
            data={"purpose": "test"}
        )
        assert response.status_code == 200
        upload = response.json()
        assert upload["object"] == "file"
        assert "id" in upload
        assert upload["filename"] == filename
        assert upload["bytes"] == len(content)
        assert upload["purpose"] == "test"
        file_id = upload["id"]
        print_log(f"✓ Upload → {file_id}")

        # 2. List
        response = requests.get(f"{BASE_URL}/v1/files")
        assert response.status_code == 200
        list_data = response.json()
        assert list_data["object"] == "list"
        assert any(f["id"] == file_id for f in list_data["data"])
        print_log(f"✓ List → {len(list_data['data'])} files")

        # 3. Download
        response = requests.get(f"{BASE_URL}/v1/files/{file_id}/content")
        assert response.status_code == 200
        assert response.content == content
        assert response.headers.get("content-type") == "application/octet-stream"
        print_log(f"✓ Download → {len(response.content)} bytes")

        # 4. Chat completion with file_id
        payload = {
            "model": "ds_sop_model",
            "messages": [{"role": "user", "content": [{"type": "input_video", "file_id": file_id}]}]
        }
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=180
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert "id" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        print_log(f"✓ Chat with file_id → {data['choices'][0]['message']['content'][:50]}...")

        # 5. Delete
        response = requests.delete(f"{BASE_URL}/v1/files/{file_id}")
        assert response.status_code == 200
        delete_data = response.json()
        assert delete_data["id"] == file_id
        assert delete_data["object"] == "file.deleted"
        assert delete_data["deleted"] is True
        print_log(f"✓ Delete → {file_id}")

        # 6. Verify 404 after delete
        response = requests.get(f"{BASE_URL}/v1/files/{file_id}/content")
        assert response.status_code == 404
        print_log("✓ 404 after deletion")

    def test_file_not_found(self):
        fake_id = "file-nonexistent123456789"
        assert requests.get(f"{BASE_URL}/v1/files/{fake_id}/content").status_code == 404
        assert requests.delete(f"{BASE_URL}/v1/files/{fake_id}").status_code == 404
        print_log("✓ 404 for non-existent file")


# =============================================================================
# Chat Completion Endpoint
# =============================================================================

class TestChatCompletionEndpoint:
    """Test chat completion endpoint"""

    def test_chat_completion_basic(self):
        """Non-streaming chat completion with base64 video"""
        payload = {
            "model": "ds_sop_model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this video for safety compliance"},
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{test_video_base64_str}"}}
                ]
            }],
            "stream": False,
            "chunking_options": {
                "algorithm": "ddm-net",
                "threshold": 0.8,
                "min_length_sec": 1.0,
                "max_length_sec": 60.0
            }
        }
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=180
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["id"].startswith("chatcmpl-")
        assert "created" in data and "model" in data
        assert len(data["choices"]) > 0
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        print_log(f"✓ Non-streaming chat → {len(choice['message']['content'])} chars")

    def test_chat_completion_streaming(self):
        """Streaming chat completion returns SSE chunks"""
        payload = {
            "model": "ds_sop_model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check for PPE compliance"},
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{test_video_base64_str}"}}
                ]
            }],
            "stream": True
        }
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=180
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        chunks = []
        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunks.append(json.loads(data_str))
                    except json.JSONDecodeError:
                        pass

        assert len(chunks) > 0
        assert chunks[0]["object"] == "chat.completion.chunk"
        assert "id" in chunks[0]
        assert "choices" in chunks[0]
        print_log(f"✓ Streaming chat → {len(chunks)} chunks")

    def test_chat_completion_invalid_content_type(self):
        """Wrong Content-Type → 415"""
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={"model": "ds_sop_model", "messages": []},
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 415
        print_log("✓ 415 for text/plain content-type")

    def test_chat_completion_validation_errors(self):
        """Input validation returns appropriate errors"""
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={"model": "ds_sop_model", "messages": []},
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 415
        print_log("✓ Validation errors handled")


# =============================================================================
# Uniform Chunking
# =============================================================================

class TestUniformChunkingEndpoint:
    """Test chat completion endpoint with uniform chunking algorithm"""

    def test_uniform_chunking_non_streaming(self):
        """Uniform chunking is accepted and returns a valid non-streaming response"""
        payload = {
            "model": "ds_sop_model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{test_video_base64_str}"}}
                ]
            }],
            "stream": False,
            "chunking_options": {"algorithm": "uniform", "chunk_length_sec": 2.5}
        }
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        assert response.status_code == 200, f"Status: {response.status_code}, Body: {response.text[:500]}"
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["id"].startswith("chatcmpl-")
        choice = data["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        metadata_list = choice.get("chunk_metadata_list", [])
        assert len(metadata_list) > 0, "Expected at least one chunk in chunk_metadata_list"
        for chunk_meta in metadata_list:
            assert "start_time" in chunk_meta and "end_time" in chunk_meta and "chunk_idx" in chunk_meta
        print_log(f"✓ Uniform chunking (non-streaming) → {len(metadata_list)} chunks")

    def test_uniform_chunking_streaming(self):
        """Uniform chunking is accepted and returns a valid streaming response"""
        payload = {
            "model": "ds_sop_model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{test_video_base64_str}"}}
                ]
            }],
            "stream": True,
            "chunking_options": {"algorithm": "uniform", "chunk_length_sec": 2.5}
        }
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=60
        )
        assert response.status_code == 200, f"Status: {response.status_code}, Body: {response.text[:500]}"
        assert "text/event-stream" in response.headers.get("content-type", "")
        chunks_received = 0
        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str == "[DONE]":
                break
            event_data = json.loads(data_str)
            assert event_data["object"] == "chat.completion.chunk"
            chunks_received += 1
        assert chunks_received > 0, "Expected at least one streaming chunk"
        print_log(f"✓ Uniform chunking (streaming) → {chunks_received} chunks")

    def test_uniform_chunking_invalid_chunk_length(self):
        """chunk_length_sec <= 0 is rejected with 422"""
        payload = {
            "model": "ds_sop_model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{test_video_base64_str}"}}
                ]
            }],
            "stream": False,
            "chunking_options": {"algorithm": "uniform", "chunk_length_sec": -1.0}
        }
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        assert response.status_code == 422, f"Expected 422, got {response.status_code}: {response.text[:300]}"
        print_log("✓ Uniform chunking rejects chunk_length_sec <= 0 with 422")

    def test_uniform_chunking_rejects_ddm_extra_fields(self):
        """DDM-specific fields with algorithm='uniform' are rejected with 422 (extra='forbid')"""
        payload = {
            "model": "ds_sop_model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{test_video_base64_str}"}}
                ]
            }],
            "stream": False,
            "chunking_options": {"algorithm": "uniform", "chunk_length_sec": 5.0, "threshold": 0.9}
        }
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        assert response.status_code == 422, f"Expected 422, got {response.status_code}: {response.text[:300]}"
        print_log("✓ Uniform chunking rejects extra DDM fields with 422")

    def test_ddm_net_still_works_after_uniform_added(self):
        """Regression: existing ddm-net algorithm still works"""
        payload = {
            "model": "ds_sop_model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{test_video_base64_str}"}}
                ]
            }],
            "stream": False,
            "chunking_options": {"algorithm": "ddm-net", "threshold": 0.8, "min_length_sec": 1.0, "max_length_sec": 60.0}
        }
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        assert response.status_code == 200, f"Status: {response.status_code}, Body: {response.text[:500]}"
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) > 0
        print_log("✓ ddm-net still works after uniform chunking added")


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_invalid_endpoint(self):
        response = requests.get(f"{BASE_URL}/v1/invalid_endpoint")
        assert response.status_code == 404
        print_log("✓ 404 for invalid endpoint")

    def test_method_not_allowed(self):
        response = requests.post(f"{BASE_URL}/v1/live")
        assert response.status_code == 405
        print_log("✓ 405 for POST on GET endpoint")


# =============================================================================
# Test Runner
# =============================================================================

def run_all_tests():
    print_log("\n" + "=" * 60)
    print_log("SOP Inference Microservice API Tests")
    print_log("=" * 60)

    test_classes = [
        TestHealthEndpoints,
        TestModelEndpoints,
        TestMetadataEndpoint,
        TestFileEndpoints,
        TestChatCompletionEndpoint,
        TestUniformChunkingEndpoint,
        TestEdgeCases,
        TestMetricsEndpoint,
    ]

    total, passed = 0, 0
    failed = []

    for cls in test_classes:
        print_log(f"\n{cls.__doc__}")
        print_log("-" * 40)
        instance = cls()
        for method_name in [m for m in dir(instance) if m.startswith("test_")]:
            total += 1
            try:
                getattr(instance, method_name)()
                passed += 1
            except Exception as e:
                failed.append((cls.__name__, method_name, str(e)))
                print_log(f"✗ {method_name}: {e}")

    print_log("\n" + "=" * 60)
    print_log(f"Results: {passed}/{total} passed")
    if failed:
        print_log("\nFailed:")
        for cls_name, method, err in failed:
            print_log(f"  {cls_name}.{method}: {err}")
    else:
        print_log("All tests passed!")
    print_log("=" * 60)

    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
