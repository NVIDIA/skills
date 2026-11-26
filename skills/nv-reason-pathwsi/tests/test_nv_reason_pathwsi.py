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

"""Offline wrapper contracts; fake CLIs here do not establish model inference."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
import yaml
from PIL import Image

SKILL_DIR = Path(__file__).resolve().parents[1]
SCRIPT = SKILL_DIR / "scripts" / "run_nv_reason_pathwsi.py"
FIXTURE = SKILL_DIR / "fixtures" / "synthetic_pathology_input.json"
SCHEMA = json.loads((SKILL_DIR / "validators" / "output_schema.json").read_text())
spec = importlib.util.spec_from_file_location("pathwsi_wrapper", SCRIPT)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def invoke(args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def make_image(path):
    Image.new("RGB", (32, 24), "pink").save(path)
    return path


def test_synthetic_cli_json_and_artifact_binding(tmp_path):
    out = tmp_path / "output"
    result = invoke([FIXTURE, "--mock", "--out-dir", out])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    jsonschema.validate(payload, SCHEMA)
    assert payload == json.loads((out / "result.json").read_text())
    assert payload["runtime"]["mode"] == "mock"
    assert payload["runtime"]["command"] == []
    assert payload["runtime"]["upstream_entrypoint_sha256"] is None
    assert payload["output"]["response_text"].startswith("MOCK:")
    image = Path(payload["input"]["image"]["path"])
    assert hashlib.sha256(image.read_bytes()).hexdigest() == payload["input"]["image"]["sha256"]
    response = Path(payload["output"]["response_path"])
    assert hashlib.sha256(response.read_bytes()).hexdigest() == payload["output"]["response_sha256"]
    assert payload["output"]["text_chars"] == len(payload["output"]["response_text"])
    assert not (out / "cache").exists()


@pytest.mark.parametrize("suffix,expected", [(".png", "png"), (".jpg", "jpeg"), (".JPEG", "jpeg")])
def test_relative_json_image_and_prompt_override(tmp_path, suffix, expected):
    image = make_image(tmp_path / ("patch" + suffix))
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"image": image.name, "prompt": "fixture prompt"}))
    result = invoke(
        [request, "--mock", "--prompt", "override prompt", "--out-dir", tmp_path / "out"]
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["input"]["prompt"] == "override prompt"
    assert payload["input"]["image"]["path"] == str(image)
    assert payload["input"]["image"]["format"] == expected
    assert payload["input"]["image"]["width"] == 32
    assert payload["input"]["image"]["height"] == 24


@pytest.mark.parametrize(
    "request_data",
    [
        [],
        {"image": ""},
        {"image": 7},
        {"image": "x.png", "mock": "false"},
        {"image": "x.png", "prompt": " "},
        {"image": "x.png", "case_id": None},
    ],
)
def test_invalid_request_fails_without_success_artifact(tmp_path, request_data):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(request_data))
    out = tmp_path / "out"
    result = invoke([path, "--out-dir", out])
    assert result.returncode == 2
    assert not result.stdout.strip()
    assert not (out / "result.json").exists()


@pytest.mark.parametrize(
    "name,content,error",
    [
        (
            "pointer.png",
            b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n",
            "Git LFS pointer",
        ),
        ("broken.png", b"not an image", "Invalid or unsafe image"),
        ("slide.svs", b"not a supported slide", "Only PNG/JPEG"),
    ],
)
def test_invalid_image_and_lfs_pointer(tmp_path, name, content, error):
    path = tmp_path / name
    path.write_bytes(content)
    result = invoke([path, "--mock", "--out-dir", tmp_path / "out"])
    assert result.returncode == 2
    assert error in result.stderr


def test_live_cannot_downgrade_fixture_to_mock(tmp_path):
    result = invoke([FIXTURE, "--live", "--out-dir", tmp_path / "out"])
    assert result.returncode == 2
    assert "refuses a mock request" in result.stderr


def test_generated_fixture_requires_mock(tmp_path):
    path = tmp_path / "live.json"
    path.write_text(json.dumps({"image": "generated://synthetic_rgb"}))
    result = invoke([path, "--out-dir", tmp_path / "out"])
    assert result.returncode == 2
    assert "mock-only" in result.stderr


def test_existing_evidence_preserved(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    original = out / "result.json"
    original.write_text("prior evidence")
    result = invoke([FIXTURE, "--mock", "--out-dir", out])
    assert result.returncode == 2
    assert "fresh --out-dir" in result.stderr
    assert original.read_text() == "prior evidence"


@pytest.mark.parametrize(
    "flag,value", [("--max-new-tokens", "0"), ("--timeout", "nan"), ("--timeout", "0")]
)
def test_invalid_limits(tmp_path, flag, value):
    result = invoke([FIXTURE, "--out-dir", tmp_path / "out", flag, value])
    assert result.returncode == 2
    assert "positive finite" in result.stderr


def fake_cli(tmp_path, content):
    root = tmp_path / "fake upstream"
    root.mkdir()
    (root / "inference.py").write_text(content)
    return root


def live_args(tmp_path, root):
    image = make_image(tmp_path / "input tile.png")
    return [str(image), "--live", "--upstream-dir", str(root), "--out-dir", str(tmp_path / "out")]


def test_cli_forwarding_offline_cache_and_stdout_contract(tmp_path, monkeypatch, capsys):
    root = fake_cli(
        tmp_path,
        """import json, os, sys
print(json.dumps({'argv': sys.argv[1:], 'offline': os.environ['HF_HUB_OFFLINE'],
                  'cache': os.environ['HF_HOME']}))
""",
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    original_env = dict(wrapper.os.environ)
    prompt = 'Describe "tissue"; $(do_not_execute)'
    args = live_args(tmp_path, root) + [
        "--model",
        "local-model",
        "--prompt",
        prompt,
        "--think",
        "--max-new-tokens",
        "128",
        "--device-map",
        "cpu",
    ]
    assert wrapper.main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    jsonschema.validate(payload, SCHEMA)
    observed = json.loads(payload["output"]["response_text"])
    assert observed["argv"] == [
        "--model",
        "local-model",
        "--image",
        str(tmp_path / "input tile.png"),
        "--prompt",
        prompt,
        "--max-new-tokens",
        "128",
        "--think",
        "--device-map",
        "cpu",
    ]
    assert observed["offline"] == "1"
    assert Path(observed["cache"]).is_relative_to(tmp_path / "out")
    assert payload["runtime"]["mode"] == "upstream_cli"
    assert payload["runtime"]["mock"] is False
    assert payload["runtime"]["truncation_status"] == "unknown"
    assert dict(wrapper.os.environ) == original_env


@pytest.mark.parametrize(
    "content,error",
    [
        ("import sys\nprint('failed', file=sys.stderr)\nsys.exit(7)\n", "exited 7"),
        ("print('   ')\n", "empty text"),
        ("import time\ntime.sleep(5)\n", "timed out"),
    ],
)
def test_upstream_failure_no_mock_fallback(tmp_path, monkeypatch, capsys, content, error):
    root = fake_cli(tmp_path, content)
    args = live_args(tmp_path, root) + ["--timeout", "0.5"]
    assert wrapper.main(args) == 2
    captured = capsys.readouterr()
    assert error in captured.err
    assert not captured.out.strip()
    assert not (tmp_path / "out" / "result.json").exists()
    assert (tmp_path / "out" / "upstream.stderr.log").exists()


def test_tokens_redacted_from_saved_logs(tmp_path, monkeypatch, capsys):
    root = fake_cli(
        tmp_path,
        """import os, sys
print(os.environ['HF_TOKEN'])
print(os.environ['HF_TOKEN'], file=sys.stderr)
sys.exit(1)
""",
    )
    monkeypatch.setenv("HF_TOKEN", "test-only-secret-sentinel")
    assert wrapper.main(live_args(tmp_path, root) + ["--allow-downloads"]) == 2
    for path in (tmp_path / "out").glob("upstream.*"):
        assert "test-only-secret-sentinel" not in path.read_text()
        assert "<redacted>" in path.read_text()
    assert "test-only-secret-sentinel" not in str(capsys.readouterr())


@pytest.mark.parametrize("allow_downloads", [False, True])
def test_upstream_receives_only_scoped_environment(tmp_path, monkeypatch, capsys, allow_downloads):
    root = fake_cli(
        tmp_path,
        """import json, os
names = ['CI_JOB_TOKEN', 'UNRELATED_API_KEY', 'PYTHONPATH', 'LD_LIBRARY_PATH',
         'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'HF_TOKEN', 'HTTPS_PROXY',
         'SSL_CERT_FILE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']
print(json.dumps({name: os.environ.get(name) for name in names}))
""",
    )
    monkeypatch.setenv("CI_JOB_TOKEN", "unrelated-ci-sentinel")
    monkeypatch.setenv("UNRELATED_API_KEY", "unrelated-api-sentinel")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "untrusted_imports"))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("HF_TOKEN", "test-only-hf-sentinel")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.invalid:8080")
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "ca.pem"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    original_env = dict(wrapper.os.environ)
    args = live_args(tmp_path, root)
    if allow_downloads:
        args.append("--allow-downloads")
    assert wrapper.main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    observed = json.loads(payload["output"]["response_text"])
    assert all(
        observed[name] is None for name in ["CI_JOB_TOKEN", "UNRELATED_API_KEY", "PYTHONPATH"]
    )
    assert observed["LD_LIBRARY_PATH"] == "/usr/local/cuda/lib64"
    assert observed["CUDA_VISIBLE_DEVICES"] == "0"
    assert observed["OMP_NUM_THREADS"] == "2"
    # Logs redact a forwarded HF token before it can become response evidence.
    assert observed["HF_TOKEN"] == ("<redacted>" if allow_downloads else None)
    assert observed["HTTPS_PROXY"] == (
        "http://proxy.example.invalid:8080" if allow_downloads else None
    )
    assert observed["SSL_CERT_FILE"] == (str(tmp_path / "ca.pem") if allow_downloads else None)
    assert observed["HF_HUB_OFFLINE"] == observed["TRANSFORMERS_OFFLINE"] == "1"
    assert dict(wrapper.os.environ) == original_env


def test_live_requires_upstream_without_guessing(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("NV_REASON_PATHWSI_ROOT", raising=False)
    image = make_image(tmp_path / "tile.png")
    assert wrapper.main([str(image), "--live", "--out-dir", str(tmp_path / "out")]) == 2
    assert "--upstream-dir" in capsys.readouterr().err


def test_setup_report_has_no_model_or_network_probe(tmp_path):
    result = invoke(
        [
            "--check-setup",
            "--model",
            str(tmp_path / "missing-model"),
            "--out-dir",
            tmp_path / "unused",
        ]
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mode"] == "setup"
    assert payload["network_access_checked"] is False
    assert payload["local_model_directory"] is False
    assert not (tmp_path / "unused").exists()


def test_schema_rejects_inconsistent_mock_label(tmp_path):
    result = invoke([FIXTURE, "--mock", "--out-dir", tmp_path / "out"])
    payload = json.loads(result.stdout)
    payload["runtime"]["mock"] = False
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, SCHEMA)


def test_setup_reports_missing_packages_without_version_gates():
    versions = {
        "torch": "99.0",
        "torchvision": None,
        "transformers": "1.0",
        "accelerate": "0.1",
        "Pillow": None,
    }
    assert wrapper.missing_packages(versions) == ["torchvision", "Pillow"]
    ready, step = wrapper.readiness(["torchvision"], "inference.py", True, None)
    assert ready is False and "torchvision" in step


def git_cmd(cwd, *args):
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=noreply@nvidia.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(cwd),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()


def test_git_commit_ignores_enclosing_repository(tmp_path):
    outer = tmp_path / "outer"
    copied = outer / "vendor" / "upstream"
    copied.mkdir(parents=True)
    git_cmd(outer, "init", "-q")
    git_cmd(outer, "commit", "-q", "--allow-empty", "-m", "outer")
    assert wrapper.git_commit(copied) is None
    git_cmd(copied, "init", "-q")
    git_cmd(copied, "commit", "-q", "--allow-empty", "-m", "upstream")
    assert wrapper.git_commit(copied) == git_cmd(copied, "rev-parse", "HEAD")


def write_snapshot(root, stop_ids):
    root.mkdir()
    for name in ("config.json", "processor_config.json"):
        (root / name).write_text("{}")
    (root / "chat_template.jinja").write_text("{{ messages }}")
    (root / "generation_config.json").write_text(json.dumps({"eos_token_id": stop_ids}))
    (root / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>"}))
    (root / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 7, "content": "<|endoftext|>"},
                    {"id": 9, "content": "<|im_end|>"},
                ]
            }
        )
    )
    (root / "model.safetensors").write_bytes(b"\0" * 16)
    return root


@pytest.mark.parametrize("stop_ids,stops", [(7, False), ([9, 7], True)])
def test_setup_reports_snapshot_stop_tokens(tmp_path, stop_ids, stops):
    result = invoke(["--check-setup", "--model", write_snapshot(tmp_path / "model", stop_ids)])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["local_model"]["missing"] == []
    assert payload["local_model"]["stops_at_tokenizer_eos"] is stops
    assert bool(payload["warnings"]) is not stops


def test_readiness_flags_incomplete_snapshot_and_unverifiable_model_id(tmp_path):
    model = write_snapshot(tmp_path / "model", [9])
    (model / "processor_config.json").unlink()
    (model / "model.safetensors").write_bytes(wrapper.LFS_POINTER_PREFIX + b"\noid sha256:abc\n")
    snapshot = wrapper.local_model_report(str(model))
    assert snapshot["missing"] == ["processor", "weights"]
    ready, step = wrapper.readiness([], "inference.py", True, snapshot)
    assert ready is False and "processor, weights" in step
    assert wrapper.readiness([], "inference.py", True, None)[0] is None
    complete = wrapper.local_model_report(str(write_snapshot(tmp_path / "complete", [9])))
    assert wrapper.readiness([], "inference.py", True, complete)[0] is True


def test_eval_attachment_matches_runtime_fixture():
    # Tier 3 reads attachments from evals/files; keep that copy identical to the fixture.
    evals = json.loads((SKILL_DIR / "evals" / "evals.json").read_text())
    attached = [name for case in evals["evals"] for name in case["files"]]
    assert attached == ["files/synthetic_pathology_input.json"]
    assert (SKILL_DIR / "evals" / attached[0]).read_bytes() == FIXTURE.read_bytes()


def test_skill_pins_no_package_versions():
    # The tested environment lives in SKILL.md prose; nothing here needs version bumps.
    manifest = yaml.safe_load((SKILL_DIR / "skill_manifest.yaml").read_text())
    runtime = manifest["runtime"]
    declared = [
        *runtime["dependencies"].values(),
        *runtime["side_effects"]["pip_packages"],
        *manifest["validation"]["env_pin"].values(),
    ]
    assert not (SKILL_DIR / "requirements.txt").exists()
    assert not any("==" in spec or "<" in spec for spec in declared)
    assert manifest["validation"]["env_pin"] == {"Pillow": runtime["dependencies"]["Pillow"]}
