# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained offline contract tests; no CT generator, model, or downloads.

Test doubles exercise loader/inference boundaries. Real-volume and CUDA parity
checks are explicitly opt-in in test_upstream_examples.py, never fallbacks.
"""

import hashlib
import json
import runpy
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parent.parent
SCRIPT = SKILL_DIR / "scripts" / "run_nv_reason_ct.py"
TEST_COMMIT = "a" * 40


def _validate(payload):
    schema = json.loads((SKILL_DIR / "validators/output_schema.json").read_text())
    jsonschema.validate(payload, schema)


@pytest.fixture
def ready_setup(wrapper, monkeypatch):
    monkeypatch.setattr(wrapper, "_installed_version", lambda _: "test-build")
    monkeypatch.setattr(
        wrapper,
        "_package_status",
        lambda name, _: {
            "installed": True,
            "importable": True,
            "version": wrapper._installed_version(name),
        },
    )
    monkeypatch.setattr(
        wrapper, "_cuda_report", lambda: {"available": True, "bfloat16_supported": True}
    )
    return wrapper


@pytest.fixture
def inference_double(wrapper, monkeypatch):
    calls = []
    runtime = {
        "device": "cuda",
        "torch_dtype": "bfloat16",
        "transformers_version": "test",
        "torch_version": "test",
        "generated_tokens": 1,
        "truncated_by_max_new_tokens": False,
    }

    def infer(**kwargs):
        calls.append(kwargs)
        return "test-double response", {
            **runtime,
            "resolved_revision": TEST_COMMIT if kwargs["revision"] else None,
        }

    monkeypatch.setattr(wrapper, "_run_transformers_inference", infer)
    return calls, runtime


def _snapshot(tmp_path, wrapper, missing=(), shards=False):
    snapshot = tmp_path / "snapshots" / TEST_COMMIT
    names = set(wrapper.REQUIRED_MODEL_FILES) | {
        "model.safetensors.index.json" if shards else "model.safetensors"
    }
    for name in names - set(missing):
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    return snapshot


def _stub_snapshot(monkeypatch, snapshot):
    def download(model_id, **kwargs):
        assert model_id == "nvidia/NV-Reason-CT"
        assert kwargs["local_files_only"] is True and kwargs["token"] is False
        if snapshot is None or kwargs["revision"] not in ("main", TEST_COMMIT):
            raise FileNotFoundError("requested revision is not cached")
        return str(snapshot)

    # No scan_cache_dir: setup must inspect only this snapshot.
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_dependency_inventory_matches_import_allowlist_without_version_constraints(
    wrapper,
):
    manifest = yaml.safe_load((SKILL_DIR / "skill_manifest.yaml").read_text())
    runtime = manifest["runtime"]
    expected = {name.lower().replace("_", "-") for name in wrapper.DEPENDENCY_IMPORTS}
    assert {name.lower().replace("_", "-") for name in runtime["dependencies"]} == expected
    assert {
        name.lower().replace("_", "-") for name in runtime["side_effects"]["pip_packages"]
    } == expected
    assert set(runtime["dependencies"].values()) == {"*"}
    assert not manifest["validation"].get("env_pin")
    assert runtime["external_assets"][0]["installation_url"] == wrapper.UPSTREAM_SETUP_URL


def test_eval_fixtures_resolve_inside_evals():
    evals_dir = SKILL_DIR / "evals"
    dataset = json.loads((evals_dir / "evals.json").read_text())
    for case in dataset["evals"]:
        for name in case.get("files", []):
            path = Path(name)
            assert not path.is_absolute() and ".." not in path.parts
            resolved = (evals_dir / path).resolve()
            assert resolved.is_relative_to(evals_dir.resolve()) and resolved.is_file()


@pytest.mark.parametrize("json_request", [False, True])
def test_cli_input_contract(
    wrapper, volume_double, inference_double, tmp_path, capsys, json_request
):
    volume, image = volume_double
    original = volume.read_bytes()
    input_path = volume
    if json_request:
        input_path = tmp_path / "request.json"
        input_path.write_text(
            json.dumps(
                {
                    "volume_path": volume.name,
                    "prompt": "request prompt",
                    "anatomy_region": "chest",
                    "enable_thinking": True,
                }
            )
        )
    assert (
        wrapper.main(
            [
                str(input_path),
                "--trust-model-code",
                "--anatomy-region",
                "abdomen",
                "--no-thinking",
                "--prompt",
                "caller prompt",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert not captured.err
    payload = json.loads(captured.out)
    _validate(payload)
    assert payload["input"]["prompt"] == "caller prompt"
    assert payload["input"]["anatomy_region"] == "abdomen"
    assert payload["input"]["enable_thinking"] is False
    metadata = payload["input"]["volume"]
    assert metadata["source"] == ("fixture_file" if json_request else "file")
    assert metadata["shape"] == list(image.shape)
    assert metadata["spacing_mm"] == list(image.header.get_zooms())
    assert metadata["sha256"] == hashlib.sha256(original).hexdigest()
    assert volume.read_bytes() == original and not (tmp_path / "out").exists()
    assert inference_double[0][0]["volume_path"] == volume
    assert inference_double[0][0]["trust_model_code"] is True


@pytest.mark.parametrize(
    "region,prompt",
    [
        ("chest", "write a structured chest CT report"),
        ("abdomen", "write a structured abdominal CT report"),
        ("none", "Describe this CT volume."),
    ],
)
def test_default_prompt(wrapper, volume_double, region, prompt):
    spec = wrapper._load_input(volume_double[0], None, region, None)
    assert spec.prompt == prompt and spec.enable_thinking is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("mock", True),
        ("mock_response", "caller text"),
        ("unknown", 1),
        ("trust_model_code", True),
    ],
)
def test_unknown_request_fields_are_rejected(wrapper, tmp_path, capsys, field, value):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"volume_path": "missing.nii", field: value}))
    assert wrapper.main([str(request), "--trust-model-code"]) == 2
    assert "unsupported request fields" in capsys.readouterr().err


def test_legacy_environment_cannot_bypass_inference(
    wrapper, volume_double, inference_double, monkeypatch, capsys
):
    monkeypatch.setenv("MOCK_NV_REASON_CT", "1")
    assert wrapper.main([str(volume_double[0]), "--trust-model-code"]) == 0
    assert len(inference_double[0]) == 1
    assert json.loads(capsys.readouterr().out)["runtime"]["mock"] is False


@pytest.mark.parametrize("args", [["--mock"], ["--device", "auto"]])
def test_removed_cli_switches_fail_explicitly(args):
    proc = _run(*args)
    assert proc.returncode == 2 and "unrecognized arguments" in proc.stderr


@pytest.mark.parametrize("content", [b"\xff\xfe", b"{broken", b"[]"])
def test_invalid_json_request_fails_cleanly(tmp_path, content):
    request = tmp_path / "request.json"
    request.write_bytes(content)
    out = tmp_path / "unused-output"
    proc = _run(request, "--out-dir", out, "--trust-model-code")
    assert proc.returncode == 2 and "error:" in proc.stderr
    assert "Traceback" not in proc.stderr and not proc.stdout and not out.exists()


@pytest.mark.parametrize("target", ["request", "volume", "hash"])
def test_unreadable_input_fails_cleanly(
    wrapper, volume_double, monkeypatch, tmp_path, capsys, target
):
    volume = volume_double[0]
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"volume_path": str(volume)}))
    original_open = Path.open
    volume_reads = 0

    def open_file(path, *args, **kwargs):
        nonlocal volume_reads
        if path == volume:
            volume_reads += 1
        if (
            (target == "request" and path == request)
            or (target == "volume" and path == volume)
            or (target == "hash" and path == volume and volume_reads == 2)
        ):
            raise PermissionError("test input is unreadable")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    assert wrapper.main([str(request), "--trust-model-code"]) == 2
    captured = capsys.readouterr()
    assert "unreadable" in captured.err and "Traceback" not in captured.err
    assert not captured.out


@pytest.mark.parametrize(
    "name,content,error",
    [
        ("bad.txt", b"text", "expected .nii or .nii.gz"),
        (
            "pointer.nii.gz",
            b"version https://git-lfs.github.com/spec/v1\n",
            "Git LFS pointer",
        ),
        ("bad.nii", b"not an image", "could not read NIfTI volume"),
    ],
)
def test_invalid_volume_fails_cleanly(tmp_path, name, content, error):
    volume = tmp_path / name
    volume.write_bytes(content)
    proc = _run(volume, "--out-dir", tmp_path / "out", "--trust-model-code")
    assert proc.returncode == 2 and error in proc.stderr
    assert "Traceback" not in proc.stderr and not proc.stdout
    assert not (tmp_path / "out").exists()


def test_missing_input_fails_cleanly(tmp_path):
    proc = _run(tmp_path / "missing.nii.gz", "--out-dir", tmp_path / "out", "--trust-model-code")
    assert proc.returncode == 2 and "input not found" in proc.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("shape", [(4, 5, 6, 2), (4, 5), (4, 0, 6)])
def test_invalid_dimensions(wrapper, volume_double, shape):
    volume, image = volume_double
    image.shape = shape
    with pytest.raises(wrapper.SkillError, match="requires one 3D NIfTI volume"):
        wrapper._volume_info(volume)


@pytest.mark.parametrize("spacing", [(0, 1, 2), (-1, 1, 2), (float("nan"), 1, 2), (1, 2)])
def test_invalid_spacing(wrapper, volume_double, spacing):
    volume, image = volume_double
    image.header.get_zooms = lambda: spacing
    with pytest.raises(wrapper.SkillError, match="three positive values"):
        wrapper._volume_info(volume)


def test_invalid_affine(wrapper, volume_double):
    volume, image = volume_double
    image.affine[0, 0] = np.inf
    with pytest.raises(wrapper.SkillError, match="non-finite"):
        wrapper._volume_info(volume)


@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("out_kind", ["new-directory", "existing-file", "permission-error"])
def test_completion_and_output_filesystem_contract(
    wrapper,
    volume_double,
    inference_double,
    tmp_path,
    capsys,
    monkeypatch,
    truncated,
    out_kind,
):
    inference_double[1]["truncated_by_max_new_tokens"] = truncated
    out = tmp_path / "out"
    if out_kind == "existing-file":
        out.write_text("preserve me")
    if out_kind == "permission-error":
        original = Path.mkdir

        def mkdir(path, *args, **kwargs):
            if path == out:
                raise PermissionError("test output is not writable")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", mkdir)
    code = wrapper.main(
        [
            str(volume_double[0]),
            "--out-dir",
            str(out),
            "--max-new-tokens",
            "1",
            "--trust-model-code",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    _validate(payload)
    assert code == (3 if truncated else 0)
    assert payload["runtime"]["truncated_by_max_new_tokens"] is truncated
    assert "Traceback" not in captured.err
    if truncated and out_kind == "new-directory":
        partial = Path(payload["output"]["partial_json_path"])
        assert partial.parent == out and json.loads(partial.read_text()) == payload
        assert str(partial) in captured.err
    else:
        assert "partial_json_path" not in payload["output"]
        assert ("retain stdout" in captured.err) is truncated
    if out_kind == "existing-file":
        assert out.read_text() == "preserve me"
    elif not truncated:
        assert not out.exists()
    checks = yaml.safe_load((SKILL_DIR / "skill_manifest.yaml").read_text())["validation"][
        "sanity_checks"
    ]
    assert {"path": "runtime.truncated_by_max_new_tokens", "eq": False} in checks
    for invalid_revision in (None, "main"):
        payload["runtime"]["resolved_revision"] = invalid_revision
        with pytest.raises(jsonschema.ValidationError):
            _validate(payload)


def test_partial_results_do_not_overwrite_previous_attempts(wrapper, tmp_path):
    payload = {"output": {"response_text": "first attempt"}}
    wrapper._preserve_partial_result(payload, tmp_path)
    first_path = Path(payload["output"]["partial_json_path"])
    first_bytes = first_path.read_bytes()
    payload["output"]["response_text"] = "second attempt"
    wrapper._preserve_partial_result(payload, tmp_path)
    assert Path(payload["output"]["partial_json_path"]) != first_path
    assert first_path.read_bytes() == first_bytes


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "recommendation",
    [
        "ready_for_live_cuda_inference",
        "install_or_repair_upstream_dependencies",
        "use_a_cuda_host",
        "use_a_cuda_gpu_with_bfloat16_support",
        "stage_complete_model_and_processor_assets",
    ],
)
def test_setup_readiness_exit_contract(wrapper, monkeypatch, capsys, strict, recommendation):
    report = {"skill": "nv_reason_ct", "setup": {"recommendation": recommendation}}
    monkeypatch.setattr(wrapper, "_setup_report", lambda *_: report)
    monkeypatch.setattr(
        wrapper,
        "_require_model_code_consent",
        lambda *_: pytest.fail("setup must not require code consent"),
    )
    args = ["--check-setup"] + (["--fail-on-not-ready"] if strict else [])
    assert wrapper.main(args) == int(strict and recommendation != "ready_for_live_cuda_inference")
    assert json.loads(capsys.readouterr().out) == report


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("input_kind", ["volume", "request"])
def test_cli_refuses_model_code_without_explicit_flag(
    wrapper, monkeypatch, tmp_path, capsys, local, input_kind
):
    # Neither environment settings nor a request-file claim can authorize code.
    monkeypatch.setenv("TRUST_REMOTE_CODE", "true")
    monkeypatch.setenv("NV_REASON_CT_TRUST_MODEL_CODE", "1")
    model = tmp_path / "model"
    model.mkdir()
    input_path = tmp_path / "input.nii.gz"
    if input_kind == "request":
        input_path = tmp_path / "request.json"
        input_path.write_text(json.dumps({"volume_path": "input.nii.gz", "trust_model_code": True}))
    for name in (
        "_load_input",
        "_resolve_model_revision",
        "_run_transformers_inference",
    ):
        monkeypatch.setattr(
            wrapper,
            name,
            lambda *_, **__: pytest.fail("consent must precede input/model work"),
        )
    out = tmp_path / "out"
    args = [str(input_path), "--out-dir", str(out), "--local-files-only"]
    args += ["--model", str(model)] if local else ["--revision", TEST_COMMIT]
    assert wrapper.main(args) == 2
    captured = capsys.readouterr()
    assert "--trust-model-code" in captured.err and "permissions" in captured.err
    assert not captured.out and "Traceback" not in captured.err
    assert not out.exists()


@pytest.mark.parametrize("revision", [None, TEST_COMMIT])
@pytest.mark.parametrize("consent", [{}, {"trust_model_code": False}])
def test_direct_inference_call_is_also_fail_closed(
    wrapper, monkeypatch, tmp_path, revision, consent
):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setattr(
        wrapper,
        "_resolve_model_revision",
        lambda *_, **__: pytest.fail("must not resolve assets"),
    )
    with pytest.raises(wrapper.SkillError, match="--trust-model-code"):
        wrapper._run_transformers_inference(
            volume_path=tmp_path / "input.nii.gz",
            prompt="test",
            anatomy_region="chest",
            enable_thinking=True,
            model_id=str(tmp_path) if revision is None else wrapper.DEFAULT_MODEL,
            revision=revision,
            max_new_tokens=3,
            local_files_only=True,
            **consent,
        )


def test_default_manifest_does_not_grant_model_code_consent(wrapper, tmp_path):
    manifest = yaml.safe_load((SKILL_DIR / "skill_manifest.yaml").read_text())
    args = manifest["runtime"]["args"]
    assert "--trust-model-code" not in args
    assert (
        wrapper._build_parser().parse_args([str(tmp_path / "input.nii.gz")]).trust_model_code
        is False
    )


@pytest.mark.parametrize(
    "args,error",
    [
        (["--fail-on-not-ready"], "requires --check-setup"),
        (["--max-new-tokens", "0"], "must be >= 1"),
        (["--model", ""], "must not be empty"),
        (["--revision", ""], "must not be empty"),
        ([], "volume_or_fixture is required"),
    ],
)
def test_argument_errors(wrapper, capsys, args, error):
    assert wrapper.main(args) == 2
    assert error in capsys.readouterr().err


@pytest.mark.parametrize("distribution,module", [("os", "os"), ("torch", "os")])
def test_setup_imports_reject_undocumented_modules(wrapper, distribution, module):
    with pytest.raises(wrapper.SkillError, match="documented dependencies"):
        wrapper._package_status(distribution, module)


def test_setup_imports_allow_documented_dependency(wrapper, monkeypatch):
    imported = []
    monkeypatch.setattr(wrapper, "_installed_version", lambda _: "25.0")
    monkeypatch.setattr(wrapper.importlib, "import_module", imported.append)
    assert wrapper._package_status("numpy", "numpy")["importable"] is True
    assert imported == ["numpy"]


@pytest.mark.parametrize("revision", ["main", TEST_COMMIT, "missing"])
def test_setup_uses_selected_snapshot_only(ready_setup, monkeypatch, tmp_path, revision):
    snapshot = _snapshot(tmp_path, ready_setup)
    _stub_snapshot(monkeypatch, snapshot)
    setup = ready_setup._setup_report(ready_setup.DEFAULT_MODEL, revision)["setup"]
    assert setup["model_cache"]["complete"] is (revision != "missing")
    assert (setup["recommendation"] == "ready_for_live_cuda_inference") is (revision != "missing")


def test_empty_cache_is_not_ready(ready_setup, monkeypatch):
    _stub_snapshot(monkeypatch, None)
    cache = ready_setup._setup_report(ready_setup.DEFAULT_MODEL, "main")["setup"]["model_cache"]
    assert cache["complete"] is False and cache["cached"] is False
    assert "config.json" in cache["missing_files"]


@pytest.mark.parametrize(
    "missing",
    [
        "config.json",
        "processor.py",
        "tokenizer.json",
        "chat_template.jinja",
        "image_processor_3d/preprocessor_config.json",
        "model.safetensors",
    ],
)
def test_other_snapshot_cannot_complete_selected_assets(
    ready_setup, monkeypatch, tmp_path, missing
):
    selected = _snapshot(tmp_path, ready_setup, missing=[missing])
    _snapshot(tmp_path / "other-cache", ready_setup)
    _stub_snapshot(monkeypatch, selected)
    cache = ready_setup._model_cache_report(ready_setup.DEFAULT_MODEL, "main")
    assert cache["complete"] is False and cache["missing_files"] == [missing]


@pytest.mark.parametrize("complete", [False, True])
def test_every_indexed_weight_shard_is_required(ready_setup, monkeypatch, tmp_path, complete):
    snapshot = _snapshot(tmp_path, ready_setup, shards=True)
    shards = ["model-00001.safetensors", "model-00002.safetensors"]
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer1": shards[0], "layer2": shards[1]}})
    )
    for name in shards if complete else shards[:1]:
        (snapshot / name).write_bytes(b"unit-test weights placeholder")
    _stub_snapshot(monkeypatch, snapshot)
    cache = ready_setup._model_cache_report(ready_setup.DEFAULT_MODEL, "main")
    assert cache["complete"] is complete
    assert cache["missing_files"] == ([] if complete else shards[1:])


@pytest.mark.parametrize(
    "index", ["not json", "[]", '{"weight_map": {}}', '{"weight_map": {"x": 2}}']
)
def test_invalid_weight_index_is_not_ready(ready_setup, monkeypatch, tmp_path, index):
    snapshot = _snapshot(tmp_path, ready_setup, shards=True)
    (snapshot / "model.safetensors.index.json").write_text(index)
    _stub_snapshot(monkeypatch, snapshot)
    cache = ready_setup._model_cache_report(ready_setup.DEFAULT_MODEL, "main")
    assert cache["complete"] is False and "could not inspect" in cache["error"]


@pytest.mark.parametrize(
    "package,installed,importable",
    [
        ("torch", True, False),
        ("nibabel", False, False),
        ("huggingface-hub", False, False),
        ("monai", True, False),
    ],
)
def test_missing_or_broken_dependency_blocks_setup(
    ready_setup, monkeypatch, tmp_path, package, installed, importable
):
    original = ready_setup._package_status
    monkeypatch.setattr(
        ready_setup,
        "_package_status",
        lambda name, module: (
            {
                "installed": installed,
                "importable": importable,
                "version": "test-build" if installed else None,
            }
            if name == package
            else original(name, module)
        ),
    )
    _stub_snapshot(monkeypatch, _snapshot(tmp_path, ready_setup))
    setup = ready_setup._setup_report(ready_setup.DEFAULT_MODEL, "main")["setup"]
    assert setup["recommendation"] == "install_or_repair_upstream_dependencies"
    assert setup["dependencies"][package]["importable"] is False
    assert setup["dependency_setup_url"] == ready_setup.UPSTREAM_SETUP_URL


@pytest.mark.parametrize("version", ["1.0", "99.0", "99.0+test"])
def test_versions_are_reported_without_enforcing_a_skill_constraint(
    ready_setup, monkeypatch, tmp_path, version
):
    monkeypatch.setattr(ready_setup, "_installed_version", lambda _: version)
    _stub_snapshot(monkeypatch, _snapshot(tmp_path, ready_setup))
    setup = ready_setup._setup_report(ready_setup.DEFAULT_MODEL, "main")["setup"]
    assert setup["recommendation"] == "ready_for_live_cuda_inference"
    assert setup["version_constraints_checked"] is False
    assert setup["model_code_opt_in_required"] is True
    assert {value["version"] for value in setup["dependencies"].values()} == {version}
    assert set(ready_setup._environment_packages().values()) == {version}


def test_missing_inference_api_points_to_upstream_setup(wrapper, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace())
    with pytest.raises(wrapper.SkillError, match="upstream inference dependencies") as error:
        wrapper._run_transformers_inference(
            volume_path=tmp_path / "ct.nii.gz",
            prompt="test prompt",
            anatomy_region="chest",
            enable_thinking=True,
            model_id=wrapper.DEFAULT_MODEL,
            revision=TEST_COMMIT,
            max_new_tokens=3,
            local_files_only=True,
            trust_model_code=True,
        )
    assert wrapper.UPSTREAM_SETUP_URL in str(error.value)


def test_dependency_import_failure_is_reported(wrapper, monkeypatch):
    monkeypatch.setattr(wrapper, "_installed_version", lambda _: "test-build")

    def broken_import(_):
        raise ImportError("test binary dependency failure")

    monkeypatch.setattr(wrapper.importlib, "import_module", broken_import)
    status = wrapper._package_status("torch", "torch")
    assert status["installed"] is True and status["importable"] is False
    assert status["version"] == "test-build"
    assert "test binary dependency failure" in status["error"]


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("region", ["chest", "none"])
@pytest.mark.parametrize(
    "new_tokens,eos_ids,truncated",
    [
        ([7, 8, 9], [2, 3], True),
        ([7, 8, 2], [2, 3], False),
        ([7, 8, 3], [2, 3], False),
        ([7, 8, 2], 2, False),
        ([7, 2], [2, 3], False),
        ([7, 8, 9], None, True),
    ],
)
def test_inference_loaders_and_generation_contract(
    wrapper,
    monkeypatch,
    tmp_path,
    capsys,
    local,
    region,
    new_tokens,
    eos_ids,
    truncated,
):
    calls = {}
    monkeypatch.setattr(wrapper, "_select_cuda_device", lambda: "cuda")
    monkeypatch.setenv("HF_TOKEN", "unit-test-token")

    def download(model_id, filename, **kwargs):
        assert not local, "local checkpoint must never reach Hub download"
        calls["download"] = (model_id, filename, kwargs)
        return str(tmp_path / "snapshots" / TEST_COMMIT / filename)

    class Inputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

        def to(self, device):
            assert device == "cuda"
            return self

    class Model:
        device = "cuda"
        generation_config = SimpleNamespace(eos_token_id=eos_ids)

        def eval(self):
            return self

        def to(self, device):
            assert device == "cuda"
            return self

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return np.array([[10, 11, *new_tokens]])

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            calls["template"] = (messages, kwargs)
            return "formatted prompt"

        def __call__(self, **kwargs):
            calls["preprocess"] = kwargs
            return Inputs(input_ids=np.array([[10, 11]]))

        def batch_decode(self, tokens, **kwargs):
            assert tokens.tolist() == [new_tokens]
            assert kwargs == {
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            }
            return ["test-double response"]

    def load_model(model_id, **kwargs):
        warning = capsys.readouterr()
        assert not warning.out
        assert "warning:" in warning.err and "--trust-model-code" in warning.err
        assert "not sandboxed" in warning.err and repr(model_id) in warning.err
        assert ("local export" if local else TEST_COMMIT) in warning.err
        calls["model"] = (model_id, kwargs)
        return Model()

    def load_processor(model_id, **kwargs):
        calls["processor"] = (model_id, kwargs)
        return Processor()

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(bfloat16="bfloat16", __version__="test", inference_mode=nullcontext),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            __version__="test",
            AutoModelForImageTextToText=SimpleNamespace(from_pretrained=load_model),
            AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
        ),
    )
    model_id = str(tmp_path / "local-export") if local else wrapper.DEFAULT_MODEL
    text, runtime = wrapper._run_transformers_inference(
        volume_path=tmp_path / "ct.nii",
        prompt="engineering prompt",
        anatomy_region=region,
        enable_thinking=True,
        model_id=model_id,
        revision=None if local else "main",
        max_new_tokens=3,
        local_files_only=True,
        trust_model_code=True,
    )
    assert text == "test-double response"
    if not local:
        assert calls["download"] == (
            model_id,
            "config.json",
            {"revision": "main", "local_files_only": True, "token": "unit-test-token"},
        )
    for name in ("model", "processor"):
        identifier, kwargs = calls[name]
        assert identifier == model_id
        assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is True
        if local:
            assert kwargs["revision"] is None and kwargs["code_revision"] is None
            assert "token" not in kwargs
        else:
            assert kwargs["revision"] == kwargs["code_revision"] == TEST_COMMIT
            assert kwargs["token"] == "unit-test-token"
    assert calls["model"][1]["dtype"] == "bfloat16"
    assert calls["model"][1]["attn_implementation"] == "sdpa"
    assert calls["preprocess"]["images3d"] == [str(tmp_path / "ct.nii")]
    assert calls["preprocess"]["anatomy_region"] == (None if region == "none" else region)
    assert calls["template"][0][0]["content"][1]["text"] == "engineering prompt"
    assert calls["template"][1] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }
    assert calls["generate"]["do_sample"] is False and calls["generate"]["use_cache"] is True
    assert calls["generate"]["max_new_tokens"] == 3
    assert runtime["resolved_revision"] == (None if local else TEST_COMMIT)
    assert runtime["truncated_by_max_new_tokens"] is truncated


def test_unresolvable_revision_fails_cleanly(wrapper, monkeypatch):
    def download(*args, **kwargs):
        raise FileNotFoundError("requested revision is not cached")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    with pytest.raises(wrapper.SkillError, match="could not resolve model revision"):
        wrapper._resolve_model_revision(
            wrapper.DEFAULT_MODEL, "missing", local_files_only=True, token=None
        )


@pytest.mark.parametrize("alias", ["--model", "--model-id"])
def test_local_export_cli_identity(
    wrapper, volume_double, inference_double, tmp_path, monkeypatch, capsys, alias
):
    export = tmp_path / "local export"
    export.mkdir()
    monkeypatch.setenv("NV_REASON_CT_REVISION", "ignored-for-local-export")
    assert wrapper.main([str(volume_double[0]), alias, str(export), "--trust-model-code"]) == 0
    payload = json.loads(capsys.readouterr().out)
    _validate(payload)
    runtime = payload["runtime"]
    assert runtime["model"] == str(export.resolve()) and runtime["model_source"] == "local"
    assert runtime["revision"] is None and runtime["resolved_revision"] is None
    assert runtime["local_files_only"] is True and inference_double[0][0]["revision"] is None
    runtime["resolved_revision"] = TEST_COMMIT
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload)


def test_local_export_setup_never_calls_hub(ready_setup, tmp_path, monkeypatch):
    export = _snapshot(tmp_path, ready_setup)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace())
    model, revision = ready_setup._model_location(str(export), None)
    setup = ready_setup._setup_report(model, revision)["setup"]
    assert setup["model_source"] == "local" and setup["revision"] is None
    assert setup["model_cache"]["complete"] is True
    assert setup["recommendation"] == "ready_for_live_cuda_inference"


def test_incomplete_local_export_setup_is_not_ready(ready_setup, tmp_path):
    setup = ready_setup._setup_report(str(tmp_path), None)["setup"]
    assert setup["model_cache"]["complete"] is False
    assert setup["recommendation"] == "stage_complete_model_and_processor_assets"


def test_local_model_rejects_explicit_hub_revision(wrapper, tmp_path):
    with pytest.raises(wrapper.SkillError, match="only to Hub models"):
        wrapper._model_location(str(tmp_path), "main")


def test_local_model_path_errors(wrapper, tmp_path):
    with pytest.raises(wrapper.SkillError, match="directory not found"):
        wrapper._model_location(str(tmp_path / "missing"), None)
    file = tmp_path / "file"
    file.touch()
    with pytest.raises(wrapper.SkillError, match="not a directory"):
        wrapper._model_location(str(file), None)


@pytest.mark.parametrize(
    "fixture_name,reason",
    [
        ("upstream_checkout", "Real-volume checks require"),
        ("live_revision", "Live model inference requires"),
    ],
)
def test_opt_in_fixtures_skip_when_options_unregistered(
    pytestconfig, monkeypatch, fixture_name, reason
):
    for option in (
        "nv_reason_ct_upstream",
        "nv_reason_ct_live",
        "nv_reason_ct_trust_model_code",
        "nv_reason_ct_revision",
    ):
        monkeypatch.delattr(pytestconfig.option, option, raising=False)
    fixtures = runpy.run_path(str(SKILL_DIR / "tests" / "conftest.py"))
    with pytest.raises(pytest.skip.Exception, match=reason):
        fixtures[fixture_name].__wrapped__(pytestconfig)
