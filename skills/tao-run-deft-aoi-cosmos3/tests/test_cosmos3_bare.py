#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import yaml


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"


def companion_skill_root(category: str, name: str) -> pathlib.Path:
    """Resolve a companion in a source checkout or flattened plugin install."""
    candidates = (
        SKILL_ROOT.parents[1] / category / name,
        SKILL_ROOT.parent / name,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"required companion skill {name!r} not found in: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


MODEL_ROOT = companion_skill_root("models", "tao-finetune-cosmos-reason")
sys.path.insert(0, str(SCRIPTS))
DATA_MINING_SCRIPTS = companion_skill_root(
    "data", "tao-mine-aoi-images"
) / "scripts"
sys.path.insert(0, str(DATA_MINING_SCRIPTS))

# The ChangeNet DEFT skill has scripts with the same top-level module names.
# Pytest may collect that suite first in one interpreter; clear only those
# ambiguous imports before loading the Cosmos3-local state machine modules.
for module_name in (
    "commit_stage",
    "finalize_run",
    "init_deft_state",
    "metric_contract",
    "record_metric_result",
    "render_report",
):
    sys.modules.pop(module_name, None)

import analyze_gaps  # noqa: E402
import assemble_training_json  # noqa: E402
import check_annotations  # noqa: E402
import commit_stage  # noqa: E402
import emit_mined_sharegpt  # noqa: E402
import emit_sdg_sharegpt  # noqa: E402
import filter_mined_by_cosine  # noqa: E402
import filter_mined_history  # noqa: E402
import finalize_run  # noqa: E402
import init_deft_state  # noqa: E402
import patch_eval_image_cap  # noqa: E402
import render_report  # noqa: E402
import validate_sharegpt  # noqa: E402
import validate_split_contract  # noqa: E402


def record(target: str, golden: str, label: str) -> dict:
    return {
        "images": [target, golden],
        "conversations": [
            {
                "from": "human",
                "value": "Compare the AOI image with the golden reference.",
            },
            {"from": "gpt", "value": label},
        ],
    }


def write_json(path: pathlib.Path, payload: object) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def write_rcca_report(path: pathlib.Path) -> pathlib.Path:
    path.write_text(
        "# Cosmos3 Proxy RCCA Report\n\n"
        "## 1. Verdict\nFixture verdict.\n\n"
        "## 2. False-Accept Breakdown\nFixture false accepts.\n\n"
        "## 3. False-Reject Breakdown\nFixture false rejects.\n\n"
        "## 4. Top-K Worst Samples\nFixture worst samples.\n\n"
        "## 5. Per-Defect Analysis\nFixture defect analysis.\n\n"
        "## 6. Recommended Actions\nFixture actions.\n",
        encoding="utf-8",
    )
    return path


def write_sdg_output(sdg_dir: pathlib.Path, stems: list[str]) -> pathlib.Path:
    """Fake one AnomalyGen SDG run: paired reconstructed/original images + CSV."""
    rows = ["reconstructed_image,original_image,psnr"]
    for stem in stems:
        for subdir in ("reconstructed_image", "original_image"):
            image = sdg_dir / subdir / f"{stem}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"\x89PNG\r\n\x1a\n")
        rows.append(
            f"reconstructed_image/{stem}.png,original_image/{stem}.png,31.5"
        )
    csv_path = sdg_dir / "SDG_result.csv"
    csv_path.write_text("\n".join(rows) + "\n")
    return csv_path


class BareAnnotationTests(unittest.TestCase):
    def test_exact_bare_labels_only(self) -> None:
        records = [record("a.png", "golden.png", "OK")]
        summary = validate_sharegpt.validate_records(
            records, media_root=pathlib.Path("/tmp"), require_files=False
        )
        self.assertEqual(summary["mode"], "bare_okng")
        self.assertEqual(summary["labels"], {"OK": 1})

        invalid = [record("b.png", "golden.png", "Final answer: NG")]
        with self.assertRaisesRegex(ValueError, "exactly OK or NG"):
            validate_sharegpt.validate_records(
                invalid,
                media_root=pathlib.Path("/tmp"),
                require_files=False,
            )

    def test_evaluation_ids_are_unique_and_filesystem_safe(self) -> None:
        """cosmos-rl-evaluate hard-indexes id and reuses it as a filename."""
        media = pathlib.Path("/tmp")
        ok = [
            {**record("a.png", "g.png", "OK"), "id": "a_93c3e56d"},
            {**record("b.png", "g.png", "NG"), "id": "b_1f2e3d4c"},
        ]
        summary = validate_sharegpt.validate_records(
            ok, media_root=media, require_files=False, require_id=True
        )
        self.assertEqual(summary["unique_ids"], 2)

        # Missing id is only an error for the evaluated splits.
        bare = [record("a.png", "g.png", "OK")]
        self.assertEqual(
            validate_sharegpt.validate_records(
                bare, media_root=media, require_files=False
            )["unique_ids"],
            0,
        )
        with self.assertRaisesRegex(ValueError, "id must be a non-empty string"):
            validate_sharegpt.validate_records(
                bare, media_root=media, require_files=False, require_id=True
            )

        duplicated = [
            {**record("a.png", "g.png", "OK"), "id": "same"},
            {**record("b.png", "g.png", "NG"), "id": "same"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            validate_sharegpt.validate_records(
                duplicated, media_root=media, require_files=False, require_id=True
            )

        # id doubles as a path segment, so separators must be rejected.
        unsafe = [{**record("a.png", "g.png", "OK"), "id": "dir/../escape"}]
        with self.assertRaisesRegex(ValueError, "filesystem-safe"):
            validate_sharegpt.validate_records(
                unsafe, media_root=media, require_files=False, require_id=True
            )

    def test_annotation_contract_is_checked_per_role(self) -> None:
        """Preflight must fail on a missing evaluation id, not the first GPU job."""
        with tempfile.TemporaryDirectory() as temporary:
            workspace = pathlib.Path(temporary)

            def write_roles(proxy_id: str | None) -> None:
                proxy = record("p.png", "g.png", "OK")
                if proxy_id:
                    proxy["id"] = proxy_id
                write_json(workspace / "annotations/proxy_kpi.json", [proxy])
                write_json(
                    workspace / "annotations/benchmark_kpi.json",
                    [{**record("b.png", "g.png", "NG"), "id": "b_1"}],
                )
                write_json(
                    workspace / "annotations/mining_pool.json",
                    [record("m.png", "g.png", "OK")],
                )

            write_roles(None)
            self.assertEqual(
                check_annotations.main(["--workspace", str(workspace)]), 2
            )

            write_roles("p_1")
            self.assertEqual(
                check_annotations.main(["--workspace", str(workspace)]), 0
            )

            # Mining carries no id and must still pass: only the evaluated
            # splits reach cosmos-rl-evaluate.
            report, failures = check_annotations.check(
                {
                    role: workspace / "annotations" / spec["filename"]
                    for role, spec in check_annotations.ROLE_CONTRACT.items()
                },
                media_root=workspace,
                require_files=False,
            )
            self.assertEqual(failures, [])
            self.assertEqual(report["mining"]["id_coverage"], "n/a")
            self.assertEqual(report["benchmark"]["id_coverage"], "1/1")

    def test_eval_image_cap_patch_targets_only_the_cap(self) -> None:
        """bare_okng needs 2 images; the image hardcodes 1 with no override."""
        source = (
            "        engine = LLM(\n"
            "            model=ckpt,\n"
            '            limit_mm_per_prompt={"video": 1, "image": 1},\n'
            "            tensor_parallel_size=tp_size,\n"
            "        )\n"
        )
        patched, current = patch_eval_image_cap.apply_cap(source, 2)
        self.assertEqual(current, 1)
        self.assertIn('{"video": 1, "image": 2}', patched)
        # Only the cap changes: same line count, video untouched.
        self.assertEqual(len(patched.splitlines()), len(source.splitlines()))
        self.assertEqual(len(patched), len(source))

        # An image that was fixed upstream reports its cap and needs no rewrite.
        fixed = source.replace('"image": 1', '"image": 4')
        _, already = patch_eval_image_cap.apply_cap(fixed, 2)
        self.assertEqual(already, 4)

        # If the image changes shape, fail loudly instead of guessing.
        with self.assertRaisesRegex(ValueError, "not found"):
            patch_eval_image_cap.apply_cap("engine = LLM(model=ckpt)\n", 2)
        with self.assertRaisesRegex(ValueError, "exactly one image cap"):
            patch_eval_image_cap.apply_cap(source + source, 2)

    def test_eval_image_cap_probe_times_out(self) -> None:
        timeout = patch_eval_image_cap.subprocess.TimeoutExpired(
            cmd=["docker", "run"], timeout=120
        )
        with mock.patch.object(
            patch_eval_image_cap.shutil, "which", return_value="/usr/bin/docker"
        ), mock.patch.object(
            patch_eval_image_cap.subprocess, "run", side_effect=timeout
        ) as run:
            with self.assertRaisesRegex(
                ValueError,
                "timed out after 120s.*pre-pull the image",
            ):
                patch_eval_image_cap.read_from_image(
                    "example/cosmos:1",
                    patch_eval_image_cap.CONTAINER_PATH,
                    docker="docker",
                )
        self.assertEqual(run.call_args.kwargs["timeout"], 120)

    def test_media_root_one_level_too_deep_is_diagnosed(self) -> None:
        """Annotations resolve from the workspace root, not workspace/images."""
        message = (
            "record[0]: missing image file(s): "
            "['/ws/images/images/board/R1.jpg']"
        )
        hint = check_annotations._media_root_hint(
            message, pathlib.Path("/ws/images")
        )
        self.assertIsNotNone(hint)
        self.assertIn("--media-root /ws", hint)
        # A correct root produces no doubling and therefore no hint.
        self.assertIsNone(
            check_annotations._media_root_hint(
                "record[0]: missing image file(s): ['/ws/images/board/R1.jpg']",
                pathlib.Path("/ws"),
            )
        )

    def test_parquet_on_a_json_flag_is_named(self) -> None:
        """Routing deals in both formats, so the mix-up needs a clear error."""
        with tempfile.TemporaryDirectory() as temporary:
            parquet = pathlib.Path(temporary) / "mining_targets.parquet"
            parquet.write_bytes(b"PAR1" + b"\x00" * 16)
            with self.assertRaisesRegex(ValueError, "got a parquet file"):
                commit_stage._required_json_file(parquet, "--mining-targets")

            not_a_list = pathlib.Path(temporary) / "obj.json"
            write_json(not_a_list, {"filepath": "a.png"})
            with self.assertRaisesRegex(ValueError, "must be a JSON array"):
                commit_stage._required_json_file(not_a_list, "--mining-targets")

            good = write_json(
                pathlib.Path(temporary) / "ok.json", [{"filepath": "a.png"}]
            )
            self.assertEqual(
                commit_stage._required_json_file(good, "--mining-targets"),
                str(good.resolve()),
            )

    def test_loader_requires_one_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            records = [
                record("a.png", "golden.png", "OK"),
                record("b.png", "golden.png", "NG"),
            ]
            json_path = write_json(root / "proxy_kpi.json", records)
            self.assertEqual(validate_sharegpt.load_records(json_path), records)
            jsonl_path = root / "proxy_kpi.jsonl"
            jsonl_path.write_text(
                "".join(json.dumps(item) + "\n" for item in records)
            )
            with self.assertRaisesRegex(ValueError, "JSONL is not supported"):
                validate_sharegpt.load_records(jsonl_path)

    def test_mined_alignment_preserves_prompt_pair_and_label(self) -> None:
        source = [record("pool/a.png", "golden/g.png", "NG")]
        output, summary = emit_mined_sharegpt.emit_records(
            ["pool/a.png"],
            source,
            media_root=pathlib.Path("/dataset"),
            relative=True,
        )
        self.assertEqual(summary["mode"], "bare_okng")
        self.assertEqual(output[0]["images"], ["pool/a.png", "golden/g.png"])
        self.assertEqual(output[0]["conversations"][-1]["value"], "NG")

    def test_first_train_is_mined_then_assembly_is_monotonic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            new = write_json(
                root / "new.json", [record("new.png", "golden.png", "NG")]
            )
            first, summary = assemble_training_json.assemble(
                None,
                [new],
                dedupe=True,
                validation_paths=[],
            )
            self.assertEqual(len(first), 1)
            self.assertIsNone(summary["seed"])

            previous = write_json(root / "iter1.json", first)
            next_input = write_json(
                root / "next.json", [record("next.png", "golden.png", "OK")]
            )
            merged, summary = assemble_training_json.assemble(
                previous,
                [next_input],
                dedupe=True,
                validation_paths=[],
            )
            self.assertEqual(len(merged), 2)
            self.assertEqual(summary["mode"], "bare_okng")
            self.assertEqual(summary["labels"], {"OK": 1, "NG": 1})

            proxy = write_json(
                root / "proxy_kpi.json", [record("next.png", "g2.png", "NG")]
            )
            with self.assertRaisesRegex(ValueError, "leakage"):
                assemble_training_json.assemble(
                    previous,
                    [next_input],
                    dedupe=True,
                    validation_paths=[proxy],
                )


class IsolationAndMetricTests(unittest.TestCase):
    def test_cosmos3_variant_aliases_are_explicit_and_canonical(self) -> None:
        self.assertEqual(
            init_deft_state.canonicalize_base_model("nano"),
            "nvidia/Cosmos3-Nano",
        )
        self.assertEqual(
            init_deft_state.canonicalize_base_model("EDGE"),
            "nvidia/Cosmos3-Edge",
        )
        self.assertEqual(
            init_deft_state.canonicalize_base_model("super"),
            "nvidia/Cosmos3-Super",
        )
        self.assertEqual(
            init_deft_state.canonicalize_base_model("/models/custom-cosmos3"),
            "/models/custom-cosmos3",
        )

    def test_model_skill_owns_train_and_evaluate_templates(self) -> None:
        info = yaml.safe_load(
            (MODEL_ROOT / "references/skill_info.yaml").read_text()
        )
        train = yaml.safe_load(
            (MODEL_ROOT / "references/spec_template_train.yaml").read_text()
        )
        evaluate = yaml.safe_load(
            (MODEL_ROOT / "references/spec_template_evaluate.yaml").read_text()
        )
        for action in ("train", "evaluate"):
            self.assertEqual(info["actions"][action]["mode"], "config")
            self.assertEqual(info["actions"][action]["config_format"], "toml")
        self.assertIn("train_dataset", train["custom"])
        self.assertIn("dataset", evaluate)
        self.assertFalse(
            (SKILL_ROOT / "references/train_spec.toml").exists()
        )
        self.assertFalse(
            (SKILL_ROOT / "references/evaluate_spec.toml").exists()
        )
        self.assertTrue(
            (
                MODEL_ROOT
                / "scripts/prepare_cosmos3_vlm_checkpoint.py"
            ).is_file()
        )
        application_contract = (SKILL_ROOT / "SKILL.md").read_text()
        self.assertIn("model_type=\"cosmos3_omni\"", application_contract)
        self.assertIn(
            "scripts/prepare_cosmos3_vlm_checkpoint.py",
            application_contract,
        )

    def test_cosine_floor_recomputes_similarity(self) -> None:
        kept, audit = filter_mined_by_cosine.filter_mined_records(
            mined_rows=[{"filepath": "keep.png"}, {"filepath": "drop.png"}],
            source_rows=[
                {"filepath": "keep.png", "embedding": [1.0, 0.0]},
                {"filepath": "drop.png", "embedding": [0.0, 1.0]},
            ],
            target_rows=[{"filepath": "target.png", "embedding": [1.0, 0.0]}],
            min_similarity=0.9,
        )
        self.assertEqual(kept, [0])
        self.assertTrue(audit[0]["kept"])
        self.assertFalse(audit[1]["kept"])

    def test_split_contract_and_frozen_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            role_paths = {
                role: write_json(
                    root / f"{role}.json",
                    [record(f"{role}.png", "golden.png", "OK")],
                )
                for role in ("proxy", "benchmark", "mining")
            }
            expected = hashlib.sha256(
                role_paths["benchmark"].read_bytes()
            ).hexdigest()
            summary = validate_split_contract.validate(
                role_paths,
                media_root=root,
                expected_benchmark_sha256=expected,
            )
            self.assertTrue(summary["benchmark_hash_verified"])

            generated_train = write_json(
                root / "train.json",
                [record("mining.png", "golden.png", "OK")],
            )
            generated_summary = validate_split_contract.validate(
                {**role_paths, "train": generated_train},
                media_root=root,
                expected_benchmark_sha256=expected,
            )
            self.assertEqual(
                generated_summary["roles"]["train"], "generated_from_mining"
            )

            # AnomalyGen output is an eligible Train source alongside Mining.
            synthetic = write_json(
                root / "synthetic.json",
                [record("sdg/PCB+bridge_00000.png", "sdg/orig.png", "NG")],
            )
            mixed_train = write_json(
                root / "train_mixed.json",
                [
                    record("mining.png", "golden.png", "OK"),
                    record("sdg/PCB+bridge_00000.png", "sdg/orig.png", "NG"),
                ],
            )
            mixed_summary = validate_split_contract.validate(
                {**role_paths, "synthetic": synthetic, "train": mixed_train},
                media_root=root,
                expected_benchmark_sha256=expected,
            )
            self.assertEqual(
                mixed_summary["roles"]["train"],
                "generated_from_mining_and_anomalygen",
            )
            self.assertEqual(mixed_summary["target_overlap"]["train:synthetic"], 1)

            # Iteration 2 must retain iteration 1, including historical
            # synthetic records that are neither in Mining nor in iteration
            # 2's AnomalyGen output.
            current_synthetic = write_json(
                root / "synthetic_iter2.json",
                [record("sdg/iter2.png", "sdg/orig2.png", "NG")],
            )
            monotonic_train = write_json(
                root / "train_iter2.json",
                [
                    record("mining.png", "golden.png", "OK"),
                    record("sdg/PCB+bridge_00000.png", "sdg/orig.png", "NG"),
                    record("sdg/iter2.png", "sdg/orig2.png", "NG"),
                ],
            )
            with self.assertRaisesRegex(ValueError, "must come from the Mining"):
                validate_split_contract.validate(
                    {
                        **role_paths,
                        "synthetic": current_synthetic,
                        "train": monotonic_train,
                    },
                    media_root=root,
                    expected_benchmark_sha256=expected,
                )

            monotonic_roles = {
                **role_paths,
                "previous_train": mixed_train,
                "synthetic": current_synthetic,
                "train": monotonic_train,
            }
            monotonic_summary = validate_split_contract.validate(
                monotonic_roles,
                media_root=root,
                expected_benchmark_sha256=expected,
            )
            self.assertEqual(
                monotonic_summary["roles"]["previous_train"],
                "previous_iteration_train",
            )
            self.assertEqual(
                monotonic_summary["target_overlap"]["train:previous_train"],
                2,
            )

            dropped_history = write_json(
                root / "train_iter2_dropped_history.json",
                [
                    record("mining.png", "golden.png", "OK"),
                    record("sdg/iter2.png", "sdg/orig2.png", "NG"),
                ],
            )
            with self.assertRaisesRegex(
                ValueError, "retain every record from --previous-train"
            ):
                validate_split_contract.validate(
                    {**monotonic_roles, "train": dropped_history},
                    media_root=root,
                    expected_benchmark_sha256=expected,
                )

            # A synthetic board that also sits in an evaluation split is leakage.
            leaking_synthetic = write_json(
                root / "synthetic_leak.json",
                [record("proxy.png", "sdg/orig.png", "NG")],
            )
            with self.assertRaisesRegex(
                ValueError, "leakage between synthetic and proxy"
            ):
                validate_split_contract.validate(
                    {**role_paths, "synthetic": leaking_synthetic},
                    media_root=root,
                    expected_benchmark_sha256=expected,
                )

            role_paths["mining"] = write_json(
                root / "mining.json",
                [record("proxy.png", "other-golden.png", "NG")],
            )
            with self.assertRaisesRegex(ValueError, "target leakage"):
                validate_split_contract.validate(
                    role_paths,
                    media_root=root,
                    expected_benchmark_sha256=expected,
                )

            outside_mining = write_json(
                root / "outside.json",
                [record("outside.png", "golden.png", "OK")],
            )
            isolated_roles = {
                "proxy": write_json(
                    root / "proxy2.json",
                    [record("proxy2.png", "golden.png", "OK")],
                ),
                "benchmark": role_paths["benchmark"],
                "mining": write_json(
                    root / "mining2.json",
                    [record("mining2.png", "golden.png", "OK")],
                ),
                "train": outside_mining,
            }
            with self.assertRaisesRegex(ValueError, "must come from the Mining"):
                validate_split_contract.validate(
                    isolated_roles,
                    media_root=root,
                )

    def test_proxy_never_gates_and_benchmark_unknown_blocks(self) -> None:
        samples = [
            {"gt": "NG", "response": "NG"},
            {"gt": "OK", "response": "OK"},
        ]
        proxy, *_ = analyze_gaps.analyze(
            samples, evaluation_role="proxy"
        )
        self.assertIsNone(proxy["kpi"]["met"])
        self.assertFalse(proxy["kpi"]["gate_eligible"])

        benchmark, *_ = analyze_gaps.analyze(
            samples, evaluation_role="benchmark"
        )
        self.assertTrue(benchmark["kpi"]["met"])
        unknown, *_ = analyze_gaps.analyze(
            [{"gt": "NG", "response": "maybe"}],
            evaluation_role="benchmark",
        )
        self.assertFalse(unknown["kpi"]["met"])
        self.assertEqual(unknown["unknown_samples"], 1)


class StateMachineTests(unittest.TestCase):
    def test_stage_commit_validates_duration_by_skip_contract(self) -> None:
        base = [
            "--results-dir",
            "/tmp/cosmos3-duration-contract",
            "--iter-label",
            "baseline",
            "--stage",
            "loop_stop",
            "--summary",
            "done",
        ]
        with self.assertRaises(SystemExit):
            commit_stage._parser().parse_args(base)
        for duration in ("0", "-1"):
            self.assertEqual(
                commit_stage.main([*base, "--duration-sec", duration]), 2
            )
        skip = [
            "--results-dir",
            "/tmp/cosmos3-duration-contract",
            "--iter-label",
            "iter1",
            "--stage",
            "anomalygen",
            "--skip",
            "--summary",
            "driving RCCA has zero false accepts",
        ]
        self.assertEqual(commit_stage.main([*skip, "--duration-sec", "-1"]), 2)

    def test_baseline_commit_to_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = root / "workspace"
            results = root / "results"
            workspace.mkdir()
            (workspace / "specs").mkdir()
            for name in ("train_spec.toml", "evaluate_spec.toml"):
                (workspace / "specs" / name).write_text("value = 1\n")
            for role, filename, label in (
                ("proxy", "proxy_kpi.json", "NG"),
                ("benchmark", "benchmark_kpi.json", "NG"),
                ("mining", "mining_pool.json", "OK"),
            ):
                write_json(
                    workspace / "annotations" / filename,
                    [record(f"{role}.png", "golden.png", label)],
                )

            rc = init_deft_state.main(
                [
                    "--results-dir",
                    str(results),
                    "--workspace",
                    str(workspace),
                    "--platform",
                    "docker",
                    "--network-mode",
                    "airgap",
                    "--network-mode-source",
                    "test-harness",
                    "--gpu-model",
                    "NVIDIA H100 80GB HBM3",
                    "--max-iterations",
                    "1",
                    "--cosmos-container",
                    "example/cosmos:1",
                    "--mining-container",
                    "example/mining:1",
                ]
            )
            self.assertEqual(rc, 0)
            state = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(state["version"], 5)
            self.assertEqual(
                state["execution_policy"]["network_mode"], "airgap"
            )
            self.assertFalse(
                state["execution_policy"]["allow_package_install"]
            )
            self.assertNotIn("train", state["config"]["annotations"])
            self.assertTrue(state["config"]["evaluation"]["proxy"]["drives_rcca"])
            self.assertFalse(
                state["config"]["evaluation"]["proxy"]["drives_loop_stop"]
            )
            self.assertTrue(
                state["config"]["evaluation"]["benchmark"]["drives_loop_stop"]
            )
            self.assertFalse(
                state["config"]["evaluation"]["benchmark"]["drives_rcca"]
            )
            self.assertIn("mining", state["config"]["annotations"])
            self.assertEqual(state["config"]["mining"]["metric"], "cosine")

            # Gate first, and it passes here, so the run must stop without ever
            # evaluating Proxy: Proxy only exists to seed a next iteration.
            benchmark_results = write_json(
                results / "baseline/evaluate_benchmark/results.json",
                [{"gt": "NG", "response": "NG"}],
            )
            self.assertEqual(
                commit_stage.main(
                    [
                        "--results-dir",
                        str(results),
                        "--iter-label",
                        "baseline",
                        "--stage",
                        "evaluate_benchmark",
                        "--benchmark-results",
                        str(benchmark_results),
                        "--duration-sec",
                        "1",
                        "--summary",
                        "Benchmark evaluation complete",
                    ]
                ),
                0,
            )
            benchmark_dir = results / "baseline/benchmark_metrics"
            self.assertEqual(
                analyze_gaps.main(
                    [
                        "--results-json",
                        str(benchmark_results),
                        "--output-dir",
                        str(benchmark_dir),
                        "--evaluation-role",
                        "benchmark",
                    ]
                ),
                0,
            )
            self.assertEqual(
                commit_stage.main(
                    [
                        "--results-dir",
                        str(results),
                        "--iter-label",
                        "baseline",
                        "--stage",
                        "benchmark_metrics",
                        "--benchmark-metrics-summary",
                        str(benchmark_dir / "metrics_summary.json"),
                        "--metric-result",
                        str(benchmark_dir / "metric_result.json"),
                        "--duration-sec",
                        "1",
                        "--summary",
                        "Benchmark KPI met",
                    ]
                ),
                0,
            )
            state = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(state["status"], "in_progress")
            self.assertEqual(state["events"][-1]["stage"], "benchmark_metrics")
            self.assertEqual(
                finalize_run.main(
                    [
                        "--results-dir", str(results),
                        "--iter-label", "baseline",
                        "--stop-reason", "metric_met",
                        "--duration-sec", "1",
                    ]
                ),
                0,
            )
            state = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["events"][-1]["stage"], "loop_stop")
            self.assertEqual(
                [event["seq"] for event in state["events"]], [1, 2, 3]
            )
            self.assertFalse((results / "loop_log.jsonl").exists())
            html_report = results / "DEFT_Loop_Report.html"
            self.assertTrue(html_report.is_file())
            self.assertIn("KPI MET", html_report.read_text())
            self.assertEqual(
                render_report.main(
                    ["--results-dir", str(results), "--require-terminal"]
                ),
                0,
            )
            phase = json.loads(
                (results / "deft_state.json").read_text()
            )["iterations"]["baseline"]
            self.assertEqual(phase["stage_completed"], "benchmark_metrics")
            self.assertNotIn("proxy_results_json", phase)

    def test_passed_gate_is_visible_in_state_without_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = root / "workspace"
            results = root / "results"
            (workspace / "specs").mkdir(parents=True)
            for name in ("train_spec.toml", "evaluate_spec.toml"):
                (workspace / "specs" / name).write_text("value = 1\n")
            for filename, label in (
                ("proxy_kpi.json", "NG"),
                ("benchmark_kpi.json", "NG"),
                ("mining_pool.json", "OK"),
            ):
                write_json(
                    workspace / "annotations" / filename,
                    [record(filename.replace(".json", ".png"), "g.png", label)],
                )
            self.assertEqual(
                init_deft_state.main(
                    [
                        "--results-dir",
                        str(results),
                        "--workspace",
                        str(workspace),
                        "--platform",
                        "docker",
                        "--gpu-model",
                        "NVIDIA H100 80GB HBM3",
                        "--max-iterations",
                        "2",
                        "--cosmos-container",
                        "example/cosmos:1",
                        "--mining-container",
                        "example/mining:1",
                    ]
                ),
                0,
            )

            def commit(stage: str, *extra: str) -> int:
                return commit_stage.main(
                    [
                        "--results-dir",
                        str(results),
                        "--iter-label",
                        "baseline",
                        "--stage",
                        stage,
                        "--duration-sec",
                        "1",
                        "--summary",
                        f"baseline {stage}",
                        *extra,
                    ]
                )

            benchmark_results = write_json(
                results / "baseline/evaluate_benchmark/results.json",
                [{"gt": "NG", "response": "NG"}],
            )
            self.assertEqual(
                commit(
                    "evaluate_benchmark",
                    "--benchmark-results",
                    str(benchmark_results),
                ),
                0,
            )
            metrics_dir = results / "baseline/benchmark_metrics"
            self.assertEqual(
                analyze_gaps.main(
                    [
                        "--results-json",
                        str(benchmark_results),
                        "--output-dir",
                        str(metrics_dir),
                        "--evaluation-role",
                        "benchmark",
                    ]
                ),
                0,
            )
            self.assertEqual(
                commit(
                    "benchmark_metrics",
                    "--benchmark-metrics-summary",
                    str(metrics_dir / "metrics_summary.json"),
                    "--metric-result",
                    str(metrics_dir / "metric_result.json"),
                ),
                0,
            )

            # The gate result is canonical state evidence; the orchestrator
            # reads it and proceeds directly to loop_stop without Proxy work.
            state = json.loads((results / "deft_state.json").read_text())
            phase = state["iterations"]["baseline"]
            self.assertEqual(phase["stage_completed"], "benchmark_metrics")
            self.assertEqual(phase["status"], "complete")
            self.assertTrue(phase["metric_result"]["passed"])
            self.assertEqual(state["status"], "in_progress")
            self.assertEqual(state["events"][-1]["stage"], "benchmark_metrics")

    def test_missing_specs_fail_before_state_exists(self) -> None:
        """State is written once and never hand-edited, so fail before writing."""
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = root / "workspace"
            results = root / "results"
            for filename, label in (
                ("proxy_kpi.json", "OK"),
                ("benchmark_kpi.json", "NG"),
                ("mining_pool.json", "OK"),
            ):
                write_json(
                    workspace / "annotations" / filename,
                    [record(filename.replace(".json", ".png"), "g.png", label)],
                )

            argv = [
                "--results-dir",
                str(results),
                "--workspace",
                str(workspace),
                "--platform",
                "docker",
                "--gpu-model",
                "NVIDIA H100 80GB HBM3",
                "--max-iterations",
                "1",
                "--cosmos-container",
                "example/cosmos:1",
                "--mining-container",
                "example/mining:1",
            ]
            self.assertEqual(init_deft_state.main(argv), 2)
            self.assertFalse((results / "deft_state.json").exists())

            (workspace / "specs").mkdir(parents=True)
            for name in ("train_spec.toml", "evaluate_spec.toml"):
                (workspace / "specs" / name).write_text("value = 1\n")
            self.assertEqual(init_deft_state.main(argv), 0)
            self.assertTrue((results / "deft_state.json").exists())

    def test_per_role_evaluate_specs_are_resolved(self) -> None:
        """Split evaluate specs are preferred; a shared one still works."""
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = root / "workspace"
            (workspace / "specs").mkdir(parents=True)
            (workspace / "specs/train_spec.toml").write_text("value = 1\n")
            for filename, label in (
                ("proxy_kpi.json", "OK"),
                ("benchmark_kpi.json", "NG"),
                ("mining_pool.json", "OK"),
            ):
                write_json(
                    workspace / "annotations" / filename,
                    [record(filename.replace(".json", ".png"), "g.png", label)],
                )

            def init(results: pathlib.Path, *extra: str) -> dict:
                self.assertEqual(
                    init_deft_state.main(
                        [
                            "--results-dir",
                            str(results),
                            "--workspace",
                            str(workspace),
                            "--platform",
                            "docker",
                            "--gpu-model",
                            "NVIDIA H100 80GB HBM3",
                            "--max-iterations",
                            "1",
                            "--cosmos-container",
                            "example/cosmos:1",
                            "--mining-container",
                            "example/mining:1",
                            *extra,
                        ]
                    ),
                    0,
                )
                state = json.loads((results / "deft_state.json").read_text())
                return state["config"]["specs"]

            # Shared evaluate spec: both roles point at the same file.
            (workspace / "specs/evaluate_spec.toml").write_text("value = 1\n")
            shared = init(root / "r_shared")
            self.assertEqual(shared["proxy"], shared["benchmark"])

            # Per-role files take precedence and stay distinct, so a Proxy job
            # cannot pick up the Benchmark annotation.
            for role in ("proxy", "benchmark"):
                (workspace / f"specs/evaluate_spec_{role}.toml").write_text(
                    "value = 1\n"
                )
            split = init(root / "r_split")
            self.assertNotEqual(split["proxy"], split["benchmark"])
            self.assertTrue(split["proxy"].endswith("evaluate_spec_proxy.toml"))
            self.assertTrue(
                split["benchmark"].endswith("evaluate_spec_benchmark.toml")
            )

            override = init(
                root / "r_override",
                "--benchmark-spec",
                str(workspace / "specs/evaluate_spec.toml"),
            )
            self.assertTrue(
                override["benchmark"].endswith("evaluate_spec.toml")
            )

    def test_error_stage_commits_without_its_artifact(self) -> None:
        """A hard stop usually dies before writing anything; record it anyway."""
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = root / "workspace"
            results = root / "results"
            (workspace / "specs").mkdir(parents=True)
            for name in ("train_spec.toml", "evaluate_spec.toml"):
                (workspace / "specs" / name).write_text("value = 1\n")
            for filename, label in (
                ("proxy_kpi.json", "OK"),
                ("benchmark_kpi.json", "NG"),
                ("mining_pool.json", "OK"),
            ):
                write_json(
                    workspace / "annotations" / filename,
                    [record(filename.replace(".json", ".png"), "g.png", label)],
                )
            self.assertEqual(
                init_deft_state.main(
                    [
                        "--results-dir",
                        str(results),
                        "--workspace",
                        str(workspace),
                        "--platform",
                        "docker",
                        "--gpu-model",
                        "NVIDIA H100 80GB HBM3",
                        "--max-iterations",
                        "1",
                        "--cosmos-container",
                        "example/cosmos:1",
                        "--mining-container",
                        "example/mining:1",
                    ]
                ),
                0,
            )
            # The evaluator crashed before writing results.json, so there is no
            # --benchmark-results to hand over.
            self.assertEqual(
                commit_stage.main(
                    [
                        "--results-dir",
                        str(results),
                        "--iter-label",
                        "baseline",
                        "--stage",
                        "evaluate_benchmark",
                        "--status",
                        "error",
                        "--duration-sec",
                        "1",
                        "--summary",
                        "cosmos-rl-evaluate exited 1 before writing results",
                    ]
                ),
                0,
            )
            state = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(state["status"], "failed")
            phase = state["iterations"]["baseline"]
            self.assertEqual(phase["status"], "failed")
            self.assertEqual(len(state["events"]), 1)
            self.assertEqual(state["events"][0]["status"], "error")

    def test_anomalygen_skip_requires_empty_driving_false_accepts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = root / "workspace"
            results = root / "results"
            (workspace / "specs").mkdir(parents=True)
            for name in ("train_spec.toml", "evaluate_spec.toml"):
                (workspace / "specs" / name).write_text("value = 1\n")
            for filename, label in (
                ("proxy_kpi.json", "OK"),
                ("benchmark_kpi.json", "NG"),
                ("mining_pool.json", "OK"),
            ):
                write_json(
                    workspace / "annotations" / filename,
                    [record(filename.replace(".json", ".png"), "g.png", label)],
                )
            self.assertEqual(
                init_deft_state.main(
                    [
                        "--results-dir",
                        str(results),
                        "--workspace",
                        str(workspace),
                        "--platform",
                        "docker",
                        "--gpu-model",
                        "NVIDIA H100 80GB HBM3",
                        "--max-iterations",
                        "2",
                        "--cosmos-container",
                        "example/cosmos:1",
                        "--mining-container",
                        "example/mining:1",
                    ]
                ),
                0,
            )

            def commit(
                label: str, stage: str, *extra: str, duration: str = "1"
            ) -> int:
                return commit_stage.main(
                    [
                        "--results-dir",
                        str(results),
                        "--iter-label",
                        label,
                        "--stage",
                        stage,
                        "--duration-sec",
                        duration,
                        "--summary",
                        f"{label} {stage}",
                        *extra,
                    ]
                )

            # Gate unmet (NG missed on Benchmark) but Proxy shows only a false
            # reject, so there is no under-detection gap for SDG to close.
            benchmark_results = write_json(
                results / "baseline/evaluate_benchmark/results.json",
                [{"gt": "NG", "response": "OK"}],
            )
            self.assertEqual(
                commit(
                    "baseline",
                    "evaluate_benchmark",
                    "--benchmark-results",
                    str(benchmark_results),
                ),
                0,
            )
            metrics_dir = results / "baseline/benchmark_metrics"
            analyze_gaps.main(
                [
                    "--results-json",
                    str(benchmark_results),
                    "--output-dir",
                    str(metrics_dir),
                    "--evaluation-role",
                    "benchmark",
                ]
            )
            self.assertEqual(
                commit(
                    "baseline",
                    "benchmark_metrics",
                    "--benchmark-metrics-summary",
                    str(metrics_dir / "metrics_summary.json"),
                    "--metric-result",
                    str(metrics_dir / "metric_result.json"),
                ),
                0,
            )
            proxy_results = write_json(
                results / "baseline/evaluate_proxy/results.json",
                [{"gt": "OK", "response": "NG"}],
            )
            self.assertEqual(
                commit(
                    "baseline",
                    "evaluate_proxy",
                    "--proxy-results",
                    str(proxy_results),
                ),
                0,
            )
            proxy_dir = results / "baseline/proxy_rcca"
            analyze_gaps.main(
                [
                    "--results-json",
                    str(proxy_results),
                    "--output-dir",
                    str(proxy_dir),
                    "--evaluation-role",
                    "proxy",
                ]
            )
            self.assertEqual(
                json.loads((proxy_dir / "false_accepts.json").read_text()), []
            )
            rcca_report = write_rcca_report(proxy_dir / "RCCA_Report.md")
            self.assertEqual(
                commit(
                    "baseline",
                    "proxy_rcca",
                    "--proxy-gaps-summary",
                    str(proxy_dir / "gaps_summary.json"),
                    "--false-accepts",
                    str(proxy_dir / "false_accepts.json"),
                    "--false-rejects",
                    str(proxy_dir / "false_rejects.json"),
                    "--rcca-report",
                    str(rcca_report),
                ),
                0,
            )
            targets = write_json(
                results / "iter1/routing/mining_targets.json",
                [{"filepath": "proxy_kpi.png", "label": "OK"}],
            )
            self.assertEqual(
                commit("iter1", "routing", "--mining-targets", str(targets)), 0
            )
            skip_reason = "iter1 anomalygen"
            self.assertEqual(
                commit("iter1", "anomalygen", "--skip", duration="0"), 0
            )
            state = json.loads((results / "deft_state.json").read_text())
            phase = state["iterations"]["iter1"]
            self.assertTrue(phase["anomalygen_skipped"])
            self.assertEqual(phase["anomalygen_skip_reason"], skip_reason)
            self.assertNotIn("anomalygen_sdg_csv", phase)
            self.assertEqual(state["status"], "in_progress")
            event = state["events"][-1]
            self.assertEqual(event["stage"], "anomalygen")
            self.assertEqual(event["status"], "skipped")
            self.assertEqual(event["skip_reason"], skip_reason)
            self.assertEqual(event["duration_sec"], 0)

    def test_unmet_baseline_runs_iteration_to_max(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            workspace = root / "workspace"
            results = root / "results"
            (workspace / "specs").mkdir(parents=True)
            for name in ("train_spec.toml", "evaluate_spec.toml"):
                (workspace / "specs" / name).write_text("value = 1\n")
            annotation_paths = {}
            for role, filename, label in (
                ("proxy", "proxy_kpi.json", "NG"),
                ("benchmark", "benchmark_kpi.json", "NG"),
                ("mining", "mining_pool.json", "OK"),
            ):
                records = [record(f"{role}.png", "golden.png", label)]
                if role == "mining":
                    records.append(record("mining_iter2.png", "golden.png", label))
                annotation_paths[role] = write_json(
                    workspace / "annotations" / filename,
                    records,
                )
            self.assertEqual(
                init_deft_state.main(
                    [
                        "--results-dir",
                        str(results),
                        "--workspace",
                        str(workspace),
                        "--platform",
                        "docker",
                        "--gpu-model",
                        "NVIDIA H100 80GB HBM3",
                        "--max-iterations",
                        "2",
                        "--cosmos-container",
                        "example/cosmos:1",
                        "--mining-container",
                        "example/mining:1",
                    ]
                ),
                0,
            )

            def commit(label: str, stage: str, *extra: str) -> None:
                self.assertEqual(
                    commit_stage.main(
                        [
                            "--results-dir",
                            str(results),
                            "--iter-label",
                            label,
                            "--stage",
                            stage,
                            "--duration-sec",
                            "1",
                            "--summary",
                            f"{label} {stage}",
                            *extra,
                        ]
                    ),
                    0,
                )

            def train(label: str) -> None:
                checkpoint = results / label / "train/safetensors/epoch_1"
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapter_model.safetensors").write_bytes(b"x")
                commit(
                    label,
                    "train",
                    "--best-ckpt",
                    str(checkpoint),
                    "--training-spec",
                    str(workspace / "specs/train_spec.toml"),
                )

            def evaluate_arc(
                label: str, *, correct: bool, continuing: bool
            ) -> None:
                """Gate first; Proxy only when the loop continues past it."""
                response = "NG" if correct else "OK"
                benchmark_results = write_json(
                    results / label / "evaluate_benchmark/results.json",
                    [{"gt": "NG", "response": response}],
                )
                commit(
                    label,
                    "evaluate_benchmark",
                    "--benchmark-results",
                    str(benchmark_results),
                )
                metrics_dir = results / label / "benchmark_metrics"
                self.assertEqual(
                    analyze_gaps.main(
                        [
                            "--results-json",
                            str(benchmark_results),
                            "--output-dir",
                            str(metrics_dir),
                            "--evaluation-role",
                            "benchmark",
                        ]
                    ),
                    0,
                )
                commit(
                    label,
                    "benchmark_metrics",
                    "--benchmark-metrics-summary",
                    str(metrics_dir / "metrics_summary.json"),
                    "--metric-result",
                    str(metrics_dir / "metric_result.json"),
                )
                if not continuing:
                    return
                proxy_results = write_json(
                    results / label / "evaluate_proxy/results.json",
                    [{"gt": "NG", "response": response}],
                )
                commit(
                    label,
                    "evaluate_proxy",
                    "--proxy-results",
                    str(proxy_results),
                )
                proxy_dir = results / label / "proxy_rcca"
                self.assertEqual(
                    analyze_gaps.main(
                        [
                            "--results-json",
                            str(proxy_results),
                            "--output-dir",
                            str(proxy_dir),
                            "--evaluation-role",
                            "proxy",
                        ]
                    ),
                    0,
                )
                rcca_report = write_rcca_report(
                    proxy_dir / "RCCA_Report.md"
                )
                commit(
                    label,
                    "proxy_rcca",
                    "--proxy-gaps-summary",
                    str(proxy_dir / "gaps_summary.json"),
                    "--false-accepts",
                    str(proxy_dir / "false_accepts.json"),
                    "--false-rejects",
                    str(proxy_dir / "false_rejects.json"),
                    "--rcca-report",
                    str(rcca_report),
                )

            evaluate_arc("baseline", correct=False, continuing=True)
            targets = write_json(
                results / "iter1/routing/mining_targets.json",
                [{"filepath": "proxy.png", "label": "NG"}],
            )
            commit(
                "iter1",
                "routing",
                "--mining-targets",
                str(targets),
            )

            sdg_csv = write_sdg_output(
                results / "iter1/anomalygen/sdg", ["PCB+bridge_00000"]
            )
            allocation = write_json(
                results / "iter1/anomalygen/sdg/allocation.json", {"bridge": 1}
            )
            synthetic_json = results / "iter1/anomalygen/sdg_sharegpt.json"
            self.assertEqual(
                emit_sdg_sharegpt.main(
                    [
                        "--sdg-csv",
                        str(sdg_csv),
                        "--media-root",
                        str(root),
                        "--prompt-from",
                        str(annotation_paths["mining"]),
                        "--output",
                        str(synthetic_json),
                    ]
                ),
                0,
            )
            synthetic_records = json.loads(synthetic_json.read_text())
            self.assertEqual(len(synthetic_records), 1)
            self.assertEqual(
                synthetic_records[0]["conversations"][-1]["value"], "NG"
            )
            commit(
                "iter1",
                "anomalygen",
                "--anomalygen-sdg",
                str(sdg_csv),
                "--anomalygen-allocation",
                str(allocation),
                "--anomalygen-sharegpt",
                str(synthetic_json),
            )
            state = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(
                state["iterations"]["iter1"]["anomalygen_amp_allocated"], 1
            )
            self.assertEqual(
                state["iterations"]["iter1"]["anomalygen_allocation_json"],
                str(allocation.resolve()),
            )

            mining_dir = results / "iter1/mining"
            mining_dir.mkdir(parents=True)
            mined = mining_dir / "mined_filtered.parquet"
            mining_candidates = mining_dir / "mined_candidates.parquet"
            mining_history_summary = mining_dir / "mining_history_summary.json"
            mining_history = results / "mining_history.json"
            source_embeddings = mining_dir / "source_embeddings.parquet"
            target_embeddings = mining_dir / "target_embeddings.parquet"
            pq.write_table(
                pa.table({"filepath": ["mining.png"]}), mining_candidates
            )
            history_result = filter_mined_history.select_novel_samples(
                candidate_parquet=mining_candidates,
                output_parquet=mined,
                history_file=mining_history,
                summary_file=mining_history_summary,
                iteration=1,
                topn=5,
            )
            embedding_table = pa.table(
                {"filepath": ["mining.png"], "embedding": [[1.0, 0.0]]}
            )
            pq.write_table(embedding_table, source_embeddings)
            pq.write_table(embedding_table, target_embeddings)
            mining_summary = write_json(
                mining_dir / "cosine_filter_summary.json",
                {"kept_rows": 1, "min_similarity": 0.9},
            )
            commit(
                "iter1",
                "data_mining",
                "--mining-parquet",
                str(mined),
                "--mining-candidates",
                str(mining_candidates),
                "--mining-summary",
                str(mining_summary),
                "--mining-history",
                str(mining_history),
                "--mining-history-summary",
                str(mining_history_summary),
                "--mining-target-embeddings",
                str(target_embeddings),
                "--mining-source-embeddings",
                str(source_embeddings),
                "--mining-count",
                str(history_result["selected_count"]),
            )

            assemble_dir = results / "iter1/assemble"
            mined_sharegpt = write_json(
                assemble_dir / "mined_sharegpt.json",
                [record("mining.png", "golden.png", "OK")],
            )
            # The train file must carry BOTH producers' targets. A mined-only
            # train file never exercises the synthetic lineage path,
            # which is how a missing "synthetic" role at that call site went
            # unnoticed while the standalone validator passed.
            synthetic_target = str(
                (results / "iter1/anomalygen/sdg/reconstructed_image"
                 / "PCB+bridge_00000.png")
            )
            synthetic_golden = str(
                (results / "iter1/anomalygen/sdg/original_image"
                 / "PCB+bridge_00000.png")
            )
            combined = write_json(
                assemble_dir / "train_iter_1.json",
                [
                    record("mining.png", "golden.png", "OK"),
                    record(synthetic_target, synthetic_golden, "NG"),
                ],
            )
            assemble_summary = write_json(
                assemble_dir / "assemble_summary.json",
                {
                    "mode": "bare_okng",
                    "output_records": 2,
                    "seed": None,
                    "validation_jsons": [
                        str(annotation_paths["proxy"]),
                        str(annotation_paths["benchmark"]),
                    ],
                },
            )
            commit(
                "iter1",
                "assemble_data",
                "--mined-sharegpt",
                str(mined_sharegpt),
                "--combined-training",
                str(combined),
                "--assemble-summary",
                str(assemble_summary),
            )
            validation_report = write_json(
                results / "iter1/validate/validation_report.json",
                {
                    "mode": "bare_okng",
                    "records": 1,
                    "labels": {"OK": 1},
                },
            )
            commit(
                "iter1",
                "validate_data",
                "--validation-report",
                str(validation_report),
            )
            train("iter1")

            # The first iteration remains below target, so its Proxy RCCA
            # drives a second augmentation cycle.
            evaluate_arc("iter1", correct=False, continuing=True)
            iter2_targets = write_json(
                results / "iter2/routing/mining_targets.json",
                [{"filepath": "proxy.png", "label": "NG"}],
            )
            commit(
                "iter2",
                "routing",
                "--mining-targets",
                str(iter2_targets),
            )

            iter2_sdg_csv = write_sdg_output(
                results / "iter2/anomalygen/sdg", ["PCB+bridge_00001"]
            )
            iter2_allocation = write_json(
                results / "iter2/anomalygen/sdg/allocation.json", {"bridge": 1}
            )
            iter2_synthetic = results / "iter2/anomalygen/sdg_sharegpt.json"
            self.assertEqual(
                emit_sdg_sharegpt.main(
                    [
                        "--sdg-csv",
                        str(iter2_sdg_csv),
                        "--media-root",
                        str(root),
                        "--prompt-from",
                        str(annotation_paths["mining"]),
                        "--output",
                        str(iter2_synthetic),
                    ]
                ),
                0,
            )
            commit(
                "iter2",
                "anomalygen",
                "--anomalygen-sdg",
                str(iter2_sdg_csv),
                "--anomalygen-allocation",
                str(iter2_allocation),
                "--anomalygen-sharegpt",
                str(iter2_synthetic),
            )

            iter2_mining_dir = results / "iter2/mining"
            iter2_mining_dir.mkdir(parents=True)
            iter2_mined = iter2_mining_dir / "mined_filtered.parquet"
            iter2_mining_candidates = (
                iter2_mining_dir / "mined_candidates.parquet"
            )
            iter2_history_summary = (
                iter2_mining_dir / "mining_history_summary.json"
            )
            iter2_source_embeddings = (
                iter2_mining_dir / "source_embeddings.parquet"
            )
            iter2_target_embeddings = (
                iter2_mining_dir / "target_embeddings.parquet"
            )
            pq.write_table(
                pa.table({"filepath": ["mining_iter2.png"]}),
                iter2_mining_candidates,
            )
            iter2_history_result = filter_mined_history.select_novel_samples(
                candidate_parquet=iter2_mining_candidates,
                output_parquet=iter2_mined,
                history_file=mining_history,
                summary_file=iter2_history_summary,
                iteration=2,
                topn=5,
            )
            iter2_embedding_table = pa.table(
                {
                    "filepath": ["mining_iter2.png"],
                    "embedding": [[1.0, 0.0]],
                }
            )
            pq.write_table(iter2_embedding_table, iter2_source_embeddings)
            pq.write_table(iter2_embedding_table, iter2_target_embeddings)
            iter2_mining_summary = write_json(
                iter2_mining_dir / "cosine_filter_summary.json",
                {"kept_rows": 1, "min_similarity": 0.9},
            )
            commit(
                "iter2",
                "data_mining",
                "--mining-parquet",
                str(iter2_mined),
                "--mining-candidates",
                str(iter2_mining_candidates),
                "--mining-summary",
                str(iter2_mining_summary),
                "--mining-history",
                str(mining_history),
                "--mining-history-summary",
                str(iter2_history_summary),
                "--mining-target-embeddings",
                str(iter2_target_embeddings),
                "--mining-source-embeddings",
                str(iter2_source_embeddings),
                "--mining-count",
                str(iter2_history_result["selected_count"]),
            )

            iter2_assemble_dir = results / "iter2/assemble"
            iter2_mined_sharegpt = write_json(
                iter2_assemble_dir / "mined_sharegpt.json",
                [record("mining_iter2.png", "golden.png", "OK")],
            )
            iter2_records, iter2_summary = assemble_training_json.assemble(
                combined,
                [iter2_mined_sharegpt, iter2_synthetic],
                dedupe=True,
                validation_paths=[
                    annotation_paths["proxy"],
                    annotation_paths["benchmark"],
                ],
            )
            iter2_combined = write_json(
                iter2_assemble_dir / "train_iter_2.json", iter2_records
            )
            iter2_assemble_summary = write_json(
                iter2_assemble_dir / "assemble_summary.json", iter2_summary
            )
            self.assertEqual(len(iter2_records), 4)
            # Regression: this commit used to roll back because iter1's
            # synthetic target was absent from both Mining and iter2 synthetic.
            commit(
                "iter2",
                "assemble_data",
                "--mined-sharegpt",
                str(iter2_mined_sharegpt),
                "--combined-training",
                str(iter2_combined),
                "--assemble-summary",
                str(iter2_assemble_summary),
            )
            iter2_validation_summary = validate_sharegpt.validate_records(
                iter2_records,
                media_root=root,
                require_files=False,
            )
            iter2_validation_report = write_json(
                results / "iter2/validate/validation_report.json",
                iter2_validation_summary,
            )
            commit(
                "iter2",
                "validate_data",
                "--validation-report",
                str(iter2_validation_report),
            )
            train("iter2")
            # max_iterations=2, so iter2 stops at the gate and skips Proxy.
            evaluate_arc("iter2", correct=False, continuing=False)
            state = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(state["current_iteration"], 2)
            self.assertEqual(
                state["iterations"]["iter2"]["stage_completed"],
                "benchmark_metrics",
            )
            commit(
                "iter2",
                "loop_stop",
                "--stop-reason",
                "max_iterations",
                "--final-report",
                str(results / "DEFT_Loop_Report.html"),
            )
            state = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["current_iteration"], 2)
            self.assertEqual(state["events"][-1]["stage"], "loop_stop")


if __name__ == "__main__":
    unittest.main()
