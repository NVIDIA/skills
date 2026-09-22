#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import yaml

SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
for module_name in (
    "align_token_usage",
    "commit_stage",
    "finalize_run",
    "init_deft_state",
    "metric_contract",
    "prepare_inference_spec",
    "stage_backbone",
    "render_report",
):
    sys.modules.pop(module_name, None)
sys.path.insert(0, str(SCRIPTS))

import align_token_usage  # noqa: E402
import commit_stage  # noqa: E402
import init_deft_state  # noqa: E402
import metric_contract  # noqa: E402
import render_report  # noqa: E402
import deft_exec  # noqa: E402
import finalize_run  # noqa: E402
import prepare_inference_spec  # noqa: E402
import stage_backbone  # noqa: E402
import resolve_mining_pool  # noqa: E402


FULL_RCA_REPORT = """# VCN Gap Analysis Report: fixture

## 1. Verdict
Fixture verdict.

## 2. Threshold Selection
Fixture threshold selection.

## 3. Kept-Sample Weakness Distribution
Fixture weakness distribution.

## 4. Top-K Weakest Samples
Fixture weakest samples.

## 5. Visual Spot Check
Fixture visual evidence.

## 6. Per-Label Breakdown
Fixture label breakdown.

## 7. Recommended Actions
Fixture actions.
"""


def _write_reachable_rca_fixture(
    output_dir: pathlib.Path,
    *,
    report_text: str = FULL_RCA_REPORT,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    output_dir.mkdir(parents=True)
    gaps = output_dir / "kpi_gaps.parquet"
    gaps.write_bytes(b"PAR1fixture")
    (output_dir / "threshold.txt").write_text("0.125\n", encoding="utf-8")
    (output_dir / "weak_samples_breakdown.txt").write_text(
        "PASS: 1 kept row\n", encoding="utf-8"
    )
    report = output_dir / "RCA_Report.md"
    report.write_text(report_text, encoding="utf-8")
    images = output_dir / "rca_images"
    images.mkdir()
    (images / "weak-pass.png").write_bytes(b"fixture-image")
    return gaps, report, images


class ReportRenderingTests(unittest.TestCase):
    def test_finalize_run_creates_handoff_before_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = self._state(results)
            state["status"] = "in_progress"
            checkpoint = results / "iter1/train/model.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            # The existing AOI host filename intentionally differs from the
            # container-side filename in baseline_spec.yaml.
            backbone = results / "backbone/c_radio_v2_b.safetensors"
            backbone.parent.mkdir(parents=True)
            backbone.write_bytes(b"backbone")
            (backbone.parent / "vit_large_dinov3.safetensors").write_bytes(b"decoy")
            training_spec = SKILL_ROOT / "references/baseline_spec.yaml"
            state["config"].update(
                {
                    "specs_file": str(training_spec),
                    "backbone_weight_dir": str(backbone.parent),
                    "images_dir": str(results / "images"),
                }
            )
            state["iterations"]["iter1"]["best_ckpt_path"] = str(checkpoint)
            state["iterations"]["iter1"]["training_spec"] = str(training_spec)
            (results / "deft_state.json").write_text(json.dumps(state))

            self.assertEqual(
                finalize_run.main(
                    [
                        "--results-dir", str(results),
                        "--iter-label", "iter1",
                        "--stop-reason", "metric_met",
                        "--duration-sec", "1",
                    ]
                ),
                0,
            )
            committed = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(committed["status"], "complete")
            self.assertTrue((results / "best_model.json").is_file())
            self.assertTrue((results / "best_model_inference_spec.yaml").is_file())
            self.assertEqual(
                committed["final_artifacts"]["best_model_json"],
                str((results / "best_model.json").resolve()),
            )
            handoff = json.loads((results / "best_model.json").read_text())
            self.assertEqual(handoff["backbone"], str(backbone.resolve()))
            self.assertEqual(
                handoff["backbone_container_path"],
                "/data/pretrained_models/C-RADIOv2_B.pth",
            )
            self.assertEqual(
                handoff["backbone_type"],
                "c_radio_v2_vit_base_patch16_224",
            )
            self.assertFalse(handoff["backbone_frozen"])
            inference_spec = yaml.safe_load(
                (results / "best_model_inference_spec.yaml").read_text()
            )
            self.assertEqual(inference_spec["train"]["classify"]["loss"], "ce")
            self.assertEqual(
                inference_spec["inference"]["checkpoint"],
                prepare_inference_spec.CHECKPOINT_MOUNT,
            )

    def test_airgap_guard_blocks_install_and_forces_no_pull(self) -> None:
        policy = {"network_mode": "airgap"}
        with self.assertRaisesRegex(ValueError, "package installation"):
            deft_exec._reject_airgap(["python3", "-m", "pip", "install", "x"], policy)
        self.assertEqual(
            deft_exec._with_no_pull(["docker", "run", "image:tag", "true"], policy),
            ["docker", "run", "--pull=never", "image:tag", "true"],
        )
        guarded = deft_exec._with_offline_container_env(
            ["docker", "run", "--pull=never", "image:tag", "true"], policy
        )
        self.assertIn("--env=HF_HUB_OFFLINE=1", guarded)
        with self.assertRaisesRegex(ValueError, "cannot override HF_HUB_OFFLINE"):
            deft_exec._reject_airgap(
                ["sudo", "docker", "run", "--env", "HF_HUB_OFFLINE=0", "image:tag"],
                policy,
            )

    def test_far_at_recall_contract_is_percentage_safe(self) -> None:
        args = init_deft_state._build_parser().parse_args(
            [
                "--results-dir", "/tmp/results",
                "--workspace", "/tmp/workspace",
                "--kpi-target", "FAR < 10% at Recall=100%",
                "--max-iterations", "1",
                "--num-gpus", "1",
                "--gpu-model", "test GPU",
                "--num-epochs", "1",
                "--num-sdg", "1",
                "--project", "nvpcb",
                "--step", "1",
            ]
        )
        contract, _ = init_deft_state._build_metric_contract(args)
        self.assertEqual(contract["name"], "far_pct")
        self.assertEqual(contract["target"], 10.0)
        self.assertEqual(contract["unit"], "%")
        self.assertEqual(contract["constraints"][0]["name"], "recall_pct")
        self.assertEqual(contract["constraints"][0]["target"], 100.0)
        with self.assertRaisesRegex(ValueError, "does not match"):
            metric_contract.result_from_iteration(
                {"metric_result": {"name": "far_pct", "value": 5, "unit": "", "constraints": {}}},
                contract,
            )

    def test_mining_pool_resolves_directory_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            images = root / "images"
            source = images / "board/scan/PerComponent/R1@1_SolderLight.jpg"
            golden = images / "golden/images/boardBOT/R1@1_SolderLight.jpg"
            source.parent.mkdir(parents=True)
            golden.parent.mkdir(parents=True)
            source.write_bytes(b"source")
            golden.write_bytes(b"golden")
            pool = root / "augmentation/mining_pool/mining_pool.csv"
            pool.parent.mkdir(parents=True)
            pool.write_text(
                "input_path,golden_path,label,object_name\n"
                "board/scan/PerComponent,golden/images/boardBOT/,PASS,R1@1\n",
                encoding="utf-8",
            )
            output = root / "resolved.csv"
            summary = resolve_mining_pool.resolve(pool, images, output)
            self.assertEqual(summary["rows"], 1)
            self.assertIn(
                "board/scan/PerComponent/R1@1_SolderLight.jpg",
                output.read_text(),
            )

    def test_allocation_proof_is_summed_and_rejects_invalid_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            allocation = pathlib.Path(temporary) / "allocation.json"
            allocation.write_text(
                json.dumps({"bridge": 3, "missing": 2}), encoding="utf-8"
            )
            path, total = commit_stage._required_allocation(
                allocation.resolve(), "--anomalygen-allocation"
            )
            self.assertEqual(path, str(allocation.resolve()))
            self.assertEqual(total, 5)

            allocation.write_text(json.dumps({"bridge": True}), encoding="utf-8")
            with self.assertRaises(ValueError):
                commit_stage._required_allocation(
                    allocation.resolve(), "--anomalygen-allocation"
                )

    def test_stage_commit_validates_duration_by_skip_contract(self) -> None:
        base = [
            "--results-dir",
            "/tmp/deft-duration-contract",
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
            "/tmp/deft-duration-contract",
            "--iter-label",
            "iter1",
            "--stage",
            "anomalygen",
            "--skip",
            "--summary",
            "routing produced zero rows",
        ]
        self.assertEqual(commit_stage.main([*skip, "--duration-sec", "-1"]), 2)

    def test_skipped_stage_accepts_zero_and_records_explicit_reason(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = self._state(results)
            state["status"] = "in_progress"
            state["iterations"]["iter1"] = {
                "status": "in_progress",
                "stage_completed": "routing",
            }
            state["events"] = []
            routed = results / "iter1/routing/anomalygen.parquet"
            routed.parent.mkdir(parents=True)
            pq.write_table(
                pa.table({"filepath": pa.array([], type=pa.string())}), routed
            )
            state["iterations"]["iter1"]["routing_anomalygen_parquet"] = str(
                routed.resolve()
            )
            (results / "deft_state.json").write_text(json.dumps(state))

            summary = "routing produced zero rows"
            self.assertEqual(
                commit_stage.main(
                    [
                        "--results-dir",
                        str(results),
                        "--iter-label",
                        "iter1",
                        "--stage",
                        "anomalygen",
                        "--skip",
                        "--duration-sec",
                        "0",
                        "--summary",
                        summary,
                    ]
                ),
                0,
            )
            committed = json.loads((results / "deft_state.json").read_text())
            phase = committed["iterations"]["iter1"]
            event = committed["events"][-1]
            self.assertTrue(phase["anomalygen_skipped"])
            self.assertEqual(phase["anomalygen_skip_reason"], summary)
            self.assertEqual(event["status"], "skipped")
            self.assertEqual(event["skip_reason"], summary)
            self.assertEqual(event["duration_sec"], 0)

    def test_rca_target_defects_serialize_one_and_repeated_labels(self) -> None:
        cases = (
            (["missing"], ["missing"]),
            (
                ["missing", "shift", "lifted_lead"],
                ["missing", "shift", "lifted_lead"],
            ),
            (
                ["missing", "shift", "missing", "lifted_lead"],
                ["missing", "shift", "lifted_lead"],
            ),
        )
        for labels, expected in cases:
            with (
                self.subTest(labels=labels),
                tempfile.TemporaryDirectory() as temporary,
            ):
                results = pathlib.Path(temporary)
                state = self._state(results)
                state["status"] = "in_progress"
                state["iterations"]["iter1"] = {"status": "in_progress"}
                state["events"] = []
                (results / "deft_state.json").write_text(json.dumps(state))
                gaps, report, _ = _write_reachable_rca_fixture(
                    results / "iter1/rca"
                )
                threshold = gaps.parent / "threshold.txt"
                breakdown = gaps.parent / "weak_samples_breakdown.txt"
                label_args = [
                    item
                    for label in labels
                    for item in ("--rca-target-defect", label)
                ]

                self.assertEqual(
                    commit_stage.main(
                        [
                            "--results-dir",
                            str(results),
                            "--iter-label",
                            "iter1",
                            "--stage",
                            "rca",
                            "--rca-gaps",
                            str(gaps),
                            "--rca-report",
                            str(report),
                            *label_args,
                            "--duration-sec",
                            "1",
                            "--summary",
                            "RCA complete",
                        ]
                    ),
                    0,
                )
                committed = json.loads((results / "deft_state.json").read_text())
                self.assertEqual(
                    committed["iterations"]["iter1"]["rca_target_defects"],
                    expected,
                )
                self.assertEqual(
                    committed["iterations"]["iter1"]["rca_threshold_file"],
                    str(threshold.resolve()),
                )
                self.assertEqual(
                    committed["iterations"]["iter1"]["rca_breakdown_file"],
                    str(breakdown.resolve()),
                )

    def test_rca_commit_requires_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = self._state(results)
            state["status"] = "in_progress"
            state["iterations"]["iter1"] = {"status": "in_progress"}
            state["events"] = []
            state_path = results / "deft_state.json"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            output_dir = results / "iter1/rca"
            output_dir.mkdir(parents=True)
            gaps = output_dir / "kpi_gaps.parquet"
            gaps.write_bytes(b"PAR1fixture")
            (output_dir / "threshold.txt").write_text("0.125\n")
            (output_dir / "weak_samples_breakdown.txt").write_text("PASS: 1\n")

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = commit_stage.main(
                    [
                        "--results-dir", str(results),
                        "--iter-label", "iter1",
                        "--stage", "rca",
                        "--rca-gaps", str(gaps),
                        "--duration-sec", "1",
                        "--summary", "RCA complete",
                    ]
                )

            self.assertEqual(rc, 2)
            self.assertIn("--rca-report is required", stderr.getvalue())
            self.assertNotIn("rca_report", json.loads(state_path.read_text())["iterations"]["iter1"])

    def test_rca_commit_rejects_report_missing_required_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = self._state(results)
            state["status"] = "in_progress"
            state["iterations"]["iter1"] = {"status": "in_progress"}
            state["events"] = []
            state_path = results / "deft_state.json"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            gaps, report, _ = _write_reachable_rca_fixture(
                results / "iter1/rca",
                report_text="# Fixture\n\n## 1. Verdict\nIncomplete report.\n",
            )

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = commit_stage.main(
                    [
                        "--results-dir", str(results),
                        "--iter-label", "iter1",
                        "--stage", "rca",
                        "--rca-gaps", str(gaps),
                        "--rca-report", str(report),
                        "--duration-sec", "1",
                        "--summary", "RCA complete",
                    ]
                )

            self.assertEqual(rc, 2)
            message = stderr.getvalue()
            self.assertIn("missing required section heading(s)", message)
            self.assertIn("Threshold Selection", message)
            self.assertIn("references/output-template.md", message)

    def test_rca_commit_records_report_and_image_paths(self) -> None:
        manifest = json.loads(commit_stage.RCA_ARTIFACT_MANIFEST.read_text())
        classes = manifest["artifact_classes"]
        self.assertEqual(
            [item["path"] for item in classes["container_produced_required"]],
            ["kpi_gaps.parquet", "threshold.txt", "weak_samples_breakdown.txt"],
        )
        self.assertEqual(
            [item["path"] for item in classes["agent_produced_required"]],
            ["RCA_Report.md"],
        )
        self.assertEqual(
            [
                item["path"]
                for item in classes["agent_produced_conditionally_required"]
            ],
            ["rca_images/"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = self._state(results)
            state["status"] = "in_progress"
            state["iterations"]["iter1"] = {"status": "in_progress"}
            state["events"] = []
            (results / "deft_state.json").write_text(json.dumps(state))
            gaps, report, images = _write_reachable_rca_fixture(
                results / "iter1/rca"
            )

            rc = commit_stage.main(
                [
                    "--results-dir", str(results),
                    "--iter-label", "iter1",
                    "--stage", "rca",
                    "--rca-gaps", str(gaps),
                    "--rca-report", str(report),
                    "--duration-sec", "1",
                    "--summary", "RCA complete",
                ]
            )

            self.assertEqual(rc, 0)
            phase = json.loads((results / "deft_state.json").read_text())["iterations"]["iter1"]
            self.assertEqual(phase["rca_report"], str(report.resolve()))
            self.assertEqual(phase["rca_images_dir"], str(images.resolve()))
            self.assertEqual(phase["rca_gaps_parquet"], str(gaps.resolve()))

    def test_rca_commit_accepts_unreachable_abridged_report_without_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = self._state(results)
            state["status"] = "in_progress"
            state["iterations"]["iter1"] = {"status": "in_progress"}
            state["events"] = []
            (results / "deft_state.json").write_text(json.dumps(state))
            output_dir = results / "iter1/rca"
            output_dir.mkdir(parents=True)
            unreachable = output_dir / "unreachable_kpi.txt"
            unreachable.write_text("maximum recall 0.75; target 1.0\n")
            report = output_dir / "RCA_Report.md"
            report.write_text(
                "# VCN Gap Analysis Report: fixture\n\n"
                "## KPI Unreachable\n"
                "No threshold meets the target; retrain or relabel.\n"
            )

            rc = commit_stage.main(
                [
                    "--results-dir", str(results),
                    "--iter-label", "iter1",
                    "--stage", "rca",
                    "--rca-report", str(report),
                    "--duration-sec", "1",
                    "--summary", "RCA found KPI unreachable",
                ]
            )

            self.assertEqual(rc, 0)
            phase = json.loads((results / "deft_state.json").read_text())["iterations"]["iter1"]
            self.assertEqual(phase["rca_report"], str(report.resolve()))
            self.assertEqual(
                phase["rca_unreachable_kpi"], str(unreachable.resolve())
            )
            self.assertNotIn("rca_images_dir", phase)
            self.assertNotIn("rca_gaps_parquet", phase)

    def test_rca_target_defect_help_and_errors_explain_repeatable_flag(self) -> None:
        help_text = commit_stage._parser().format_help()
        self.assertIn("[--rca-target-defect <label>]...", help_text)
        base = [
            "--results-dir",
            "/tmp/deft-rca-label-contract",
            "--iter-label",
            "baseline",
            "--stage",
            "rca",
            "--rca-gaps",
            "/tmp/gaps.parquet",
            "--duration-sec",
            "1",
            "--summary",
            "RCA complete",
        ]
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            commit_stage._parser().parse_args(
                [*base, "--rca-target-defect", "   "]
            )
        self.assertIn("label must not be empty", stderr.getvalue())

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            commit_stage._parser().parse_args(
                [
                    *base,
                    "--rca-target-defect",
                    "missing",
                    "shift",
                    "lifted_lead",
                ]
            )
        message = stderr.getvalue()
        self.assertIn("accepts exactly one label per occurrence", message)
        self.assertIn("repeat the flag for each label", message)

    def test_loop_stop_is_recorded_only_in_deft_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = self._state(results)
            # build85 copied a completed run into a different CI artifact directory.
            # The state must stay portable instead of rejecting its original path.
            state["results_dir"] = "/original/run/location"
            state["status"] = "in_progress"
            state["events"] = state["events"][:1]
            (results / "deft_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            best_model = results / "best_model.json"
            best_model.write_text("{}\n", encoding="utf-8")
            inference_spec = results / "best_model_inference_spec.yaml"
            inference_spec.write_text("model: {}\n", encoding="utf-8")

            rc = commit_stage.main(
                [
                    "--results-dir",
                    str(results),
                    "--iter-label",
                    "iter1",
                    "--stage",
                    "loop_stop",
                    "--summary",
                    "iteration budget reached",
                    "--duration-sec",
                    "1",
                    "--stop-reason",
                    "metric_met",
                    "--best-model",
                    str(best_model),
                    "--inference-spec",
                    str(inference_spec),
                ]
            )

            self.assertEqual(rc, 0)
            committed = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(committed["status"], "complete")
            self.assertEqual(committed["events"][-1]["stage"], "loop_stop")
            self.assertEqual(committed["events"][-1]["seq"], 2)
            self.assertFalse((results / "loop_log.jsonl").exists())

    def test_loop_stop_requires_complete_baseline_and_final_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = self._state(results)
            state["status"] = "in_progress"
            state["iterations"]["iter1"]["status"] = "in_progress"
            state["events"] = []
            state_path = results / "deft_state.json"
            state_path.write_text(json.dumps(state), encoding="utf-8")

            rc = commit_stage.main(
                [
                    "--results-dir",
                    str(results),
                    "--iter-label",
                    "iter1",
                    "--stage",
                    "loop_stop",
                    "--summary",
                    "premature stop",
                    "--duration-sec",
                    "1",
                ]
            )

            self.assertEqual(rc, 2)
            committed = json.loads(state_path.read_text())
            self.assertEqual(committed["status"], "in_progress")
            self.assertEqual(committed["events"], [])

    def test_token_alignment_updates_state_events_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            state_path = root / "deft_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "events": [
                            {
                                "seq": 1,
                                "ts": "2026-08-04T00:01:00+00:00",
                                "iter": "baseline",
                                "stage": "train",
                                "context_tokens": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            transcript = root / "session.jsonl"
            transcript.write_text(
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": "2026-08-04T00:00:30+00:00",
                        "message": {
                            "model": "test-model",
                            "usage": {
                                "input_tokens": 10,
                                "output_tokens": 2,
                                "cache_read_input_tokens": 3,
                                "cache_creation_input_tokens": 4,
                            },
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            state, events, messages = align_token_usage.align(
                state_path, [transcript]
            )
            align_token_usage.write_atomic(state_path, state)

            committed = json.loads(state_path.read_text())
            self.assertEqual(len(events), 1)
            self.assertEqual(len(messages), 1)
            self.assertEqual(committed["status"], "complete")
            self.assertEqual(committed["events"][0]["context_tokens"], 17)
            self.assertEqual(committed["events"][0]["tokens"]["output"], 2)

    def test_stage_commit_validates_mined_count_and_real_logs(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = self._state(results)
            state["status"] = "in_progress"
            state["iterations"]["iter1"] = {
                "status": "in_progress",
                "stage_completed": "routing",
            }
            state["events"] = []
            (results / "deft_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )

            phase_root = results / "iter1"
            artifacts = {}
            for name in (
                "summary.csv",
                "history-summary.json",
                "target.log",
                "source.log",
                "knn.log",
            ):
                path = phase_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("command completed with 0 rows\n", encoding="utf-8")
                artifacts[name] = path
            for name in ("mined.parquet", "candidates.parquet", "target.parquet", "source.parquet"):
                path = phase_root / name
                pq.write_table(pa.table({"filepath": pa.array([], type=pa.string())}), path)
                artifacts[name] = path
            history = results / "mining_history.json"
            history.write_text("{}\n", encoding="utf-8")

            rc = commit_stage.main(
                [
                    "--results-dir", str(results),
                    "--iter-label", "iter1",
                    "--stage", "data_mining",
                    "--summary", "mining completed with no retained rows",
                    "--duration-sec", "1",
                    "--mining-parquet", str(artifacts["mined.parquet"]),
                    "--mining-candidates", str(artifacts["candidates.parquet"]),
                    "--mining-summary", str(artifacts["summary.csv"]),
                    "--mining-history", str(history),
                    "--mining-history-summary", str(artifacts["history-summary.json"]),
                    "--mining-target-embeddings", str(artifacts["target.parquet"]),
                    "--mining-source-embeddings", str(artifacts["source.parquet"]),
                    "--mining-target-log", str(artifacts["target.log"]),
                    "--mining-source-log", str(artifacts["source.log"]),
                    "--mining-knn-log", str(artifacts["knn.log"]),
                    "--mining-count", "0",
                ]
            )

            self.assertEqual(rc, 0)
            committed = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(
                committed["iterations"]["iter1"]["mining_mined_count"], 0
            )
            self.assertEqual(committed["events"][-1]["stage"], "data_mining")

            combined = phase_root / "dataset" / "train_combined_iter1.csv"
            combined.parent.mkdir(parents=True, exist_ok=True)
            combined.write_text(
                "input_path,golden_path,label,object_name,source\n"
                "input,golden,PASS,part,base_train\n",
                encoding="utf-8",
            )
            provenance = phase_root / "dataset" / "provenance.csv"
            provenance.write_text("source\nbase_train\n", encoding="utf-8")
            validation = phase_root / "dataset" / "merge_validation.json"
            validation.write_text(
                '{"rows_checked": 1, "missing_file_count": 0}\n',
                encoding="utf-8",
            )
            rc = commit_stage.main(
                [
                    "--results-dir", str(results),
                    "--iter-label", "iter1",
                    "--stage", "data_merge",
                    "--summary", "validated combined training CSV",
                    "--duration-sec", "1",
                    "--combined-csv", str(combined),
                    "--provenance-csv", str(provenance),
                    "--merge-validation-report", str(validation),
                ]
            )

            self.assertEqual(rc, 0)
            committed = json.loads((results / "deft_state.json").read_text())
            self.assertEqual(committed["events"][-1]["stage"], "data_merge")

    def test_backbone_profiles_preserve_default_and_cover_dinov3(self) -> None:
        expected_dinov3 = {
            "vit_small_dinov3",
            "vit_small_plus_dinov3",
            "vit_base_dinov3",
            "vit_large_dinov3",
            "vit_huge_plus_dinov3",
            "vit_7b_dinov3",
        }
        self.assertTrue(
            expected_dinov3.issubset(stage_backbone.BACKBONE_PROFILES)
        )

        args = SimpleNamespace(
            backbone_type=stage_backbone.DEFAULT_BACKBONE_TYPE,
            repo_id=None,
            filename=None,
            stage_name=None,
        )
        self.assertEqual(
            stage_backbone.resolve_source(args),
            (
                stage_backbone.DEFAULT_REPO_ID,
                stage_backbone.DEFAULT_FILENAME,
                stage_backbone.DEFAULT_STAGE_NAME,
            ),
        )

        args.backbone_type = "vit_large_dinov3"
        self.assertEqual(
            stage_backbone.resolve_source(args),
            (
                "timm/vit_large_patch16_dinov3.lvd1689m",
                "model.safetensors",
                "vit_large_dinov3.safetensors",
            ),
        )

    def test_custom_source_overrides_bypass_workspace_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = pathlib.Path(temporary)
            specs = workspace / "specs"
            specs.mkdir()
            (specs / "baseline_spec.yaml").write_text(
                "model:\n"
                "  backbone:\n"
                "    type: customer_backbone\n"
            )
            destination = workspace / "custom.safetensors"
            args = SimpleNamespace(
                backbone_type=None,
                workspace=str(workspace),
                dest=str(destination),
                repo_id="customer/backbone",
                filename="weights.safetensors",
                stage_name=None,
            )

            self.assertEqual(
                stage_backbone.resolve_source(args),
                (
                    "customer/backbone",
                    "weights.safetensors",
                    stage_backbone.DEFAULT_STAGE_NAME,
                ),
            )
            self.assertEqual(
                stage_backbone.resolve_dest(args),
                str(destination.resolve()),
            )

    def test_backbone_profile_is_inferred_from_frozen_baseline_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = pathlib.Path(temporary)
            specs = workspace / "specs"
            specs.mkdir()
            baseline = specs / "baseline_spec.yaml"
            baseline.write_text(
                "model:\n"
                "  backbone:\n"
                "    type: vit_large_dinov3\n"
                "    freeze_backbone: true\n"
            )
            args = SimpleNamespace(
                backbone_type=None,
                workspace=str(workspace),
                repo_id=None,
                filename=None,
                stage_name=None,
            )

            self.assertEqual(
                stage_backbone.resolve_source(args),
                (
                    "timm/vit_large_patch16_dinov3.lvd1689m",
                    "model.safetensors",
                    "vit_large_dinov3.safetensors",
                ),
            )

            baseline.write_text(baseline.read_text().replace("true", "false"))
            with self.assertRaisesRegex(SystemExit, "freeze_backbone"):
                stage_backbone.resolve_source(args)

            args.backbone_type = "vit_large_dinov3"
            with self.assertRaisesRegex(SystemExit, "freeze_backbone"):
                stage_backbone.resolve_source(args)

    def test_backbone_mount_references_use_resolved_container_path(self) -> None:
        visual_reference = (
            SKILL_ROOT / "references" / "visual-changenet.md"
        ).read_text()
        inference_reference = (
            SKILL_ROOT / "references" / "prepare-for-inference.md"
        ).read_text()

        self.assertIn(
            '-v "${STAGED}:${BACKBONE_CONTAINER_PATH}:ro"',
            visual_reference,
        )
        self.assertIn(
            "BACKBONE_CONTAINER_PATH=$(jq -er",
            inference_reference,
        )
        self.assertIn(
            '-v "${BACKBONE}:${BACKBONE_CONTAINER_PATH}:ro"',
            inference_reference,
        )
        self.assertNotIn(
            ".backbone ${RESULTS_DIR}/best_model.json):"
            "/data/pretrained_models/C-RADIOv2_B.pth",
            inference_reference,
        )

    def test_configured_converted_dinov3_checkpoint_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = pathlib.Path(temporary)
            specs = workspace / "specs"
            backbone_dir = workspace / "augmentation" / "backbone"
            specs.mkdir()
            backbone_dir.mkdir(parents=True)
            converted = backbone_dir / "customer-converted.safetensors"
            converted.write_bytes(b"converted")
            (specs / "baseline_spec.yaml").write_text(
                "model:\n"
                "  backbone:\n"
                "    type: vit_large_dinov3\n"
                "    pretrained_backbone_path: "
                "/data/pretrained_models/customer-converted.safetensors\n"
                "    freeze_backbone: true\n"
            )
            args = SimpleNamespace(
                backbone_type=None,
                workspace=str(workspace),
                dest=None,
                repo_id=None,
                filename=None,
                stage_name=None,
            )

            self.assertEqual(
                stage_backbone.resolve_dest(args), str(converted.resolve())
            )

            converted.unlink()
            with self.assertRaisesRegex(SystemExit, "configured DINOv3"):
                stage_backbone.resolve_dest(args)

    def test_handoff_resolves_exact_configured_backbone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backbone_dir = pathlib.Path(temporary)
            expected = backbone_dir / "vit_large_dinov3.safetensors"
            expected.write_bytes(b"converted")
            (backbone_dir / "unrelated.pth").write_bytes(b"decoy")
            train_spec = {
                "model": {
                    "backbone": {
                        "type": "vit_large_dinov3",
                        "pretrained_backbone_path": (
                            "/data/pretrained_models/vit_large_dinov3.safetensors"
                        ),
                        "freeze_backbone": True,
                    }
                }
            }
            state = {"config": {"backbone_weight_dir": str(backbone_dir)}}

            resolved = prepare_inference_spec._resolve_backbone(train_spec, state)
            self.assertEqual(resolved[0], str(expected.resolve()))
            self.assertEqual(
                resolved[1],
                "/data/pretrained_models/vit_large_dinov3.safetensors",
            )
            self.assertEqual(resolved[2:], ("vit_large_dinov3", True))

            expected.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "exact pretrained"):
                prepare_inference_spec._resolve_backbone(train_spec, state)

    def test_handoff_rejects_missing_backbone_path(self) -> None:
        train_spec = {
            "model": {
                "backbone": {
                    "type": "vit_large_dinov3",
                    "pretrained_backbone_path": None,
                    "freeze_backbone": True,
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "has no .*pretrained"):
            prepare_inference_spec._resolve_backbone(train_spec, {"config": {}})

    def test_handoff_rejects_unrelated_sole_backbone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backbone_dir = pathlib.Path(temporary)
            (backbone_dir / "vit_large_dinov3.safetensors").write_bytes(b"wrong")
            train_spec = {
                "model": {
                    "backbone": {
                        "type": "c_radio_v2_vit_base_patch16_224",
                        "pretrained_backbone_path": (
                            "/data/pretrained_models/C-RADIOv2_B.pth"
                        ),
                        "freeze_backbone": False,
                    }
                }
            }
            state = {"config": {"backbone_weight_dir": str(backbone_dir)}}
            with self.assertRaisesRegex(FileNotFoundError, "exact pretrained"):
                prepare_inference_spec._resolve_backbone(train_spec, state)

    def _state(self, results: pathlib.Path) -> dict:
        contract = {
            "name": "escape_cost",
            "display_name": "Weighted escape cost",
            "operator": "<=",
            "target": 0.02,
            "unit": "cost/board",
            "evaluator": {
                "type": "artifact",
                "producer": "test",
                "path_template": str(
                    results / "{iter_label}/evaluate/metric_result.json"
                ),
            },
            "constraints": [],
        }
        return {
            "version": 4,
            "started_at": "2026-08-04T00:00:00+00:00",
            "status": "complete",
            "kpi_target": "Weighted escape cost <= 0.02 cost/board",
            "metric_contract": contract,
            "results_dir": str(results),
            "max_iterations": 2,
            "current_iteration": 1,
            "config": {
                "kpi_test_csv": str(results / "kpi.csv"),
                "training_csv": str(results / "training.csv"),
                "num_gpus": 1,
                "gpu_model": "NVIDIA RTX PRO 6000 Blackwell (96 GB)",
                "mining_filter": {"metric": "cosine", "min_similarity": 0.9},
            },
            "iterations": {
                "baseline": {
                    "status": "complete",
                    "stage_completed": "evaluate",
                    "best_ckpt_path": str(results / "baseline/train/model.ckpt"),
                    "threshold": 0.5,
                    "metric_result": {
                        "name": "escape_cost",
                        "value": 0.031,
                        "unit": "cost/board",
                        "constraints": {},
                        "passed": False,
                    },
                },
                "iter1": {
                    "status": "complete",
                    "stage_completed": "evaluate",
                    "best_ckpt_path": "</div><script>alert('x')</script>",
                    "threshold": 0.47,
                    "mining_mined_count": 8,
                    "metric_result": {
                        "name": "escape_cost",
                        "value": 0.018,
                        "unit": "cost/board",
                        "constraints": {},
                        "passed": True,
                    },
                },
            },
            "events": [
                {
                    "seq": 1,
                    "ts": "2026-08-04T00:01:00Z",
                    "iter": "iter1",
                    "stage": "evaluate",
                    "status": "ok",
                    "summary": "done",
                    "duration_sec": 120,
                },
                {
                    "seq": 2,
                    "ts": "2026-08-04T00:02:00Z",
                    "iter": "iter1",
                    "stage": "loop_stop",
                    "status": "ok",
                    "summary": "target met",
                    "duration_sec": 1,
                },
            ],
            "_completed_step_values": [],
            "_status_values": [],
        }

    def test_renders_release_style_template_and_escapes_disk_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = self._state(results)
            (results / "training.csv").write_text(
                "input_path,label\n" + "".join(f"base-{index}.png,PASS\n" for index in range(118)),
                encoding="utf-8",
            )
            (results / "kpi.csv").write_text(
                "input_path,label\na.png,PASS\nb.png,PASS\nc.png,NG-Missing\n",
                encoding="utf-8",
            )
            combined = results / "iter1-combined.csv"
            combined.write_text(
                "input_path,label\n" + "".join(f"train-{index}.png,PASS\n" for index in range(120)),
                encoding="utf-8",
            )
            mining_summary = results / "iter1-knn-summary.csv"
            mining_summary.write_text(
                "candidate_count,kept_count,rejected_count,similarity_threshold\n5,2,3,0.9\n",
                encoding="utf-8",
            )
            sdg_csv = results / "iter1-sdg.csv"
            sdg_csv.write_text(
                "image,label\n"
                + "".join(f"sdg-{index}.png,NG\n" for index in range(5)),
                encoding="utf-8",
            )
            state["iterations"]["iter1"]["combined_training_csv"] = str(combined)
            state["iterations"]["iter1"]["mining_summary"] = str(mining_summary)
            state["iterations"]["iter1"]["anomalygen_sdg_csv"] = str(sdg_csv)
            (results / "deft_state.json").write_text(json.dumps(state), encoding="utf-8")
            output = render_report.render(results)
            text = output.read_text(encoding="utf-8")
            self.assertIn("DEFT Loop Final Report", text)
            self.assertIn("--nvidia-green: #76b900", text)
            self.assertIn("KPI MET", text)
            self.assertIn("Run Configuration &amp; Outcome", text)
            self.assertIn("NVIDIA RTX PRO 6000 Blackwell (96 GB)", text)
            self.assertIn("1 iters × ~2m 0s = 2m 0s total time", text)
            self.assertIn("KNN Raw Mined", text)
            self.assertIn("SDG Generated", text)
            self.assertIn("New Unique Images (After Dedup)", text)
            self.assertIn(">118</td>", text)
            self.assertIn(">120</td>", text)
            self.assertIn(">+2</td>", text)
            self.assertNotRegex(text, r"\{\{\s+[A-Z0-9_]+\s+\}\}")
            self.assertNotIn("</div><script>alert('x')</script>", text)
            self.assertIn("&lt;/div&gt;&lt;script&gt;alert", text)
            self.assertEqual(state["status"], "complete")

    def test_terminal_gap_has_no_informational_kpi_banner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = self._state(results)
            state["iterations"]["iter1"]["metric_result"]["value"] = 0.025
            (results / "deft_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )

            text = render_report.render(results).read_text(encoding="utf-8")
            self.assertIn("0.005 cost/board from target", text)
            self.assertNotIn("Best result so far", text)
            self.assertNotIn(">i</div>", text)
            self.assertNotIn("KPI MET", text)

    def test_partial_state_still_produces_a_complete_live_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = self._state(results)
            state["iterations"] = {}
            state["current_iteration"] = 0
            state["status"] = "in_progress"
            state["events"] = []
            (results / "deft_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            output = render_report.render(results)
            text = output.read_text(encoding="utf-8")
            self.assertIn("IN PROGRESS", text)
            self.assertIn("No completed evaluation yet", text)
            self.assertNotRegex(text, r"\{\{\s+[A-Z0-9_]+\s+\}\}")

    def test_inline_chart_json_cannot_close_the_script_element(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            state = self._state(results)
            state["metric_contract"]["unit"] = "</script><script>alert(1)</script>"
            for phase in state["iterations"].values():
                phase["metric_result"]["unit"] = state["metric_contract"]["unit"]
            (results / "deft_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            text = render_report.render(results).read_text(encoding="utf-8")
            self.assertNotIn('const metricUnit = "</script>', text)
            self.assertIn(r"\u003c/script\u003e", text)

    def test_initialization_hook_writes_the_live_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            results = root / "results"
            rc = init_deft_state.main(
                [
                    "--results-dir",
                    str(results),
                    "--workspace",
                    str(root / "workspace"),
                    "--kpi-target",
                    "FAR <= 1% at recall=100%",
                    "--max-iterations",
                    "1",
                    "--num-gpus",
                    "1",
                    "--gpu-model",
                    "NVIDIA RTX PRO 6000 Blackwell (96 GB)",
                    "--num-epochs",
                    "1",
                    "--num-sdg",
                    "2",
                    "--project",
                    "nvpcb",
                    "--step",
                    "1",
                    "--train-container",
                    "example/train:1",
                    "--ag-container",
                    "example/anomalygen:1",
                ]
            )
            self.assertEqual(rc, 0)
            state = json.loads((results / "deft_state.json").read_text(encoding="utf-8"))
            self.assertEqual(
                state["config"]["images_dir"],
                str((root / "workspace" / "images").resolve()),
            )
            report = results / "DEFT_Loop_Report.html"
            self.assertTrue(report.is_file())
            self.assertIn("IN PROGRESS", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
