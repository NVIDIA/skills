#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import re
import sys
import tempfile
import unittest


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
for module_name in ("align_token_usage", "metric_contract", "render_report"):
    sys.modules.pop(module_name, None)
sys.path.insert(0, str(SCRIPTS))

import align_token_usage  # noqa: E402
import render_report  # noqa: E402


CHANGE_TEMPLATE = (
    SKILL_ROOT.parent
    / "tao-run-deft-aoi"
    / "references"
    / "DEFT_Loop_Report.html"
)
COSMOS_TEMPLATE = SKILL_ROOT / "references" / "DEFT_Loop_Report.html"


def css_rules(template: pathlib.Path) -> dict[str, dict[str, str]]:
    text = template.read_text(encoding="utf-8")
    css = text.split("<style>", 1)[1].split("</style>", 1)[0]
    css = css.split("@media", 1)[0]
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    rules: dict[str, dict[str, str]] = {}
    for selector_group, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        for selector in selector_group.split(","):
            selector = selector.strip()
            if selector.startswith("/*") or selector.startswith("@"):
                continue
            declarations = rules.setdefault(selector, {})
            for declaration in body.split(";"):
                if ":" not in declaration:
                    continue
                name, value = declaration.split(":", 1)
                declarations[name.strip()] = " ".join(value.split())
    return rules


def record(label: str, prompt: str = "Inspect.") -> dict:
    return {
        "images": ["target.png", "golden.png"],
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": label},
        ],
    }


class CosmosReportRenderingTests(unittest.TestCase):
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
                                "stage": "evaluate_benchmark",
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

    def test_matches_changenet_visual_contract_and_single_column_skeleton(self) -> None:
        change = css_rules(CHANGE_TEMPLATE)
        cosmos = css_rules(COSMOS_TEMPLATE)
        contract = {
            ":root": (
                "--nvidia-green",
                "--nvidia-green-light",
                "--nvidia-yellow",
                "--nvidia-red",
                "--bg-dark",
                "--bg-card",
                "--bg-panel",
                "--text-primary",
                "--text-secondary",
                "--text-muted",
                "--border-soft",
            ),
            "body": (
                "font-family",
                "background",
                "color",
                "display",
                "flex-direction",
                "align-items",
                "gap",
                "padding",
                "font-size",
            ),
            ".card": (
                "width",
                "background",
                "border-radius",
                "padding",
                "box-shadow",
                "position",
                "overflow",
            ),
            ".card::before": ("height", "background"),
            ".header": (
                "display",
                "justify-content",
                "align-items",
                "margin-bottom",
            ),
            ".chart-title": (
                "font-size",
                "font-weight",
                "letter-spacing",
                "color",
                "line-height",
            ),
            ".chart-subtitle": (
                "font-size",
                "font-weight",
                "color",
                "margin-top",
            ),
            ".hero": (
                "width",
                "background",
                "border-radius",
                "padding",
                "box-shadow",
                "position",
                "overflow",
            ),
            ".hero::before": ("height", "background"),
            ".hero h1": (
                "font-size",
                "font-weight",
                "letter-spacing",
                "color",
            ),
            ".hero .meta": (
                "display",
                "gap",
                "font-size",
                "color",
                "margin-top",
                "flex-wrap",
            ),
            ".hero .meta-item strong": ("color", "font-weight"),
            ".hero .meta-item .value": ("color", "font-weight"),
            ".data-table": (
                "width",
                "border-collapse",
                "margin-top",
                "font-size",
            ),
            ".data-table th": (
                "text-align",
                "padding",
                "font-size",
                "font-weight",
                "letter-spacing",
                "text-transform",
                "color",
                "background",
                "border-bottom",
            ),
            ".data-table td": (
                "padding",
                "color",
                "border-bottom",
                "vertical-align",
            ),
        }
        for selector, properties in contract.items():
            with self.subTest(selector=selector):
                self.assertIn(selector, change)
                self.assertIn(selector, cosmos)
                self.assertEqual(
                    {name: change[selector][name] for name in properties},
                    {name: cosmos[selector][name] for name in properties},
                )

        cosmos_html = COSMOS_TEMPLATE.read_text(encoding="utf-8")
        self.assertIn(".page, .grid { display: contents; }", cosmos_html)
        self.assertNotIn('class="card half"', cosmos_html)
        self.assertEqual(cosmos_html.count('class="card"'), 9)
        self.assertEqual(cosmos_html.count('class="header"'), 9)

    def test_nvidia_sections_terminal_proxy_semantics_and_escaping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary) / "results"
            workspace = pathlib.Path(temporary) / "workspace"
            results.mkdir()
            annotations: dict[str, str] = {}
            inspection_prompt = "Compare <script>alert('prompt')</script> with golden."
            for role, label in (("proxy", "NG"), ("benchmark", "NG"), ("mining", "OK")):
                path = workspace / "annotations" / f"{role}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps([record(label, inspection_prompt)]), encoding="utf-8"
                )
                annotations[role] = str(path)
            contract = {
                "name": "recall_ng",
                "display_name": "NG recall",
                "operator": ">=",
                "target": 1.0,
                "unit": "",
                "evaluator": {
                    "type": "artifact",
                    "producer": "scripts/analyze_gaps.py",
                    "path_template": str(
                        results / "{iter_label}/benchmark_metrics/metric_result.json"
                    ),
                },
                "constraints": [
                    {
                        "name": "unknown_predictions",
                        "display_name": "Unknown predictions",
                        "operator": "<=",
                        "target": 0,
                        "unit": "",
                    }
                ],
            }
            state = {
                "version": 5,
                "workflow": "tao-run-deft-aoi-cosmos3",
                "started_at": "2026-08-04T00:00:00+00:00",
                "status": "complete",
                "kpi_target": "NG recall >= 1",
                "metric_contract": contract,
                "results_dir": str(results),
                "max_iterations": 1,
                "current_iteration": 1,
                "config": {
                    "platform": "docker",
                    "base_model": "nvidia/Cosmos3-Nano",
                    "annotation_mode": "bare_okng",
                    "annotations": annotations,
                    "evaluation": {"benchmark": {"sha256": "abc123"}},
                    "training": {
                        "num_nodes": 1,
                        "num_gpus": 2,
                        "gpu_model": "NVIDIA H100 80GB HBM3",
                    },
                    "anomalygen": {"num_SDG": 20},
                },
                "iterations": {
                    "iter1": {
                        "status": "complete",
                        "stage_completed": "benchmark_metrics",
                        "benchmark_results_json": str(results / "iter1/benchmark/results.json"),
                        "metric_result": {
                            "name": "recall_ng",
                            "value": 0.75,
                            "unit": "",
                            "constraints": {"unknown_predictions": 0},
                            "metrics": {
                                "accuracy": 0.8,
                                "recall_ng": 0.75,
                                "precision_ng": 1.0,
                                "f1_ng": 0.857,
                            },
                            "confusion": {
                                "fn_ng_to_ok_false_accept": 1,
                                "fp_ok_to_ng_false_reject": 0,
                            },
                        },
                    }
                },
                "_completed_step_values": [],
                "_status_values": [],
            }
            mining_summary = results / "iter1/mining/mining_summary.json"
            mining_summary.parent.mkdir(parents=True, exist_ok=True)
            mining_summary.write_text(
                json.dumps({"candidate_count": 170, "kept_count": 170}),
                encoding="utf-8",
            )
            assemble_summary = results / "iter1/assemble/assemble_summary.json"
            assemble_summary.parent.mkdir(parents=True, exist_ok=True)
            assemble_summary.write_text(
                json.dumps(
                    {
                        "output_records": 170,
                        "unique_target_images": {"new_after_dedup": 170},
                    }
                ),
                encoding="utf-8",
            )
            sdg_csv = results / "iter1/anomalygen/sdg/SDG_result.csv"
            sdg_csv.parent.mkdir(parents=True, exist_ok=True)
            sdg_csv.write_text("image,label\na.png,NG\nb.png,NG\n", encoding="utf-8")
            allocation = results / "iter1/anomalygen/sdg/allocation.json"
            allocation.write_text(json.dumps({"bridge": 20}), encoding="utf-8")
            state["iterations"]["iter1"].update(
                {
                    "mining_summary": str(mining_summary),
                    "assemble_summary": str(assemble_summary),
                    "anomalygen_sdg_csv": str(sdg_csv),
                    "anomalygen_allocation_json": str(allocation),
                    "anomalygen_amp_allocated": 20,
                }
            )
            entries = [
                {
                    "seq": 1,
                    "ts": "2026-08-04T00:01:00Z",
                    "iter": "iter1",
                    "stage": "benchmark_metrics",
                    "status": "ok",
                    "summary": "<img src=x onerror=alert(1)>",
                    "duration_sec": 12,
                },
                {
                    "seq": 2,
                    "ts": "2026-08-04T00:02:00Z",
                    "iter": "iter1",
                    "stage": "loop_stop",
                    "status": "ok",
                    "summary": "max iterations",
                    "duration_sec": 0,
                },
            ]
            state["events"] = entries
            (results / "deft_state.json").write_text(json.dumps(state), encoding="utf-8")
            output = render_report.render(results)
            text = output.read_text(encoding="utf-8")
            for heading in (
                "NVIDIA TAO · DEFT AOI",
                "Run Configuration &amp; Outcome",
                "Benchmark KPI Trend",
                "Dataset Isolation",
                "Prompt Examples",
                "Iteration Metrics",
                "Pipeline Execution",
                "Augmentation Volume",
                "Artifacts",
                "Hard Stops / Warnings",
            ):
                self.assertIn(heading, text)
            self.assertIn("not run (terminal iteration)", text)
            self.assertIn("1 node(s) · 2 GPU(s) · NVIDIA H100 80GB HBM3", text)
            self.assertIn("1 iters × ~12s = 12s total time", text)
            self.assertIn("KNN Raw Mined", text)
            self.assertIn("SDG Generated", text)
            self.assertIn("New Unique Images (After Dedup)", text)
            self.assertIn(">170</td>", text)
            self.assertIn(">+170</td>", text)
            self.assertIn(">20</td><td class=\"num\">2</td>", text)
            self.assertLess(
                text.index("Run Configuration &amp; Outcome"),
                text.index("Benchmark KPI Trend"),
            )
            self.assertLess(
                text.index("Benchmark KPI Trend"), text.index("Dataset Isolation")
            )
            self.assertIn("Proxy · Benchmark · Mining", text)
            self.assertIn("3 RECORDS", text)
            self.assertIn("Compare &lt;script&gt;alert(&#x27;prompt&#x27;)&lt;/script&gt; with golden.", text)
            self.assertNotIn(inspection_prompt, text)
            self.assertIn("Exact assistant output", text)
            self.assertNotIn("BEST RESULT RECORDED", text)
            self.assertNotIn("after the approved iteration budget", text)
            self.assertNotIn('<div class="icon">i</div>', text)
            self.assertNotIn("kpi-banner warn", text)
            self.assertIn("&lt;img src=x onerror=alert(1)&gt;", text)
            self.assertNotIn("<img src=x onerror=alert(1)>", text)
            self.assertNotRegex(text, r"\{\{\s+[A-Z0-9_]+\s+\}\}")

    def test_growth_uses_cumulative_delta_not_batch_unique_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = pathlib.Path(temporary)
            iterations: dict[str, dict[str, str]] = {}
            for number, raw, generated, total, batch_unique in (
                (1, 13, 20, 33, 33),
                (2, 10, 20, 36, 23),
            ):
                root = results / f"iter{number}"
                mining = root / "mining_summary.json"
                mining.parent.mkdir(parents=True)
                mining.write_text(json.dumps({"input_rows": raw}), encoding="utf-8")
                sdg = root / "SDG_result.csv"
                sdg.write_text(
                    "image,label\n"
                    + "".join(f"sdg-{index}.png,NG\n" for index in range(generated)),
                    encoding="utf-8",
                )
                assemble = root / "assemble_summary.json"
                assemble.write_text(
                    json.dumps(
                        {
                            "output_records": total,
                            "unique_target_images": {
                                "new_after_dedup": batch_unique
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                iterations[f"iter{number}"] = {
                    "mining_summary": str(mining),
                    "anomalygen_sdg_csv": str(sdg),
                    "assemble_summary": str(assemble),
                }

            rows = render_report._growth_rows({"iterations": iterations})
            self.assertIn(
                '<strong>Iter1</strong></td><td class="num">13</td>'
                '<td class="num">20</td><td class="num">33</td>'
                '<td class="num">33</td><td class="num">+33</td>',
                rows,
            )
            self.assertIn(
                '<strong>Iter2</strong></td><td class="num">10</td>'
                '<td class="num">20</td><td class="num">3</td>'
                '<td class="num">36</td><td class="num">+3</td>',
                rows,
            )


if __name__ == "__main__":
    unittest.main()
