#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check an NV-MONAI-Autoresearch challenge folder for common traceability drift."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import pathlib
import re
import urllib.parse
from dataclasses import dataclass
from typing import Optional

EXP_ID_RE = re.compile(r"\bexp\d{3}_[A-Za-z0-9_-]+\b")
FOCUS_ID_RE = re.compile(r"\bF\d{4,}\b")
EXPLORATION_ID_RE = re.compile(r"\bE\d{3,}\b")
TRAJECTORY_ID_RE = re.compile(r"\bT\d{4,}\b")
LITERATURE_EVIDENCE_ID_RE = re.compile(r"\bL\d{3,}\b")
URL_RE = re.compile(r"^https?://")
BASELINE_TITLE_RE = re.compile(r"^baseline\d{3}[A-Za-z0-9_-]*$", re.IGNORECASE)
BASELINE_REFERENCE_RE = re.compile(r"\bbaseline\d{3}[A-Za-z0-9_-]*\b", re.IGNORECASE)
LITERATURE_SCHEMA = "literature-evidence-v7"
LITERATURE_SCHEMA_V6 = "literature-evidence-v6"
LITERATURE_SCHEMA_V5 = "literature-evidence-v5"
LITERATURE_SCHEMA_V4 = "literature-evidence-v4"
LITERATURE_SCHEMA_V3 = "literature-evidence-v3"
LITERATURE_SCHEMA_V2 = "literature-evidence-v2"
CANDIDATE_SCHEMA = "candidate-selection-v10"
CANDIDATE_SCHEMA_V9 = "candidate-selection-v9"
CANDIDATE_SCHEMA_V8 = "candidate-selection-v8"
CANDIDATE_SCHEMA_V7 = "candidate-selection-v7"
CANDIDATE_SCHEMA_V6 = "candidate-selection-v6"
CANDIDATE_SCHEMA_V5 = "candidate-selection-v5"
CANDIDATE_SCHEMA_V4 = "candidate-selection-v4"
CANDIDATE_SCHEMA_V3 = "candidate-selection-v3"
CANDIDATE_SCHEMA_V2 = "candidate-selection-v2"
BASELINE_SCHEMA = "baseline-direction-v5"
BASELINE_SCHEMA_V4 = "baseline-direction-v4"
BASELINE_SCHEMA_V3 = "baseline-direction-v3"
BASELINE_SCHEMA_V2 = "baseline-direction-v2"
DASHBOARD_SCHEMA = "dashboard-v6"
DASHBOARD_SCHEMA_V5 = "dashboard-v5"
DASHBOARD_SCHEMA_V4 = "dashboard-v4"
DASHBOARD_SCHEMA_V3 = "dashboard-v3"
DASHBOARD_SCHEMA_V2 = "dashboard-v2"
STATE_SCHEMA = "state-stop-check-v8"
STATE_SCHEMA_V7 = "state-stop-check-v7"
STATE_SCHEMA_V6 = "state-stop-check-v6"
STATE_SCHEMA_V5 = "state-stop-check-v5"
STATE_SCHEMA_V4 = "state-stop-check-v4"
STATE_SCHEMA_V3 = "state-stop-check-v3"
STATE_SCHEMA_V2 = "state-stop-gate-v2"
EXPERIMENT_SCHEMA = "experiment-record-v2"
DISCOVERY_GATE_SCHEMA = "discovery-gate-v1"
DISCOVERY_GATE_PATH = pathlib.Path(".miagent/discovery_gate.json")
SUSPICIOUS_HUMAN_STEERING_RE = re.compile(
    r"\b("
    r"goal remains active|"
    r"target [^|.\n]{0,80}remains unmet|"
    r"current best [^|.\n]{0,80}(does not meet|below target)|"
    r"workflow (requires|treats)|"
    r"continue the workflow loop|"
    r"experiments remain stopped by prior human request|"
    r"failed [^|.\n]{0,80}engineering repairs"
    r")\b",
    re.IGNORECASE,
)
DASHBOARD_SOFT_MAX_LINES = 300
DASHBOARD_TABLE_LINE_SOFT_MAX = 520
STATE_SOFT_MAX_LINES = 80
DASHBOARD_SECTION_ROW_LIMITS = {
    "Snapshot": 16,
    "Challenge Readiness Checklist": 8,
    "Global Blockers": 8,
    "Blockers": 8,
}
DASHBOARD_BASELINE_SUBSECTION_ROW_LIMITS = {
    "Watch Next": 5,
    "Active Jobs": 20,
    "Planned Jobs": 5,
}


@dataclass
class Finding:
    level: str
    message: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("challenge_dir", help="Path to challenge/<name>.")
    parser.add_argument(
        "--json", action="store_true", help="Emit a structured consistency report on stdout."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero for warnings as well as errors; use as the non-smoke admission gate.",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "discovery"),
        default="all",
        help="Check the full challenge or only literature/candidate discovery artifacts.",
    )
    parser.add_argument(
        "--seal",
        action="store_true",
        help="After a clean strict discovery check, seal the approved scientific decision before method-specific work.",
    )
    args = parser.parse_args()

    if args.seal and (args.phase != "discovery" or not args.strict):
        parser.error("--seal requires --phase discovery --strict")

    challenge_dir = pathlib.Path(args.challenge_dir).expanduser().resolve()
    findings = (
        check_discovery(challenge_dir)
        if args.phase == "discovery"
        else check_challenge(challenge_dir)
    )
    if args.seal and not any(f.level in {"WARN", "ERROR"} for f in findings):
        findings.extend(seal_discovery(challenge_dir))
    if args.json:
        print(
            json.dumps(
                {
                    "skill": "nv-monai-autoresearch",
                    "operation": "check",
                    "phase": args.phase,
                    "strict": args.strict,
                    "passed": not any(
                        f.level == "ERROR" or (args.strict and f.level == "WARN") for f in findings
                    ),
                    "findings": [{"level": f.level, "message": f.message} for f in findings],
                },
                indent=2,
            )
        )
    else:
        for finding in findings:
            print(f"[{finding.level}] {finding.message}")
    if any(f.level == "ERROR" or (args.strict and f.level == "WARN") for f in findings):
        raise SystemExit(1)


def check_discovery(challenge_dir: pathlib.Path) -> list[Finding]:
    if not challenge_dir.exists():
        return [Finding("ERROR", f"challenge directory does not exist: {challenge_dir}")]

    findings: list[Finding] = []
    challenge_path = challenge_dir / "challenge.md"
    if not challenge_path.is_file():
        findings.append(Finding("ERROR", "missing required file: challenge.md"))
    elif uses_active_template_contract(read_text(challenge_path)):
        check_filled_template_contracts(challenge_dir, findings, discovery_only=True)

    literature_path = challenge_dir / "literature.md"
    if not literature_path.is_file():
        findings.append(Finding("ERROR", "missing required file: literature.md"))
    else:
        check_literature(literature_path, findings)
        check_human_steering(literature_path, "literature.md", findings)

    candidates_path = challenge_dir / "candidates.md"
    if not candidates_path.is_file():
        findings.append(Finding("ERROR", "missing required file: candidates.md"))
    else:
        check_candidates(candidates_path, findings)
        check_human_steering(candidates_path, "candidates.md", findings)

    if literature_path.is_file() and candidates_path.is_file():
        check_literature_candidate_composition(challenge_dir, findings)

    if not findings:
        findings.append(Finding("OK", "discovery artifacts have no consistency issues"))
    return findings


def seal_discovery(challenge_dir: pathlib.Path) -> list[Finding]:
    if challenge_dir.name == "challenge_template":
        return [Finding("ERROR", "the raw challenge template cannot be discovery-sealed")]

    candidates_path = challenge_dir / "candidates.md"
    candidates_text = read_text(candidates_path)
    approval = clean_markdown_cell(
        read_field(markdown_section(candidates_text, "Selected Workflow"), "approval status")
    ).lower()
    if not approval.startswith("approved"):
        return [Finding("ERROR", "discovery seal requires an approved candidate gate")]

    gate_path = challenge_dir / DISCOVERY_GATE_PATH
    prior_gate = read_discovery_gate(gate_path)
    artifacts = method_specific_artifacts(challenge_dir)
    if artifacts and not valid_discovery_gate_record(prior_gate):
        shown = ", ".join(str(path.relative_to(challenge_dir)) for path in artifacts[:4])
        if len(artifacts) > 4:
            shown += ", ..."
        return [
            Finding(
                "ERROR",
                "initial discovery seal cannot be created after method-specific artifacts exist: "
                + shown,
            )
        ]

    selected = first_baseline_id(
        read_field(markdown_section(candidates_text, "Selected Workflow"), "selected baseline")
    )
    record = {
        "schema": DISCOVERY_GATE_SCHEMA,
        "sealed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "literature_sha256": sha256_file(challenge_dir / "literature.md"),
        "candidates_sha256": sha256_file(candidates_path),
        "selected_baseline": selected,
    }
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = gate_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(gate_path)
    return [
        Finding(
            "OK",
            f"discovery gate sealed for {selected}: {gate_path.relative_to(challenge_dir)}",
        )
    ]


def check_discovery_gate(challenge_dir: pathlib.Path, findings: list[Finding]) -> None:
    if challenge_dir.name == "challenge_template":
        return

    candidates_text = read_text(challenge_dir / "candidates.md")
    gate_path = challenge_dir / DISCOVERY_GATE_PATH
    artifacts = method_specific_artifacts(challenge_dir)
    approval = clean_markdown_cell(
        read_field(markdown_section(candidates_text, "Selected Workflow"), "approval status")
    ).lower()
    latest_schema = has_template_schema(candidates_text, CANDIDATE_SCHEMA)
    requires_gate = (
        latest_schema and (approval.startswith("approved") or bool(artifacts))
    ) or gate_path.exists()
    if not requires_gate:
        return

    record = read_discovery_gate(gate_path)
    if not valid_discovery_gate_record(record):
        findings.append(
            Finding(
                "WARN",
                "missing or invalid .miagent/discovery_gate.json; seal discovery before method-specific work",
            )
        )
        return

    for field, source in (
        ("literature_sha256", challenge_dir / "literature.md"),
        ("candidates_sha256", challenge_dir / "candidates.md"),
    ):
        if record.get(field) != sha256_file(source):
            findings.append(
                Finding(
                    "WARN",
                    f"discovery gate is stale because {source.name} changed; rerun the discovery seal before method-specific work",
                )
            )

    selected = first_baseline_id(
        read_field(markdown_section(candidates_text, "Selected Workflow"), "selected baseline")
    )
    if record.get("selected_baseline") != selected:
        findings.append(
            Finding(
                "WARN",
                "discovery gate selected baseline does not match candidates.md; rerun the discovery seal",
            )
        )


def method_specific_artifacts(challenge_dir: pathlib.Path) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for directory_name in ("baselines", "code", "slurm", "jobs", ".venv", "venv", "env"):
        root = challenge_dir / directory_name
        if not root.exists():
            continue
        files = []
        for path in root.rglob("*"):
            try:
                if path.is_file():
                    files.append(path)
            except OSError:
                continue
        if files:
            paths.extend(files)
        else:
            paths.append(next(root.rglob("*"), root))
    for pattern in ("*.pt", "*.pth", "*.ckpt", "*.safetensors"):
        paths.extend(path for path in challenge_dir.glob(pattern) if path.is_file())
    return sorted(set(paths))


def read_discovery_gate(path: pathlib.Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def valid_discovery_gate_record(record: dict[str, object]) -> bool:
    return (
        record.get("schema") == DISCOVERY_GATE_SCHEMA
        and isinstance(record.get("sealed_at_utc"), str)
        and bool(record.get("sealed_at_utc"))
        and isinstance(record.get("literature_sha256"), str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", str(record.get("literature_sha256"))))
        and isinstance(record.get("candidates_sha256"), str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", str(record.get("candidates_sha256"))))
        and isinstance(record.get("selected_baseline"), str)
        and bool(BASELINE_TITLE_RE.fullmatch(str(record.get("selected_baseline"))))
    )


def sha256_file(path: pathlib.Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_challenge(challenge_dir: pathlib.Path) -> list[Finding]:
    findings: list[Finding] = []
    dashboard_text = ""
    modern_template_contract = False
    if not challenge_dir.exists():
        return [Finding("ERROR", f"challenge directory does not exist: {challenge_dir}")]

    challenge_path = challenge_dir / "challenge.md"
    if not challenge_path.exists():
        findings.append(Finding("ERROR", "missing required file: challenge.md"))
    else:
        challenge_text = read_text(challenge_path)
        modern_template_contract = uses_active_template_contract(challenge_text)
        if not re.search(r"^\|\s*MIAgent harness version\s*\|", challenge_text, re.MULTILINE):
            findings.append(Finding("WARN", "challenge.md missing MIAgent harness version"))
        elif challenge_dir.name != "challenge_template" and not meaningful_value(
            read_field(challenge_text, "MIAgent harness version")
        ):
            findings.append(
                Finding("WARN", "challenge.md has no concrete MIAgent version/revision")
            )
        if challenge_dir.name != "challenge_template":
            for field in (
                "objective/core problem class",
                "input / modality",
                "output / labels",
                "metrics, metric meaning, and aggregation",
            ):
                if not meaningful_value(read_field(challenge_text, field)):
                    findings.append(
                        Finding("WARN", f"challenge.md missing challenge-contract field: {field}")
                    )
    if modern_template_contract and challenge_dir.name != "challenge_template":
        check_filled_template_contracts(challenge_dir, findings)
    dashboard_path = challenge_dir / "dashboard.md"
    if not dashboard_path.exists() and (challenge_dir / "baselines").exists():
        findings.append(Finding("WARN", "missing read-first file: dashboard.md"))
    elif dashboard_path.exists():
        dashboard_text = read_text(dashboard_path)
        if not any(
            section in dashboard_text
            for section in (
                "## Challenge Readiness Checklist",
                "## Data Checklist",
                "## Harness Checklist",
            )
        ):
            findings.append(
                Finding("WARN", "dashboard.md has no Challenge Readiness Checklist section")
            )
        else:
            check_dashboard_readiness(
                dashboard_text,
                findings,
                require_status=challenge_dir.name != "challenge_template",
            )
        result_section = markdown_section(dashboard_text, "Result Table") or markdown_section(
            dashboard_text, "Result Board"
        )
        if not result_section:
            findings.append(Finding("WARN", "dashboard.md has no Result Table section"))
        else:
            result_header = set(first_markdown_table_header(result_section))
            for column in ("model / checkpoint", "training data", "evaluation data", "result"):
                if column not in result_header:
                    findings.append(
                        Finding("WARN", f"dashboard.md Result Table is missing column: {column}")
                    )
            if challenge_dir.name != "challenge_template" and (
                has_template_schema(dashboard_text, DASHBOARD_SCHEMA)
                or has_template_schema(dashboard_text, DASHBOARD_SCHEMA_V5)
            ):
                for row in first_markdown_table_rows(result_section):
                    result_type = clean_markdown_cell(row.get("result type", "")).lower()
                    if meaningful_value(result_type) and not re.search(
                        r"official|leaderboard|\bfinal\b|\bbest\b|\btop\b", result_type
                    ):
                        findings.append(
                            Finding(
                                "WARN",
                                f"dashboard.md Result Table contains non-summary result type '{result_type}'; keep intermediate runs only in Important Completed Runs",
                            )
                        )
        has_resource_budget = "## Resource Budget" in dashboard_text or re.search(
            r"^\|\s*Compute Budget\s*\|", dashboard_text, re.MULTILINE
        )
        if not has_resource_budget:
            findings.append(Finding("WARN", "dashboard.md has no resource budget field or section"))
        if (challenge_dir / "baselines").exists() and uses_active_template_contract(dashboard_text):
            dashboard_fields = [
                "Run authorization",
                "Run state",
                "Current loop",
                "Active baseline",
                "Heartbeat",
                "Last monitoring pass / next wake",
                "GPU efficiency / evidence",
                "Latest trajectory decision",
            ]
            if (
                has_template_schema(dashboard_text, DASHBOARD_SCHEMA)
                or has_template_schema(dashboard_text, DASHBOARD_SCHEMA_V5)
                or has_template_schema(dashboard_text, DASHBOARD_SCHEMA_V4)
                or has_template_schema(dashboard_text, DASHBOARD_SCHEMA_V3)
            ):
                dashboard_fields.append("Final freeze readiness")
            for field in dashboard_fields:
                if not meaningful_value(read_field(dashboard_text, field)):
                    findings.append(
                        Finding("WARN", f"dashboard.md Snapshot missing long-run field: {field}")
                    )
        if (challenge_dir / "baselines").exists() and not dashboard_baseline_sections(
            dashboard_text
        ):
            findings.append(Finding("WARN", "dashboard.md has no baseline sections"))
        check_dashboard_readability(dashboard_path, dashboard_text, findings)
        check_dashboard_baseline_sections(dashboard_text, findings)
        check_dashboard_single_active_baseline(dashboard_text, findings)
        if (challenge_dir / "baselines").exists():
            check_dashboard_planned_experiments(challenge_dir, dashboard_text, findings)
            check_dashboard_key_visualizations(challenge_dir, dashboard_text, findings)
        check_human_steering(challenge_dir / "dashboard.md", "dashboard.md", findings)

    state_path = challenge_dir / "state.md"
    if not state_path.exists() and (challenge_dir / "baselines").exists():
        findings.append(Finding("WARN", "missing read-first research memory: state.md"))
    elif state_path.exists():
        check_state(state_path, findings)

    trajectory_path = challenge_dir / "trajectory.md"
    current_state_contract = state_path.exists() and (
        uses_active_template_contract(read_text(state_path))
        or "## Current Focus Contract" in read_text(state_path)
    )
    if not trajectory_path.exists() and current_state_contract:
        findings.append(Finding("WARN", "missing decision trajectory: trajectory.md"))
    elif trajectory_path.exists():
        check_trajectory(trajectory_path, findings)

    check_shared_visualization(challenge_dir, findings)

    has_legacy_state = (challenge_dir / "status.md").exists() or (
        challenge_dir / "ideas.md"
    ).exists()
    if not (challenge_dir / "candidates.md").exists():
        if has_legacy_state:
            findings.append(
                Finding(
                    "WARN",
                    "legacy challenge is missing candidates.md; new workflow uses candidates.md, state.md, and baseline.md",
                )
            )
        else:
            findings.append(Finding("ERROR", "missing required file: candidates.md"))
    else:
        check_candidates(challenge_dir / "candidates.md", findings)
        check_human_steering(challenge_dir / "candidates.md", "candidates.md", findings)
        if has_legacy_state:
            findings.append(
                Finding(
                    "INFO",
                    "legacy status.md or ideas.md exists; new workflow uses candidates.md, state.md, and baseline.md",
                )
            )

    literature_path = challenge_dir / "literature.md"
    if not literature_path.exists():
        if has_legacy_state:
            findings.append(Finding("WARN", "legacy challenge is missing literature.md"))
        else:
            findings.append(Finding("ERROR", "missing required file: literature.md"))
    else:
        check_literature(literature_path, findings)
        check_human_steering(literature_path, "literature.md", findings)

    if literature_path.exists() and (challenge_dir / "candidates.md").exists():
        check_literature_candidate_composition(challenge_dir, findings)

    check_discovery_gate(challenge_dir, findings)

    if challenge_dir.name == "challenge_template":
        template_steering_files = (
            challenge_dir / "baseline_template" / "baseline.md",
            challenge_dir / "baseline_template" / "runs" / "exp_template" / "experiment.md",
        )
        for template_path in template_steering_files:
            check_human_steering(
                template_path,
                str(template_path.relative_to(challenge_dir)),
                findings,
            )
        check_template_contracts(challenge_dir, findings)

    baseline_files = collect_baseline_files(challenge_dir)
    if not baseline_files:
        findings.append(Finding("INFO", "no baseline folders found yet"))

    for baseline_path in baseline_files:
        check_baseline(challenge_dir, baseline_path, findings)

    if dashboard_text and baseline_files:
        check_final_freeze_readiness(challenge_dir, dashboard_text, findings)

    if trajectory_path.exists() and challenge_dir.name != "challenge_template":
        check_trajectory_references(challenge_dir, findings)

    if not findings:
        findings.append(Finding("OK", "no consistency issues found"))
    return findings


def check_candidates(candidates_path: pathlib.Path, findings: list[Finding]) -> None:
    text = read_text(candidates_path)
    lower = text.lower()
    active_contract = uses_active_template_contract(text)
    latest_schema = (
        has_template_schema(text, CANDIDATE_SCHEMA)
        or candidates_path.parent.name == "challenge_template"
    )
    prior_vanilla_first_schema = has_template_schema(text, CANDIDATE_SCHEMA_V9)
    prior_judgment_schema = prior_vanilla_first_schema or has_template_schema(
        text, CANDIDATE_SCHEMA_V8
    )
    scored_comparison_schema = has_template_schema(text, CANDIDATE_SCHEMA_V7)
    comparison_schema = (
        latest_schema
        or prior_judgment_schema
        or scored_comparison_schema
        or has_template_schema(text, CANDIDATE_SCHEMA_V6)
    )
    detail_schema = has_template_schema(text, CANDIDATE_SCHEMA_V5)
    ranked_schema = detail_schema or has_template_schema(text, CANDIDATE_SCHEMA_V4)
    evidence_id_schema = ranked_schema or has_template_schema(text, CANDIDATE_SCHEMA_V3)
    current_schema = evidence_id_schema or has_template_schema(text, CANDIDATE_SCHEMA_V2)
    if (
        active_contract
        and candidates_path.parent.name != "challenge_template"
        and not (latest_schema or prior_judgment_schema or scored_comparison_schema)
    ):
        findings.append(
            Finding(
                "WARN",
                f"candidates.md uses a legacy selection record; reload the raw template and migrate it to {CANDIDATE_SCHEMA} before revising selection",
            )
        )
    if comparison_schema:
        check_candidate_comparison(
            candidates_path,
            text,
            findings,
            require_queue=latest_schema or prior_vanilla_first_schema,
            judgment_schema=latest_schema or prior_judgment_schema,
            vanilla_first_schema=latest_schema or prior_vanilla_first_schema,
            scientific_selection_schema=latest_schema,
        )
        return
    if has_candidate_detail_blocks(text):
        if "## selected workflow" not in lower:
            findings.append(
                Finding(
                    "WARN", "candidates.md has candidate blocks but no Selected Workflow section"
                )
            )
        required_fields: list[str] = []
        if active_contract:
            if detail_schema:
                required_fields.extend(
                    (
                        "direction summary",
                        "scientific rank / rationale",
                        "component composition / evidence",
                        "alternative comparison",
                        "implementation feasibility / changes",
                        "promotion / kill criteria",
                        "risks / unresolved questions",
                    )
                )
            else:
                required_fields.extend(("why it fits the challenge", "why it can reach the target"))
                required_fields.extend(
                    (
                        "direction summary",
                        "key insight / novelty",
                        "component composition / evidence",
                        "changes needed for the challenge",
                        "papers / evidence",
                        "code / checkpoint",
                        "risks",
                        "external critic",
                    )
                )
                if current_schema:
                    required_fields.extend(("alternative comparison", "promotion / kill criteria"))
            if not current_schema and (
                "alternative comparison" not in lower or "promotion / kill criteria" not in lower
            ):
                findings.append(
                    Finding(
                        "INFO",
                        "candidates.md uses a legacy selection schema; migrate it before revising candidate selection",
                    )
                )
            if ranked_schema and "scientific rank / rationale" not in required_fields:
                required_fields.append("scientific rank / rationale")
        else:
            required_fields.append("method summary")
        for required_field in required_fields:
            if required_field not in lower:
                findings.append(
                    Finding(
                        "WARN", f"candidates.md candidate blocks missing field: {required_field}"
                    )
                )
        candidate_blocks = {
            normalize_baseline_title(title): body
            for title, body in markdown_subsections(text, "## ")
            if is_candidate_title(title) and "direction summary" in body.lower()
        }
        candidate_ranks: dict[str, int] = {}
        if active_contract and candidates_path.parent.name != "challenge_template":
            for title, body in candidate_blocks.items():
                missing = [
                    field
                    for field in required_fields
                    if not meaningful_value(read_field(body, field))
                ]
                if missing:
                    findings.append(
                        Finding(
                            "WARN",
                            f"{title} is incomplete; missing: {', '.join(missing)}",
                        )
                    )
                if ranked_schema:
                    rank = candidate_scientific_rank(
                        read_field(body, "scientific rank / rationale")
                    )
                    if rank is None:
                        findings.append(
                            Finding(
                                "WARN",
                                f"{title} scientific rank must start with a positive integer and give a rationale",
                            )
                        )
                    else:
                        candidate_ranks[title] = rank
                if evidence_id_schema:
                    composition = read_field(body, "component composition / evidence")
                    alternatives = read_field(body, "alternative comparison")
                    if not LITERATURE_EVIDENCE_ID_RE.search(clean_markdown_cell(composition)):
                        findings.append(
                            Finding(
                                "WARN",
                                f"{title} component composition does not cite an L001-style literature Evidence ID",
                            )
                        )
                    if not LITERATURE_EVIDENCE_ID_RE.search(clean_markdown_cell(alternatives)):
                        findings.append(
                            Finding(
                                "WARN",
                                f"{title} alternative comparison does not cite an L001-style literature Evidence ID",
                            )
                        )
                    feasibility_field = (
                        "implementation feasibility / changes"
                        if detail_schema
                        else "code / checkpoint"
                    )
                    code_status = clean_markdown_cell(candidate_feasibility_value(body)).lower()
                    if not candidate_feasibility_status_is_valid(
                        code_status,
                        require_hard_blocker=ranked_schema,
                    ):
                        findings.append(
                            Finding(
                                "WARN",
                                f"{title} {feasibility_field} must be verified: <evidence link>, blocked: hard blocker: <reason and evidence link>, or not checked",
                            )
                        )
            if ranked_schema and candidate_ranks:
                duplicate_ranks = sorted(
                    rank
                    for rank in set(candidate_ranks.values())
                    if list(candidate_ranks.values()).count(rank) > 1
                )
                if duplicate_ranks:
                    findings.append(
                        Finding(
                            "WARN",
                            f"candidates.md scientific ranks are not unique: {duplicate_ranks}",
                        )
                    )
                expected_ranks = set(range(1, len(candidate_blocks) + 1))
                if set(candidate_ranks.values()) != expected_ranks:
                    findings.append(
                        Finding("WARN", "candidates.md scientific ranks must be contiguous from 1")
                    )
        if active_contract and "trajectory id" not in lower:
            findings.append(
                Finding("WARN", "candidates.md Selected Workflow missing trajectory ID")
            )
        if active_contract and candidates_path.parent.name != "challenge_template":
            selected = markdown_section(text, "Selected Workflow")
            selected_baseline = read_field(selected, "selected baseline")
            preferred_baseline = read_field(
                selected, "scientifically preferred direction before feasibility"
            )
            trajectory_id = read_field(selected, "trajectory ID")
            alternative_decision = read_field(selected, "alternative coverage decision")
            approval_status = clean_markdown_cell(read_field(selected, "approval status")).lower()
            proposal_only = approval_status.startswith("proposed")
            if (
                current_schema
                and meaningful_value(selected_baseline)
                and not meaningful_value(alternative_decision)
            ):
                findings.append(
                    Finding("WARN", "selected baseline has no alternative coverage decision")
                )
            if (
                evidence_id_schema
                and meaningful_value(selected_baseline)
                and not meaningful_value(preferred_baseline)
            ):
                findings.append(
                    Finding(
                        "WARN",
                        "selected baseline has no scientific preference recorded before feasibility",
                    )
                )
            if (
                meaningful_value(selected_baseline)
                and not proposal_only
                and not meaningful_value(trajectory_id)
            ):
                findings.append(Finding("WARN", "selected baseline has no trajectory decision"))
            elif (
                not proposal_only
                and meaningful_value(trajectory_id)
                and not TRAJECTORY_ID_RE.search(trajectory_id)
            ):
                findings.append(Finding("WARN", "candidates.md trajectory ID is not T0001-style"))
            selected_id = first_baseline_id(selected_baseline)
            preferred_id = first_baseline_id(preferred_baseline)
            if (
                evidence_id_schema
                and meaningful_value(selected_baseline)
                and meaningful_value(preferred_baseline)
            ):
                for label, baseline_id in (
                    ("selected", selected_id),
                    ("scientifically preferred", preferred_id),
                ):
                    if not baseline_id or baseline_id not in candidate_blocks:
                        findings.append(
                            Finding(
                                "WARN", f"{label} direction does not name one candidate baseline"
                            )
                        )
                if not proposal_only and selected_id in candidate_blocks:
                    selected_code = candidate_feasibility_value(candidate_blocks[selected_id])
                    if not candidate_feasibility_is_verified(selected_code):
                        findings.append(
                            Finding(
                                "WARN",
                                f"selected direction {selected_id} lacks verified implementation/checkpoint probe evidence",
                            )
                        )
                if (
                    ranked_schema
                    and preferred_id in candidate_ranks
                    and candidate_ranks[preferred_id] != 1
                ):
                    findings.append(
                        Finding(
                            "WARN",
                            "scientifically preferred direction must be the rank-1 candidate",
                        )
                    )
                if not ranked_schema and (
                    not proposal_only
                    and selected_id
                    and preferred_id
                    and selected_id != preferred_id
                    and preferred_id in candidate_blocks
                ):
                    preferred_code = candidate_feasibility_value(candidate_blocks[preferred_id])
                    if not candidate_feasibility_is_blocked(preferred_code):
                        findings.append(
                            Finding(
                                "WARN",
                                f"selected direction differs from scientific preference {preferred_id} without a verified hard blocker",
                            )
                        )
            if ranked_schema and selected_id in candidate_blocks:
                check_promotion_obligations(
                    text,
                    candidate_blocks,
                    candidate_ranks,
                    selected_id,
                    findings,
                )
        return

    findings.append(
        Finding(
            "WARN",
            "candidates.md should use one concise baselineXXX_<direction> block per candidate",
        )
    )


def check_candidate_comparison(
    candidates_path: pathlib.Path,
    text: str,
    findings: list[Finding],
    require_queue: bool,
    judgment_schema: bool = False,
    vanilla_first_schema: bool = False,
    scientific_selection_schema: bool = False,
) -> None:
    section = markdown_section(text, "Candidate Comparison")
    if not section:
        findings.append(Finding("WARN", "candidates.md is missing Candidate Comparison"))
        return

    required_columns = (
        [
            "candidate / direction",
            "committed end-to-end composition",
            "why it may win",
            "main uncertainty",
            "next decisive evidence",
            "status / evidence",
        ]
        if vanilla_first_schema
        else (
            [
                "candidate / direction",
                "committed end-to-end composition / evidence ids",
                "expected scientific advantage",
                "main uncertainty",
                "next decisive evidence",
                "status / evidence",
            ]
            if judgment_schema
            else [
                "candidate / direction",
                "component composition / evidence ids",
                "expected target performance (1-5)",
                "evidence strength (1-5)",
                "bottleneck coverage (1-5)",
                "novelty / complementarity (1-5)",
                "scientific score / rank",
                "status / evidence",
            ]
        )
    )
    if require_queue and "next decisive evidence" not in required_columns:
        required_columns.insert(-1, "next decisive evidence")
    header = first_markdown_table_header(section)
    for column in required_columns:
        if column not in header:
            findings.append(
                Finding("WARN", f"candidates.md Candidate Comparison missing column: {column}")
            )

    if candidates_path.parent.name == "challenge_template":
        return

    rows = [
        row
        for row in first_markdown_table_rows(section)
        if meaningful_value(row.get("candidate / direction", ""))
    ]
    if vanilla_first_schema and not rows:
        findings.append(
            Finding("WARN", "candidates.md Candidate Comparison has no serious direction")
        )
    elif not vanilla_first_schema and not 3 <= len(rows) <= 6:
        findings.append(
            Finding("WARN", "candidates.md Candidate Comparison needs 3-6 serious directions")
        )

    candidates: dict[str, dict[str, object]] = {}
    allowed_statuses = (
        "open",
        "required",
        "selected",
        "active",
        "supported",
        "rejected",
        "blocked",
    )
    for row_number, row in enumerate(rows, start=1):
        candidate_id = first_baseline_id(row.get("candidate / direction", ""))
        if not candidate_id:
            findings.append(
                Finding(
                    "WARN", f"Candidate Comparison row {row_number} lacks a baselineXXX direction"
                )
            )
            continue
        if candidate_id in candidates:
            findings.append(Finding("WARN", f"Candidate Comparison repeats {candidate_id}"))
            continue

        composition_field = (
            "committed end-to-end composition"
            if vanilla_first_schema
            else (
                "committed end-to-end composition / evidence ids"
                if judgment_schema
                else "component composition / evidence ids"
            )
        )
        composition = row.get(composition_field, "")
        evidence_ids = set(LITERATURE_EVIDENCE_ID_RE.findall(clean_markdown_cell(composition)))
        if not meaningful_value(composition):
            findings.append(
                Finding("WARN", f"{candidate_id} lacks a committed end-to-end composition")
            )
        elif not vanilla_first_schema and not evidence_ids:
            findings.append(
                Finding("WARN", f"{candidate_id} composition needs L001-style literature evidence")
            )
        total: Optional[int] = None
        rank: Optional[int] = None
        if judgment_schema:
            judgment_fields = (
                ("why it may win", "main uncertainty")
                if vanilla_first_schema
                else ("expected scientific advantage", "main uncertainty")
            )
            for field in judgment_fields:
                if not meaningful_value(row.get(field, "")):
                    findings.append(Finding("WARN", f"{candidate_id} lacks {field}"))
            composition_lower = clean_markdown_cell(composition).lower()
            if re.search(
                r"\b(optional|promotion(?:-gated| gate| path)?|promote if|control plus|fallback)\b",
                composition_lower,
            ) or re.search(r"control[_+ -]+plus[_+ -]+promotion", candidate_id, re.IGNORECASE):
                findings.append(
                    Finding(
                        "WARN",
                        f"{candidate_id} is a conditional control/promotion bundle; make every defining component committed or split the control into its own candidate",
                    )
                )
        else:
            score_fields = (
                "expected target performance (1-5)",
                "evidence strength (1-5)",
                "bottleneck coverage (1-5)",
                "novelty / complementarity (1-5)",
            )
            scores: list[int] = []
            for field in score_fields:
                score = candidate_rating(row.get(field, ""))
                if score is None:
                    findings.append(
                        Finding("WARN", f"{candidate_id} {field} must be an integer from 1 to 5")
                    )
                else:
                    scores.append(score)

            total, rank = candidate_score_and_rank(row.get("scientific score / rank", ""))
            if total is None or rank is None:
                findings.append(
                    Finding(
                        "WARN", f"{candidate_id} scientific score / rank must be '<score> / <rank>'"
                    )
                )
            elif len(scores) == 4:
                expected_total = 2 * scores[0] + sum(scores[1:])
                if total != expected_total:
                    findings.append(
                        Finding(
                            "WARN",
                            f"{candidate_id} scientific score {total} does not match the performance-weighted total {expected_total}",
                        )
                    )

        status = clean_markdown_cell(row.get("status / evidence", "")).lower()
        if not status.startswith(allowed_statuses):
            findings.append(
                Finding("WARN", f"{candidate_id} has unknown Candidate Comparison status: {status}")
            )
        elif status.startswith(("supported", "rejected")) and not has_experiment_evidence_link(
            row.get("status / evidence", "")
        ):
            findings.append(
                Finding(
                    "WARN", f"{candidate_id} resolved status lacks comparable experiment evidence"
                )
            )
        elif (
            judgment_schema
            and status.startswith(("supported", "rejected"))
            and not any(
                candidate_id.lower() in href.lower() and "experiment.md" in href.lower()
                for href in markdown_links(row.get("status / evidence", ""))
            )
        ):
            findings.append(
                Finding(
                    "WARN",
                    f"{candidate_id} resolved status must link a comparable experiment under that candidate's baseline",
                )
            )
        elif status.startswith("blocked") and not verified_hard_blocker(
            row.get("status / evidence", "")
        ):
            findings.append(Finding("WARN", f"{candidate_id} lacks verified hard-blocker evidence"))

        if require_queue:
            next_evidence = row.get("next decisive evidence", "")
            if not meaningful_value(next_evidence):
                findings.append(Finding("WARN", f"{candidate_id} lacks Next Decisive Evidence"))
            elif status.startswith(("open", "required", "selected", "active")) and not (
                markdown_links(next_evidence)
                or EXPLORATION_ID_RE.search(next_evidence)
                or LITERATURE_EVIDENCE_ID_RE.search(next_evidence)
                or EXP_ID_RE.search(next_evidence)
            ):
                findings.append(
                    Finding(
                        "WARN",
                        f"{candidate_id} Next Decisive Evidence needs a literature, exploration, or experiment reference",
                    )
                )

        candidates[candidate_id] = {
            "row": row,
            "total": total,
            "rank": rank,
            "status": status,
            "evidence_ids": evidence_ids,
        }

    valid_ranks = [value["rank"] for value in candidates.values() if isinstance(value["rank"], int)]
    if not judgment_schema and valid_ranks:
        if len(valid_ranks) != len(set(valid_ranks)):
            findings.append(Finding("WARN", "Candidate Comparison ranks must be unique"))
        if set(valid_ranks) != set(range(1, len(candidates) + 1)):
            findings.append(Finding("WARN", "Candidate Comparison ranks must be contiguous from 1"))
        ordered = sorted(
            (
                (value["rank"], value["total"])
                for value in candidates.values()
                if isinstance(value["rank"], int) and isinstance(value["total"], int)
            ),
            key=lambda item: item[0],
        )
        if any(left[1] < right[1] for left, right in zip(ordered, ordered[1:])):
            findings.append(
                Finding("WARN", "Candidate Comparison rank contradicts the scientific scores")
            )

    selected = markdown_section(text, "Selected Workflow")
    required_selection_fields = (
        (
            "pre-search thesis direction",
            "pre-search thesis disposition",
            (
                "scientifically strongest direction before feasibility"
                if scientific_selection_schema
                else "strongest unblocked direction"
            ),
            "selected baseline",
            "selection reason",
            "trajectory ID",
            "approval status",
        )
        if vanilla_first_schema
        else (
            (
                "strongest unblocked direction",
                "selected baseline",
                "selection reason / anchor comparison",
                "trajectory ID",
                "approval status",
            )
            if judgment_schema
            else (
                "highest-scoring unblocked direction",
                "selected baseline",
                "hard-blocker exception",
                "trajectory ID",
                "approval status",
            )
        )
    )
    missing = [
        field
        for field in required_selection_fields
        if not meaningful_value(read_field(selected, field))
    ]
    if missing:
        findings.append(
            Finding("WARN", f"candidates.md Selected Workflow is incomplete: {', '.join(missing)}")
        )
        return

    approval_status = clean_markdown_cell(read_field(selected, "approval status")).lower()
    proposal_only = approval_status.startswith("proposed")
    selected_id = first_baseline_id(read_field(selected, "selected baseline"))
    preferred_field = (
        "scientifically strongest direction before feasibility"
        if scientific_selection_schema
        else (
            "strongest unblocked direction"
            if judgment_schema
            else "highest-scoring unblocked direction"
        )
    )
    preferred_id = first_baseline_id(read_field(selected, preferred_field))

    if judgment_schema:
        if preferred_id not in candidates:
            label = (
                "scientifically strongest direction"
                if scientific_selection_schema
                else "strongest unblocked direction"
            )
            findings.append(Finding("WARN", f"{label} does not name a candidate"))
        elif not scientific_selection_schema and str(candidates[preferred_id]["status"]).startswith(
            ("blocked", "rejected")
        ):
            findings.append(Finding("WARN", "strongest unblocked direction is blocked or rejected"))
        if selected_id not in candidates:
            findings.append(Finding("WARN", "selected baseline does not name a candidate"))
        elif not proposal_only and selected_id != preferred_id:
            preferred_status = (
                str(candidates[preferred_id]["status"]) if preferred_id in candidates else ""
            )
            if not (
                scientific_selection_schema
                and preferred_status.startswith("blocked")
                and verified_hard_blocker(
                    candidates[preferred_id]["row"].get("status / evidence", "")
                )
            ):
                findings.append(
                    Finding(
                        "WARN",
                        (
                            "selected baseline differs from the scientifically strongest direction without a verified hard blocker"
                            if scientific_selection_schema
                            else "selected baseline must be the strongest unblocked direction"
                        ),
                    )
                )
        elif not proposal_only:
            selected_status = str(candidates[selected_id]["status"])
            if not selected_status.startswith(("selected", "active", "supported")):
                findings.append(
                    Finding(
                        "WARN",
                        f"selected baseline {selected_id} has unresolved status: {selected_status}",
                    )
                )

        selection_reason_field = (
            "selection reason" if vanilla_first_schema else "selection reason / anchor comparison"
        )
        selection_reason = read_field(selected, selection_reason_field)
        reason_ids = {
            normalize_baseline_title(candidate_id)
            for candidate_id in BASELINE_REFERENCE_RE.findall(clean_markdown_cell(selection_reason))
        }
        if not proposal_only and selected_id and selected_id not in reason_ids:
            findings.append(Finding("WARN", "selection reason must name the selected direction"))
        if not proposal_only and len(candidates) > 1 and len(reason_ids) < 2:
            findings.append(
                Finding("WARN", "selection reason must compare the strongest alternative")
            )
        if scientific_selection_schema and not proposal_only:
            if convenience_selection_language(selection_reason):
                findings.append(
                    Finding(
                        "WARN",
                        "selection reason uses implementation convenience instead of scientific performance or a verified hard blocker",
                    )
                )
            if selected_id in candidates:
                selected_row = candidates[selected_id]["row"]
                if convenience_selection_language(selected_row.get("why it may win", "")):
                    findings.append(
                        Finding(
                            "WARN",
                            f"{selected_id} Why It May Win relies on execution convenience rather than scientific advantage",
                        )
                    )
                if soft_implementation_uncertainty(selected_row.get("main uncertainty", "")):
                    findings.append(
                        Finding(
                            "WARN",
                            f"{selected_id} Main Uncertainty contains setup convenience; keep scientific uncertainty here and record only verified hard blockers in status",
                        )
                    )
        if (
            not vanilla_first_schema
            and not proposal_only
            and not LITERATURE_EVIDENCE_ID_RE.search(selection_reason)
        ):
            findings.append(
                Finding("WARN", "selection reason must cite decisive literature Evidence IDs")
            )
        if vanilla_first_schema:
            thesis_direction = first_baseline_id(
                read_field(selected, "pre-search thesis direction")
            )
            if thesis_direction not in candidates:
                findings.append(
                    Finding("WARN", "pre-search thesis direction does not name a candidate")
                )
            thesis_disposition = clean_markdown_cell(
                read_field(selected, "pre-search thesis disposition")
            ).lower()
            if not thesis_disposition.startswith(("verify:", "refine:", "replace:", "reject:")):
                findings.append(
                    Finding(
                        "WARN",
                        "pre-search thesis disposition must start with verify/refine/replace/reject",
                    )
                )
    else:
        actionable = [
            (candidate_id, value)
            for candidate_id, value in candidates.items()
            if str(value["status"]).startswith(("open", "required", "selected", "active"))
            and isinstance(value["rank"], int)
        ]
        expected_id = (
            min(actionable, key=lambda item: int(item[1]["rank"]))[0] if actionable else ""
        )

        if preferred_id not in candidates:
            findings.append(
                Finding("WARN", "highest-scoring unblocked direction does not name a candidate")
            )
        elif expected_id and preferred_id != expected_id:
            findings.append(
                Finding(
                    "WARN",
                    "highest-scoring unblocked direction contradicts Candidate Comparison rank/status",
                )
            )
        if selected_id not in candidates:
            findings.append(Finding("WARN", "selected baseline does not name a candidate"))
        elif not proposal_only and expected_id and preferred_id and selected_id != preferred_id:
            findings.append(
                Finding("WARN", "selected baseline must be the highest-scoring unblocked direction")
            )
        elif not proposal_only and expected_id:
            selected_status = str(candidates[selected_id]["status"])
            if not selected_status.startswith(("selected", "active", "supported")):
                findings.append(
                    Finding(
                        "WARN",
                        f"selected baseline {selected_id} has unresolved status: {selected_status}",
                    )
                )

        if (
            expected_id
            and selected_id in candidates
            and isinstance(candidates[selected_id]["rank"], int)
        ):
            selected_rank = int(candidates[selected_id]["rank"])
            blocker_exception = read_field(selected, "hard-blocker exception")
            if selected_rank == 1 and not clean_markdown_cell(blocker_exception).lower().startswith(
                "not applicable:"
            ):
                findings.append(
                    Finding(
                        "WARN",
                        "rank-1 selection should mark the hard-blocker exception not applicable",
                    )
                )
            if selected_rank > 1 and not markdown_links(blocker_exception):
                findings.append(
                    Finding(
                        "WARN", "selection below rank 1 must link verified hard-blocker evidence"
                    )
                )
            for candidate_id, value in candidates.items():
                rank = value["rank"]
                status = str(value["status"])
                if (
                    isinstance(rank, int)
                    and rank < selected_rank
                    and not status.startswith("blocked")
                ):
                    findings.append(
                        Finding(
                            "WARN",
                            f"higher-scoring candidate {candidate_id} cannot be skipped without a verified hard blocker",
                        )
                    )

    trajectory_id = read_field(selected, "trajectory ID")
    if not proposal_only and not TRAJECTORY_ID_RE.search(trajectory_id):
        findings.append(Finding("WARN", "candidates.md trajectory ID is not T0001-style"))


def has_candidate_detail_blocks(text: str) -> bool:
    for title, body in markdown_subsections(text, "## "):
        if is_candidate_title(title) and (
            "direction summary" in body.lower() or "method summary" in body.lower()
        ):
            return True
    return False


def is_candidate_title(title: str) -> bool:
    return bool(re.match(r"^baseline\d{3}", normalize_baseline_title(title), re.IGNORECASE))


def check_vanilla_first_literature(
    literature_path: pathlib.Path,
    text: str,
    findings: list[Finding],
) -> None:
    required_sections = (
        "## Independent Scientific Thesis",
        "## Open Research Findings",
        "## Revised Scientific Direction",
        "## Post-Search Coverage Audit",
        "## External Critic Summary",
    )
    for section in required_sections:
        if section not in text:
            findings.append(Finding("WARN", f"literature.md missing section: {section[3:]}"))

    if literature_path.parent.name == "challenge_template":
        return

    thesis = markdown_section(text, "Independent Scientific Thesis")
    for field in (
        "agent-chosen decomposition",
        "strongest complete pipeline before search",
        "causal rationale",
        "serious alternatives",
        "memory / uncertainty boundary",
    ):
        if not meaningful_literature_evidence(read_field(thesis, field)):
            findings.append(
                Finding("WARN", f"literature.md Independent Scientific Thesis is missing: {field}")
            )

    research = markdown_section(text, "Open Research Findings")
    rows = [
        row
        for row in first_markdown_table_rows(research)
        if meaningful_value(row.get("evidence id", ""))
    ]
    if not rows:
        findings.append(
            Finding("WARN", "literature.md has no decision-changing Open Research Findings")
        )
    evidence_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        evidence_id = clean_markdown_cell(row.get("evidence id", ""))
        if not LITERATURE_EVIDENCE_ID_RE.fullmatch(evidence_id):
            findings.append(
                Finding(
                    "WARN",
                    f"literature.md research row {row_number} lacks one L001-style Evidence ID",
                )
            )
        elif evidence_id in evidence_ids:
            findings.append(Finding("WARN", f"literature.md repeats Evidence ID {evidence_id}"))
        else:
            evidence_ids.add(evidence_id)
        for field in (
            "finding / method",
            "decision-relevant evidence",
            "primary source",
            "implementation / weights / data",
            "effect on thesis",
        ):
            if not meaningful_literature_evidence(row.get(field, "")):
                findings.append(
                    Finding(
                        "WARN",
                        f"literature.md research row {row_number} is missing: {field}",
                    )
                )
        primary_source = row.get("primary source", "")
        if meaningful_literature_evidence(primary_source) and not markdown_links(primary_source):
            findings.append(
                Finding(
                    "WARN",
                    f"literature.md {evidence_id or row_number} primary source needs an inspectable link",
                )
            )
        effect = clean_markdown_cell(row.get("effect on thesis", "")).lower().lstrip("*_")
        if meaningful_literature_evidence(effect) and not effect.startswith(
            ("verify:", "refine:", "replace:", "reject:", "uncertain:")
        ):
            findings.append(
                Finding(
                    "WARN",
                    f"literature.md {evidence_id or row_number} effect must start with verify/refine/replace/reject/uncertain",
                )
            )

    revised = markdown_section(text, "Revised Scientific Direction")
    for field in (
        "chosen complete pipeline after search",
        "what changed from the thesis",
        "why this is now strongest",
        "strongest serious alternative",
        "remaining scientific uncertainty",
    ):
        if not meaningful_literature_evidence(read_field(revised, field)):
            findings.append(
                Finding("WARN", f"literature.md Revised Scientific Direction is missing: {field}")
            )
    if convenience_selection_language(read_field(revised, "why this is now strongest")):
        findings.append(
            Finding(
                "WARN",
                "literature.md justifies the revised direction with implementation convenience rather than scientific performance",
            )
        )

    audit = markdown_section(text, "Post-Search Coverage Audit")
    audit_rows = {
        clean_markdown_cell(row.get("coverage area", "")).lower(): row
        for row in first_markdown_table_rows(audit)
        if meaningful_value(row.get("coverage area", ""))
    }
    required_area_aliases = (
        ("strongest exact/close-task methods, organizer baselines, and prior winners",),
        (
            "recent task-specific and general cv/ml frontier through the current date, including predecessor, related-work, forward-citation, and official-release successors",
            "recent task-specific and general cv/ml frontier, including related-work or citation successors",
        ),
        ("current general and domain/task-family foundation or pretrained models",),
        ("usable official implementations, checkpoints, and applicable miagent skills",),
        ("external/pretraining data, training-distribution overlap, license, and leakage risk",),
    )
    for aliases in required_area_aliases:
        area = aliases[0]
        row = next((audit_rows[alias] for alias in aliases if alias in audit_rows), None)
        if row is None:
            findings.append(Finding("WARN", f"literature.md post-search audit is missing: {area}"))
            continue
        for field in ("evidence ids / concrete findings", "decision / remaining gap"):
            if not meaningful_literature_evidence(row.get(field, "")):
                findings.append(
                    Finding("WARN", f"literature.md post-search audit {area} lacks {field}")
                )

    critic = markdown_section(text, "External Critic Summary")
    for field in (
        "search pass / trigger",
        "critic report",
        "main objections",
        "accepted changes",
        "rejected or deferred objections",
    ):
        if not meaningful_literature_evidence(read_field(critic, field)):
            findings.append(
                Finding("WARN", f"literature.md External Critic Summary is missing: {field}")
            )
    critic_report = clean_markdown_cell(read_field(critic, "critic report"))
    if meaningful_value(critic_report):
        if "anticipated" in critic_report.lower() or "not run yet" in critic_report.lower():
            findings.append(
                Finding(
                    "WARN", "literature.md records an anticipated rather than completed critic pass"
                )
            )
        elif critic_report.lower().startswith("blocked:"):
            findings.append(
                Finding(
                    "WARN",
                    "literature.md critic provider failure is not a completed critic pass; use a fresh critic subagent or dedicated self-critique report",
                )
            )
        elif not markdown_links(critic_report):
            findings.append(
                Finding("WARN", "literature.md critic report must link the completed critic report")
            )
        for href in markdown_links(critic_report):
            check_local_link(literature_path, href, findings, "literature.md critic report")


def check_literature(literature_path: pathlib.Path, findings: list[Finding]) -> None:
    text = read_text(literature_path)
    if not uses_active_template_contract(text):
        for section in (
            "## Challenge Decomposition",
            "## Search Log",
            "## Method Evidence Table",
            "## External Critic Summary",
        ):
            if section not in text:
                findings.append(Finding("WARN", f"literature.md missing section: {section[3:]}"))
        return
    latest_schema = (
        has_template_schema(text, LITERATURE_SCHEMA)
        or literature_path.parent.name == "challenge_template"
    )
    if latest_schema:
        check_vanilla_first_literature(literature_path, text, findings)
        return
    prior_latest_schema = has_template_schema(text, LITERATURE_SCHEMA_V6)
    prior_current_schema = prior_latest_schema or has_template_schema(text, LITERATURE_SCHEMA_V5)
    evidence_id_schema = (
        latest_schema or prior_current_schema or has_template_schema(text, LITERATURE_SCHEMA_V4)
    )
    current_schema = evidence_id_schema or has_template_schema(text, LITERATURE_SCHEMA_V3)
    v2_schema = has_template_schema(text, LITERATURE_SCHEMA_V2)

    required_sections = (
        "## Challenge Decomposition",
        "## Overall Thinking",
        "## Component Search Evidence",
        "## Unified Direction",
        "## External Critic Summary",
    )
    miagent_anchor_row: Optional[dict[str, str]] = None
    if current_schema:
        required_sections = (
            "## Challenge Decomposition",
            "## Overall Thinking",
            "## Challenge Anchor Evidence",
            "## Component Search Evidence",
            "## Unified Direction",
            "## External Critic Summary",
        )
    elif v2_schema:
        required_sections = (
            "## Challenge Decomposition",
            "## Overall Thinking",
            "## Search Coverage Evidence",
            "## Component Search Evidence",
            "## Unified Direction",
            "## External Critic Summary",
        )
    for section in required_sections:
        if section not in text:
            findings.append(Finding("WARN", f"literature.md missing section: {section[3:]}"))

    v2_required_coverage = (
        "official baselines, prior winners, and controls",
        "same-domain and task-specific methods",
        "abstract-task frontier outside the medical domain",
        "foundation and pretrained models",
        "runnable implementations, checkpoints, and weights",
        "forward citations and related-work mining",
        "external public data and pretraining data",
        "task-specific miagent evidence",
    )
    required_anchors = (
        "organizer baselines and model/data assumptions",
        "strongest reproducible exact/close-task pipelines",
        (
            "applicable miagent methods, pretrained assets, and prior-run evidence"
            if latest_schema or prior_current_schema
            else "task-specific miagent skill and prior-run evidence"
        ),
    )
    required_search_lenses = (
        "exact-task / domain anchor",
        "cross-domain medical analogue",
        "general cv abstract-task frontier",
        "general vision foundation / pretraining",
        "medical or task-family foundation / pretraining",
        "bottleneck-specific methods",
        "citation / related-work expansion",
        "external data / pretraining data",
    )
    legacy_required_categories = (
        "official baselines, prior winners, and controls",
        "same-domain solutions",
        "current sota for the abstract task",
        "recent foundation and general methods",
        "task-specific miagent evidence",
        "forward citations and related work",
        "external public data",
    )
    lower = text.lower()
    if current_schema:
        for anchor in required_anchors:
            if anchor not in lower:
                findings.append(
                    Finding("WARN", f"literature.md missing challenge anchor area: {anchor}")
                )
    elif v2_schema:
        for area in v2_required_coverage:
            if area not in lower:
                findings.append(
                    Finding("WARN", f"literature.md missing search coverage area: {area}")
                )
    elif "## Search Coverage Evidence" not in text:
        findings.append(
            Finding(
                "INFO",
                "literature.md uses a legacy search-evidence schema; migrate it before focused re-search",
            )
        )

    if literature_path.parent.name == "challenge_template":
        return

    decomposition = markdown_section(text, "Challenge Decomposition")
    component_field = (
        "prediction-relevant component" if current_schema else "decomposed problem / component"
    )
    decomposition_rows = [
        row
        for row in first_markdown_table_rows(decomposition)
        if meaningful_value(row.get(component_field, ""))
    ]
    components = [clean_markdown_cell(row.get(component_field, "")) for row in decomposition_rows]
    if not components:
        findings.append(Finding("WARN", "literature.md has no decomposed challenge problems"))
        return

    for row, component in zip(decomposition_rows, components):
        if latest_schema and re.search(
            r"\b(evaluator|evaluation parity|metric implementation|metric parity|packag(?:e|ing)|submission adapter|output format|file format|environment|scheduler|scheduling|pipeline (?:readiness|plumbing))\b",
            component,
            re.IGNORECASE,
        ):
            findings.append(
                Finding(
                    "WARN",
                    f"literature.md decomposition contains operational component '{component}'; route it to challenge.md/dashboard.md and keep only its prediction-relevant scientific implication",
                )
            )
        decomposition_fields = (
            (
                "challenge-specific form",
                "abstract technical formulation(s)",
                "metric impact / failure mode",
                "inputs / interfaces / scientific constraints",
            )
            if current_schema
            else (
                "importance / metric impact",
                "what must be solved",
                "dependencies / constraints",
            )
        )
        missing = [
            field for field in decomposition_fields if not meaningful_value(row.get(field, ""))
        ]
        if missing:
            findings.append(
                Finding(
                    "WARN",
                    f"literature.md decomposition for {component} is incomplete; missing: {', '.join(missing)}",
                )
            )

    overall = markdown_section(text, "Overall Thinking")
    current_view = markdown_subsection(overall, "### Current View")
    current_view_text = current_view.removeprefix("### Current View").strip()
    if not meaningful_value(current_view_text):
        findings.append(Finding("WARN", "literature.md has no challenge-specific Overall Thinking"))

    if current_schema:
        anchor_section = markdown_section(text, "Challenge Anchor Evidence")
        anchor_rows = {
            clean_markdown_cell(row.get("anchor area", "")).lower(): row
            for row in first_markdown_table_rows(anchor_section)
            if meaningful_value(row.get("anchor area", ""))
        }
        for anchor in required_anchors:
            row = anchor_rows.get(anchor)
            if row is None:
                findings.append(
                    Finding("WARN", f"literature.md missing challenge anchor evidence: {anchor}")
                )
                continue
            missing = [
                field
                for field in (
                    "concrete findings",
                    "evidence inspected",
                    "decision / remaining gap",
                )
                if not meaningful_literature_evidence(row.get(field, ""))
            ]
            if missing:
                findings.append(
                    Finding(
                        "WARN",
                        f"literature.md has incomplete {anchor} anchor; missing: {', '.join(missing)}",
                    )
                )
            if "miagent" in anchor:
                miagent_anchor_row = row
            if "miagent" in anchor and not row_mentions_local_skill(row):
                findings.append(
                    Finding(
                        "WARN",
                        "literature.md MIAgent-method anchor does not cite a consulted skills/ path",
                    )
                )
    elif v2_schema:
        coverage_section = markdown_section(text, "Search Coverage Evidence")
        coverage_rows = {
            clean_markdown_cell(row.get("coverage area", "")).lower(): row
            for row in first_markdown_table_rows(coverage_section)
            if meaningful_value(row.get("coverage area", ""))
        }
        for area in v2_required_coverage:
            row = coverage_rows.get(area)
            if row is None:
                findings.append(
                    Finding("WARN", f"literature.md missing search coverage evidence: {area}")
                )
                continue
            missing = [
                field
                for field in (
                    "concrete findings",
                    "evidence inspected",
                    "decision / remaining gap",
                )
                if not meaningful_literature_evidence(row.get(field, ""))
            ]
            if missing:
                findings.append(
                    Finding(
                        "WARN",
                        f"literature.md has incomplete {area} coverage; missing: {', '.join(missing)}",
                    )
                )

    evidence_section = markdown_section(text, "Component Search Evidence")
    component_blocks = {
        clean_markdown_cell(title).lower(): body
        for title, body in markdown_subsections(evidence_section, "### ")
        if meaningful_value(title)
    }
    current_method_rows: list[tuple[str, dict[str, str]]] = []
    evidence_ids: set[str] = set()
    for component in components:
        body = component_blocks.get(component.lower())
        if body is None:
            findings.append(
                Finding(
                    "WARN", f"literature.md has no search evidence block for component: {component}"
                )
            )
            continue
        initial_analysis_header = (
            "#### Agent's Initial Analysis And Abstraction"
            if current_schema
            else "#### Agent's Initial Analysis"
        )
        initial_analysis = markdown_subsection(body, initial_analysis_header)
        initial_analysis_text = initial_analysis.removeprefix(initial_analysis_header).strip()
        if not meaningful_value(initial_analysis_text):
            findings.append(
                Finding(
                    "WARN",
                    f"literature.md {component} has no agent initial analysis and abstraction before search",
                )
            )
        if current_schema:
            component_rows = [
                row
                for row in first_markdown_table_rows(body)
                if meaningful_value(row.get("search lens", ""))
            ]
            current_method_rows.extend((component, row) for row in component_rows)
            rows_by_lens: dict[str, list[dict[str, str]]] = {}
            for row in component_rows:
                lens_name = clean_markdown_cell(row.get("search lens", "")).lower()
                rows_by_lens.setdefault(lens_name, []).append(row)
            if not component_has_local_skill_grounding(initial_analysis_text, component_rows):
                findings.append(
                    Finding(
                        "WARN",
                        f"literature.md {component} does not cite task-specific skills or vision-foundation skill grounding",
                    )
                )
            abstract_targets: dict[str, str] = {}
            for lens, lens_rows in rows_by_lens.items():
                for row_number, row in enumerate(lens_rows, start=1):
                    evidence_fields = [
                        "component-specific abstract search target",
                        "method / family",
                        "year / venue",
                        "concrete primary evidence",
                        "paper / primary source",
                        "official implementation + weights / pretraining / license",
                        "transfer fit / component decision",
                    ]
                    if evidence_id_schema:
                        evidence_fields.insert(0, "evidence id")
                    missing = [
                        field
                        for field in evidence_fields
                        if not meaningful_literature_evidence(row.get(field, ""))
                    ]
                    if missing:
                        findings.append(
                            Finding(
                                "WARN",
                                f"literature.md {component} has incomplete {lens} evidence row {row_number}; missing: {', '.join(missing)}",
                            )
                        )
                    if evidence_id_schema:
                        evidence_id_text = clean_markdown_cell(row.get("evidence id", ""))
                        evidence_id_match = LITERATURE_EVIDENCE_ID_RE.fullmatch(evidence_id_text)
                        if not evidence_id_match:
                            findings.append(
                                Finding(
                                    "WARN",
                                    f"literature.md {component} {lens} evidence row {row_number} lacks one L001-style Evidence ID",
                                )
                            )
                        elif evidence_id_text in evidence_ids:
                            findings.append(
                                Finding(
                                    "WARN", f"literature.md repeats Evidence ID {evidence_id_text}"
                                )
                            )
                        else:
                            evidence_ids.add(evidence_id_text)
                        disposition = clean_markdown_cell(
                            row.get("transfer fit / component decision", "")
                        ).lower()
                        if meaningful_literature_evidence(
                            disposition
                        ) and not disposition.startswith(
                            ("candidate:", "control:", "support:", "reject:")
                        ):
                            findings.append(
                                Finding(
                                    "WARN",
                                    f"literature.md {evidence_id_text or component} needs candidate/control/support/reject disposition",
                                )
                            )
                        if (
                            "foundation / pretraining" in lens
                            and disposition.startswith(("candidate:", "control:"))
                            and not actionable_pretraining_row(row)
                        ):
                            findings.append(
                                Finding(
                                    "WARN",
                                    f"literature.md {evidence_id_text or component} pretrained contender is not action-ready",
                                )
                            )
                    if (
                        lens
                        in {
                            "general cv abstract-task frontier",
                            "general vision foundation / pretraining",
                            "medical or task-family foundation / pretraining",
                        }
                        and named_literature_method(row.get("method / family", ""))
                        and not row_mentions_vision_foundation_skill(row)
                    ):
                        findings.append(
                            Finding(
                                "WARN",
                                f"literature.md {component} {lens} row {row_number} does not cite a consulted skills/vision-foundation index",
                            )
                        )
                    if (
                        not missing
                        and lens
                        in {
                            "general cv abstract-task frontier",
                            "general vision foundation / pretraining",
                            "medical or task-family foundation / pretraining",
                            "citation / related-work expansion",
                        }
                        and weak_frontier_search_row(row)
                    ):
                        findings.append(
                            Finding(
                                "WARN",
                                f"literature.md {component} {lens} records an execution excuse instead of opened current-source evidence",
                            )
                        )
                abstract_target = clean_markdown_cell(
                    lens_rows[0].get("component-specific abstract search target", "")
                )
                normalized_target = re.sub(r"\W+", " ", abstract_target.lower()).strip()
                if (
                    meaningful_literature_evidence(abstract_target)
                    and not starts_with_literature_status(abstract_target)
                    and normalized_target
                ):
                    prior_lens = abstract_targets.get(normalized_target)
                    if prior_lens is not None:
                        findings.append(
                            Finding(
                                "WARN",
                                f"literature.md {component} repeats one abstract search target for {prior_lens} and {lens}",
                            )
                        )
                    else:
                        abstract_targets[normalized_target] = lens

            named_component_methods = {
                clean_markdown_cell(row.get("method / family", "")).lower()
                for row in component_rows
                if named_literature_method(row.get("method / family", ""))
            }
            if len(named_component_methods) < 2:
                findings.append(
                    Finding(
                        "WARN",
                        f"literature.md {component} needs at least two concrete, materially different methods",
                    )
                )
        elif v2_schema:
            rows = [
                row
                for row in first_markdown_table_rows(body)
                if meaningful_value(row.get("concrete method / finding", ""))
            ]
            if not rows:
                findings.append(
                    Finding("WARN", f"literature.md {component} has no concrete search findings")
                )
                continue
            for row_number, row in enumerate(rows, start=1):
                missing = [
                    field
                    for field in (
                        "concrete method / finding",
                        "evidence inspected",
                        "implementation / weights",
                        "effect on component decision",
                    )
                    if not meaningful_literature_evidence(row.get(field, ""))
                ]
                if missing:
                    findings.append(
                        Finding(
                            "WARN",
                            f"literature.md {component} finding {row_number} is incomplete; missing: {', '.join(missing)}",
                        )
                    )
        else:
            rows = {
                clean_markdown_cell(row.get("category", "")).lower(): row
                for row in first_markdown_table_rows(body)
                if meaningful_value(row.get("category", ""))
            }
            for category in legacy_required_categories:
                row = rows.get(category)
                if row is None:
                    findings.append(
                        Finding(
                            "WARN", f"literature.md {component} missing search category: {category}"
                        )
                    )
                    continue
                missing = [
                    field
                    for field in (
                        "what was found",
                        "paper / source evidence",
                        "implementation / weights evidence",
                        "agent conclusion",
                    )
                    if not meaningful_literature_evidence(row.get(field, ""))
                ]
                if missing:
                    findings.append(
                        Finding(
                            "WARN",
                            f"literature.md {component} has incomplete {category}; missing: {', '.join(missing)}",
                        )
                    )

    if current_schema:
        covered_lenses = {
            clean_markdown_cell(row.get("search lens", "")).lower()
            for _, row in current_method_rows
            if meaningful_value(row.get("search lens", ""))
        }
        for lens in required_search_lenses:
            if lens not in covered_lenses:
                findings.append(
                    Finding(
                        "WARN", f"literature.md has no component evidence for search lens: {lens}"
                    )
                )
        check_current_method_evidence(current_method_rows, components, findings)
        if prior_latest_schema:
            check_local_method_inventory(
                miagent_anchor_row,
                current_method_rows,
                evidence_ids,
                findings,
            )

    unified = markdown_section(text, "Unified Direction")
    unified_component_field = (
        "prediction-relevant component" if current_schema else "decomposed problem / component"
    )
    unified_rows = {
        clean_markdown_cell(row.get(unified_component_field, "")).lower(): row
        for row in first_markdown_table_rows(unified)
        if meaningful_value(row.get(unified_component_field, ""))
    }
    for component in components:
        row = unified_rows.get(component.lower())
        if row is None:
            findings.append(
                Finding("WARN", f"literature.md Unified Direction does not cover: {component}")
            )
            continue
        unified_evidence_field = "evidence ids" if evidence_id_schema else "evidence"
        missing = [
            field
            for field in (
                "selected evidence-backed response(s)",
                unified_evidence_field,
                "role / interaction in the unified direction",
                "open risk / alternative",
            )
            if not meaningful_literature_evidence(row.get(field, ""))
        ]
        if missing:
            findings.append(
                Finding(
                    "WARN",
                    f"literature.md Unified Direction has incomplete coverage for {component}; missing: {', '.join(missing)}",
                )
            )
        if evidence_id_schema:
            referenced_ids = set(
                LITERATURE_EVIDENCE_ID_RE.findall(clean_markdown_cell(row.get("evidence ids", "")))
            )
            if not referenced_ids:
                findings.append(
                    Finding(
                        "WARN", f"literature.md Unified Direction {component} cites no Evidence IDs"
                    )
                )
            for evidence_id in sorted(referenced_ids - evidence_ids):
                findings.append(
                    Finding(
                        "WARN",
                        f"literature.md Unified Direction {component} references unknown Evidence ID {evidence_id}",
                    )
                )

    current_direction = markdown_subsection(unified, "### Current Unified Direction")
    current_direction_text = current_direction.removeprefix("### Current Unified Direction").strip()
    if not meaningful_value(current_direction_text):
        findings.append(Finding("WARN", "literature.md has no current Unified Direction"))

    critic = markdown_section(text, "External Critic Summary")
    critic_fields = (
        "search pass / trigger",
        "critic report",
        "main objections",
        "accepted changes",
        "rejected or deferred objections",
    )
    for field in critic_fields:
        if not meaningful_literature_evidence(read_field(critic, field)):
            findings.append(
                Finding("WARN", f"literature.md External Critic Summary is missing: {field}")
            )
    critic_report = clean_markdown_cell(read_field(critic, "critic report"))
    critic_lower = critic_report.lower()
    if meaningful_value(critic_report):
        if "anticipated" in critic_lower or "not run yet" in critic_lower:
            findings.append(
                Finding(
                    "WARN", "literature.md records an anticipated rather than completed critic pass"
                )
            )
        elif not markdown_links(critic_report) and not critic_lower.startswith("blocked:"):
            findings.append(
                Finding(
                    "WARN",
                    "literature.md critic report must link the actual report or record blocked: <tool/provider failure>",
                )
            )
        for href in markdown_links(critic_report):
            check_local_link(literature_path, href, findings, "literature.md critic report")


def check_literature_candidate_composition(
    challenge_dir: pathlib.Path,
    findings: list[Finding],
) -> None:
    literature_text = read_text(challenge_dir / "literature.md")
    candidates_text = read_text(challenge_dir / "candidates.md")
    if has_template_schema(literature_text, LITERATURE_SCHEMA) and (
        has_template_schema(candidates_text, CANDIDATE_SCHEMA)
        or has_template_schema(candidates_text, CANDIDATE_SCHEMA_V9)
    ):
        known_ids = {
            clean_markdown_cell(row.get("evidence id", ""))
            for row in first_markdown_table_rows(
                markdown_section(literature_text, "Open Research Findings")
            )
            if LITERATURE_EVIDENCE_ID_RE.fullmatch(clean_markdown_cell(row.get("evidence id", "")))
        }
        candidate_evidence_text = "\n".join(
            (
                markdown_section(candidates_text, "Candidate Comparison"),
                markdown_section(candidates_text, "Selected Workflow"),
            )
        )
        referenced_ids = set(LITERATURE_EVIDENCE_ID_RE.findall(candidate_evidence_text))
        for evidence_id in sorted(referenced_ids - known_ids):
            findings.append(
                Finding(
                    "WARN", f"candidates.md references unknown literature Evidence ID {evidence_id}"
                )
            )
        return

    literature_has_ids = (
        has_template_schema(literature_text, LITERATURE_SCHEMA_V6)
        or has_template_schema(literature_text, LITERATURE_SCHEMA_V5)
        or has_template_schema(literature_text, LITERATURE_SCHEMA_V4)
    )
    candidates_have_ids = (
        has_template_schema(candidates_text, CANDIDATE_SCHEMA_V8)
        or has_template_schema(candidates_text, CANDIDATE_SCHEMA_V7)
        or has_template_schema(candidates_text, CANDIDATE_SCHEMA_V6)
        or has_template_schema(candidates_text, CANDIDATE_SCHEMA_V5)
        or has_template_schema(candidates_text, CANDIDATE_SCHEMA_V4)
        or has_template_schema(candidates_text, CANDIDATE_SCHEMA_V3)
    )
    if not (literature_has_ids and candidates_have_ids):
        return

    evidence_section = markdown_section(literature_text, "Component Search Evidence")
    evidence_rows: list[dict[str, str]] = []
    for _, body in markdown_subsections(evidence_section, "### "):
        evidence_rows.extend(
            row
            for row in first_markdown_table_rows(body)
            if meaningful_value(row.get("evidence id", ""))
        )

    known_ids = {
        clean_markdown_cell(row.get("evidence id", ""))
        for row in evidence_rows
        if LITERATURE_EVIDENCE_ID_RE.fullmatch(clean_markdown_cell(row.get("evidence id", "")))
    }
    required_composition_ids = {
        clean_markdown_cell(row.get("evidence id", ""))
        for row in evidence_rows
        if clean_markdown_cell(row.get("transfer fit / component decision", ""))
        .lower()
        .startswith("candidate:")
    }
    required_control_ids = {
        clean_markdown_cell(row.get("evidence id", ""))
        for row in evidence_rows
        if clean_markdown_cell(row.get("transfer fit / component decision", ""))
        .lower()
        .startswith("control:")
    }

    composition_ids: set[str] = set()
    comparison_ids: set[str] = set()
    candidate_comparison_schema = (
        has_template_schema(candidates_text, CANDIDATE_SCHEMA_V8)
        or has_template_schema(candidates_text, CANDIDATE_SCHEMA_V7)
        or has_template_schema(candidates_text, CANDIDATE_SCHEMA_V6)
    )
    if candidate_comparison_schema:
        candidate_rows = [
            row
            for row in first_markdown_table_rows(
                markdown_section(candidates_text, "Candidate Comparison")
            )
            if meaningful_value(row.get("candidate / direction", ""))
        ]
        for row in candidate_rows:
            composition_field = (
                "committed end-to-end composition / evidence ids"
                if has_template_schema(candidates_text, CANDIDATE_SCHEMA_V8)
                else "component composition / evidence ids"
            )
            row_ids = LITERATURE_EVIDENCE_ID_RE.findall(
                clean_markdown_cell(row.get(composition_field, ""))
            )
            composition_ids.update(row_ids)
            comparison_ids.update(row_ids)
    else:
        candidate_blocks = [
            body
            for title, body in markdown_subsections(candidates_text, "## ")
            if is_candidate_title(title) and "direction summary" in body.lower()
        ]
        for body in candidate_blocks:
            composition_ids.update(
                LITERATURE_EVIDENCE_ID_RE.findall(
                    clean_markdown_cell(read_field(body, "component composition / evidence"))
                )
            )
            comparison_ids.update(
                LITERATURE_EVIDENCE_ID_RE.findall(
                    clean_markdown_cell(read_field(body, "alternative comparison"))
                )
            )

    for evidence_id in sorted(required_composition_ids - composition_ids):
        findings.append(
            Finding(
                "WARN",
                f"literature.md {evidence_id} is marked candidate but is not used in a candidates.md component composition",
            )
        )
    for evidence_id in sorted(required_control_ids - comparison_ids):
        location = (
            "Candidate Comparison composition"
            if candidate_comparison_schema
            else "alternative comparison"
        )
        findings.append(
            Finding(
                "WARN",
                f"literature.md {evidence_id} is marked control but is not used in a candidates.md {location}",
            )
        )
    for evidence_id in sorted((composition_ids | comparison_ids) - known_ids):
        findings.append(
            Finding(
                "WARN", f"candidates.md references unknown literature Evidence ID {evidence_id}"
            )
        )

    challenge_text = read_text(challenge_dir / "challenge.md")
    pretrained_policy = clean_markdown_cell(
        read_field(challenge_text, "pretrained model policy")
    ).lower()
    pretraining_allowed = any(
        term in pretrained_policy for term in ("allow", "permit")
    ) and not any(term in pretrained_policy for term in ("prohibit", "forbid", "not allowed"))
    if pretraining_allowed:
        carried_pretrained = [
            row
            for row in evidence_rows
            if "foundation / pretraining" in clean_markdown_cell(row.get("search lens", "")).lower()
            and clean_markdown_cell(row.get("transfer fit / component decision", ""))
            .lower()
            .startswith(("candidate:", "control:"))
            and actionable_pretraining_row(row)
        ]
        if not carried_pretrained:
            findings.append(
                Finding(
                    "WARN",
                    "pretraining is allowed but literature.md carries no action-ready pretrained contender into candidate comparison",
                )
            )


def check_state(state_path: pathlib.Path, findings: list[Finding]) -> None:
    text = read_text(state_path)
    line_count = len(text.splitlines())
    if line_count > STATE_SOFT_MAX_LINES:
        findings.append(
            Finding(
                "WARN",
                f"state.md is {line_count} lines; consolidate the attention reset before {STATE_SOFT_MAX_LINES}",
            )
        )
    current_state = has_template_schema(text, STATE_SCHEMA)
    prior_living_state = has_template_schema(text, STATE_SCHEMA_V7)
    if current_state or prior_living_state:
        check_current_state(state_path, text, findings)
        experiment_refs = set(re.findall(r"\bexp\d{3}\b", text, re.IGNORECASE))
        experiment_limit = 2 if current_state else 8
        if len(experiment_refs) > experiment_limit:
            findings.append(
                Finding(
                    "WARN",
                    f"state.md links {len(experiment_refs)} experiments; keep result history and decisive evidence in their owning artifacts and trajectory.md",
                )
            )
        check_state_steering_refs(state_path, text, findings)
        return
    current_contract = uses_active_template_contract(text) or "## Current Focus Contract" in text
    if "## Current Focus Contract" in text and not uses_active_template_contract(text):
        findings.append(
            Finding(
                "WARN",
                "state.md uses the current focus schema but lacks the active template-contract marker",
            )
        )
    latest_stop_check = has_template_schema(text, STATE_SCHEMA_V6)
    prior_current_stop_check = has_template_schema(text, STATE_SCHEMA_V5)
    legacy_stop_check = has_template_schema(text, STATE_SCHEMA_V4) or has_template_schema(
        text, STATE_SCHEMA_V3
    )
    legacy_stop_gate = has_template_schema(text, STATE_SCHEMA_V2) or "## Stop Gate" in text
    if latest_stop_check or prior_current_stop_check:
        stop_section = "## Stop Check"
    elif legacy_stop_check:
        stop_section = "## Stop Check"
    elif legacy_stop_gate:
        stop_section = "## Stop Gate"
    elif "## Auto-Continue Decision" in text:
        stop_section = "## Auto-Continue Decision"
    else:
        stop_section = "## Stop Check"
    if (
        current_contract
        and state_path.parent.name != "challenge_template"
        and not (latest_stop_check or prior_current_stop_check)
    ):
        findings.append(
            Finding(
                "WARN",
                f"state.md uses a legacy continuation record; migrate it to {STATE_SCHEMA} Stop Check before continuing",
            )
        )
    if current_contract:
        next_moves_section = (
            "## Next Move Shortlist"
            if latest_stop_check or prior_current_stop_check
            else "## Next Focus Candidates"
        )
        required_sections = [
            "## Current Focus Contract",
            stop_section,
            "## Current Knowledge Needed For This Focus",
            "## Requirement And Capability Coverage",
            next_moves_section,
            "## How We Got Here",
            "## Active Human Steering",
        ]
    else:
        required_sections = [
            "## High-Level Challenge And Metric Meaning",
            "## Current Research Direction",
            "## Development Stages",
            "## Major Performance Issues And Hypotheses",
            "## History: Tried Baselines and Directions",
            "## Research Lessons",
            "## Human Steering In Force",
        ]
    for section in required_sections:
        if section not in text:
            findings.append(Finding("WARN", f"state.md missing section: {section[3:]}"))
    focus_sections = [section for section in required_sections if section != stop_section]
    if (
        current_contract
        and state_path.parent.name != "challenge_template"
        and all(section in text for section in focus_sections)
    ):
        focus_fields = (
            "focus ID / last changed",
            "active baseline / direction",
            "baseline exploration item",
            "challenge target / remaining gap",
            "objective now",
            "core bottleneck / problem now",
            "why this is the focus",
            "stage-level decision to resolve",
            "evidence needed / exit condition",
            "protected guardrails / non-regression",
            "out of scope / do not drift into",
            "focus decision / authority",
        )
        missing = [field for field in focus_fields if not meaningful_value(read_field(text, field))]
        if missing:
            findings.append(
                Finding(
                    "WARN", f"state.md Current Focus Contract is incomplete: {', '.join(missing)}"
                )
            )
        focus_value = read_field(text, "focus ID / last changed")
        if meaningful_value(focus_value) and not FOCUS_ID_RE.search(focus_value):
            findings.append(Finding("WARN", "state.md current focus has no F0001-style ID"))
        authority = read_field(text, "focus decision / authority")
        if meaningful_value(authority) and not TRAJECTORY_ID_RE.search(authority):
            findings.append(
                Finding("WARN", "state.md current focus authority has no T0001-style trajectory ID")
            )
        exploration_item = read_field(text, "baseline exploration item")
        if (
            meaningful_value(exploration_item)
            and not clean_markdown_cell(exploration_item).lower().startswith("not applicable:")
            and not EXPLORATION_ID_RE.search(exploration_item)
        ):
            findings.append(
                Finding("WARN", "state.md current focus has no E001-style Exploration View ID")
            )
        active_baseline = read_field(text, "active baseline / direction")
        if meaningful_value(active_baseline) and not first_baseline_id(active_baseline):
            findings.append(
                Finding("WARN", "state.md active baseline does not name a baselineXXX direction")
            )
        if latest_stop_check or prior_current_stop_check:
            next_move_rows = first_markdown_table_rows(
                markdown_section(text, "Next Move Shortlist")
            )
            if len(next_move_rows) > 3:
                findings.append(
                    Finding("WARN", "state.md has more than three Next Move Shortlist items")
                )
            for row in next_move_rows:
                scope = clean_markdown_cell(row.get("scope", "")).lower()
                item = clean_markdown_cell(row.get("item", ""))
                trigger = row.get("trigger / why it may be next", "")
                if scope not in ("internal", "external"):
                    findings.append(
                        Finding(
                            "WARN", f"state.md Next Move has unknown scope: {scope or '<missing>'}"
                        )
                    )
                elif scope == "internal" and not EXPLORATION_ID_RE.search(item):
                    findings.append(
                        Finding(
                            "WARN", f"state.md internal Next Move lacks an E001-style ID: {item}"
                        )
                    )
                elif scope == "external" and not first_baseline_id(item):
                    findings.append(
                        Finding(
                            "WARN",
                            f"state.md external Next Move lacks a baselineXXX direction: {item}",
                        )
                    )
                if not meaningful_value(trigger):
                    findings.append(
                        Finding(
                            "WARN",
                            f"state.md Next Move {item or '<unknown>'} is missing its trigger",
                        )
                    )
        else:
            next_focus_rows = first_markdown_table_rows(
                markdown_section(text, "Next Focus Candidates")
            )
            if len(next_focus_rows) > 3:
                findings.append(
                    Finding("WARN", "state.md has more than three Next Focus Candidates")
                )
            for row in next_focus_rows:
                item = clean_markdown_cell(row.get("exploration item", ""))
                if not EXPLORATION_ID_RE.search(item):
                    findings.append(
                        Finding(
                            "WARN", f"state.md Next Focus Candidate lacks an E001-style ID: {item}"
                        )
                    )
                if not meaningful_value(row.get("why it may be next", "")):
                    findings.append(
                        Finding(
                            "WARN",
                            f"state.md Next Focus Candidate {item or '<unknown>'} is missing: why it may be next",
                        )
                    )
        knowledge_rows = first_markdown_table_rows(
            markdown_section(text, "Current Knowledge Needed For This Focus")
        )
        if len(knowledge_rows) > 5:
            findings.append(Finding("WARN", "state.md has more than five current beliefs"))

        coverage_rows = first_markdown_table_rows(
            markdown_section(text, "Requirement And Capability Coverage")
        )
        for row in coverage_rows:
            status = clean_markdown_cell(row.get("current status", "")).lower()
            if status.startswith(("verified", "done", "complete", "passed")):
                requirement = clean_markdown_cell(row.get("requirement / capability", ""))
                findings.append(
                    Finding(
                        "WARN",
                        f"state.md Requirement And Capability Coverage retains verified item {requirement or '<unknown>'}; keep verified facts in their owning artifact",
                    )
                )

        history_rows = first_markdown_table_rows(markdown_section(text, "How We Got Here"))
        if len(history_rows) > 4:
            findings.append(
                Finding("WARN", "state.md How We Got Here has more than four causal milestones")
            )

        if (latest_stop_check or prior_current_stop_check) and "## Stop Check" in text:
            stop_fields = (
                "result versus target / validity",
                "compute used / remaining / decision",
                "internal-loop exit gate",
                "external-loop exit gate",
                "agent judgment / last checked",
                "next action or stop rationale",
                "continuity proof before yield",
            )
            missing = [
                field for field in stop_fields if not meaningful_value(read_field(text, field))
            ]
            if missing:
                findings.append(
                    Finding("WARN", f"state.md Stop Check is incomplete: {', '.join(missing)}")
                )

            result_check = read_field(text, "result versus target / validity")
            if meaningful_value(result_check) and not markdown_links(result_check):
                findings.append(
                    Finding(
                        "WARN", "state.md Stop Check result/target judgment needs an evidence link"
                    )
                )

            compute_check = read_field(text, "compute used / remaining / decision")
            if meaningful_value(compute_check) and (
                "dashboard.md" not in compute_check or not markdown_links(compute_check)
            ):
                findings.append(
                    Finding("WARN", "state.md Stop Check compute judgment must link dashboard.md")
                )

            evidence_fields = (
                "internal-loop exit gate",
                "external-loop exit gate",
            )
            for field in evidence_fields:
                value = read_field(text, field)
                value_lower = clean_markdown_cell(value).lower()
                if (
                    meaningful_value(value)
                    and not value_lower.startswith("not required:")
                    and not (
                        markdown_links(value)
                        or EXPLORATION_ID_RE.search(value)
                        or LITERATURE_EVIDENCE_ID_RE.search(value)
                    )
                ):
                    findings.append(
                        Finding(
                            "WARN", f"state.md Stop Check {field} needs linked or ID-based evidence"
                        )
                    )

            internal_gate = clean_markdown_cell(read_field(text, "internal-loop exit gate")).lower()
            external_gate = clean_markdown_cell(read_field(text, "external-loop exit gate")).lower()
            allowed_gate_statuses = ("continue:", "passed:", "not required:")
            for label, value, raw_value in (
                ("internal-loop", internal_gate, read_field(text, "internal-loop exit gate")),
                ("external-loop", external_gate, read_field(text, "external-loop exit gate")),
            ):
                if meaningful_value(value) and not value.startswith(allowed_gate_statuses):
                    findings.append(
                        Finding("WARN", f"state.md {label} exit gate has unknown judgment: {value}")
                    )
                elif (
                    latest_stop_check
                    and value.startswith("not required:")
                    and not valid_stop_gate_exemption(raw_value)
                ):
                    findings.append(
                        Finding(
                            "WARN",
                            f"state.md {label} exit gate exemption requires an explicit terminal target with evidence or a linked human-stop decision",
                        )
                    )
            internal_raw = read_field(text, "internal-loop exit gate")
            if (
                meaningful_value(internal_raw)
                and not internal_gate.startswith("not required:")
                and ("baseline" not in internal_gate or not markdown_links(internal_raw))
            ):
                findings.append(
                    Finding(
                        "WARN",
                        "state.md internal-loop exit gate must link the active baseline Exploration View",
                    )
                )
            if internal_gate.startswith("passed:"):
                baseline_id = first_baseline_id(read_field(text, "active baseline / direction"))
                baseline_path = state_path.parent / "baselines" / baseline_id / "baseline.md"
                if not baseline_id or not baseline_path.is_file():
                    findings.append(
                        Finding(
                            "WARN",
                            "state.md internal-loop exit gate cannot verify the active baseline Exploration View",
                        )
                    )
                else:
                    unresolved_items = []
                    latest_baseline = has_template_schema(read_text(baseline_path), BASELINE_SCHEMA)
                    for row in first_markdown_table_rows(
                        markdown_section(read_text(baseline_path), "Exploration View")
                    ):
                        priority = clean_markdown_cell(row.get("priority / trigger", ""))
                        status = clean_markdown_cell(row.get("status", "")).lower()
                        if re.search(r"\bP[01]\b", priority, re.IGNORECASE) and status.startswith(
                            ("open", "planned", "active", "deferred")
                            if latest_baseline
                            else ("open", "planned", "active")
                        ):
                            unresolved_items.append(
                                clean_markdown_cell(row.get("id", "")) or "<unknown>"
                            )
                    if unresolved_items:
                        findings.append(
                            Finding(
                                "WARN",
                                "state.md internal-loop exit gate passed with unresolved P0/P1 Exploration View items: "
                                + ", ".join(unresolved_items),
                            )
                        )
            external_raw = read_field(text, "external-loop exit gate")
            if (
                meaningful_value(external_raw)
                and not external_gate.startswith("not required:")
                and ("candidates.md" not in external_gate or not markdown_links(external_raw))
            ):
                findings.append(
                    Finding(
                        "WARN",
                        "state.md external-loop exit gate must link candidates.md Candidate Comparison",
                    )
                )
            if external_gate.startswith("passed:"):
                candidates_path = state_path.parent / "candidates.md"
                candidates_text = read_text(candidates_path)
                unresolved_candidates = []
                required_candidate_schema = (
                    CANDIDATE_SCHEMA if latest_stop_check else CANDIDATE_SCHEMA_V7
                )
                if not has_template_schema(candidates_text, required_candidate_schema):
                    findings.append(
                        Finding(
                            "WARN",
                            f"state.md external-loop exit gate requires candidates.md {required_candidate_schema}",
                        )
                    )
                else:
                    for row in first_markdown_table_rows(
                        markdown_section(candidates_text, "Candidate Comparison")
                    ):
                        candidate_id = first_baseline_id(row.get("candidate / direction", ""))
                        status = clean_markdown_cell(row.get("status / evidence", "")).lower()
                        if candidate_id and status.startswith(
                            ("open", "required", "selected", "active")
                        ):
                            unresolved_candidates.append(candidate_id)
                if unresolved_candidates:
                    findings.append(
                        Finding(
                            "WARN",
                            "state.md external-loop exit gate passed with unresolved Candidate Comparison directions: "
                            + ", ".join(unresolved_candidates),
                        )
                    )

            judgment_raw = clean_markdown_cell(read_field(text, "agent judgment / last checked"))
            judgment = judgment_raw.lower()
            allowed_verdicts = (
                "continue: internal",
                "continue: external",
                "continue: monitoring",
                "blocked:",
                "ready to freeze:",
                "paused: human",
                "stopped: human",
                "completed: final evaluation",
            )
            if meaningful_value(judgment) and not judgment.startswith(allowed_verdicts):
                findings.append(
                    Finding("WARN", f"state.md has unknown Stop Check judgment: {judgment}")
                )

            if judgment.startswith("continue: internal") and not internal_gate.startswith(
                "continue:"
            ):
                findings.append(
                    Finding("WARN", "continue: internal requires an open internal-loop exit gate")
                )
            if judgment.startswith("continue: external"):
                if not internal_gate.startswith("passed:"):
                    findings.append(
                        Finding(
                            "WARN",
                            "continue: external requires the internal-loop exit gate to pass",
                        )
                    )
                if not external_gate.startswith("continue:"):
                    findings.append(
                        Finding(
                            "WARN", "continue: external requires an open external-loop exit gate"
                        )
                    )

            if judgment.startswith(("ready to freeze:", "completed: final evaluation")):
                if not TRAJECTORY_ID_RE.search(judgment_raw):
                    findings.append(
                        Finding(
                            "WARN",
                            "state.md ready/completed Stop Check judgment has no T0001-style trajectory ID",
                        )
                    )
                if "trajectory.md" not in judgment or not markdown_links(
                    read_field(text, "agent judgment / last checked")
                ):
                    findings.append(
                        Finding(
                            "WARN",
                            "state.md ready/completed Stop Check judgment must link trajectory.md",
                        )
                    )

            if judgment.startswith(
                ("paused: human", "stopped: human")
            ) and not TRAJECTORY_ID_RE.search(judgment_raw):
                findings.append(
                    Finding(
                        "WARN",
                        "state.md human pause/stop judgment has no T0001-style trajectory ID",
                    )
                )

            if judgment.startswith(("continue:", "blocked:")):
                continuity = clean_markdown_cell(
                    read_field(text, "continuity proof before yield")
                ).lower()
                if meaningful_value(continuity) and "heartbeat" not in continuity:
                    findings.append(
                        Finding(
                            "WARN",
                            "state.md continuing/blocked run has no heartbeat in continuity proof",
                        )
                    )
                if meaningful_value(continuity) and (
                    "dashboard.md" not in continuity
                    or not markdown_links(read_field(text, "continuity proof before yield"))
                ):
                    findings.append(
                        Finding(
                            "WARN",
                            "state.md continuing/blocked run continuity proof must link dashboard.md",
                        )
                    )
        elif "## Auto-Continue Decision" in text:
            loop_fields = (
                "loop decision / last checked",
                "continuation / readiness reason",
                "unresolved completion gates",
                "next automatic action",
                "freeze / stop authority",
            )
            missing = [
                field for field in loop_fields if not meaningful_value(read_field(text, field))
            ]
            if missing:
                findings.append(
                    Finding(
                        "WARN",
                        f"state.md Auto-Continue Decision is incomplete: {', '.join(missing)}",
                    )
                )
            decision = clean_markdown_cell(read_field(text, "loop decision / last checked")).lower()
            allowed_decisions = (
                "continue: internal",
                "continue: external",
                "ready to freeze",
                "blocked:",
                "paused: human",
                "stopped: human",
                "completed: final evaluation",
            )
            if meaningful_value(decision) and not decision.startswith(allowed_decisions):
                findings.append(
                    Finding("WARN", f"state.md has unknown Auto-Continue Decision: {decision}")
                )
            if decision.startswith(
                (
                    "ready to freeze",
                    "paused: human",
                    "stopped: human",
                    "completed: final evaluation",
                )
            ):
                authority = read_field(text, "freeze / stop authority")
                if meaningful_value(authority) and not TRAJECTORY_ID_RE.search(authority):
                    findings.append(
                        Finding(
                            "WARN",
                            "state.md freeze/stop authority has no T0001-style trajectory ID",
                        )
                    )
    experiment_refs = set(re.findall(r"\bexp\d{3}\b", text, re.IGNORECASE))
    if len(experiment_refs) > 12:
        findings.append(
            Finding(
                "WARN",
                f"state.md links {len(experiment_refs)} experiments; retain only evidence needed to restore the current focus",
            )
        )
    check_state_steering_refs(state_path, text, findings)


def check_current_state(
    state_path: pathlib.Path,
    text: str,
    findings: list[Finding],
) -> None:
    for section in ("## Current Focus", "## Continue / Switch / Freeze Judgment"):
        if section not in text:
            findings.append(Finding("WARN", f"state.md missing section: {section[3:]}"))
    if state_path.parent.name == "challenge_template":
        return

    focus_fields = (
        "focus ID / last changed",
        "stage / active direction",
        "objective now",
        "bottleneck / current diagnosis",
        "internal exploration pointer",
        "external exploration pointer",
        "focus decision / authority",
    )
    missing = [field for field in focus_fields if not meaningful_value(read_field(text, field))]
    if missing:
        findings.append(
            Finding("WARN", f"state.md Current Focus is incomplete: {', '.join(missing)}")
        )
    if not read_field(text, "active human constraint").strip():
        findings.append(
            Finding("WARN", "state.md Current Focus is incomplete: active human constraint")
        )
    human_constraint = read_field(text, "active human constraint")
    if clean_markdown_cell(human_constraint).lower() != "none" and not markdown_links(
        human_constraint
    ):
        findings.append(
            Finding("WARN", "state.md active human constraint must link its owning artifact")
        )

    focus_value = read_field(text, "focus ID / last changed")
    if meaningful_value(focus_value) and not FOCUS_ID_RE.search(focus_value):
        findings.append(Finding("WARN", "state.md current focus has no F0001-style ID"))
    authority = read_field(text, "focus decision / authority")
    if meaningful_value(authority) and (
        not TRAJECTORY_ID_RE.search(authority) or not markdown_links(authority)
    ):
        findings.append(
            Finding("WARN", "state.md focus authority must link a T0001-style trajectory decision")
        )

    stage = clean_markdown_cell(read_field(text, "stage / active direction")).lower()
    if any(name in stage for name in ("implementation", "internal", "monitoring", "freeze")):
        if not first_baseline_id(stage):
            findings.append(
                Finding("WARN", "state.md active stage does not name its baselineXXX direction")
            )

    internal_pointer = read_field(text, "internal exploration pointer")
    internal_lower = clean_markdown_cell(internal_pointer).lower()
    if meaningful_value(internal_pointer) and not internal_lower.startswith(
        ("not yet created:", "not applicable:")
    ):
        if not EXPLORATION_ID_RE.search(internal_pointer):
            findings.append(Finding("WARN", "state.md internal pointer has no E001-style item"))
        if "baseline" not in internal_lower or not markdown_links(internal_pointer):
            findings.append(
                Finding(
                    "WARN",
                    "state.md internal pointer must link the active baseline Internal Exploration",
                )
            )

    external_pointer = read_field(text, "external exploration pointer")
    external_lower = clean_markdown_cell(external_pointer).lower()
    if meaningful_value(external_pointer) and not external_lower.startswith("not applicable:"):
        if not first_baseline_id(external_pointer):
            findings.append(
                Finding("WARN", "state.md external pointer has no baselineXXX direction")
            )
        if "candidates.md" not in external_lower or not markdown_links(external_pointer):
            findings.append(Finding("WARN", "state.md external pointer must link candidates.md"))

    judgment_fields = (
        "internal improvement potential",
        "external improvement potential",
        "resource effect",
    )
    current_judgment_field = (
        "agent judgment"
        if has_template_schema(text, STATE_SCHEMA)
        else "agent judgment / last checked"
    )
    judgment_fields = judgment_fields + (current_judgment_field,)
    missing = [field for field in judgment_fields if not meaningful_value(read_field(text, field))]
    if missing:
        findings.append(
            Finding(
                "WARN",
                "state.md Continue / Switch / Freeze Judgment is incomplete: " + ", ".join(missing),
            )
        )

    potentials: dict[str, tuple[str, str]] = {}
    for scope in ("internal", "external"):
        field = f"{scope} improvement potential"
        raw = read_field(text, field)
        value = clean_markdown_cell(raw).lower()
        potentials[scope] = (raw, value)
        if meaningful_value(value) and not value.startswith(
            ("continue:", "no likely large gain:", "not applicable:")
        ):
            findings.append(Finding("WARN", f"state.md has unknown {field} judgment: {value}"))
            continue
        if value.startswith("continue:"):
            if not markdown_links(raw):
                findings.append(
                    Finding("WARN", f"state.md {field} needs a linked exploration owner")
                )
            if (
                scope == "internal"
                and not EXPLORATION_ID_RE.search(raw)
                and not awaiting_first_baseline_result(state_path, stage, internal_lower)
            ):
                findings.append(
                    Finding(
                        "WARN", "state.md continuing internal potential needs an E001-style item"
                    )
                )
        elif value.startswith("no likely large gain:"):
            if any(
                phrase in value
                for phrase in (
                    "because final evaluation",
                    "after final evaluation",
                    "because frozen",
                    "queue is empty",
                    "default run completed",
                    "job failed",
                )
            ):
                findings.append(
                    Finding(
                        "WARN",
                        f"state.md {field} uses an operational event as scientific exit evidence",
                    )
                )
        elif value.startswith("not applicable:") and not valid_current_potential_exemption(raw):
            findings.append(
                Finding(
                    "WARN",
                    f"state.md {field} exemption requires linked terminal-target evidence or a linked human-stop decision",
                )
            )

    resource = read_field(text, "resource effect")
    if meaningful_value(resource) and (
        "dashboard.md" not in resource or not markdown_links(resource)
    ):
        findings.append(Finding("WARN", "state.md resource effect must link dashboard.md"))

    verdict_raw = clean_markdown_cell(read_field(text, current_judgment_field))
    verdict = verdict_raw.lower()
    allowed_verdicts = (
        "continue: internal",
        "continue: external",
        "continue: monitoring",
        "blocked:",
        "ready to freeze:",
        "paused: human",
        "stopped: human",
        "completed: final evaluation",
    )
    if meaningful_value(verdict) and not verdict.startswith(allowed_verdicts):
        findings.append(Finding("WARN", f"state.md has unknown scientific judgment: {verdict}"))
    if verdict.startswith("continue: internal") and not potentials["internal"][1].startswith(
        "continue:"
    ):
        findings.append(
            Finding("WARN", "continue: internal requires continuing internal improvement potential")
        )
    if verdict.startswith("continue: external") and not potentials["external"][1].startswith(
        "continue:"
    ):
        findings.append(
            Finding("WARN", "continue: external requires continuing external improvement potential")
        )
    if verdict.startswith(("ready to freeze:", "completed: final evaluation")):
        for scope, (raw, value) in potentials.items():
            if not (
                value.startswith("no likely large gain:")
                or (value.startswith("not applicable:") and valid_current_potential_exemption(raw))
            ):
                findings.append(
                    Finding(
                        "WARN", f"ready to freeze requires resolved {scope} improvement potential"
                    )
                )
        if not TRAJECTORY_ID_RE.search(verdict_raw) or not markdown_links(
            read_field(text, current_judgment_field)
        ):
            findings.append(
                Finding("WARN", "ready/completed judgment must link a T0001 trajectory decision")
            )


def awaiting_first_baseline_result(
    state_path: pathlib.Path,
    stage: str,
    internal_pointer: str,
) -> bool:
    """Allow the workflow's unseeded exploration state only before any Outcome."""
    if not internal_pointer.startswith("not yet created:"):
        return False
    baseline_id = first_baseline_id(stage)
    if not baseline_id:
        return False
    baseline_dir = state_path.parent / "baselines" / baseline_id
    baseline_path = baseline_dir / "baseline.md"
    if not baseline_path.is_file():
        return False
    exploration = markdown_section(read_text(baseline_path), "Internal Exploration")
    if any(
        meaningful_value(row.get("id", ""))
        or meaningful_value(row.get("possible improvement / question", ""))
        for row in first_markdown_table_rows(exploration)
    ):
        return False
    return not any(
        markdown_section(read_text(path), "Outcome")
        for path in collect_experiment_files(baseline_dir)
    )


def valid_current_potential_exemption(value: str) -> bool:
    status = clean_markdown_cell(value).lower()
    if not status.startswith("not applicable:"):
        return False
    if "human stop" in status:
        return bool(TRAJECTORY_ID_RE.search(value) and markdown_links(value))
    if "explicit terminal target" in status:
        return bool(markdown_links(value))
    return False


def check_state_steering_refs(
    state_path: pathlib.Path,
    state_text: str,
    findings: list[Finding],
) -> None:
    section = markdown_section(state_text, "Active Human Steering") or markdown_section(
        state_text, "Human Steering In Force"
    )
    for row in first_markdown_table_rows(section):
        value = (
            row.get("trajectory link", "")
            or row.get("source", "")
            or row.get("steering id / link", "")
        )
        if not meaningful_value(value):
            continue
        match = re.fullmatch(r"\[[^\]]+\]\(([^)]+)\)", value.strip())
        if not match:
            findings.append(
                Finding("WARN", f"state.md active steering source is not a Markdown link: {value}")
            )
            continue
        href = match.group(1)
        path_part, _, _ = href.partition("#")
        target = (state_path.parent / path_part).resolve()
        if not target.is_file():
            findings.append(
                Finding("WARN", f"state.md active steering target does not exist: {path_part}")
            )


def check_trajectory(trajectory_path: pathlib.Path, findings: list[Finding]) -> None:
    text = read_text(trajectory_path)
    section = markdown_section(text, "Decision Trace")
    if not section:
        findings.append(Finding("WARN", "trajectory.md missing Decision Trace section"))
        return
    required = {
        "id",
        "time",
        "actor",
        "decision / steering",
        "reasoning summary",
        "evidence / owning artifact",
        "applied action",
    }
    header = set(first_markdown_table_header(section))
    for column in sorted(required - header):
        findings.append(
            Finding("WARN", f"trajectory.md Decision Trace is missing column: {column}")
        )
    seen: set[str] = set()
    is_raw_template = trajectory_path.parent.name == "challenge_template"
    for row in first_markdown_table_rows(section):
        trajectory_id = clean_markdown_cell(row.get("id", ""))
        if not meaningful_value(trajectory_id):
            continue
        if not TRAJECTORY_ID_RE.fullmatch(trajectory_id):
            findings.append(
                Finding("WARN", f"trajectory.md has invalid decision ID: {trajectory_id}")
            )
        if trajectory_id in seen:
            findings.append(Finding("WARN", f"trajectory.md repeats decision ID: {trajectory_id}"))
        seen.add(trajectory_id)
        if is_raw_template:
            continue
        for column in required - {"id"}:
            if not meaningful_value(clean_markdown_cell(row.get(column, ""))):
                findings.append(
                    Finding(
                        "WARN",
                        f"trajectory.md {trajectory_id} has no meaningful value for: {column}",
                    )
                )
        evidence = row.get("evidence / owning artifact", "")
        links = markdown_links(evidence)
        if not links:
            findings.append(
                Finding(
                    "WARN",
                    f"trajectory.md {trajectory_id} evidence/owner is not an inspectable Markdown link",
                )
            )
        for href in links:
            check_local_link(trajectory_path, href, findings, f"trajectory.md {trajectory_id}")


def check_trajectory_references(
    challenge_dir: pathlib.Path,
    findings: list[Finding],
) -> None:
    trajectory_path = challenge_dir / "trajectory.md"
    section = markdown_section(read_text(trajectory_path), "Decision Trace")
    known_ids = {
        clean_markdown_cell(row.get("id", ""))
        for row in first_markdown_table_rows(section)
        if TRAJECTORY_ID_RE.fullmatch(clean_markdown_cell(row.get("id", "")))
    }

    references: list[tuple[str, str]] = []
    state_path = challenge_dir / "state.md"
    if state_path.is_file():
        state_text = read_text(state_path)
        references.append(
            (
                "state.md focus decision / authority",
                read_field(state_text, "focus decision / authority"),
            )
        )
        steering = markdown_section(state_text, "Active Human Steering")
        for row_number, row in enumerate(first_markdown_table_rows(steering), start=1):
            references.append(
                (
                    f"state.md Active Human Steering row {row_number}",
                    row.get("trajectory link", ""),
                )
            )

    dashboard_path = challenge_dir / "dashboard.md"
    if dashboard_path.is_file():
        references.append(
            (
                "dashboard.md Latest trajectory decision",
                read_field(read_text(dashboard_path), "Latest trajectory decision"),
            )
        )

    candidates_path = challenge_dir / "candidates.md"
    if candidates_path.is_file():
        selected = markdown_section(read_text(candidates_path), "Selected Workflow")
        references.append(
            ("candidates.md Selected Workflow", read_field(selected, "trajectory ID"))
        )

    for baseline_path in collect_baseline_files(challenge_dir):
        for experiment_path in collect_experiment_files(baseline_path.parent):
            outcome = markdown_section(read_text(experiment_path), "Outcome")
            trajectory_value = read_field(outcome, "downstream updates") or read_field(
                outcome, "trajectory decision"
            )
            references.append(
                (
                    str(experiment_path.relative_to(challenge_dir)),
                    trajectory_value,
                )
            )

    for label, value in references:
        if not meaningful_value(value):
            continue
        ids = set(TRAJECTORY_ID_RE.findall(clean_markdown_cell(value)))
        if not ids and "trajectory" in label.lower():
            findings.append(Finding("WARN", f"{label} has no T0001-style decision reference"))
            continue
        for trajectory_id in sorted(ids - known_ids):
            findings.append(
                Finding(
                    "WARN",
                    f"{label} references missing trajectory decision {trajectory_id}",
                )
            )


def markdown_links(markdown: str) -> list[str]:
    return [match.strip() for match in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", markdown)]


def check_local_link(
    source_path: pathlib.Path,
    href: str,
    findings: list[Finding],
    label: str,
) -> None:
    target_text = href.strip().strip("<>")
    if (
        not target_text
        or target_text.startswith("#")
        or re.match(r"^(?:https?|mailto|data|javascript):", target_text, re.IGNORECASE)
        or "<" in target_text
        or ">" in target_text
    ):
        return
    path_text = urllib.parse.unquote(target_text.split("#", 1)[0].split("?", 1)[0])
    if not path_text:
        return
    candidate = pathlib.Path(path_text)
    if candidate.is_absolute():
        findings.append(
            Finding("WARN", f"{label} uses a non-portable absolute Markdown link: {path_text}")
        )
        return
    target = (source_path.parent / candidate).resolve()
    if not target.exists():
        findings.append(Finding("WARN", f"{label} has a broken local link: {path_text}"))


def check_shared_visualization(
    challenge_dir: pathlib.Path,
    findings: list[Finding],
) -> None:
    shared_dir = challenge_dir / "shared" / "visualization"
    contract = shared_dir / "README.md"
    if not contract.is_file():
        findings.append(
            Finding("WARN", "missing shared visualization contract: shared/visualization/README.md")
        )
        return
    if challenge_dir.name == "challenge_template":
        return
    contract_text = read_text(contract)
    required_fields = (
        "task type",
        "implementation",
        "canonical command",
        "normalized inputs",
        "validated by",
    )
    missing = [
        field for field in required_fields if not meaningful_value(read_field(contract_text, field))
    ]
    if missing:
        findings.append(
            Finding(
                "WARN",
                f"shared visualization contract is incomplete; missing: {', '.join(missing)}",
            )
        )
    has_runs = any(challenge_dir.glob("baselines/*/runs/exp*/experiment.md")) or any(
        challenge_dir.glob("baselines/*/exp*/experiment.md")
    )
    if not has_runs:
        return

    implementation = shared_dir / "visualizer.py"
    if not implementation.is_file():
        findings.append(
            Finding(
                "WARN",
                "experiment records exist but shared/visualization/visualizer.py is missing",
            )
        )
        return

    implementation_text = read_text(implementation)
    if re.search(r"^\s*TEMPLATE_STUB\s*=\s*True\s*$", implementation_text, re.MULTILINE):
        findings.append(Finding("WARN", "shared visualization still contains the template stub"))
        return

    try:
        tree = ast.parse(implementation_text)
    except SyntaxError as exc:
        findings.append(Finding("WARN", f"shared visualization is not valid Python: {exc.msg}"))
        return

    function_names = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing_api = sorted({"visualize_case", "visualize_run"} - function_names)
    if missing_api:
        findings.append(
            Finding(
                "WARN",
                f"shared visualization is missing callable API: {', '.join(missing_api)}",
            )
        )
    if (
        'if __name__ == "__main__"' not in implementation_text
        and "if __name__ == '__main__'" not in implementation_text
    ):
        findings.append(Finding("WARN", "shared visualization has no canonical CLI entry point"))


def dashboard_baseline_sections(dashboard_text: str) -> list[tuple[str, str]]:
    sections = direct_dashboard_baseline_sections(dashboard_text)

    for wrapper_name in ("Baseline Boards", "Baseline Summary"):
        wrapper = markdown_section(dashboard_text, wrapper_name)
        if not wrapper:
            continue
        for title, body in markdown_subsections(wrapper, "### "):
            if is_baseline_title(title):
                sections.append((normalize_baseline_title(title), body))
    return sections


def direct_dashboard_baseline_sections(dashboard_text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    for title, body in markdown_subsections(dashboard_text, "## "):
        if is_baseline_title(title):
            sections.append((normalize_baseline_title(title), body))
    return sections


def markdown_heading(line: str) -> tuple[int, str] | None:
    match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
    if not match:
        return None
    return len(match.group(1)), match.group(2).strip()


def is_baseline_title(title: str) -> bool:
    return bool(BASELINE_TITLE_RE.match(normalize_baseline_title(title)))


def normalize_baseline_title(title: str) -> str:
    return clean_markdown_cell(title).strip().lower().replace(" ", "_")


def first_baseline_id(value: str) -> str:
    match = BASELINE_REFERENCE_RE.search(clean_markdown_cell(value))
    return normalize_baseline_title(match.group(0)) if match else ""


def candidate_scientific_rank(value: str) -> Optional[int]:
    status = clean_markdown_cell(value)
    match = re.match(r"^(?:rank\s*)?(\d+)\s*[:.)-]\s*(.+)$", status, re.IGNORECASE)
    if not match or not meaningful_value(match.group(2)):
        return None
    rank = int(match.group(1))
    return rank if rank > 0 else None


def candidate_rating(value: str) -> Optional[int]:
    text = clean_markdown_cell(value)
    if not re.fullmatch(r"[1-5]", text):
        return None
    return int(text)


def candidate_score_and_rank(value: str) -> tuple[Optional[int], Optional[int]]:
    text = clean_markdown_cell(value)
    match = re.fullmatch(r"(\d+)\s*/\s*(?:rank\s*)?(\d+)", text, re.IGNORECASE)
    if not match:
        return None, None
    total = int(match.group(1))
    rank = int(match.group(2))
    if not 5 <= total <= 25 or rank < 1:
        return None, None
    return total, rank


def candidate_feasibility_value(candidate_block: str) -> str:
    return read_field(candidate_block, "implementation feasibility / changes") or read_field(
        candidate_block, "code / checkpoint"
    )


def check_promotion_obligations(
    candidates_text: str,
    candidate_blocks: dict[str, str],
    candidate_ranks: dict[str, int],
    selected_id: str,
    findings: list[Finding],
) -> None:
    section = markdown_section(candidates_text, "Promotion Obligations")
    if not section:
        findings.append(Finding("WARN", "candidates.md is missing Promotion Obligations"))
        return
    rows = [
        row
        for row in first_markdown_table_rows(section)
        if meaningful_value(row.get("candidate", ""))
    ]
    obligation_rows: dict[str, dict[str, str]] = {}
    allowed_statuses = ("required", "active", "promoted", "rejected", "blocked")
    for row_number, row in enumerate(rows, start=1):
        candidate_id = first_baseline_id(row.get("candidate", ""))
        if not candidate_id or candidate_id not in candidate_blocks:
            findings.append(
                Finding(
                    "WARN",
                    f"Promotion Obligation row {row_number} does not name one candidate baseline",
                )
            )
            continue
        if candidate_id in obligation_rows:
            findings.append(Finding("WARN", f"Promotion Obligations repeats {candidate_id}"))
            continue
        obligation_rows[candidate_id] = row
        probe = row.get("bounded probe / comparison", "")
        status = clean_markdown_cell(row.get("status", "")).lower()
        evidence = row.get("evidence", "")
        if not meaningful_value(probe):
            findings.append(
                Finding("WARN", f"Promotion Obligation {candidate_id} has no bounded probe")
            )
        if not status.startswith(allowed_statuses):
            findings.append(
                Finding("WARN", f"Promotion Obligation {candidate_id} has unknown status: {status}")
            )
            continue
        if status.startswith("rejected") and not has_experiment_evidence_link(evidence):
            findings.append(
                Finding(
                    "WARN",
                    f"Promotion Obligation {candidate_id} rejection lacks linked experiment evidence",
                )
            )
        elif status.startswith("blocked") and not verified_hard_blocker(status, evidence):
            findings.append(
                Finding(
                    "WARN",
                    f"Promotion Obligation {candidate_id} lacks verified hard-blocker evidence",
                )
            )
        elif status.startswith("promoted") and not markdown_links(evidence):
            findings.append(
                Finding(
                    "WARN",
                    f"Promotion Obligation {candidate_id} promotion lacks a decision or experiment link",
                )
            )

    selected_rank = candidate_ranks.get(selected_id)
    if selected_rank is None:
        return
    for candidate_id, rank in candidate_ranks.items():
        if rank >= selected_rank:
            continue
        candidate_code = candidate_feasibility_value(candidate_blocks[candidate_id])
        if candidate_feasibility_is_blocked(candidate_code, require_hard_blocker=True):
            continue
        if candidate_id not in obligation_rows:
            findings.append(
                Finding(
                    "WARN",
                    f"higher-ranked candidate {candidate_id} is neither a Promotion Obligation nor a verified hard blocker",
                )
            )


def has_experiment_evidence_link(value: str) -> bool:
    return any(
        EXP_ID_RE.search(href) or re.search(r"(?:^|/)runs/[^/]+/experiment\.md", href)
        for href in markdown_links(value)
    )


def verified_hard_blocker(status: str, evidence: str = "") -> bool:
    combined = f"{clean_markdown_cell(status)} {clean_markdown_cell(evidence)}".lower()
    return (
        combined.startswith("blocked: hard blocker")
        and meaningful_value(combined.removeprefix("blocked: hard blocker").lstrip(" :-—"))
        and bool(markdown_links(f"{status} {evidence}"))
    )


def candidate_feasibility_status_is_valid(
    value: str,
    *,
    require_hard_blocker: bool = False,
) -> bool:
    status = clean_markdown_cell(value).lower()
    if status == "not checked":
        return True
    return candidate_feasibility_is_verified(value) or candidate_feasibility_is_blocked(
        value,
        require_hard_blocker=require_hard_blocker,
    )


def candidate_feasibility_is_verified(value: str) -> bool:
    status = clean_markdown_cell(value).lower()
    return (
        status.startswith("verified:")
        and meaningful_value(status.removeprefix("verified:").strip())
        and bool(markdown_links(value))
    )


def candidate_feasibility_is_blocked(
    value: str,
    *,
    require_hard_blocker: bool = False,
) -> bool:
    status = clean_markdown_cell(value).lower()
    if require_hard_blocker:
        return verified_hard_blocker(status)
    return (
        status.startswith("blocked:")
        and meaningful_value(status.removeprefix("blocked:").strip())
        and bool(markdown_links(value))
    )


def check_dashboard_readability(
    dashboard_path: pathlib.Path, dashboard_text: str, findings: list[Finding]
) -> None:
    lines = dashboard_text.splitlines()
    if len(lines) > DASHBOARD_SOFT_MAX_LINES:
        findings.append(
            Finding(
                "WARN",
                f"dashboard.md is {len(lines)} lines; keep live sections compact and consolidate non-viewer index content before {DASHBOARD_SOFT_MAX_LINES}",
            )
        )

    for line_number, line in enumerate(lines, start=1):
        if line.strip().startswith("|") and len(line) > DASHBOARD_TABLE_LINE_SOFT_MAX:
            findings.append(
                Finding(
                    "WARN",
                    f"dashboard.md line {line_number} has a very long table row; move detail to the experiment Outcome",
                )
            )

    for section_name, row_limit in DASHBOARD_SECTION_ROW_LIMITS.items():
        section = markdown_section(dashboard_text, section_name)
        if not section:
            continue
        rows = count_markdown_table_data_rows(section)
        if rows > row_limit:
            findings.append(
                Finding(
                    "WARN",
                    f"dashboard.md {section_name} has {rows} rows; target at most {row_limit}",
                )
            )


def check_dashboard_readiness(
    dashboard_text: str,
    findings: list[Finding],
    *,
    require_status: bool,
) -> None:
    section = (
        markdown_section(dashboard_text, "Challenge Readiness Checklist")
        or markdown_section(dashboard_text, "Harness Checklist")
        or markdown_section(dashboard_text, "Data Checklist")
    )
    rows = first_markdown_table_rows(section)
    labels = " ".join(
        clean_markdown_cell(
            row.get("gate", "") or row.get("component", "") or row.get("item", "")
        ).lower()
        for row in rows
    )
    required_concepts = {
        "raw data": ("raw data", "data downloaded"),
        "split": ("split",),
        "metric/evaluator": ("metric", "evaluator"),
        "submission": ("submission", "packaging"),
        "scheduler/process status": ("scheduler", "process status"),
        "shared visualization": ("shared visualization", "visualizer"),
    }
    for label, alternatives in required_concepts.items():
        if not any(term in labels for term in alternatives):
            findings.append(
                Finding("WARN", f"dashboard.md readiness checklist is missing: {label}")
            )
    if require_status:
        for row in rows:
            gate = clean_markdown_cell(
                row.get("gate", "") or row.get("component", "") or row.get("item", "")
            )
            status = (
                row.get("status / evidence", "") or row.get("status", "") or row.get("answer", "")
            )
            if gate and not meaningful_value(status):
                findings.append(
                    Finding("WARN", f"dashboard.md readiness gate has no status/evidence: {gate}")
                )
            elif gate and not gate_is_complete(status):
                findings.append(
                    Finding(
                        "WARN",
                        f"dashboard.md readiness gate is not admission-ready: {gate} ({clean_markdown_cell(status)})",
                    )
                )


def check_dashboard_baseline_sections(dashboard_text: str, findings: list[Finding]) -> None:
    sections = dashboard_baseline_sections(dashboard_text)
    if not sections:
        return

    if direct_dashboard_baseline_sections(dashboard_text):
        for top_level in ("## What To Watch Next", "## Active Jobs"):
            if re.search(rf"^{re.escape(top_level)}\s*$", dashboard_text, re.MULTILINE):
                findings.append(
                    Finding(
                        "WARN",
                        f"dashboard.md uses {top_level}; active work belongs inside each baseline section",
                    )
                )

    for baseline, body in sections:
        for subsection_name, row_limit in DASHBOARD_BASELINE_SUBSECTION_ROW_LIMITS.items():
            subsection = markdown_subsection(body, f"### {subsection_name}") or markdown_subsection(
                body, f"#### {subsection_name}"
            )
            if not subsection:
                continue
            rows = count_markdown_table_data_rows(subsection)
            if rows > row_limit:
                findings.append(
                    Finding(
                        "WARN",
                        f"{baseline} {subsection_name} has {rows} rows; target at most {row_limit}",
                    )
                )
        completed = markdown_subsection(
            body, "### Important Completed Runs"
        ) or markdown_subsection(body, "#### Important Completed Runs")
        if completed:
            header = first_markdown_table_header(completed)
            if header and "result" not in header:
                findings.append(
                    Finding(
                        "WARN",
                        f"{baseline} Important Completed Runs table is missing a Result column",
                    )
                )
        active = markdown_subsection(body, "### Active Jobs") or markdown_subsection(
            body, "#### Active Jobs"
        )
        if active:
            exp_ids = [
                first_exp_id(" | ".join(row.values())) for row in first_markdown_table_rows(active)
            ]
            duplicates = sorted(
                {exp_id for exp_id in exp_ids if exp_id and exp_ids.count(exp_id) > 1}
            )
            if duplicates:
                findings.append(
                    Finding(
                        "WARN",
                        f"{baseline} Active Jobs repeats experiments by stage; group stage/job IDs into one row: {', '.join(duplicates)}",
                    )
                )
        planned = markdown_subsection(body, "### Planned Jobs") or markdown_subsection(
            body, "#### Planned Jobs"
        )
        if (
            planned
            and has_template_schema(dashboard_text, DASHBOARD_SCHEMA)
            and "launch group" not in first_markdown_table_header(planned)
        ):
            findings.append(
                Finding("WARN", f"{baseline} Planned Jobs is missing a Launch Group column")
            )


def check_dashboard_single_active_baseline(
    dashboard_text: str,
    findings: list[Finding],
) -> None:
    active_directions: list[str] = []
    for baseline, body in dashboard_baseline_sections(dashboard_text):
        active = markdown_subsection(body, "### Active Jobs") or markdown_subsection(
            body, "#### Active Jobs"
        )
        planned = markdown_subsection(body, "### Planned Jobs") or markdown_subsection(
            body, "#### Planned Jobs"
        )
        if count_markdown_table_data_rows(active) or count_markdown_table_data_rows(planned):
            active_directions.append(baseline)
    if len(active_directions) > 1:
        findings.append(
            Finding(
                "WARN",
                "dashboard.md has active/planned work under multiple directions: "
                + ", ".join(active_directions),
            )
        )


def check_dashboard_planned_experiments(
    challenge_dir: pathlib.Path, dashboard_text: str, findings: list[Finding]
) -> None:
    current_baseline = ""
    in_planned = False
    header: list[str] = []
    for line in dashboard_text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower().rstrip(":")
        heading = markdown_heading(stripped)
        if heading:
            level, title = heading
            title_lower = title.lower().rstrip(":")
            if is_baseline_title(title):
                current_baseline = normalize_baseline_title(title)
                in_planned = False
                header = []
                continue
            if title_lower == "planned jobs":
                in_planned = True
                header = []
                continue
            if title_lower in {"active jobs", "important completed runs", "key evidence"}:
                in_planned = False
                header = []
                continue
            if level <= 2:
                current_baseline = ""
                in_planned = False
                header = []
                continue
            in_planned = False
            header = []
            continue
        if lowered == "planned jobs":
            in_planned = True
            header = []
            continue
        if lowered in {"active jobs", "important completed runs", "key evidence"}:
            in_planned = False
            header = []
            continue
        if not in_planned or not stripped.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in stripped.strip("|").split("|")]
        if not cells or set(cells[0]) <= {"-"}:
            continue
        lower_cells = [cell.lower() for cell in cells]
        record_headers = {"experiment record", "record", "intent"}
        if (
            "exp" in lower_cells or "run" in lower_cells or "experiment" in lower_cells
        ) and record_headers.intersection(lower_cells):
            header = lower_cells
            continue
        exp = first_exp_id(" | ".join(cells))
        if not exp:
            continue
        record = next(
            (
                cell_by_header(cells, header, name)
                for name in record_headers
                if cell_by_header(cells, header, name)
            ),
            "",
        )
        if not meaningful_value(record):
            findings.append(
                Finding(
                    "WARN", f"{exp} planned experiment has no experiment.md path in dashboard.md"
                )
            )
            continue
        record_path = resolve_dashboard_path(challenge_dir, current_baseline, record)
        if record_path and not record_path.exists():
            findings.append(
                Finding("WARN", f"{exp} dashboard experiment path does not exist: {record}")
            )
        elif record_path:
            check_experiment_admission(exp, record_path, challenge_dir, findings)


def check_experiment_admission(
    exp_id: str,
    experiment_path: pathlib.Path,
    challenge_dir: pathlib.Path,
    findings: list[Finding],
    enforce_current_focus: bool = True,
) -> None:
    text = read_text(experiment_path)
    outcome = markdown_section(text, "Outcome")
    outcome_status = clean_markdown_cell(read_field(outcome, "result status / limitations")).lower()
    completed_outcome = outcome_status == "completed" or outcome_status.startswith(
        "completed with limitations"
    )
    if (
        uses_active_template_contract(text)
        and not has_template_schema(text, EXPERIMENT_SCHEMA)
        and not completed_outcome
    ):
        findings.append(
            Finding(
                "WARN",
                f"{exp_id} is a live legacy experiment record; reload the raw template and migrate it to {EXPERIMENT_SCHEMA} before launch or repair",
            )
        )
    focus_evidence = read_field(text, "current focus / exploration item")
    required_fields = {
        "parent / comparator": read_field(text, "parent / comparator"),
        "current focus / exploration item": focus_evidence,
        "question / hypothesis": read_field(text, "question / hypothesis"),
        "intervention": read_field(text, "intervention"),
        "reasoning / alternatives": read_field(text, "reasoning / alternatives"),
        "distinguishing evidence / falsifier": read_field(
            text, "distinguishing evidence / falsifier"
        ),
        "training / evaluation contract": read_field(text, "training / evaluation contract"),
        "execution / evidence plan": read_field(text, "execution / evidence plan"),
    }
    missing = [field for field, value in required_fields.items() if not meaningful_value(value)]
    if missing:
        findings.append(
            Finding(
                "WARN",
                f"{exp_id} Intent is incomplete; missing: {', '.join(missing)}",
            )
        )
    focus_value = clean_markdown_cell(focus_evidence)
    experiment_focus = FOCUS_ID_RE.search(focus_value)
    state_path = challenge_dir / "state.md"
    state_focus = None
    if state_path.is_file():
        state_focus = FOCUS_ID_RE.search(
            clean_markdown_cell(read_field(read_text(state_path), "focus ID / last changed"))
        )
    if meaningful_value(focus_value) and not experiment_focus:
        findings.append(Finding("WARN", f"{exp_id} Intent has no F0001-style focus ID"))
    state_uses_living_exploration = state_path.is_file() and (
        has_template_schema(read_text(state_path), STATE_SCHEMA)
        or has_template_schema(read_text(state_path), STATE_SCHEMA_V7)
    )
    internal_pointer = (
        read_field(read_text(state_path), "internal exploration pointer")
        if state_uses_living_exploration
        else ""
    )
    requires_exploration_id = (
        not completed_outcome
        and "smoke" not in exp_id.lower()
        and (not state_uses_living_exploration or bool(EXPLORATION_ID_RE.search(internal_pointer)))
    )
    if (
        meaningful_value(focus_value)
        and requires_exploration_id
        and not EXPLORATION_ID_RE.search(focus_value)
    ):
        findings.append(
            Finding("WARN", f"{exp_id} Intent has no E001-style Internal Exploration ID")
        )
    if (
        enforce_current_focus
        and experiment_focus
        and state_focus
        and experiment_focus.group(0) != state_focus.group(0)
    ):
        findings.append(
            Finding(
                "WARN",
                f"{exp_id} Intent targets {experiment_focus.group(0)} but state.md current focus is {state_focus.group(0)}",
            )
        )
    focus_remainder = FOCUS_ID_RE.sub("", EXPLORATION_ID_RE.sub("", focus_value))
    if experiment_focus and not re.sub(r"[\W_]+", "", focus_remainder):
        findings.append(
            Finding("WARN", f"{exp_id} Intent names its focus but has no relevance or evidence")
        )


def check_dashboard_key_visualizations(
    challenge_dir: pathlib.Path,
    dashboard_text: str,
    findings: list[Finding],
) -> None:
    baseline_sections = dashboard_baseline_sections(dashboard_text)
    indexed_baselines = {baseline for baseline, _ in baseline_sections}
    for baseline_dir in (challenge_dir / "baselines").glob("baseline*"):
        if not baseline_dir.is_dir() or baseline_dir.name.lower() in indexed_baselines:
            continue
        viewers = list(baseline_dir.glob("runs/exp*/result_viewer.html")) + list(
            baseline_dir.glob("exp*/result_viewer.html")
        )
        if viewers:
            findings.append(
                Finding(
                    "WARN",
                    f"{baseline_dir.name}: result viewers exist but the dashboard has no baseline section to index them",
                )
            )

    for baseline, body in baseline_sections:
        completed = markdown_subsection(
            body, "### Important Completed Runs"
        ) or markdown_subsection(body, "#### Important Completed Runs")
        baseline_dir = challenge_dir / "baselines" / baseline
        viewer_exp_ids = {
            viewer.parent.name
            for pattern in ("runs/exp*/result_viewer.html", "exp*/result_viewer.html")
            for viewer in baseline_dir.glob(pattern)
            if not any(marker in viewer.parent.name.lower() for marker in ("smoke", "canary"))
        }
        indexed_exp_ids = {
            exp_id
            for row in first_markdown_table_rows(completed)
            if (exp_id := first_exp_id(" | ".join(row.values())))
        }
        missing_from_index = sorted(viewer_exp_ids - indexed_exp_ids)
        if missing_from_index:
            findings.append(
                Finding(
                    "WARN",
                    f"{baseline}: completed runs with result_viewer.html are absent from Important Completed Runs: {', '.join(missing_from_index)}",
                )
            )
        if not completed:
            continue

        missing_record: list[str] = []
        missing_outcome: list[str] = []
        missing_viewer: list[str] = []
        for row in first_markdown_table_rows(completed):
            exp_id = first_exp_id(" | ".join(row.values()))
            if not exp_id:
                continue

            record_candidates = (
                baseline_dir / "runs" / exp_id / "experiment.md",
                baseline_dir / exp_id / "experiment.md",
            )
            record_path = next((path for path in record_candidates if path.is_file()), None)
            if record_path is None:
                missing_record.append(exp_id)
                continue

            record_text = read_text(record_path)
            outcome = markdown_section(record_text, "Outcome")
            if not outcome or not meaningful_value(
                read_field(outcome, "result status / limitations")
            ):
                missing_outcome.append(exp_id)
                continue
            viewer_path = record_path.parent / "result_viewer.html"
            viewer_answer = clean_markdown_cell(
                read_field(outcome, "result viewer / diagnostic evidence")
            )
            answer_lower = viewer_answer.lower()
            waived = (
                "waiv" in answer_lower and "human" in answer_lower
            ) or "no prediction" in answer_lower
            if not viewer_path.is_file() and not waived:
                missing_viewer.append(exp_id)

        for label, exp_ids in (
            ("missing experiment.md", missing_record),
            ("missing a valid Outcome", missing_outcome),
            ("missing result_viewer.html or an explicit human waiver", missing_viewer),
        ):
            if exp_ids:
                findings.append(
                    Finding(
                        "WARN",
                        f"{baseline}: Important Completed Runs {label}: {', '.join(exp_ids)}",
                    )
                )


def markdown_subsections(markdown: str, marker: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_lines: list[str] = []
    for line in markdown.splitlines():
        if line.startswith(marker):
            if current_title:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = line[len(marker) :].strip()
            current_lines = []
        elif current_title:
            current_lines.append(line)
    if current_title:
        sections.append((current_title, "\n".join(current_lines)))
    return sections


def markdown_subsection(markdown: str, header: str) -> str:
    start = markdown.find(header)
    if start == -1:
        return ""
    level = len(header) - len(header.lstrip("#"))
    pattern = re.compile(rf"\n#{{1,{level}}} ")
    match = pattern.search(markdown[start + len(header) :])
    if not match:
        return markdown[start:]
    return markdown[start : start + len(header) + match.start()]


def check_baseline(
    challenge_dir: pathlib.Path, baseline_path: pathlib.Path, findings: list[Finding]
) -> None:
    baseline_text = read_text(baseline_path)
    baseline_dir = baseline_path.parent
    baseline_id = read_field(baseline_text, "baseline id") or baseline_dir.name
    latest_baseline = has_template_schema(baseline_text, BASELINE_SCHEMA)
    check_human_steering(baseline_path, f"{baseline_id} baseline.md", findings)
    experiment_files = collect_experiment_files(baseline_dir)
    for experiment_path in experiment_files:
        check_human_steering(
            experiment_path, f"{baseline_id} {experiment_path.parent.name} experiment.md", findings
        )
        check_experiment_admission(
            experiment_path.parent.name,
            experiment_path,
            challenge_dir,
            findings,
            enforce_current_focus=False,
        )
    exploration_section = "## Internal Exploration" if latest_baseline else "## Exploration View"
    required_sections = (
        "## Implementation Plan",
        exploration_section,
        "## Visualization And Logging For Human Inspection",
        "## Implementation Acceptance",
    )
    missing_sections = [
        section[3:] for section in required_sections if section not in baseline_text
    ]
    if missing_sections:
        findings.append(
            Finding(
                "WARN",
                f"{baseline_id} baseline plan is incomplete; missing: {', '.join(missing_sections)}",
            )
        )
    else:
        check_baseline_acceptance(
            baseline_text,
            baseline_id,
            findings,
            require_committed_composition=(
                latest_baseline or has_template_schema(baseline_text, BASELINE_SCHEMA_V4)
            ),
        )
    if uses_active_template_contract(baseline_text):
        current_schema = (
            has_template_schema(baseline_text, BASELINE_SCHEMA)
            or has_template_schema(baseline_text, BASELINE_SCHEMA_V4)
            or has_template_schema(baseline_text, BASELINE_SCHEMA_V3)
            or has_template_schema(baseline_text, BASELINE_SCHEMA_V2)
            or baseline_path.parent.name == "baseline_template"
        )
        direction_fields = (
            "selected direction / component composition",
            "repo link / source strategy",
            "compute / GPU efficiency",
        )
        if current_schema:
            direction_fields = (
                "repo link / source strategy",
                "compute / GPU efficiency",
            )
        elif "alternatives and escape criteria" not in baseline_text.lower():
            findings.append(
                Finding(
                    "INFO",
                    f"{baseline_id} baseline uses a legacy direction schema; migrate it before changing direction",
                )
            )
        missing_direction = [
            field
            for field in direction_fields
            if not meaningful_value(read_field(baseline_text, field))
        ]
        if missing_direction:
            findings.append(
                Finding(
                    "WARN",
                    f"{baseline_id} baseline has an incomplete direction plan; missing: {', '.join(missing_direction)}",
                )
            )
        has_decision_result = any(
            clean_markdown_cell(
                read_field(
                    markdown_section(read_text(path), "Outcome"), "result status / limitations"
                )
            )
            .lower()
            .startswith("completed")
            for path in experiment_files
        )
        check_exploration_map(
            baseline_text,
            baseline_id,
            findings,
            require_items=has_decision_result,
        )

    for experiment_path in experiment_files:
        experiment = read_text(experiment_path)
        exp_id = experiment_path.parent.name
        current_experiment_schema = has_template_schema(experiment, EXPERIMENT_SCHEMA)
        outcome = markdown_section(experiment, "Outcome")
        metric_files = list(experiment_path.parent.glob("metrics*/metrics.json")) + list(
            experiment_path.parent.glob("metrics*/per_case*.csv")
        )
        if not outcome:
            if metric_files:
                findings.append(
                    Finding(
                        "WARN",
                        f"{exp_id} has metric files but no Outcome; complete result analysis",
                    )
                )
            continue
        raw_status = clean_markdown_cell(read_field(outcome, "result status / limitations"))
        status_lower = raw_status.lower()
        has_valid_outcome = status_lower == "completed" or status_lower.startswith(
            "completed with limitations"
        )
        if not has_valid_outcome:
            if meaningful_value(raw_status):
                findings.append(
                    Finding(
                        "WARN",
                        f"{exp_id} Outcome is not a valid completed result; keep routine execution state under jobs/ and dashboard.md",
                    )
                )
            else:
                findings.append(
                    Finding(
                        "WARN",
                        f"{exp_id} has an empty Outcome section; omit it until a valid result exists",
                    )
                )
            continue
        required_outcome_fields = {
            "execution / artifact evidence": read_field(outcome, "execution / artifact evidence"),
            "actual model / checkpoint / config": read_field(
                outcome, "actual model / checkpoint / config"
            ),
            "actual training data / split": read_field(outcome, "actual training data / split"),
            "actual evaluation provenance": read_field(outcome, "actual evaluation provenance"),
            "numerical result / comparison": read_field(outcome, "numerical result / comparison"),
            "result viewer / diagnostic evidence": read_field(
                outcome, "result viewer / diagnostic evidence"
            ),
            "interpretation / uncertainty": read_field(outcome, "interpretation / uncertainty"),
            "decision / next action": read_field(outcome, "decision / next action"),
        }
        if current_experiment_schema:
            required_outcome_fields["downstream updates"] = read_field(
                outcome, "downstream updates"
            )
        else:
            required_outcome_fields.update(
                {
                    "Exploration Map impact": read_field(outcome, "Exploration Map impact"),
                    "state.md impact": read_field(outcome, "`state.md` impact")
                    or read_field(outcome, "state.md impact"),
                    "trajectory decision": read_field(outcome, "trajectory decision"),
                }
            )
        for field, value in required_outcome_fields.items():
            if not meaningful_value(value):
                findings.append(Finding("WARN", f"{exp_id} completed Outcome is missing: {field}"))
        trajectory_answer = required_outcome_fields.get(
            "downstream updates", required_outcome_fields.get("trajectory decision", "")
        )
        if (
            meaningful_value(trajectory_answer)
            and "trajectory" in trajectory_answer.lower()
            and not TRAJECTORY_ID_RE.search(trajectory_answer)
            and "no trajectory change" not in trajectory_answer.lower()
        ):
            findings.append(
                Finding(
                    "WARN",
                    f"{exp_id} downstream trajectory update must link T0001 or say no trajectory change",
                )
            )


def check_exploration_map(
    baseline_text: str,
    baseline_id: str,
    findings: list[Finding],
    *,
    require_items: bool = False,
) -> None:
    if has_template_schema(baseline_text, BASELINE_SCHEMA):
        section = markdown_section(baseline_text, "Internal Exploration")
        rows = [
            row
            for row in first_markdown_table_rows(section)
            if meaningful_value(row.get("id", ""))
            or meaningful_value(row.get("possible improvement / question", ""))
        ]
        if require_items and not rows:
            findings.append(
                Finding(
                    "WARN",
                    f"{baseline_id} has a decision-bearing result but no task-specific Internal Exploration item",
                )
            )
        for row in rows:
            item = clean_markdown_cell(row.get("id", ""))
            judgment = clean_markdown_cell(row.get("current judgment", ""))
            evidence = row.get("evidence", "")
            for field, value in (
                ("ID", item),
                ("Possible Improvement / Question", row.get("possible improvement / question", "")),
                ("Why It May Materially Help Now", row.get("why it may materially help now", "")),
                ("Current Judgment", judgment),
            ):
                if not meaningful_value(value):
                    findings.append(
                        Finding(
                            "WARN",
                            f"{baseline_id} Internal Exploration {item or '<unknown>'} is missing: {field}",
                        )
                    )
            if item and not EXPLORATION_ID_RE.fullmatch(item):
                findings.append(
                    Finding(
                        "WARN",
                        f"{baseline_id} Internal Exploration item lacks an E001-style ID: {item}",
                    )
                )
            judgment_lower = judgment.lower()
            if judgment_lower.startswith(
                ("testing", "adopted", "rejected by evidence")
            ) and not has_experiment_evidence_link(evidence):
                findings.append(
                    Finding(
                        "WARN",
                        f"{baseline_id} Internal Exploration {item} needs linked experiment evidence",
                    )
                )
            if judgment_lower.startswith("not worthwhile now:") and not meaningful_value(
                judgment.split(":", 1)[1] if ":" in judgment else ""
            ):
                findings.append(
                    Finding(
                        "WARN", f"{baseline_id} Internal Exploration {item} needs a causal judgment"
                    )
                )
            if judgment_lower.startswith("blocked:") and not verified_hard_blocker(
                judgment, evidence
            ):
                findings.append(
                    Finding(
                        "WARN",
                        f"{baseline_id} Internal Exploration {item} lacks verified blocker evidence",
                    )
                )
        return

    section = markdown_section(baseline_text, "Exploration View") or markdown_section(
        baseline_text, "Exploration Map"
    )
    view = markdown_subsection(section, "### Agent's Exploration View") or section
    view_text = (
        view.removeprefix("### Agent's Exploration View")
        .removeprefix("## Exploration View")
        .removeprefix("## Exploration Map")
        .split("| ID |", 1)[0]
        .strip()
    )
    if not meaningful_value(view_text):
        findings.append(
            Finding("WARN", f"{baseline_id} Exploration View has no agent-defined research agenda")
        )
    rows = first_markdown_table_rows(section)
    if not rows:
        findings.append(Finding("WARN", f"{baseline_id} Exploration View has no items"))
        return
    allowed = (
        "open",
        "planned",
        "active",
        "supported",
        "rejected",
        "deferred",
        "blocked",
        "not applicable",
    )
    for row in rows:
        item = clean_markdown_cell(row.get("id", ""))
        area = row.get("area / question", "")
        status = clean_markdown_cell(row.get("status", ""))
        evidence = row.get("evidence / experiment", "")
        if not EXPLORATION_ID_RE.search(item):
            findings.append(
                Finding(
                    "WARN", f"{baseline_id} Exploration View item lacks an E001-style ID: {item}"
                )
            )
        for field, value in (
            ("Area / Question", area),
            ("Why It Could Matter", row.get("why it could matter", "")),
            ("Priority / Trigger", row.get("priority / trigger", "")),
            ("Status", status),
        ):
            if not meaningful_value(value):
                findings.append(
                    Finding(
                        "WARN",
                        f"{baseline_id} Exploration View {item or '<unknown>'} is missing: {field}",
                    )
                )
        status_lower = status.lower()
        if meaningful_value(status) and not status_lower.startswith(allowed):
            findings.append(
                Finding(
                    "WARN", f"{baseline_id} Exploration View {item} has unknown status: {status}"
                )
            )
        if status_lower.startswith(
            ("planned", "active", "supported", "rejected")
        ) and not meaningful_value(evidence):
            findings.append(
                Finding(
                    "WARN",
                    f"{baseline_id} Exploration View {item} needs evidence or an experiment link for status {status}",
                )
            )
        if status_lower.startswith("blocked") and not verified_hard_blocker(status, evidence):
            findings.append(
                Finding(
                    "WARN",
                    f"{baseline_id} Exploration View {item} lacks verified hard-blocker evidence",
                )
            )
        if (
            status_lower.startswith(("deferred", "not applicable"))
            and ":" not in status
            and not meaningful_value(evidence)
        ):
            findings.append(
                Finding(
                    "WARN",
                    f"{baseline_id} Exploration View {item} needs a reason for status {status}",
                )
            )
    areas = " ".join(clean_markdown_cell(row.get("area / question", "")) for row in rows).lower()
    coverage = {
        "data/domain/external data": r"\bdata\b|domain|external",
        "input/preprocessing/context/patch": r"input|preprocess|context|resolution|patch",
        "sampling/augmentation": r"sampl|augment",
        "architecture/initialization/pretraining": r"architect|initiali|pretrain|foundation",
        "loss/objective": r"\bloss\b|objective",
        "optimizer/LR/schedule/training length/effective batch": r"optimi|\blr\b|schedule|training length|effective batch",
        "inference/postprocessing": r"inference|postprocess",
        "ensemble/robustness": r"ensemble|robust",
        "compute/GPU efficiency": r"compute|\bgpu\b|efficien",
    }
    for label, pattern in coverage.items():
        if not re.search(pattern, areas):
            findings.append(
                Finding("WARN", f"{baseline_id} Exploration View does not disposition: {label}")
            )


def check_final_freeze_readiness(
    challenge_dir: pathlib.Path,
    dashboard_text: str,
    findings: list[Finding],
) -> None:
    latest_dashboard = has_template_schema(dashboard_text, DASHBOARD_SCHEMA)
    current_dashboard = latest_dashboard or has_template_schema(dashboard_text, DASHBOARD_SCHEMA_V5)
    if not current_dashboard and not (
        has_template_schema(dashboard_text, DASHBOARD_SCHEMA_V4)
        or has_template_schema(dashboard_text, DASHBOARD_SCHEMA_V3)
    ):
        return

    readiness = clean_markdown_cell(read_field(dashboard_text, "Final freeze readiness"))
    readiness_lower = readiness.lower()
    run_state = clean_markdown_cell(read_field(dashboard_text, "Run state")).lower()
    run_authorization = clean_markdown_cell(read_field(dashboard_text, "Run authorization")).lower()
    run_is_complete = run_state.startswith(
        ("complete", "completed")
    ) or run_authorization.startswith(("complete", "completed"))
    if run_is_complete and not readiness_lower.startswith("ready:"):
        findings.append(Finding("WARN", "completed run has not passed Final freeze readiness"))
    if not readiness_lower.startswith("ready:"):
        return
    if not TRAJECTORY_ID_RE.search(readiness):
        findings.append(
            Finding("WARN", "Final freeze readiness must cite its T0001-style trajectory decision")
        )
    if current_dashboard:
        canary_match = re.search(r"final-path canary:\s*(.+)$", readiness, re.IGNORECASE)
        if not canary_match:
            findings.append(
                Finding(
                    "WARN", "Final freeze readiness must record the final-path train-only canary"
                )
            )
        else:
            canary = canary_match.group(1)
            if not markdown_links(canary) and not clean_markdown_cell(canary).lower().startswith(
                "not applicable: unchanged validated checkpoint and inference path"
            ):
                findings.append(
                    Finding(
                        "WARN",
                        "Final freeze readiness canary must link train-only evidence or state that the validated checkpoint and inference path are unchanged",
                    )
                )

    state_text = read_text(challenge_dir / "state.md")
    latest_state = has_template_schema(state_text, STATE_SCHEMA)
    prior_living_state = has_template_schema(state_text, STATE_SCHEMA_V7)
    prior_current_state = has_template_schema(state_text, STATE_SCHEMA_V6) or has_template_schema(
        state_text, STATE_SCHEMA_V5
    )
    current_state = latest_state or prior_living_state or prior_current_state
    if not current_state:
        findings.append(Finding("WARN", f"Final freeze readiness requires state.md {STATE_SCHEMA}"))
    stop_verdict_raw = clean_markdown_cell(
        read_field(
            state_text,
            (
                "agent judgment"
                if latest_state
                else (
                    "agent judgment / last checked"
                    if current_state
                    else "loop decision / last checked"
                )
            ),
        )
    )
    stop_verdict = stop_verdict_raw.lower()
    if not stop_verdict.startswith(("ready to freeze", "completed: final evaluation")):
        findings.append(
            Finding(
                "WARN",
                f"Final freeze readiness conflicts with state.md Stop Check: {stop_verdict or 'missing'}",
            )
        )
    stop_authority = (
        stop_verdict_raw if current_state else read_field(state_text, "freeze / stop authority")
    )
    if not TRAJECTORY_ID_RE.search(stop_authority):
        findings.append(
            Finding(
                "WARN",
                "Final freeze readiness requires state.md Stop Check authority with a T0001-style trajectory ID",
            )
        )

    if latest_state or prior_living_state:
        for field in ("internal improvement potential", "external improvement potential"):
            potential_raw = read_field(state_text, field)
            potential = clean_markdown_cell(potential_raw).lower()
            if not (
                potential.startswith("no likely large gain:")
                or (
                    potential.startswith("not applicable:")
                    and valid_current_potential_exemption(potential_raw)
                )
            ):
                findings.append(
                    Finding(
                        "WARN",
                        f"Final freeze readiness requires resolved {field}: {potential or 'missing'}",
                    )
                )
    elif current_state:
        for field in ("internal-loop exit gate", "external-loop exit gate"):
            gate_raw = read_field(state_text, field)
            gate = clean_markdown_cell(gate_raw).lower()
            if not gate.startswith(("passed:", "not required:")):
                findings.append(
                    Finding(
                        "WARN",
                        f"Final freeze readiness requires a passed or target/human-stop exemption for {field}: {gate or 'missing'}",
                    )
                )
            elif (
                latest_state
                and gate.startswith("not required:")
                and not valid_stop_gate_exemption(gate_raw)
            ):
                findings.append(
                    Finding(
                        "WARN",
                        f"Final freeze readiness has an invalid {field} exemption",
                    )
                )

    candidates_text = read_text(challenge_dir / "candidates.md")
    if current_dashboard:
        if not has_template_schema(candidates_text, CANDIDATE_SCHEMA):
            findings.append(
                Finding(
                    "WARN",
                    f"Final freeze readiness requires candidates.md {CANDIDATE_SCHEMA} Candidate Comparison",
                )
            )
        else:
            candidate_rows = [
                row
                for row in first_markdown_table_rows(
                    markdown_section(candidates_text, "Candidate Comparison")
                )
                if meaningful_value(row.get("candidate / direction", ""))
            ]
            external_gate = clean_markdown_cell(
                read_field(state_text, "external-loop exit gate")
            ).lower()
            if not latest_state and external_gate.startswith("passed:"):
                for row in candidate_rows:
                    candidate_id = (
                        first_baseline_id(row.get("candidate / direction", "")) or "<unknown>"
                    )
                    status = clean_markdown_cell(row.get("status / evidence", "")).lower()
                    if status.startswith(("open", "required", "selected", "active")):
                        findings.append(
                            Finding(
                                "WARN",
                                f"Final freeze readiness leaves external candidate {candidate_id} unresolved: {status}",
                            )
                        )

    selected = markdown_section(candidates_text, "Selected Workflow")
    baseline_id = first_baseline_id(read_field(selected, "selected baseline")) or first_baseline_id(
        read_field(dashboard_text, "Active baseline")
    )
    if not baseline_id:
        findings.append(
            Finding("WARN", "Final freeze readiness has no selected baseline to verify")
        )
        return
    if has_template_schema(candidates_text, CANDIDATE_SCHEMA):
        selected_row = next(
            (
                row
                for row in first_markdown_table_rows(
                    markdown_section(candidates_text, "Candidate Comparison")
                )
                if first_baseline_id(row.get("candidate / direction", "")) == baseline_id
            ),
            None,
        )
        selected_status = clean_markdown_cell(
            selected_row.get("status / evidence", "") if selected_row else ""
        ).lower()
        if not selected_status.startswith("supported:"):
            findings.append(
                Finding(
                    "WARN",
                    f"Final freeze readiness requires a serious supported run of the selected candidate {baseline_id}",
                )
            )
        elif not any(
            baseline_id in href.lower() and "experiment.md" in href.lower()
            for href in markdown_links(selected_row.get("status / evidence", ""))
        ):
            findings.append(
                Finding(
                    "WARN",
                    f"Final freeze readiness requires selected-candidate experiment evidence under {baseline_id}",
                )
            )
    baseline_path = challenge_dir / "baselines" / baseline_id / "baseline.md"
    if not baseline_path.is_file():
        findings.append(
            Finding(
                "WARN",
                f"Final freeze readiness cannot find selected baseline artifact: {baseline_id}",
            )
        )
        return

    baseline_text = read_text(baseline_path)
    if has_template_schema(baseline_text, BASELINE_SCHEMA):
        return
    exploration = markdown_section(baseline_text, "Exploration View") or markdown_section(
        baseline_text, "Exploration Map"
    )
    high_priority_rows = [
        row
        for row in first_markdown_table_rows(exploration)
        if re.search(
            r"\bP[01]\b", clean_markdown_cell(row.get("priority / trigger", "")), re.IGNORECASE
        )
    ]
    if not high_priority_rows:
        findings.append(
            Finding(
                "WARN",
                f"Final freeze readiness has no P0/P1 Exploration View items in {baseline_id}",
            )
        )
        return

    latest_baseline = has_template_schema(baseline_text, BASELINE_SCHEMA)
    resolved_statuses = (
        ("supported", "rejected", "blocked", "not applicable")
        if latest_baseline
        else ("supported", "rejected", "deferred", "blocked", "not applicable")
    )
    for row in high_priority_rows:
        item = clean_markdown_cell(row.get("id", "")) or "<unknown>"
        status = clean_markdown_cell(row.get("status", ""))
        status_lower = status.lower()
        evidence = row.get("evidence / experiment", "")
        if not status_lower.startswith(resolved_statuses):
            findings.append(
                Finding(
                    "WARN",
                    f"Final freeze readiness leaves {baseline_id} Exploration View {item} unresolved: {status}",
                )
            )
            continue
        if status_lower.startswith("blocked") and not verified_hard_blocker(status, evidence):
            findings.append(
                Finding(
                    "WARN",
                    f"Final freeze readiness lacks verified hard-blocker evidence for {baseline_id} {item}",
                )
            )
        elif (
            latest_baseline
            and status_lower.startswith(("supported", "rejected"))
            and not has_experiment_evidence_link(evidence)
        ):
            findings.append(
                Finding(
                    "WARN",
                    f"Final freeze readiness lacks linked experiment evidence for {baseline_id} {item}: {status}",
                )
            )
        elif not meaningful_value(evidence) or not markdown_links(evidence):
            findings.append(
                Finding(
                    "WARN",
                    f"Final freeze readiness lacks linked evidence for {baseline_id} {item}: {status}",
                )
            )


def check_baseline_acceptance(
    baseline_text: str,
    baseline_id: str,
    findings: list[Finding],
    *,
    require_committed_composition: bool = False,
) -> None:
    section = markdown_section(baseline_text, "Implementation Acceptance")
    rows = first_markdown_table_rows(section)
    if not rows:
        findings.append(
            Finding("WARN", f"{baseline_id} Implementation Acceptance has no checklist rows")
        )
        return
    committed_gate = "selected candidate's complete committed composition and every defining component implemented; no control-only substitute"
    if require_committed_composition and not any(
        clean_markdown_cell(row.get("gate", "") or row.get("component", "")).lower()
        == committed_gate
        for row in rows
    ):
        findings.append(
            Finding(
                "WARN",
                f"{baseline_id} Implementation Acceptance lacks the complete selected-composition gate",
            )
        )
    for row in rows:
        gate = clean_markdown_cell(row.get("gate", "") or row.get("component", ""))
        status = row.get("status / evidence", "") or row.get("status", "") or row.get("answer", "")
        if require_committed_composition and gate.lower() == committed_gate:
            if not clean_markdown_cell(status).lower().startswith("done:") or not markdown_links(
                status
            ):
                findings.append(
                    Finding(
                        "WARN",
                        f"{baseline_id} complete selected-composition gate requires done: <implementation evidence link>",
                    )
                )
            continue
        if gate and not meaningful_value(status):
            findings.append(
                Finding("WARN", f"{baseline_id} implementation gate has no status/evidence: {gate}")
            )
        elif gate and not (
            gate_is_complete(status) or ("deferred" in gate.lower() and gate_is_deferred(status))
        ):
            findings.append(
                Finding(
                    "WARN",
                    f"{baseline_id} implementation gate is not admission-ready: {gate} ({clean_markdown_cell(status)})",
                )
            )


def check_filled_template_contracts(
    challenge_dir: pathlib.Path,
    findings: list[Finding],
    *,
    discovery_only: bool = False,
) -> None:
    top_level_names = (
        ("challenge.md", "literature.md", "candidates.md")
        if discovery_only
        else (
            "challenge.md",
            "dashboard.md",
            "state.md",
            "trajectory.md",
            "literature.md",
            "candidates.md",
        )
    )
    paths: list[tuple[pathlib.Path, str]] = [
        (challenge_dir / name, f"assets/challenge_template/{name}") for name in top_level_names
    ]
    if discovery_only:
        check_filled_template_pointer_rows(paths, challenge_dir, findings)
        return
    paths.extend(
        (
            challenge_dir / relative_path,
            f"assets/challenge_template/{relative_path}",
        )
        for relative_path in (
            "critics/README.md",
            "shared/visualization/README.md",
            "shared/visualization/visualizer.py",
        )
    )
    paths.extend(
        (path, "assets/challenge_template/baseline_template/baseline.md")
        for path in challenge_dir.glob("baselines/*/baseline.md")
    )
    for pattern, source in (
        (
            "baselines/*/runs/exp*/experiment.md",
            "assets/challenge_template/baseline_template/runs/exp_template/experiment.md",
        ),
        (
            "baselines/*/exp*/experiment.md",
            "assets/challenge_template/baseline_template/runs/exp_template/experiment.md",
        ),
        (
            "baselines/*/runs/exp*/jobs/README.md",
            "assets/challenge_template/baseline_template/runs/exp_template/jobs/README.md",
        ),
        (
            "baselines/*/exp*/jobs/README.md",
            "assets/challenge_template/baseline_template/runs/exp_template/jobs/README.md",
        ),
    ):
        paths.extend((path, source) for path in challenge_dir.glob(pattern))
    check_filled_template_pointer_rows(paths, challenge_dir, findings)


def check_filled_template_pointer_rows(
    paths: list[tuple[pathlib.Path, str]],
    challenge_dir: pathlib.Path,
    findings: list[Finding],
) -> None:
    for path, expected_source in paths:
        if not path.is_file():
            continue
        text = read_text(path)
        label = str(path.relative_to(challenge_dir))
        legacy_source = expected_source.replace(
            "assets/challenge_template/", "challenge/challenge_template/"
        )
        if expected_source not in text and legacy_source not in text:
            findings.append(
                Finding(
                    "WARN",
                    f"{label} lost its miagent-template-source pointer; reload its raw template before the next material update",
                )
            )
        if "miagent-template-contract: active" not in text:
            findings.append(
                Finding(
                    "WARN",
                    f"{label} lost its active template contract; reload its raw template before the next material update",
                )
            )


def check_template_contracts(
    challenge_dir: pathlib.Path,
    findings: list[Finding],
) -> None:
    raw_templates = {
        "challenge.md": "assets/challenge_template/challenge.md",
        "dashboard.md": "assets/challenge_template/dashboard.md",
        "state.md": "assets/challenge_template/state.md",
        "trajectory.md": "assets/challenge_template/trajectory.md",
        "literature.md": "assets/challenge_template/literature.md",
        "candidates.md": "assets/challenge_template/candidates.md",
        "critics/README.md": "assets/challenge_template/critics/README.md",
        "shared/visualization/README.md": "assets/challenge_template/shared/visualization/README.md",
        "shared/visualization/visualizer.py": "assets/challenge_template/shared/visualization/visualizer.py",
        "baseline_template/baseline.md": "assets/challenge_template/baseline_template/baseline.md",
        "baseline_template/runs/exp_template/experiment.md": "assets/challenge_template/baseline_template/runs/exp_template/experiment.md",
        "baseline_template/runs/exp_template/jobs/README.md": "assets/challenge_template/baseline_template/runs/exp_template/jobs/README.md",
    }
    for relative_path, expected_source in raw_templates.items():
        path = challenge_dir / relative_path
        text = read_text(path)
        if expected_source not in text:
            findings.append(
                Finding(
                    "ERROR", f"raw template {relative_path} has no correct miagent-template-source"
                )
            )
        if "miagent-template-contract: active" not in text:
            findings.append(
                Finding("ERROR", f"raw template {relative_path} has no active template contract")
            )
    schema_templates = {
        "dashboard.md": DASHBOARD_SCHEMA,
        "state.md": STATE_SCHEMA,
        "literature.md": LITERATURE_SCHEMA,
        "candidates.md": CANDIDATE_SCHEMA,
        "baseline_template/baseline.md": BASELINE_SCHEMA,
        "baseline_template/runs/exp_template/experiment.md": EXPERIMENT_SCHEMA,
    }
    for relative_path, schema in schema_templates.items():
        path = challenge_dir / relative_path
        if not has_template_schema(read_text(path), schema):
            findings.append(
                Finding("ERROR", f"raw template {relative_path} has no schema marker: {schema}")
            )

    state_text = read_text(challenge_dir / "state.md")
    for section in (
        "## Current Focus",
        "## Continue / Switch / Freeze Judgment",
    ):
        if section not in state_text:
            findings.append(Finding("ERROR", f"state template missing section: {section[3:]}"))
    for field in (
        "focus ID / last changed",
        "stage / active direction",
        "objective now",
        "bottleneck / current diagnosis",
        "internal exploration pointer",
        "external exploration pointer",
        "active human constraint",
        "focus decision / authority",
        "internal improvement potential",
        "external improvement potential",
        "resource effect",
        "agent judgment",
    ):
        if not has_table_field(state_text, field):
            findings.append(Finding("ERROR", f"state template missing field: {field}"))

    trajectory_text = read_text(challenge_dir / "trajectory.md")
    if "## Decision Trace" not in trajectory_text:
        findings.append(Finding("ERROR", "trajectory template missing Decision Trace"))

    literature_text = read_text(challenge_dir / "literature.md")
    for section in (
        "## Independent Scientific Thesis",
        "## Open Research Findings",
        "## Revised Scientific Direction",
        "## Post-Search Coverage Audit",
        "## External Critic Summary",
    ):
        if section not in literature_text:
            findings.append(Finding("ERROR", f"literature template missing section: {section[3:]}"))
    for field in (
        "evidence id",
        "finding / method",
        "decision-relevant evidence",
        "primary source",
        "implementation / weights / data",
        "effect on thesis",
        "coverage area",
        "evidence ids / concrete findings",
        "decision / remaining gap",
    ):
        if not has_table_column(literature_text, field):
            findings.append(Finding("ERROR", f"literature template missing field: {field}"))
    for field in (
        "agent-chosen decomposition",
        "strongest complete pipeline before search",
        "chosen complete pipeline after search",
        "what changed from the thesis",
    ):
        if not has_table_field(literature_text, field):
            findings.append(Finding("ERROR", f"literature template missing field: {field}"))

    candidates_text = read_text(challenge_dir / "candidates.md")
    comparison = markdown_section(candidates_text, "Candidate Comparison")
    for field in (
        "candidate / direction",
        "committed end-to-end composition",
        "why it may win",
        "main uncertainty",
        "next decisive evidence",
        "status / evidence",
    ):
        if not has_table_column(comparison, field):
            findings.append(
                Finding("ERROR", f"candidates template Candidate Comparison missing field: {field}")
            )
    for field in (
        "pre-search thesis direction",
        "pre-search thesis disposition",
        "scientifically strongest direction before feasibility",
        "selected baseline",
        "selection reason",
        "trajectory ID",
        "approval status",
    ):
        if not has_table_field(markdown_section(candidates_text, "Selected Workflow"), field):
            findings.append(
                Finding("ERROR", f"candidates template Selected Workflow missing field: {field}")
            )

    dashboard_text = read_text(challenge_dir / "dashboard.md")
    for field in (
        "Run authorization",
        "Active baseline",
        "Heartbeat",
        "GPU efficiency / evidence",
        "Latest trajectory decision",
        "Final freeze readiness",
    ):
        if not has_table_field(dashboard_text, field):
            findings.append(Finding("ERROR", f"dashboard template missing field: {field}"))

    baseline_text = read_text(challenge_dir / "baseline_template" / "baseline.md")
    for section in (
        "## Implementation Plan",
        "## Internal Exploration",
        "## Visualization And Logging For Human Inspection",
        "## Implementation Acceptance",
    ):
        if section not in baseline_text:
            findings.append(Finding("ERROR", f"baseline template missing section: {section[3:]}"))
    for field in (
        "repo link / source strategy",
        "compute / GPU efficiency",
    ):
        if not has_table_field(baseline_text, field):
            findings.append(Finding("ERROR", f"baseline template missing field: {field}"))
    internal_exploration = markdown_section(baseline_text, "Internal Exploration")
    for field in (
        "id",
        "possible improvement / question",
        "why it may materially help now",
        "current judgment",
        "evidence",
        "revisit trigger",
    ):
        if not has_table_column(internal_exploration, field):
            findings.append(
                Finding("ERROR", f"baseline template Internal Exploration missing field: {field}")
            )

    planned_jobs = markdown_section(dashboard_text, "baseline001_nnunet")
    if "Launch Group" not in markdown_subsection(planned_jobs, "### Planned Jobs"):
        findings.append(Finding("ERROR", "dashboard template Planned Jobs is missing Launch Group"))
    for required_gate in (
        "selected candidate's complete committed composition and every defining component implemented; no control-only substitute",
        "representative GPU profile recorded; bottleneck and selected resource settings justified",
        "quality-affecting compute changes are controlled experiments, or explicitly deferred",
    ):
        if required_gate not in baseline_text:
            findings.append(
                Finding("ERROR", f"baseline template missing implementation gate: {required_gate}")
            )

    experiment_text = read_text(
        challenge_dir / "baseline_template" / "runs" / "exp_template" / "experiment.md"
    )
    for field in (
        "parent / comparator",
        "current focus / exploration item",
        "question / hypothesis",
        "intervention",
        "reasoning / alternatives",
        "distinguishing evidence / falsifier",
        "training / evaluation contract",
        "execution / evidence plan",
        "result status / limitations",
        "execution / artifact evidence",
        "actual model / checkpoint / config",
        "actual training data / split",
        "actual evaluation provenance",
        "numerical result / comparison",
        "result viewer / diagnostic evidence",
        "interpretation / uncertainty",
        "decision / next action",
        "downstream updates",
    ):
        if not has_table_field(experiment_text, field):
            findings.append(Finding("ERROR", f"experiment template missing field: {field}"))
    for section in ("## Intent", "## Human Steering", "## Outcome", "## Diagnosis"):
        if section not in experiment_text:
            findings.append(Finding("ERROR", f"experiment template missing section: {section[3:]}"))


def check_human_steering(path: pathlib.Path, label: str, findings: list[Finding]) -> None:
    text = read_text(path)
    for section_name in ("Human Steering", "Human Queue Steering", "Human Result Steering"):
        section = markdown_section(text, section_name)
        if not section:
            continue
        if "challenge_template" in path.parts:
            check_template_human_steering_shape(section, label, section_name, findings)
        for line_number, human_input in human_steering_inputs(section):
            cleaned = clean_markdown_cell(human_input)
            if not meaningful_value(cleaned):
                continue
            if ":/auto" in cleaned.lower():
                findings.append(
                    Finding(
                        "WARN",
                        f"{label} {section_name} line {line_number} duplicates :/auto authorization; keep it in dashboard Snapshot and the initial trajectory decision",
                    )
                )
            if SUSPICIOUS_HUMAN_STEERING_RE.search(cleaned):
                findings.append(
                    Finding(
                        "WARN",
                        f"{label} {section_name} line {line_number} looks like workflow/agent state, not human input: {shorten(cleaned)}",
                    )
                )


def check_template_human_steering_shape(
    section: str, label: str, section_name: str, findings: list[Finding]
) -> None:
    header = first_markdown_table_header(section)
    if not header:
        return
    required = {"date", "human decision", "reason", "applied action"}
    if required <= set(header):
        return
    findings.append(
        Finding(
            "WARN",
            f"{label} {section_name} table should use Date/Human Decision/Reason/Applied Action",
        )
    )


def markdown_section(markdown: str, section_name: str) -> str:
    header = f"## {section_name}"
    start = markdown.find(header)
    if start == -1:
        return ""
    next_header = re.search(r"\n## ", markdown[start + len(header) :])
    if not next_header:
        return markdown[start:]
    return markdown[start : start + len(header) + next_header.start()]


def count_markdown_table_data_rows(section: str) -> int:
    rows = 0
    seen_header = False
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells or all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if not seen_header:
            seen_header = True
            continue
        rows += 1
    return rows


def first_markdown_table_header(section: str) -> list[str]:
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [clean_markdown_cell(cell).lower() for cell in stripped.strip("|").split("|")]
        if not cells or all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        return cells
    return []


def first_markdown_table_rows(section: str) -> list[dict[str, str]]:
    header: list[str] = []
    rows: list[dict[str, str]] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells or all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if not header:
            header = [clean_markdown_cell(cell).lower() for cell in cells]
            continue
        rows.append(
            {key: cells[index] if index < len(cells) else "" for index, key in enumerate(header)}
        )
    return rows


def human_steering_inputs(section: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    header: list[str] = []
    human_input_index = -1
    for offset, line in enumerate(section.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if not cells or all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            lower_cells = [clean_markdown_cell(cell).lower() for cell in cells]
            candidate_index = first_human_input_column(lower_cells)
            if candidate_index >= 0:
                header = lower_cells
                human_input_index = candidate_index
                continue
            if header and human_input_index >= 0 and human_input_index < len(cells):
                rows.append((offset, cells[human_input_index]))
            continue

        lower = stripped.lower()
        for prefix in ("- human input / decision:", "- human input:", "- human decision:"):
            if lower.startswith(prefix):
                rows.append((offset, stripped.split(":", 1)[1].strip()))
                break
    return rows


def first_human_input_column(cells: list[str]) -> int:
    for index, cell in enumerate(cells):
        if "human input" in cell or "human decision" in cell:
            return index
    return -1


def collect_baseline_files(challenge_dir: pathlib.Path) -> list[pathlib.Path]:
    paths = sorted(challenge_dir.glob("baselines/*/baseline.md"))
    return [path for path in paths if "baseline_template" not in path.parts]


def collect_experiment_files(baseline_dir: pathlib.Path) -> list[pathlib.Path]:
    paths = []
    paths.extend(sorted(baseline_dir.glob("runs/exp*/experiment.md")))
    paths.extend(sorted(baseline_dir.glob("exp*/experiment.md")))  # legacy layout
    return [path for path in paths if "exp_template" not in path.parts]


def first_exp_id(text: str) -> str:
    match = EXP_ID_RE.search(text)
    return match.group(0) if match else ""


def cell_by_header(cells: list[str], header: list[str], wanted: str) -> str:
    if wanted in header:
        index = header.index(wanted)
        if index < len(cells):
            return cells[index].strip()
    return ""


def resolve_dashboard_path(
    challenge_dir: pathlib.Path, baseline: str, value: str
) -> pathlib.Path | None:
    text = markdown_link_target(value)
    if not text or URL_RE.match(text):
        return None
    text = urllib.parse.unquote(text.split("#", 1)[0].split("?", 1)[0])
    if text.startswith("runs/") and baseline:
        return challenge_dir / "baselines" / baseline / text
    if text.startswith("baselines/"):
        return challenge_dir / text
    return challenge_dir / text


def markdown_link_target(value: str) -> str:
    text = value.strip().strip("`")
    match = re.fullmatch(r"\[[^\]]+\]\(([^)]+)\)", text)
    if match:
        return match.group(1).strip()
    return text


def first_choice(value: str) -> str:
    if not value:
        return ""
    return value.strip().split()[0].strip("|,;").lower()


def normalize_status(value: str) -> str:
    if value == "yes":
        return "completed"
    if value == "no":
        return ""
    return value


def clean_markdown_cell(value: str) -> str:
    text = value.strip()
    text = text.replace("`", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def shorten(value: str, limit: int = 140) -> str:
    text = clean_markdown_cell(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def meaningful_value(value: str) -> bool:
    text = value.strip().lower()
    if not text:
        return False
    if text.startswith("example:") or re.search(r"<[^>]+>", text):
        return False
    return text not in {"n/a", "na", "none", "unknown", "todo", "tbd", "-"}


def meaningful_literature_evidence(value: str) -> bool:
    text = clean_markdown_cell(value).lower()
    if not meaningful_value(value) or text in {
        "yes",
        "no",
        "searched",
        "checked",
        "done",
        "complete",
        "completed",
    }:
        return False
    for prefix in ("none found", "blocked", "not applicable"):
        if text == prefix or text.startswith(prefix + " "):
            return False
        if text.startswith(prefix + ":"):
            return meaningful_value(text.removeprefix(prefix + ":").strip())
    return True


def convenience_selection_language(value: str) -> bool:
    text = clean_markdown_cell(value).lower()
    patterns = (
        r"\bfastest\b",
        r"\bstrongest executable\b",
        r"\bimmediately (?:available|reliable|runnable|executable)\b",
        r"\bonly immediately\b",
        r"\b(?:easy|easier|easiest) (?:implementation|path|option|baseline)\b",
        r"\bsetup (?:cost|effort|risk|time)\b",
        r"\bimplementation (?:cost|effort|risk|time)\b",
        r"\bcritical path\b",
        r"\bif (?:setup|integration|installation) is (?:clean|quick|easy)\b",
        r"\bif (?:it|weights|the method) (?:is|are) (?:easy|quick) to run\b",
        r"\bafter (?:the )?(?:easy |generic )?baseline (?:is weak|plateaus|underperforms)\b",
        r"\bbefore serious training\b",
        r"\bunder (?:the )?(?:limited )?(?:wall[- ]clock|time) budget\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def soft_implementation_uncertainty(value: str) -> bool:
    text = clean_markdown_cell(value).lower()
    return any(
        phrase in text
        for phrase in (
            "setup cost",
            "setup risk",
            "setup time",
            "implementation cost",
            "implementation risk",
            "implementation time",
            "interface setup",
            "uncached",
            "not cached",
            "package unavailable",
            "package not installed",
        )
    )


def starts_with_literature_status(value: str) -> bool:
    text = clean_markdown_cell(value).lower()
    return any(
        text == prefix or text.startswith(prefix + ":")
        for prefix in ("none found", "blocked", "not applicable")
    )


def check_current_method_evidence(
    component_rows: list[tuple[str, dict[str, str]]],
    components: list[str],
    findings: list[Finding],
) -> None:
    serious_rows: list[tuple[str, dict[str, str]]] = []
    for component, row in component_rows:
        method = row.get("method / family", "")
        if not named_literature_method(method):
            continue
        serious_rows.append((component, row))
        source = row.get("paper / primary source", "")
        if meaningful_literature_evidence(source) and not has_external_markdown_link(source):
            findings.append(
                Finding(
                    "WARN",
                    f"literature.md {component} method {clean_markdown_cell(method)} must link an opened primary source",
                )
            )
        year_venue = row.get("year / venue", "")
        if meaningful_literature_evidence(year_venue) and not re.search(
            r"\b(?:19|20)\d{2}\b", clean_markdown_cell(year_venue)
        ):
            findings.append(
                Finding(
                    "WARN",
                    f"literature.md {component} method {clean_markdown_cell(method)} is missing a publication or release year",
                )
            )
    for component in components:
        if not any(row_component.lower() == component.lower() for row_component, _ in serious_rows):
            findings.append(
                Finding(
                    "WARN",
                    f"literature.md has no named serious method evidence for component: {component}",
                )
            )

    if not any(
        "control" in clean_markdown_cell(row.get("transfer fit / component decision", "")).lower()
        for _, row in serious_rows
    ):
        findings.append(
            Finding(
                "WARN",
                "literature.md component evidence does not identify a mature control",
            )
        )

    recent_general = [
        row
        for _, row in serious_rows
        if clean_markdown_cell(row.get("search lens", "")).lower()
        == "general cv abstract-task frontier"
        and has_recent_literature_year(row.get("year / venue", ""))
        and has_external_markdown_link(row.get("paper / primary source", ""))
    ]
    recent_foundation = [
        row
        for _, row in serious_rows
        if "foundation / pretraining" in clean_markdown_cell(row.get("search lens", "")).lower()
        and has_recent_literature_year(row.get("year / venue", ""))
        and has_external_markdown_link(row.get("paper / primary source", ""))
    ]
    for label, rows in (
        ("recent general CV/ML frontier", recent_general),
        ("current foundation/pretrained", recent_foundation),
    ):
        method_names = {
            normalize_literature_name(row.get("method / family", ""))
            for row in rows
            if normalize_literature_name(row.get("method / family", ""))
        }
        if len(method_names) < 2:
            findings.append(
                Finding(
                    "WARN",
                    f"literature.md needs at least two materially distinct named {label} methods with recent years and primary-source links",
                )
            )


def normalize_literature_name(value: str) -> str:
    return re.sub(r"\W+", " ", clean_markdown_cell(value).lower()).strip()


def named_literature_method(value: str) -> bool:
    text = clean_markdown_cell(value).lower()
    if not meaningful_literature_evidence(value) or starts_with_literature_status(value):
        return False
    return not any(
        text == phrase or text.startswith(phrase + ":")
        for phrase in (
            "deferred",
            "not checked",
            "not searched",
            "not selected",
            "not used",
            "foundation model",
            "foundation models",
            "pretrained model",
            "pretrained models",
            "general method",
            "general methods",
            "recent method",
            "recent methods",
        )
    )


def has_external_markdown_link(value: str) -> bool:
    return any(URL_RE.match(href.strip().strip("<>")) for href in markdown_links(value))


def component_has_local_skill_grounding(
    initial_analysis: str,
    rows: list[dict[str, str]],
) -> bool:
    return mentions_local_skill(initial_analysis) or any(
        row_mentions_local_skill(row) for row in rows
    )


def check_local_method_inventory(
    anchor_row: Optional[dict[str, str]],
    component_rows: list[tuple[str, dict[str, str]]],
    evidence_ids: set[str],
    findings: list[Finding],
) -> None:
    if anchor_row is None:
        return
    anchor_text = " ".join(anchor_row.values())
    anchor_lower = clean_markdown_cell(anchor_text).lower()
    inventory_ids = set(LITERATURE_EVIDENCE_ID_RE.findall(anchor_text))
    no_applicable_method = "not applicable:" in anchor_lower
    if not inventory_ids and not no_applicable_method:
        findings.append(
            Finding(
                "WARN",
                "literature.md MIAgent-method inventory names no L001-style component evidence",
            )
        )
        return

    local_row_ids = {
        clean_markdown_cell(row.get("evidence id", ""))
        for _, row in component_rows
        if row_mentions_local_skill(row)
        and LITERATURE_EVIDENCE_ID_RE.fullmatch(clean_markdown_cell(row.get("evidence id", "")))
    }
    for evidence_id in sorted(inventory_ids - evidence_ids):
        findings.append(
            Finding(
                "WARN",
                f"literature.md MIAgent-method inventory references unknown Evidence ID {evidence_id}",
            )
        )
    for evidence_id in sorted(inventory_ids - local_row_ids):
        findings.append(
            Finding(
                "WARN",
                f"literature.md MIAgent-method inventory {evidence_id} is not represented by a skill-linked component evidence row",
            )
        )

    soft_rejection_terms = (
        "not installed",
        "missing package",
        "missing dependency",
        "uncached",
        "not cached",
        "setup work",
        "setup cost",
    )
    for component, row in component_rows:
        if not row_mentions_local_skill(row):
            continue
        decision = clean_markdown_cell(row.get("transfer fit / component decision", "")).lower()
        if decision.startswith("reject:") and any(
            term in decision for term in soft_rejection_terms
        ):
            evidence_id = clean_markdown_cell(row.get("evidence id", "")) or component
            findings.append(
                Finding(
                    "WARN",
                    f"literature.md {evidence_id} rejects a repository method for setup/cache convenience rather than scientific incompatibility",
                )
            )


def row_mentions_local_skill(row: dict[str, str]) -> bool:
    return mentions_local_skill(" ".join(row.values()))


def row_mentions_vision_foundation_skill(row: dict[str, str]) -> bool:
    return mentions_vision_foundation_skill(" ".join(row.values()))


def mentions_local_skill(value: str) -> bool:
    return bool(
        re.search(r"(?:^|[^A-Za-z0-9_-])(?:\.\./)*skills/[^)\s|]+", clean_markdown_cell(value))
    )


def mentions_vision_foundation_skill(value: str) -> bool:
    return bool(
        re.search(
            r"(?:^|[^A-Za-z0-9_-])(?:\.\./)*skills/vision-foundation/[^)\s|]+",
            clean_markdown_cell(value),
        )
    )


def has_recent_literature_year(value: str) -> bool:
    years = [int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", clean_markdown_cell(value))]
    return any(year >= dt.date.today().year - 3 for year in years)


def weak_frontier_search_row(row: dict[str, str]) -> bool:
    method = clean_markdown_cell(row.get("method / family", "")).lower()
    finding = clean_markdown_cell(row.get("concrete primary evidence", "")).lower()
    evidence = clean_markdown_cell(row.get("paper / primary source", "")).lower()
    inspected_current_source = has_external_markdown_link(row.get("paper / primary source", ""))
    weak_patterns = (
        "not checked",
        "not searched",
        "no source inspected",
        "no sources inspected",
        "no checkpoint source inspected",
        "no cached weight",
        "container probe",
        "dependency missing",
        "dependencies missing",
        "not installed",
        "time budget favors",
        "hot path",
        "before first run",
        "if the baseline plateaus",
        "if baseline plateaus",
    )
    if (
        method.startswith("not applicable:")
        and inspected_current_source
        and meaningful_literature_evidence(row.get("concrete primary evidence", ""))
    ):
        return False
    if not named_literature_method(method):
        return True
    combined = f"{method} {finding} {evidence}"
    if not inspected_current_source:
        return True
    return any(pattern in combined for pattern in weak_patterns)


def actionable_pretraining_row(row: dict[str, str]) -> bool:
    implementation = clean_markdown_cell(
        row.get("official implementation + weights / pretraining / license", "")
    ).lower()
    decision = clean_markdown_cell(row.get("transfer fit / component decision", "")).lower()
    interface_pattern = r"backbone|encoder|decoder|head|adapter|fine[- ]?tun|feature|initiali|prompt|reject|incompatib"
    return (
        has_external_markdown_link(
            row.get("official implementation + weights / pretraining / license", "")
        )
        and "weights:" in implementation
        and "pretraining data:" in implementation
        and bool(re.search(interface_pattern, decision))
    )


def gate_is_complete(value: str) -> bool:
    status = clean_markdown_cell(value).lower()
    prefixes = ("done", "verified", "complete", "completed", "passed")
    for prefix in prefixes:
        if status.startswith(prefix):
            evidence = status[len(prefix) :].lstrip(" :,-;")
            return meaningful_value(evidence)
    if status.startswith("not applicable:"):
        return meaningful_value(status.removeprefix("not applicable:").strip())
    return False


def gate_is_deferred(value: str) -> bool:
    status = clean_markdown_cell(value).lower()
    return status.startswith("deferred:") and meaningful_value(
        status.removeprefix("deferred:").strip()
    )


def valid_stop_gate_exemption(value: str) -> bool:
    status = clean_markdown_cell(value).lower()
    if not status.startswith("not required:"):
        return False
    if "human stop" in status:
        return bool(TRAJECTORY_ID_RE.search(value) and markdown_links(value))
    if "explicit terminal target met" in status:
        return bool(markdown_links(value))
    return False


def read_text(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def uses_active_template_contract(markdown: str) -> bool:
    return "miagent-template-contract: active" in markdown


def has_template_schema(markdown: str, schema: str) -> bool:
    return f"miagent-template-schema: {schema}" in markdown


def read_field(markdown: str, field_name: str) -> str:
    wanted = field_name.strip().lower()
    for line in markdown.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        if cells[0].strip().lower() == wanted:
            return cells[1].strip()
    return ""


def has_table_field(markdown: str, field_name: str) -> bool:
    wanted = clean_markdown_cell(field_name).lower()
    for line in markdown.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and clean_markdown_cell(cells[0]).lower() == wanted:
            return True
    return False


def has_table_column(markdown: str, column_name: str) -> bool:
    wanted = clean_markdown_cell(column_name).lower()
    for line in markdown.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if any(clean_markdown_cell(cell).lower() == wanted for cell in cells):
            return True
    return False


if __name__ == "__main__":
    main()
