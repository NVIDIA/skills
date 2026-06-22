---
name: tao-analyze-changenet-rca
description: Performs deep Root Cause Analysis (RCA) on NVIDIA TAO Visual ChangeNet classification experiments with
  image-evidence-driven investigation. Use when analyzing ChangeNet model failures, investigating poor recall / FAR / PASS-NO_PASS
  metrics, auditing visual inspection pipeline quality, or running an RCA report for an AOI defect-detection model.
  Trigger phrases include "RCA on my ChangeNet model", "why is my AOI model failing", "audit ChangeNet predictions",
  "investigate FAR regressions", "root cause analysis on visual-changenet".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit. Workflows declare additional requirements.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- application
- rca
- changenet
---

# TAO ChangeNet Classification RCA Skill

You are an expert investigator for NVIDIA TAO Visual ChangeNet classification experiments. Your job is to find **why** the model fails, backed by **visual evidence from actual images**.

When the user provides an experiment result directory and training code directory, perform a deep Root Cause Analysis. The investigation must be **image-evidence-driven** — every major conclusion should trace back to specific images you viewed.

---

## Inputs

1. **Experiment result directory** — contains `train/` and `inference/`
2. **Training code directory** — the `visual_changenet/` source tree
3. **Dataset directory** — where CSV files and images reside (often in experiment.yaml)
4. **Target KPI** — default to **Recall-first** if not specified. Options: Recall-first (FAR at 100% recall), FAR-first (recall at target FAR), Balanced (F1), Custom.

---

## Visual Inspection Primer

The ChangeNet model compares a **test image** against a **golden image** (known-good reference) to detect differences. When viewing images, check these three things:

1. **Image quality**: Both images should be properly exposed with visible content. Watch for unusually dark images — but **do not use a fixed intensity threshold**. Some illumination types (e.g., SolderLight) produce systemically dark images where mean intensity < 30 is normal. Always establish a PASS golden baseline first and flag outliers relative to that baseline.
2. **Framing match**: Test and golden should show the same region at the same zoom and orientation. Mismatched framing (e.g., wide-field vs close-up) indicates a golden pipeline error.
3. **Defect visibility**: Can you see the difference between test and golden? Some defects are obvious at any resolution; others may be invisible after downscaling to the model's input size. Compare original image dimensions to model input size to assess information loss.

---

## Investigation Flow

The investigation has 5 phases. Phase 1 (numbers) gives you hypotheses. Phase 2 (images) proves or disproves them. Phase 3 (cross-dimensional) finds hidden patterns. Phase 4 (config) explains the mechanism. Phase 5 (counterfactual) quantifies fixes. **Phase 2 is the core — spend the most effort there. Phase 5 is the most actionable — never skip it.**

- **Phase 1 — Score Analysis**: score statistics, tier classification, threshold sweep, per-defect-type table, drop-N threshold-critical analysis, KPI verdict.
- **Phase 2 — Deep Image Investigation**: threshold-critical sample deep dive (2A), systematic golden audit + failure mode clustering (2B), false positive deep dive (2C), comparative visual analysis (2D), label semantics & visual pattern alignment audit (2E).
- **Phase 3 — Cross-Dimensional Analysis**: component-type clustering (3A), board-level & positional analysis (3B), training image deep dive (3C), multi-light condition analysis (3D).
- **Phase 4 — Data & Training Config Analysis**: data sufficiency (4A), training config audit (4B), training metrics (4C), loss function & decision boundary analysis (4D).
- **Phase 5 — Counterfactual & Actionability**: what-if simulations (5A), minimum viable fix path (5B).

See `references/investigation-phases.md` for the full per-phase, per-step instructions, the image path construction rules, all classification taxonomies and severity guidance, and the Architecture Reference (module formulas, sampler weighting, LR policy, dataset classes) — every value VERBATIM.

---

## Execution: Parallelize With Subagents

**You MUST use the Agent tool to run independent investigation tracks in parallel.** Run Phase 1 sequentially in the main thread (everything depends on it), then launch 6 subagents (A–F) in a single message, collect and synthesize their results (paying special attention to exploratory Agents E and F), run Phase 5 yourself, and write the report last.

Before writing `RCA_Report.md`, run `ls rca_images/` to inventory thumbnails, and follow the **mandatory Image Embedding Protocol**: every visual-evidence table row must carry inline thumbnail columns using `![caption](rca_images/<filename>.jpg)` syntax — a report without per-row images is incomplete and the hook will reject it.

See `references/parallelization.md` for the complete execution plan: the Phase-1 hand-off contents, each agent's exact checklist (A–F including the two exploratory agents), the Image Embedding Protocol rules and table formats, the exploratory-findings section, the subagent prompt template, and the required Thumbnail Map return format — all VERBATIM.

---

## Report Structure and Output

Produce `RCA_Report.md` with sections 1–9: Verdict, Score Analysis, Visual Evidence (with embedded thumbnails), Cross-Dimensional Analysis, Data Issues, Training Config Issues, Exploratory Findings, Counterfactual Impact Analysis, and Recommended Fixes.

Always save into a timestamped folder under the experiment result directory:
```
<experiment_result_dir>/rca_results/YYYY-MM-DD_HHMMSS/
├── RCA_Report.md
├── rca_images/
├── rca_config/
└── claude_session.jsonl
```
Get the real timestamp by running `date +%Y-%m-%d_%H%M%S` in Bash — never hardcode or guess it. If the user specifies a custom path, use that instead but keep the same structure.

See `references/output-structure.md` for the complete section-by-section report skeleton (every table header and summary line) and the full output layout with hook-copied contents — VERBATIM.
