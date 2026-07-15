---
name: tao-run-automl-deft-pipeline
description: >
  Run the canonical NVIDIA AOI three-phase training pipeline — Phase 1 AutoML baseline (HPO),
  Phase 2 DEFT loop (RCA → SDG → mining → plain-train retrain), Phase 3 AutoML refinement on
  the DEFT-augmented dataset. Use when the user asks to "run the AOI workflow",
  "fine-tune my PCB AOI model end-to-end", "improve my AOI ChangeNet model", or "AOI workflow
  with AutoML" request — route here instead of tao-run-deft-aoi directly unless the user
  explicitly asks for the DEFT loop ONLY (e.g. "run JUST the DEFT loop", "skip AutoML, only
  DEFT"). Also handles the same three-phase pattern for non-AOI DEFT applications — AutoML
  baseline then DEFT loop warm-started from AutoML's winning HPs then post-DEFT AutoML
  refinement on the iteration-augmented dataset. Trigger phrases include "run the AOI
  workflow", "AOI end-to-end", "AutoML + DEFT", "AutoML then DEFT", "tune hyperparameters then
  DEFT", "DEFT with AutoML at both ends", "warm-start DEFT", "improve my AOI model".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit. Workflows (tao-run-automl, tao-run-deft-aoi) declare additional requirements.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Skill Bash Write
tags:
- tao
- applications
---

# AutoML + DEFT Pipeline

A workflow-bridge skill that runs **three phases** in sequence by delegating to two existing skills — `tao-run-automl` for HPO and a DEFT application skill (default `tao-run-deft-aoi` for AOI; other `skills/applications/deft-*` skills for non-AOI cases) for the iterative data-improvement loop.

This skill **does not** re-implement AutoML or DEFT. It owns only the connective tissue: HPO spec inputs, the spec-handoff between AutoML and DEFT, and the post-DEFT AutoML re-run on the augmented dataset.

## Routing policy

- User asks to "run the AOI workflow" or "improve my AOI ChangeNet model" — **default to this skill**, not `tao-run-deft-aoi` directly. The bare DEFT loop is the inner stage of this pipeline.
- User wants AutoML and DEFT chained on the same model/dataset
- User says "AutoML at both ends", "tune HPs then DEFT", "warm-start DEFT", "AutoML before and after DEFT"
- User has an AutoML-tuned spec and asks how to feed it into DEFT

## When this skill does NOT apply

- User explicitly asks for the DEFT loop only ("run JUST the DEFT loop", "skip AutoML") → use `tao-run-deft-aoi` directly
- User wants only AutoML with no follow-on DEFT → use `tao-run-automl` directly
- User is doing zero-shot eval, RAG, or non-training workflows

---

## The mental model

```
Phase 1 (AutoML baseline)        Phase 2 (DEFT loop, plain train)        Phase 3 (AutoML refinement)
─────────────────────────        ────────────────────────────────        ───────────────────────────
specs/baseline_spec.yaml         (Phase 1 winner pre-seeds baseline      ${RESULTS_DIR}/iter${N}/dataset/
train/base/training_set.csv       — DEFT skips its baseline train)       train_combined_iter${N}.csv
        │                                       │                                       │
        ▼                                       ▼                                       ▼
[ AutoML HPO sweep ]               [ DEFT: baseline-inference → RCA       [ AutoML HPO sweep ]
   N recommendations                 → iter 1..N (plain retrain) ]        re-tunes HPs against the
   pick best by val_loss / FAR      RCA / route / SDG / mining             DEFT-augmented dataset
        │                                       │                                       │
        ▼                                       ▼                                       ▼
best HPs spec + ckpt ─────►      DEFT-augmented CSV ───────────►        final best checkpoint
                                 + iter winner checkpoint               (the deliverable; no
                                 (Phase 3 warm-starts from it)           further retrain)
```

The two handoffs are:

- **Phase 1 → Phase 2**: a *spec file* AND the *winning checkpoint* — the bridge deep-merges Phase 1's HPs onto `specs/baseline_spec.yaml`, copies the checkpoint into `${RESULTS_DIR}/baseline/train/`, and pre-populates `deft_state.json` / `loop_log.jsonl` so DEFT skips its baseline train and resumes at baseline inference → evaluate → RCA → iter 1. DEFT stays plain-train (`automl_policy: off` preserved).
- **Phase 2 → Phase 3**: a *training CSV* (`train_combined_iter${N_final}.csv`) AND the *iter winner's checkpoint* — the checkpoint is wired into each rec's `train.pretrained_model_path` so Phase 3 fine-tunes from Phase 2's winner. Phase 3's winning checkpoint is the deliverable; no separate retrain after Phase 3.

See `references/phase-handoffs.md` for the exact steps, code, and DEFT-honors-this-handoff details of both handoffs.

## Why three phases instead of two

- **Phase 1 alone** finds good HPs on the *original* training distribution, but the model still has the distributional gaps DEFT is designed to fill.
- **Phase 2 alone** (just DEFT) fills the gaps but uses whatever HPs `specs/baseline_spec.yaml` was hand-authored with — usually not optimal.
- **Phase 3 alone** would run AutoML against the augmented dataset, but without a tuned baseline the DEFT loop's iteration cost is higher (slower convergence, more iterations to hit the KPI).

Running all three: AutoML cheap-tunes once on the original data, DEFT does the heavy data work with reasonable HPs, then AutoML tunes again on the now-richer dataset. Phase 3 is the most important of the three for the final deployed FAR/recall.

## Cost up-front

The pipeline is sequential. Total wall-clock ≈ Phase 1 (N_automl × per-rec train) + Phase 2 (M iterations × per-iter cost) + Phase 3 (N_automl × per-rec train).

Note that **Phase 2 has no separate baseline train** — Phase 1's winning checkpoint is reused as DEFT's baseline, so the baseline cost lands inside Phase 1's N_automl trainings rather than as an extra retrain. Surface this to the user before kickoff. Typically Phase 2's iterations still dominate (each includes SDG + retrain), but Phase 1 and Phase 3 each add several hours on a single-GPU box. Use the per-job estimate from the user's setup (if they have one) rather than guessing minutes. See `references/pitfalls-and-quality-checks.md` (**Compute budget**) for the per-phase term breakdown.

---

## Consolidated Pre-Flight — one gate, all three phases

**The pipeline has exactly one user gate.** Before any side-effecting action (docker pull, docker login, any job-launch call delegated to a downstream skill, file mutations under `${RESULTS_DIR}/`), the agent must produce a single consolidated Pre-Flight Summary that subsumes every downstream skill's preflight. Once the user approves, the run is autonomous through all three phases — no further interactive pauses.

The user explicitly does not want to be paged between phases. The DEFT loop's own inline `## Pre-Flight Summary` gate becomes a **zero-question display step** (every value pre-supplied from this consolidated gate), as does `tao-run-automl`'s shared launch preflight in Phase 1 and Phase 3.

Before printing the summary, the agent must open and read every downstream skill's preflight section in full, run every read-only check those sections prescribe, and surface the *outcome* of each check. The summary has nine mandatory sections (workspace/host/platform/network; credentials status; container images; dataset table; Phase 1 config; Phase 2 config; Phase 3 config; compute estimate; confirmation line). After the gate, every downstream interactive gate is suppressed by passing through the collected values. The only allowed post-gate pauses are mid-run hard-stop safety gates the downstream skill cannot bypass.

See `references/consolidated-preflight.md` for: the full list of preflight sections to read, the required DEFT `## Pre-Flight` run, the exact nine-section summary contents, the value pass-through for gate suppression, and the procedure when the skill bank version doesn't yet support gate suppression.

---

## Phase 1 — AutoML baseline

Invoke `tao-skill-bank:tao-run-automl` with:

| Input | AOI default | Notes |
|---|---|---|
| `network_arch` | `visual-changenet` | Same model the DEFT loop expects |
| `train_dataset_uri` | `<workspace>/train/base/training_set.csv` | Same training set DEFT will start from |
| `eval_dataset_uri` | `<workspace>/train/base/validation_set.csv` | Held-out — must NOT be the KPI test set (`<workspace>/kpi/testing_set.csv`), since that set is reserved for DEFT's final reporting |
| `metric` | FAR @ 100% recall (preferred) or `val_loss` | See **Metric pitfalls** in `references/pitfalls-and-quality-checks.md` — ChangeNet AOI is class-imbalanced, val_loss alone can mode-collapse |
| `algorithm` | `bayesian` | LLM-brain or `autoresearch` if compute is tight |
| `automl_max_recommendations` | 5–10 for AOI | More recs = better HPs but linear in compute |
| `spec_overrides` | Pin epochs / batch_size; sweep optimizer-related HPs only | Otherwise AutoML wanders into long-train regimes that blow Phase 2's budget |

After the sweep finishes, AutoML's `result["best"]["specs"]` is the winning hyperparameter dict.

### Handoff to Phase 2

Phase 1 hands over **two artifacts**: the winning *spec* and the winning *checkpoint*. Retraining the same HPs in DEFT's baseline step is wasted compute — instead, pre-seed DEFT's baseline state from Phase 1's outputs so DEFT starts at baseline inference → evaluate → RCA → iter 1. This is a four-step bridge (write merged spec → pre-seed `baseline/train/` → initialise `deft_state.json` with baseline already done → invoke DEFT), followed by a quality check of the winning checkpoint (per-class prediction counts; compare to zero-shot ChangeNet).

See `references/phase-handoffs.md` for the verbatim Steps 1–4 (including the `cp` command, the `deft_state.json` patch code, and the `loop_log.jsonl` append) and the quality-check checklist.

---

## Phase 2 — DEFT loop (plain training, baseline pre-seeded from Phase 1)

Invoke `tao-skill-bank:tao-run-deft-aoi` (read its `SKILL.md` for the full interface). For non-AOI applications, invoke the matching DEFT skill; the handoff shape is the same.

**The DEFT loop's baseline-train sub-step is skipped.** Phase 1 already produced a checkpoint trained at the winning HPs, and Phase 1's handoff (see `references/phase-handoffs.md`) pre-populated `${RESULTS_DIR}/baseline/train/` and `${RESULTS_DIR}/deft_state.json` so DEFT resumes at baseline inference → evaluate → RCA → iter 1. The rest of the DEFT loop runs unchanged. **Do not modify its `automl_policy: off` invariant.**

The DEFT loop owns: its Pre-Flight Summary display step (**not** a fresh user gate — the Consolidated Pre-Flight above is the single gate; the DEFT summary still prints as an audit-trail display of the pre-seeded `baseline/train/` source and must not re-prompt); baseline inference → evaluate → RCA on the pre-seeded checkpoint; the full per-iteration RCA → routing → SDG → mining → assemble → train cycle; KPI gating and stop conditions; and the `${RESULTS_DIR}/` layout (`deft_state.json`, `loop_log.jsonl`, `DEFT_Loop_Report.html`).

After the loop exits (KPI met or `max_iterations` reached), capture two values from `deft_state.json`: `iterations.<best>.best_ckpt_path` (the loop's best plain-train checkpoint) and the final iteration label `N_final` (used to locate the augmented training CSV).

If the DEFT loop hard-stops on an unrecoverable gate, **skip Phase 3**. There is no validated augmented CSV to feed AutoML.

---

## Phase 3 — AutoML refinement on the DEFT-augmented dataset

Re-invoke `tao-skill-bank:tao-run-automl` with the augmented training CSV as the train dataset, the same held-out validation CSV as before, and **Phase 2's iter winner checkpoint as the warm-start**:

| Input | AOI value |
|---|---|
| `network_arch` | `visual-changenet` |
| `train_dataset_uri` | `${RESULTS_DIR}/iter${N_final}/dataset/train_combined_iter${N_final}.csv` |
| `eval_dataset_uri` | Same as Phase 1 (`<workspace>/train/base/validation_set.csv`) — keep the comparison apples-to-apples |
| `metric` | Same metric as Phase 1 |
| `algorithm` | Same as Phase 1 |
| `automl_max_recommendations` | 5–10 |
| Initial spec | Start from `<workspace>/specs/baseline_spec_automl.yaml` (Phase 1's winner) — gives the sweep a strong centroid to refine around |
| **Warm-start checkpoint** | **`iterations.<best>.best_ckpt_path` from `${RESULTS_DIR}/deft_state.json`** — set `spec_overrides["train"]["pretrained_model_path"]` to this path. Each Phase 3 rec then **fine-tunes from Phase 2's winner** instead of training from scratch. |

The warm-start is mandatory: without it every rec starts from random init with only 10-20 epochs to reconverge, `val_loss` regresses by 0.03-0.05 vs iter1, and the `_pick_best` safety net silently rolls back to the iter winner. Output goes to `${RESULTS_DIR}/final_automl/`; the winning checkpoint of this sweep is the pipeline's deliverable. After the sweep, register Phase 3's checkpoint under `iterations.final_automl` in `deft_state.json` and re-run `prepare_inference_spec.py` so the handoff sees it (falling back to the loop's best if Phase 3 regressed).

See `references/phase-handoffs.md` for: the full "why the warm-start is mandatory" rationale and tradeoff, the concrete `spec_overrides` selection code, the exact two-step wiring of Phase 3's output back into the DEFT report, and the safety note on regression.

---

## Pitfalls and quality checks

These apply to both AutoML phases. Bake them into agent behavior — don't just paste once. The full detail lives in `references/pitfalls-and-quality-checks.md`; in brief:

- **Metric pitfalls — AOI is class-imbalanced.** ChangeNet AOI datasets are PASS-dominant (90%+), so a `val_loss` winner can be a mode-collapsed model. Prefer FAR @ 100%-recall directly, or guard val_loss with a `pred_counts` sanity check, or eval top-K by FAR @ 100%-recall before picking. For balanced / regression tasks, val_loss is fine.
- **Run-to-run noise.** AutoML can show 2–3× variance for the same HP config. If the winner is suspiciously better than the runner-up, re-run with a fresh seed before committing the spec to Phase 2.
- **Cleanliness (data leakage).** Both AutoML phases use a validation set distinct from the KPI test set (`<workspace>/kpi/testing_set.csv`), which is reserved for DEFT's final reporting. Phase 3 trains on the augmented CSV but keeps the same validation set so Phase 1 and Phase 3 numbers stay comparable.
- **Compute budget.** Phase 1 `N_automl × per-rec train`; Phase 2 `M_iter × (RCA + SDG + mining + retrain)` (usually largest); Phase 3 `N_automl × per-rec train` on the larger augmented dataset. Ask the user for their per-job time before quoting wall-clock.

---

## Quick Start (AOI worked example)

When starting fresh from "run the AOI workflow", the agent presents a three-phase plan to the user (Phase 1 AutoML baseline → Phase 2 DEFT loop → Phase 3 AutoML refinement), states the total cost structure (no extra baseline retrain at the front, no extra retrain at the end), asks for the user's per-run time for a wall-clock estimate, and waits for approval. After confirmation it invokes Phase 1, writes the merged spec, pre-seeds `deft_state.json`, invokes the DEFT loop with every input pre-supplied, then invokes Phase 3 — with no further pauses unless a downstream skill hits an unrecoverable hard-stop. It summarizes the trajectory at the end (baseline AutoML best → DEFT iter 1 → ... → DEFT iter N_final → Phase 3 best).

See `references/quick-start-example.md` for the verbatim customer-facing message block and the exact post-confirmation invocation sequence.

## Non-AOI DEFT applications

Same three-phase pattern applies to other DEFT skills. Swap:

- `network_arch` to the relevant model
- The DEFT skill invoked in Phase 2
- The "best HP spec file" and "best HP checkpoint" path conventions to whatever the target DEFT skill expects
- The augmented-CSV path in Phase 3 to whatever the target DEFT skill produces

The handoff shape — Phase 1 emits a *spec + checkpoint* (the checkpoint pre-seeds the DEFT baseline), Phase 2 consumes both and emits an augmented dataset, Phase 3 emits the final checkpoint — is identical. The Phase 1 → Phase 2 baseline-skip mechanism is generic: any DEFT-style loop that exposes a resumable baseline state can be seeded the same way.

---

## See also

- `tao-skill-bank:tao-run-automl` — AutoML interface, algorithms, HP ranges
- `tao-skill-bank:tao-run-deft-aoi` — full DEFT AOI loop (Phase 2 default)
- `tao-skill-bank:tao-train-visual-changenet` — underlying ChangeNet train/eval/infer skill (used by both AutoML and DEFT)
- Other `skills/applications/deft-*` skills — non-AOI Phase 2 targets
- `references/consolidated-preflight.md` — the single-gate preflight in full
- `references/phase-handoffs.md` — both handoffs, baseline pre-seed, and Phase 3 warm-start, verbatim
- `references/pitfalls-and-quality-checks.md` — metric pitfalls, run-to-run noise, leakage, compute budget
- `references/quick-start-example.md` — the customer-facing worked-example message
