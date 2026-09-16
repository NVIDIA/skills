---
name: tao-run-deft-aoi-cosmos3
description: >
  Run the disk-backed DEFT AOI improvement loop for NVIDIA Cosmos Reason 3 /
  Cosmos3 models, using Nano by default and Edge or Super when explicitly
  requested: evaluate the base model on Proxy and frozen Benchmark splits,
  mine real image pairs from Proxy gaps, assemble a per-iteration Train JSON
  from selected Mining samples, train with cosmos-rl LoRA SFT, and repeat
  through the selected platform's submit/status/logs/cancel contract.
  This migration supports bare labels only: the assistant response must be
  exactly OK or NG. Use for "run Cosmos3 DEFT AOI", "CR3 AOI loop", or
  "improve Cosmos3 PCB inspection with bare OK/NG"; do not use for
  rich/reasoning annotation, one-off Cosmos training, or generic anomaly
  generation.
license: Apache-2.0 AND CC-BY-4.0
compatibility: Requires the companion TAO skill-bank skills from `eval.config`, host Python with `pyarrow` and `yaml`, and the selected platform's native CLI.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Task Bash Write
tags:
- application
- workflow
- deft
- aoi
- cosmos
---

# Skill: tao-run-deft-aoi-cosmos3

## Installation

Install this application as part of the full TAO skill-bank root, not as only
the companion skill folders: `TAO_SKILL_BANK_PATH` must point at a directory
containing `versions.yaml`, `scripts/resolve_versions_key.py`, and the
Cosmos model resolver `scripts/resolve_tao_image.py`, plus the
`skills/{applications,models,data,platform,core}/...` tree listed in
`eval.config`. Run bundled validation with the skill Python so dependencies
match runtime: `PYTHON=$(bash scripts/deft_python.sh); "$PYTHON" -m unittest
tests.test_cosmos3_bare`. Resolve network mode first. Missing air-gap imports
are a hard stop; network-enabled setup lives only in
`references/network-bootstrap.md`.

## Execution Contract

Treat a run as a disk-backed state machine.

1. Preserve every explicit user value and show the source of each effective
   value (`user`, `spec`, or `default`) in the Pre-Flight Summary.
2. Ask which installed platform to use. Do not select Docker, SLURM,
   Kubernetes, Brev, virtualenv, or an external platform by default.
3. Resolve network mode, then read exactly one of `references/air-gap.md` or
   `references/network-bootstrap.md`. Run the selected platform skill's
   Preflight and stop on a missing system/native-CLI prerequisite.
4. Before any mutation or launch, invoke `tao-launch-workflow` and show its
   launch review plus this skill's Pre-Flight Summary. Wait for one explicit
   approval.
5. After approval, set `PYTHON=$(bash scripts/deft_python.sh)` and initialize
   `${RESULTS_DIR}/deft_state.json` once with
   `"$PYTHON" scripts/init_deft_state.py`. Pass the exact GPU model reported by the
   selected platform's Preflight through `--gpu-model` (include accelerator
   memory when available), plus the resolved network mode/source and selected
   absolute Python. Never reinitialize a resumed run or edit
   `deft_state.json` by hand.
6. Before every stage, after context compaction, and before a completion claim,
   run `"$PYTHON" scripts/deft_context.py --state ... --stage ...`. Use its durable
   `next_stage` and the state file's `status`,
   `current_iteration`, `iterations.*.status`, `stage_completed`, and latest
   `events` entry to resume. Do not infer progress from assistant prose or
   from an artifact that is not recorded in state.
7. Run every command that can install, fetch, log in, or launch a local
   container through `"$PYTHON" scripts/deft_exec.py --state ... -- <command>`. In an
   air-gap it rejects egress/package operations and enforces no-pull. Remote
   platforms must apply the equivalent immutable no-pull/offline policy.
8. Submit each GPU stage through the chosen platform's four verbs:
   `submit` / `status` / `logs` / `cancel`. The `submit` verb must open the
   job-record before native launch; the returned id is the only launch handle.
   Poll the backend, not the job-record, and map state to
   `PENDING RUNNING COMPLETE ERROR CANCELED UNKNOWN`.
9. Commit every completed or failed DEFT stage with
   `"$PYTHON" scripts/commit_stage.py`. It verifies the stage inputs and atomically
   updates both the resume snapshot and ordered `events` array in state. Every
   executed-stage commit requires a positive, measured `--duration-sec`: use
   backend elapsed wall time for submitted jobs and a host wall-clock timer for
   inline stages. A documented `--skip` may record `0`; negative durations are
   always rejected.
10. Claim completion only after `"$PYTHON" scripts/finalize_run.py` verifies final
   Benchmark evidence, successfully commits `loop_stop`, and a fresh
   read of `deft_state.json` shows `status == "complete"`,
   `iterations.baseline.status == "complete"`, and the final iteration's
   `status == "complete"`.

Never place secrets in a spec, command, transcript, job-record, or chat. Check
credential presence only, for example
`[ -n "$HF_TOKEN" ] && echo SET || echo UNSET`. Credentials come from the
user's exported shell environment or from a user-approved env file —
`~/.tao/secrets.env`, `~/.config/tao/.env`, or a path the user points at, never
one merely found in the workspace — loaded with
`set -a; source /path/to/.env; set +a`. Never print the file's contents or any
credential value.

## Cosmos3 Model Contract

- Model skill: `tao-finetune-cosmos-reason`.
- Supported canonical base models:
  - `nvidia/Cosmos3-Nano` — default;
  - `nvidia/Cosmos3-Edge` — only when explicitly requested;
  - `nvidia/Cosmos3-Super` — only when explicitly requested.
- Normalize the user aliases `nano`, `edge`, and `super` to those canonical
  IDs. Preserve any variant selected in the prompt. When no variant is
  selected, use Nano.
- Give hardware recommendations for the selected variant and report when the
  available compute is insufficient. If the prompt asks for a variant
  recommendation based on hardware or workload, recommend one with the
  tradeoff, but require an explicit selection before state initialization.
  Never silently switch or fall back to another variant.
- Keep the selected canonical ID as source-model lineage, but do not pass the
  native online checkpoint directly to Cosmos-RL.
- The published Cosmos Reason 3 reasoners ship in Cosmos3's own native Omni
  format (`model_type="cosmos3_omni"`), which Cosmos-RL cannot load. After
  launch approval and before baseline evaluation, run
  `"$PYTHON" <model_skill>/scripts/prepare_cosmos3_vlm_checkpoint.py`
  to convert the selected reasoner
  into a Qwen3-VL safetensors PTM, or validate and reuse an existing prepared
  output.
- Use the prepared PTM consistently for zero-shot evaluation, Train
  `policy.model_name_or_path`, and LoRA `model.base_model_path`. The model
  being trained is still the selected Cosmos Reason 3 reasoner — keep its
  canonical ID as checkpoint lineage; the Qwen3-VL PTM is only the on-disk
  format Cosmos-RL consumes.
- Nano may use the helper's packaged Qwen3-VL default. Edge and Super require
  a variant-specific, validated VLM base; never reuse Nano's conversion
  arguments.
- Container image: resolve the `cosmos-rl` backend from
  `tao-finetune-cosmos-reason/references/skill_info.yaml` with
  `"$PYTHON" "$TAO_SKILL_BANK_PATH/scripts/resolve_tao_image.py"`; never copy a Cosmos image pin into this
  application skill.
- Train action: `cosmos-rl --config <spec.toml>
  /opt/cosmos_rl/tao_sft_example.py`.
- Before the first evaluate job, run `"$PYTHON" scripts/patch_eval_image_cap.py` to
  source-classify the selected image. Mount its output read-only into every
  evaluation container only when it reports `patch_required`; no mount is
  needed for `already_sufficient` or `cap_absent`. An unrecognized cap/vLLM
  shape is a hard stop; see `references/cosmos-reason.md`.
- Workflow override: `automl_policy: off`. DEFT owns iteration and checkpoint
  selection; this is a workflow argument, not a TOML key.
- Default adaptation: LoRA over the language-side projections
  `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
  "down_proj"]`, leaving the vision tower's pretrained weights untouched. The
  schema also accepts `"all-linear"`, which additionally adapts the vision
  linear layers; use it only when the user explicitly requests it. Derive all
  other Train defaults from the model skill's current template.
- Every spec is a nested dictionary serialized to TOML. Never write literal
  flat dotted keys into a spec.
- Do not mount user data over `/workspace`; cosmos-rl is installed there.
- Run every Docker container with a writable host mount as the invoking
  UID:GID with `USER`, `LOGNAME`, `HOME=/tmp`, and the read-only host
  passwd/group databases; never fall back to a root repair container. This
  covers checkpoint preparation, Train, Proxy/Benchmark evaluate, AnomalyGen,
  and mining. See `references/cosmos-reason.md` and
  `references/tao-mine-aoi-images.md`.

Read `skills/models/tao-finetune-cosmos-reason/SKILL.md` and its
`references/skill_info.yaml` before authoring a spec. Start from the model
skill's current packaged template for the selected action and apply only the
AOI workflow overrides in `references/cosmos-reason.md`. Replace every
dataset/output path with the chosen platform's compute-frame path. Prove that
the selected Cosmos-RL image can load the prepared PTM and train the requested
variant; do not reuse Nano conversion, parallelism, or memory assumptions for
Edge or Super.

## Bare OK/NG Contract

This migration supports one annotation mode: `bare_okng`.

- Each record is ShareGPT JSON with exactly two images in
  `[AOI, golden_reference]` order.
- The first human/user turn contains the inspection prompt.
- The final assistant/gpt response is exactly `OK` or `NG`; reasoning,
  prefixes, explanations, and final-answer wrappers are invalid training
  labels.
- `NG` is the positive class. `NG -> OK` is a false accept; `OK -> NG` is a
  false reject.
- Evaluation may normalize a model response by its last standalone `OK`/`NG`
  token, but training labels remain exact.
- Rich, reasoning, BCQ/MCQ, and task fan-out modes are outside this migration.
  Stop instead of silently accepting them.

Run `"$PYTHON" scripts/validate_sharegpt.py` on Proxy, Benchmark, Mining, and each
generated iteration training file. There is no input Train annotation.
Run `"$PYTHON" scripts/validate_split_contract.py` to prove that Proxy, Benchmark, and
Mining targets are disjoint and that the frozen Benchmark annotation hash has
not changed. When a generated Train file is supplied, the same validator
requires its targets to come from Mining, the immediate `--previous-train`
seed, or the current iteration's `--synthetic` AnomalyGen output, and to remain
disjoint from Proxy and Benchmark. For iteration N>1, `--previous-train` is
required and the validator proves that every preceding Train record was
retained.

## KPI Isolation

- **Proxy:** `annotations/proxy_kpi.json`. It is the only error source
  for RCCA, routing, mining targets, and data-mixture decisions. It never stops
  the loop.
- **Benchmark:** `annotations/benchmark_kpi.json`. It is frozen, evaluated
  at baseline and every iteration, and is the only stop-gate source. Benchmark
  sample errors never feed routing or mining.
- Default gate: `recall_ng >= 1.0`. If the user asks for accuracy, use
  `accuracy >= <target>`.
- Unknown model responses block the gate through the
  `unknown_predictions <= 0` metric constraint.

`scripts/analyze_gaps.py` writes Proxy RCCA artifacts or Benchmark aggregate
metrics plus `metric_result.json`. `scripts/record_metric_result.py` binds the
Benchmark metric evidence to the configured metric contract.

## Workspace Contract

```text
workspace/
├── annotations/                 # user-supplied
│   ├── benchmark_kpi.json
│   ├── proxy_kpi.json
│   └── mining_pool.json
├── images/                      # user-supplied
└── specs/                       # produced by this workflow, after approval
    ├── train_spec.toml
    ├── evaluate_spec_proxy.toml
    └── evaluate_spec_benchmark.toml
```

Per-role evaluate specs are preferred over one shared `evaluate_spec.toml`;
both are accepted. See `references/data-layout.md`.

The user supplies annotations and images. The specs are **not** an input to
ask for: build them from the `tao-finetune-cosmos-reason` templates plus the
AOI overrides, and write them after the approval gate and before
`init_deft_state.py`, which refuses to initialize without them. A workspace
carrying its own specs is still valid — reuse them rather than overwriting —
but their absence is normal and is never a reason to stop and ask the user for
a TOML file.

Non-default paths are valid when passed explicitly to
`scripts/init_deft_state.py`; downstream stages must read the recorded paths
instead of re-inferring conventions. Record absolute host/compute-frame
artifact paths under `${RESULTS_DIR}/baseline` or
`${RESULTS_DIR}/iterN`.

Read `references/data-layout.md` for the dataset roles, allowed source
categories, and commercial-training eligibility.

## Launch Intake and Pre-Flight

Read `references/preflight.md` and run every ordered check:

1. select and preflight the platform;
2. resolve workspace, annotations, media root, and `max_iterations`;
3. validate bare ShareGPT and Proxy/Benchmark/Mining target isolation;
4. hash and freeze Benchmark annotations;
5. resolve current Cosmos-RL and data-services images from `versions.yaml`;
6. plan conversion of the selected Cosmos Reason 3 reasoner into a Qwen3-VL
   PTM, and that output's platform-visible path;
7. check only required environment-variable presence;
8. construct Proxy / Benchmark TOML specs and validate the Train template;
9. verify compute shape and path visibility from the selected platform;
10. run the model/platform launch preflight;
11. show the full Pre-Flight Summary and stop for approval.

No results directory, state file, spec mutation, dependency install, image
pull, or native launch is allowed before this gate, except the TAO policy's
small-Python-helper remediation.

## Workflow

The full transition graph is in `references/pipeline-and-state.md`.

The frozen Benchmark gate is always evaluated before any Proxy work. Proxy
evaluate and RCCA exist only to seed the next iteration's mining, so they run
only when the gate is unmet. A run that passes the gate stops without spending
a Proxy evaluation.

Baseline starts with zero-shot frozen Benchmark evaluation of the unmodified
base model, which establishes the zero-shot KPI:

1. `evaluate_benchmark`
2. `benchmark_metrics` — stop here when the gate already passes.
3. `evaluate_proxy` — only when the gate is unmet.
4. `proxy_rcca`

Before every `proxy_rcca` commit, write `proxy_rcca/RCCA_Report.md` from the
three Proxy RCCA JSON artifacts using `references/RCCA_REPORT_TEMPLATE.md`,
then pass it with `--rcca-report`. Artifact requirements, section headings,
and state fields come from `references/rcca-artifact-manifest.json`.

For each `iterN` when the frozen Benchmark gate is unmet:

1. `routing` — derive mining targets from Proxy false accepts/rejects only.
   Write both formats from the same rows: `mining_targets.json` for state
   (`--mining-targets` takes the JSON) and a `filepath[,label]` parquet for the
   embedding container. Gap rows carry no image paths, so join back to Proxy by
   `id` — see `references/gap-analysis.md`.
2. `anomalygen` — generate synthetic defects with `tao-generate-anomalies` in
   `inference_only` mode, then turn each generated pair into a bare `NG`
   record with `"$PYTHON" scripts/emit_sdg_sharegpt.py`. `--skip` is permitted only when
   the driving Proxy RCCA recorded zero false accepts, and even then generating
   is often still worthwhile. The emitter accepts PAIDF 1.0.1 repo-root-relative
   and documented output-dir-relative paths, with `--sdg-root` as an explicit
   additional base — see `references/tao-generate-anomalies.md`.
3. `data_mining` — invoke `tao-mine-aoi-images`, apply the configured cosine
   floor with `"$PYTHON" scripts/filter_mined_by_cosine.py`, then run the mapped skill's
   history-aware post-processing so a filepath selected by a prior iteration
   cannot enter Train again. The default top-K remains 5; preserve an explicit
   user value and increase it only when the history summary shows low novelty.
4. `assemble_data` — align mined target paths to Mining source prompts,
   golden references, and exact labels with `"$PYTHON" scripts/emit_mined_sharegpt.py`;
   create `train_iter_1.json` from the mined and synthetic records only after
   Proxy RCA and Mining selection, then append monotonically into
   `train_iter_N.json` in later iterations with
   `"$PYTHON" scripts/assemble_training_json.py`.
5. `validate_data` — validate exact bare labels, files, duplicates, and
   generated-Train lineage plus Proxy/Benchmark leakage.
6. `train`
7. `evaluate_benchmark`
8. `benchmark_metrics` — stop here when the gate passes or
   `N = max_iterations`.
9. `evaluate_proxy` — only when the loop continues.
10. `proxy_rcca`

`init_deft_state.py` writes the first `DEFT_Loop_Report.html`; every successful
`commit_stage.py` call then refreshes it through the deterministic
`scripts/render_report.py` post-commit hook. Stop when the Benchmark contract
passes, `max_iterations` is reached, or a hard stop occurs. For an ordinary
stop, run `"$PYTHON" scripts/finalize_run.py` with the explicit reason, then run
`"$PYTHON" scripts/render_report.py --require-terminal` after optional token alignment.
The Cosmos-only report addition is a bounded prompt showcase sourced from
recorded annotations; keep every other visual convention aligned with
ChangeNet. See `references/REPORT_RENDERING.md`. Never delegate or hand-author
report rendering.

## Stage References

| Stage | Producer | Read first |
|---|---|---|
| Train | `tao-finetune-cosmos-reason` train, `automl_policy: off` | `references/cosmos-reason.md`, `references/example_lora_config.toml` |
| Proxy / Benchmark evaluate | `tao-finetune-cosmos-reason` evaluate | `references/cosmos-reason.md` |
| Proxy RCCA / Benchmark metric | bundled `analyze_gaps.py` | `references/gap-analysis.md` |
| Routing / mining | Proxy gaps + `tao-mine-aoi-images` | `references/tao-mine-aoi-images.md` |
| AnomalyGen | `tao-generate-anomalies`, `mode=inference_only` | `references/tao-generate-anomalies.md` |
| Assemble / validate | bundled bare ShareGPT scripts | `references/aoi-annotation.md` |
| State/report | bundled state commit + deterministic report hook | `references/scripts-and-agents.md` |

## Hard Stops

Commit an error stage and do not auto-retry for: invalid disk state; a rich or
non-exact training label; a JSONL or non-array annotation input; an
an unconverted Cosmos Reason 3 checkpoint still in native Omni format at a
Cosmos-RL boundary;
missing/ambiguous mined-to-source alignment; missing/tampered mining history,
cross-iteration mined filepath duplication; target overlap among
Proxy/Benchmark/Mining; a generated Train target outside Mining and AnomalyGen
output, or overlapping Proxy/Benchmark; a changed Benchmark hash; any Benchmark
error used for routing; missing/empty mining output; a failed or empty
AnomalyGen run while Proxy false accepts remain outstanding; an `anomalygen`
skip not backed by zero false accepts in the driving RCCA; a synthetic record
whose label is not `NG` or whose paired image is missing; a
PAIDF-incompatible AnomalyGen fine-tuned checkpoint; a missing AnomalyGen
Guardrail checkpoint or an SDG log showing disabled screening; a checkpoint outside
the iteration result tree; an invalid nested TOML spec; unknown evaluator
ground truth; or a program error.

Infrastructure errors may follow the chosen platform skill's bounded retry
policy with a new job-record linked by `--retry-of`; the DEFT stage is committed
only once, after a successful terminal backend result.
