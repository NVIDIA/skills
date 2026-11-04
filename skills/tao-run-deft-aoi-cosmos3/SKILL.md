---
name: tao-run-deft-aoi-cosmos3
description: >
  Run the disk-backed DEFT AOI improvement loop for NVIDIA Cosmos Reason 3 /
  Cosmos3 models, using Nano by default and Edge or Super when explicitly
  requested: evaluate the base model on Proxy and frozen Benchmark splits,
  mine real image pairs from Proxy gaps, generate AnomalyGen synthetic NG
  pairs, assemble a per-iteration Train JSON from both producers, train with
  Cosmos Framework LoRA SFT, and repeat
  through the selected platform's submit/status/logs/cancel contract.
  This migration supports bare labels only: the assistant response must be
  exactly OK or NG. Use for "run Cosmos3 DEFT AOI", "CR3 AOI loop", or
  "improve Cosmos3 PCB inspection with bare OK/NG"; do not use for
  rich/reasoning annotation, one-off Cosmos training, or generic anomaly
  generation.
license: Apache-2.0 AND CC-BY-4.0
compatibility: Requires the companion TAO skill-bank skills from `eval.config`, host Python with `pyarrow`, `yaml`, and either Python 3.11+ `tomllib` or Python 3.10 with the `tomli` fallback used by the shared `cosmos_workflow.py` planner, the selected platform's native CLI, and PR 230 model helper `--backend`.
metadata:
  author: NVIDIA Corporation
  version: "0.2.0"
allowed-tools: Read Task Bash Write
tags:
- application
- workflow
- deft
- aoi
- cosmos
---

# tao-run-deft-aoi-cosmos3

## Installation

Install this application as part of the full TAO skill-bank root, not as only
the companion skill folders: a usable install places `skills/`, `scripts/`,
`templates/`, and `versions.yaml` under the same `TAO_SKILL_BANK_PATH`. Any
install that ships only the skill folders, for example an agent plugin or
skills-only install, must also provide the bank-level scripts, templates, and
`versions.yaml` and point `TAO_SKILL_BANK_PATH` at their common root before
running `scripts/resolve_tao_model.py`. Run bundled validation with the skill
Python so dependencies match runtime: `PYTHON=$(bash scripts/deft_python.sh); "$PYTHON"
-m unittest tests.test_cosmos3_bare`. Resolve network mode first. Missing
air-gap imports are a hard stop; network-enabled setup lives only in
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
   `"$PYTHON" scripts/init_deft_state.py`. Require `--base-model-path` to name
   the prepared Qwen3-VL PTM, freeze the image with `--framework-image-digest`,
   and pass its container URI with `--framework-container`. Pass the exact GPU model reported by the
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
   updates both the resume snapshot and ordered `events` array in state.
   `commit_stage.py --stage train` requires `--framework-config` with the saved
   Hydra `config.yaml`. Every
   executed-stage commit requires a positive, measured `--duration-sec`: use
   backend elapsed wall time for submitted jobs and a host wall-clock timer for
   inline stages. A documented `--skip` may record `0`; negative durations are
   always rejected.
10. Claim completion only after `"$PYTHON" scripts/finalize_run.py` verifies final
   Benchmark evidence, successfully commits `loop_stop`, and a fresh
   read of `deft_state.json` shows `status == "complete"`,
   `version == 6`, non-empty `final_artifacts`,
   `iterations.baseline.status == "complete"`, and the final iteration's
   `status == "complete"`.

Never place secrets in a spec, command, transcript, job-record, or chat. Check
credential presence only. Credentials come from the
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
- Resolve actions with `resolve_tao_model.py`, then `cosmos_workflow.py
  resolve --backend cosmos-framework --workload training`.
- Keep the selected canonical ID as source-model lineage, but do not pass the
  native online checkpoint directly to Framework.
- The published Cosmos Reason 3 reasoners ship in Cosmos3's own native Omni
  format (`model_type="cosmos3_omni"`), which Framework cannot load. After
  launch approval and before baseline evaluation, run
  `"$PYTHON" <model_skill>/scripts/prepare_cosmos3_vlm_checkpoint.py`
  to convert the selected reasoner into a Qwen3-VL safetensors PTM, or validate
  and reuse an existing prepared output.
- Use the prepared PTM consistently for zero-shot evaluation, Framework Train,
  Inference, and DCP vision loading. The model being trained is still the
  selected Cosmos Reason 3 reasoner — keep its canonical ID as checkpoint
  lineage; the Qwen3-VL PTM is only the on-disk format Framework consumes.
- Nano may use the helper's packaged Qwen3-VL default. Edge and Super require
  a variant-specific, validated VLM base; never reuse Nano's conversion
  arguments.
- Pass helper `--backend cosmos-framework`; use the resolved Framework
  image/digest for `--runtime-image`/`--runtime-image-digest` and the same
  immutable image for Train, Evaluate, and Inference. Never copy its pin here.
- Train with `cosmos-framework-train --sft-toml=/tao/config/train.toml`;
  Evaluate with `cosmos-framework-evaluate --config /tao/config/evaluate.toml`;
  use the public Framework inference entrypoint for standalone inference.
- Render Train through `"$PYTHON" scripts/render_cfw_sft.py` and each role's
  Evaluate TOML through `"$PYTHON" scripts/render_cfw_evaluate.py`. Never reuse a packaged
  workspace TOML: the renderers/submitters fail closed on stale backend paths,
  tokenizer/generation-era fields, or missing Framework-native keys.
- Train keeps `model.attn_implementation="cosmos"` because `sdpa` collapses
  Framework LoRA SFT to the majority label on this image. Evaluate uses `sdpa`
  because its Hugging Face loader rejects `cosmos`.
- Framework Train writes a native DCP checkpoint and a saved Hydra `config.yaml`
  beside it. Record both; iteration Evaluate sets `model.config_file` to that
  `config.yaml`, never the input SFT TOML, and uses the prepared PTM for vision.
  The next Train sets `checkpoint.load_path` to the preceding DCP. Commit the
  DCP right after a successful Train and clear checkpoints before an intentional retrain.
- Every spec is a nested dictionary serialized to TOML. Never write literal
  flat dotted keys into a spec.
- Do not mount user data over `/workspace`; Framework is installed there.
- Run every Docker container with a writable host mount as the invoking
  UID:GID with `USER`, `LOGNAME`, `HOME=/tmp`, and read-only host
  passwd/group databases. The submit helpers add automatic read-only identity
  mounts for absolute model, annotation, media, config, and checkpoint paths.
  See `references/cosmos-reason.md` and
  `references/tao-mine-aoi-images.md`.

Read `skills/models/tao-finetune-cosmos-reason/SKILL.md`, its
`references/skill_info.yaml`, and the selected Framework backend contract
before authoring a spec. Replace every dataset/output path with the chosen
platform's compute-frame path and preserve the selected model variant.

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
ask for: render them with this application's Framework renderers after the
approval gate and before `init_deft_state.py`, which refuses to initialize
without them. A workspace carrying its own specs is valid data input, but every packaged
TOML is stale evidence: regenerate Train, Proxy Evaluate, and Benchmark
Evaluate specs with this checkout's Framework renderers. Their absence is
normal and is never a reason to stop and ask the user for a TOML file.

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
5. resolve current Framework and data-services images from `versions.yaml`;
6. plan its Qwen3-VL PTM conversion and platform-visible output;
7. check only required environment-variable presence;
8. render Framework Train / Proxy Evaluate / Benchmark Evaluate TOML specs;
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
Follow `references/REPORT_RENDERING.md`; never delegate or hand-author report
rendering.

## Stage References

Read the stage-to-producer/reference table in
`references/scripts-and-agents.md` before each stage.

## Hard Stops

Commit an error stage and do not auto-retry for: invalid disk state; a rich or
non-exact training label; a JSONL or non-array annotation input; a
native Omni at a Framework boundary; stale TOML; an invalid Framework DCP; or
a missing saved Train Hydra `config.yaml`;
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
