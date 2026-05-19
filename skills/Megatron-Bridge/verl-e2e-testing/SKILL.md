---
name: verl-e2e-testing
description: External verl end-to-end validation workflow for Megatron-Bridge model/provider changes. Covers running a small verl Megatron backend job from a Bridge checkout, choosing LoRA/DDP plus optional save/resume and parallelism variants, setting PYTHONPATH so verl imports the local Bridge tree, and reporting pass/fail evidence.
when_to_use: Adding or changing a Megatron-Bridge model/provider and needing downstream verl compatibility validation; checking non-vanilla Bridge provider paths; testing PEFT/LoRA, DDP, checkpoint behavior, or explicitly requested advanced variants through verl; 'does this model work in verl', 'run verl e2e', 'external RL loop validation'.
---

# verl E2E Testing

Validate a Megatron-Bridge model addition through verl's Megatron backend. This catches integration issues that Bridge-only conversion tests miss: provider configuration, HF import through Bridge, PEFT wrapping, DDP wrapping, optimizer setup, rollout/ref wiring, and checkpoint ownership by an external RL loop.

Use this as an external compatibility smoke test after the Bridge unit and functional tests for a new model provider are green.

This is not a replacement for Bridge model parity tests. The default verl PPO run proves that the provider can survive an external RL training loop; architecture-specific correctness still comes from Bridge import/export, logits/roundtrip, and model-specific inference tests.

## Scope

Think in coverage levels. Start with Level 0 and add only the levels justified by the change.

| Level | Required when | What it proves |
|---|---|---|
| 0: LoRA + DDP smoke | Any new provider or provider config change that claims verl compatibility | verl can import the local Bridge provider, apply PEFT, wrap with Megatron DDP, build optimizer state, run rollout/ref/critic wiring, and finish one PPO step |
| 1: Save/resume | PEFT, checkpointing, HF export, adapter export, optimizer state, or resume behavior changed | verl-owned checkpoint scheduling can save and reload Bridge-built model state |
| 2: Parallelism stress | Provider finalization, mpu-derived settings, TP/PP/CP/EP, sequence parallel, or dispatcher behavior changed | provider settings remain correct under non-trivial Megatron parallel state |
| 3: Optional Megatron-FSDP | Only when downstream explicitly asks for verl Megatron-FSDP coverage or the change directly touches that integration path | the same provider works when verl selects Megatron-FSDP instead of DDP |
| 4: Architecture-specific e2e | VLM, MoE, MTP, QAT/ModelOpt, quantized weights, or custom layer behavior is involved | the part of the architecture not exercised by text-only GSM8K also has a targeted runtime check |
| 5: Convergence / learning signal | Optimizer, scheduler, loss, reward, PEFT trainability, gradient flow, or model-specific training stability changed | metrics move in the expected direction over a short run and do not silently produce zero/NaN/unstable updates |

The default Level 0 target is a short, non-vanilla Bridge run in verl with LoRA enabled and Megatron DDP selected:

```bash
USE_MBRIDGE=True
VANILLA_MBRIDGE=False
VALUE_VANILLA_MBRIDGE=False
LORA_RANK=4
USE_MEGATRON_FSDP=False
TOTAL_TRAIN_STEPS=1
```

This is intentionally small. It exercises the Bridge-facing path in verl without making Megatron-Bridge own rollout scheduling, reward handling, optimizer scheduling, or checkpoint orchestration.

Level 0 is not a convergence test. It only proves the training loop can complete one update. Use Level 5 when the question is whether the model actually learns under verl.

Megatron-FSDP is not part of the default validation expected for current provider compatibility work. Run it only for Level 3 coverage when FSDP is explicitly in scope:

```bash
USE_MEGATRON_FSDP=True
ALL_OFFLOAD=False
COMMON_PP=1
COMMON_VPP=null
COMMON_CP=1
COMMON_TP=1
INFER_TP=1
```

## Repos

Use explicit repo variables. Do not rely on an installed `megatron-bridge` wheel; the purpose is to test the current Bridge checkout.

Use the upstream verl repository as the default source:

```text
https://github.com/verl-project/verl
```

If a checkout is not already available, clone it next to the Bridge checkout or into the site's standard workspace:

```bash
git clone https://github.com/verl-project/verl.git /path/to/verl
```

```bash
export BRIDGE_REPO=${BRIDGE_REPO:-/path/to/Megatron-Bridge}
export VERL_REPO=${VERL_REPO:-/path/to/verl}
export PYTHONPATH="${BRIDGE_REPO}/src:${BRIDGE_REPO}/3rdparty/Megatron-LM:${VERL_REPO}:${PYTHONPATH:-}"
```

Before running, record both states:

```bash
git -C "$BRIDGE_REPO" status --short
git -C "$VERL_REPO" status --short
git -C "$BRIDGE_REPO" rev-parse --short HEAD
git -C "$VERL_REPO" rev-parse --short HEAD
```

If testing on a remote GPU machine, sync the exact local changes first. Do not reset or overwrite unrelated changes in either tree.

Verify that Python imports the checkout under test:

```bash
python - <<'PY'
import megatron.bridge
print(megatron.bridge.__file__)
PY
```

The printed path must live under `$BRIDGE_REPO/src`. If it points at site-packages, fix `PYTHONPATH` before trusting any result.

When running against an existing Ray cluster, the driver import is not enough. Ray workers may not inherit the shell `PYTHONPATH`, and the run can fail later with `ModuleNotFoundError: No module named 'megatron.bridge'` even though the driver import passed. Verify a worker import:

```bash
python - <<'PY'
import os
import ray

ray.init(address="auto", runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}})

@ray.remote
def bridge_path():
    import megatron.bridge
    return megatron.bridge.__file__

print(ray.get(bridge_path.remote()))
PY
```

If the worker import fails, pass the checkout path through Ray's runtime environment when launching the wrapper:

```bash
bash tests/special_e2e/run_ppo_trainer_megatron.sh \
  "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${PYTHONPATH}"
```

If this import fails before model construction, fix the runtime environment first. The official verl image may not contain every Bridge dependency; for example, Bridge imports `modelopt` through `AutoBridge`, so a missing `nvidia-modelopt` can fail the smoke before verl exercises the provider:

```bash
python -m pip show nvidia-modelopt || \
  python -m pip install --extra-index-url https://pypi.nvidia.com nvidia-modelopt
```

Treat ad-hoc installs as container setup evidence, not repository changes. If the image lacks `uv`, run focused Bridge unit tests in a Bridge development environment instead of forcing them through the verl container.

## Model Choice

Prefer the smallest public HF checkpoint that uses the changed provider family. For example, use a 0.5B or 0.6B dense checkpoint for dense provider changes before testing larger variants.

If there is no small public checkpoint for the new architecture, use verl's dummy-model path with a minimal HF config from that architecture:

```bash
USE_DUMMY_MODEL=True
DUMMY_MODEL_CONFIG_PATH=/path/to/minimal_config.json
MODEL_ID=<org>/<representative-model-name>
```

Report dummy-model results carefully: they validate model construction and training mechanics, not pretrained weight compatibility.

For VLMs, the generic GSM8K PPO run is text-only. It can validate the language-side Bridge provider and external-loop wrapping, but it does not prove image/video preprocessing or vision encoder execution. Pair it with the VLM conversion/inference tests from @skills/adding-model-support/tests-and-examples.md, or use a verl multimodal training command if one exists for the model family.

For MoE models, Level 0 with `COMMON_EP=1` still catches many provider and PEFT issues, but it does not stress expert parallel routing. Add a Level 2 run with expert parallelism when the change touches expert layout, dispatcher config, router replay, or expert tensor parallelism.

For MTP, QAT/ModelOpt, or quantized checkpoint support, the generic wrapper may not activate the feature. Use the closest verl example or model-family script that turns the feature on, and record the extra Hydra overrides in the report.

## Bridge Checks First

Run focused Bridge tests before the external verl e2e. Include any model-specific tests added by the change.

```bash
cd "$BRIDGE_REPO"
uv run python -m pytest -q \
  tests/unit_tests/models/test_model_provider_mixin.py \
  tests/unit_tests/models/test_param_mapping.py \
  tests/unit_tests/training/test_integration.py \
  <model-specific-test-paths>
```

For a new model family, also run the relevant conversion or roundtrip test from the model's PR. See @skills/adding-model-support/tests-and-examples.md for model-test patterns.

Minimum Bridge-side evidence for a new model/provider:

- provider/config unit tests
- parameter mapping tests
- HF to Megatron import or roundtrip on a small model
- model-specific generation or logits comparison when available
- this verl external-loop smoke after the above pass

## verl Data Setup

verl's Megatron PPO smoke wrapper expects GSM8K parquet files by default. Prepare them once from the verl checkout if they are missing:

```bash
cd "$VERL_REPO"
export PYTHONPATH="$VERL_REPO:${PYTHONPATH:-}"
python3 examples/data_preprocess/gsm8k.py \
  --local_save_dir "${GSM8K_DIR:-$HOME/data/gsm8k}"
```

Use `--local_dataset_path "$GSM8K_SOURCE_DIR"` only when that raw local dataset path actually exists. Otherwise let `datasets` fetch `openai/gsm8k`.

Set explicit paths when running in a container or shared filesystem:

```bash
export TRAIN_FILES=/path/to/gsm8k/train.parquet
export VAL_FILES=/path/to/gsm8k/test.parquet
```

The wrapper also enables a reward model by default. Ensure the default reward model path exists, or set:

```bash
export RM_MODEL_PATH=/path/to/local/reward/model
```

For a Level 0 rule-reward smoke, it is acceptable to disable the reward-model rollout when no local reward model is available:

```bash
bash tests/special_e2e/run_ppo_trainer_megatron.sh \
  reward.reward_model.enable=False
```

Report this as a limitation; it still tests Bridge actor/ref/critic construction, LoRA, DDP wrapping, rollout, and one PPO update, but not reward-model serving.

## Minimal verl Run

Use verl's maintained wrapper rather than constructing a long Hydra command manually:

```bash
cd "$VERL_REPO"
ray stop --force || true

export MODEL_ID=<small-compatible-hf-model>
export TRAIN_FILES=${TRAIN_FILES:-/path/to/gsm8k/train.parquet}
export VAL_FILES=${VAL_FILES:-/path/to/gsm8k/test.parquet}
export RM_MODEL_PATH=${RM_MODEL_PATH:-/path/to/local/reward/model}
export ENGINE=vllm
export USE_MBRIDGE=True
export VANILLA_MBRIDGE=False
export VALUE_VANILLA_MBRIDGE=False
export LORA_RANK=4
export USE_MEGATRON_FSDP=False
export COMMON_PP=1
export COMMON_VPP=null
export COMMON_CP=1
export COMMON_TP=1
export INFER_TP=1
export ALL_OFFLOAD=False
export TOTAL_TRAIN_STEPS=1
export SAVE_FREQ=-1
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=-1

bash tests/special_e2e/run_ppo_trainer_megatron.sh
```

Use `MODEL_ID` when the checkpoint is available through the wrapper's default cache layout. Add `MODEL_PATH=/path/to/local/hf/model` only when testing a local or converted checkpoint.

When `$HOME` is small or shared slowly, put HF caches and downloaded checkpoints on a larger shared or node-local scratch path and pass `MODEL_PATH` explicitly. Pre-download large models once in the same container environment to avoid Ray workers racing the cache:

```bash
export HF_HOME=${HF_HOME:-/scratch/$USER/verl_hf}
export HF_HUB_CACHE=$HF_HOME/hub
MODEL_PATH=${MODEL_PATH:-/scratch/$USER/models/<org>/<model>}
hf download <org>/<model> --local-dir "$MODEL_PATH"
```

Capture logs to a file for review:

```bash
mkdir -p "${LOG_DIR:-$PWD/verl_e2e_logs}"
LOG_FILE="${LOG_DIR:-$PWD/verl_e2e_logs}/verl_lora_ddp_$(date +%Y%m%d_%H%M%S).log"
bash tests/special_e2e/run_ppo_trainer_megatron.sh \
  "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${PYTHONPATH}" \
  2>&1 | tee "$LOG_FILE"
grep -E "Training Progress|VANILLA_MBRIDGE|Traceback|RuntimeError|KeyError|ValueError" "$LOG_FILE"
```

Prefer a saved log over a pasted terminal excerpt in PR descriptions.

For time-limited smoke runs where vLLM CUDA graph capture dominates setup and the goal is Bridge provider validation, it is acceptable to add:

```bash
bash tests/special_e2e/run_ppo_trainer_megatron.sh \
  actor_rollout_ref.rollout.enforce_eager=True
```

Report this override as a limitation. It still validates Bridge import, HF import, LoRA, Megatron DDP, rollout wiring, and one PPO update, but not vLLM CUDA graph capture.

## Save/Resume Coverage

After the minimal run passes, add checkpoint coverage if the change touches PEFT, checkpointing, export, or optimizer state:

```bash
# Save once.
SAVE_FREQ=1 TOTAL_TRAIN_STEPS=1 \
bash tests/special_e2e/run_ppo_trainer_megatron.sh

# Resume and train one more step.
RESUME_MODE=auto SAVE_FREQ=1 TOTAL_TRAIN_STEPS=2 \
bash tests/special_e2e/run_ppo_trainer_megatron.sh
```

Remove stale verl `checkpoints/` output between unrelated experiments. Keep it for resume validation.

## Parallelism Stress

Use Level 2 when the provider reads or mutates parallelism-related fields, or when the change touches `provider.configure(...)`, Megatron `mpu`, sequence parallel, context parallel, MoE dispatcher behavior, or tensor/expert tensor parallel settings.

The variants below assume the Level 0 exports above are still in the shell; each command overrides only the values being tested.

Example dense stress variant:

```bash
COMMON_TP=2 \
COMMON_PP=2 \
COMMON_VPP=null \
COMMON_CP=1 \
INFER_TP=2 \
USE_MEGATRON_FSDP=False \
bash tests/special_e2e/run_ppo_trainer_megatron.sh
```

Example MoE stress variant, only for compatible MoE checkpoints:

```bash
COMMON_EP=2 \
COMMON_ETP=1 \
ROUTING_REPLAY_MODE=disabled \
bash tests/special_e2e/run_ppo_trainer_megatron.sh
```

Keep these as follow-up runs. Do not make them the first debugging surface for a new provider.

## Optional Megatron-FSDP Variant

Use Level 3 after Level 0 passes only when downstream explicitly requests Megatron-FSDP coverage or the Bridge change directly touches FSDP wrapping, sharding, checkpoint format, or distributed optimizer behavior:

```bash
USE_MEGATRON_FSDP=True \
ALL_OFFLOAD=False \
COMMON_PP=1 \
COMMON_VPP=null \
COMMON_CP=1 \
COMMON_TP=1 \
INFER_TP=1 \
bash tests/special_e2e/run_ppo_trainer_megatron.sh \
  ++actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
  ++actor_rollout_ref.ref.megatron.override_transformer_config.gradient_accumulation_fusion=False \
  ++critic.megatron.override_transformer_config.gradient_accumulation_fusion=False
```

For Bridge-native FSDP behavior and constraints, also read @skills/perf-megatron-fsdp/SKILL.md.

## Convergence / Learning Signal

Use Level 5 only when the change affects trainability or when downstream validation explicitly asks for convergence. Do not require it for every provider-only PR; RL convergence is slower, noisier, and more environment-dependent than the compatibility smoke.

The goal is a short learning-signal run, not a full benchmark. Prefer a small model, fixed data, fixed seed when available, and enough steps to observe non-random metric movement:

```bash
TOTAL_TRAIN_STEPS=20 \
SAVE_FREQ=-1 \
VAL_BEFORE_TRAIN=True \
TEST_FREQ=10 \
LORA_RANK=4 \
USE_MBRIDGE=True \
VANILLA_MBRIDGE=False \
VALUE_VANILLA_MBRIDGE=False \
USE_MEGATRON_FSDP=False \
ENGINE=vllm \
bash tests/special_e2e/run_ppo_trainer_megatron.sh
```

For a stronger signal, run 50-100 steps if GPU time allows. Keep rollout, reward model, dataset, batch sizes, and model checkpoint fixed between baseline and candidate runs.

Acceptable convergence evidence depends on the task, but the report should include at least:

- no NaNs or infs in loss, reward, KL, entropy, grad norm, or logprob metrics
- nonzero trainable parameter count when PEFT is enabled
- actor/critic losses and reward-related metrics logged for multiple steps
- validation or reward trend compared against the starting point or a known-good baseline
- no repeated zero gradients, frozen LoRA adapters, or constant logprobs unless expected

Do not call a 20-step run "converged" in the benchmark sense. Call it "learning-signal passed" unless it reaches a pre-agreed metric threshold.

## Container Image

Use the official verl Docker images as the default source:

```text
https://hub.docker.com/r/verlai/verl
```

For this skill's default PPO smoke path, pick a vLLM-flavored `verlai/verl` image tag unless the test intentionally changes the rollout engine. The maintained wrapper defaults to vLLM, and the command should make that explicit with:

```bash
ENGINE=vllm
```

Avoid using sglang, TRT-LLM, or generic images for the default Level 0 run unless the point of the test is to validate that rollout backend. A backend-specific image can fail before Bridge model construction, which makes the result a poor signal for a Megatron-Bridge provider change.

Pin the exact image tag in the test log or PR description:

```bash
export VERL_IMAGE=${VERL_IMAGE:-verlai/verl:<vllm-compatible-tag>}
```

If the cluster requires Enroot/SquashFS images, convert or mirror the selected `verlai/verl` tag through the site's normal process, but keep the source tag visible in the report.

## Slurm or Container Runs

Use the cluster's standard container and mount both checkouts into the container. Keep setup and the actual PPO run in the same container step when using node-local paths such as `/tmp`; node-local model caches and ad-hoc pip installs disappear when a fresh container step starts. Keep paths generic in scripts committed to Megatron-Bridge:

```bash
export VERL_IMAGE=${VERL_IMAGE:-verlai/verl:<vllm-compatible-tag>}

srun <site-specific-slurm-options> \
  --container-image="${VERL_IMAGE}" \
  --container-mounts="${BRIDGE_REPO}:/workspace/Megatron-Bridge,${VERL_REPO}:/workspace/verl,<data-root>:<data-root>" \
  --container-workdir=/workspace/verl \
  bash -lc '
    export BRIDGE_REPO=/workspace/Megatron-Bridge
    export VERL_REPO=/workspace/verl
    export PYTHONPATH=$BRIDGE_REPO/src:$BRIDGE_REPO/3rdparty/Megatron-LM:$VERL_REPO
    ray stop --force || true
    MODEL_ID=<small-compatible-hf-model> \
    ENGINE=vllm \
    USE_MBRIDGE=True VANILLA_MBRIDGE=False VALUE_VANILLA_MBRIDGE=False \
    LORA_RANK=4 USE_MEGATRON_FSDP=False TOTAL_TRAIN_STEPS=1 SAVE_FREQ=-1 \
    bash tests/special_e2e/run_ppo_trainer_megatron.sh
  '
```

Use a persistent log directory on a shared filesystem or `$HOME` for long Slurm jobs. Logs written only to node-local `/tmp` can disappear when the allocation expires or is canceled. If a cluster attach helper runs the command through `srun --pty`, do not background the workload inside that attached shell; the step cleanup can terminate it immediately. To detach safely, background the attach helper itself from the login node:

```bash
mkdir -p "$HOME/verl_e2e_logs"
nohup env COMMAND='bash /path/to/run_verl_e2e.sh' \
  bash /path/to/<jobid>-attach.sh \
  > "$HOME/verl_e2e_logs/attach_driver_$(date +%Y%m%d_%H%M%S).out" 2>&1 &
```

If an attach helper enters a container that no longer sees the expected checkouts or log directory, treat that helper as stale. Start a fresh `srun` step against the existing allocation with explicit `--container-image`, `--container-mounts`, and `--container-workdir`.

On CUDA/H100 clusters, some launchers set both `CUDA_VISIBLE_DEVICES` and ROCm variables such as `ROCR_VISIBLE_DEVICES`. If verl workers fail before model construction with `Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set`, fix the launcher/container environment or apply a temporary local verl workaround that drops `ROCR_VISIBLE_DEVICES` when CUDA is already set. Do not report this as a Bridge provider failure.

For general Slurm debugging and multi-node patterns, read @skills/multi-node-slurm/SKILL.md.

## Pass Criteria

A useful pass has all of the following:

- Focused Bridge tests pass for provider/config/mapping behavior.
- verl uses the local Bridge checkout through `PYTHONPATH`.
- The verl log shows `VANILLA_MBRIDGE=False`.
- One training step reaches completion, for example `Training Progress: 100%|1/1|`.
- No exception occurs during Bridge provider setup, HF import, LoRA wrapping, DDP wrapping, optional FSDP wrapping when enabled, optimizer setup, checkpoint manager setup, or the training step.

Ray shutdown, Python resource-tracker warnings, or post-completion DataLoader worker termination can be acceptable if the training step completed, metrics for `training/global_step:1` were logged, and the process exits successfully. Mention them as residual log noise.

Do not claim full model e2e if the run used a dummy config, text-only data for a VLM, `COMMON_EP=1` for an expert-parallel change, or disabled save/resume for a checkpointing change. Call it the exact level that passed.

Do not claim convergence from Level 0. Claim convergence only from Level 5, and distinguish "learning signal" from "benchmark convergence" in the report.

## Failure Triage

If model construction fails, check whether the Bridge provider is finalized with the same parallel sizes that verl initialized through Megatron `mpu`.

If LoRA fails, check target module names and whether the provider path uses the non-vanilla Bridge PEFT helpers expected by verl.

If checkpoint save/load fails, first rerun without save/resume (`SAVE_FREQ=-1`) to separate model construction from checkpoint behavior.

If rollout fails before actor construction, this may be a verl rollout engine issue rather than a Bridge provider issue. Report the boundary clearly.

If the log shows the wrong Bridge path, stop. Any later failure or pass is not evidence for the local Bridge change.

If the baseline fails before model build because of data, reward model, Ray, vLLM, or package setup, fix the environment first and do not report it as a provider failure.

If model download fails with `No space left on device`, move `HF_HOME`, `HF_HUB_CACHE`, and `MODEL_PATH` to a larger shared or node-local path, then rerun with the explicit local `MODEL_PATH`.

## Summary Format

End every run with a short user-facing summary that answers "Did the requested deliverables pass?" before adding details. Use `Pass`, `Fail`, `Skipped`, or `Blocked` for each deliverable, and do not report an overall `Pass` unless the pass criteria for the requested coverage level were met.

```text
Result: <Pass/Fail/Blocked> - <one sentence stating what was validated>
Requested coverage: <Level 0/1/2/3/4/5 and requested variants>
Model: <MODEL_ID or MODEL_PATH>

Deliverables:
- Bridge-side checks: <Pass/Fail/Skipped> - <test command or skipped reason>
- Local Bridge import in verl: <Pass/Fail> - <PYTHONPATH or imported Bridge path>
- verl Megatron backend run: <Pass/Fail/Skipped> - <LoRA + DDP or requested variant>
- Requested variants: <Pass/Fail/Skipped/Not requested> - <save/resume, parallelism stress, Megatron-FSDP, architecture-specific run, or learning-signal/convergence>
- Log capture: <Pass/Fail> - <log path>

Evidence:
- Bridge repo: <commit> plus dirty files
- verl repo: <commit> plus dirty files
- Command: <exact command or script path>
- Key lines: <VANILLA_MBRIDGE=False, Training Progress completion, training/global_step:1, or the first relevant error>

Limitations:
- <dummy model, skipped save/resume, COMMON_EP=1, text-only data for VLM, no convergence claim, known shutdown warnings, etc.>

Follow-ups:
- <needed rerun, environment fix, provider fix, or "none">
```

If the job is blocked before Bridge model/provider construction by data, reward model, Ray, vLLM, dependency, disk, or cluster setup, mark the overall result as `Blocked`, not `Fail`, and state that it is not evidence against the Bridge provider.

If any requested deliverable was not run, mark it `Skipped` or `Not requested` with the reason. Do not leave it implicit in the limitations.
