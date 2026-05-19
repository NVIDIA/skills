---
name: nemo-rl-e2e-testing
description: External NeMo-RL end-to-end validation workflow for Megatron-Bridge model/provider changes, including downstream compatibility checks, external RL lifecycle behavior, Megatron policy setup, HF import/export, checkpoint/resume, non-colocated vLLM refit, delta weight transfer, optional LoRA/generation variants, and questions such as "does this model work in NeMo-RL", "run NeMo-RL e2e", or "external RL loop validation". Covers running NeMo-RL Megatron policy jobs from a Bridge checkout, choosing GRPO/SFT/checkpoint/non-colocated refit variants, setting PYTHONPATH so NeMo-RL imports the local Bridge tree, and reporting pass/fail evidence.
when_to_use: Adding or changing a Megatron-Bridge model/provider and needing downstream NeMo-RL compatibility validation; checking non-vanilla Bridge provider paths; testing PEFT/LoRA, checkpoint behavior, non-colocated vLLM refit, or explicitly requested advanced variants through NeMo-RL; 'does this model work in NeMo-RL', 'run NeMo-RL e2e', 'external RL loop validation'.
---

# NeMo-RL E2E Testing

Validate a Megatron-Bridge model or training API change through NeMo-RL's Megatron backend. This catches integration issues that Bridge-only tests miss: NeMo-RL-owned rollout scheduling, reward handling, policy/reference setup, HF import/export through Bridge, optimizer setup, checkpoint ownership, and policy-to-generation weight transfer.

Use this as an external compatibility smoke test after the focused Bridge tests for the model/provider change pass.

This is not a replacement for Bridge model parity tests. A NeMo-RL GRPO or SFT run proves that Bridge can survive an external RL training loop; architecture correctness still comes from Bridge import/export, logits, roundtrip, and model-specific inference tests.

## Scope

Think in coverage levels. Start with Level 0 and add only the levels justified by the change.

| Level | Required when | What it proves |
|---|---|---|
| 0: Megatron policy GRPO smoke | Any new provider or provider config change that claims NeMo-RL compatibility | NeMo-RL can import the local Bridge provider, build a Megatron policy, initialize optimizer/scheduler state, run rollout/ref/logprob wiring, and finish a short GRPO job |
| 1: LoRA/checkpoint variant | Checkpointing, HF export, optimizer state, resume behavior, or a NeMo-RL-supported PEFT path changed | NeMo-RL can save through its checkpoint schedule, resume without losing training state, and, when PEFT is enabled in that NeMo-RL checkout, apply Bridge LoRA hooks |
| 2: Non-colocated vLLM refit | HF export, weight mapping, policy-to-generation refit, delta compression, packed transfer, or vLLM update behavior changed | Bridge-exported weights can be transferred from the Megatron policy worker into separate vLLM generation workers |
| 3: Optional Megatron generation backend | Only when the NeMo-RL checkout still supports `policy.generation.backend=megatron` and the change explicitly targets that path | NeMo-RL can use Megatron for both policy and generation rather than only vLLM generation |
| 4: Parallelism stress | TP/PP/CP/EP, sequence parallel, MoE dispatch, pipeline stage layout, or distributed optimizer behavior changed | Provider settings remain correct under non-trivial Megatron parallel state |
| 5: Architecture-specific e2e | VLM, audio, MoE, MTP/draft models, FP8/QAT/ModelOpt, quantized weights, or custom layers are involved | The architecture-specific runtime path is exercised, not just a text-only dense GRPO smoke |
| 6: Learning signal | Optimizer, scheduler, loss, reward, PEFT trainability, gradient flow, or training stability changed | Metrics move in the expected direction over a short run and do not silently produce zero/NaN/unstable updates |

The default Level 0 target is NeMo-RL's maintained Megatron GRPO functional:

```bash
uv run bash tests/functional/grpo_megatron.sh
```

This is intentionally small. It exercises NeMo-RL's external RL loop without making Megatron-Bridge own rollout scheduling, rewards, checkpoint cadence, or trainer state.

Level 0 is not a convergence test. It only proves the job can complete a small number of updates. Use Level 6 when the question is whether the model actually learns under NeMo-RL.

## Repos

Use explicit repo variables. Do not rely on an installed `megatron-bridge` wheel; the purpose is to test the current Bridge checkout.

Use the upstream NeMo-RL repository as the default source:

```text
https://github.com/NVIDIA-NeMo/RL
```

If a checkout is not already available, clone it next to the Bridge checkout or into the site's standard workspace:

```bash
git clone https://github.com/NVIDIA-NeMo/RL.git /path/to/nemo-rl
```

```bash
export BRIDGE_REPO=${BRIDGE_REPO:-/path/to/Megatron-Bridge}
export NEMO_RL_REPO=${NEMO_RL_REPO:-/path/to/nemo-rl}
export PYTHONPATH="${BRIDGE_REPO}/src:${BRIDGE_REPO}/3rdparty/Megatron-LM:${NEMO_RL_REPO}:${PYTHONPATH:-}"
```

NeMo-RL checkouts often also contain a vendored Bridge tree under:

```text
3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
```

When testing a local Bridge change, either put the local Bridge checkout ahead of everything else in `PYTHONPATH`, or sync the exact local Bridge changes into that vendored checkout. Do not assume the vendored tree matches the Bridge PR under test.

Before running, record both states:

```bash
git -C "$BRIDGE_REPO" status --short
git -C "$NEMO_RL_REPO" status --short
git -C "$BRIDGE_REPO" rev-parse --short HEAD
git -C "$NEMO_RL_REPO" rev-parse --short HEAD
```

If testing on a remote GPU machine, sync the exact local changes first. Do not reset or overwrite unrelated changes in either tree.

Verify that Python imports the checkout under test:

```bash
python - <<'PY'
import megatron.bridge
print(megatron.bridge.__file__)
PY
```

The printed path must live under `$BRIDGE_REPO/src`, or under the NeMo-RL vendored Bridge checkout only if that vendored checkout was intentionally synced to the Bridge change. If it points at site-packages or an unexpected 3rdparty path, fix `PYTHONPATH` before trusting any result.

## Bridge Checks First

Run focused Bridge tests before the external NeMo-RL e2e. Include any model-specific tests added by the change.

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
- this NeMo-RL external-loop smoke after the above pass

## NeMo-RL Unit Checks

Run the NeMo-RL unit checks that match the surface being exercised. Keep them focused; the e2e is the expensive signal.

```bash
cd "$NEMO_RL_REPO"
uv run pytest -q \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/utils/test_weight_transfer.py
```

For checkpoint changes, add:

```bash
uv run pytest -q \
  tests/unit/utils/test_checkpoint.py \
  tests/unit/utils/test_native_checkpoint.py
```

For vLLM refit or generation-worker changes, add the relevant vLLM unit tests:

```bash
uv run pytest -q \
  tests/unit/models/generation/test_vllm_generation.py \
  tests/unit/models/generation/test_vllm_utils.py
```

## Model Choice

Prefer the smallest public HF checkpoint that uses the changed provider family. The maintained Megatron GRPO functional uses `Qwen/Qwen2.5-0.5B` because it is small enough for a 2-GPU smoke and is supported by NeMo-RL's Megatron path.

If there is no small public checkpoint for the new architecture, use the closest NeMo-RL recipe that constructs the model with a minimal config or small local checkpoint, and report that the run validates construction/training mechanics rather than pretrained weight compatibility.

For VLM or audio models, a text-only GRPO smoke is not enough. Pair the Level 0 policy smoke with the relevant NeMo-RL VLM/audio functional, for example:

```bash
uv run bash tests/functional/vlm_grpo.sh
uv run bash tests/functional/audio_grpo_megatron.sh
```

For MoE models, Level 0 with trivial expert parallelism catches many provider issues, but it does not stress expert routing. Add a Level 4 run with expert parallelism when the change touches expert layout, dispatcher config, router behavior, or expert tensor parallelism.

For MTP/draft models, use an Eagle/MTP-specific functional:

```bash
uv run bash tests/functional/grpo_megatron_eagle3_online.sh
```

For FP8/QAT/ModelOpt or quantized checkpoint support, use the closest recipe or functional that explicitly enables the feature. Do not claim the generic GRPO smoke validated quantization unless the config turns it on.

## Environment Setup

Use the NeMo-RL development environment or the site-approved NeMo-RL container. Make caches explicit on shared clusters:

```bash
export HF_HOME=${HF_HOME:-/scratch/$USER/nemo_rl_hf}
export HF_HUB_CACHE=$HF_HOME/hub
export NEMO_RL_HOME=${NEMO_RL_HOME:-$NEMO_RL_REPO}
export PYTHONPATH="${BRIDGE_REPO}/src:${BRIDGE_REPO}/3rdparty/Megatron-LM:${NEMO_RL_REPO}:${PYTHONPATH:-}"
```

If the container has a dependency fingerprint mismatch, note it in the report. Prefer rebuilding the container or virtualenv when possible; use environment overrides only as test-environment evidence, not repository changes.

If model downloads fail with `No space left on device`, move `HF_HOME`, `HF_HUB_CACHE`, and any local `MODEL_PATH` to a larger shared or node-local path.

If Hugging Face API calls fail with rate limits after the model is already cached, point both the model and tokenizer at the local snapshot and run offline:

```bash
export MODEL_PATH=/scratch/$USER/hf/hub/models--<org>--<model>/snapshots/<snapshot-sha>
export HF_HOME=/scratch/$USER/hf
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Then pass both overrides to NeMo-RL:

```bash
policy.model_name="$MODEL_PATH" \
policy.tokenizer.name="$MODEL_PATH"
```

Before trusting the snapshot, verify it loads locally:

```bash
uv run python - <<'PY'
from transformers import AutoConfig, AutoTokenizer

path = "<local-snapshot-path>"
config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
print(type(config).__name__, getattr(config, "model_type", None), type(tokenizer).__name__, tokenizer.vocab_size)
PY
```

## Minimal NeMo-RL Run

Use NeMo-RL's maintained functional wrapper for the default smoke:

```bash
cd "$NEMO_RL_REPO"
ray stop --force || true

export PYTHONPATH="${BRIDGE_REPO}/src:${BRIDGE_REPO}/3rdparty/Megatron-LM:${NEMO_RL_REPO}:${PYTHONPATH:-}"

uv run bash tests/functional/grpo_megatron.sh
```

The wrapper writes:

```text
tests/functional/grpo_megatron/run.log
tests/functional/grpo_megatron/metrics.json
```

Capture the exact command and keep the log path. Prefer a saved log over a pasted terminal excerpt in PR descriptions.

If the test needs a different provider or model, pass Hydra overrides through the wrapper:

```bash
uv run bash tests/functional/grpo_megatron.sh \
  policy.model_name=<small-compatible-hf-model> \
  policy.megatron_cfg.converter_type=<BridgeConverterType>
```

Keep the first smoke small. Increase model size or parallelism only after a small run proves the basic path works.

## LoRA And Checkpoint Coverage

Use Level 1 when the change touches checkpoint save/load, HF export, optimizer state, resume behavior, or a NeMo-RL PEFT path that is known to work in the checkout being tested.

NeMo-RL PEFT support is backend- and revision-dependent. Do not block a provider-only compatibility smoke solely on a known-broken or unsupported NeMo-RL PEFT path. In that case, record Level 1 PEFT as not applicable or blocked by NeMo-RL, keep the Level 0 GRPO smoke as the required downstream signal, and cover Bridge PEFT behavior with focused Bridge tests.

LoRA + checkpoint save smoke, when the NeMo-RL checkout supports this path:

```bash
uv run bash tests/functional/grpo_megatron_lora.sh
```

SFT resume parity across dtensor and Megatron policy paths:

```bash
uv run bash tests/functional/sft_resume_diamond.sh
```

The LoRA functional intentionally saves checkpoints. Remove stale checkpoint outputs between unrelated experiments, but keep them while validating resume behavior.

Do not claim PEFT coverage from `grpo_megatron.sh`; use the LoRA functional or an equivalent Hydra override with `policy.megatron_cfg.peft.enabled=true`.

## Non-Colocated vLLM Refit

Use Level 2 when the change touches Bridge HF export, parameter mapping, NeMo-RL weight refit, packed tensor transfer, vLLM loading, delta compression, or policy/generation worker synchronization.

Small 2-GPU non-colocated smoke with the Megatron policy backend:

```bash
cd "$NEMO_RL_REPO"
uv run coverage run -a --data-file=tests/.coverage --source=nemo_rl \
  examples/run_grpo.py \
  --config examples/configs/grpo_math_1B_megatron.yaml \
  policy.model_name=Qwen/Qwen2.5-0.5B \
  grpo.num_prompts_per_step=2 \
  grpo.num_generations_per_prompt=4 \
  policy.train_global_batch_size=4 \
  policy.train_micro_batch_size=1 \
  policy.logprob_batch_size=4 \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.gpus_per_node=1 \
  policy.generation.vllm_cfg.async_engine=true \
  cluster.gpus_per_node=2 \
  grpo.max_num_steps=2 \
  logger.tensorboard_enabled=true \
  logger.log_dir=tests/functional/grpo_megatron_non_colocated/logs \
  logger.wandb_enabled=false \
  checkpointing.enabled=false
```

After the run, dump metrics:

```bash
uv run tests/json_dump_tb_logs.py \
  tests/functional/grpo_megatron_non_colocated/logs \
  --output_path tests/functional/grpo_megatron_non_colocated/metrics.json
```

Metric assertion helpers differ across NeMo-RL revisions. Inspect `tests/check_metrics.py` or the maintained functional wrapper before assuming an interface. Some checkouts expect positional expressions:

```bash
uv run tests/check_metrics.py tests/functional/grpo_megatron_non_colocated/metrics.json \
  'max(data["train/token_mult_prob_error"]) < 1.05' \
  'min(data["train/probs_ratio_clamped_min"]) > 0.79' \
  'max(data["train/probs_ratio_clamped_max"]) < 1.21'
```

For delta-compression testing, add these overrides:

```bash
policy.generation.delta_compression.enabled=true \
policy.generation.delta_compression.dtype=bfloat16 \
policy.generation.delta_compression.transport=sparse_indices \
policy.generation.delta_compression.full_sync_interval=20 \
policy.generation.delta_compression.sparse_bucket_size_bytes=5368709120 \
policy.generation.delta_compression.delta_load_batch_size_bytes=536870912
```

Report weight-transfer timing metrics when available, especially:

- `timing/train/prepare_for_generation/total`
- `timing/train/prepare_for_generation/transfer_and_update_weights`
- `timing/train/prepare_for_generation/weight_transfer/producer/collect_tensors`
- `timing/train/prepare_for_generation/weight_transfer/producer/sparse_encode`
- `timing/train/prepare_for_generation/weight_transfer/producer/sparse_nonzero`
- `timing/train/prepare_for_generation/weight_transfer/consumer/decode_sparse`
- `timing/train/prepare_for_generation/weight_transfer/consumer/load_delta`

If the payload broadcast time is tiny but sparse encode/decode dominates, report that boundary clearly. It is a weight-preparation bottleneck, not a NCCL broadcast bottleneck.

## Megatron Generation Backend

Use Level 3 only when the NeMo-RL checkout under test supports the Megatron generation backend and the Bridge change explicitly affects that downstream path. Do not require this for normal provider compatibility, HF import/export, vLLM-backed generation, or generic Bridge inference tests.

```bash
uv run bash tests/functional/grpo_megatron_generation.sh
```

This exercises `policy.generation.backend=megatron`, so it validates NeMo-RL's Megatron generation construction and runtime behavior more directly than the default vLLM-backed GRPO functional.

Some NeMo-RL revisions declare `mcore` and `vllm` extras as mutually incompatible. In that environment, a vLLM-backed Level 0 run may be blocked even though the Megatron policy path is testable. Use `policy.generation.backend=megatron` for a Megatron-only smoke, record vLLM as skipped or blocked, and do not claim non-colocated vLLM refit coverage.

## Parallelism Stress

Use Level 4 when provider finalization, model-parallel settings, sequence parallel, context parallel, MoE dispatch, pipeline layout, or distributed optimizer behavior changed.

Start from a maintained recipe that already matches the intended GPU count. For example, use one of the recipe configs under:

```text
examples/configs/recipes/llm/*megatron*.yaml
examples/configs/recipes/llm/performance/*megatron*.yaml
examples/configs/recipes/vlm/*megatron*.yaml
```

For a small manual stress variant, override the Megatron sizes explicitly:

```bash
uv run bash tests/functional/grpo_megatron.sh \
  policy.megatron_cfg.tensor_model_parallel_size=2 \
  policy.megatron_cfg.pipeline_model_parallel_size=1 \
  policy.megatron_cfg.context_parallel_size=1 \
  policy.megatron_cfg.sequence_parallel=false \
  cluster.gpus_per_node=2
```

For MoE, use a MoE recipe and set expert parallelism only when the model and GPU count support it:

```bash
policy.megatron_cfg.expert_model_parallel_size=2 \
policy.megatron_cfg.expert_tensor_parallel_size=1
```

Keep these as follow-up runs. Do not make them the first debugging surface for a new provider.

## Learning Signal

Use Level 6 only when the change affects trainability or when downstream validation explicitly asks for learning behavior. Do not require it for every provider-only PR; RL learning is slower, noisier, and more environment-dependent than compatibility smoke tests.

The goal is a short learning-signal run, not a benchmark. Prefer a small model, fixed data, fixed seed when available, and enough steps to observe non-random metric movement:

```bash
uv run bash tests/functional/grpo_megatron_lora.sh \
  grpo.max_num_steps=20 \
  data.shuffle=false \
  checkpointing.enabled=false
```

Acceptable learning-signal evidence depends on the task, but the report should include at least:

- no NaNs or infs in loss, reward, KL, entropy, grad norm, or logprob metrics
- nonzero trainable parameter count when PEFT is enabled
- actor losses and reward-related metrics logged for multiple steps
- validation or reward trend compared against the starting point or a known-good baseline
- no repeated zero gradients, frozen LoRA adapters, or constant logprobs unless expected

Do not call a 20-step run "converged" in the benchmark sense. Call it "learning-signal passed" unless it reaches a pre-agreed metric threshold.

## Slurm Or Container Runs

Use the cluster's standard NeMo-RL container and mount both checkouts into the container. Keep setup and the actual run in the same container step when using node-local paths such as `/tmp`; node-local model caches and ad-hoc installs disappear when a fresh container step starts.

If the home filesystem is full or Megatron-Core tries to build helper extensions into a read-only/full checkout, copy the MCore submodule to node-local storage and put that copy on `PYTHONPATH` instead of editing `3rdparty/Megatron-LM/`:

```bash
export MCORE_REPO=${MCORE_REPO:-/tmp/$USER/Megatron-LM}
if [[ ! -d "$MCORE_REPO/.git" ]]; then
  cp -a "$BRIDGE_REPO/3rdparty/Megatron-LM" "$MCORE_REPO"
fi

EXT_SUFFIX=$(uv run python - <<'PY'
import sysconfig

print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
PY
)
make -C "$MCORE_REPO/megatron/core/datasets" LIBEXT="$EXT_SUFFIX"
export PYTHONPATH="${BRIDGE_REPO}/src:${MCORE_REPO}:${NEMO_RL_REPO}:${PYTHONPATH:-}"
```

Overriding `LIBEXT` avoids a suffixless `helpers_cpp` binary on containers where `python3-config` is absent from `PATH`. Verify the built file is named like `helpers_cpp.cpython-<ver>-<platform>.so` before launching a long run.

For NeMo-RL multi-node jobs, prefer NeMo-RL's own `ray.sub` launcher when it is available. It starts the Ray head and worker nodes under Slurm, mounts the requested container/filesystems, and executes `COMMAND` from the NeMo-RL root. Launch it from `$NEMO_RL_REPO`, not from the Bridge checkout:

```bash
cd "$NEMO_RL_REPO"

COMMAND="uv run ./examples/run_grpo.py \
  --config examples/configs/grpo_math_1B_megatron.yaml \
  cluster.num_nodes=2 \
  cluster.gpus_per_node=8 \
  logger.log_dir=results/grpo_megatron_2n \
  logger.wandb_enabled=false" \
CONTAINER="$NEMO_RL_IMAGE" \
MOUNTS="$BRIDGE_REPO:$BRIDGE_REPO,$NEMO_RL_REPO:$NEMO_RL_REPO,$HF_HOME:$HF_HOME" \
sbatch \
  --nodes=2 \
  --account=<account> \
  --partition=<partition> \
  --job-name=nemo-rl-bridge-e2e \
  --time=4:00:00 \
  --gres=gpu:8 \
  ray.sub
```

Include the local Bridge checkout in `MOUNTS` and in `PYTHONPATH` inside `COMMAND` when the container does not already see the same path. If using a vendored Bridge under `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`, sync the exact Bridge changes there instead and report that path.

Use a direct `srun` only when `ray.sub` is unavailable, stale for the target cluster, or when debugging the container/Slurm layer itself. Keep paths generic in scripts committed to Megatron-Bridge:

```bash
srun <site-specific-slurm-options> \
  --container-image="${NEMO_RL_IMAGE}" \
  --container-mounts="${BRIDGE_REPO}:/workspace/Megatron-Bridge,${NEMO_RL_REPO}:/workspace/nemo-rl,<data-root>:<data-root>" \
  --container-workdir=/workspace/nemo-rl \
  bash -lc '
    export BRIDGE_REPO=/workspace/Megatron-Bridge
    export NEMO_RL_REPO=/workspace/nemo-rl
    export PYTHONPATH=$BRIDGE_REPO/src:$BRIDGE_REPO/3rdparty/Megatron-LM:$NEMO_RL_REPO
    ray stop --force || true
    uv run bash tests/functional/grpo_megatron.sh
  '
```

If an attach helper enters a container that no longer sees the expected checkouts or log directory, treat that helper as stale. Start a fresh `srun` step against the existing allocation with explicit `--container-image`, `--container-mounts`, and `--container-workdir`.

Attach helpers that use `--no-container-mount-home` can enter a minimal `/home/$USER` in follow-up steps even when the original run saw the real checkout. Keep metric dumping and assertions in the same container step as the run when possible. If a follow-up step must inspect compute-local artifacts, use paths under the node-local run directory and do not assume `$NEMO_RL_REPO` is visible.

For general Slurm debugging and multi-node patterns, read @skills/multi-node-slurm/SKILL.md.

## Pass Criteria

A useful pass has all of the following:

- Focused Bridge tests pass for provider/config/mapping behavior.
- NeMo-RL imports the intended Bridge checkout, verified by `megatron.bridge.__file__`.
- The NeMo-RL config has `policy.megatron_cfg.enabled=true` for Megatron policy validation.
- The run reaches the requested step count and writes `metrics.json`.
- `tests/check_metrics.py` passes when the maintained functional includes metric assertions.
- No exception occurs during Bridge provider setup, HF import/export, enabled PEFT/LoRA wrapping, Megatron initialization, optimizer setup, checkpoint manager setup, weight transfer, or the training step.

Ray shutdown warnings, Python resource-tracker warnings, or post-completion process-group warnings can be acceptable if the training step completed, metrics were written, and the process exits successfully. Mention them as residual log noise.

Do not claim full model e2e if the run used a dummy config, text-only data for a VLM/audio model, trivial expert parallelism for an expert-parallel change, or disabled save/resume for a checkpointing change. Call it the exact level that passed.

Do not claim convergence from Level 0. Claim learning signal only from Level 6, and distinguish "learning signal" from benchmark convergence in the report.

## Failure Triage

If model construction fails, verify that NeMo-RL is importing the Bridge checkout under test and that `policy.megatron_cfg.converter_type` matches the provider.

If the config silently uses dtensor instead of Megatron, set `policy.dtensor_cfg.enabled=false` and `policy.megatron_cfg.enabled=true`, or use `grpo_megatron.sh`.

If LoRA fails, check NeMo-RL PEFT config names and Bridge target module names. Reproduce with `grpo_megatron_lora.sh` before adding larger model or parallelism changes.

If checkpoint save/load fails, first rerun with `checkpointing.enabled=false` to separate model construction from checkpoint behavior, then use `sft_resume_diamond.sh` for resume parity.

If non-colocated refit fails, separate the boundary:

- producer export and metadata preparation on the policy worker
- payload packing/broadcast
- consumer decode and model loading on the generation worker
- vLLM-specific weight-loader behavior

If NeMo-RL rejects TP >= 4 with the batch-variant accuracy guard, prefer TP 1 or 2 for the smoke, or set `policy.train_micro_batch_size` and `policy.logprob_batch_size` equal. Do not bypass with `NRL_IGNORE_TP_ACCURACY_CHECK=1` for pass/fail evidence unless the user explicitly wants an unsupported diagnostic run.

If Megatron generation fails during `cuda graph warmup` with `CUDA error: an illegal memory access was encountered`, rerun the same config with:

```bash
policy.generation.mcore_generation_config.num_cuda_graphs=null \
policy.generation.mcore_generation_config.use_cuda_graphs_for_non_decode_steps=false
```

If the no-graph run passes, report the original result as a Megatron generation CUDA-graph failure and the no-graph run as a reduced-optimization pass. Keep both logs.

If the run reaches the requested step count but `tests/check_metrics.py` fails on `train/token_mult_prob_error`, treat it as a real metric failure, not a harness failure. NeMo-RL computes this metric from `exp(abs(generation_logprobs - prev_logprobs))`; huge values mean the generation backend logprobs disagree with the policy logprobs recomputed for training. Isolate by retrying with simpler parallelism or kernels such as `policy.megatron_cfg.sequence_parallel=false`, `policy.megatron_cfg.apply_rope_fusion=false`, shorter sequence lengths, or vLLM generation when available. Do not relax the metric threshold or use sequence masking to claim a pass; run Bridge logits/import/export parity to localize whether the mismatch is in Bridge conversion, Megatron generation logprob collection, or NeMo-RL recomputation.

If model download fails, move HF caches to a larger path and rerun with explicit cache settings.

If Hugging Face returns `429 Too Many Requests` during tokenizer/config setup, first check whether the snapshot already exists under `$HF_HUB_CACHE`. If it does, switch `policy.model_name` and `policy.tokenizer.name` to the local snapshot path and enable offline mode. This is an environment failure unless the local snapshot cannot load with `local_files_only=True`.

If `helpers_cpp` fails to link with `No space left on device`, or if logs show `make: python3-config: No such file or directory`, rebuild the helper in a node-local copy of Megatron-LM with `LIBEXT` set from `sysconfig.get_config_var("EXT_SUFFIX")`. Do not patch files under `3rdparty/Megatron-LM/` in the Bridge checkout.

If a baseline fails before model build because of data, Ray, vLLM, package setup, or container mismatch, fix the environment first and do not report it as a Bridge provider failure.

## Summary Format

End every run with a short user-facing summary that answers "Did the requested deliverables pass?" before adding details. Use `Pass`, `Fail`, `Skipped`, or `Blocked` for each deliverable, and do not report an overall `Pass` unless the pass criteria for the requested coverage level were met.

```text
Result: <Pass/Fail/Blocked> - <one sentence stating what was validated>
Requested coverage: <Level 0/1/2/3/4/5/6 and requested variants>
Model: <policy.model_name or local model path>

Deliverables:
- Bridge-side checks: <Pass/Fail/Skipped> - <test command or skipped reason>
- Local Bridge import in NeMo-RL: <Pass/Fail> - <megatron.bridge.__file__ path>
- NeMo-RL Megatron policy run: <Pass/Fail/Skipped> - <GRPO Megatron or requested variant>
- Requested variants: <Pass/Fail/Skipped/Not requested> - <LoRA/checkpoint, non-colocated vLLM refit, Megatron generation, parallelism stress, architecture-specific, or learning-signal>
- Metrics/log capture: <Pass/Fail> - <log path, metrics path, and metric assertion status>

Evidence:
- Bridge repo: <commit> plus dirty files
- NeMo-RL repo: <commit> plus dirty files
- Command: <exact command or script path>
- Key lines: <policy.megatron_cfg.enabled=true, step completion, metrics.json creation, tests/check_metrics.py result, or the first relevant error>

Limitations:
- <dummy model, skipped save/resume, text-only VLM/audio smoke, trivial EP, no learning-signal claim, known shutdown warnings, etc.>

Follow-ups:
- <needed rerun, environment fix, provider fix, NeMo-RL issue, or "none">
```

If the job is blocked before Bridge model/provider construction by data, Ray, vLLM, dependency, disk, container, or cluster setup, mark the overall result as `Blocked`, not `Fail`, and state that it is not evidence against the Bridge provider.

If any requested deliverable was not run, mark it `Skipped` or `Not requested` with the reason. Do not leave it implicit in the limitations.
