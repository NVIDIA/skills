---
name: multi-node-slurm
description: Convert single-node scripts to multi-node Slurm sbatch jobs and debug common multi-node failures. Covers srun-native vs uv run torch.distributed approaches, container setup, NCCL timeouts, OOM sizing for MoE models, and interactive allocation. Use when creating Slurm scripts, scaling to multi-node, or debugging multi-node job failures.
---

# Multi-Node Slurm

Convert single-node `uv run python -m torch.distributed.run` commands into multi-node Slurm sbatch scripts with Enroot container support, and debug common multi-node failures.

## Two Approaches: srun-native vs uv run torch.distributed

| Approach | `ntasks-per-node` | Process spawning | Best for |
|---|---|---|---|
| **srun-native** (preferred) | 8 | Slurm spawns 8 tasks/node | Conversion, inference, Bridge scripts |
| **uv run torch.distributed** (legacy) | 1 | `uv run python -m torch.distributed.run` spawns 8 procs/node | MLM pretrain_gpt.py |

**Prefer srun-native** — simpler, avoids shell escaping issues with TRAIN_CMD. Megatron Bridge auto-derives `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT` from SLURM env vars (`SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_LOCALID`, `SLURM_NODELIST`) via `common_utils.py` helpers called during `initialize.py` distributed init, so you never need to set them manually.

## Cluster Environment

### Container

```bash
CONTAINER_IMAGE="<PATH_TO_YOUR_CONTAINER>.sqsh"
CONTAINER_MOUNTS="<SHARED_FS>:<SHARED_FS>,<PATH_TO_MEGATRON_BRIDGE>:/opt/Megatron-Bridge,<PATH_TO_DATA>:/opt/data"
```

### Standard Paths

```bash
WORKDIR="/opt/Megatron-Bridge"
DATA_PATH="<PATH_TO_PREPROCESSED_DATA>/dclm_01_01_text_document"
```

### Tokens / Caches

```bash
export GH_TOKEN=<YOUR_GITHUB_TOKEN>
export HF_TOKEN=<YOUR_HF_TOKEN>
export HF_HOME=<SHARED_FS>/HF_HOME
export UV_CACHE_DIR="<SHARED_FS>/uv_cache"
export NEMO_HOME="<SHARED_FS>/cache/nemo"
```

**Important**: `NEMO_HOME` must point to a shared filesystem (e.g. Lustre) for multi-node SFT/PEFT jobs.
The default (`/root/.cache/nemo`) is container-local and not shared across nodes.
Without this, packed-sequence data files prepared on node 0 are invisible to other
nodes, causing `TypeError: 'NoneType' object is not an iterator`.

### Log Directory

```text
<SHARED_FS>/logs/<job_name>_<suffix>
```

## srun-native Approach (Preferred)

Slurm spawns all processes directly. No `torch.distributed.run`, no TRAIN_CMD escaping.

### SBATCH Headers

```bash
#SBATCH --job-name=<model>-<task>
#SBATCH --nodes=<NNODES>
#SBATCH --ntasks-per-node=8          # Slurm spawns 8 tasks per node
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=batch
#SBATCH --output=<SHARED_FS>/logs/<job_name>_%j.log
#SBATCH --exclusive
```

### Build and Launch

Two-phase srun: first a single-process srun to populate the uv cache, then the full multi-node srun.

```bash
# Env exports at sbatch level (before srun)
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# Phase 1: Single-process uv sync to build/populate the shared cache
srun --mpi=pmix -N 1 --ntasks=1 \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --no-container-mount-home \
  bash -c "cd $WORKDIR && uv sync"

# Phase 2: Full multi-node run (uv sync is a fast no-op since cache is warm)
srun --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --no-container-mount-home \
  bash -c "cd $WORKDIR && uv sync && uv run --no-sync python <script.py> <args>"
```

### srun-native Key Points

- Phase 1 runs `uv sync` once on a single node/process, building all wheels into the shared cache on Lustre
- Phase 2's `uv sync` is a fast no-op (everything is cached) — safe to run on all ranks without sleep guards
- `initialize.py` + `common_utils.py` auto-set `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT` from SLURM env vars
- Env vars like `HF_TOKEN`, `HF_HOME`, `UV_CACHE_DIR` exported at sbatch level are inherited by srun tasks
- Reference: `examples/models/vlm/glm_45v/slurm_sft.sh`, `examples/models/minimax_m2/slurm_conversion.sh`

---

## uv run torch.distributed Approach (Legacy)

Use when the script requires `torch.distributed.run` (e.g., MLM pretrain_gpt.py) or when Bridge's `initialize.py` is not in the call path.

### 1. Add SBATCH Headers

```bash
#SBATCH --job-name=<model>-<framework>
#SBATCH --nodes=<NNODES>
#SBATCH --ntasks-per-node=1          # ALWAYS 1 — torchrun handles per-node spawning
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=batch
#SBATCH --output=<SHARED_FS>/logs/<job_name>_%j.log
#SBATCH --exclusive
```

**Critical**: `--ntasks-per-node=1`, NOT 8. `uv run python -m torch.distributed.run --nproc_per_node=8` spawns 8 processes per node. Using `ntasks-per-node=8` causes EADDRINUSE port collisions (8 tasks x 8 procs = 64 per node).

### 2. Convert to Multi-Node

Replace single-node:

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
  <script> <args>
```

With multi-node (inside `TRAIN_CMD` string):

```bash
uv run python -m torch.distributed.run \
  --nproc_per_node=8 \
  --nnodes=\${SLURM_JOB_NUM_NODES} \
  --node_rank=\${SLURM_NODEID} \
  <script> <args>
```

`MASTER_ADDR` and `MASTER_PORT` are auto-derived from SLURM env vars by `initialize.py` / `common_utils.py` — no need to set them.

### 3. Wrap in TRAIN_CMD + two-phase srun

Use the same two-phase pattern: first a single-process srun to warm the uv cache, then the full run.

**Environment exports go inside TRAIN_CMD** (they must be set inside the container):

```bash
TRAIN_CMD="
export CUDA_DEVICE_MAX_CONNECTIONS=1 && \
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
export NCCL_NVLS_ENABLE=0 && \
export GH_TOKEN=$GH_TOKEN && \
export HF_TOKEN=$HF_TOKEN && \
export HF_HOME=$HF_HOME && \
export UV_CACHE_DIR=$UV_CACHE_DIR && \
wandb login \$WANDB_API_KEY && \
mkdir -p $LOGDIR && \
cd $WORKDIR && \
uv sync && \
<training command here>
"
```

### 4. Launch (two-phase)

```bash
# Phase 1: Single-process uv sync to build/populate the shared cache
srun --mpi=pmix -N 1 --ntasks=1 \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --no-container-mount-home \
  bash -c "cd $WORKDIR && uv sync"

# Phase 2: Full multi-node run (uv sync in TRAIN_CMD is a fast no-op)
srun --mpi=pmix --no-kill \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --no-container-mount-home \
  bash -c "$TRAIN_CMD" 2>&1 | tee "$LOGDIR/<prefix>_${SLURM_JOB_ID}.log"
```

### 5. (Optional) Add Loss Extraction Footer

```bash
echo "======================================"
echo "Done. Losses:"
echo "======================================"
grep -E "iteration\s+" "$LOGDIR/<prefix>_${SLURM_JOB_ID}.log" | grep -iE "lm loss|reduced_train_loss" | head -25
```

---

## Interactive GPU Allocation (`salloc` + `srun`)

For ad-hoc testing (inference, conversion debugging), always follow these 3 steps:

### Step 1: Allocate the node

```bash
salloc --account <YOUR_ACCOUNT> -N 1 \
  -J <YOUR_ACCOUNT>-debug \
  -p interactive --gpus-per-node=8 -t 240
```

### Step 2: Launch container shell

```bash
srun --mpi=pmix --no-kill \
  --container-image $CONTAINER_IMAGE \
  --container-mounts $CONTAINER_MOUNTS \
  --account <YOUR_ACCOUNT> -N 1 \
  -J <YOUR_ACCOUNT>-debug \
  --no-container-mount-home --gpus-per-node=8 \
  -p interactive --pty bash
```

### Step 3: Set up environment inside container

```bash
export GH_TOKEN=<YOUR_GITHUB_TOKEN>
wandb login <YOUR_WANDB_KEY>
export HF_TOKEN=<YOUR_HF_TOKEN>
export HF_HOME=<SHARED_FS>/HF_HOME
export UV_CACHE_DIR="<SHARED_FS>/uv_cache"
export NEMO_HOME="<SHARED_FS>/cache/nemo"
uv sync
```

Then run commands with `uv run` (uses the synced virtualenv):

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
  examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path <org>/<model> --prompt "What is AI?" --max_new_tokens 50 --ep 8
```

**Pitfalls with interactive allocation:**

| Error | Cause | Fix |
|---|---|---|
| `Cannot find GPU specification` | Missing `--gpus-per-node` | Always include `--gpus-per-node=8` in both `salloc` and `srun` |
| `invalid partition specified: pool0` | Wrong partition name | Use `interactive` for interactive, `batch` for sbatch. Check: `sinfo --summarize` |
| `Invalid account or account/partition combination` | Partition not available for account | Check combos: `sacctmgr -nP show assoc where user=$USER format=account,partition` |
| `Unable to create step for job... Requested node configuration is not available` | `-w <node>` conflicts with allocation | Remove `-w` flag — HF cache is on shared filesystem, accessible from any node |
| `uv: command not found` inside container | Container doesn't have `uv` pre-installed | Use a container with `uv` pre-installed, or `pip install uv` |
| `No space left on device` during `uv` or `pip` | Container's `/root/.cache/` is full | Redirect: `export UV_CACHE_DIR=<SHARED_FS>/uv_cache` |
| `ModuleNotFoundError: No module named 'megatron.core.activations'` | Container's pre-installed megatron-core conflicts with local `3rdparty/Megatron-LM` | Install local: `pip install -e 3rdparty/Megatron-LM --no-deps --no-build-isolation` |

---

## Debugging Multi-Node Failures

### Quick Diagnosis

Check the log for these patterns (in order):

```bash
# 1. Find the actual error (filter noise)
grep -a 'Error\|OOM\|CUDA out of memory\|FAILED\|Killed' job.log \
  | grep -v 'UserWarning\|AllocatorConfig\|transformer_engine\|frame\|srun: error'

# 2. Check which rank crashed first
grep -a 'Failures:' -A 20 job.log | head -25

# 3. Check for NCCL timeout
grep -a 'ncclUniqueId\|timeout\|crash on rank 0' job.log | head -5
```

### Debugging Checklist

When a multi-node job fails:

1. **Check exit code**: 1 = Python error, 9 = OOM killed, 143 = SIGTERM (timeout or cascade)
2. **Find first failure**: Which task/node crashed first? Others get SIGTERM (143) as cascade
3. **grep the actual error**: Filter out UserWarnings, NCCL frame dumps
4. **Check rank 0 specifically**: Most save/export errors happen on rank 0
5. **Verify EP sizing**: For MoE models, ensure `num_experts / EP` fits in GPU memory with headroom
6. **Try interactive first**: Use `salloc -N 2 -p interactive` to iterate faster than sbatch queue

### NCCL Timeout at `dist.barrier()` — "crash on rank 0"

**Symptom**: All ranks on node 2+ show:
```text
[rank8] is setting up NCCL communicator and retrieving ncclUniqueId from [0]
... wait timeout after 600000ms
This may indicate a possible application crash on rank 0
```

**Root causes** (check in order):

| Cause | How to verify | Fix |
|---|---|---|
| `save_artifacts` hangs on rank 0 | Error is in `save_hf_weights` → `dist.barrier()` | Increase timeout: `init_process_group("nccl", timeout=timedelta(minutes=60))` |
| `ImportError` in custom model code | `grep ImportError job.log` | Catch `ImportError` in `save_artifacts` (see below) |
| Rank 0 OOM during export | `grep 'OutOfMemory' job.log` | Increase EP or nodes |
| Network issue between nodes | Error only on cross-node ranks | Check `sinfo`, try different nodes |

**The `save_artifacts` problem**: When `trust_remote_code=True`, rank 0 runs `save_artifacts()` (downloads tokenizer, config, custom modeling code) while all other ranks skip directly to `dist.barrier()`. If `save_artifacts` is slow or crashes, other ranks timeout.

**Fix for ImportError in save_artifacts** (`hf_pretrained/base.py`):
```python
# Change:
except OSError:
    pass
# To:
except (OSError, ImportError):
    pass
```

### OOM for MoE Models

**Symptom**: `torch.OutOfMemoryError: CUDA out of memory` during model loading or forward pass.

**Key insight**: TP does NOT reduce expert memory. Only EP splits experts across GPUs.

**Sizing formula**:
```text
experts_per_gpu = num_experts / EP
expert_memory_gb ≈ experts_per_gpu * expert_params * 2 / 1e9  (bf16)
total_per_gpu ≈ expert_memory_gb + attention_memory_gb + kv_cache_gb
```

**MiniMax-M2 example** (256 experts, ~230GB fp8 → ~460GB bf16):

| Config | Nodes | GPUs | Experts/GPU | Result |
|---|---|---|---|---|
| TP=2, EP=4 | 1 | 8 | 64 | OOM (too many experts) |
| TP=2, EP=8 | 2 | 16 | 32 | Works for roundtrip (weight-only), OOM for inference |
| TP=1, EP=16 | 2 | 16 | 16 | Works for inference |
| TP=2, EP=32 | 8 | 64 | 8 | Comfortable for training |

**Rules of thumb**:
- Roundtrip (weight-only): can use more experts per GPU (~60GB model params OK)
- Inference (forward pass + KV cache): needs headroom (~40GB model params max)
- Training (activations + optimizer): needs even more headroom (~30GB model params max)

### `ModuleNotFoundError: No module named 'megatron.core.tensor_parallel'`

**Cause**: Container's pre-installed megatron-core conflicts with local `3rdparty/Megatron-LM`.

**Fix**: Add `uv sync` before running:
```bash
CMD="if [ \"\$SLURM_LOCALID\" -eq 0 ]; then uv sync; else sleep 10; fi && "
CMD="${CMD}uv run --no-sync python <script> <args>"
```

### FP8 Weight Mismatch in Roundtrip

**Symptom**: Roundtrip completes but shows ❌ for all expert weights and raises `ValueError: Weight mismatch detected`.

**Cause**: Original HF weights are FP8, Megatron stores in BF16. Exported weights are BF16. Comparison against original FP8 exceeds `atol=1e-1`.

**This is expected for FP8 models.** The conversion is correct; the comparison tolerance is insufficient for the FP8→BF16 precision gap.

### `WORLD_SIZE` Not Set with srun

**Symptom**: Script exits with "must be launched with torchrun".

**Cause**: Scripts check `os.environ.get("WORLD_SIZE")` which torchrun sets but srun doesn't.

**Fix**: Also check `SLURM_NTASKS`:
```python
if os.environ.get("WORLD_SIZE") is None and os.environ.get("SLURM_NTASKS") is None:
    sys.exit(1)
```

Bridge's `common_utils.py` helpers (called by `initialize.py`) populate env vars from SLURM:
```python
if "RANK" not in os.environ:
    os.environ["RANK"] = str(get_rank_safe())          # uses SLURM_PROCID
if "WORLD_SIZE" not in os.environ:
    os.environ["WORLD_SIZE"] = str(get_world_size_safe())  # uses SLURM_NTASKS
if "MASTER_ADDR" not in os.environ:
    os.environ["MASTER_ADDR"] = get_master_addr_safe()     # parses SLURM_NODELIST
if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = str(get_master_port_safe()) # derives from SLURM_JOB_ID
```

---

## Key Gotchas

1. **Two-phase srun for `uv sync`**: Run a single-process srun first to warm the cache, then the full multi-node srun. The second `uv sync` is a fast no-op since everything is already cached on the shared filesystem.

2. **`--no-container-mount-home`** is an `srun` flag, NOT an `#SBATCH` directive.

3. **Escaping inside TRAIN_CMD**: Since `TRAIN_CMD` is a double-quoted string, escape inner `$` for Slurm variables that must expand at runtime (not sbatch time):
   - `\${SLURM_PROCID}`, `\${SLURM_JOB_NUM_NODES}`, `\${SLURM_NODEID}`
   - Host-side variables like `$GH_TOKEN`, `$LOGDIR`, `$WORKDIR` expand at sbatch time — no escaping needed.

4. **Bridge `rm -rf nemo_experiments`**: Add before training to avoid stale checkpoint auto-resume.

5. **MLM needs PYTHONPATH**: For pretrain_gpt.py scripts, add inside TRAIN_CMD:
   ```bash
   PYTHONPATH=${WORKDIR}/3rdparty/Megatron-LM:\${PYTHONPATH:-} \
   ```

6. **Node count heuristic**: Total GPUs = `NNODES * 8`. Must satisfy: `TP * PP * EP * DP >= total_GPUs` where `DP = total_GPUs / (TP * PP * EP)`.

7. **`NEMO_HOME` on shared filesystem for multi-node SFT**: The default nemo cache (`/root/.cache/nemo`)
   is container-local. Multi-node SFT with packed sequences prepares `.npy` files on one node
   that are invisible to others. Set `export NEMO_HOME=<SHARED_FS>/cache/nemo` so packed data
   is shared. Without this, ranks on other nodes fail with `TypeError: 'NoneType' object is not an iterator`.

## Full Template

```bash
#!/bin/bash
# ==============================================================================
# <MODEL_NAME> <pretrain|sft> — <Framework: MLM | Megatron Bridge>
#
# Default: TP<X> PP<Y> EP<Z>, NNODES=<N> (<N*8> GPUs), MBS=<M>, GBS=<G>
#
# Usage:
#   sbatch <script_name>.sh
# ==============================================================================

#SBATCH --job-name=<job-name>
#SBATCH --nodes=<NNODES>
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=batch
#SBATCH --output=<SHARED_FS>/logs/<job_name>_%j.log
#SBATCH --exclusive

# ── Container ────────────────────────────────────────────────────────────
CONTAINER_IMAGE="<PATH_TO_YOUR_CONTAINER>.sqsh"
CONTAINER_MOUNTS="<SHARED_FS>:<SHARED_FS>,<PATH_TO_MEGATRON_BRIDGE>:/opt/Megatron-Bridge,<PATH_TO_DATA>:/opt/data"

# ── Paths ────────────────────────────────────────────────────────────────
WORKDIR="/opt/Megatron-Bridge"
LOGDIR="<SHARED_FS>/logs/<logdir_name>"
DATA_PATH="<PATH_TO_PREPROCESSED_DATA>/dclm_01_01_text_document"

# ── Parallelism ──────────────────────────────────────────────────────────
TP=1; PP=1; EP=1

# ── Training ─────────────────────────────────────────────────────────────
MBS=1; GBS=256
SEQ=4096
SEED=1234
TRAIN_ITERS=20

# ── Tokens / Caches ──────────────────────────────────────────────────────
export GH_TOKEN=<YOUR_GITHUB_TOKEN>
export HF_TOKEN=<YOUR_HF_TOKEN>
export HF_HOME=<SHARED_FS>/HF_HOME
export UV_CACHE_DIR="<SHARED_FS>/uv_cache"
export NEMO_HOME="<SHARED_FS>/cache/nemo"

# ── Build training command ───────────────────────────────────────────────
TRAIN_CMD="
export CUDA_DEVICE_MAX_CONNECTIONS=1 && \
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
export NCCL_NVLS_ENABLE=0 && \
export GH_TOKEN=$GH_TOKEN && \
export HF_TOKEN=$HF_TOKEN && \
export HF_HOME=$HF_HOME && \
export UV_CACHE_DIR=$UV_CACHE_DIR && \
export NEMO_HOME=$NEMO_HOME && \
wandb login \$WANDB_API_KEY && \
mkdir -p $LOGDIR && \
cd $WORKDIR && \
uv sync && \
<TRAINING_COMMAND_HERE>
"

echo \"======================================\"
echo \"<MODEL_NAME> <Framework> Pretrain\"
echo \"Job: \$SLURM_JOB_ID | Nodes: \$SLURM_JOB_NUM_NODES\"
echo \"TP=\$TP PP=\$PP EP=\$EP MBS=\$MBS GBS=\$GBS\"
echo \"======================================\"

# Phase 1: Single-process uv sync to build/populate the shared cache
srun --mpi=pmix -N 1 --ntasks=1 \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --no-container-mount-home \
  bash -c "cd $WORKDIR && uv sync"

# Phase 2: Full multi-node run (uv sync in TRAIN_CMD is a fast no-op)
srun --mpi=pmix --no-kill \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --no-container-mount-home \
  bash -c "$TRAIN_CMD" 2>&1 | tee "$LOGDIR/<prefix>_${SLURM_JOB_ID}.log"

echo ""
echo "======================================"
echo "Done. Losses:"
echo "======================================"
grep -E "iteration\s+" "$LOGDIR/<prefix>_${SLURM_JOB_ID}.log" | grep -iE "lm loss|reduced_train_loss" | head -25
```

## Bridge-Specific TRAIN_CMD Body

```bash
rm -rf nemo_experiments && \
uv run python -m torch.distributed.run \
  --nproc_per_node=8 \
  --nnodes=\${SLURM_JOB_NUM_NODES} \
  --node_rank=\${SLURM_NODEID} \
  scripts/training/run_recipe.py \
  --recipe <recipe_name> \
  model.tensor_model_parallel_size=$TP \
  model.pipeline_model_parallel_size=$PP \
  ...overrides...
```

## MLM-Specific TRAIN_CMD Body

```bash
PYTHONPATH=${WORKDIR}/3rdparty/Megatron-LM:\${PYTHONPATH:-} \
uv run python -m torch.distributed.run \
  --nproc_per_node=8 \
  --nnodes=\${SLURM_JOB_NUM_NODES} \
  --node_rank=\${SLURM_NODEID} \
  3rdparty/Megatron-LM/pretrain_gpt.py \
  --tensor-model-parallel-size $TP \
  --pipeline-model-parallel-size $PP \
  ...args...
```
