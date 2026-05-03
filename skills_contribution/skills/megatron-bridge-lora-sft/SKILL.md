---
name: megatron-bridge-lora-sft
description: Configure and run LoRA, DoRA, and full SFT fine-tuning in Megatron-Bridge. Covers PEFT recipe selection, target module wiring, adapter merging, and HuggingFace checkpoint export. Use when applying LoRA or DoRA to any Bridge-supported model, setting up SFT datasets, debugging PEFT config errors, or exporting fine-tuned weights back to HuggingFace format.
when_to_use: LoRA or DoRA fine-tuning, SFT recipe setup, PEFT config errors, adapter merging, HuggingFace export after fine-tuning; 'peft_config', 'LoRA', 'DoRA', 'lora_rank', 'target_modules', 'merge_lora', 'sft_config', 'fine-tune', 'adapter export'.
license: Apache-2.0
---

# LoRA / DoRA / SFT Fine-Tuning

Stable docs: @docs/training/peft.md
Card: @skills/megatron-bridge-lora-sft/card.yaml

## Quick Decision

| Goal | Recipe type | Min GPUs |
|---|---|---|
| LoRA on 8B model | `*_peft_config` | 1 |
| LoRA on 70B model | `*_peft_config` | 8 |
| LoRA on 235B MoE | `*_peft_config` | 16 |
| Full SFT on 8B | `*_sft_config` | 2 |
| Full SFT on 70B | `*_sft_config` | 16 |
| Merge adapters + export to HF | Post-training step | Same as training |

Use PEFT recipes when GPU count is the constraint. Use SFT recipes when
you need full gradient flow through all parameters.

## Enablement

### LoRA (minimal)

```python
from megatron.bridge.recipes.llama import llama3_8b_peft_config

cfg = llama3_8b_peft_config()

# Default: rank=16, alpha=32, target_modules=["linear_qkv", "linear_proj"]
# Override rank and alpha:
cfg.peft.lora_rank = 32
cfg.peft.lora_alpha = 64

# Add MLP layers to target modules:
cfg.peft.target_modules = [
    "linear_qkv",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
]
```

### DoRA

```python
cfg.peft.use_dora = True
cfg.peft.lora_rank = 16
cfg.peft.lora_alpha = 16  # alpha == rank is the DoRA convention
```

### SFT (full fine-tune)

```python
from megatron.bridge.recipes.llama import llama3_8b_sft_config

cfg = llama3_8b_sft_config()
cfg.dataset.data_path = ["/data/train.jsonl"]
cfg.dataset.seq_length = 4096
cfg.train.global_batch_size = 128
cfg.train.micro_batch_size = 2
cfg.optimizer.lr = 1e-5
```

### MoE LoRA — expert layer targeting

For MoE models (Qwen3-MoE, DeepSeek, GLM-4.5), expert weights are
registered as `nn.Parameter`, not `nn.Linear`. `match_all_linear=True`
silently skips them. Set `target_modules` explicitly:

```python
cfg = qwen3_30b_a3b_peft_config()
cfg.peft.target_modules = [
    "linear_qkv",       # attention
    "linear_proj",      # attention output
    "gate_proj",        # expert gate
    "up_proj",          # expert up
    "down_proj",        # expert down
]
cfg.peft.lora_rank = 16
cfg.peft.lora_alpha = 32
```

### Adapter merge and HuggingFace export

```python
from megatron.bridge.peft.merge import merge_lora_weights
from megatron.bridge.convert import export_to_hf

# Step 1: merge adapters into base weights
merge_lora_weights(
    checkpoint_dir="/checkpoints/lora_run",
    output_dir="/checkpoints/merged",
)

# Step 2: export merged checkpoint to HuggingFace format
export_to_hf(
    megatron_checkpoint="/checkpoints/merged",
    hf_output_dir="/hf_model/",
    model_type="llama3",
)
```

Or via CLI:

```bash
python scripts/convert/megatron_to_hf.py \
  --checkpoint /checkpoints/merged \
  --output /hf_model/ \
  --model-type llama3
```

## Entry Points

```bash
# LoRA fine-tune (1 GPU)
uv run python -m torch.distributed.run --nproc_per_node=1 \
  scripts/training/run_recipe.py \
  --recipe llama3_8b_peft_config \
  --dataset llm-finetune

# SFT fine-tune (2 GPUs)
uv run python -m torch.distributed.run --nproc_per_node=2 \
  scripts/training/run_recipe.py \
  --recipe llama3_8b_sft_config \
  --dataset llm-finetune

# Override LoRA rank via CLI
uv run python -m torch.distributed.run --nproc_per_node=1 \
  scripts/training/run_recipe.py \
  --recipe llama3_8b_peft_config \
  --dataset llm-finetune \
  'peft.lora_rank=32' \
  'peft.lora_alpha=64'
```

## Code Anchors

PEFT config definition:

```python
# src/megatron/bridge/training/config.py
@dataclass
class PEFTConfig:
    lora_rank: int = 16
    lora_alpha: float = 32
    lora_dropout: float = 0.0
    use_dora: bool = False
    target_modules: list[str] = field(default_factory=lambda: ["linear_qkv", "linear_proj"])
    match_all_linear: bool = False
```

LoRA adapter application:

```python
# src/megatron/bridge/training/peft.py
def apply_lora(model, peft_config):
    # wraps target modules with LoraLinear / DoraLinear
    # match_all_linear iterates nn.Linear only — misses nn.Parameter MoE experts
```

Merge utility:

```python
# src/megatron/bridge/peft/merge.py
def merge_lora_weights(checkpoint_dir, output_dir):
    # loads base + adapter shards, merges in-place, writes merged checkpoint
```

PEFT recipe examples:

```python
# src/megatron/bridge/recipes/llama.py
def llama3_8b_peft_config() -> ConfigContainer:
    cfg = llama3_8b_sft_config()
    cfg.peft = PEFTConfig(lora_rank=16, lora_alpha=32)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    return cfg
```

## Pitfalls

1. **MoE expert layers silently skipped**: `match_all_linear=True` only
   matches `nn.Linear`. Expert weights in fused MoE blocks (Qwen3-MoE,
   DeepSeek, GLM-4.5) are `nn.Parameter` — they are invisible to the
   matcher. Always set `target_modules` explicitly for MoE models.

2. **DoRA alpha convention**: DoRA expects `lora_alpha == lora_rank`. Using
   the standard LoRA convention (`alpha = 2 * rank`) will not error but
   produces suboptimal scaling. Set `alpha = rank` for DoRA.

3. **Merge before export**: Exporting a LoRA checkpoint to HuggingFace
   without merging produces a broken HF model — the base weights do not
   include adapter contributions. Always run `merge_lora_weights()` first.

4. **TP > 1 with PEFT**: LoRA adapters are sharded along with the base
   layer when `tensor_model_parallel_size > 1`. The adapter shapes must be
   consistent across TP ranks. Mismatched `lora_rank` between ranks causes
   a shape error at initialization, not at the first forward pass.

5. **SFT with packed sequences requires MBS=1**: When `PackedSequenceSpecs`
   is active, setting `micro_batch_size > 1` raises a `ValueError`. PEFT
   recipes default to `MBS=1`; SFT recipes may need explicit adjustment.

6. **`calculate_per_token_loss` for SFT with CP**: When context parallelism
   (`context_parallel_size > 1`) is enabled for SFT, set
   `cfg.model.calculate_per_token_loss = True` and
   `cfg.ddp.average_in_collective = False`. Omitting either causes
   incorrect loss scaling across CP ranks.

7. **LoRA dropout and inference**: `lora_dropout > 0` is training-only.
   Ensure the adapter is saved and merged in eval mode or dropout
   will be applied during export, corrupting merged weights.

## Verification

Unit test coverage for PEFT config validation:

```bash
uv run python -m pytest tests/unit_tests/training/test_config.py \
  -k "peft or lora" -v
```

Smoke test LoRA on 1 GPU with mock data:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m torch.distributed.run --nproc_per_node=1 \
  scripts/training/run_recipe.py \
  --recipe llama3_8b_peft_config \
  --dataset llm-finetune \
  'train.train_iters=5' \
  'logger.log_interval=1'
```

Success criteria:

- Exit code 0
- Finite loss at iteration 5 (e.g. `lm loss: 9.8E+00`)
- Log shows `PEFTConfig` with expected `lora_rank` and `target_modules`
- No `KeyError` or shape mismatch during adapter initialization
