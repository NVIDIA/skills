---
name: megatron-bridge-lora-sft
description: Configure and run LoRA, DoRA, and full SFT fine-tuning in Megatron-Bridge. Covers LoRA dataclass setup, target module wiring, normalize_moe_lora for MoE models, and adapter export via AutoBridge.export_adapter_ckpt. Use when applying LoRA or DoRA to any Bridge-supported model, setting up SFT datasets, or exporting fine-tuned adapters to HuggingFace PEFT format.
when_to_use: LoRA or DoRA fine-tuning, SFT recipe setup, normalize_moe_lora, MoE expert targeting, adapter export to HuggingFace, peft_scheme lora dora, dim alpha target_modules LoRA dataclass, torchrun recipe fine-tune, export_adapter_ckpt AutoBridge.
---

# LoRA / DoRA / SFT Fine-Tuning

Card: @skills/megatron-bridge-lora-sft/card.yaml

## Quick Decision

| Goal | peft_scheme | Min GPUs |
|---|---|---|
| LoRA on 1B model | `"lora"` | 1 |
| DoRA on 1B model | `"dora"` | 1 |
| Full SFT on 8B | sft recipe | 2 |
| Export adapter to HF PEFT | CPU only | 0 GPUs |

## Enablement

### LoRA (minimal)

```python
from megatron.bridge.recipes.llama import llama32_1b_peft_config

config = llama32_1b_peft_config(peft_scheme="lora")

# Default target_modules: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
# Default dim=32, alpha=32

# Override rank and alpha:
config.peft.dim = 16
config.peft.alpha = 32
```

Launch:

```bash
torchrun --nproc_per_node=1 tutorials/recipes/llama/01_quickstart_finetune.py \
  --pretrained-checkpoint /path/to/checkpoint
```

### DoRA

```python
config = llama32_1b_peft_config(peft_scheme="dora")
config.peft.dim = 16
config.peft.alpha = 64   # DoRA default alpha is 64
```

### MoE LoRA — expert layer targeting

For MoE models, add expert projection names to `target_modules` and enable
`normalize_moe_lora` to scale down expert rank proportionally:

```python
from megatron.bridge.peft.lora import LoRA

lora = LoRA(
    target_modules=[
        "linear_qkv",       # attention
        "linear_proj",      # attention output
        "linear_fc1",       # MLP gate/up (dense fallback)
        "linear_fc2",       # MLP down (dense fallback)
    ],
    dim=32,
    alpha=32,
    normalize_moe_lora=True,  # dim // moe_router_topk for expert layers
)
```

With `normalize_moe_lora=True`:
- Expert linear layers: effective dim = `dim // moe_router_topk`
- Non-expert layers: effective dim = `dim` (unchanged)
- `dim` must be evenly divisible by `moe_router_topk`

### Adapter export to HuggingFace

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge(hf_model_path="/path/to/hf/model")

bridge.export_adapter_ckpt(
    peft_checkpoint="/checkpoints/lora_run",
    output_path="./my_adapter",
)
# produces: ./my_adapter/adapter_config.json
#           ./my_adapter/adapter_model.safetensors
```

Or via CLI script:

```bash
python examples/conversion/adapter/export_adapter.py \
  --hf-model-path /path/to/hf/model \
  --lora-checkpoint /checkpoints/lora_run \
  --output ./my_adapter
```

The exported adapter loads directly with HuggingFace PEFT:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./my_adapter")
```

Export runs on CPU — no GPU required.

## Code Anchors

LoRA dataclass:

```python
# src/megatron/bridge/peft/lora.py
@dataclass
class LoRA(PEFT, ModuleMatcher):
    target_modules: List[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "pre"
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"
    a2a_experimental: bool = False
    lora_dtype: torch.dtype = None
    normalize_moe_lora: bool = False
```

DoRA dataclass:

```python
# src/megatron/bridge/peft/dora.py
@dataclass
class DoRA(PEFT, ModuleMatcher):
    target_modules: List[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    dim: int = 32
    alpha: int = 64   # DoRA default differs from LoRA default
```

Recipe function:

```python
# tutorials/recipes/llama/01_quickstart_finetune.py
from megatron.bridge.recipes.llama import llama32_1b_peft_config

config = llama32_1b_peft_config(peft_scheme="lora")  # or "dora"
config.peft.dim = 16
config.peft.alpha = 32
```

Export:

```python
# examples/conversion/adapter/export_adapter.py
bridge = AutoBridge(hf_model_path=...)
bridge.export_adapter_ckpt(peft_checkpoint=..., output_path=...)
```

## Pitfalls

1. **MoE expert layers silently skipped without normalize_moe_lora or explicit targets**:
   The default `target_modules` covers attention and MLP layers for dense models.
   For MoE models, expert weights may not be covered — verify with a forward pass
   that expert parameters have `requires_grad=True`.

2. **DoRA alpha convention**: DoRA default `alpha=64`, not 32. Check the `DoRA`
   dataclass defaults before overriding.

3. **normalize_moe_lora requires evenly divisible dim**: `dim` must be divisible by
   `moe_router_topk`. Indivisible `dim` values will error.

4. **Export produces HF PEFT adapter — no merge step needed**: Unlike some frameworks,
   `export_adapter_ckpt` produces `adapter_config.json` + `adapter_model.safetensors`
   which load directly via `PeftModel.from_pretrained`. No separate merge step is
   required before HuggingFace use.

5. **TP > 1 with PEFT**: LoRA adapter shapes are sharded with the base layer when
   `tensor_model_parallel_size > 1`. Adapter `dim` must be consistent across TP ranks.
   Mismatched `dim` causes a shape error at initialization.

## Verification

Smoke test LoRA on 1 GPU with mock data:

```bash
torchrun --nproc_per_node=1 tutorials/recipes/llama/01_quickstart_finetune.py \
  --pretrained-checkpoint /path/to/checkpoint
```

Success criteria:

- Exit code 0
- Finite loss in logs
- Adapter files generated: `adapter_config.json` + `adapter_model.safetensors`
- `PeftModel.from_pretrained(base_model, output_path)` loads without error
