---
name: automodel-expert-lora
description: Apply LoRA to fused MoE expert layers in NeMo AutoModel using HuggingFace Transformers v5+ models. Covers expert parameter detection, rank_pattern configuration, and the validation warning emitted when match_all_linear silently skips expert weights. Use when fine-tuning MoE models (Mixtral, Qwen3-MoE, DeepSeek) with LoRA and needing expert layers adapted, or when diagnosing why only attention layers are changing during MoE LoRA training.
when_to_use: LoRA on MoE models in NeMo AutoModel, expert weight adaptation, rank_pattern configuration, silent skip diagnosis; 'match_all_linear MoE', 'expert LoRA', 'fused expert parameters', 'target_modules MoE', 'Mixtral LoRA', 'Qwen3-MoE LoRA', 'DeepSeek LoRA', 'nn.Parameter expert'.
license: Apache-2.0
---

# Expert LoRA for Fused MoE Models

Card: @skills/automodel-expert-lora/card.yaml

## The Problem

In Transformers v5+, fused MoE models (Mixtral, Qwen3-MoE, DeepSeek-V3,
GLM-4.5) register expert weights as `nn.Parameter` inside a combined linear
layer — not as individual `nn.Linear` modules. `match_all_linear=True` iterates
`nn.Linear` only. Expert parameters are invisible to it.

Result: LoRA appears to run, loss changes only from attention adaptation, and
the expert layers are never modified. No error is raised.

As of NeMo AutoModel v0.x (issue #1151), `apply_lora()` now emits a
`UserWarning` when this condition is detected, and three utilities are
available to configure expert LoRA correctly.

## Quick Decision

| Model family | Expert param pattern | Correct target_modules |
|---|---|---|
| Mixtral | `block_sparse_moe.w1/w2/w3` | `["w1", "w2", "w3"]` |
| Qwen3-MoE | `mlp.experts.gate_proj/up_proj/down_proj` | `["gate_proj", "up_proj", "down_proj"]` |
| DeepSeek-V3 | `mlp.experts.gate_proj/up_proj/down_proj` | `["gate_proj", "up_proj", "down_proj"]` |
| GLM-4.5 | `mlp.experts.gate_proj/up_proj/down_proj` | `["gate_proj", "up_proj", "down_proj"]` |

If unsure, run `detect_fused_moe_experts(model)` — it returns the correct
list for any supported model.

## Enablement

### Step 1 — Detect expert parameter names

```python
from nemo_automodel.components._peft.lora import detect_fused_moe_experts

targets = detect_fused_moe_experts(model)
# e.g. returns ["w1", "w2", "w3"] for Mixtral
#              ["down_proj", "gate_proj", "up_proj"] for Qwen3-MoE
```

### Step 2 — Build rank_pattern (optional: per-expert rank sizing)

```python
from nemo_automodel.components._peft.lora import build_expert_lora_rank_pattern

rank_pattern = build_expert_lora_rank_pattern(
    model,
    base_rank=16,
    expert_rank_multiplier=0.5,  # smaller rank for experts to save memory
)
# e.g. {"block_sparse_moe": 8}
```

### Step 3 — Apply LoRA with explicit target_modules

```python
from peft import LoraConfig, get_peft_model
from nemo_automodel.components._peft.lora import detect_fused_moe_experts, build_expert_lora_rank_pattern

targets = detect_fused_moe_experts(model)
rank_pattern = build_expert_lora_rank_pattern(model, base_rank=16)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=targets,
    rank_pattern=rank_pattern,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Should show expert layers in trainable params, not just attention
```

### Validation warning

If `apply_lora()` is called with `match_all_linear=True` and no
`target_modules`, and the model has fused expert parameters, a `UserWarning`
is emitted with the detected parameter names and a fix snippet. Treat this
as an error — silent expert-skip produces wrong training dynamics.

```
UserWarning: [NeMo AutoModel] Fused MoE expert parameters detected but will
NOT be adapted by LoRA.

  Detected expert parameter names: ['w1', 'w2', 'w3']

  To apply LoRA to expert layers, pass target_modules explicitly:
    lora_config = LoraConfig(
        target_modules=['w1', 'w2', 'w3'],
        rank_pattern=build_expert_lora_rank_pattern(model, base_rank=16),
    )
```

## Code Anchors

Expert detection utility:

```python
# nemo_automodel/components/_peft/lora.py
_FUSED_EXPERT_PARAM_PATTERNS = (
    "block_sparse_moe",   # Mixtral
    "mlp.experts",        # Qwen3-MoE, DeepSeek
    "moe.experts",        # generic
    "ffn.experts",        # generic
)

def detect_fused_moe_experts(model: nn.Module) -> list[str]:
    # inspects named_parameters() for known fused MoE patterns
    # returns sorted list of leaf parameter name suffixes
```

Rank pattern builder:

```python
# nemo_automodel/components/_peft/lora.py
def build_expert_lora_rank_pattern(
    model: nn.Module,
    base_rank: int,
    expert_rank_multiplier: float = 1.0,
) -> dict[str, int]:
    # maps MoE pattern keys to int(base_rank * multiplier)
    # returns {} for dense models
```

Validation hook in apply_lora:

```python
# nemo_automodel/components/_peft/lora.py
def apply_lora(model, lora_config, match_all_linear=False, target_modules=None):
    validate_lora_config_for_moe(model, match_all_linear, target_modules)
    # ... existing LoRA application logic ...
```

Tests:

```python
# tests/unit/components/peft/test_expert_lora.py
class TestDetectFusedMoeExperts   # 4 tests
class TestBuildExpertLoraRankPattern  # 4 tests
class TestValidateLoraConfigForMoe    # 3 tests
```

## Pitfalls

1. **Silent failure with match_all_linear**: The most dangerous failure mode.
   Training appears normal, loss decreases, but expert weights are never
   adapted. Only detectable by checking `model.print_trainable_parameters()`
   and confirming expert layers appear — or by observing that expert-heavy
   tasks show no improvement vs attention-only LoRA.

2. **rank_pattern key must match parameter path substring**: The keys in
   `rank_pattern` are matched against full parameter names. Use the pattern
   as returned by `detect_fused_moe_experts` or `build_expert_lora_rank_pattern`
   — do not abbreviate.

3. **expert_rank_multiplier < 1 floors at rank 1**: Setting
   `expert_rank_multiplier=0.1` with `base_rank=4` gives rank 1, not 0.
   This is intentional — rank 0 is invalid. Verify effective rank with
   `model.print_trainable_parameters()`.

4. **Dense model returns empty pattern**: `build_expert_lora_rank_pattern`
   returns `{}` for dense models. Passing an empty `rank_pattern` to
   `LoraConfig` is safe — PEFT falls back to the global `r` value.

5. **target_modules suppresses the warning**: Once `target_modules` is
   provided, `validate_lora_config_for_moe` returns immediately and
   does not check whether the provided names actually cover expert layers.
   Use `detect_fused_moe_experts` to generate the list rather than
   guessing module names.

## Verification

Unit tests for all three utilities:

```bash
pytest tests/unit/components/peft/test_expert_lora.py -v
```

Expected: `12 passed`

Confirm expert layers are trainable after apply_lora:

```python
model = get_peft_model(model, lora_config)
trainable = {n for n, p in model.named_parameters() if p.requires_grad}
expert_patterns = detect_fused_moe_experts(model.base_model)
assert any(
    any(pat in name for pat in expert_patterns)
    for name in trainable
), "No expert parameters in trainable set — check target_modules"
```

Success criteria:

- `12 passed` on unit tests
- `model.print_trainable_parameters()` shows expert layer names in the
  trainable parameter count
- No `UserWarning` about fused MoE expert skip when target_modules is set
