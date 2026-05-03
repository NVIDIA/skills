---
name: automodel-expert-lora
description: Apply LoRA to fused MoE expert layers in NeMo AutoModel. Covers PeftConfig setup, moe_rank_scaling for automatic per-expert rank reduction, target_modules wildcard matching for expert layers, and the GroupedExpertsTE limitation. Use when fine-tuning MoE models (models using GroupedExperts or GroupedExpertsDeepEP) with LoRA and needing expert layers adapted, or when diagnosing why expert weights are not being trained.
when_to_use: LoRA on MoE models in NeMo AutoModel, expert weight adaptation, moe_rank_scaling, target_modules for MoE, expert LoRA patching, GroupedExperts LoRA, dim scaling by n_activated_experts, apply_lora_to_linear_modules MoE.
---

# Expert LoRA for Fused MoE Models

Card: @skills/automodel-expert-lora/card.yaml

## The Problem

In NeMo AutoModel, fused MoE expert layers (`GroupedExperts`, `GroupedExpertsDeepEP`) are
not `nn.Linear` modules. `match_all_linear=True` iterates `nn.Linear` only and silently
skips expert parameters.

Result: LoRA runs but only attention or dense linear layers are adapted. Expert weights are
never modified. No error is raised.

Additionally, `GroupedExpertsTE` (Transformer Engine expert layers) are not supported —
passing them raises `NotImplementedError`.

## Quick Decision

| Scenario | PeftConfig setting |
|---|---|
| Adapt expert layers | `target_modules=["*experts*"]` |
| Adapt specific expert name | `target_modules=["experts"]` |
| Reduce expert rank proportionally | `moe_rank_scaling=True` |
| Dense model only | `match_all_linear=True` (skips MoE) |
| TE expert layers | Not supported — raises NotImplementedError |

## Enablement

### Step 1 — Configure PeftConfig for expert layers

```python
from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules

peft_config = PeftConfig(
    target_modules=["*experts*"],  # wildcard matches modules with "experts" in the name
    dim=16,
    alpha=32,
)

n_patched = apply_lora_to_linear_modules(model, peft_config)
# returns count of modules patched
```

For exact name matching instead of wildcard:

```python
peft_config = PeftConfig(
    target_modules=["experts"],  # exact substring match
    dim=8,
    alpha=32,
)
```

### Step 2 — Use moe_rank_scaling for proportional rank reduction

`moe_rank_scaling=True` divides `dim` by `n_activated_experts` for expert modules while
keeping the full `dim` for dense linear layers. This normalizes total adapter capacity.

```python
peft_config = PeftConfig(
    target_modules=["experts", "linear"],  # both MoE and dense
    dim=16,
    alpha=32,
    moe_rank_scaling=True,
)
# model.config.n_activated_experts = 2
# → expert lora_dim = 16 // 2 = 8
# → linear lora_dim = 16 (unchanged)

n_patched = apply_lora_to_linear_modules(model, peft_config)
```

Constraints:
- `dim` must be >= `n_activated_experts`; otherwise raises `ValueError`
- Non-evenly-divisible `dim` is allowed (floor division) but emits a warning
- `moe_rank_scaling=False` (default): all modules use the full `dim`

### Step 3 — Verify expert layers are trainable

```python
trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
# Confirm expert parameter names appear in the list
assert any("experts" in n for n, _ in trainable), \
    "No expert parameters are trainable — check target_modules"
```

## Code Anchors

PeftConfig and application function:

```python
# nemo_automodel/components/_peft/lora.py
@dataclass
class PeftConfig:
    target_modules: list = field(default_factory=list)
    exclude_modules: list = field(default_factory=list)
    match_all_linear: bool = False
    dim: int = 8
    alpha: int = 32
    use_dora: bool = False
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "post"
    lora_A_init: str = "xavier"
    lora_dtype: Optional[torch.dtype] = None
    use_triton: bool = False
    moe_rank_scaling: bool = False

def apply_lora_to_linear_modules(
    model: nn.Module,
    peft_config: PeftConfig,
    quantization_config=None,
    skip_freeze: bool = False,
) -> int:
    # patches matched nn.Linear and MoE expert modules
    # returns count of patched modules
```

MoE module patching:

```python
# nemo_automodel/components/_peft/lora.py
def patch_moe_module(
    orig_module,
    dim=8,
    alpha=32,
    lora_A_init_method="xavier",
    lora_dtype=None,
) -> nn.Module:
    # GroupedExperts     → GroupedExpertsLoRA
    # GroupedExpertsDeepEP → GroupedExpertsDeepEPLoRA
    # GroupedExpertsTE   → raises NotImplementedError
```

Tests:

```python
# tests/unit_tests/_peft/test_lora_experts.py
test_apply_lora_equivalence        # wildcard target_modules=["*experts*"]
test_apply_lora_patching_logic     # exact and wildcard matching
test_moe_rank_scaling_basic        # dim=16, n_activated_experts=2 → lora_dim=8
test_moe_rank_scaling_default_off  # moe_rank_scaling=False keeps full dim
test_moe_rank_scaling_floor_division_warning  # non-divisible dim
test_moe_rank_scaling_dim_too_small           # dim < n_activated_experts → ValueError
test_moe_rank_scaling_output_equivalence      # zero-init B → identical baseline output
```

## Pitfalls

1. **Silent expert-skip with match_all_linear**: `match_all_linear=True` iterates
   `nn.Linear` modules only. Expert modules are not `nn.Linear` — they are silently
   skipped. Training appears to run but only dense/attention layers are adapted.
   Always set `target_modules` explicitly when working with MoE models.

2. **GroupedExpertsTE not supported**: Models using Transformer Engine expert layers
   (`GroupedExpertsTE`) raise `NotImplementedError` when `patch_moe_module` is called.
   There is no workaround — TE expert LoRA is not implemented.

3. **dim too small with moe_rank_scaling**: Setting `dim < n_activated_experts` with
   `moe_rank_scaling=True` raises a `ValueError`. Increase `dim` to at least
   `n_activated_experts`.

4. **Floor division warning**: When `dim` is not evenly divisible by `n_activated_experts`,
   floor division is applied and a warning is logged. The resulting `lora_dim` may be
   unexpectedly small. Verify effective rank with trainable parameter inspection.

5. **target_modules must match module names, not parameter names**: Wildcard patterns
   like `"*experts*"` are matched against module names from `model.named_modules()`,
   not parameter names from `model.named_parameters()`.

## Verification

Run unit tests:

```bash
pytest tests/unit_tests/_peft/test_lora_experts.py -v
```

Confirm expert modules are patched:

```python
peft_config = PeftConfig(target_modules=["*experts*"], dim=8)
n = apply_lora_to_linear_modules(model, peft_config)
assert n > 0, "No modules were patched — check target_modules pattern"

trainable = {n for n, p in model.named_parameters() if p.requires_grad}
assert any("experts" in name for name in trainable), \
    "Expert parameters not in trainable set"
```

Success criteria:

- Unit tests pass
- `n_patched > 0` after `apply_lora_to_linear_modules`
- Expert parameter names appear in `model.named_parameters()` with `requires_grad=True`
- No `NotImplementedError` (i.e., model does not use `GroupedExpertsTE`)
