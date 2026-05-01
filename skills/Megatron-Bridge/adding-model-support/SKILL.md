---
name: adding-model-support
description: Guide for adding support for new LLM or VLM models in Megatron-Bridge. Covers bridge, provider, recipe, tests, docs, and examples. Use when the user asks to add, support, onboard, or integrate a new model, or when creating bridges, providers, or recipes for a new model family.
---

# Adding New Model Support in Megatron-Bridge

## Phase 1: Discovery

### Step 1 — Get the HF model link

Ask the user for the HuggingFace model link (e.g. `https://huggingface.co/Qwen/Qwen3.5-VL-27B`).

If the model is **not public**, ask the user to provide the `config.json` file directly.

### Step 2 — Fetch and analyze config.json

Read the model's `config.json` from HuggingFace (or from the user-provided file). Key fields to extract:

- `model_type` — used for `@register_bridge(model_type=...)`
- `architectures` — the HF model class name (used for `source=...` in registration)
- `tie_word_embeddings` — critical for weight tying
- Architecture fields: `num_hidden_layers`, `hidden_size`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads`, `vocab_size`, `max_position_embeddings`, `rope_theta`, etc.
- MoE fields (if present): `num_local_experts`, `num_experts_per_tok`, `moe_intermediate_size`
- MLA fields (if present): `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`

If there are config fields you don't recognize from previously supported models (check `CONFIG_MAPPING` in `model_bridge.py` and existing bridges), this likely indicates a **new architectural block** (e.g., a novel attention variant, custom normalization, or a new layer type). Ask the user to provide the HuggingFace `modeling_*.py` implementation of that block so you can understand the computation and create the correct Megatron-side mapping or custom module.

### Step 3 — Determine VLM vs LLM

**VLM** (Vision-Language Model) if config.json contains:
- `text_config` AND `vision_config` sub-configs
- Note: VLMs may or may not have "VL" in the name

**LLM** (Text-only) if:
- No `text_config` / `vision_config`
- Single flat config for the language model

This distinction affects:
- Which files to create (VLMs need a model.py combining vision + language)
- Where to read config fields from (`text_config` vs top-level for VLMs)
- Test patterns (VLMs need vision inputs in functional tests)

### Step 4 — Check for quantized weights (FP8 / FP4)

Inspect the HF checkpoint's `model.safetensors` (or `model.safetensors.index.json`) for quantized
weight dtypes such as `float8_e4m3fn` (FP8) or `uint8`/`uint4` with accompanying `*_scale_inv` or
`*_scale` tensors. Common signs:

- `config.json` mentions `quantization_config` or dtype fields like `"torch_dtype": "float8_e4m3fn"`
- Safetensors contain `weight_scale_inv` keys alongside the main weight keys
- The model card mentions FP8/FP4/INT4 weights

**Why this matters:** The bridge's `import_ckpt` path does **not** automatically dequantize — it
loads raw quantized values as-is. This produces a silently broken model (random-level loss, huge
grad norms) instead of raising an error.

**Fix:** Dequantize before conversion. Two approaches:

1. **Standalone script** (recommended for user-facing models) — Write a
   `dequant_fp8_for_bridge.py` in the model's examples folder.
   Reference: `examples/models/vlm/ministral3/dequant_fp8_for_bridge.py`.
   The pattern is: `w_bf16 = fp8_weight.to(bfloat16) * weight_scale_inv`.

2. **In-bridge hook** — Override `maybe_modify_loaded_hf_weight()` in the bridge class to
   dequantize on the fly during import:

   ```python
   def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
       weight = hf_state_dict[hf_param]
       scale_key = hf_param + "_scale_inv"
       if weight.dtype == torch.float8_e4m3fn and scale_key in hf_state_dict:
           return weight.to(torch.bfloat16) * hf_state_dict[scale_key].to(torch.bfloat16)
       return weight
   ```

Always add a sanity check in the verification workflow (e.g., print `std` of a weight tensor —
quantized models typically have `std ≈ 13` before dequantization vs `std ≈ 0.006` after).

## Phase 2: Bridge Support

### File structure

**LLM** — Reference: Qwen2 (`src/megatron/bridge/models/qwen/qwen2_bridge.py`)

```
src/megatron/bridge/models/<model>/
├── __init__.py
├── <model>_bridge.py      # Config + weight mappings (no provider file needed)
└── modeling_<model>/      # (optional) Custom nn.Module implementations if needed
    └── ...
```

**VLM** — Reference: Qwen3.5-VL (`src/megatron/bridge/models/qwen_vl/`)

```
src/megatron/bridge/models/<model>/
├── __init__.py
├── <model>_bridge.py         # Config + weight mappings
├── <model>_provider.py       # Only for VLMs that need custom provide()
└── modeling_<model>/         # If using Megatron vision encoder
    ├── __init__.py
    └── model.py              # Combines vision + language
```

OR with HF vision encoder (Reference: Gemma3-VL):

```
src/megatron/bridge/models/<model>/
├── __init__.py
├── <model>_bridge.py
├── <model>_provider.py       # Only for VLMs that need custom provide()
└── modeling_<model>.py       # HF vision + Megatron language wrapper
```

**Model-specific modeling code:** If the model requires custom `nn.Module` implementations
(e.g. a custom RoPE variant, non-standard transformer config, custom thinker/talker
architecture), place them in a `modeling_<model>/` directory or a single `modeling_<model>.py`
file inside the model family folder. Use a directory when there are multiple files (model,
transformer config, custom ops); use a single file when one module suffices. Never put
model-specific modeling code in shared directories or as loose files in the bridge family
directory — keep them namespaced under the `modeling_<model>` prefix.

### Implementation order

**LLM:**
1. **Bridge only** — Register bridge, implement `provider_bridge()` and `mapping_registry()`.
   The bridge calls `super().provider_bridge()` to get a `GPTModelProvider` from `CONFIG_MAPPING`,
   then sets model-specific attributes on it. **Do not create a provider file** — the stock
   provider returned by `super().provider_bridge()` is usually sufficient for LLMs
   (e.g., `GPTModelProvider`, or another base provider selected via `PROVIDER_CLASS`).

**VLM:**
1. **Bridge** — Register bridge, implement config and weight mappings.
2. **Provider** (when needed) — Only VLMs that require a custom `provide()` to instantiate a
   combined vision+language model need a provider subclass. The bridge manually calls
   `hf_config_to_provider_kwargs(text_config)` and instantiates the custom provider.
3. **Model class** — Combine vision encoder + language decoder.

For detailed patterns, see:
- VLM: @skills/adding-model-support/vlm-patterns.md
- LLM: @skills/adding-model-support/llm-patterns.md

### Critical: `tie_word_embeddings` for VLMs

For VLMs, `tie_word_embeddings` lives on the **top-level** HF config, NOT on `text_config`. Always read from the parent config:

```python
provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
```

### Critical: Config field location for VLMs

When reading HF config for VLMs, check whether each field is in:
- `hf_config` (top-level) — e.g. `tie_word_embeddings`, `image_token_id`, `video_token_id`
- `hf_config.text_config` — e.g. `num_hidden_layers`, `hidden_size`, etc.
- `hf_config.vision_config` — e.g. vision encoder dimensions

### Encapsulating model-specific layers

When a new model introduces custom or non-standard layers (novel attention variants, custom
normalization, fused expert layouts, MTP heads, etc.), **keep all model-specific logic inside
the model family directory**. Do not modify shared files in `src/megatron/bridge/models/conversion/`
(e.g. `param_mapping.py`, `model_bridge.py`, `quant_mapping.py`) unless the change is genuinely
reusable across multiple model families.

**Principle:** The bridge and provider files for a model family are your primary extension surface.
Shared conversion infrastructure provides hooks and base classes — subclass them locally rather
than adding conditionals to shared code.

#### Strategy 1: Create a local mapping subclass

If the model has a layer whose weight layout doesn't match any existing mapping class, create a
private mapping class in the bridge file or a `<model>_mappings.py` file in the family directory.

Example — GLM's fused expert down-projection disables grouped-export transpose:

```python
# src/megatron/bridge/models/glm/glm_moe_mappings.py
class GLMExpertDownProjMapping(FusedExpertMapping):
    def __init__(self, megatron_param, hf_param, permute_dims=None):
        super().__init__(megatron_param, hf_param, permute_dims, transpose_on_export=False)
```

Example — Nemotron-H's MTP layers flatten indices during resolve:

```python
# Inside nemotron_h_bridge.py (private to the module)
class _MTPFlatteningMapping(MegatronParamMapping):
    def resolve(self, captures):
        return AutoMapping(self._flatten(captures), ...)
```

Example — MiniMax-M2's non-standard QK norm layout:

```python
# Inside minimax_m2_bridge.py (private to the module)
class _FullDimQKNormMapping(MegatronParamMapping):
    def hf_to_megatron(self, hf_weights):
        # Custom scatter logic for full-dim QK norm
        ...
    def megatron_to_hf(self, megatron_weights):
        # Custom gather logic
        ...
```

#### Strategy 2: Override bridge hooks

`MegatronModelBridge` provides several override hooks — use them instead of modifying the base class:

| Hook | When to use |
|------|-------------|
| `mapping_registry()` | Define all weight name mappings (abstract, always overridden) |
| `provider_bridge()` | Configure the provider with model-specific flags (call `super()` then setattr) |
| `maybe_modify_loaded_hf_weight()` | Dequantize, rename, or reshape HF weights before conversion |
| `maybe_modify_converted_hf_weight()` | Synthesize extra HF keys on export (e.g. `inv_freq`) |
| `megatron_to_hf_config()` | Build HF `config.json` for export |
| `hf_config_to_provider_kwargs()` | Override CONFIG_MAPPING behavior for specific fields |

**Accessing HF config in `mapping_registry()`:** The bridge instance has `self.hf_config`
available during conversion — it is set automatically by the dispatch system before
`mapping_registry()` is called. Use it when your mapping registry needs config-dependent
logic (e.g. dynamic MTP layer count, number of experts):

```python
def mapping_registry(self) -> MegatronMappingRegistry:
    hf_config = getattr(self, "hf_config", None)
    num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0) if hf_config else 0
    ...
```

Do **not** override `build_conversion_tasks()` to stash `self._hf_config` — that pattern is
deprecated.

#### Strategy 3: Custom provider subclass (VLMs only)

Most models do **not** need a provider file — the stock provider (e.g., `GPTModelProvider`, or
another base selected via `PROVIDER_CLASS`) is usually sufficient for LLMs. Only create a provider subclass when a VLM needs custom `provide()` logic to instantiate
a combined vision+language model:

```python
# src/megatron/bridge/models/<model>/<model>_provider.py
class MyVLModelProvider(GPTModelProvider):
    image_token_id: int = 0

    def provide(self, ...):
        # Custom model construction combining vision encoder + language decoder
        ...
```

The bridge then references it via `PROVIDER_CLASS = MyVLModelProvider` or instantiates it directly
in `provider_bridge()`.

#### When shared file changes ARE justified

Modify `param_mapping.py` or `model_bridge.py` only when the pattern is **reusable by 2+ model
families**. Examples of justified shared changes:

- `FusedExpertMapping` / `FusedGatedExpertMapping` — used by GLM, DeepSeek, OLMoE, etc.
- `RMSNorm2ZeroCenteredRMSNormMapping` — used by Gemma, Nemotron, etc.
- New `CONFIG_MAPPING` entries — when a standard HF config key maps to a standard provider attribute

If you're tempted to add a model-specific `if model_type == "..."` branch in shared code, or
pattern-matching on specific weight names in shared conversion logic, that's a signal to use a
local subclass or hook override instead.

### Update FLOPs calculator for new architectural blocks

If the model introduces a new computational block that differs from standard attention or MLP
(e.g., Gated DeltaNet / GDN linear attention, Multi-Token Prediction / MTP heads, Mamba SSM layers),
update the FLOPs calculator in `src/megatron/bridge/training/utils/flop_utils.py` so that
training throughput metrics (TFLOPs/GPU) are accurate.

**When to update:** Any time the new block has different FLOPs-per-token than standard self-attention
or standard MLP. Common cases:
- Linear attention variants (GDN, RetNet, RWKV) — replace the `O(s²)` attention term with the
  block's actual operation count
- MTP / speculative decoding heads — add FLOPs for the extra projection and norm layers
- SSM layers (Mamba) — different recurrence FLOPs than attention
- Novel MoE routing — may change the effective expert count

**How to update:**

1. Read the existing `transformer_flops()` function in `flop_utils.py` to understand the structure.
2. Add a conditional block gated on a config attribute (e.g., `experimental_attention_variant`,
   `mtp_num_layers`). Follow the existing MoE pattern for config validation — raise on invalid
   types, assert list lengths, and use direct attribute access instead of `getattr` with fallback
   defaults so that misconfigurations fail explicitly.
3. Compute the per-layer FLOPs for the new block and blend it with the standard attention term
   based on the layer pattern.
4. Add unit tests in `tests/unit_tests/training/utils/test_flop_utils.py` that verify:
   - New-block FLOPs differ from pure-attention baseline
   - Exact formula matches hand-computed expected values
   - Varying the block ratio (e.g., `linear_attention_freq`) changes FLOPs

Reference PR: [#2925 — GDN FLOPs calculator](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2925)
adds GDN support with both the calculator code and comprehensive tests.

## Phase 3: Recipe Support

Recipes provide pre-configured training settings for each model size.

**LLM recipes:** `src/megatron/bridge/recipes/<family>/<model>.py`
**VLM recipes:** `src/megatron/bridge/recipes/<family>/<model>.py`

Each recipe file defines functions for each model size + training mode:
- `<model>_<size>_sft_config()` — Full supervised fine-tuning
- `<model>_<size>_peft_config()` — LoRA/DoRA parameter-efficient fine-tuning
- `<model>_<size>_pretrain_config()` — Pretraining (LLM only, usually)

For detailed recipe patterns, see @skills/adding-model-support/recipe-patterns.md.

### Export checklist

1. Family `__init__.py` — import and add to `__all__`
2. Top-level `src/megatron/bridge/recipes/__init__.py` — wildcard import
3. `train_any_basic.py` — add to `config_map`, docstring, and `--model` choices

## Phase 4: Tests

### Unit tests (no GPU)

```text
tests/unit_tests/models/<model>/
├── __init__.py
├── test_<model>_bridge.py    # Mock HF config → verify provider mapping
└── test_<model>_provider.py  # (optional) Only if custom provider subclass exists
```

### Functional tests (GPU)

```text
tests/functional_tests/models/<model>/
├── __init__.py
├── test_<model>_conversion.py  # Toy model HF↔Megatron roundtrip
└── test_<model>_provider.py    # compare_provider_configs (optional)
```

For detailed test patterns, see @skills/adding-model-support/tests-and-examples.md.

## Phase 5: Docs and Examples

### Examples

LLM examples: `examples/models/<model>/`
VLM examples: `examples/models/vlm/<model>/`

```text
examples/models/<model>/          # LLM
examples/models/vlm/<model>/      # VLM
├── README.md
├── conversion.sh        # HF↔Megatron conversion commands (real model)
├── inference.sh         # Generation commands (real model, reasonable output)
├── slurm_sft.sh         # SFT training on SLURM
└── slurm_peft.sh        # PEFT training on SLURM
```

**Key deliverable requirement:** `conversion.sh` and `inference.sh` must target a real published model (e.g. `Qwen/Qwen3-8B`, not a toy). The inference script must produce reasonable output — for LLMs a coherent text continuation, for VLMs a plausible image description. This is the acceptance bar: conversion runs cleanly and generation makes sense.

### Documentation

Add a model page at `docs/models/<type>/<model>.md` covering:
- Supported variants and sizes
- Conversion commands
- Training examples (SFT, PEFT)
- Known limitations

## Verification Workflow

After implementing bridge support, prompt the user to run these commands on the cluster:

### 1. Smoke test (single GPU)

```bash
uv run python -c "
from megatron.bridge import AutoBridge
bridge = AutoBridge.from_hf_pretrained('<org>/<model>')
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
bridge.load_hf_weights(model)
for i, (name, tensor) in enumerate(bridge.export_hf_weights(model, cpu=True)):
    print(name, tuple(tensor.shape))
    if i > 10: break
"
```

### 2. Conversion roundtrip (multi-GPU)

```bash
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model <org>/<model> \
    --megatron-path /workspace/<model> \
    --torch-dtype bfloat16

uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model <org>/<model> \
    --megatron-path /workspace/<model>/iter_0000000 \
    --hf-path /workspace/<model>-hf-export
```

### 3. Generation test

For LLMs:
```bash
uv run python examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path <org>/<model> --prompt "Hello"
```

For VLMs:
```bash
uv run python examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path <org>/<model> \
    --image_path "https://example.com/image.jpeg" \
    --prompt "Describe this image."
```

### 4. Run tests

```bash
uv run python -m pytest tests/unit_tests/models/<model>/ -v
uv run python -m pytest tests/functional_tests/models/<model>/ -v --run-gpu
```

## Quick Decision Tree

```
User wants to add a model
│
├─ Has HF link? ─── No ──→ Ask for link (or config.json if private)
│
├─ Has text_config + vision_config? ─── Yes ──→ VLM path
│   ├─ Has Megatron vision encoder? ──→ Megatron encoder (Qwen3.5 pattern)
│   └─ No Megatron encoder ──→ HF encoder (Gemma3 pattern)
│
└─ No vision config ──→ LLM path (bridge only, no provider file)
    ├─ Standard GPT-style? ──→ Bridge with stock mappings
    └─ Custom layers? ──→ Bridge + local mapping subclasses / hook overrides
        ├─ Custom weight layout? ──→ Local mapping subclass in family dir
        └─ Custom import/export? ──→ Override bridge hooks (maybe_modify_*)
```
