---
name: nemo-automodel-model-onboarding
description: Guide for onboarding new model architectures into NeMo AutoModel, including architecture discovery, implementation patterns, registration, and validation.
when_to_use: Adding or modifying model architecture support in NeMo AutoModel, such as LLM/VLM/MoE model files, custom layers, state-dict adapters, registry entries, Hugging Face config mapping, or capability flags.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - nemo-automodel
    - model-onboarding
---

# Adding Model Support to NeMo AutoModel

## Purpose

This skill guides implementation of new model architectures in NeMo AutoModel. Follow the five phases in order.
<!-- NVSkills signature refresh requested after PR #2998 (2026-07-31). -->

## Instructions

When answering an onboarding question, keep the response in this order:

1. Classify the architecture from `config.json`.
2. Name the exact implementation files under `components/models/<name>/`.
3. Identify registry and optional custom-config updates.
4. State the validation tests that must be added before full checkpoint use.

For conceptual onboarding questions, answer from this skill without opening the
pattern files unless the user asks you to edit code. Mention pattern filenames
as references, then give the direct checklist.

Use direct action verbs: classify the model, name the files, map the weights,
register the class, and add tests. Do not discuss distributed strategy,
launcher configuration, or general recipe authoring unless the user explicitly
connects it to onboarding a new architecture.

## Examples

Use these compact answer patterns for common questions:

- Dense causal LM: classify as dense only when `architectures` contains a
  `ForCausalLM` class and expert fields such as `num_local_experts`,
  `n_routed_experts`, or `num_experts_per_tok` are absent. Create
  `components/models/<name>/model.py` and `__init__.py`; add `state_dict_adapter.py`
  only for checkpoint weight conversion and `config.py` only if needed. Register `MODEL_ARCH_MAPPING` in
  `_transformers/registry.py`, add example YAML, and add tiny-config unit tests
  plus layer-equivalence tests for rewritten layers.
- MoE state dict: identify expert fields in `config.json`, reference
  `moe-patterns.md`, map router tensors separately, preserve routed-expert
  index order, map routed experts, shared experts, and gate/up/down projections,
  add adapter key-map tests and tiny-config numerical equivalence tests, and do
  not rely only on `from_pretrained()` or silent tensor reshapes.
- VLM onboarding: classify as VLM only when `vision_config`, `text_config`, and
  a `ForConditionalGeneration` architecture are present. Reference
  `vlm-patterns.md` and existing VLM implementations such as `mistral4`,
  `kimivl`, or `kimi_k25_vl`; check text backbone, vision tower, projector,
  processor assumptions, text and vision checkpoint compatibility (adapter mappings when needed),
  registry registration, and tiny image-text tests before full checkpoints.
  Do not treat VLM onboarding as a pure causal-LM path or skip processor/image
  tests.

For MoE state-dict and VLM questions, apply the checklists in Sections 2.4 and 2.5.

## Routing Boundary

Use this skill only when the user is adding or modifying model architecture support: model files, custom layers, state-dict adapters, Hugging Face config mapping, registry entries, or model capability flags.

Do not use this skill for standalone training recipe YAML questions about optimizers, datasets, schedulers, validation datasets, or trainer wiring unless they are explicitly part of onboarding a new model architecture. Those recipe questions belong to the nemo-automodel-recipe-development skill.

In-scope examples:

- "Add support for a new Hugging Face causal LM architecture."
- "Map MoE router and expert weights from a Hugging Face checkpoint."
- "Register a new model class in NeMo AutoModel."

Out-of-scope examples:

- "Write a finetuning recipe YAML with optimizer and dataset sections."
- "Choose FSDP2, DDP, tensor parallel, or context parallel settings."
- "Configure Slurm, SkyPilot, containers, mounts, or launch dispatch."

## Phase 1: Discovery

Before writing code, gather information about the target model.

### 1.1 Fetch HuggingFace config.json

Download the model's `config.json` from the HuggingFace Hub (or use `AutoConfig.from_pretrained`). Key fields to extract:

- `architectures` -- determines the class name and registration key (e.g., `"LlamaForCausalLM"`, `"Qwen3MoeForCausalLM"`, `"Mistral3ForConditionalGeneration"`)
- `model_type` -- used for custom config registration in `_CUSTOM_CONFIG_REGISTRATIONS` if HF does not have a built-in config class
- `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads` -- sizing
- `vocab_size` -- needed for tiny test configs
- `tie_word_embeddings` -- the saved setting in each supported checkpoint; do not infer it from a bare config constructor
- `hidden_act` -- activation function (e.g., `"silu"` for SwiGLU)

### 1.2 Determine model type

| Type | Indicators | Pattern file |
|------|-----------|-------------|
| **Dense LLM** | `ForCausalLM` in architectures, no expert fields | [llm-patterns.md](./llm-patterns.md) |
| **MoE LLM** | `n_routed_experts`, `num_local_experts`, `num_experts_per_tok` in config | [moe-patterns.md](./moe-patterns.md) |
| **VLM** | `ForConditionalGeneration` in architectures, has `vision_config` + `text_config` | [vlm-patterns.md](./vlm-patterns.md) |

### 1.3 Check for existing similar architectures

Look in `components/models/` for architectures with similar attention or MLP patterns:

```
components/models/
  llama/           # Standard GQA + SwiGLU with separate HF-compatible projections
  qwen2/           # Same as Llama but with attention bias + QKV bias
  baichuan/        # ALiBi attention variant
  deepseek_v3/     # MLA attention + MoE (DeepSeek-style grouped experts)
  mistral4/        # MLA + MoE + VLM (Pixtral vision)
  kimivl/          # DeepSeek-V3 backbone + MoonVit vision
  kimi_k25_vl/     # Updated KimiVL with different projector
  qwen3_moe/       # Qwen3 with MoE layers
  nemotron_v3/     # Hybrid mamba-attention
```

### 1.4 Identify custom components

Check whether the model needs:

- **Custom attention**: GQA (standard), MLA (DeepSeek/Mistral4), sliding window, bidirectional
- **Custom RoPE**: Standard (Llama), YaRN scaling, NTK-aware, complex-number (DeepSeek)
- **Custom normalization**: RMSNorm (standard), LayerNorm, different eps values
- **Custom MLP**: SwiGLU (standard), GeGLU, ReLU-squared, MoE routing
- **Custom config class**: Needed only if HF `AutoConfig` cannot parse the model's `config.json` (check `auto_map` field)

### 1.5 Note dimensions for test config

For unit tests, create a tiny config. Target: ~1M parameters or less.

```python
# Example tiny config for a Llama-like model:
tiny_config = LlamaConfig(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=256,
    max_position_embeddings=128,
)
```

---

## Phase 2: Implementation

### 2.1 Create directory structure

```
components/models/<name>/
  __init__.py
  model.py
  state_dict_adapter.py # Only if HF weight names or tensor layouts need conversion
  config.py            # Only if HF config is insufficient
  layers.py            # Only for MoE / MLA / other non-standard layers
  rope_utils.py        # Only for custom RoPE
```

### 2.2 Implementation order

Implement files in dependency order:

1. **config.py** (if needed) -- Custom `PretrainedConfig` subclass
2. **rope_utils.py** (if needed) -- RoPE implementation
3. **layers.py** (if needed) -- Attention, MLP, decoder block classes
4. **model.py** -- The main `ForCausalLM` (or `ForConditionalGeneration`) class
5. **state_dict_adapter.py** (if needed) -- HF weight conversion. Treat checkpoint I/O
   performance as part of the implementation, and evaluate the low-memory DCP
   capability as described in Section 2.6.
6. **__init__.py** -- Re-export the main model class

See the pattern files for detailed implementation guidance:

- Dense LLM: [llm-patterns.md](./llm-patterns.md)
- MoE: [moe-patterns.md](./moe-patterns.md)
- VLM: [vlm-patterns.md](./vlm-patterns.md)
- Capabilities and fp32 precision: [capabilities-and-precision.md](./capabilities-and-precision.md)

Most custom models need `state_dict_adapter.py` for HF weight conversion.
Omit the file and attribute only when HF names and tensor layouts already match
across supported backend/config variants, as in Llama, Qwen2, and Qwen3.
Weight tying remains the model's responsibility (Section 2.3).

### 2.3 Causal LM weight tying

Every registered model class with a causal `lm_head` must:

- Declare `tie_word_embeddings_support: TieSupport` as `BOTH`, `TIED_ONLY`, or
  `UNTIED_ONLY`.
- Call `reject_unsupported_tie_word_embeddings(type(self), config)` at the top
  of `__init__`, using the original config before unwrapping `text_config` or
  `thinker_config`.

Only classes with no causal LM head may be explicitly exempted from the registry
test.

Choose the policy from the implementation and the actual supported checkpoint
configs, not from a bare config constructor:

- `BOTH`: tied and untied configurations are both supported.
- `TIED_ONLY`: only a tied configuration is supported.
- `UNTIED_ONLY`: only an untied configuration is supported.

Runtime helpers must treat `TIED_ONLY` and `UNTIED_ONLY` as authoritative and
only resolve a per-checkpoint config flag for `BOTH`. All current `BOTH` VLMs
honor the outer `tie_word_embeddings` flag, so do not add a model-specific
resolver until a supported `BOTH` model actually requires another config path.

For `BOTH` and `TIED_ONLY`, always declare `_tied_weights_keys` and implement
`tie_weights()` with the actual `lm_head` and input-embedding FQNs. Do not rely
on inherited Hugging Face tying, and re-tie after any language-model swap.

Add policy-specific tests:

- `BOTH`: tied aliases; untied does not alias.
- `TIED_ONLY`: tied aliases; untied is rejected.
- `UNTIED_ONLY`: weights stay separate; tied is rejected.

Do not tie architectures with intentionally separate heads, asymmetric vocab
sizes, or stages that do not own both tensors.

For `from_pretrained`, the checkpoint's saved `tie_word_embeddings` value is
authoritative, even for `BOTH`. The `NeMoAuto*` bridge rejects flips in either
direction. A model-owned `from_pretrained` that bypasses that bridge must call
`reject_tie_word_embeddings_flip(checkpoint_config, requested_config,
model_class_name)`.

### 2.4 MoE state-dict adapter checklist

For MoE models, verify all weights below. When their HF and native layouts differ,
the adapter must explicitly map:

- Router weights, including gate bias or correction-bias tensors when the Hugging Face model has them.
- Expert weights, preserving expert index order across local and routed experts.
- Gate/up/down projections, including combined or split projection layouts.
- Shared experts separately from routed experts when the architecture has both.

Add tests that assert expected key mappings and run numerical equivalence with tiny configs before trying full checkpoints.

Do not use these shortcuts:

- Do not validate the adapter only by calling `from_pretrained()`.
- Do not accept missing or extra expert keys without an explicit mapping reason.
- Do not change dtype, transpose dimensions, or reshape tensors unless the HF
  and NeMo layouts require it and a test proves the conversion is reversible.
- Do not skip router or shared-expert tests because dense-layer tests pass.

### 2.5 VLM onboarding checklist

For VLMs, confirm the Hugging Face config has `vision_config` and `text_config`
and that `architectures` points to a conditional-generation class. Start from
the closest VLM pattern file, usually [vlm-patterns.md](./vlm-patterns.md), and
compare existing implementations such as `mistral4`, `kimivl`, or
`kimi_k25_vl`.

The implementation should explicitly cover:

- Text backbone, vision tower, projector, and processor or image preprocessing assumptions.
- Checkpoint compatibility for both text and vision modules, with adapter mappings where needed.
- Registration of the `ForConditionalGeneration` class in `_transformers/registry.py`.
- Tiny tests that exercise image-text inputs and verify checkpoint load/export, plus adapter round-trip when present.

### 2.6 Checkpoint I/O performance

Treat checkpoint performance as an implementation requirement, not a later
optimization. For every new or materially changed state-dict adapter, evaluate
both latency and peak host/device memory for loading and for any save or export
path the change affects. In particular:

- Avoid full-checkpoint or model-sized temporary copies when tensors can load
  directly into final model storage or be transformed in bounded parts.
- Keep distributed reads and conversions rank-local when a rank needs only its
  shard; do not materialize a global tensor on every rank unnecessarily.
- Avoid repeated tensor merges, copies, full-heap garbage collections, shard
  scans, or file opens inside model-sized loops.
- Record representative before/after latency and peak-memory evidence for an
  optimized path, including the model, dtype, backend, and topology.

Every adapter must evaluate `supports_low_memory_dcp_load`. Set
`_supports_low_memory_dcp_load = True` only when most checkpoint tensors write
directly into final model storage and every remaining allocating conversion has
a small, bounded temporary footprint for every runtime variant that reports
support. Keep it false when a backend, topology, dtype, quantization mode, or
model option requires model-sized rebuilding. A false value selects the safe
fallback; it does not mean checkpoint loading is unsupported.

An opt-in needs focused tests that write sentinel values through direct
destinations and prove the final model storage changes, bound any allocating
conversions, and verify unsafe runtime variants report the capability as false.
This storage test is also a correctness requirement: a false positive can cause
the adapter to treat a temporary tensor as loaded in place and skip rebuilding
the real parameter.

### 2.7 Register in registry

Add the model to `MODEL_ARCH_MAPPING` in `_transformers/registry.py`:

```python
# In _transformers/registry.py
MODEL_ARCH_MAPPING = OrderedDict([
    # ... existing entries ...
    (
        "NewModelForCausalLM",
        ("nemo_automodel.components.models.new_model.model", "NewModelForCausalLM"),
    ),
])
```

If the model has a custom config class with `auto_map` in its `config.json`, also register in `_CUSTOM_CONFIG_REGISTRATIONS`:

```python
_CUSTOM_CONFIG_REGISTRATIONS: Dict[str, Tuple[str, str]] = {
    # ... existing entries ...
    "new_model": ("nemo_automodel.components.models.new_model.configuration", "NewModelConfig"),
}
```

### 2.8 Declare capabilities and precision-sensitive params

Every class registered in `MODEL_ARCH_MAPPING` must declare parallelism
capabilities, either with a static nested `ModelCapabilities` dataclass or a
variant-aware `get_capabilities(cls, config)` method. Pick exactly one pattern.
Capabilities should reflect recipe YAMLs that have been validated end to end.

If the model has precision-sensitive parameters such as Mamba `A_log` /
`dt_bias`, MoE sigmoid gate bias, attention-sink bias, or per-head `scale`,
declare `_keep_in_fp32_modules_strict` so sharding keeps those params in fp32
compute. See [capabilities-and-precision.md](./capabilities-and-precision.md)
for examples, variant dispatch rules, and frozen-submodule dtype guidance.

---

## Phase 3: Onboarding Example Config

This phase is only for adding a minimal example config that proves the newly
onboarded architecture can load and run. Use nemo-automodel-recipe-development for general
recipe authoring or existing recipe modifications.

### 3.1 Create example YAML config

Create an example config under `examples/llm_finetune/<name>/` (or `examples/vlm_finetune/<name>/`):

For new full-parameter Adam/AdamW examples, set `model.dtype: float32`.
See [training precision](../nemo-automodel-recipe-development/SKILL.md#full-parameter-training-precision)
for compute precision and other training modes.

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: <org>/<model-name>
  dtype: float32

trainer:
  max_steps: 100
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1

# ... data, optimizer config ...
```

### 3.2 Verify model loads

Test that the model loads from a HuggingFace checkpoint:

```python
from nemo_automodel import NeMoAutoModelForCausalLM

model = NeMoAutoModelForCausalLM.from_pretrained("<org>/<model-name>")
```

### 3.3 Test with tiny config first

Before using full-size models, verify with a tiny config (1-2 layers, small hidden dim) to catch shape mismatches early.

## Phase 4: Tests

Create `tests/unit_tests/models/<name>/` and cover the checks below before
loading full checkpoints:

- Forward-shape smoke test with a tiny config.
- State-dict adapter round-trip, when present: `from_hf -> to_hf` preserves
  mapped names, shapes, dtypes, and values.
- HF load/export and native save/reload preserve weights and ties, with or
  without an adapter; see `tests/unit_tests/checkpoint/test_native_hf_state_dict.py`.
- Layer equivalence tests for every rewritten attention, MLP, normalization,
  RoPE, or MoE layer. Use the model dtype from config, identical seeded weights,
  identical inputs, and dtype-appropriate `torch.allclose` tolerances.
- Short functional test that verifies loss decreases over a few training steps.

---

## Phase 5: Documentation

### 5.1 Update model coverage page

Edit the appropriate file in `docs/model-coverage/`:
- LLM/MoE: `docs/model-coverage/llm/index.md`
- VLM: `docs/model-coverage/vlm/index.md`

Add a row with the model name, supported features (TP, PP, FSDP, LoRA, QLoRA), and any limitations.

---

## Phase 6: Parity Testing

After implementation and unit tests are complete, run the full parity-testing
workflow to verify that the new model produces numerically equivalent results to
the reference HuggingFace implementation.

Run three levels of comparison:

1. State-dict round-trip: load a reference HuggingFace checkpoint, convert it
   into the NeMo AutoModel layout, export it back, and verify that all mapped
   tensors match the reference names, shapes, dtypes, and values within the
   expected tolerance.
2. Component-level parity: compare rewritten attention, MLP, normalization,
   RoPE, and MoE components against the HuggingFace implementation with fixed
   seeds and identical dtype.
3. End-to-end forward pass: run the full NeMo AutoModel and HuggingFace model
   on the same tokenized input and compare logits, hidden states, and loss.

Do not skip this phase. A model that passes unit tests can still diverge from HF
due to subtle weight-conversion bugs, backend differences, or RoPE mismatches
that only surface in a full parity comparison.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `_transformers/registry.py` | `MODEL_ARCH_MAPPING` and `_CUSTOM_CONFIG_REGISTRATIONS` |
| `components/models/common/__init__.py` | Exports `BackendConfig`, `HFCheckpointingMixin`, and backend construction utilities |
| `components/models/llama/model.py` | Separate attention and MLP projections with HF-compatible weights |
| `components/checkpoint/state_dict_adapter.py` | Optional `StateDictAdapter` conversion contract |
| `components/models/common/hf_checkpointing_mixin.py` | `HFCheckpointingMixin` for save/load |
| `components/models/common/utils.py` | `BackendConfig`, `initialize_rms_norm_module`, `initialize_linear_module`, `get_rope_config` |
| `components/moe/config.py` | `MoEConfig` dataclass |
| `components/moe/fsdp_mixin.py` | `MoEFSDPSyncMixin` for distributed expert handling |
| `components/moe/layers.py` | `MoE` layer, `MLP` (dense) for MoE blocks |
| `components/moe/experts.py` | `GroupedExperts`, `GroupedExpertsDeepEP`, `GroupedExpertsTE` |

---

## Checklist

- [ ] Fetched and analyzed `config.json` from HuggingFace
- [ ] Determined model type (dense LLM / MoE / VLM)
- [ ] Identified custom components (attention, RoPE, normalization, MLP)
- [ ] Created `components/models/<name>/` directory
- [ ] Implemented config.py (if custom config needed)
- [ ] Implemented layers.py (if custom layers needed)
- [ ] Implemented rope_utils.py (if custom RoPE needed)
- [ ] Implemented model.py with `HFCheckpointingMixin`
- [ ] Implemented state_dict_adapter.py if needed; evaluated checkpoint latency and peak memory per Section 2.6
- [ ] For adapters, evaluated `supports_low_memory_dcp_load`; any opt-in proves direct destinations reach model storage, bounds
  allocating conversions, and reports `False` for unsafe variants
- [ ] Implemented __init__.py with re-export
- [ ] Registered in `MODEL_ARCH_MAPPING` in `_transformers/registry.py`
- [ ] Registered custom config in `_CUSTOM_CONFIG_REGISTRATIONS` (if applicable)
- [ ] Declared `ModelCapabilities` nested dataclass (static) OR `get_capabilities(cls, config)` classmethod (variant dispatch, e.g. ERNIE-4.5 MoE vs dense) — never both, never neither
- [ ] Declared `TieSupport` and called the constructor guard for every class with a causal `lm_head` (or added an explicit no-head exemption) -- see §2.3
- [ ] Added explicit `_tied_weights_keys` and `tie_weights()` for `BOTH` / `TIED_ONLY`, plus policy-specific alias and rejection tests -- see §2.3
- [ ] Guarded any model-owned `from_pretrained` that bypasses the `NeMoAuto*` bridge against checkpoint flips -- see §2.3
- [ ] Created example YAML config
- [ ] Verified model loads via `NeMoAutoModelForCausalLM.from_pretrained()`
- [ ] Created unit tests (forward shape, state_dict round-trip)
- [ ] Declared `_keep_in_fp32_modules_strict` for every intrinsically-fp32 param (SSM `A_log`/`dt_bias`, Mamba `D` when reference-fp32, MoE gate bias, attention-sink bias, `scale`, …) — see §2.8
- [ ] Created layer equivalence tests for every rewritten layer (matching model dtype)
- [ ] Created functional tests (training loss decreases)
- [ ] Updated docs/model-coverage page
- [ ] Ran state-dict round-trip, component parity, and E2E forward-pass parity checks
- [ ] Set `ModelClass = <Name>ForCausalLM` at module bottom
