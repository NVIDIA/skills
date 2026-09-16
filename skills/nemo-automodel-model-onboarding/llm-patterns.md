# Dense LLM Implementation Patterns

This document describes the standard patterns for adding a dense (non-MoE) causal language model to NeMo AutoModel.

Reference implementations:
- `components/models/llama/model.py` -- canonical dense LLM (inherits PreTrainedModel)
- `components/models/qwen2/model.py` -- dense LLM with attention/QKV bias

---

## Directory Structure

A dense LLM typically needs these files:

```
components/models/<name>/
  __init__.py
  model.py
  state_dict_adapter.py   # Only if HF weight names or tensor layouts need conversion
  rope_utils.py           # Only if RoPE differs from Llama
```

Start from the current Llama, Qwen2, or Qwen3 implementation and verify numerical
equivalence before reusing its attention or MLP pattern. These models preserve
separate HF-compatible projections; combined projections are not a requirement.

---

## Common Imports

```python
from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
```

---

## Attention and MLP Weight Layouts

Use the current `LlamaAttention` and `LlamaMLP` in
`components/models/llama/model.py` as references for separate projections:

- Attention has `q_proj`, `k_proj`, `v_proj`, and `o_proj`. Q weights have shape
  `[num_attention_heads * head_dim, hidden_size]`; K/V weights have shape
  `[num_key_value_heads * head_dim, hidden_size]`. Preserve the reference bias
  policy and apply the model's backend selection to each projection.
- SwiGLU has `gate_proj`, `up_proj`, and `down_proj`. Gate/up weights have shape
  `[intermediate_size, hidden_size]`; down weights have shape
  `[hidden_size, intermediate_size]`. Preserve the reference activation and bias.

Implement the model's own attention and MLP classes with these names and layouts
when they match HF. Thread `BackendConfig` through construction and use
`initialize_linear_module` for the selected linear backend. Verify forward and
backward parity for every rewritten layer.

If a model combines projections that HF stores separately, document its exact
packing order and provide a model-owned state dict adapter. Matching tensor sizes alone does not
prove the same axis order or interleaving. Do not fuse or split projections
solely to add or remove an adapter.

---

## Decoder Layer

Inherit from `GradientCheckpointingLayer` for activation checkpointing support:

```python
from transformers.modeling_layers import GradientCheckpointingLayer

class NewModelDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int, backend: BackendConfig):
        super().__init__()
        self.self_attn = NewModelAttention(config=config, layer_idx=layer_idx, backend=backend)
        self.mlp = NewModelMLP(config=config, backend=backend)
        self.input_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps,
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, cache_position=None,
                position_embeddings=None, **kwargs):
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

---

## Model Backbone (PreTrainedModel)

The backbone holds embeddings, layers, final norm, and RoPE:

```python
class NewModelModel(NewModelPreTrainedModel):
    def __init__(self, config, backend: BackendConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([
            NewModelDecoderLayer(config=config, layer_idx=i, backend=backend)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps,
        )
        self.rotary_emb = NewModelRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()
```

---

## ForCausalLM Class (Top-Level)

This is the main class that gets registered. It must inherit `HFCheckpointingMixin` and the model's `PreTrainedModel` base:

```python
class NewModelForCausalLM(HFCheckpointingMixin, NewModelPreTrainedModel):
    # Required attributes for TP/PP
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    @classmethod
    def from_config(cls, config, backend=None, **kwargs):
        return cls(config, backend, **kwargs)

    def __init__(self, config, backend=None):
        super().__init__(config)
        self.config = config
        self.backend = backend or BackendConfig()
        self.model = NewModelModel(config=config, backend=self.backend)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Separate HF-compatible projections need no state dict adapter.
        self.post_init()
        if getattr(config, "tie_word_embeddings", False):
            self.tie_weights()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self, *_args, **_kwargs):
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                logits_to_keep=0, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             return_dict=True, **kwargs)
        hidden_states = outputs.last_hidden_state

        # logits_to_keep optimization for training
        if isinstance(logits_to_keep, int) and logits_to_keep == 0:
            logits = self.lm_head(hidden_states)
        else:
            slice_indices = slice(-logits_to_keep, None)
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels,
                                     vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(loss=loss, logits=logits,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states)

# Module-level alias for registry
ModelClass = NewModelForCausalLM
```

### Required class attributes

| Attribute | Purpose | Example |
|-----------|---------|---------|
| `_tied_weights_keys` | Maps output embed to input embed for weight tying | `{"lm_head.weight": "model.embed_tokens.weight"}` |
| `_tp_plan` | Tensor parallelism sharding plan | `{"lm_head": "colwise_rep"}` |
| `_pp_plan` | Pipeline parallelism split plan | `{"lm_head": (["hidden_states"], ["logits"])}` |

### PreTrainedModel base class attributes

```python
class NewModelPreTrainedModel(PreTrainedModel):
    config_class = NewModelConfig  # or LlamaConfig, Qwen2Config, etc.
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NewModelDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
```

---

## BackendConfig Integration

`BackendConfig` controls which implementations to use for attention, linear layers, norms, and RoPE. Models receive it in `__init__` and pass it down.

Key backend fields:
- `backend.attn` -- attention implementation (`"sdpa"`, `"te"`, `"flex"`)
- `backend.linear` -- linear layer implementation (`"torch"`, `"te"`)
- `backend.rms_norm` -- RMSNorm implementation (`"torch"`, `"te"`)
- `backend.rope_fusion` -- whether to use fused RoPE kernels

Usage:
```python
from nemo_automodel.components.models.common import (
    initialize_rms_norm_module,
    initialize_linear_module,
)

# Norm: selects TE or torch implementation
self.norm = initialize_rms_norm_module(backend.rms_norm, hidden_size, eps=eps)

# Linear: selects the configured implementation while preserving weight names and shapes
self.proj = initialize_linear_module(backend.linear, in_features, out_features, bias=False)
```

For standard dense LLMs inheriting `PreTrainedModel`, the attention backend is controlled by HF's `_attn_implementation` (set via `attn_implementation` kwarg to `from_pretrained`). The model's attention class uses `ALL_ATTENTION_FUNCTIONS` to dispatch:

```python
attention_interface = eager_attention_forward
if self.config._attn_implementation != "eager":
    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
```

---

## State Dict Adapter

Most custom models need a `StateDictAdapter` from
`components/checkpoint/state_dict_adapter.py`. Implement `from_hf()` and
`to_hf()` in the model package and test both conversion directions.

Llama, Qwen2, and Qwen3 are exceptions: their HF names and tensor layouts match,
so they omit the adapter file and attribute. Keep the model's weight-tying logic.

Evaluate `supports_low_memory_dcp_load` per Section 2.6 of [SKILL.md](./SKILL.md).
For distributed conversions, test tensor shapes, DTensor placements, values,
and rank ownership with real distributed execution.

---

## __init__.py

Keep the init file simple -- just re-export the main class:

```python
from nemo_automodel.components.models.<name>.model import NewModelForCausalLM

__all__ = ["NewModelForCausalLM"]
```

---

## Registration in registry.py

Add to the `MODEL_ARCH_MAPPING` ordered dict in `_transformers/registry.py`:

```python
(
    "NewModelForCausalLM",
    ("nemo_automodel.components.models.new_model.model", "NewModelForCausalLM"),
),
```

The tuple format is `(module_path, class_name)`. An optional third element is a set of tags:

```python
(
    "NewModelForSequenceClassification",
    ("nemo_automodel.components.models.new_model.model", "NewModelForSequenceClassification", {"retrieval"}),
),
```

---

## _tp_plan and _pp_plan Format

### _tp_plan

Maps module names (relative to the ForCausalLM class) to TP sharding strategies:

```python
_tp_plan = {"lm_head": "colwise_rep"}
```

Common values:
- `"colwise_rep"` -- shard output dim (columns), replicate input; used for `lm_head`
- `"colwise"` -- shard output dim
- `"rowwise"` -- shard input dim (rows)

The TP plan for internal layers (attention projections, MLP) is typically handled by the parallelizer based on attribute names (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

### _pp_plan

Maps module names to `(input_names, output_names)` tuples for pipeline stage boundaries:

```python
_pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
```

---

## Module-Level ModelClass

Always set `ModelClass` at the bottom of `model.py`:

```python
ModelClass = NewModelForCausalLM
```

This allows the registry to lazy-import and find the class.
