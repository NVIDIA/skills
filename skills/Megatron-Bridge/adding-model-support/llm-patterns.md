# LLM Bridge Patterns

Reference implementations:
- Simple dense: Qwen2 (`src/megatron/bridge/models/qwen/qwen2_bridge.py`)
- MoE: GLM-4.5 (`src/megatron/bridge/models/glm/glm45_bridge.py`)
- MoE with custom layer spec: OLMoE (`src/megatron/bridge/models/olmoe/olmoe_bridge.py`)
- Advanced (YARN, MoE, provider re-wrap): GPT-OSS (`src/megatron/bridge/models/gpt_oss/`)

## Provider Pattern

Most bridges do **not** need a custom provider subclass. The base `provider_bridge()` uses
`CONFIG_MAPPING` to auto-create a `GPTModelProvider` from HF config. The bridge then sets
model-specific attributes directly on the returned provider instance.

```python
def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
    provider = super().provider_bridge(hf_pretrained)

    provider.normalization = "RMSNorm"
    provider.gated_linear_unit = True
    provider.position_embedding_type = "rope"
    provider.add_bias_linear = False
    provider.hidden_dropout = 0.0
    provider.autocast_dtype = torch.bfloat16

    # MoE settings (if applicable)
    provider.moe_grouped_gemm = True
    provider.moe_token_dispatcher_type = "alltoall"

    return provider
```

### When you DO need a provider subclass

Create a `GPTModelProvider` subclass only when:

1. **Extra dataclass fields** — The provider has fields not on `GPTModelProvider` (e.g., YARN
   RoPE params, custom MoE fields) that need to serialize into `run_config.yaml`.
2. **Custom `provide()` logic** — The model needs special instantiation (e.g., TE version
   checks, sink attention, custom layer specs that require runtime logic).
3. **Predefined size variants for recipes** — Hardcoded configs like `LlamaModelProvider8B`
   used by recipe functions (not by the bridge itself).

```python
@dataclass
class MyModelProvider(GPTModelProvider):
    yarn_rotary_scaling_factor: Optional[float] = None
    yarn_original_max_position_embeddings: Optional[int] = None

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        # Custom logic only if needed
        return super().provide(pre_process, post_process, vp_stage)
```

If the bridge uses a custom provider, re-wrap the base provider in `provider_bridge()`:

```python
def provider_bridge(self, hf_pretrained) -> MyModelProvider:
    provider = super().provider_bridge(hf_pretrained)
    provider = MyModelProvider(**{f.name: getattr(provider, f.name) for f in fields(provider)})
    provider.yarn_rotary_scaling_factor = ...
    return provider
```

### Predefined size variants (for recipes only)

Size-specific subclasses are used by recipes, not by the bridge:

```python
@dataclass
class MyModelProvider7B(MyModelProvider):
    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 8
    ffn_hidden_size: int = 14336
    vocab_size: int = 128256
```

## Bridge Pattern

```python
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping, GatedMLPMapping
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

@MegatronModelBridge.register_bridge(
    source=MyModelForCausalLM,    # HF class (or string "MyModelForCausalLM")
    target=GPTModel,               # Megatron target
    model_type="my_model",         # HF model_type
)
class MyModelBridge(MegatronModelBridge):

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        provider = super().provider_bridge(hf_pretrained)

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.hidden_dropout = 0.0
        provider.autocast_dtype = torch.bfloat16

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        return MegatronMappingRegistry(
            # Embeddings
            AutoMapping(
                megatron_param="embedding.word_embeddings.weight",
                hf_param="model.embed_tokens.weight",
            ),
            # Output layer
            AutoMapping(
                megatron_param="output_layer.weight",
                hf_param="lm_head.weight",
            ),
            # Final layernorm
            AutoMapping(
                megatron_param="decoder.final_layernorm.weight",
                hf_param="model.norm.weight",
            ),
            # QKV (fused)
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
            # Attention output projection
            AutoMapping(
                megatron_param="decoder.layers.*.self_attention.linear_proj.weight",
                hf_param="model.layers.*.self_attn.o_proj.weight",
            ),
            # MLP (gated)
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc2.weight",
                hf_param="model.layers.*.mlp.down_proj.weight",
            ),
            # Layer norms
            AutoMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                hf_param="model.layers.*.input_layernorm.weight",
            ),
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                hf_param="model.layers.*.post_attention_layernorm.weight",
            ),
        )
```

### Base CONFIG_MAPPING

The base class provides automatic mapping for common fields — no need to duplicate:

```text
(num_hidden_layers, num_layers), (hidden_size, hidden_size),
(intermediate_size, ffn_hidden_size), (num_attention_heads, num_attention_heads),
(num_key_value_heads, num_query_groups), (head_dim, kv_channels),
(vocab_size, vocab_size), (max_position_embeddings, seq_length),
(rms_norm_eps, layernorm_epsilon), (rope_theta, rotary_base),
(tie_word_embeddings, share_embeddings_and_output_weights),
(attention_bias, add_qkv_bias), (mlp_bias, add_bias_linear),
```

### MoE weight mappings

For models with Mixture of Experts, use expert-specific mappings:

```python
ExpertMLPGateUpProjMapping(
    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
    up="model.layers.*.mlp.experts.*.up_proj.weight",
),
ExpertMLPDownProjMapping(
    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
),
AutoMapping(
    megatron_param="decoder.layers.*.mlp.router.weight",
    hf_param="model.layers.*.mlp.gate.weight",
),
```

### Optional weight modification hooks

Override these for special handling (e.g., quantized weights, expert layout):

```python
def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
    """Transform HF weights before loading into Megatron (e.g., dequantize)."""
    return hf_state_dict[hf_param]

def maybe_modify_converted_hf_weight(self, task, converted_weights_dict, hf_state_dict):
    """Transform weights after Megatron→HF conversion (e.g., merge expert shards)."""
    return converted_weights_dict
```

## Registration Options

| Parameter | Required | Description |
|-----------|----------|-------------|
| `source` | Yes | HF model class or string class name |
| `target` | Yes | Megatron model class (usually `GPTModel`) |
| `provider` | No | Provider class (defaults to `GPTModelProvider`) |
| `model_type` | No | HF `model_type` string for export config |

If `source` is a string (model not importable), the bridge is matched by class name.
