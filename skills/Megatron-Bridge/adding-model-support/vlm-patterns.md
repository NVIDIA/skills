# VLM Bridge Patterns

Reference implementations:
- **Megatron vision encoder:** Qwen3.5-VL (`src/megatron/bridge/models/qwen_vl/`)
- **HF vision encoder:** Gemma3-VL (`src/megatron/bridge/models/gemma_vl/`)

## Provider Pattern

Subclass `GPTModelProvider`. VLM providers add vision-specific fields on top of standard LLM fields.

```python
@dataclass
class MyVLModelProvider(GPTModelProvider):
    # Vision config (passed as a HF config object)
    vision_config: Optional[Any] = None

    # VLM-specific token IDs
    image_token_id: Optional[int] = None
    video_token_id: Optional[int] = None

    # Freeze options
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    # Whether to use HF vision model (vs Megatron)
    use_hf_vision_model: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> MyVLModel:
        # Build language layer spec
        language_transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(...)
        # Build vision config if needed
        # Instantiate combined model
        model = MyVLModel(config=self, ...)
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(self.freeze_language_model, self.freeze_vision_model, self.freeze_vision_projection)
        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None):
        """Returns language-only model (for text-only inference)."""
        return GPTModel(config=self, ...)

    def validate_parallelism(self):
        if self.num_query_groups < self.tensor_model_parallel_size:
            raise ValueError(f"TP ({self.tensor_model_parallel_size}) must be <= num_query_groups ({self.num_query_groups})")
```

### Key provider fields by source

Read these from the correct config level:

| Field | Source (VLM) | Notes |
|-------|-------------|-------|
| `num_layers`, `hidden_size`, `ffn_hidden_size` | `text_config` | Core architecture |
| `num_attention_heads`, `num_key_value_heads` | `text_config` | Attention config |
| `vocab_size`, `max_position_embeddings` | `text_config` | Tokenizer/position |
| `rope_theta` | `text_config` | RoPE |
| `tie_word_embeddings` | **top-level** `hf_config` | CRITICAL: not text_config |
| `vision_config` | **top-level** `hf_config` | Vision encoder config |
| `image_token_id`, `video_token_id` | **top-level** `hf_config` | Special token IDs |

## Bridge Pattern

```python
@MegatronModelBridge.register_bridge(
    source="MyModelForConditionalGeneration",   # HF class name (string if not importable)
    target=MyVLModel,                            # Megatron model class
    provider=MyVLModelProvider,                  # Provider class
    model_type="my_model",                       # HF model_type for export
)
class MyVLBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> MyVLModelProvider:
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config

        # Map text config to provider kwargs using base class helper
        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        provider = MyVLModelProvider(**provider_kwargs)

        # CRITICAL: tie_word_embeddings from top-level config
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)

        # Vision config
        provider.vision_config = hf_config.vision_config

        # VLM-specific fields from top-level config
        provider.image_token_id = getattr(hf_config, "image_token_id", None)
        provider.video_token_id = getattr(hf_config, "video_token_id", None)

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        return MegatronMappingRegistry(
            # Language model mappings (prefixed with language_model.*)
            AutoMapping(megatron_param="language_model.embedding.word_embeddings.weight",
                       hf_param="model.embed_tokens.weight"),
            AutoMapping(megatron_param="language_model.output_layer.weight",
                       hf_param="model.lm_head.weight"),
            # ... language decoder layers ...
            QKVMapping(
                megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.language_model.layers.*.self_attn.q_proj.weight",
                k="model.language_model.layers.*.self_attn.k_proj.weight",
                v="model.language_model.layers.*.self_attn.v_proj.weight",
            ),
            # Vision model mappings
            AutoMapping(megatron_param="vision_model.patch_embed.proj.**",
                       hf_param="model.visual.patch_embed.proj.**"),
            # ... vision layers ...
        )
```

### Import types

```python
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM      # VLM
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM  # LLM
```

## VLM Model Class Patterns

### Option A: Megatron Vision Encoder (Qwen3.5 pattern)

Both vision and language use Megatron modules. Full parallelism support.

```python
class MyVLModel(MegatronModule):
    def __init__(self, config, pre_process=True, post_process=True, ...):
        if pre_process:
            self.vision_model = MyVisionModel(config.vision_config, ...)
        self.language_model = MyGPTModel(config, ...)

    def forward(self, input_ids, pixel_values, image_grid_thw, ...):
        # 1. Vision: pixel_values → vision_embeds
        vision_embeds = self.vision_model(pixel_values, image_grid_thw)
        # 2. Text embeddings
        text_embeds = self.language_model.embedding(input_ids)
        # 3. Scatter vision into text at image token positions
        combined = text_embeds.clone()
        combined[vision_mask] = vision_embeds
        # 4. Language model forward
        return self.language_model(decoder_input=combined, ...)

    def freeze(self, freeze_language, freeze_vision, freeze_projection):
        if freeze_language:
            for p in self.language_model.parameters(): p.requires_grad = False
        if freeze_vision:
            for p in self.vision_model.parameters(): p.requires_grad = False
        # projection freeze logic
```

### Option B: HF Vision Encoder (Gemma3 pattern)

HF vision encoder + Megatron projector + Megatron language model. Simpler to implement.

```python
class MyVLModel(MegatronModule):
    def __init__(self, config, pre_process=True, post_process=True, ...):
        if pre_process:
            self.vision_tower = AutoModel.from_config(config.vision_config)
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_tower)
            self.multi_modal_projector = MyProjector(config)
        self.language_model = config.provide_language_model(pre_process, post_process)

    def forward(self, input_ids, pixel_values, ...):
        text_embeds = self.language_model.embedding(input_ids)
        if pixel_values is not None:
            image_features = self.vision_tower(pixel_values).pooler_output
            image_features = self.multi_modal_projector(image_features)
            text_embeds.masked_scatter_(special_image_mask, image_features)
        return self.language_model(decoder_input=text_embeds, ...)
```

## Weight Mapping Naming Conventions

VLM weight names typically have these prefixes:

| Megatron prefix | HF prefix | Component |
|----------------|-----------|-----------|
| `language_model.*` | `model.language_model.*` or `model.layers.*` | Text decoder |
| `language_model.embedding.*` | `model.embed_tokens.*` | Text embeddings |
| `language_model.output_layer.*` | `model.lm_head.*` or `lm_head.*` | Output head |
| `vision_model.*` | `model.visual.*` or `vision_tower.*` | Vision encoder |

Check the actual HF model's `state_dict()` keys to determine exact naming.

## Common Mapping Types for VLMs

| Mapping Class | Use Case |
|--------------|----------|
| `AutoMapping` | 1:1 name mapping (most weights) |
| `QKVMapping` | Fused Q/K/V projections |
| `ConcatenatedQKVMapping` | Vision QKV (different from language) |
| `GatedMLPMapping` | gate_proj + up_proj → linear_fc1 |
| `ReplicatedMapping` | Weights replicated across TP ranks (e.g. patch_embed) |
| `ExpertMLPGateUpProjMapping` | MoE gate+up projections |
| `ExpertMLPDownProjMapping` | MoE down projections |
