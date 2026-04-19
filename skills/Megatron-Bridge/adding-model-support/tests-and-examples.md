# Test and Example Patterns

## Unit Tests

Location: `tests/unit_tests/models/<model>/`

### Bridge Unit Test

Mock the HF config and pretrained model, then verify `provider_bridge()` and `mapping_registry()`.

```python
import pytest
from unittest.mock import Mock
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM  # or .causal_lm

def _make_mock_config():
    """Create a mock HF config with model-specific attributes."""
    config = Mock()
    config.num_hidden_layers = 4
    config.hidden_size = 256
    config.intermediate_size = 512
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.vocab_size = 32000
    config.max_position_embeddings = 2048
    config.rope_theta = 10000.0
    config.rms_norm_eps = 1e-6
    config.tie_word_embeddings = False
    # For VLMs: add text_config and vision_config
    # config.text_config = _make_text_config()
    # config.vision_config = _make_vision_config()
    return config

def _make_mock_pretrained(config):
    pretrained = Mock(spec=PreTrainedVLM)  # or PreTrainedCausalLM
    pretrained.config = config
    return pretrained

class TestMyModelBridgeProviderBridge:
    @pytest.fixture
    def bridge(self):
        return MyModelBridge()

    @pytest.fixture
    def mock_pretrained(self):
        return _make_mock_pretrained(_make_mock_config())

    def test_provider_type(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert isinstance(provider, GPTModelProvider)  # or custom provider class if one exists

    def test_config_mapping(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.num_layers == 4
        assert provider.hidden_size == 256
        assert provider.num_attention_heads == 4

    def test_tie_word_embeddings(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.share_embeddings_and_output_weights == False

class TestMyModelBridgeMappingRegistry:
    @pytest.fixture
    def bridge(self):
        return MyModelBridge()

    def test_has_embedding_mapping(self, bridge):
        registry = bridge.mapping_registry()
        hf_params = {m.hf_param for m in registry.mappings if hasattr(m, 'hf_param')}
        assert "model.embed_tokens.weight" in hf_params

    def test_has_output_layer_mapping(self, bridge):
        registry = bridge.mapping_registry()
        megatron_params = {m.megatron_param for m in registry.mappings}
        assert any("output_layer" in p for p in megatron_params)
```

### Provider Unit Test (only if custom provider subclass exists)

Skip this if the bridge uses `GPTModelProvider` directly (most LLM bridges).
Only needed for VLM providers or LLM providers with custom fields/`provide()` logic.

```python
class TestMyModelProvider:
    def test_defaults(self):
        provider = MyModelProvider(
            num_layers=32, hidden_size=4096,
            num_attention_heads=32, num_query_groups=8,
        )
        assert provider.normalization == "RMSNorm"

    def test_tp_validation(self):
        with pytest.raises(ValueError):
            provider = MyModelProvider(
                num_query_groups=2,
                tensor_model_parallel_size=4,
            )
            provider.validate_parallelism()
```

### Skip conditions

```python
# Module-level skip for optional dependencies
pytestmark = pytest.mark.skipif(
    not _HAS_MODEL_CLASS,
    reason="transformers version does not support MyModel"
)

# Class-level skip
@pytest.mark.skipif(not _HAS_MOE_CLASS, reason="MoE class not available")
class TestMyMoEBridge:
    ...
```

## Functional Tests

Location: `tests/functional_tests/models/<model>/`

### Conversion Functional Test

Tests HF ↔ Megatron roundtrip on GPU with a toy model.

```python
import subprocess
import pytest

# Toy model config (reduced sizes for fast testing)
HF_TOY_MODEL_CONFIG = {
    "model_type": "my_model",
    "num_hidden_layers": 4,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 2048,
    "max_position_embeddings": 512,
    # ... model-specific fields
}

@pytest.fixture(scope="class")
def toy_model_path(tmp_path_factory):
    """Create a small HF model for testing."""
    from transformers import AutoConfig
    model_dir = tmp_path_factory.mktemp("toy_model")
    config = AutoConfig.for_model(**HF_TOY_MODEL_CONFIG)
    model = MyModelForCausalLM(config)
    model.save_pretrained(str(model_dir), safe_serialization=True)
    return str(model_dir)

@pytest.mark.run_only_on("GPU")
class TestMyModelConversion:
    @pytest.mark.parametrize("tp,pp", [(1, 1), (2, 1)])
    def test_roundtrip(self, toy_model_path, tp, pp, tmp_path):
        result = subprocess.run(
            [
                "uv", "run", "python", "-m", "torch.distributed.run",
                f"--nproc_per_node={tp * pp}",
                "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
                f"--hf-model-id={toy_model_path}",
                f"--output-dir={tmp_path}",
                f"--tp={tp}", f"--pp={pp}",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Conversion failed: {result.stderr}"
```

### VLM toy model creation

VLM toy models need both text and vision configs:

```python
HF_VLM_TOY_CONFIG = {
    "model_type": "my_vlm",
    "text_config": {
        "num_hidden_layers": 4,
        "hidden_size": 256,
        # ...
    },
    "vision_config": {
        "hidden_size": 128,
        "num_hidden_layers": 2,
        # ...
    },
    "image_token_id": 151655,
    "video_token_id": 151656,
    "tie_word_embeddings": False,
}
```

### MoE toy model: fuse expert weights

Some MoE models store experts in fused format. After creating the model, fuse:

```python
def _fuse_moe_expert_weights(model_dir):
    """Convert per-expert weights to fused gate_up_proj/down_proj layout."""
    # Load safetensors, reshape per-expert into combined tensors, save back
    ...
```

### Test marks

```python
@pytest.mark.run_only_on("GPU")       # Requires GPU
@pytest.mark.parametrize("tp,pp", [(2, 1)])  # Parallelism variants
@pytest.mark.skipif(...)               # Conditional skip
```

## Example Scripts

Example scripts target **real published models** (e.g. `Qwen/Qwen3-8B`), not toy configs.
The inference script must produce reasonable output — a coherent text completion for LLMs,
a plausible image description for VLMs. This is the acceptance bar for the deliverable.

### Conversion example (`examples/models/<type>/<model>/conversion.sh`)

```bash
#!/usr/bin/env bash
set -e

WORKSPACE=${WORKSPACE:-/workspace}
MODEL_NAME=<default-model-name>
HF_MODEL=<org>/${MODEL_NAME}
TP=1; PP=8; EP=1  # Adjust per model

# Import HF → Megatron
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model ${HF_MODEL} \
    --megatron-path ${WORKSPACE}/${MODEL_NAME} \
    --torch-dtype bfloat16

# Compare logits
uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/compare_hf_and_megatron/compare.py \
    --hf_model_path ${HF_MODEL} \
    --megatron_model_path ${WORKSPACE}/${MODEL_NAME} \
    --prompt "Hello, how are you?" \
    --tp ${TP} --pp ${PP} --ep ${EP}

# Export Megatron → HF
uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model ${HF_MODEL} \
    --megatron-path ${WORKSPACE}/${MODEL_NAME}/iter_0000000 \
    --hf-path ${WORKSPACE}/${MODEL_NAME}-hf-export

# Roundtrip validation
uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id ${HF_MODEL} --tp ${TP} --pp ${PP} --ep ${EP}
```

### Inference example (`examples/models/<type>/<model>/inference.sh`)

For LLMs:
```bash
uv run python examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path ${HF_MODEL} --prompt "Hello"
```

For VLMs:
```bash
uv run python examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path ${HF_MODEL} \
    --image_path "https://example.com/image.jpeg" \
    --prompt "Describe this image."
```

### VLM inference adds `--model_class` for non-default HF classes:
```bash
--model_class "MyModelForConditionalGeneration"
```

## Documentation Page

Create `docs/models/<type>/<model>.md`:

```markdown
# <Model Name>

## Supported Variants

| Variant | Parameters | HF Path |
|---------|-----------|---------|
| <Model>-7B | 7B | <org>/<model>-7B |
| <Model>-70B | 70B | <org>/<model>-70B |

## Conversion

\`\`\`bash
# HF → Megatron
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model <org>/<model> --megatron-path /workspace/<model>
\`\`\`

## Training

See `examples/models/<type>/<model>/slurm_sft.sh` and `slurm_peft.sh` for full Slurm scripts.
Single-node quick-start:

### SFT
\`\`\`bash
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe <model>_<size>_sft_config \
    checkpoint.pretrained_checkpoint=/workspace/models/<model> \
    model.tensor_model_parallel_size=<TP> \
    model.pipeline_model_parallel_size=<PP> \
    train.train_iters=1000 \
    train.global_batch_size=<GBS>
\`\`\`

### PEFT (LoRA)
\`\`\`bash
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe <model>_<size>_peft_config \
    checkpoint.pretrained_checkpoint=/workspace/models/<model> \
    model.tensor_model_parallel_size=<TP> \
    model.pipeline_model_parallel_size=<PP> \
    train.train_iters=1000 \
    train.global_batch_size=<GBS>
\`\`\`

## Known Limitations
- [List any known issues]
```
