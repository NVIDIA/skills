---
name: parity-testing
description: Structured framework for verifying numerical parity of HF<->MCore weight conversions. References existing tools and the add-model-support skill.
when_to_use: Debugging weight mismatches, verifying HF↔MCore checkpoint round-trips, choosing verification tools, or investigating a commit that changed weight conversion and caused parity failures; 'weights don't match', 'parity test', 'roundtrip check', 'logit equivalence'.
---

# Parity Testing for Megatron Bridge

This skill provides the decision framework for choosing the right
verification tool and interpreting results. For the full model onboarding
workflow (which includes parity testing as milestones 1 and 2), see the
`add-model-support` skill.

## Quick Decision: Which Tool to Run

| What you want to verify | Tool | GPU? | When to use |
|---|---|---|---|
| All weights round-trip exactly (single GPU) | `hf_megatron_roundtrip.py` | No | First check after writing a bridge |
| Weights round-trip with TP/PP/EP | `hf_megatron_roundtrip_multi_gpu.py` | Yes | After single-GPU passes |
| Forward-pass logit equivalence | `compare_hf_and_megatron/compare.py` | Yes | After round-trip passes |
| Text generation sanity | `hf_to_megatron_generate_text.py` | Yes | Large models that OOM compare.py |
| Programmatic weight check | `weights_verification_table()` | Yes | Inside Python scripts |
| VLM generation sanity | `hf_to_megatron_generate_vlm.py` | Yes | VLM models |

All tools live under `examples/conversion/`.

## 3-Level Test Strategy

### Level 1: State Dict Round-Trip (exact match)

The fastest and most fundamental check. If mappings can't perfectly
round-trip weights, nothing else will work.

```bash
# Single-GPU round-trip
uv run python examples/conversion/hf_megatron_roundtrip.py \
    --hf-model-id <org>/<model>

# Multi-GPU with TP=2
uv run python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id <org>/<model> --tp 2

# Multi-GPU with PP=2
uv run python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id <org>/<model> --pp 2
```

**Expected:** Every weight shows "Matches Original: checkmark". Any "X"
means the param mapping has an error.

**Tolerance:** Exact match (`max_diff == 0.0`). Round-trip conversions are
pure tensor reshaping — no floating-point arithmetic is involved.

For programmatic verification inside scripts, use the built-in verifier:

```python
from megatron.bridge.models.conversion.utils import weights_verification_table
weights_verification_table(bridge, hf_pretrained, megatron_model)
```

### Level 2: Forward-Pass Parity (GPU / bfloat16)

After round-trip passes, verify that converted weights produce identical
forward-pass output.

```bash
# Compare logits (loads both HF and Megatron models)
uv run python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/compare_hf_and_megatron/compare.py \
    --hf_model_path <org>/<model> --tp 2 \
    --prompt "The capital of France is"
```

**Expected:** Cosine similarity > 99.99%, matching next-token predictions.

For large models that OOM `compare.py` (which loads both models), use text
generation instead:

```bash
uv run python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path <org>/<model> --tp 2 \
    --prompt "The capital of France is" --max_new_tokens 50
```

### Level 3: Training Parity (optional)

Verify that a few training steps produce decreasing loss. This catches
gradient computation issues that forward-pass tests miss. Use a toy model
with 2 layers and small dimensions. See the functional test pattern in the
`add-model-support` skill (Milestone 3, Phase 6).

## Tolerance Table

| Test Level | Dtype | Device | Max Diff | Cosine Sim |
|---|---|---|---|---|
| Round-trip | float32 | CPU | 0.0 (exact) | 1.0 (exact) |
| Forward pass | bfloat16 | GPU | < 1e-2 | > 0.9999 |
| Forward pass | float16 | GPU | < 1e-3 | > 0.99999 |

## Comparison Utilities

These functions are useful when writing custom verification scripts or
debugging failures. They are not part of the Bridge library — copy them
into your script as needed.

```python
import torch


def compare_tensors(a, b, name=""):
    """Compare two tensors and report similarity metrics."""
    max_diff = (a - b).abs().max().item()
    mean_diff = (a - b).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0,
    ).item()
    print(f"{name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, cosine_sim={cos_sim:.8f}")
    return max_diff, mean_diff, cos_sim


def compare_state_dicts(sd_a, sd_b, prefix=""):
    """Compare two state dicts key-by-key, reporting per-parameter differences."""
    keys_a, keys_b = set(sd_a.keys()), set(sd_b.keys())
    missing, extra = keys_a - keys_b, keys_b - keys_a
    if missing:
        print(f"{prefix}Missing keys: {sorted(missing)}")
    if extra:
        print(f"{prefix}Extra keys: {sorted(extra)}")
    max_diffs = {}
    for key in sorted(keys_a & keys_b):
        diff = (sd_a[key].float() - sd_b[key].float()).abs().max().item()
        if diff > 0:
            max_diffs[key] = diff
            print(f"{prefix}{key}: max_diff={diff:.6e}")
    if not max_diffs and not missing and not extra:
        print(f"{prefix}All {len(keys_a & keys_b)} parameters match exactly.")
    return missing, extra, max_diffs
```

## Debugging Workflow

When a parity test fails, follow this sequence:

1. **Run single-GPU round-trip** — if this fails, the mapping itself is
   wrong. Check the `mapping_registry()` in the bridge file.

2. **If single-GPU passes but multi-GPU fails** — the TP/PP scatter/gather
   is wrong. Compare the TP=1 result against each TP shard. See the
   `nccl-contiguous-tensors` skill for NCCL-specific issues.

3. **If round-trip passes but forward pass fails** — weights loaded
   correctly but the model architecture differs. Check `provider_bridge()`
   config mapping (normalization, activation, RoPE, etc.).

4. **Use the debugging script template** from the `add-model-support` skill
   to inspect runtime vs safetensors key naming and bridge config mapping.

For the full catalog of pitfalls (QKV interleaving, MoE fused exports, tied
embeddings, FP8 dequantization, TE LayerNorm aliases, etc.), see the
Pitfalls section of the `add-model-support` skill.

## Code Anchors

| Component | Path |
|---|---|
| Single-GPU round-trip | `examples/conversion/hf_megatron_roundtrip.py` |
| Multi-GPU round-trip | `examples/conversion/hf_megatron_roundtrip_multi_gpu.py` |
| Forward-pass comparison | `examples/conversion/compare_hf_and_megatron/compare.py` |
| Text generation | `examples/conversion/hf_to_megatron_generate_text.py` |
| VLM generation | `examples/conversion/hf_to_megatron_generate_vlm.py` |
| Checkpoint CLI | `examples/conversion/convert_checkpoints.py` |
| Toy model creator | `examples/conversion/create_hf_toy_model.py` |
| Verification utility | `src/megatron/bridge/models/conversion/utils.py` |
| Adapter verification | `examples/conversion/adapter/verify_adapter.py` |
