# Model Capabilities and Precision

Use this reference when adding a model class to `MODEL_ARCH_MAPPING`.

## ModelCapabilities

Every registered class must declare parallelism capabilities. Pick exactly one
of the two patterns below. CI enforces this via
`tests/unit_tests/_transformers/test_model_capabilities.py`.

The canonical `ModelCapabilities` dataclass has four bool fields:
`supports_tp`, `supports_cp`, `supports_pp`, and `supports_ep`. It is exported
as `nemo_automodel.ModelCapabilities`.

Declare what has been verified by a recipe YAML. A flag is `True` only when at
least one `examples/*/*.yaml` for this class sets that parallelism axis above 1.
Otherwise leave it `False`.

### Static `ModelCapabilities`

Use a frozen nested dataclass when every checkpoint mapped to the class shares
the same capability profile.

```python
from dataclasses import dataclass

class NewModelForCausalLM(HFCheckpointingMixin, nn.Module):
    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False
```

### Variant `get_capabilities(cls, config)`

Use `get_capabilities` when one registered class serves checkpoints with
different capability profiles. Do not also define a nested `ModelCapabilities`
dataclass.

```python
from nemo_automodel import ModelCapabilities


class Ernie4_5_MoeForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    @classmethod
    def get_capabilities(cls, config) -> ModelCapabilities:
        """Return parallelism capabilities for a specific ERNIE-4.5 config."""
        if getattr(config, "moe_num_experts", 0) > 0:
            return ModelCapabilities(supports_ep=True)
        return ModelCapabilities()
```

The dispatch field must be stable and present on every HF config the class
sees. Good signals include model-specific expert counts, known boolean flags
such as `enable_moe_block`, or `num_hidden_layers` when variants differ clearly.
Avoid heuristics that silently misclassify new checkpoints.

The public API dispatches between these patterns:

```python
from nemo_automodel import query_capabilities

caps = query_capabilities("baidu/ERNIE-4.5-21B-A3B-PT")
```

`query_capabilities` accepts an HF model id, a `PretrainedConfig`, a model
instance, or the registered class itself. Variant-dispatched classes reject the
bare-class form because they need a config.

## Precision-Sensitive Params

Some parameters are numerically unstable in low precision and must be computed
in fp32 even when the rest of the model computes in bf16. Examples include
SSM/Mamba `A_log` and `dt_bias`, `D` for Mamba variants whose reference
checkpoints keep it fp32, MoE sigmoid-gate bias (`e_score_correction_bias`),
attention-sink bias, and per-head `scale`.

If the model has such params, declare `_keep_in_fp32_modules_strict` as
parameter-name substrings. Sharding (`fully_shard_by_dtype`) reads this list and
uses fp32 compute dtype for matching params while other params use
`mp_policy.param_dtype`.

For a NeMo-native model class:

```python
class NewMoEForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
```

Trainable fp32 params inside mixed modules should live in a small
`_fp32_params` holder rather than as bare params beside bf16 bulk weights. Call
the holder in `forward`, keep it out of broad dtype casts with
`cast_model_to_dtype(..., skip_modules=("_fp32_params",))`, and make the
state-dict adapter strip or route holder keys plus upcast loaded tensors to
fp32.

For HF-derived models with fp32 runtime params, build the fp32 structure in the
model or layer constructor. Do not use a runtime monkeypatch, and do not infer
the contract globally from a broad module path or a parameter name alone.

Always declare the pin. Checkpoint load can auto-record original HF dtypes as a
fallback, but quantized, from-scratch, and unusual checkpoint paths may skip
that recording.

## Frozen Submodules

Frozen submodules such as VLM vision towers can create dtype mismatches under
the fp32-master pattern. A frozen part that stays fp32 can feed bf16 trainable
modules and trip matmul dtype checks. After materialization, checkpoint load,
and sharding, fully frozen submodules are cast toward `mp_policy.param_dtype`
unless they match `_keep_in_fp32_modules_strict`.

Parameters in frozen unsharded modules are cast. Parameters in frozen sharded
modules are left to FSDP all-gather casting. Buffers are always plain tensors,
so fp32 buffers are cast unconditionally unless protected by the strict list.

If a frozen part is also numerically sensitive and must compute in fp32, list it
in `_keep_in_fp32_modules_strict`. A model whose vision path forces fp32 inside
a forward op that is not a parameter or buffer needs a per-model activation cast
at that seam.
