# Synthetic, trimmed excerpt for an evaluation exercise. Not runnable.
import torch

from ..attention.mla import MLA


def load_kv_b_proj_and_k_b_proj_trans(module: MLA, hf_kv_b_proj: torch.Tensor, tp_rank: int,
                                      tp_size: int):
    heads = module.num_heads
    kv = hf_kv_b_proj.view(heads * tp_size, module.qk_nope_head_dim + module.v_head_dim,
                           module.kv_lora_rank)
    kv = kv[tp_rank * heads:(tp_rank + 1) * heads]
    k_nope, v = kv.split([module.qk_nope_head_dim, module.v_head_dim], dim=1)
    module.kv_b_proj.weight.data.copy_(kv.reshape(-1, module.kv_lora_rank))
    module.k_b_proj_trans.data.copy_(k_nope.transpose(1, 2).contiguous())
    module.v_b_proj.data.copy_(v.contiguous())


class DeepseekV3WeightLoader:

    def __init__(self, mapping):
        self.mapping = mapping

    def load_weights(self, model, weights):
        for name, module in model.named_modules():
            if isinstance(module, MLA):
                load_kv_b_proj_and_k_b_proj_trans(module, weights[f"{name}.kv_b_proj.weight"],
                                                  self.mapping.tp_rank, self.mapping.tp_size)
                for proj in ("q_proj", "kv_a_proj_with_mqa", "o_proj"):
                    getattr(module, proj).load_weights(weights, prefix=f"{name}.{proj}")
