# Synthetic, trimmed excerpt for an evaluation exercise. Not runnable.
import torch
from torch import nn


def apply_rotary_pos_emb(x, cos, sin, is_neox: bool = True):
    if is_neox:
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * cos + rotated * sin


class RotaryEmbedding(nn.Module):

    def __init__(self, rope_params, head_dim: int, is_neox: bool = True):
        super().__init__()
        self.head_dim = head_dim
        self.is_neox = is_neox
        self.register_buffer("cos_sin_cache", self._build_cache(rope_params), persistent=False)

    def _build_cache(self, rope_params):
        return torch.empty(0)

    def forward(self, position_ids, q, k):
        cos, sin = self.cos_sin_cache[position_ids].chunk(2, dim=-1)
        q_rope = apply_rotary_pos_emb(q[..., -self.head_dim:], cos, sin, self.is_neox)
        k_rope = apply_rotary_pos_emb(k, cos, sin, self.is_neox)
        q = torch.cat([q[..., :-self.head_dim], q_rope], dim=-1)
        return q, k_rope
