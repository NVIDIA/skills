# Synthetic, trimmed excerpt for an evaluation exercise. Not runnable.
import torch
from torch import nn

from ..modules.linear import Linear
from ..utils import get_sm_version, maybe_compiled_cat
from .backends.trtllm import create_attention
from .rotary_embedding import RotaryEmbedding


class MLA(nn.Module):
    """Multi-head latent attention for DeepSeek-V3-style models (including DSA models)."""

    def __init__(self, config, mapping, layer_idx: int, rope_fusion: bool = True,
                 is_dsa: bool = False):
        super().__init__()
        self.mapping = mapping
        self.layer_idx = layer_idx
        self.is_dsa = is_dsa
        self.num_heads = config.num_attention_heads // mapping.tp_size
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        self.q_proj = Linear(config.hidden_size,
                             self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim))
        self.kv_a_proj_with_mqa = Linear(config.hidden_size,
                                         self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_b_proj = Linear(self.kv_lora_rank,
                                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.o_proj = Linear(self.num_heads * self.v_head_dim, config.hidden_size)
        self.k_b_proj_trans = nn.Parameter(
            torch.empty(self.num_heads, self.kv_lora_rank, self.qk_nope_head_dim),
            requires_grad=False)
        self.v_b_proj = nn.Parameter(
            torch.empty(self.num_heads, self.v_head_dim, self.kv_lora_rank),
            requires_grad=False)

        self.apply_rotary_emb = not rope_fusion
        self.rotary_emb = RotaryEmbedding(config.rope_scaling, head_dim=self.qk_rope_head_dim)

        self.mqa = create_attention(config, layer_idx, kind="mqa", rope_fusion=rope_fusion,
                                    sparse=is_dsa)
        if is_dsa:
            self.mha = None
        else:
            self.mha = create_attention(config, layer_idx, kind="mha", rope_fusion=rope_fusion)

        if self.is_dsa:
            assert self.mha is None, "DSA models run context attention through self.mqa"

    def forward(self, position_ids, hidden_states, attn_metadata):
        q, compressed_kv, k_pe, latent_cache = self._project_qkv(position_ids, hidden_states)
        output = hidden_states.new_empty(hidden_states.shape[0],
                                         self.num_heads * self.v_head_dim)
        n_ctx = attn_metadata.num_ctx_tokens
        if attn_metadata.num_contexts > 0:
            ctx = slice(0, n_ctx)
            if self.is_dsa:
                self.forward_context_dsa(q[ctx], compressed_kv[ctx], k_pe[ctx], attn_metadata,
                                         latent_cache[ctx], output[ctx])
            else:
                self.forward_context(q[ctx], compressed_kv[ctx], k_pe[ctx], attn_metadata,
                                     latent_cache[ctx], output[ctx])
        if attn_metadata.num_generations > 0:
            gen = slice(n_ctx, None)
            self.forward_generation(q[gen], compressed_kv[gen], k_pe[gen], attn_metadata,
                                    latent_cache[gen], output[gen])
        return self.o_proj(output)

    def _project_qkv(self, position_ids, hidden_states):
        q = self.q_proj(hidden_states)
        compressed_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        if self.apply_rotary_emb:
            q, k_pe = self.rotary_emb(position_ids, q, k_pe)
        latent_cache = maybe_compiled_cat([compressed_kv, k_pe], dim=-1)
        return q, compressed_kv, k_pe, latent_cache

    def forward_context(self, q, compressed_kv, k_pe, attn_metadata, latent_cache, output):
        if attn_metadata.num_cached_ctx_tokens > 0:
            if attn_metadata.enable_chunked_prefill and get_sm_version() >= 100:
                return self.forward_context_with_chunked_prefill(
                    q, compressed_kv, k_pe, attn_metadata, latent_cache, output)
            return self.forward_context_with_cached_kv(
                q, compressed_kv, k_pe, attn_metadata, latent_cache, output)
        return self.forward_context_default(q, compressed_kv, k_pe, attn_metadata,
                                            latent_cache, output)

    def forward_context_default(self, q, compressed_kv, k_pe, attn_metadata, latent_cache,
                                output):
        kv = self.kv_b_proj(compressed_kv).view(-1, self.num_heads,
                                                self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe_heads = k_pe.view(-1, 1, self.qk_rope_head_dim).expand(-1, self.num_heads, -1)
        k = maybe_compiled_cat([k_nope, k_pe_heads], dim=-1)
        return self.mha.forward(q, k, v, attn_metadata, latent_cache=latent_cache, out=output)

    def forward_context_with_cached_kv(self, q, compressed_kv, k_pe, attn_metadata,
                                       latent_cache, output):
        full_compressed_kv, full_k_pe = self.mha.load_cached_latent(
            attn_metadata, compressed_kv, k_pe)
        kv = self.kv_b_proj(full_compressed_kv).view(-1, self.num_heads,
                                                     self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe_heads = full_k_pe.view(-1, 1, self.qk_rope_head_dim).expand(-1, self.num_heads, -1)
        k = maybe_compiled_cat([k_nope, k_pe_heads], dim=-1)
        return self.mha.forward(q, k, v, attn_metadata, latent_cache=latent_cache, out=output)

    def forward_context_with_chunked_prefill(self, q, compressed_kv, k_pe, attn_metadata,
                                             latent_cache, output):
        for chunk in attn_metadata.cached_kv_chunks():
            self.mha.forward_chunk(q, chunk, attn_metadata, out=output, merge=True)
        return self.forward_context_default(q, compressed_kv, k_pe, attn_metadata,
                                            latent_cache, output)

    def forward_context_dsa(self, q, compressed_kv, k_pe, attn_metadata, latent_cache, output):
        topk_indices = self.mqa.indexer(q, k_pe, attn_metadata)
        return self.forward_absorption(q, compressed_kv, k_pe, attn_metadata, latent_cache,
                                       output, topk_indices=topk_indices)

    def forward_generation(self, q, compressed_kv, k_pe, attn_metadata, latent_cache, output):
        topk_indices = self.mqa.indexer(q, k_pe, attn_metadata) if self.is_dsa else None
        return self.forward_absorption(q, compressed_kv, k_pe, attn_metadata, latent_cache,
                                       output, topk_indices=topk_indices)

    def forward_absorption(self, q, compressed_kv, k_pe, attn_metadata, latent_cache, output,
                           topk_indices=None):
        q = q.view(-1, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope_absorbed = torch.bmm(q_nope.transpose(0, 1), self.k_b_proj_trans).transpose(0, 1)
        fused_q = maybe_compiled_cat([q_nope_absorbed, q_pe], dim=-1)
        latent_out = self.mqa.forward(fused_q, None, None, attn_metadata,
                                      latent_cache=latent_cache, topk_indices=topk_indices)
        out = torch.bmm(latent_out.transpose(0, 1), self.v_b_proj.transpose(1, 2))
        output.copy_(out.transpose(0, 1).reshape(output.shape))
        return output
