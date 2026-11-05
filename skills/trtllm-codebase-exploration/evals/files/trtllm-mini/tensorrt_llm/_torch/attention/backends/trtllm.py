# Synthetic, trimmed excerpt for an evaluation exercise. Not runnable.
import torch


class TrtllmAttention:
    """Fused attention backend implemented by TensorRT-LLM C++ kernels (torch.ops.trtllm).

    Operates on packed (ragged) token batches using the per-sequence lengths in
    AttentionMetadata, and reads/writes the paged KV cache itself.
    """

    def __init__(self, layer_idx, num_heads, head_dim, num_kv_heads, rope_fusion: bool,
                 is_mla: bool = False):
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.rope_fusion = rope_fusion
        self.is_mla = is_mla

    def forward(self, q, k, v, attn_metadata, latent_cache=None, out=None, topk_indices=None):
        return torch.ops.trtllm.attention(
            q, k, v, out, latent_cache,
            attn_metadata.kv_cache_params,
            layer_idx=self.layer_idx,
            apply_rope=self.rope_fusion,
            topk_indices=topk_indices)

    def forward_chunk(self, q, chunk, attn_metadata, out, merge: bool):
        return torch.ops.trtllm.attention_chunk(q, chunk, out, merge=merge,
                                                apply_rope=self.rope_fusion)

    def load_cached_latent(self, attn_metadata, compressed_kv, k_pe):
        return torch.ops.trtllm.load_paged_latent(attn_metadata.kv_cache_params, compressed_kv,
                                                  k_pe, layer_idx=self.layer_idx)


class DSATrtllmAttention(TrtllmAttention):
    """Sparse (DSA) variant: an indexer picks the top-k KV positions for each query."""

    def __init__(self, *args, indexer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.indexer = indexer


def create_attention(config, layer_idx, kind: str, rope_fusion: bool, sparse: bool = False):
    if kind == "mqa":
        head_dim = config.kv_lora_rank + config.qk_rope_head_dim
        cls = DSATrtllmAttention if sparse else TrtllmAttention
        return cls(layer_idx, config.num_attention_heads, head_dim, 1, rope_fusion=rope_fusion,
                   is_mla=True)
    head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    return TrtllmAttention(layer_idx, config.num_attention_heads, head_dim,
                           config.num_attention_heads, rope_fusion=rope_fusion, is_mla=True)
