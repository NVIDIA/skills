# Synthetic, trimmed excerpt for an evaluation exercise. Not runnable.
# tensorrt_llm.layers: attention layers expressed with tensorrt_llm.functional ops.
from ..functional import gpt_attention
from ..module import Module


class Attention(Module):

    def __init__(self, hidden_size, num_attention_heads, rotary_embedding_dim=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.rotary_embedding_dim = rotary_embedding_dim

    def forward(self, hidden_states, kv_cache_params=None, attention_params=None):
        return gpt_attention(hidden_states, kv_cache_params, attention_params,
                             rotary_embedding_dim=self.rotary_embedding_dim)


class DeepseekV2Attention(Module):

    def __init__(self, hidden_size, num_attention_heads, kv_lora_rank, qk_rope_head_dim):
        super().__init__()
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

    def forward(self, hidden_states, kv_cache_params=None, attention_params=None):
        return gpt_attention(hidden_states, kv_cache_params, attention_params,
                             is_mla_enabled=True,
                             rotary_embedding_dim=self.qk_rope_head_dim)
