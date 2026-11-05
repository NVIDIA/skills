# Synthetic, trimmed excerpt for an evaluation exercise. Not runnable.
import torch


def create_kv_cache_manager(config, mapping, max_batch_size, max_seq_len, is_dsa=False):
    """Create the cache manager used by MLA/DSA attention tests."""
    ...


def create_test_metadata(seq_lens, kv_cache_manager, num_cached_tokens_per_seq=None,
                         enable_chunked_prefill=False, num_generations=0):
    """Build an AttentionMetadata for unit tests.

    Handles the context/generation split, cached-token bookkeeping and KV-cache
    parameters so tests do not have to assemble AttentionMetadata by hand.
    """
    ...


def load_mla_weights_for_test(module, hf_state_dict, mapping):
    """Load HF-format test weights into an MLA module through the production loader."""
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3WeightLoader
    DeepseekV3WeightLoader(mapping).load_weights(module, hf_state_dict)
