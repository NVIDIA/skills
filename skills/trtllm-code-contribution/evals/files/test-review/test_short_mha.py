# Synthetic unit-test draft prepared for a review exercise.
# Not runnable outside a TensorRT-LLM GPU environment.
import pytest
import torch

from tensorrt_llm._torch.attention.attention import MLA
from tensorrt_llm._torch.attention.backends.sparse import DSACacheManager
from tensorrt_llm._torch.attention.interface import AttentionMetadata
from tensorrt_llm.mapping import Mapping

from .configs import tiny_dsa_config


def _build_metadata(seq_lens, cache_manager):
    md = AttentionMetadata.__new__(AttentionMetadata)
    md.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    md.num_contexts = len(seq_lens)
    md.num_generations = 0
    md.num_ctx_tokens = int(sum(seq_lens))
    md.kv_cache_manager = cache_manager
    return md


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [16, 32, 64, 128, 256, 512, 1024])
def test_short_mha_matches_sparse(seq_len, batch, dtype, monkeypatch):
    cfg = tiny_dsa_config()
    mapping = Mapping(world_size=1, tp_size=1, rank=0)

    monkeypatch.setenv("TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD", "0")
    ref = MLA(cfg, mapping, layer_idx=0, is_dsa=True).to(dtype)
    monkeypatch.setenv("TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD", "4096")
    new = MLA(cfg, mapping, layer_idx=0, is_dsa=True).to(dtype)

    # HuggingFace checkpoint layout: [num_heads * (qk_nope + v), kv_lora_rank]
    hf_kv_b = torch.randn(cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
                          cfg.kv_lora_rank, dtype=dtype)
    ref.kv_b_proj.weight.data.copy_(hf_kv_b)

    # Make the two modules identical for the A/B comparison.
    with torch.no_grad():
        for (_, p_ref), (_, p_new) in zip(ref.named_parameters(), new.named_parameters()):
            p_new.copy_(p_ref)

    ref.to("cuda")
    new.to("cuda")

    cache_manager = DSACacheManager(cfg, mapping, max_batch_size=batch, max_seq_len=seq_len)
    metadata = _build_metadata([seq_len] * batch, cache_manager)

    hidden = torch.randn(batch * seq_len, cfg.hidden_size, dtype=dtype, device="cuda")
    position_ids = torch.cat([torch.arange(seq_len) for _ in range(batch)]).cuda()

    out_ref = ref(position_ids, hidden, metadata)
    out_new = new(position_ids, hidden, metadata)
    torch.testing.assert_close(out_new, out_ref, atol=2e-2, rtol=2e-2)
