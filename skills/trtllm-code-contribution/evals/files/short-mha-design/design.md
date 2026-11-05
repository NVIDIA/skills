# Design draft: short-sequence dense MHA for DSA MLA context attention

Synthetic design note prepared for a review exercise.

## Motivation

For DSA models, MLA context attention always runs through the sparse absorbed (MQA)
path. For short prompts, dense MHA over the expanded K/V is faster.

## Plan

1. In `MLA.__init__`, add a separate backend for DSA models so that the existing
   DSA invariant `assert self.mha is None` keeps holding:

        if self.is_dsa:
            self._short_seq_mha = create_attention(config, layer_idx, kind="mha",
                                                   rope_fusion=rope_fusion)

2. Gate the new path with a module-level constant, on by default:

        SHORT_SEQ_MHA_THRESHOLD = 10240

3. In `forward_context_dsa`, take the dense path when the longest context is short:

        def forward_context_dsa(self, q, compressed_kv, k_pe, attn_metadata, latent_cache, output):
            if (SHORT_SEQ_MHA_THRESHOLD > 0 and not self.apply_rotary_emb
                    and attn_metadata.max_ctx_seq_len <= SHORT_SEQ_MHA_THRESHOLD):
                return self.forward_context_default(q, compressed_kv, k_pe, attn_metadata,
                                                    latent_cache, output)
            ...  # existing sparse path

4. In `_prepare_workspace`, the same condition decides whether to reserve the dense
   K/V workspace:

        if (SHORT_SEQ_MHA_THRESHOLD > 0 and not self.apply_rotary_emb
                and attn_metadata.max_ctx_seq_len <= SHORT_SEQ_MHA_THRESHOLD):
            self._reserve_dense_kv_workspace(attn_metadata.num_ctx_tokens)

## Validation so far

- Accuracy checked on B200 (SM100) with TP=8 and EP=8; matches the sparse path.
- Chunked prefill (contexts that already have cached tokens) is expected to behave the
  same way on every GPU generation, so it was not tested separately.
- Pipeline parallelism and context parallelism have not been considered yet.
