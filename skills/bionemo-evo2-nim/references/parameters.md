# Evo 2 Parameter Guidance

Use small generation requests for examples and smoke tests. Request large
outputs, logits, or layer tensors only when the user needs them.

## Generation Parameters

- `sequence`: DNA prompt string. Normalize whitespace and uppercase before use.
- `num_tokens`: number of tokens to generate. Use `64` for examples unless the
  user asks otherwise.
- `temperature`: default-style value `0.7`; higher values increase randomness.
- `top_k`: `0` to `6`; `3` is a practical default, `0` considers all tokens.
- `top_p`: `0` to `1`; `0.0` disables nucleus sampling.
- `random_seed`: optional development reproducibility.
- `enable_sampled_probs`: request when the user wants probability validation.
- `enable_elapsed_ms_per_token`: request when timing matters.
- `enable_logits`: avoid unless needed because logits can make responses large.

## Forward Parameters

- Local only: `POST /biology/arc/evo2/forward`.
- Required fields are `sequence` and `output_layers`.
- Useful layer names include `output_layer`, `decoder.final_norm`, and selected
  `decoder.layers.<index>.*` entries.
- 7B layer indices are `0` to `31`; 40B layer indices are `0` to `49`.

## Local Variant Parameters

- Default local model is 40B.
- Set `NIM_VARIANT=7b` before container startup for 7B.
- Use `NIM_TEST_GPUS=0,1` for 2x H100 80GB 40B, or `NIM_TEST_GPUS=0` for
  single-H200 40B or 7B.
