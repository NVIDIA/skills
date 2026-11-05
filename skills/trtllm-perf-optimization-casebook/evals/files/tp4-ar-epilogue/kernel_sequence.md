# Llama-3.1-70B FP8, TP=4 decode on 4x H100 - per-layer kernel sequence

Setup: Llama-3.1-70B-Instruct with FP8 per-tensor static quantization of linear-layer
activations and weights (activations between ops are BF16), TensorRT-LLM PyTorch
backend, TP=4 on 4x H100 SXM, decode batch 8, CUDA graphs on.
`torch_compile_config` is not set (torch.compile is off). AllReduce strategy: AUTO;
UserBuffers are not enabled.

## One decoder layer, decode step (nsys, rank 0)

```
fp8_gemm (qkv_proj)        21 us
xqa decode attention       38 us
fp8_gemm (o_proj)          14 us
allreduce (oneshot)        17 us
residual_add                3 us
rmsnorm                     5 us
static_quantize_e4m3        4 us   # quantizes the normed output for gate_up_proj
fp8_gemm (gate_up_proj)    44 us
silu_and_mul + quant        6 us
fp8_gemm (down_proj)       25 us
allreduce (oneshot)        17 us
residual_add                3 us
rmsnorm                     5 us
static_quantize_e4m3        4 us   # quantizes the normed output for the next qkv_proj
```

- Decoder layers take ~16.5 ms of an 18.2 ms decode step (LM head, sampling and the
  rest take the remainder).
- The `allreduce -> residual_add -> rmsnorm -> static_quantize_e4m3` chain appears
  twice per layer, 80 layers: 160 chains per step, ~29 us each (~4.6 ms per step).
- Each small kernel in the chain re-reads the hidden state from HBM.

Question: what does prior art say we should do about this chain, how do we turn it on,
and what are the risks?
