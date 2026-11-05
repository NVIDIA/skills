Synthetic, heavily trimmed excerpt of a TensorRT-LLM source tree, prepared for an
evaluation exercise. Only a handful of files are included, method bodies are
simplified, and nothing here is importable or runnable. Treat it as read-only
reference material.

Layout:
  tensorrt_llm/_torch/attention/mla.py              MLA attention module
  tensorrt_llm/_torch/attention/rotary_embedding.py RoPE helpers
  tensorrt_llm/_torch/attention/backends/trtllm.py  fused attention backend
  tensorrt_llm/_torch/utils.py                      shared helpers
  tensorrt_llm/_torch/models/modeling_deepseekv3.py DeepSeek-V3 weight loading
  tensorrt_llm/layers/attention.py                  engine-building attention layers
  tests/unittest/_torch/attention/attention_test_utils.py  test helpers
