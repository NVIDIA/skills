# Round 0 drill-down run

- Model: Llama-3.1-8B-Instruct (text-only, no multimodal inputs), TensorRT-LLM PyTorch backend
- Hardware: 4x B200, TP=4; ISL 1024 / OSL 1024, concurrency 128
- Profiled window: 4,100 forward steps, ~68 generation requests per step on average
  (278,800 generation-request iterations in total), 12,300 context requests
- Launch: `EXTRA_SUFFIX=round0_drilldown bash profile.sh` with

```bash
export TLLM_LINE_PROFILER_ENABLED=True
export TLLM_LINE_PROFILER_PATH=$WORKDIR/lp_results/profile_round0_drilldown.txt
export TLLM_LINE_PROFILER_FUNCTIONS="tensorrt_llm._torch.pyexecutor.model_engine.PyTorchModelEngine._prepare_tp_inputs"
```

- CPU affinity unchanged from the baseline (`numactl --cpunodebind=0 --membind=0`)
- Benchmark result: output throughput 3,200 tok/s, mean TPOT 37.4 ms
- Notes: `LlmRequest` is a nanobind-bound C++ class; `state_value`,
  `get_last_tokens()` and `max_beam_num_tokens` go through the binding, while
  `py_request_id`, `py_seq_slot` and `py_multimodal_data` are plain Python attributes
  (`py_multimodal_data` is `None` for every request in this text-only run).
  `LlmRequestState` is a Python `enum.Enum`; >99% of generation requests are in
  `GENERATION_IN_PROGRESS`. `self.mapping.has_cp_helix()` returns the same value for
  the whole run. `self.max_beam_width` is 1 (no beam search).
