# Flaky spec-decode perf-sanity case

Case: Llama-3.3-70B with Eagle3 speculative decoding (max_draft_len 3), TensorRT-LLM
PyTorch backend, 8x H100, ISL/OSL 1024/2048, concurrency 32, perf-sanity harness
(`tests/integration/defs/perf/test_perf_sanity.py`). Same commit, same nodes.

| Rep | Mean accepted tokens per step (acceptance length) | Output tok/s | Median step time |
|---|---|---|---|
| 1 | 2.31 | 2,940 | 31.4 ms |
| 2 | 1.87 | 2,410 | 31.2 ms |
| 3 | 2.64 | 3,330 | 31.6 ms |
| 4 | 1.95 | 2,500 | 31.3 ms |
| 5 | 2.48 | 3,150 | 31.5 ms |

Throughput tracks acceptance length; step time is stable across reps.

## Benchmark command emitted by the harness for this case

```
python -m tensorrt_llm.serve.scripts.benchmark_serving --model <served-model> \
  --dataset-name random --random-input-len 1024 --random-output-len 2048 \
  --ignore-eos --max-concurrency 32 --num-prompts 320
```

The same `--ignore-eos` flag is emitted for every case in the suite (it keeps
plain-decode cases comparable by forcing exactly OSL tokens per request).

In an ad-hoc run without `--ignore-eos`, requests stopped naturally after ~640 output
tokens on average.

## Proposed fix (from a teammate)

Raise the CV tolerance for spec-decode cases from 5% to 15% and increase `run_count`
from 3 to 10 so the bar stops flaking.
