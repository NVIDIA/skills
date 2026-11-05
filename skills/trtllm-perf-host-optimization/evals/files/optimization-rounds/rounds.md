# Host-optimization loop log: `PyTorchModelEngine._prepare_tp_inputs`

Workload (identical for every round): Llama-3.1-8B-Instruct (text-only), TP=4 on
4x B200, ISL 1024 / OSL 1024, concurrency 128, 4,100 forward steps profiled,
CPU affinity `numactl --cpunodebind=0 --membind=0` for every run. Related unit tests
(`tests/unittest/_torch/pyexecutor/test_model_engine.py`) passed after every kept change.

## Rounds

| Round (EXTRA_SUFFIX) | Hotspot (file:line) | Class | Change kept | `_prepare_tp_inputs` total | Output throughput |
|---|---|---|---|---|---|
| round0_drilldown | - | - | baseline | 43.20 s | 3,200 tok/s |
| round1_skip_multimodal | model_engine.py:1586-1588 | DEAD_WORK | guard `MultimodalParams` construction with `if request.py_multimodal_data:` | 35.50 s (-17.8%) | 3,520 tok/s |
| round2_eliminate_redundant_iter | model_engine.py:1598 | REDUNDANT_ITER | build `previous_state` inside the categorization loop | 31.10 s (-12.4%) | 3,705 tok/s |
| round3_cache_invariants | model_engine.py:1566-1583 | FUNCALL | hoist `has_cp_helix()`, drop `first_beam` (option B) | 29.30 s (-5.8%) | 3,762 tok/s |
| round4_enum_raw_int | model_engine.py:1576 | ENUM_CONSTRUCT | compare raw `state_value` ints, build the enum only on the rare path | 28.90 s (-1.4%) | 3,779 tok/s |
| round5_cache_lookup | model_engine.py:1599 | CONTAINER | cache per-slot dict lookups in `_can_use_incremental_update` | 28.65 s (-0.9%) | 3,786 tok/s |

Reverted attempt: in round 3, option C (also rewriting line 1580 as
`beam_width = request.sampling_config.beam_width if _use_beam_search else 1`) made the
line slower (Per Hit 5.1 us -> 14.3 us) and the function slower (31.10 s -> 31.85 s);
it was rolled back with `git checkout -- tensorrt_llm/_torch/pyexecutor/model_engine.py`
and option B was applied instead.

## Benchmark metrics (benchmark client, same workload)

| Metric | round0 | round5 |
|---|---|---|
| Output throughput | 3,200 tok/s | 3,786 tok/s |
| Mean TPOT | 37.4 ms | 31.5 ms |
| Mean ITL | 37.1 ms | 31.3 ms |
| Mean E2EL | 39.6 s | 33.4 s |

## Top remaining lines after round 5 (`round5_cache_lookup` profile)

| Line | Content | Time | Notes |
|---|---|---|---|
| 1620 | `attn_metadata.prepare()` | 3.91 s | drilled down in round 5: host-side construction of per-step metadata tensors, no syncs or allocations in loops |
| 1577 | `input_ids.append(request.get_last_tokens(0))` | 3.01 s | per-request token fetch through the C++ request object |
| 1552 | `input_ids.extend(request.get_tokens(0))` | 2.35 s | context-request prompt tokens |
| 1578 | `position_ids.append(request.max_beam_num_tokens - 1)` | 1.62 s | per-request position |

Only remaining proposal on the table: rewrite per-request bookkeeping as a
struct-of-arrays token/position buffer (or a Cython extension) so the loop no longer
touches each request object - estimated 4-6 s, high risk, touches the request
lifecycle in py_executor.py, sampler.py and resource_manager.py.
