# Cosmos3 DEFT AOI Data Layout

## Workspace

The default layout matches `~/workspace`:

```text
workspace/
├── annotations/
│   ├── benchmark_kpi.json
│   ├── proxy_kpi.json
│   └── mining_pool.json
├── augmentation/
│   └── anomalygen/
│       ├── base_checkpoints/          # Cosmos base checkpoints cache
│       ├── checkpoints/<project>/     # HF checkpoint auto-downloads by default; optional BYO override
│       │                                # holds ag_config.yaml + fine-tuned checkpoint.
│       └── datasets/<project>/        # clean boards, cad masks, defect_spec.jsonl
├── images/
│   ├── ... AOI images ...
│   └── golden/images/... golden references ...
├── models/
│   ├── Cosmos3-Nano/                 # downloaded native Omni source
│   └── Cosmos3-Nano-VLM/             # prepared Qwen3-VL PTM
└── specs/
    ├── evaluate_spec_proxy.toml
    ├── evaluate_spec_benchmark.toml
    └── train_spec.toml
```

A per-role evaluate spec is preferred: each file is concrete, already carrying
its own `annotation_path`, `results_dir`, and `save_folder`, so no job mutates a
shared file at launch and a Proxy job cannot pick up the Benchmark annotation.
A single `evaluate_spec.toml` is still accepted and is then materialized per
stage. `init_deft_state.py` prefers the per-role files, falls back to the
shared one, and accepts `--train-spec` / `--proxy-spec` / `--benchmark-spec`.

The `augmentation/anomalygen/` tree is required only when the AnomalyGen stage
runs. Its contents and bootstrap are owned by `references/tao-generate-anomalies.md`.

The default `models/` tree is produced after approval, not required from the
user. Download the selected published Cosmos Reason 3 source checkpoint, then
run the model skill's `prepare_cosmos3_vlm_checkpoint.py` to create the
Qwen3-VL safetensors PTM consumed by Framework. The helper may reuse the
prepared output only when it verifies the same complete, provenance-bearing
target; explicit non-default source and output paths remain valid.
Pass helper `--backend cosmos-framework`; its `--runtime-image` and
`--runtime-image-digest` use the same immutable Framework image that runs Train,
Evaluate, and Inference on the prepared PTM.

The three `.json` files are the only input annotation sets. Each file contains
one non-empty JSON array of bare OK/NG ShareGPT records; JSONL is not accepted
by the Framework dataset loaders. There is no input Train annotation. Relative
image paths resolve from the workspace root. Ignore legacy `.jsonl` siblings
when both formats are present.

The workspace TOML files are concrete or staged job specs, not application
reference templates. They must exist before `init_deft_state.py` runs — it
refuses to write state without them, because state is initialized exactly once
and may never be hand-edited, so a state pointing at absent specs would leave
the run unable to proceed. Render them with this application's `render_cfw_sft.py` and
`render_cfw_evaluate.py` scripts:

- Give Proxy and Benchmark their own evaluate spec, each already pointed at its
  own annotation and bound output path. Only when a single shared
  `evaluate_spec.toml` is used does it have to be materialized per stage.
- Keep the initial `train_spec.toml` annotation path pointed at
  `mining_pool.json`. This identifies the initial training-eligible source
  pool; it is not a pre-zero-shot Train annotation.
- Always evaluate the frozen Benchmark gate first, then zero-shot Proxy only
  when that gate is unmet. After Proxy RCA selects Mining samples, write
  `train_iter_<N>.json` and point the next staged Train job at that file.

`references/cosmos_framework_sft_full.toml` and
`references/cosmos_framework_sft_smoke.toml` are the native Framework Train
templates; rendered workspace specs contain concrete absolute paths.

Explicit non-default paths are valid. Record every path as an absolute path in
the selected platform's compute frame.

## Dataset roles

| Role | Purpose | Coverage | May train? | Workflow authority |
|---|---|---|---:|---|
| Benchmark KPI | Measure the frozen product KPI across one or more defined use cases | Target domain, fixed scenarios, and task definitions comparable to the intended use cases | no | The only set that may stop the loop |
| Proxy KPI | Test whether improvements generalize beyond the Benchmark examples | Built internally in the same target domain with similar scenario coverage; its task set may differ from Benchmark | no | The only set that drives RCCA, routing, and mining |
| Mining Pool | Supply eligible candidates for new training examples | May come from the target domain or another useful domain | source only | Supplies candidates; never gates |
| AnomalyGen SDG | Manufacture defects the pool lacks, closing Proxy false accepts | Synthetic defects painted onto clean target-domain boards | source only | Supplies candidates; never gates |

## Allowed data sources

Benchmark and Proxy are evaluation-only. Proxy may be assembled from public
data whose license is non-commercial but explicitly permits benchmarking.
Such data must never enter Mining or generated Train. Keep Benchmark frozen
and evaluation-only regardless of its source.

Every Mining record must be approved for commercial training use. Eligible
sources include:

- real internal data;
- real data purchased from a vendor with training rights;
- public data whose license permits commercial training;
- synthetic data approved for commercial training.

Track provenance and usage approval outside the annotation payload. A source
being public, purchased, internal, or synthetic does not by itself establish
training rights.

AnomalyGen output is generated per iteration under the results tree, not staged
as a workspace input. It is still training data, so the commercial-training
approval above applies to it: use only synthetic data approved for commercial
training.

`train_iter_<N>.json` is an iteration artifact, not a fourth input split.
Create it only after Proxy RCA and Mining selection finish. Later iterations
may append newly selected Mining records and newly generated AnomalyGen records
to the preceding Train artifact.

## Isolation

Proxy, Benchmark, and Mining target AOI images must be pairwise disjoint.
Generated Train targets must come from Mining or the iteration's AnomalyGen
output, and remain disjoint from Proxy and Benchmark. Synthetic targets are
held to the same isolation. Golden reference images may be shared.

## Results

```text
results/run_<id>/
├── deft_state.json
├── DEFT_Loop_Report.html
├── baseline/
│   ├── evaluate_proxy/
│   ├── proxy_rcca/
│   ├── evaluate_benchmark/
│   └── benchmark_metrics/
└── iterN/
    ├── routing/
    ├── anomalygen/
    │   ├── sdg/                       # SDG_result.csv, reconstructed_image/, original_image/
    │   └── sdg_sharegpt.json          # generated bare NG records
    ├── mining/
    ├── assemble/
    ├── validate/
    ├── train/                        # native DCP plus saved Hydra config.yaml
    ├── evaluate_proxy/
    ├── proxy_rcca/
    ├── evaluate_benchmark/
    └── benchmark_metrics/
```

Every GPU subdirectory is the bound output of its stage job-record. State
records the final absolute artifact paths, not backend object names or
unresolved symlinks.
