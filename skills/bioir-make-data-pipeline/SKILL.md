---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
name: make-data-pipeline
description: Port an open-source bioinformatics data pipeline into the BioIR pipeline architecture. Use when the user asks to create, port, convert, or write a new data pipeline from OSS code, or add a new model's data processing to the BioIR system. Level 2 is one full build_processor run of the sample manifest, serial and Ray.
license: Apache-2.0
metadata:
  author: Tien Pham
---

# Port an Open-Source Data Pipeline to BioIR

**Input:** the open-source model's name and source location. **Output:** a
registered BioIR pipeline, validation evidence, and a summary report.

The procedure lives in the BioIR checkout, written `$BIOIR` below. Without
a checkout, clone
`https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime`
and work on a branch. Read both guides in full before you start. Then follow
the port guide's steps in order, and use the validation guide for the Level 1
and Level 2 checks. Start each step only after the step before it passes.
When a guide step branches, take the branch its stated condition selects, and
record that choice in `implementation-notes.md`:

1. `$BIOIR/docs/advanced/port-data-pipeline.md`
1. `$BIOIR/docs/advanced/validate-data-pipeline.md`

This file lists only what an agent must do on top of them. Level 2 is one
full `build_processor` run of the manifest, defined below. The port guide's
`examples/folding/run_demo.py` command folds one bundled file and only shows
that the stages connect. That command does not satisfy Level 2.

## Instructions

Answer an offline audit from the recorded input. Do not run a pipeline,
clone a repository, or use the network for that request. A port follows
the steps below.

1. Gather every input before you start: the upstream source location and
   revision, the WORKDIR (default `workdir/<model_name>/`), the checkpoint,
   the samples, and the scorers. If more than one source revision is
   visible and the user has not named one, ask the user. Do not guess.
2. Follow `port-data-pipeline.md`, then `validate-data-pipeline.md`. Start
   each step only after the previous one passes.
3. Run Level 1 on the whole sample manifest. Record the shared settings in
   `NOTES.md`.
4. Run Level 2, the full pipeline on serial and Ray, only after Level 1
   passes and the settings are recorded. Copy the seed and `runtime_args`
   from `NOTES.md` into `run_pipeline.py`. Never copy example numbers from
   the guide.
5. Score every structure with the OpenStructure and DockQ commands. Apply
   the 0.05 rule and the INVESTIGATE rule from the validation guide.
6. Log each decision in `implementation-notes.md`, then write the final
   report.

## Examples

- **Audit a Level 2 run.** Given a recorded run and its manifest IDs, compare
  the processed IDs with the manifest. Missing, extra, or duplicate IDs
  mean the run does not satisfy Level 2.
- **A demo is not Level 2.** A passing `run_demo.py` on one bundled file
  shows only that the stages connect. It does not satisfy Level 2.
- **A score above upstream.** A BioIR lDDT above upstream, with matching
  serial and Ray scores, is INVESTIGATE. It stays INVESTIGATE until the
  validation guide's Check Feature Equivalence section passes, `NOTES.md`
  records identical settings for both runs, and one scoring command
  produced both scores.
- **Requests for other skills.** A module swap goes to
  `accelerate-custom-model`. A folding benchmark goes to `bench-perf-oss`.
  Neither starts a data pipeline port.
- **An ambiguous revision.** Two source revisions are visible and the user
  has not named one, and the hub does not resolve a checkpoint. Ask the
  user which revision to use, and do not start the port.

## Full pipeline run

Level 1 compares features and stops before the engine and the writer. Level 2
is the full data pipeline: parser, tokenizer, feature generator, engine, and
writer, on every id in `ref_data/sample_manifest.json`. Run that set once
through the serial processor and once through Ray. One request, or one file
under `examples/data/samples/`, is not this run.

Start Level 2 only after Level 1 passes on the whole manifest, and only after
`ref_data/upstream_config.json` and `NOTES.md` record the shared settings.
`runtime_args` and the seed come from `NOTES.md`. Copy the numbers in the
validation guide only when `NOTES.md` records those numbers.

The script below uses the folding example's `load_requests` and the CIF
writer. For a non-folding model, replace `load_requests` with the model's own
loader, and use the output format the port guide names. Record that change in
`implementation-notes.md`. If the output cannot be scored with OpenStructure
or DockQ, score it with the metric the validation guide names for that output
type, and record that substitution. Do not report an OpenStructure or DockQ
result for it.

Write `$WORKDIR/run_pipeline.py` and execute it. The validation guide's
snippet leaves `REPO`, `WORKDIR`, `load_requests`, and
`assert_manifest_coverage` to earlier sections, so the script has to define
them. The script does not parse `NOTES.md`. Copy the seed, `runtime_args`,
and any modified configuration into `SEED`, `RUNTIME_ARGS`, and
`ENGINE_KWARGS` by hand. The script refuses to run while any of those
three is `None`. If `NOTES.md` does not record one of them, stop and report
which value is missing. Do not guess a seed or a configuration. Inside the
development container, drop the `uv run --locked` prefix.

```bash
export WORKDIR="$HOME/workdir/<model_name>"
uv run --locked python "$WORKDIR/run_pipeline.py"
```

```python
import json
import os
import sys
from pathlib import Path

import ray

import bionemo_ir
from bionemo_ir.pipeline.processor.engine_proc import EngineProcessorConfig, build_processor
from bionemo_ir.pipeline.stages.configs import FeatureGeneratorStageConfig, WriterStageConfig

REPO = Path(bionemo_ir.__file__).resolve().parents[1]
WORKDIR = Path(os.environ["WORKDIR"])
sys.path.insert(0, str(REPO / "examples" / "folding"))
from run_demo import load_requests  # noqa: E402

MODEL_NAME = "<registered model name>"
SEED = None  # the seed recorded in NOTES.md
RUNTIME_ARGS = None  # the runtime_args recorded in NOTES.md
ENGINE_KWARGS = None  # {} or {"config": config} recorded for Level 1


def assert_manifest_coverage(processed_ids):
    manifest_path = WORKDIR / "ref_data" / "sample_manifest.json"
    with open(manifest_path) as handle:
        expected = {sample["sample_id"] for sample in json.load(handle)["samples"]}
    processed = list(processed_ids)
    if len(processed) != len(set(processed)):
        raise SystemExit("duplicate sample IDs")
    missing, extra = expected - set(processed), set(processed) - expected
    if missing or extra:
        raise SystemExit(f"missing={sorted(missing)} extra={sorted(extra)}")


def load_rows():
    with open(WORKDIR / "ref_data" / "sample_manifest.json") as handle:
        manifest = json.load(handle)
    rows = []
    for sample in manifest["samples"]:
        matched = [
            request
            for request in load_requests(REPO / sample["input"])
            if request["input_id"] == sample["sample_id"]
        ]
        if not matched:
            raise SystemExit(f"no request matches manifest id {sample['sample_id']}")
        rows.extend({"record": r, "__record_id": r["input_id"], "random_seed": SEED} for r in matched)
    return rows


def processor_config(executor_backend, output_path):
    return EngineProcessorConfig(
        model_source=MODEL_NAME,
        executor_backend=executor_backend,
        engine_kwargs=ENGINE_KWARGS,
        runtime_args=RUNTIME_ARGS,
        feature_generator_stage=FeatureGeneratorStageConfig(init_context={"random_seed": SEED}),
        writer_stage=WriterStageConfig(output_path=output_path, format="cif"),
    )


def run(executor_backend, output_path):
    if output_path.exists() and any(output_path.iterdir()):
        raise SystemExit(f"output directory is not empty: {output_path}")
    rows = load_rows()
    processor = build_processor(processor_config(executor_backend, str(output_path)))
    if executor_backend == "ray":
        results = list(processor(ray.data.from_items(rows)).materialize().iter_rows())
    else:
        results = processor(rows)
    failed = [row["__record_id"] for row in results if (row.get("__inference_error__") or {}).get("error_msg")]
    if failed:
        raise SystemExit(f"inference failed: {failed}")
    assert_manifest_coverage(row["__record_id"] for row in results)
    print(f"{executor_backend or 'serial'}: {len(results)} structures in {output_path}")
    del processor, results


def release_serial_cuda():
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if SEED is None or RUNTIME_ARGS is None or ENGINE_KWARGS is None:
        raise SystemExit("set SEED, RUNTIME_ARGS, and ENGINE_KWARGS before running")
    run(None, WORKDIR / "bioir" / "serial")
    # The serial processor ran in this process. Return its cached CUDA blocks
    # before Ray workers allocate.
    release_serial_cuda()
    run("ray", WORKDIR / "bioir" / "ray")
```

The two runs share the driver process. After the serial run returns, the
script drops that processor and its outputs, collects garbage, and calls
`torch.cuda.empty_cache()`. PyTorch otherwise keeps the unused CUDA blocks
in the driver cache, and Ray workers cannot allocate that memory.

If Level 1 used a modified configuration, set
`ENGINE_KWARGS = {"config": config}` in the script. Both runs read that one
object, so the serial and Ray runs cannot diverge.

A passing script exits 0, prints both lines, and leaves one structure per
manifest id under `bioir/serial` and `bioir/ray`. The script stops when
either directory already holds files, so a stale structure cannot pass
coverage. Remove that directory before a rerun. Cite the command, the exit
code, both directories, and the sample ids from the manifest. Then score
every structure with the OpenStructure and DockQ commands in the validation
guide, and apply that guide's 0.05 rule. Serial and Ray scores for a sample
must match. A BioIR score above upstream is INVESTIGATE, not PASS, until
feature equivalence, identical settings, and the same scoring command are
confirmed.

## Hard Gates

Treat the rules for trustworthy results in
`$BIOIR/docs/advanced/validate-data-pipeline.md` as hard gates. In addition:

- Every PASS you report cites the command, its exit code, the artifact path,
  and the exact sample set. Never report a result from memory or partial
  output.
- A gate you cannot meet is a blocker. Stop and report it, and do not claim
  completion.
- Changing a frozen tolerance, threshold, or feature category, or narrowing
  the sample scope, needs user approval and a rationale in `NOTES.md`.

## Ask the User

- **First:** where the upstream source lives, and the WORKDIR (default
  `workdir/<model_name>/`). Do not guess which version or fork is
  authoritative.
- **Before:** extending an existing pipeline instead of creating one,
  adding a dependency, or downloading a checkpoint that `bionemo_ir.hubs`
  does not resolve.
- **Immediately:** when you find an upstream bug.

The user might step away. Gather every resource — source, checkpoint, samples,
scorers — at the start.

## Working Directory

Set up the environment as in `$BIOIR/docs/dev.md`, and the WORKDIR as in
`$BIOIR/docs/advanced/validate-data-pipeline.md`. Never commit the WORKDIR.

Keep `$WORKDIR/implementation-notes.md` as a running log, written when a
decision happens rather than at the end. Each entry has a timestamp, a
category (design decision, deviation, tradeoff, or open question), the
context, the decision, the reason, alternatives considered, the expected
impact, and whether it needs user confirmation.

## Final Report

Consolidate `implementation-notes.md` so a future agent can resume from it,
then print a summary:

1. Model, upstream location and commit, and pipeline shape.
1. The function inventory: upstream function to BioIR class. Record each
   mapping in `implementation-notes.md` as you make it, so this report only
   consolidates it.
1. Files created or modified.
1. Level 1 results, then the full pipeline run: command, exit code, serial
   and Ray artifact directories, and the manifest ids. Include the
   OpenStructure and DockQ comparison.
1. Limitations, and anything that needs human review.
1. The WORKDIR, the path to `implementation-notes.md`, and every open
   question.
