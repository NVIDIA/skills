# DEFT OD AOI pipeline

Read this for stage order and completion gates. Read only the stage-specific
reference and leaf skill needed for the current step.

## 0. Prepare sources and freeze the contract

If the customer supplied source COCO files instead of an existing four-role
handoff, author `dataset_sources.json`, run `prepare_deft_od_aoi_sources.py
--check-only`, then materialize it in the pinned Data Services image. The
preparer writes canonical per-source COCO shards, delegates each role merge to
`annotations merge`, and validates the merged outputs before publishing the
new normalized directory. Use the emitted `sources.json` as the policy's
`sources` mapping.

Copy and complete `assets/default_policy.yaml`, then run
`init_deft_od_aoi.py` into a new result directory. Gate on
`deft_od_aoi_policy.yaml`, `input_contract.json`, and `deft_state.json`.

When a synthesis route lacks task weights, run
`resolve_deft_od_aoi_synthesis.py` and complete every emitted fine-tuning
request before building the candidate cache.

## 1. Build candidate embeddings

Run `prepare_deft_od_aoi_retrieval.py candidates`. Submit only the emitted real
and clean embedding specs through `tao-generate-image-embeddings`. An initially
empty clean role records count zero and emits neither a candidate parquet nor an
embedding spec. The completed outputs are immutable and reused by every
iteration. Commit the `candidate_cache` stage.

## 2. Baseline measurement and gaps

Run `prepare_deft_od_aoi_measurement.py --baseline` with the frozen base
checkpoint. In default `cold_start` mode, it emits empty KPI prediction files
and cardinality-matched empty test prediction evidence, but no inference specs,
so every KPI ground-truth box begins as an FN while the KPI/test completion
contract remains intact. In explicit `checkpoint` mode, submit its KPI and
report-only test inference specs using a binary-compatible checkpoint. Then
submit the emitted loose and strict gap specs via `tao-analyze-gaps-od-map`.
Commit baseline measurement and gap artifacts.

## 3. Per-iteration retrieval

Run `prepare_deft_od_aoi_retrieval.py queries` with the previous strict and
loose gap parquets and the prior cumulative COCO when one exists. Submit each
enabled query-embedding action. Then run each enabled `tao-mine-od-images`
action for max-similarity, or run `deft_od_aoi_round_robin_selection.py` once
for round-robin. Both paths must materialize each enabled role's
`mine_<role>/final_unique_files.parquet`. Round-robin additionally registers
`round_robin_selection_report.json` as the `selection_report` completion
artifact and `round_robin_admission_index.npy` as the `admission_index`
completion artifact. Commit `iteration_retrieval` with
each queried role's query parquet and exclusion parquet; the latter preserves
the exact history/configured exclusion audit even when it is empty.
Queries routed to an initially empty clean role are typed `EXHAUSTED` with all
three candidate audit counts at zero, while real retrieval continues. A real
role may become exhausted only after its nonempty pool has been consumed; clean
retrieval may then continue within its cumulative real-relative cap. When no
retrieval role is ready and synthesis cannot add data, commit convergence
without admission, training, or another unchanged iteration.

Both selection strategies accept valid empty mined artifacts. Max-similarity requires the
`filepath` column even when there are no rows; admission then records
`NO_MATCHES`. Round-robin additionally reconciles the empty selection against
its report's `NO_MATCHES` outcome and zero selected count. Missing or unreadable
artifacts and invalid completion evidence remain errors, not convergence.
Candidates below `minimum_similarity` can produce no matches without pool
exhaustion.

## 4. Admit real and clean data

Run `admit_deft_od_aoi_coco.py`. For iteration 2 and later, pass the prior
iteration's cumulative `train.json`. Gate on the new binary COCO,
`admitted_sources.parquet`, `admission_report.json`, and round-robin's copied
`admission_index.npy`. Commit `iteration_admission`. Admission records the same
`SELECTED`/`NO_MATCHES`
per-role outcome for either selection strategy. When synthesis is enabled,
the controller requires a nonnegative integer
`synthetic_admission.available_room_before_admission`. Positive capacity
routes to synthesis, including unused allowance from prior real admissions.
Zero capacity records `synthesis_decision: {status: SKIPPED, reason:
no_synthetic_budget, available_room_before_admission: 0}` in the admission
event and bypasses synthesis. Continue to training only if
`new_training_images` is positive; otherwise converge with
`retrieval_no_matches` or `retrieval_no_new_data`. Never infer zero capacity
from missing evidence or merely from zero new real images.

## 5. Optional synthesis

Only when the committed next stage is `iteration_synthesis`, run
`prepare_deft_od_aoi_synthesis.py` against strict FN gaps with the current
iteration and admitted real COCO. The default generated plan bounds work before
passing the filtering YAML through `tao-prepare-anomalygennext-inputs`; complete
its embedding and AMP actions, then invoke `tao-generate-od-defects`.

Re-run admission with the generated binary COCO and image root, the
same-iteration real admission `train.json` as `--previous-coco`, and
`--synthetic-only`. The explicit flag records mining admission as skipped;
supplying synthetic and previous COCO inputs alone does not disable mining.
Resolve the binary COCO from the producer action contract before invoking
admission: join the generation action's results directory to
`actions.generate.outputs.binary_coco.relative_path`, pass that file through
`--synthetic-coco`, and pass the results directory through `--synthetic-images`.
Do not make the admission script discover the producer skill or its metadata.
In the current producer contract, `binary_coco.relative_path` is
`pseudo_labels/coco_annotations_od_defect.json`; the similarly named
`pseudo_labels/coco_annotations.json` is `native_coco` and must not be passed
to admission. Treat this value as a sanity-check example rather than a
hardcoded override: re-read the producer contract, require a normalized
non-absolute path without `..`, and verify the resolved file exists.
Commit `iteration_synthesis`. Never synthesize from a box without its exact
mask.
If post-generation admission adds nothing, the controller also verifies the
committed initial admission report: real/clean images added earlier in this
iteration still justify training. Retained images from older iterations do not.

When combined with bounded planning, a positive image allowance can still be
too small for one whole FN. Its existing `synthesis_request.json` may report
`SKIPPED / no_synthetic_budget`. Commit it as `synthesis_request` together
with the unchanged, previously committed `admission_report` (and
`admission_index` for round-robin). The controller verifies the report hash,
remaining image budget, and zero whole-FN allocation; no `generation_report`
is required or allowed for this skip. Training still requires positive
`new_training_images` from admission. Whole-FN allocation is unchanged.

When the request has no routed FNs, commit `synthesis_request.json` as the
`synthesis_request` artifact. When mask preparation has no eligible FNs, commit
its `input_contract.json` as `synthesis_preparation`. These typed skips advance
to training without embedding, AMP, generation, or synthetic re-admission.
Preparation uses top-level reason `no_eligible_false_negatives`; detailed
`skip_counts` and per-FN records retain `no_clean_reference_images` without
disabling real/clean retrieval. If the committed retrieval manifest has no
enabled producer either, the synthesis commit records convergence instead of
starting training or an identical next iteration.

## 6. Train and select

Run `prepare_deft_od_aoi_training.py`. Iterations 1–2 emit direct
`train.yaml`. Later iterations may emit three probe specs. Submit probes,
then run `select_deft_od_aoi_training.py probes` to materialize the selected
main spec. Submit main training from the frozen base checkpoint.

Run `select_deft_od_aoi_training.py checkpoint` over all status phases. If it
emits `extension.yaml`, submit that same-iteration resume once and reselect
with `--extension-applied`. Commit `iteration_training`.

## 7. Measure, gap, and advance

Prepare measurement with the KPI-best selected checkpoint, then repeat KPI/test
inference and dual gap analysis. Commit measurement and gaps. Start the next
iteration only from the committed state.

After every application-owned stage, use `commit_deft_od_aoi_stage.py` with
the stage evidence it names: retrieval manifest and enabled-role parquets,
admission report, generation report when synthesis runs, checkpoint-selection
report, measurement manifest, and both gap reports and gap parquets. The
committer validates their identities and counts before advancing. The state is
durable history; native platform status remains authoritative while work is live.
