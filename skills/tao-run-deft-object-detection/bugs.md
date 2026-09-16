[P1] The skill cannot invoke the leaf skills it requires.
Frontmatter allows Read Task Bash Write, but omits Skill; later it explicitly requires invoking tao-skill-bank:* through the Skill tool. Other compositional skills declare allowed-tools: Read Skill Bash Write. If the whitelist is enforced, the workflow cannot execute its own stages. SKILL.md

[P1] Preflight violates the single user gate.
The skill says no side effects before approval, but preflight instructs the agent to:
create a venv and install packages;
download a 1.93 GB checkpoint from NGC.
Both happen before the Pre-Flight Summary. Gate contract, venv installation, checkpoint download
Resolve planned paths before approval; perform installations/downloads afterward.

[P1] Fresh class_stratified prep cannot initialize.
A fresh prep run is supposed to generate source_pool/coco.json, but initialization requires that file to exist already for class_stratified. Therefore the state machine can’t start the stage that would produce its required input. init_deft_state.py

[P1] Target classes can default to the wrong dataset.
When --target-classes and thresholds are omitted, initialization silently defaults to the reference ITS classes. It does not derive classes from the supplied KPI mapping, prepared pool report, or label map. A workflow advertised for arbitrary object-detection classes can therefore initialize as bicycle/car/person for an unrelated dataset. init_deft_state.py

[P1] Known-unmountable paths are accepted.
Initialization detects paths outside $WORKSPACE, explains that containers cannot see them, and records only a warning. The run is still initialized and fails later. The embedding snapshot is explicitly exempted even though the stage also needs to access it. init_deft_state.py
Related: the pool input builder resolves symlinks but never checks whether the resolved target remains inside the mounted workspace. A valid symlink to /data/images/x.jpg is written into the parquet and becomes invisible inside the container. prepare_input_for_image_embeddings.py

[P1] The ODVG validator accepts malformed annotations.
Malformed JSON and records missing file_name increment a counter and are skipped, but malformed > 0 is never included in failures. One valid record plus 100 malformed records returns success. validate_odvg_images.py

[P1] Caption/class ordering remains unenforced.
The Grounding DINO overlay correctly explains that caption order defines label identity, but nothing verifies that these all match:
KPI ground-truth classes;
inference captions and their order;
target classes in state;
staged ODVG labelmap.json;
source-pool class mapping.
A mismatch produces valid-looking predictions assigned to the wrong classes. The PR description itself acknowledges that KPI class coverage is not currently enforced. PR #99

[P2] Missing exclusion history silently permits duplicate mining.
On iteration 2+, if the previous cumulative parquet is missing or mistyped, prepare_exclude_for_mining.py logs a note and continues with only the current iteration. Earlier exclusions are forgotten, so later iterations may re-mine already-trained images. prepare_exclude_for_mining.py
Missing --parquet-b should only be legal when an explicit --iteration 1 is supplied.

[P2] Failed spec validation still leaves the invalid spec on disk.
apply_spec_overrides.py writes the output and report first, then checks for unresolved ??? values. With --require-no-mandatory, it returns non-zero but leaves the invalid modified file behind. A resume can mistake that for a usable stage spec. apply_spec_overrides.py

[P2] Pool compatibility proof is optional.
pool_report.json is the only artifact that cross-checks the prepared pool against the requested classes, but --pool-report is optional. Without it, a pool prepared for different classes can pass initialization and return plausible but irrelevant neighbors.

[P2] “Idempotent prep” is unsafe.
The skill says prep skips artifacts when they exist. Existence alone cannot establish that an artifact was generated from the same checkpoint, target-class mapping, encoder, image set, or container version. A partially stale pool can be silently reused. The prep report needs input fingerprints and the resume logic must compare them.

Coverage problem
The four skill evals mostly test routing and whether the agent can repeat the documented stage order. None executes:
fresh prep;
class-stratified initialization;
non-reference classes;
malformed ODVG;
caption-order mismatch;
external mounts/symlinks;
resume with missing exclusion history;
failed spec generation.

in evals.json




wordiness w/ post mortems

Biggest offenders
references/grounding-dino.md: 293 lines / 1,872 words
The sections on class_embed_bias, log_scale, checkpoint tensor inspection, error-message diagnosis, and the measured 20-frame experiment belong in tao-train-grounding-dino troubleshooting documentation.
Similarly, the story about pgrep causing a 10-hour wait belongs in platform/job-monitoring guidance.
What DEFT needs to retain is approximately:
Apply the Grounding DINO workflow overlay. Require checkpoint/spec compatibility. Train every iteration from the base checkpoint using the accumulated dataset. Require a newly created checkpoint before proceeding.


references/prep-source-pool.md: 367 lines / 3,115 words
This is the largest offender. It contains:
diagnosis of the unregistered Co-DETR console command;
an explanation of post-fold group NMS and Co-DETR internals;
a long checkpoint architecture mismatch postmortem;
tensor-shape instructions for deriving backbone, query count and class count;
detailed KITTI converter behavior and silent class-drop explanations.
Those belong respectively in:
tao-train-codetr troubleshooting;
Co-DETR inference/category-mapping documentation;
tao-convert-annotations or shared annotation-conversion troubleshooting.
The workflow reference should retain only the six prep transitions, their inputs, outputs, and validation gates.

references/preflight.md: 276 lines / 2,126 words
Preflight contains implementation tutorials rather than checks:
how to create and populate a Python environment;
exact image-version history and why another data-services image lacks commands;
NGC checkpoint provenance, including the claim that 951 tensors were compared;
a shell algorithm for searching Hugging Face cache layouts.
Those belong in setup, checkpoint-resolution, and embedding leaf skills. DEFT preflight only needs assertions such as:
host runtime ready
platform selected and ready
images resolvable
checkpoint resolved and compatible
pool valid and class-compatible
encoder identity matches pool
KPI inputs valid
configuration frozen

Smaller examples
tao-analyze-detection-kpi.md lines 66–97 teaches optional multi-threshold KPI sweeps and explains capitalization, W&B and is_internal pitfalls. That belongs in tao-analyze-detection-kpi. DEFT only uses one canonical scoring configuration.

tao-analyze-gaps-od-map.md lines 18–30 explains the internal fallback behavior of several threshold fields. The leaf gap-analysis skill should explain that. DEFT should retain the chosen invariant: every target has an explicit AP50 threshold and all fallback thresholds are zero.

tao-mine-od-images.md lines 35–56 contains a design proposal for adding rare_class_list: auto to TAO DS. That belongs in an issue/design document, not inside an executable workflow skill.

What is appropriately placed
pipeline-and-state.md, data-layout.md, and most of stage-mined-data.md are conceptually correct workflow material. They describe:
stage ordering;
state transitions;
run-wide invariants;
artifact ownership;
stop conditions;
inputs and outputs.
They may still be compressible, but they are in the right layer.
Pasteable PR feedback
The reference documentation currently mixes workflow orchestration with extensive TAO product troubleshooting. For example, grounding-dino.md contains detailed class_embed_bias/log_scale diagnosis, tensor inspection, and a pgrep postmortem; prep-source-pool.md contains Co-DETR entrypoint, architecture-loading, NMS, and annotation-converter internals; and the KPI/gap/mining overlays document leaf-command quirks and even future TAO DS design proposals.
This material is useful, but it should live in the corresponding leaf skills or dedicated troubleshooting references. The DEFT workflow should primarily define stage inputs, outputs, transitions, fixed configuration, validation gates, and stop conditions, linking to leaf documentation for command-specific diagnosis. Moving ownership downward would materially shorten this skill and prevent TAO behavior from being documented independently in both the workflow and leaf skills.



-- bugs 2

Fresh source-pool prep is ordered incorrectly. Step 1 consumes codetr_category_mapping.yaml, explicitly described as Step 2’s output, but Step 2 creates it later. A clean run cannot follow the documented order. Step 1, premature consumption, actual creation.

Neither documented training command works as written. The primary recipe omits required checkpoint and validation arguments. The other recipe includes them but points at iterN/staged/..., while staging writes iterN/tmm/.... Primary recipe, actual CLI contract, wrong paths.

The new mining-budget command is invalid shell. A missing \ after --report-json makes --pool-size execute as a separate command. Broken block.

A false pool-exhaustion assertion can mark a zero-iteration run complete. I reproduced this after baseline with max_iterations=3:
--iter-label baseline --stage loop_stop
--pool-exhausted --pool-remaining 999999
audit_deft_run.py --require-complete succeeded with 0/3 iterations. loop_stop bypasses normal transition validation, and the audit trusts the boolean without evidence. Commit handling, audit completion.

An accepted stage can be rolled back on the next resume. State and log become durable and pass audit before the recovery journal is cleared. A crash in that window leaves audit saying “run the next stage”; the following commit sees the journal and restores the pre-commit snapshot. I reproduced this with an embed commit. Commit window, unconditional rollback.

Evaluation leakage remains unprotected. The skill explicitly assumes the mining pool and KPI set are disjoint, although overlap moves evaluation images into training and makes every mAP optimistic. Additionally, training validation is sampled from the mining pool and those images are never added to mining exclusions. KPI leakage assumption, validation sampling.

KPI failures can appear successful. Docker is piped to tee without pipefail or checking PIPESTATUS, so TAO can exit nonzero while the command returns success. Separately, map_value is optional, so the audit can declare a run complete without any aggregate mAP. KPI invocation, optional mAP.

The advertised staging hard-stop is not enforced. --min-success-rate defaults to 0.0; one annotated image out of thousands succeeds. Missing annotations become “harmless” orphan images, and the entire mined batch can then be added to the exclusion set. Staging gate, orphans accepted.

Retries accept stale output as fresh success. await_stage.py accepts any existing nonempty checkpoint or any historical “finished successfully” status line. A restarted failed training job can therefore immediately reuse the old artifact. Await logic.


The documented KPI mapping example uses - car: car, the exact scalar form the bundled validator says silently produces zero metrics. The validator is never actually invoked. Example, validator warning.
JSONL artifact validation stops after the first valid record, so valid line 1 plus corrupted remainder passes. Short-circuit.
Selected platform is effectively ignored: the workflow asks for Docker/SLURM/Kubernetes/Brev, but its executable references hardcode Docker and do not use the bank’s submit/status/logs/cancel job contract.
Four named leaf-skill dependencies are absent from this checkout, leaving the Docker-only fallback as the practical execution path.
Preflight downloads a 1.93-GB checkpoint before the workflow’s own approval gate.