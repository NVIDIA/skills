# Evaluation notes

## Dataset

`evals.json` contains 18 offline cases: twelve workflow cases and six routing
negatives. The workflow cases ask for mode selection, payload authoring, count
and merge reasoning, GPU capacity review, result provenance, safe submission
recovery, missing-input handling, component overrides, board adaptation, pose
defects, full-board coverage for a named part, and a hypothetical local image. The
negatives cover clothing augmentation, event video generation, video style
transfer, standalone AnomalyGen fine-tuning, controller redeployment and custom
DAG authoring. No case needs credentials, a cluster, an inference endpoint or a
GPU. The Day-1 image path is a planning assumption, not a shipped fixture or an
attachment the agent must inspect.

## Grading notes

- The skill name is `cdf-defect-image-generation`. Negative cases have
  `expected_skill: null`, and the skill should stay inactive for them.
- Grade every `expected_behavior` entry separately. Routing negatives distinguish
  target-skill non-activation, avoiding a DIG payload, and naming the correct
  destination. Keep routing criteria in `expected_behavior`, the existing dataset
  format; do not assume the evaluator supports a new `route_to` field. Standalone
  AnomalyGen training has no matching bundled CDF workflow skill: accept a
  component-skill route when available or an explicit availability limitation.
- No case expects a script run: the bundled checker needs a CDF Orchestration
  checkout and its Airflow Python environment, which an evaluation sandbox may
  not have.
- For payload-authoring cases, grade the JSON the agent writes as well as its
  explanation. The pose and texture coverage cases explicitly request both;
  they prohibit execution, not explanatory prose. Grade any explicit JSON-only
  output requirement separately if one is added in a future case.
- Count and provenance cases should keep requested defect counts, raw exports
  and validated selections distinct.
- An offline case's lack of cluster access is a scope constraint, not evidence
  that readiness or submission works. Report constraint compliance separately
  from capability assertions. Require affirmative explanations for service
  requirements and workflow-managed fields rather than passing mere omission.
- In action cases, assert the required outcome directly. For example, require
  rejection and correction of an unsupported key rather than accepting either a
  checker invocation or a vague explanation. Do not require a particular helper
  solely to make the with-skill arm score higher.

## Benchmark isolation and reporting

Before the next benchmark, configure the harness so each arm sees only its task,
permitted input fixtures, and the intended product checkout. Stage runtime skill
instructions, references and scripts only in the with-skill arm. Keep
`evals/`, benchmark reports, ground truth, grading instructions, other case
responses and transcripts outside both agents' accessible filesystem and tool
search scope. Prompt instructions alone are not an isolation boundary.

Use separate workspaces for every case and arm. Deny out-of-workspace reads and
searches, and audit tool transcripts for isolation violations. If the harness
cannot enforce these boundaries, disclose that limitation and do not describe
the run as isolated.

Rerun `dig-texture-full-coverage` in both arms under these boundaries. Exclude the
reported contaminated attempt from the clean comparison and retain it only as
labeled historical evidence. Apply revised assertions before running either
arm; do not silently regrade or overwrite historical benchmark scores. Record
the new dataset digest, model, harness, repetitions and scoring policy. Existing
`BENCHMARK.md` results describe the earlier dataset and require regeneration.

## Offline checker tests

Before each benchmark, run all 12 checker tests from a CDF Orchestration checkout,
using its Airflow Python environment. Resolve failures before benchmarking:

```bash
.venv/bin/python skills/cdf-defect-image-generation/evals/test_payload.py
```

Set `CDF_ORCHESTRATION_ROOT` to the checkout when the skill is installed
elsewhere. The tests run the checker against the DAG's payload model and
work-item compiler, with no cluster or storage access.

## Live runs

The dataset launches no GPU work. To evaluate live behavior, use a small
authorized dataset and compute budget, and compare runs with and without the
skill for readiness reporting, a single confirmed submission, monitoring, service
cleanup, and result provenance in each mode. Orchestration success does not
measure defect realism or downstream inspection accuracy.
