---
name: tao-artifacts
description: The contract home for TAO's SDK-free execution pipeline — authoritative JSON Schemas for the four typed artifacts (spec-bundle, job-record, results_dir layout, best_rec) plus the fixed job-status vocabulary and the nested-not-dotted spec rule. Use when authoring or validating a spec-bundle before submit, writing or reading a .tao/jobs job-record, resolving where results land, or consuming AutoML's best_rec.json. Trigger phrases include "validate the spec bundle", "job record schema", "status vocabulary", "results_dir layout", "best_rec schema".
license: Apache-2.0
compatibility: Python 3.10+ with jsonschema for validation. No nvidia-tao-sdk.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- core
---

# tao-artifacts

Four typed artifacts flow through every TAO job. Their schemas live **here and
nowhere else** — producers (model/data skills) and consumers (platform skills)
both validate against this skill's `references/`.

| Artifact | Schema | Produced by → consumed by |
|---|---|---|
| **spec-bundle** | `references/spec_bundle.schema.json` | model/data skill → platform skill (at the submit seam) |
| **job-record** | `references/job_record.schema.json` | `scripts/tao_job_record.py` (the ONLY writer) → any re-attaching agent/poller |
| **results_dir layout** | `references/results_dir.contract.md` | platform skill at submit → whoever collects outputs |
| **best_rec** | `references/best_rec.schema.json` | tao-run-automl adapter → DEFT warm-start |

## Quick Start — validate an artifact

```bash
python - <<'PY'
import json, yaml, jsonschema, pathlib
ref = pathlib.Path("${TAO_SKILL_BANK_PATH:?}/skills/core/tao-artifacts/references")
schema = json.loads((ref / "spec_bundle.schema.json").read_text())
bundle = yaml.safe_load(open("/path/to/bundle.yaml"))   # or a dict built in-context
jsonschema.validate(bundle, schema)                      # raises on violation
print("bundle OK")
PY
```

Validate the bundle **before** the verify-before-launch gate; validate a
job-record only when debugging (the writer script already enforces the schema).

## The two rules the schemas enforce structurally

1. **Nested, not dotted.** A `spec` is a nested dict mirroring the container's
   config shape — `{"train": {"num_epochs": 12}}`. Any key containing `.` at
   any depth is rejected (`{"train.num_epochs": 12}` is the #1 authoring
   mistake). Note the distinction: `declared_inputs[].spec_key` and
   `gpu_spec_key` are dotted/indexed **pointers** into the spec
   (`dataset.train_data_sources[0].image_dir`) — dots are correct there.
2. **Mode discrimination.** `mode: config` requires `spec` + `config_format`
   and a `command` containing `{config_path}`, and forbids `args`.
   `mode: args` requires `args` and forbids `spec`. There is no other mode.

## Optional action lifecycle

Use `execution` when an action needs more than its primary command. This is the
shared model-to-platform seam; do not add a model-specific Docker, Kubernetes,
or SLURM renderer merely to carry runtime environment, attestations,
post-processing, or helper dependencies.

- The producing model/data skill owns `environment`, ordered `pre_commands`,
  ordered `post_commands`, distributed-launch intent, and completion evidence.
- The platform owns container mounts, scheduler/container syntax, task/rank
  binding, timeouts, log paths, and preservation of the real child exit code.
- `environment` is non-secret. Credential values continue to use the selected
  platform's secret/sidecar contract and never enter a spec-bundle.
- Commands, environment values, and string values in `spec` may use
  `{config_path}`, `{job_id}`, and `{results_dir}`. The platform binds them
  only after the job record has been opened; the job record's `results_dir` is
  authoritative over any pre-review display path. Persist hashes of both the
  producer bundle and the bound runtime config.
- `supporting_files` names checked-in orchestration helpers relative to the
  producing skill root. The platform stages the closed set, verifies every
  declared SHA256, and rejects traversal, undeclared siblings, or overwrite of
  a different bundle. Supporting files orchestrate an action; they must never
  shadow or patch code inside the selected image.
- A `torchrun` declaration expresses process topology, not SLURM/Kubernetes
  syntax. Each platform maps it to its native distributed launcher.

## Fixed status vocabulary

Every job state anywhere in the pipeline is exactly one of:

`PENDING · RUNNING · COMPLETE · ERROR · CANCELED · UNKNOWN`

Platform-native sub-states (`ImagePullBackOff`, `PENDING`-because-resources,
`Insufficient-GPU`, slurm `COMPLETING`…) are never new states — they ride in
the transition's `message` field. Terminal = `COMPLETE | ERROR | CANCELED`.
This is what lets the in-turn poll loop and the detached poller share one code
path across docker/slurm/kubernetes/brev.

## Ordering invariants (enforced at the seam, stated here)

- The verify-before-launch gate runs on the **spec-bundle**, before any job id
  exists.
- `tao_job_record.py open` writes `PENDING` + the resolved `results_dir`
  **first** and returns the id — the only handle a launch can use. A submit
  that skipped the gate has no id, so it cannot launch.
- `transitions` is append-only; `.tao/` lives outside every synced results
  tree.
