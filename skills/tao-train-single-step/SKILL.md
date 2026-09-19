---
name: tao-train-single-step
description: Standard single-step train/eval/export workflow for any TAO model. Use when training a TAO model on a dataset
  without iterative data augmentation, AutoML, or DEFT loops. Trigger phrases include "single train run", "train then evaluate
  then export", "plain TAO training", "normal training", "no AutoML", "skip the loop". Routes through the per-model SKILL.md
  for action specifics and through `tao-launch-workflow` for platform/credentials/dataset intake.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit. Workflows declare additional requirements.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash Write
tags:
- training
- single-step
- generic
---

# Normal Train

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

Standard supervised fine-tuning: train a model on a labeled dataset, optionally evaluate, then optionally export. The most common TAO workflow for adapting a pretrained model to a new dataset.

## Steps

1. **train** — executed through AutoML when the selected model has
   `automl_enabled: true` and `automl_policy` is `on`; set
   `automl_policy=off` for a plain single training run
2. **eval** — executed if `eval_dataset_uri` is resolved
3. **export** — optional, on user request after training

## Prerequisites

The selected model skill's resolved `container_image` is the default training
runtime. Do not replace it with a host venv, `uv` environment, generic training
image, or hand-written trainer unless the user explicitly requests that
execution mode. SDK/controller Python environments are control-plane-only; the
model action remains container-backed.

### Required
- **model**: A compatible TAO model (e.g., clip, nvdinov2, grounding_dino)
- **train_dataset_uri**: URI of the training dataset (e.g., `s3://bucket/train/`)
- **platform**: Discover the execution platforms from the installed platform skills (tao-run-on-docker / -slurm / -kubernetes / -brev, plus any external one); on a runtime that surfaces only the core router skills, read `skills/platform/tao-run-on-*/SKILL.md` frontmatter.
- **container image confirmation**: resolve the default image from the selected
  model/action config, show it to the user, and require confirmation or
  `image=<override>` before creating runner files or submitting training.

### Optional
- **eval_dataset_uri**: Some model skills mark this as required — check the resolved model skill before treating it as optional.
- **base_checkpoint**: If not provided, defaults to the NGC pretrained checkpoint listed in the model skill, or trains from scratch if no NGC checkpoint exists.
- **automl_policy**: `on` by default; set `off` to bypass model-level AutoML for this run while leaving model metadata unchanged. Use only `on` / `off` in new launch settings.
- **image override**: Use `image=<override>` to pin a specific TAO toolkit build
  after reviewing the resolved default.

## Launch Intake

After the user confirms they want this standard train/eval/export workflow,
ask which supported platform they intend to run on. Discover the execution
platforms from the installed platform skills (tao-run-on-docker / -slurm /
-kubernetes / -brev, plus any external one); on a runtime that surfaces only the
core router skills, read `skills/platform/tao-run-on-*/SKILL.md` frontmatter.

Before creating a plain train runner, inspect the selected model's metadata
with `scripts/list_tao_models.py --scope automl --format json` or read
`skills/models/<network>/references/skill_info.yaml`. If `automl_enabled` is true and
the helper reports a valid train schema for that model, route the train stage
through `skills/applications/tao-run-automl` by default. Only stay on the plain train path
when `automl_policy=off`, the user explicitly asks for no HPO/AutoML, or AutoML
is enabled but not runnable because the model's train schema is not packaged
yet.

Also ask whether long-running monitoring should stay enabled and how many
minutes between status updates. Defaults: enabled, 5 minutes.

After the model/action are known, run `scripts/resolve_tao_image.py --model
<network> --action train --format text` and ask whether to use the resolved
image or an `image=<override>`. Do not create the tao-train-single-step runner until the
image is confirmed.

After platform selection, read the chosen platform skill's `## Credentials`
section and `references/skill_info.yaml` (required_credentials /
credential_groups) and ask only for credentials relevant to that platform, plus
any selected-model credentials. Do not ask for unrelated platform credentials.
