# Evaluation scope

The CI dataset has eight positive cases and two negative cases. It covers
framework-managed backends, CUDA flavor selection, release metadata, extras
versus dependency groups, C++ development installs, NGC packages, loader
conflicts, CPU-only verification, and scoped upgrades. The negatives distinguish
an execution-plan task and a built-in Python task from installation.

These are self-contained installation-plan and diagnostic tasks. Prompts forbid
installation or other environment changes. No GPU, cuDNN package, checkout, or
network access is required; case-specific environment facts are in the prompt.
Both arms receive identical inputs and are graded on observable task outcomes.
Package versions in a case are evidence for that scenario, not release claims.

## Run

With a configured provider and compatible agent credentials, run from the
source repo root:

```bash
skillevaluator tier3 validate skills/cudnn-install --strict
skillevaluator tier3 evaluate skills/cudnn-install \
  --agents codex,claude-code --env-mode docker --n-attempts 1
```

Keep the baseline enabled. Run every case and record resolved agent/model
versions, paired results, tokens, elapsed time, failures, and skips. The config
uses one attempt and two concurrent cases. Confirm the full run fits the CI
runner's one-hour limit; use team infrastructure and BYOE if it does not.

## Extended validation

The smoke set does not establish that an installation works. In isolated
environments, test supported CUDA/backend/frontend combinations with Python
wheels and source builds, a C++-only development install, and a compatible NGC
image. Include framework-managed backends, conflicting loader paths, and the
requested upgrade scope. Preserve production environments and dependency pins.

Record OS/architecture, driver/toolkit, exact frontend/backend/framework
versions, loaded library paths, commands, dependency checks, and a real GPU
operation. Validate without a compatibility override. Keep unavailable platform
or GPU coverage explicit. Do not claim a Windows path was tested on Linux.

Use BYOE when the needed hardware or runtime exceeds CI capacity. Keep raw logs
and trajectories outside the skill directory. Do not hand-author a PASS
benchmark, skill card, or signature; review and retain the CI-generated files.
