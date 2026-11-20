# Evaluation scope

The CI dataset has eight positive cases and two negative cases. It covers
binary plan serialization, graph ownership during parallel builds, deviceless
deployment, plan selection, cross-process bindings, cache failures, version
limits, and status propagation. The negatives distinguish package installation
and data loading from plan compilation.

These are self-contained code-review and integration-fragment tasks. They do
not require a GPU, cuDNN libraries, a source checkout, or network access.
Revision-sensitive facts needed to grade a case are supplied in its prompt.
Both evaluation arms receive the same prompt and are graded on the same
outcomes. No assertion requires reading an installed skill or a particular
instruction file.

## Run

With a configured provider and compatible agent credentials, run from the
source repo root:

```bash
skillevaluator tier3 validate skills/cudnn-compile-and-serialize --strict
skillevaluator tier3 evaluate skills/cudnn-compile-and-serialize \
  --agents codex,claude-code --env-mode docker --n-attempts 1
```

Keep the baseline enabled. Run every case and record the resolved agent/model
versions, paired results, tokens, elapsed time, failures, and skips. The config
uses one attempt and two concurrent cases. Confirm the full run fits the CI
runner's one-hour limit; if it does not, use team infrastructure and BYOE rather
than reporting an incomplete run as publication evidence.

## Extended validation

The smoke set does not establish executable C++ correctness, GPU compatibility,
or compilation speedup. On supported Linux GPU hosts, compile the converted
application and compare original/restored outputs with an independent numerical
reference. Exercise fresh UID bindings, plan filters, workspace limits, invalid
artifacts, and one/multiple workers. Capture device properties on a GPU, run the
AOT middle stage with no GPU accessible, then reload and execute on the target.

Measure cold build, parallel build, serialization, reload, and execution
separately. Record frontend/backend/CUDA/compiler versions, target device,
source commit, cache state, worker count, commands, and actual results. The
sample's skipped parallel benchmark supplies no execution evidence.

Use BYOE when required hardware or runtime exceeds CI capacity. Retain raw
trajectories and logs outside the skill directory. Do not hand-author a PASS
benchmark, skill card, or signature; review and retain the CI-generated files.
