# Validation Checklist

Run validation at the narrowest useful scope first, then broaden before the
branch is ready.

## Code Checks

Use the repo commands when time and dependencies allow:

```bash
make lint-check
make mypy
make test
```

For focused work, run the relevant package or service tests directly with
pytest, then run the full commands before final submission.

## Asset Checks

When changing skills or cookbooks, run:

```bash
git diff --check
rg "services/workflow_runner/cookbooks|\\.agents/skills" --glob '!skills/paidf-auto-labeling/references/workflow-stage-integration/validation-checklist.md'
```

The scan should return no stale service-local cookbook paths or old skill-root
paths. For cookbook changes, also run a focused workflow-runner dry run or test
that loads the changed cookbook.

## Runner Dry-Run Checks

Before a real workflow run, run the workflow runner with `--container-dry-run`
and inspect:

- Stage order or topological order.
- Image names and build targets.
- Runtime, GPUs, network mode, and environment pass-through.
- Model cache mounts.
- Media, output, question-bank, prompt, and config mounts.
- VLM/LLM endpoint URLs and model names.
- Stage args, especially windowing and generation modes.

## Output Checks

After execution, validate each scene directory:

- `sidecars/pipeline_state.json` parses with the core model.
- Media-transforming stages preserve raw media and update active media only when
  expected.
- Detection/tracking, captioning, VQA, and export artifacts exist for the
  enabled stages.
- The `reasoning` stage writes expected DAFT `task/` (and `contextual/`)
  artifacts, and `training_export` writes aggregated dataset outputs under its
  owned namespace.
- Empty or missing outputs are explained by policy, skipped stages, or explicit
  disabled settings.

## MR Readiness

Before committing:

- Keep unrelated changes out of the branch.
- Document new stage names, cookbook fields, dry-run examples, and known gaps.
- Include tests for error cases, not only the happy path.
- State any pending service/image dependencies in the MR description.
