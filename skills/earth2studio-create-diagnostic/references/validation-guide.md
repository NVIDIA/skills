# Validation Guide for Diagnostic Model Wrappers

This guide covers reference comparison, sanity plots, PR text, and review
follow-up for diagnostic model wrappers.

## Step 10 - Reference Comparison and Validation

Create validation scripts in the repo root and do not commit them:

1. `reference_<model>_vanilla.py` - runs the original inference path without Earth2Studio imports.
2. `reference_<model>_e2s.py` - runs the Earth2Studio wrapper on matched inputs.
3. `reference_<model>_compare.py` - computes numerical agreement between vanilla and E2S outputs.
4. `reference_<model>_sanity.py` - generates visual sanity-check plots from realistic inputs.

The vanilla and Earth2Studio scripts must use the same input variables, grid,
normalization, stochastic seeds, sample count, and deterministic settings. If the
original implementation uses a different tensor order or latitude convention,
convert internally for comparison but keep Earth2Studio public coordinates in
north-to-south latitude order.

Run scripts with `uv run`:

```bash
uv run python reference_<model>_vanilla.py
uv run python reference_<model>_e2s.py
uv run python reference_<model>_compare.py
uv run python reference_<model>_sanity.py
```

Do not commit validation scripts, generated `.pt` files, NetCDF files, plots,
images, downloaded checkpoints, or cache outputs. Keep them as local review
artifacts.

## Acceptance Criteria

- Output shape matches `output_coords` exactly.
- Coordinate order is diagnostic-only: `batch`, optional `sample`, `variable`, `lat`, `lon`.
- Deterministic single-step max absolute and mean absolute differences are below model-appropriate tolerance.
- Correlation or structural agreement is high for continuous fields, usually `0.9999` or better when exact agreement is expected.
- Outputs have no NaN or Inf values except documented masked fields.
- Visual sanity plots look physically plausible and use Earth2Studio public coordinate conventions.

For stochastic or generative diagnostics, compare fixed seeded samples when
possible. When exact agreement is impossible because of stochastic sampling,
interpolation, non-deterministic kernels, or mixed precision, explain the
tolerance source and report reproducible seeds, sample count, and settings.

## PR Body

Use `pr-body-template.md`. Keep the structured sections:

- Description
- Diagnostic details
- Dependencies added
- Reference comparison
- Validation
- Checklist

Do not collapse the PR body into a one-line summary. Do not include machine
names, hostnames, absolute filesystem paths, cache paths, device inventory, or
image upload links. Say that plots are available in the validation comment as
placeholders for manual upload.

## PR Validation Comment

Use `pr-comment-template.md`. The comment should include:

- `## Reference Comparison Validation`
- Model and comparison date
- Sanitized environment label, such as `GPU validation run` or `CPU validation run`
- Results summary table
- Key findings
- Reference plot placeholders
- Full review-safe validation scripts inside expandable details blocks

Each script must be placed in a fenced `python` code block under its own
`<details>` / `<summary>` section. Do not replace scripts with summaries.

Do not upload images from automation and do not paste image links. Use TODO
placeholders so the PR author can upload plots manually in the browser.

## Branch, Commit, and PR

Before opening or updating the PR, verify that packaged diagnostics have a
`pyproject.toml` optional dependency extra, that the `all` extra includes it,
that install docs include both pip and uv commands, and that the model is listed
in `docs/modules/models_dx.rst` and `CHANGELOG.md` when it is public.

Stage only implementation, tests, docs, changelog, dependency metadata, and skill
updates that belong in the branch. Exclude validation scripts and outputs.

```bash
git add \
  earth2studio/models/dx/<filename>.py \
  earth2studio/models/dx/__init__.py \
  test/models/dx/test_<filename>.py \
  pyproject.toml \
  CHANGELOG.md \
  docs/modules/models_dx.rst \
  docs/userguide/about/install.md
```

Create the PR with the body template and then post the validation comment from
`pr-comment-template.md`.

## Code Review Follow-up

After the PR is open, inspect review and bot feedback. Fix correctness,
coordinate, dependency, documentation, and test issues. Rerun focused tests after
each change. Update the validation comment only when results or scripts change.
