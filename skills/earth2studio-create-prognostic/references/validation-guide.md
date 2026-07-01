# Validation Guide for Prognostic Model Wrappers

This guide covers reference comparison, sanity plots, PR text, and PR review
comments for prognostic model wrappers.

## Step 10 - Reference Comparison and Validation

Create validation scripts in the repo root and do not commit them:

1. `reference_<model>_vanilla.py` - runs the third-party/original inference path without Earth2Studio imports.
2. `reference_<model>_e2s.py` - runs the Earth2Studio wrapper on matched inputs.
3. `reference_<model>_compare.py` - computes numerical agreement between vanilla and E2S outputs.
4. `reference_<model>_sanity.py` or equivalent - generates visual sanity-check plots from realistic inputs.

The vanilla and E2S scripts must use the same initial time, variables, grid,
lead times, stochastic seeds, and deterministic settings whenever the model
supports them. If the original implementation uses a different public latitude
or tensor order, convert internally for comparison but keep Earth2Studio public
coordinates in 90 to -90 latitude order.

Run the scripts with `uv run`:

```bash
uv run python reference_<model>_vanilla.py
uv run python reference_<model>_e2s.py
uv run python reference_<model>_compare.py
uv run python reference_<model>_sanity.py
```

Do not commit validation scripts, generated `.pt` files, NetCDF files, plots,
or image outputs. Keep them local review artifacts.

## Acceptance Criteria

- Single-step max absolute difference is below the model-appropriate tolerance.
- Multi-step correlation remains above the model-appropriate tolerance, usually `0.9999` or better for deterministic comparisons.
- Outputs have no NaN/Inf values except documented masked fields.
- Visual sanity plots look physically plausible and use the Earth2Studio public coordinate convention.

When exact agreement is impossible because of stochastic sampling, interpolation,
non-deterministic kernels, or mixed precision, explain the tolerance source in
the PR comment and report reproducible seeds/settings.

## PR Body

Use `pr-body-template.md`. Keep the structured sections from the template:

- Description
- Model details
- Dependencies added
- Reference comparison
- Validation
- Checklist

Do not collapse the PR body into a one-line summary. Do not include machine
names, hostnames, absolute filesystem paths, cache paths, device inventory, or
image upload links. The PR body should say that plots are available in the PR
comment as placeholders for manual upload.

## PR Validation Comment

Use `pr-comment-template.md`. The comment should include:

- `## Reference Comparison Validation`
- Model and comparison date
- Sanitized environment label only, such as `GPU validation run`
- Results summary table
- Key findings
- Reference plot placeholders, including multi-step vanilla-vs-Earth2Studio comparison plots
- Full review-safe validation scripts inside expandable details blocks

Each script must be placed in a fenced `python` code block under its own `<details>` / `<summary>` section. Do not replace the scripts with summaries.

The comparison script should preserve the scalar numerical comparison and also include plotting code for the multi-step comparison tensors. The generated comparison plots should place Earth2Studio output on the top row, vanilla reference output on the middle row, and relative error on the bottom row across multiple forecast lead times. Use an explicit seed for stochastic or ensemble-style reference paths and state the denominator convention for relative error, such as `max(abs(vanilla), 1e-6)`.

Do not upload images from the automation and do not paste `<img ...>` links.
Use TODO placeholders so the PR author can upload plots manually in the browser.
Do not include absolute paths, hostnames, machine names, cache directories, or
machine-identifying details.

## Branch, Commit, and PR

Before opening or updating the PR, verify that new prognostic models have a
`pyproject.toml` optional dependency extra, even if empty, that the `all` extra
includes it, that install docs include model notes plus both pip and uv commands,
and that the model is listed in `docs/modules/models_px.rst` and `CHANGELOG.md`.
Stage only implementation, tests, docs, changelog, dependency metadata, and skill
updates that belong in the branch. Exclude validation scripts and outputs.

```bash
git add earth2studio/models/px/<filename>.py         earth2studio/models/px/__init__.py         test/models/px/test_<filename>.py         pyproject.toml         CHANGELOG.md         docs/modules/models_px.rst         docs/userguide/about/install.md
```

Create the PR with the body template and then post the validation comment from
`pr-comment-template.md`.

## Code Review Follow-up

After the PR is open, inspect review and bot feedback. Fix correctness,
coordinate, dependency, documentation, and test issues. For each follow-up,
rerun focused tests and update the validation comment only when results change.
