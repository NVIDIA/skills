# Contributing to FastGen (selected canonical sections)

> Current authored contribution policy from the task repository; this policy update is not part of the released code revision. Exact source ranges and hashes are recorded in inference_text_sources.json.

## Code Style and Formatting

FastGen uses **Ruff** for linting and code formatting. Follow existing conventions when adding new code.

```bash
make format  # Auto-format code
make lint    # Check compliance
```

**Note:** The `fastgen/third_party/` directory is excluded from checks.

## Design and Maintenance

- Write clear, modular code that is easy to use and extend. Prefer refactoring and
  shared helpers over workarounds and duplication.
- Put model-specific behavior in networks and preprocessors, and dataset-specific
  behavior in loaders. Keep methods, the trainer and shared loading infrastructure
  as model- and dataset-agnostic as possible.
- Backward compatibility is not required unless requested. Update callers together;
  avoid compatibility-only aliases or wrappers. Use keyword-only parameters only
  when they improve API clarity.
- Avoid redundant checks when existing code already reports the error clearly.
  When it improves readability, group input checks in a separate function that raises
  informative errors without modifying inputs or returning a value.
- Follow existing style, except in third-party code.
- Update relevant tests, documentation and skills when behavior, interfaces or usage
  change. Keep READMEs, docstrings and skills concise; link instead of repeating details.

## Continuous Integration

The CI pipeline runs automatically on every push to `main` and every pull request targeting `main`:

| Stage | Commands |
|-------|----------|
| Lint | `make lint`, `make mypy` |
| Test | `make pytest` |
| Install | `make install-fastgen` |

**Note:** Before submitting a pull request, ensure all checks pass locally.

The test suite also checks that the built wheel includes configs and runtime
data and works outside the checkout. Run only this check with
`python -m pytest tests/test_packaging.py`.

## Documentation

Edit component READMEs for the generated guides, or `fern/docs/*.mdx` for
authored pages. The result is published at
[docs.nvidia.com/fastgen](https://docs.nvidia.com/fastgen); see
[fern/README.md](fern/README.md) for validation, local previews, and
deployment. Documentation changes require no GPU or FastGen installation.
