---
name: linting-and-formatting
description: Code style and quality rules for Megatron Bridge — ruff configuration, naming conventions, type hints, mypy rules, docstrings, copyright headers, logging, and the code review checklist.
when_to_use: Writing or reviewing code for style compliance, fixing ruff or mypy errors, pre-commit hook failures, copyright header questions, naming or docstring conventions.
---

# Linting and Formatting

Single source of truth for code style in Megatron Bridge. Read this before
writing new code or reviewing PRs.

## Style Guides

- Python: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Shell: [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)

Target Python 3.10+.

## Formatting and Linting

Run before every commit:

```bash
uv run ruff check --fix .
uv run ruff format .
```

Pre-commit hooks run these automatically. If hooks auto-fix files, re-stage
and re-run until clean:

```bash
git add -u
pre-commit run
# if it auto-fixed files:
git add -u
pre-commit run
```

### Ruff Rules (from `ruff.toml`)

| Rule | ID | Description |
|---|---|---|
| Line length | — | 119 characters (formatter) |
| Quote style | — | Double quotes |
| f-string without placeholders | F541 | Error |
| Unused local variable | F841 | Auto-removed by `--fix` |
| Unused import | F401 | Auto-removed by `--fix` (ignored in `__init__.py`) |
| Ambiguous variable name | E741 | Error (e.g., `l`, `O`, `I`) |
| Undefined name | F821 | Error |
| Block comment format | E266 | Error (too many `#`) |
| Import sorting | I | isort-compatible, auto-fixed |
| Public class docstring | D101 | Warning (ignored in test files) |
| Public function docstring | D103 | Warning (ignored in test files) |

**Per-file overrides:**
- `__init__.py`: F401 and F403 ignored (re-exports expected).
- `test_*.py`, `*_test.py`, `tests/*.py`: D101 and D103 ignored.

## Naming Conventions

| Kind | Convention | Example |
|---|---|---|
| Files | snake_case | `model_bridge.py` |
| Classes | PascalCase | `MegatronModelBridge` |
| Functions/methods | snake_case | `load_weights_hf_to_megatron` |
| Local variables | snake_case | `megatron_weights` |
| Variables starting with digit | prefix `k` | `k_99th_percentile` |
| Global variables | UPPER_SNAKE + prefix `G` | `G_LOGGER` |
| Constants | UPPER_SNAKE | `DEFAULT_HIDDEN_SIZE` |

- Avoid shadowing variables from an outer scope.
- Initialize all externally visible class members in the constructor.

## Import Order

1. `__future__` imports
2. Standard library
3. Third-party (`megatron.core`, `torch`, `transformers`, etc.)
4. First-party (`megatron.bridge.*`)
5. Local folder imports

Separate groups with blank lines. ruff auto-fixes via the `I` rule.

## Type Hints

Required on all public API functions and methods.

- Use `T | None` instead of `Optional[T]`
- Use `X | Y` instead of `Union[X, Y]`
- Use built-in generics (`list`, `dict`, `tuple`) instead of `typing` equivalents

```python
def get_module_by_name(
    model: torch.nn.Module,
    name: str,
    default: torch.nn.Module | None = None,
) -> torch.nn.Module | None:
    ...
```

### Mypy

Run on changed files before submitting:

```bash
uv run mypy --strict path/to/file.py
```

Key rules:

- **No `Any` leaks** — use `object` for unknown types or a `TypeVar` for generics.
- **No untyped defs** — every function must have parameter and return annotations.
- **No implicit `Optional`** — write `x: int | None = None`, never `x: int = None`.
- **Explicit casts** — use `typing.cast()` only when inference fails; add a comment.
- **Typed dictionaries** — prefer `TypedDict` over `dict[str, Any]` for structured dicts.
- **Callable signatures** — use `Callable[[ArgType], ReturnType]` or `Protocol`.
- **Ignore sparingly** — `# type: ignore[code]` must include the error code and justification.

## Keyword-Only Arguments for Ambiguous Parameters

When a function has multiple parameters of the same type that could be swapped
by mistake, use `*` to force keyword-only arguments.

```python
# Don't
def scatter_weights(tensor: Tensor, tp_group: ProcessGroup, ep_group: ProcessGroup): ...

# Do
def scatter_weights(tensor: Tensor, *, tp_group: ProcessGroup, ep_group: ProcessGroup): ...
```

## Docstrings

Google-style for public classes and functions:

```python
def convert_weights(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    mapping: MegatronParamMapping,
) -> dict[str, torch.Tensor]:
    """Convert weights from source to target model format.

    Args:
        source_model: The source model containing weights to convert.
        target_model: The target model that will receive converted weights.
        mapping: Parameter mapping defining the conversion rules.

    Returns:
        Dictionary mapping parameter names to converted weight tensors.

    Raises:
        ValueError: If source and target models have incompatible shapes.
    """
```

## Comments

- Commented-out code must have a comment explaining why. Otherwise remove it.
- Comments explain non-obvious intent, trade-offs, or constraints — not what the code does.

## Logging

Use `logging.getLogger(__name__)` for module-level loggers. Use `print_rank_0`
/ `warn_rank_0` for user-facing messages in distributed contexts.

```python
# Don't
print(f"Loading weights for {model_name}")

# Do
logger = logging.getLogger(__name__)
logger.info("Loading weights for %s", model_name)
```

## Error Handling

Use specific exceptions. Keep try bodies minimal.

```python
try:
    state_dict = torch.load(path)
except FileNotFoundError:
    raise ValueError(f"Checkpoint not found at {path}") from None
else:
    result = convert(state_dict)
```

## Avoid Reflection

```python
# Don't
def make_config(*args):
    x, y = args
    return dict(**locals())

# Do
def make_config(x, y):
    return {"x": x, "y": y}
```

## Configuration and Dataclasses

- Use `dataclasses` or `NamedTuple` for configuration objects.
- Do not add arbitrary defaults — be explicit about required vs optional fields.

## NVIDIA Copyright Header

Add to all new Python files and shell scripts (not test files). Use the current year.

```python
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## Code Review Checklist

1. **Copyright header** present on new Python files (not test files)
2. **Type hints** on public functions and methods
3. **Docstrings** on public classes and functions (Google style)
4. **Specific exceptions** in try-except blocks
5. **No bare `print()`** — use `logger` or `print_rank_0`
6. **No hidden defaults** in config function parameters
7. **Keyword-only args** for ambiguous same-type parameters
8. **Double quotes** for strings
9. **Import order** follows the 5-group convention
10. **No commented-out code** without explanation
11. **Mypy clean** — no untyped defs, no `Any` in public APIs, no bare `# type: ignore`
