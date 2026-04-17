---
name: code-style
description: Code style and quality guidelines for Megatron Bridge. Covers naming, type hints, ruff enforcement, keyword-arg safety, copyright headers, logging, and common anti-patterns. Auto-invoked during code review and when writing new code.
---

# Code Style for Megatron Bridge

This is the single source of truth for code style conventions in
Megatron Bridge, combining the ruff/pre-commit configuration with
project-specific rules. Read this before writing new code or reviewing PRs.

## Style Guides

- Python: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Shell: [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)

This repository is Python-first. Target Python 3.10+.

## Formatting and Linting

Run before every commit:

```bash
uv run ruff check --fix .
uv run ruff format .
```

Pre-commit hooks run these automatically. If hooks auto-fix files, re-stage
and re-run until clean.

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
- `__init__.py`: F401 and F403 are ignored (re-exports are expected).
- `test_*.py`, `*_test.py`, `tests/*.py`: D101 and D103 are ignored.

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

Organize imports in this order, separated by blank lines:

1. `__future__` imports
2. Standard library
3. Third-party (`megatron.core`, `torch`, `transformers`, etc.)
4. First-party (`megatron.bridge.*`)
5. Local folder imports

ruff auto-fixes import ordering via the `I` rule. First-party is configured
as `known-first-party = ["megatron.bridge"]`.

## Type Hints

Required on all public API functions and methods.

- Use `T | None` instead of `Optional[T]`
- Use `X | Y` instead of `Union[X, Y]`
- Use built-in generics (`list`, `dict`, `tuple`) instead of `typing` equivalents
- Use `TypeVar` for generic type parameters

```python
def get_module_by_name(
    model: torch.nn.Module,
    name: str,
    default: torch.nn.Module | None = None,
) -> torch.nn.Module | None:
    ...
```

### Mypy

Run mypy on changed files before submitting:

```bash
uv run mypy --strict path/to/file.py
```

Key rules enforced by mypy:

- **No `Any` leaks** — avoid `Any` in public signatures. Use `object` for truly
  unknown types or a `TypeVar` for generic patterns.
- **No untyped defs** — every function must have parameter and return annotations.
  Use `-> None` for procedures.
- **No implicit `Optional`** — write `x: int | None = None`, never `x: int = None`.
- **Explicit casts** — use `typing.cast()` only when the type system cannot infer
  the correct type; add a comment explaining why.
- **Typed dictionaries** — prefer `TypedDict` over `dict[str, Any]` for
  structured dictionaries with known keys.
- **Callable signatures** — use `Callable[[ArgType], ReturnType]` or
  `Protocol` instead of bare `Callable`.
- **Ignore sparingly** — `# type: ignore[code]` must include the specific error
  code and a comment justifying the suppression.

## Enforce Keyword Arguments for Ambiguous Parameters

When a function has multiple parameters of the same type that could be
swapped by mistake, use a bare `*` to force keyword-only arguments.

**Don't:**
```python
def scatter_weights(tensor: Tensor, tp_group: ProcessGroup, ep_group: ProcessGroup):
    ...
scatter_weights(t, ep_group, tp_group)  # silently wrong
```

**Do:**
```python
def scatter_weights(tensor: Tensor, *, tp_group: ProcessGroup, ep_group: ProcessGroup):
    ...
scatter_weights(t, tp_group=tp_group, ep_group=ep_group)  # clear
```

## Docstrings

Use Google-style docstrings for public classes and functions. These are
parseable by Sphinx.

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
    ...
```

For interfaces used outside a file, prefer docstrings over comments. Comments
are for code within a function or file-local interfaces.

## Comments

- Commented-out code must have a comment explaining why. Otherwise remove it.
- Do not add comments that merely narrate what the code does.
- Comments should explain non-obvious intent, trade-offs, or constraints.

## Logging

Use `logging.getLogger(__name__)` for module-level loggers. Use
`print_rank_0` / `warn_rank_0` for user-facing messages in distributed
contexts.

**Don't:**
```python
print(f"Loading weights for {model_name}")
```

**Do:**
```python
logger = logging.getLogger(__name__)
logger.info("Loading weights for %s", model_name)

# Or for distributed-aware output:
from megatron.bridge.utils.common_utils import print_rank_0
print_rank_0(f"Loading weights for {model_name}")
```

## Error Handling

Use specific exceptions. Keep try bodies minimal.

**Don't:**
```python
try:
    result = load_and_convert(path)
except:
    print("Conversion failed")
```

**Do:**
```python
try:
    state_dict = torch.load(path)
except FileNotFoundError:
    raise ValueError(f"Checkpoint not found at {path}") from None
else:
    result = convert(state_dict)
```

When using try-except for duck typing, keep the try body as small as possible
and use the else block for logic:

```python
try:
    f.seek  # probe, do not call
except AttributeError:
    ...  # not file-like
else:
    f.seek(0)
    f.read()
```

## Avoid Reflection

Do not use reflection when functionality can be achieved without it.

**Don't:**
```python
def make_config(*args):
    x, y = args
    return dict(**locals())
```

**Do:**
```python
def make_config(x, y):
    return {"x": x, "y": y}
```

## Configuration and Dataclasses

- Use `dataclasses` or `NamedTuple` for configuration objects.
- Be explicit about required vs optional fields.
- Do not add arbitrary defaults — be as explicit as possible.

## NVIDIA Copyright Header

Add this header to all Python files and shell scripts. Use the current year.
Exclude test files under `tests/`.

```python
# Copyright (c) <CURRENT_YEAR>, NVIDIA CORPORATION.  All rights reserved.
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

## String Quotes

Use double quotes for all strings (matching ruff formatter configuration).

## Testing Conventions

- Unit tests go in `tests/unit_tests/`, named `test_*.py`.
- Functional tests go in `tests/functional_tests/`.
- Use pytest fixtures for common setup.
- Use pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`.
- Keep unit test configs tiny: small hidden dims, 1-2 layers, short sequences.
- Functional tests are capped at 2 GPUs.
- Set `CUDA_VISIBLE_DEVICES` explicitly for multi-GPU tests.

## Code Review Checklist

When reviewing code, check for:

1. **Copyright header** present on new Python files (not test files)
2. **Type hints** on public functions and methods
3. **Docstrings** on public classes and functions (Google style)
4. **Specific exceptions** in try-except blocks
5. **No bare `print()`** — use `logger` or `print_rank_0`
6. **No hidden defaults** in function parameters for config values
7. **Keyword-only args** for ambiguous same-type parameters
8. **Double quotes** for strings
9. **Import order** follows the 5-group convention
10. **No commented-out code** without explanation
11. **Mypy clean** — no untyped defs, no `Any` in public APIs, no bare `# type: ignore`
