---
name: linting-and-formatting
description: Code style guidelines for NeMo-RL (Python and shell). Covers naming, indentation, comments, docstrings, reflection avoidance, and uv usage.
when_to_use: Reviewing code style; writing or checking Python or shell code; 'is this naming right', 'docstring format', 'style violation', 'how should I format this', during code review.
---

# Code Style

## Style Guides

- Python: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Shell: [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)

This repository is Python-first.

## uv

Use `uv run` to execute scripts. Do not activate a virtual environment and call `python` directly.

**Do:**
```bash
uv run examples/run_grpo.py
```

**Don't:**
```bash
source .venv/bin/activate
python examples/run_grpo.py
```

Exception: `Dockerfile.ngc_pytorch` is exempt from this rule.

## Python Standard

Code must conform to Python 3.13.13+.

## Indentation

Indent with 4 spaces. Do not use tabs.

## Naming

| Kind | Convention | Example |
|------|-----------|---------|
| Files | snake_case | `some_file.py` |
| Classes | PascalCase | `class SomeClass` |
| Functions/methods | snake_case | `def my_awesome_function():` |
| Local variables | snake_case | `my_variable = ...` |
| Variables starting with a number | prefix `k` | `k_99th_percentile = ...` |
| Global variables | upper snake_case + prefix `G` | `G_MY_GLOBAL = ...` |
| Constants | upper snake_case | `MY_CONSTANT = ...` |

- Avoid shadowing variables declared in an outer scope.
- Initialize all externally visible members of a class in the constructor.

## Comments

- For interfaces used outside a file, prefer docstrings over comments.
- Comments are for code within a function or file-local interfaces.
- Commented-out code must have a comment explaining why it is commented out. Otherwise remove it before merging.

## Docstrings

Use [Google style](https://google.github.io/styleguide/pyguide.html) docstrings (parseable by Sphinx).

## Enforce Keyword Arguments for Ambiguous Parameters

When a function has multiple parameters of the same type that could easily be swapped by mistake, use a bare `*` to force keyword-only arguments starting from where the ambiguity begins. This prevents callers from accidentally transposing arguments.

**Don't:**
```python
def loss_fn(input: Tensor, cp_group: ProcessGroup, tp_group: ProcessGroup, cp_rank: int, tp_rank: int):
    ...

# Caller can silently swap cp_group/tp_group or cp_rank/tp_rank
loss_fn(x, tp_group, cp_group, tp_rank, cp_rank)  # wrong order, no error
```

**Do:**
```python
def loss_fn(input: Tensor, *, cp_group: ProcessGroup, tp_group: ProcessGroup, cp_rank: int, tp_rank: int):
    ...

# Caller must name every argument — swaps are impossible
loss_fn(x, cp_group=cp_group, tp_group=tp_group, cp_rank=cp_rank, tp_rank=tp_rank)
```

## Avoid Reflection

Do not use reflection when functionality can be achieved without it.

**Don't:**
```python
def make_complex(*args):
    x, y = args
    return dict(**locals())
```

**Do:**
```python
def make_complex(x, y):
    return {'x': x, 'y': y}
```
