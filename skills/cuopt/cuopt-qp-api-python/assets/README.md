# Assets — reference QP models

QP reference implementations (Python, beta). Use as reference when building new applications; do not edit in place.

| Model | Description |
|-------|-------------|
| [portfolio](portfolio/) | Minimize portfolio variance; budget and min-return constraints |
| [least_squares](least_squares/) | Minimize (x-3)² + (y-4)² (closest point) |
| [maximization_workaround](maximization_workaround/) | Maximize quadratic via minimize -f(x) |

**Run:** From each subdir, `python model.py`. QP is **beta** and supports **MINIMIZE** only. See [resources/examples.md](../resources/examples.md) for more.
