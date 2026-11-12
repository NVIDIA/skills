---
name: newton-usage
description: Set up, build, or debug physics simulations with the Newton Python package. Use when installing Newton, choosing simulation components, finding examples, or diagnosing simulation and runtime problems.
---

# Using Newton

Help users of the released Newton wheel find and apply the relevant public
documentation and examples. Work in the user's Python environment; a Newton
source checkout is optional.

1. Identify the installed Newton version with
   `python -c "from importlib.metadata import version; print(version('newton'))"`.
   Establish the user's goal or failure, solver, and CPU/GPU environment.
2. Start at the [Newton documentation](https://newton-physics.github.io/newton/).
   Select documentation matching the installed release where available; flag
   any version mismatch before relying on an API or feature. Read only the
   sections relevant to the task:
   - Setup and modeling: installation, tutorials, conventions, solver guides,
     and the API reference.
   - Debugging: FAQ, simulation tuning, collisions and contacts, and viewers
     and debugging; follow the relevant solver-specific guidance.
   - If a matching source checkout is already available, these sources are
     under `docs/guide/*.rst`, `docs/concepts/*.rst`, `docs/solvers/*.rst`,
     `docs/api/*.rst`, and `docs/faq.rst`. Do not require a clone to use Newton.
3. Discover examples shipped in the wheel with
   `python -m newton.examples --list`. Run a relevant example with
   `python -m newton.examples <example_name>` and inspect its options with
   `python -m newton.examples <example_name> --help`. Locate the installed
   package with `python -c "import newton; print(newton.__file__)"`; example
   source is in the adjacent `examples/` directory. If example dependencies
   are missing, follow the installation guide for the `newton[examples]`
   extra, keeping the user's Newton version pinned.
4. Adapt the closest supported example or reduce a failure to a small
   reproducer. Use documented public APIs, make a focused change, and verify
   the requested behavior or original symptom. Report the documentation used,
   what was checked, and anything still unverified.
