# cuOpt Numerical Optimization — CLI

Solve LP, MILP, and QP problems from MPS or LP files via `cuopt_cli`. The same command and options apply across all three; QP is supported in both MPS (QPS) and LP files.

Confirm problem type and formulation (variables, objective, constraints, variable types) before coding.

The CLI is included with the `cuopt` Python package — install via pip or conda, then verify with `cuopt_cli --help`.

## Basic Usage

```bash
cuopt_cli <problem-file> [options]
```

The first positional argument is the input file. The format is chosen automatically from the extension — MPS, QPS, and LP files are all accepted (including `.gz` / `.bz2` compressed variants); run `cuopt_cli --help` for the exact list of supported extensions.

## Options

**`cuopt_cli --help` is the authoritative list — don't work from a hard-coded subset.** The CLI exposes every solver setting as a flag, generated from the parameter list at runtime: take any parameter documented in the cuOpt settings reference and replace underscores with hyphens (`time_limit` → `--time-limit`, `mip_relative_gap` → `--mip-relative-gap`). So if a parameter is documented, the flag exists; `--help` and the [solver-settings docs](https://docs.nvidia.com/cuopt/user-guide/latest/) are the sources of truth for names, meanings, and defaults.

A few options are CLI-specific (not solver parameters) and worth knowing because you wouldn't derive them from a parameter name:

- `--params-file <file>` — supply many parameters from a `key = value` config file instead of repeating flags.
- `--relaxation` — solve the continuous relaxation of a MILP (drop integrality).
- `--initial-solution <file>` — warm-start from a solution file.

Run `cuopt_cli --help` for the complete, current set.

## Authoring input files

MPS, QPS, and LP are precise, externally-specified file formats. Don't hand-author them from memory or from a partial recollection of the layout — column-position rules, marker conventions, the quadratic-objective encoding, and sign/scaling conventions are easy to get subtly wrong, and a malformed file either fails to parse or **silently encodes a different model than intended**.

If you're building a model from data (rather than solving a file you already have), **define the problem through the cuOpt Python API or your preferred modeling interface — don't generate an MPS or LP file by hand to feed the CLI.** These interfaces take coefficients, bounds, and variable types as native data structures, so there's no text format to get wrong. The cuOpt Python API solves and returns the solution in the same program (see [python_api.md](python_api.md)); a separate modeling library (e.g. PuLP, Pyomo, JuMP) can export a valid file for the CLI or call its own solver. Reach for `cuopt_cli` when you *already* have a model file — e.g. a benchmark instance or a file exported by one of these tools.

When a file is genuinely the right artifact, still generate it programmatically rather than by hand:

- **From cuOpt's Python API** — build the model, then export it with `model.writeMPS("problem.mps")` (emits a valid MPS file, including the quadratic objective for QP) and solve with `cuopt_cli problem.mps`.
- **From another modeling tool** — most LP/MILP modeling libraries and solvers can export standard MPS or LP files; pass the exported file straight to `cuopt_cli`.

If you must read or write these formats by hand anyway, work from the format's full specification (and the cuOpt repo docs at `docs/cuopt/source/cuopt-cli/` for cuOpt-specific conventions such as the quadratic-objective encoding) — not from an example alone.

Either way, **validate before trusting the result**: `cuopt_cli` logs `Read file ...` on a successful parse, and reports the variable/constraint counts and objective — sanity-check those against your intended model.

## MPS Format (required sections, in order)

1. **NAME** — problem name
2. **ROWS** — `N` (objective), `L`/`G`/`E` (constraints)
3. **COLUMNS** — variable names, row names, coefficients
4. **RHS** — right-hand side values
5. **BOUNDS** (optional) — `LO`, `UP`, `FX`, `BV`, `LI`, `UI`
6. **ENDATA**

Integer variables: wrap columns with `'MARKER' 'INTORG'` before and `'MARKER' 'INTEND'` after.

## QP via CLI (beta)

Quadratic objectives are **MINIMIZE only** (for maximization, negate the objective, including the quadratic terms) and require **continuous variables only** (no integer variables mixed with a quadratic objective). Quadratic objectives use the standard MPS quadratic-objective (QPS) extension and are also supported in LP files. Check `cuopt_cli --help` for QP-specific flags; see `docs/cuopt/source/cuopt-cli/` for the format.

## Troubleshooting

- **Parsing input file failed** — Confirm the extension matches the format (`.lp` vs `.mps`/`.qps`); an unrecognized extension is rejected before parsing. The parser error names the offending line — fix it against the format spec, or (more reliably) regenerate the file from a modeling tool rather than patching it by hand. Also check `ENDATA`, section order, and integer markers.
- **Infeasible** — Re-check the model against your intended formulation: constraint directions (`L`/`G`/`E`), right-hand sides, and variable bounds.

## Reference Models

| Model | Type | Location |
|-------|------|----------|
| Minimal LP | LP | [assets/cli/lp_simple/](../assets/cli/lp_simple/) |
| Production planning | LP | [assets/cli/lp_production/](../assets/cli/lp_production/) |
| Facility location | MILP | [assets/cli/milp_facility/](../assets/cli/milp_facility/) |
