# GenMol Parameter Guidance

GenMol uses one `/generate` endpoint. The `smiles` field name is misleading:
it expects SAFE notation, not ordinary SMILES, for conditioned generation.

## SAFE Patterns

- De novo: `[*{20-30}]`
- Scaffold decoration: `<scaffold_safe>.[*{10-15}]`
- Motif extension: `[*{5-10}].<core_safe>.[*{5-10}]`
- Lead optimization: encode the hit molecule, then replace one fragment with
  `[*{5-12}]`

Wider mask ranges increase diversity. Tight mask ranges keep analog size more
controlled.

## Request Parameters

Field types, defaults, and ranges live in the canonical schema table in
[`api.md`](api.md) under **Request body schema**. Usage tips: request more than
the desired display count since invalid molecules are filtered out after
generation; keep `step_size` at `1` unless tuning speed; set `unique` to `true`
for non-duplicate analogs.

## SAFE Conversion

Convert SMILES to SAFE with the `safe-mol` package. The reusable
`scaffold_to_safe` helper (encode with a fragmentation-error fallback) lives in
[`examples.md`](examples.md) under **Scaffold Decoration**; `safe-mol` is not
needed for pure de novo generation.
