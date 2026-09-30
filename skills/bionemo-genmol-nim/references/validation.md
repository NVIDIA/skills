# GenMol Validation

Validate generated molecules before treating the output as useful.

## Response Checks

- `status` should be `"success"`.
- `molecules` should be a list of dictionaries.
- Each molecule should include `smiles` and `score`.
- The returned count may be less than `num_molecules`; explain this rather than
  treating it as an API failure.

## Chemical Checks

- Use RDKit or an equivalent parser to confirm returned SMILES are valid.
- Deduplicate outputs when the user asks for unique molecules.
- Sort by `score` descending before presenting ranked output.
- For QED, scores should usually be between 0 and 1.
- For LogP, high values may indicate lipophilicity rather than better drug
  quality; call this out when ranking by LogP.

## Artifact Checks

- Save a tabular `.smi` or `.tsv` file with `smiles` and `score` columns.
- Keep raw JSON when downstream inspection matters.
- Include the SAFE input pattern in notes or metadata so the generation can be
  reproduced.

## Scientific Review

- Flag reactive groups, salts, disconnected fragments, extreme molecular size,
  and implausible analogs for medicinal chemistry review.
- Do not claim binding affinity, potency, selectivity, or safety from GenMol
  output alone.
