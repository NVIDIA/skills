# OpenFold2 Validation Guide

## Request Checks

- `sequence` contains only valid amino-acid symbols.
- `input_id` is stable and file-name safe.
- `selected_models` contains integers from 1 to 5.
- A3M alignments start with `>` and include the query sequence.
- mmCIF templates start with `data_` and contain `_atom_site` records.
- Local requests do not include `Authorization`.

## Response Checks

Always save:

- `openfold2_request.json`
- `openfold2_response.json`
- Every PDB/mmCIF structure artifact found in the response.

Scientific checks:

- Structure artifact is parseable as PDB or mmCIF.
- Residue count approximately matches the input sequence length.
- Backbone atoms are present for most modeled residues.
- Any confidence/ranking fields are finite and recorded.
- Predictions from multiple selected models are compared, not silently mixed.

## Toy Input Caveat

Short sequences and single-sequence MSAs are valid smoke tests. They are not
evidence that a predicted fold is useful. For meaningful validation, use a
sequence with a realistic domain boundary, a non-trivial MSA, and relevant
templates only when justified.

## Suggested Live Validation

This portable skill collection does not include live hosted/local validation harnesses. Save request and response artifacts from your own NIM calls and apply the checks above.

If local validation is unavailable, record the missing dependency: Docker,
NVIDIA Container Toolkit, supported GPU, `NGC_API_KEY`/`NVIDIA_API_KEY`, or
`LOCAL_NIM_CACHE`.
