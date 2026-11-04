# MSA-Search Parameter Guidance

Use exact database names and keep local-depth settings consistent with the
container configuration.

## Standard Search

- `sequence`: one protein sequence, 1-4096 amino acids.
- `databases`: case-sensitive names such as `Uniref30_2302` and
  `colabfold_envdb_202108`; use `all` only when appropriate.
- `output_alignment_formats`: usually `["a3m"]`; include `fasta` only when the
  user asks.
- `e_value`: default `0.0001`; relax for remote homologs, tighten for precision.
- `max_msa_sequences`: default `500`; for local GPU server mode it must match
  `NIM_GLOBAL_MAX_MSA_DEPTH`.

## Paired Search

- Use `/paired/predict`.
- Use `sequences`, not `sequence`.
- Provide at least two chains.
- Parse `alignments_by_chain`; do not expect the standard `alignments` shape.
- `pairing_strategy` may be `greedy` or `complete`.

## Template Search

- Use local `/structure-templates/predict`.
- Use canonical template database names such as `pdb70_220313`.
- Set `max_structures` to the requested count.
- Include `max_msa_sequences=500` unless the local `NIM_GLOBAL_MAX_MSA_DEPTH`
  was changed.

## Startup Database Selection (local Docker)

`databases` in the request body selects among **already-downloaded** databases only. To
control what is downloaded at container startup (and thus storage and launch time), set the
`NIM_MODEL_PROFILE` environment variable to a profile hash from `list-model-profiles`:

- `databases:pdb70` — ~100 MB, quick test.
- `databases:uniref30` — ~500 GB, paired/complex MSA search (UniRef30 is the only taxonomy
  DB used for pairing).
- `databases:uniref30,pdb70,pdb` — ~700 GB, template search.
- `databases:all` — ~1.2 TB, full sensitivity (adds ColabFold envdb, PDB100).

Hashes change between NIM releases — always read them from `list-model-profiles`.
