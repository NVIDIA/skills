# ProteinMPNN Validation

Validate both response shape and biological plausibility before presenting
designed sequences as useful.

## Response Checks

- `mfasta` exists and is non-empty.
- FASTA headers and sequences parse cleanly.
- Designed sequence count matches `num_seq_per_target` after accounting for any
  native/WT row.
- `scores`, when present, are reported for designed sequences only.
- If the leading FASTA record is native/WT, determine whether the JSON score
  array includes it before matching scores. Reject unexplained count mismatches.
- When JSON scores are absent, preserve scores from the corresponding design
  headers and record that source. Do not substitute `global_score` or a WT score.

## Artifact Checks

- Save `mfasta` as `.fa` or `.fasta`.
- Preserve the raw HTTP body and parsed response; keep a summary of the actual
  design count, identifiers, sequences, matched scores, and artifact paths.
- Keep request metadata including input PDB name, chains, temperatures, omitted
  residues, and soluble-model flag.
- Do not overwrite outputs from multiple temperatures.
- Confirm the files were written before saying the task is complete. An HTTP
  error, a pending response, or malformed output is not a completed design.

## Scientific Checks

- Confirm designed chains and fixed chains match the user request.
- Check for excluded amino acids in designed sequences.
- Flag unusual cysteine/methionine exclusions or extreme composition choices.
- Recommend fold-back validation with OpenFold3 or Boltz2 for serious use.
