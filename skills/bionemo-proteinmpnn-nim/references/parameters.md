# ProteinMPNN Parameter Guidance

Use modest defaults for examples and make design constraints explicit.

## Core Fields

- `input_pdb`: raw PDB content as an inline string.
- `input_pdb_chains`: chains to design. If omitted, all chains may be designed.
- `num_seq_per_target`: 1-100; use the user-requested number.
- `sampling_temp`: list of floats, even for one value. Lower values are more
  conservative; multiple values explore diversity.
- `omit_AAs`: one-letter amino-acid codes to exclude globally.
- `use_soluble_model`: set `True` only when the user asks for soluble bias.
- `ca_only`: use only for CA-only backbones.

## Constraint Fields

- `fixed_positions_jsonl`: keep specific residues fixed.
- `omit_AA_jsonl`: chain- or position-specific amino-acid exclusions.
- `bias_AA_jsonl` and `bias_by_res_jsonl`: composition or per-residue biasing.
- `tied_positions_jsonl`: enforce identical residues across symmetric positions.

## Score Handling

The `mfasta` response can include a native/WT row. If scores are returned,
report them only for designed sequences and avoid pairing a score with the WT
header unless the response explicitly does so.
