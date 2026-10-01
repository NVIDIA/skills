# OpenFold3 Parameter Reference

## Top-Level Fields

| Field | Effect | Guidance |
|---|---|---|
| `request_id` | Optional request identifier echoed by the service. | Use when saving artifacts from multiple jobs. |
| `inputs` | Required array of prediction inputs. | OpenFold3 accepts exactly one input per request. |

## Input Fields

| Field | Effect | Guidance |
|---|---|---|
| `input_id` | Optional input identifier echoed in the response. | Set a stable value that maps to output filenames. |
| `output_format` | Structure format: `pdb` or `cif`. | Use `pdb` for broad tooling compatibility, `cif` when preserving richer metadata matters. |
| `molecules` | Biomolecular assembly to predict. | Include 1-32 molecule objects. |

## Molecule Fields

| Field | Applies To | Effect | Guidance |
|---|---|---|---|
| `type` | all | Molecule type: `protein`, `dna`, `rna`, or `ligand`. | Required. Use exact lowercase values. |
| `id` | all | Chain/entity identifier. | Use stable IDs such as `A`, `B`, or `L` to simplify interpretation. |
| `sequence` | protein, DNA, RNA | Primary sequence. | Protein uses amino-acid letters; DNA uses ATCG; RNA uses AUCG. |
| `smiles` | ligand | Ligand chemistry from SMILES. | Mutually exclusive with `ccd_codes`; validate chemistry when possible. |
| `ccd_codes` | ligand | Ligand by PDB Chemical Component Dictionary code. | Good for standard cofactors such as `ATP` or `NAG`. |
| `msa` | protein, RNA | Unpaired alignment context. | Use real alignments when available; minimal fallback is `>query\nSEQUENCE`. |
| `paired_msa` | protein | Paired alignment context for complexes. | Use when chain pairing information is available. |
| `diffusion_samples` | all molecules | Number of sampled structures, 1-5. | Increase for alternative hypotheses; save and compare all samples. |
| `structural_templates` | all | Template-guided prediction from CIF structures. | Use only when the template is relevant and chain mapping is clear. |

## MSA Shape

The nested `msa` object shape (by database and format name) is documented in
[`api.md`](api.md) under **MSA Object**. The `alignment` string must start with
a FASTA-style header. For a generated
MSA from another tool, preserve the format and only normalize it if the API
requires that shape.

## Tuning Guidance

- Start with `diffusion_samples: 1` for smoke tests and fast examples.
- Use more samples when ranking alternative complexes or ligand poses, up to 5.
- Prefer biologically meaningful sequences and real MSAs over toy examples for
  scientific runs.
- Use `output_format: "pdb"` when examples or user tooling expect PDB files;
  use `cif` when downstream tools prefer mmCIF.
- Do not mix SMILES and `ccd_codes` in one ligand object.
