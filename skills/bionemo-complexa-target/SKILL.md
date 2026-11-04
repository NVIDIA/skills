---
name: complexa-target
description: >
  Add, edit, inspect, or validate Proteina-Complexa target definitions for protein
  binders, ligand binders, and AME motif scaffolding. Use for target registries,
  chain/residue selectors, hotspots, ligand fields, and incomplete motif inputs.
  Covers direct YAML edits and complexa target/validate target commands; running
  a campaign with an already registered target belongs to complexa-design.
allowed-tools: Bash, Read, Write, AskUserQuestion
metadata:
  author: Ohad Mosafi <omosafi@nvidia.com>
---


# complexa-target

Add or edit a design target in Proteina-Complexa. Targets live in **three YAML files**, one per `complexa design` pipeline:

- `configs/targets/targets_dict.yaml` — protein binder (**default pipeline**)
- `configs/targets/ligand_targets_dict.yaml` — ligand binder
- `configs/design_tasks/ame_dict_v2.yaml` — AME / enzyme scaffolding

The `complexa target` CLI only manages the first two. AME tasks use an extended schema (the same core fields as ligand targets plus `contig_atoms` for hand-curated per-residue motif atom selections) and are file-edit-only.

## Instructions

`complexa target add` is a thin wrapper around "load YAML → build dict → append a block" (see `src/proteinfoundation/cli/target_manager.py:add_target_cli` and `append_target_to_dict`). For agentic use, **just edit the targets dict directly** — the schema is short, the existing entries are great copy templates, and you skip 14 CLI flags / SMILES shell-escaping. Use the CLI only if you explicitly want its YAML auto-quoter or its overwrite-prompt safety.

For an offline edit, work on the supplied registry or requested output copy.
When a copy is requested, create its parent directory and copy into an absent
destination. If the destination already exists, inspect and preserve it: reuse
it when it is the intended working copy, otherwise report the path conflict.
Do not delete the output tree to restart a configuration edit.
An installed Complexa runtime, GPU, or target structure is not required to write
the configuration. Use the field schema below, preserve unrelated entries, and
read back the saved block. Coordinate validation remains pending when the
structure is absent; finish the configuration task without installing a runtime.
With Read/Edit tools, read the destination copy before editing it; reading the
original input path does not satisfy Edit's requirement for the output path.

Use the schema below and the selected registry; the CLI reference is needed
only for CLI use. Finish with the saved entry, its consumer pipeline, the
downstream `generation.task_name`, and the validation status.

| Step | Direct file edit (preferred) | CLI |
|---|---|---|
| Look at existing targets | Read `configs/targets/{,ligand_}targets_dict.yaml` (or `configs/design_tasks/ame_dict_v2.yaml` for AME) | `complexa target list` (protein/ligand only) |
| Look at one target | Read the dict and grep for the key | `complexa target show NAME` |
| Add a new target | Append a YAML block (Step 3a) | `complexa target add ...` (Step 3b) |
| Verify configuration | Parse the saved YAML and compare the changed entry and untouched keys | `complexa target show NAME --dict PATH` |
| Check runtime PDB-path resolution | Requires the full pipeline and environment | `complexa validate target CONFIG --target NAME` |

## Step 1: Decide the target type

Targets live in **three different files**, one per `complexa design` pipeline. Pick the file by working out which pipeline the user will run, then add the entry to that file. Each file is consumed by exactly one pipeline.

| User intent | Pipeline | Targets dict file | This skill? |
|---|---|---|---|
| Bind a protein surface (PD-L1, IFNAR2, TNF-α, …) — **default** | `configs/search_binder_local_pipeline.yaml` | `configs/targets/targets_dict.yaml` | Yes — Step 2 (protein) |
| Bind a small-molecule pocket (FAD, SAM, OQO, …) | `configs/search_ligand_binder_local_pipeline.yaml` | `configs/targets/ligand_targets_dict.yaml` | Yes — Step 2 (ligand) |
| AME / enzyme scaffolding (motif + ligand, `M####_<pdb>` names) | `configs/search_ame_local_pipeline.yaml` | `configs/design_tasks/ame_dict_v2.yaml` | Partial — see "AME tasks" below |

### AME tasks (enzyme pipeline)

AME tasks live under `motif_target_dict_cfg:` with `source`, `target_filename`,
`ligand`, `contig_atoms`, `binder_length`, and `use_bonds_from_file`. The
`contig_atoms` string contains curated chain/residue/atom selections. Existing
entries illustrate the field structure, not the selections for a new motif.
If motif atoms, ligand selection, or structure location are missing, list the
missing inputs and write a separate, explicitly incomplete template if useful.
Leave the registry unchanged until those scientific inputs are supplied. A task
name or PDB ID alone does not determine them. In the missing-input handoff,
explain that registration will be a direct edit under `motif_target_dict_cfg`
once the curated selections are supplied; `complexa target add` cannot register
AME tasks. Completed entries are selected with `++generation.task_name=<NAME>`
and `configs/search_ame_local_pipeline.yaml`.

## Step 2: Gather required info

Use the supplied values and ask only for missing required information. Required
fields differ for protein, ligand, and AME definitions.

### Protein target

| Field | Question | Example | Required |
|---|---|---|---|
| name | "Target name (used as the dict key and `task_name`)?" | `02_PDL1`, `MyTarget_v1` | yes |
| source | "Source directory under `$DATA_PATH/target_data/`?" | `bindcraft_targets`, `custom_targets` | yes (or `target_path`) |
| target_filename | "PDB filename (no `.pdb` extension)?" | `PD-L1`, `IFNAR2` | yes (or `target_path`) |
| target_input | "Chain + residue range — see reference for grammar." | `A1-115`, `A1-50,B1-50` | yes |
| hotspot_residues | "Hotspot residues (interface contact residues)?" | `["A33", "A95", "A102"]` | optional, recommended |
| binder_length | "Binder length range `[min, max]` or single `[length]`?" | `[80, 150]`, `[100]` | optional (default `[60, 120]`) |
| pdb_id | "Reference PDB ID (optional, metadata only)?" | `"2lag"` | optional |

Hotspot selectors encode **chain + residue number**, not amino-acid identity.
For example, Tyr42 (`Y42`) on explicitly specified chain B becomes `B42`;
already supplied chain/residue selectors stay unchanged. Preserve the user's
residue numbering. Without coordinates, report that residue identities and
membership in the structure are unverified.

### Ligand target — protein fields above (minus `target_input`), plus:

| Field | Question | Example | Required |
|---|---|---|---|
| ligand | "3-letter PDB ligand residue code?" | `FAD`, `OQO`, `SAM` | yes (presence marks target as ligand) |
| SMILES | "SMILES string for the ligand?" | `"O=C2C3=Nc1cc(c(...)..."` | recommended; uppercase YAML key |
| ligand_only | "Generate pocket around ligand only (no protein-protein interface)?" | `true` / `false` | optional (default `true`) |
| use_bonds_from_file | "Use bond info from the input PDB/CIF?" | `true` / `false` | optional (default `true`) |
| target_input | not required for ligand targets | — | no |

Check whether the proposed key already exists in the selected YAML mapping.
Preserve an existing entry unless the user requested that entry's update.

Copy a supplied SMILES exactly. Chain-specific ligand selection belongs to
preparation of the input structure; do not invent a `chain` field or protein
`target_input` to represent it in the ligand entry.

## Examples

### Step 3a: Append the YAML block directly (preferred)

Open `configs/targets/targets_dict.yaml` (or `ligand_targets_dict.yaml` for ligand targets), find a similar existing entry as a style template, and append the new block under `target_dict_cfg:`. Two-space indent, single blank line between entries.

#### Protein template

```yaml
  02_PDL1:
    source: bindcraft_targets
    target_filename: PD-L1
    target_input: "A1-115"
    hotspot_residues: ["A37", "A39", "A49", "A98"]
    binder_length: [64, 155]
    pdb_id: "4z18"
```

Rules to match the existing file style (mirrors what `complexa target add` would emit):

- Quote chain/residue ranges (`"A1-115"`, `"A33"`) and any SMILES — flow-style strings get tripped up by `:` and brackets otherwise.
- Use flow-style lists (`["A33", "A95"]`, `[64, 155]`) — that's what the on-disk dump produces.
- `target_input`, `source`, `target_filename` are required for protein. `target_path` (absolute path) can replace `source + target_filename` if the PDB lives outside `$DATA_PATH/target_data/`.

#### Ligand template

```yaml
  41_7BKC_LIGAND:
    source: ligand_targets
    target_filename: 7BKC_ligand_centered
    hotspot_residues: [null]
    binder_length: [100]
    pdb_id: 7BKC
    ligand: 'FAD'
    ligand_only: True
    SMILES: "O=C2C3=Nc1cc(c(cc1N(C3=NC(=O)N2)CC(O)C(O)C(O)COP(=O)(O)OP(=O)(O)OCC6OC(n5cnc4c(ncnc45)N)C(O)C6O)C)C"
    use_bonds_from_file: True
```

The presence of the `ligand:` key flips the target into ligand mode — there is no separate `is_ligand` flag. `target_input` is not required for ligands.

After saving, skip to Step 4 to verify.

## Step 3b: CLI alternative (`complexa target add`)

Use the CLI when you want its automatic chain/residue quoting, the overwrite-confirm prompt, or to wire target creation into a non-Python script. Prefer non-interactive mode for agentic use (`-i / --editor` opens an editor and blocks).

### Confirmed flags (from `src/proteinfoundation/cli/target_cli.py`)

Read [target_schema.md](references/target_schema.md#cli-options) for the flag
table and worked CLI examples when this path is needed. Select the intended
dictionary with `--dict`; `--source` and `--target-filename` should be explicit
when they differ from the CLI defaults. AME registration remains a direct edit.

## Step 4: Verify

First parse the saved YAML with an available YAML parser and check the selected
mapping, field types, requested values, and preservation of other entries.
Read back the resulting block. This verifies configuration editing only.

With a prepared Complexa installation, pipeline configs, environment, and target
structure, additionally check runtime path resolution from that checkout:

```bash
# 1. Select the dictionary that was actually edited.
complexa target show 02_PDL1 --dict configs/targets/targets_dict.yaml

# 2. Resolve the PDB path and validate it exists on disk.
#    This traverses Hydra defaults to find the target dict.
complexa validate target configs/search_binder_local_pipeline.yaml --target 02_PDL1
```

`complexa validate target` (from `src/proteinfoundation/cli/validate.py::validate_target`) checks:

- `DATA_PATH` is set and `target_data/` exists.
- The target name resolves in `target_dict_cfg`.
- Either `target_path` exists, or `$DATA_PATH/target_data/<source>/<target_filename>.pdb` exists.
- `target_input`, `hotspot_residues`, `binder_length` are reported back so a human can sanity-check them.

For ligand targets, use `configs/targets/ligand_targets_dict.yaml` for `show`
and `configs/search_ligand_binder_local_pipeline.yaml` for validation; `--target`
is the target key, not a config path. This validator does not check residue
identities, ligand chemistry, or AME `contig_atoms`; do not describe a path check
as coordinate or scientific validation.

## Step 5: Emit artifact

Save the edited entry and validation status at the user's requested output
location (otherwise `./target_<name>/`). For an offline edit, the saved YAML
block is sufficient; no CLI-generated artifact is required. On a prepared
installation, a target-show record can also be saved:

```bash
mkdir -p target_02_PDL1
complexa target show 02_PDL1 > target_02_PDL1/target_show.txt
```

If a standalone definition is requested, save the actual entry under its
registry key in `target_definition.yaml`. The handoff should include the saved
path, target key, consumer pipeline from Step 1, and
`++generation.task_name=<NAME>`, together with what validation actually ran.
For an incomplete AME request, report the separate template, missing inputs,
and the direct registry edit needed after those inputs arrive.

## Hardware requirements

None for target definition — this is a YAML edit, not a training/inference step. Disk impact is a few KB appended to the targets dict (plus an automatic `.yaml.bak` backup written by `save_targets_dict`).

For the downstream design / evaluate runs that consume the target, defer to `complexa-design` and `references/hardware.md`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Target PDB file: File not found` from `validate target` | `<source>/<target_filename>.pdb` does not exist under `$DATA_PATH/target_data/` | Confirm the PDB stem and source dir. Use `--target-path /full/path.pdb` if the file lives outside `target_data/`. |
| "Target 'X' already exists! Overwrite? (y/N)" | Name collision in dict | Either pick a new name, or pass `-f / --force` to overwrite. |
| Hotspot residue not in PDB | Wrong chain or residue number | Open the PDB, re-check chain letters (case-sensitive) and residue indices. Hotspots use the format `<CHAIN><RESNUM>` — see reference. |
| Chain not found | `target_input` references a chain that does not exist in the PDB | Inspect the PDB with `grep "^ATOM" target.pdb \| awk '{print $5}' \| sort -u`. |
| Ligand code missing from PDB | The 3-letter `ligand` code does not appear as a `HETATM` residue name in the file | Open the PDB and check `HETATM` lines; you may need a `_ligand_centered` variant of the PDB. |
| SMILES parse failure downstream | Bad SMILES string | Validate with `rdkit.Chem.MolFromSmiles(smiles)` before adding. Quote SMILES in shell to escape brackets and parens. |
| Target name with leading digit breaks Hydra interpolation | YAML treats `02_PDL1` as a string fine; some override syntaxes need quoting | Use `++generation.task_name=02_PDL1`; quote if the shell strips characters. |
| `target_input` appears ignored for a ligand target | By design — ligand targets do not use `target_input`; pocket is defined by the ligand | Leave it unset for ligand targets. |

## Reference

- `references/target_schema.md` — every field, chain-spec grammar, AME task-name grammar, three worked examples.
- `configs/targets/targets_dict.yaml` — live protein entries (copy a known-good one as a template).
- `configs/targets/ligand_targets_dict.yaml` — live ligand entries.
- `configs/design_tasks/ame_dict_v2.yaml` — AME task definitions (file-edit only, not exposed via `complexa target` CLI).
- `src/proteinfoundation/cli/target_cli.py` — argparse source of truth.
- `src/proteinfoundation/cli/target_manager.py` — `add_target_cli`, `list_targets`, `show_target`, schema in `TARGET_FIELDS`.
- `src/proteinfoundation/cli/validate.py` — `validate_target` implementation.
