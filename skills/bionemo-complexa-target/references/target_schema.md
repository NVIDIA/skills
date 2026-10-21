# Target schema reference

Authoritative source: `src/proteinfoundation/cli/target_manager.py::TARGET_FIELDS` and the live YAML dicts under `configs/targets/`. This document mirrors them.

## Protein target schema

Stored in `configs/targets/targets_dict.yaml` under `target_dict_cfg.<name>`. Either (`source` + `target_filename`) OR `target_path` is required, plus `target_input`.

| Field | Type | Required | Default | Example | Controls |
|---|---|---|---|---|---|
| `source` | str | yes* | `custom_targets` | `bindcraft_targets` | Subdirectory of `$DATA_PATH/target_data/` where the PDB lives. |
| `target_filename` | str | yes* | (name of target) | `PD-L1` | PDB stem (no `.pdb` extension). Combined with `source` to build the path. |
| `target_path` | str | yes* | `null` | `/abs/path/target.pdb` | Full path to a PDB. If set, `source`+`target_filename` are optional. |
| `target_input` | str | **yes** | `A1-100` | `A1-115`, `A1-50,B1-50` | Chain + residue range (the "target_input" chain-spec). |
| `hotspot_residues` | list[str] | no | `[]` | `["A33", "A95", "A102"]` | Interface residues focused on during binder design. |
| `binder_length` | list[int] | no | `[60, 120]` | `[80, 150]`, `[100]` | Binder length. Two-element list = uniform `[min, max]`; one-element = fixed length. |
| `pdb_id` | str | no | `null` | `"2lag"` | Reference PDB ID, metadata only. |

(* "yes" with the OR rule: either `target_path` OR (`source` AND `target_filename`).)

## Ligand target schema

Stored in `configs/targets/ligand_targets_dict.yaml`. The presence of the `ligand` key is what marks an entry as a ligand target (no separate `is_ligand` flag). `target_input` is not used for ligand targets.

All protein fields above (with `target_input` optional / omitted), plus:

| Field | Type | Required | Default | Example | Controls |
|---|---|---|---|---|---|
| `ligand` | str OR list[str] | **yes** | `null` | `"FAD"`, `["DHZ", "ZN"]` | Three-letter PDB residue name(s) for the ligand. Presence marks target as ligand. |
| `ligand_only` | bool | no | `true` | `true` | If `true`, generate the binding pocket around the ligand only (no protein-protein interface). |
| `SMILES` | str | no | `null` | `"O=C2C3=Nc1cc(...)C"` | SMILES string for the ligand (single-ligand targets only). Optional but recommended. |
| `use_bonds_from_file` | bool | no | `true` | `true` | If `true`, use bond information from the input PDB/CIF file rather than inferring from coordinates. |

Note: `hotspot_residues` is still present for ligand entries but is typically `[null]` since the pocket is ligand-defined.

## `target_input` chain-spec grammar

`target_input` selects which residues of the target PDB are exposed to the binder design model.

| Form | Meaning | Example |
|---|---|---|
| `<CHAIN><START>-<END>` | One contiguous chain segment | `A1-115` = chain A, residues 1 through 115 |
| `<CHAIN><START>-<END>,<CHAIN><START>-<END>` | Multiple chain segments (comma-separated) | `A1-50,B1-50` = chain A residues 1-50 plus chain B residues 1-50 |
| `<CHAIN><START>-<END>/0 <NRES>-<NRES>` | Chain break + contigs syntax (RFdiffusion-style) | `A1-115/0 50-100` = target A1-115 plus a 50-100 residue binder placeholder |

`<CHAIN>` is a single uppercase letter (case-sensitive — `A` and `a` are different). `<START>` and `<END>` are integer residue numbers as they appear in the PDB (not 0-indexed; respects insertion codes only if the PDB does).

When is `target_input` required vs optional?

- **Protein targets**: required. Default is `A1-100`.
- **Ligand targets**: not required (and not used at runtime). The pocket is defined by the ligand position, not a chain range.

## Hotspot residue format

A hotspot is a single residue, identified by chain + residue number:

```
"A33"     # chain A, residue 33
"B17"     # chain B, residue 17
```

Rules:
- Must be a string (always quote in YAML — the YAML dumper auto-quotes any `<chain><digits>` pattern).
- Chain letter is case-sensitive and must match a chain in the PDB.
- An amino-acid label plus an explicitly supplied chain must be translated: Tyr42 (`Y42`) on chain B is selector `B42`. Do not infer the chain from the amino-acid letter or claim the identity was checked without coordinates.
- Residue number must exist in the PDB (use `grep "^ATOM" target.pdb | awk '{print $5, $6}' | sort -u` to enumerate).
- The list may be empty (`[]`) — no hotspots = no special interface focus.
- The list may contain only `[null]` for ligand targets where the pocket is ligand-defined.

CLI form: `--hotspot-residues A33 A95 A102` (space-separated, no quotes needed on the command line).

## `binder_length` semantics

| Value | Meaning |
|---|---|
| `[60, 120]` | Sample binder length uniformly in [60, 120] inclusive. |
| `[100]` | Always generate length-100 binders. |
| `[80, 150]` | Sample uniformly in [80, 150]. |
| `[]` or unset | Falls back to default `[60, 120]`. |

CLI form: `--binder-length 60 120` (two ints) or `--binder-length 100` (one int).

## Source directory convention

`source` is **not** a full path. It is a subdirectory name under `$DATA_PATH/target_data/`. The full path resolution is:

```
${DATA_PATH}/target_data/<source>/<target_filename>.pdb
```

Examples:

| `source` | `target_filename` | Resolved path |
|---|---|---|
| `bindcraft_targets` | `PD-L1` | `${DATA_PATH}/target_data/bindcraft_targets/PD-L1.pdb` |
| `ligand_targets` | `7BKC_ligand_centered` | `${DATA_PATH}/target_data/ligand_targets/7BKC_ligand_centered.pdb` |
| `custom_targets` | `MyTarget_v1` | `${DATA_PATH}/target_data/custom_targets/MyTarget_v1.pdb` |

Override the convention with `--target-path /absolute/path/to/file.pdb` (overrides both `source` and `target_filename`).

Common existing `source` directories: `bindcraft_targets`, `alpha_proteo_targets`, `ligand_targets`, `custom_targets`.

## AME task names (NOT `complexa target`)

The AME pipeline uses **task names**, not targets-dict entries. Task names are defined in `configs/design_tasks/ame_dict_v2.yaml` under `motif_target_dict_cfg:` and are file-edit-only — they have a richer schema than protein/ligand targets and the `complexa target` CLI does not touch them.

### Grammar

```
M{NNNN}_{pdb_id}
```

| Component | Values | Meaning |
|---|---|---|
| `M{NNNN}` | `M0001` … | Zero-padded sequential ID assigned when the task is added. |
| `pdb_id` | 4-char PDB code (lowercase) | Source PDB for the motif + ligand context. |

Examples (drawn from `configs/design_tasks/ame_dict_v2.yaml`): `M0024_1nzy`, `M0096_1chm`.

### AME task schema

Each entry under `motif_target_dict_cfg.<name>` has:

| Field | Type | Example | Notes |
|---|---|---|---|
| `source` | str | `ame_targets` | Subdirectory under `$DATA_PATH/target_data/`. |
| `target_filename` | str | `1nzy_v2` | PDB stem (no `.pdb`). |
| `ligand` | str | `"FAD"` | 3-letter PDB ligand code in the motif PDB. |
| `contig_atoms` | str | `"A64: [O, C]; A86: [CB, CA, N, C]; ..."` | Hand-curated per-residue atom selection for the motif. |
| `binder_length` | list[int] | `[100, 160]` | Length range for the scaffold. |
| `use_bonds_from_file` | bool | `true` | Use bond info from the PDB. |
| `pdb_id` | str | `"1nzy"` | Reference PDB ID, metadata only. |

The motif atom selections and ligand identity must come from the user or a
curated structure/task definition. A task name alone is insufficient. When
those inputs are absent, preserve the registry and put placeholders in a
separate template explicitly marked incomplete.

To run an AME task:

```bash
complexa design configs/search_ame_local_pipeline.yaml \
  ++run_name=ame_1nzy \
  ++generation.task_name=M0024_1nzy
```

**Do not use `complexa target add` for AME tasks** — they live in a different dict and add the `contig_atoms` field (and use `motif_target_dict_cfg` instead of `target_dict_cfg`) that the CLI does not know how to construct.

## CLI options

For `complexa target add`, confirmed by `src/proteinfoundation/cli/target_cli.py`:

| Flag | Type | Applies to | Notes |
|---|---|---|---|
| `name` (positional) | str | both | dict key |
| `--dict PATH` | path | both | select target dictionary explicitly |
| `-i, --interactive` | flag | both | opens an editor; avoid for unattended use |
| `-e, --editor NAME` | str | both | editor command |
| `--source NAME` | str | both | directory under `$DATA_PATH/target_data/` |
| `--target-filename NAME` | str | both | PDB stem without `.pdb` |
| `--target-path PATH` | str | both | full path, instead of source + filename |
| `--target-input SPEC` | str | protein | chain/residue range |
| `--hotspot-residues R [R ...]` | list | both | chain/residue selectors |
| `--binder-length N [N ...]` | int list | both | range or fixed length |
| `--pdb-id ID` | str | both | metadata only |
| `--ligand CODE` | str | ligand | marks ligand mode |
| `--ligand-only` | flag | ligand | pocket-only mode |
| `--smiles STR` | str | ligand | writes uppercase `SMILES` |
| `--use-bonds-from-file` | flag | ligand | use input bonds |
| `-f, --force` | flag | both | overwrite existing without prompting |

`--source` defaults to `custom_targets` for protein or `ligand_targets` for
ligand entries; `--target-filename` defaults to the target name. Ligand entries
default to `ligand_only: true` and `use_bonds_from_file: true`. These commands
do not manage AME entries.

## Worked examples

### 1. Protein target from PDB ID + chain (new BindCraft target)

User wants to design binders against chain A, residues 1-115, of an existing PDB at `${DATA_PATH}/target_data/bindcraft_targets/PD-L1.pdb`, hotspot residues A37, A39, A49, A98, binder length 64-155.

```bash
complexa target add 02_PDL1 \
  --source bindcraft_targets \
  --target-filename PD-L1 \
  --target-input A1-115 \
  --hotspot-residues A37 A39 A49 A98 \
  --binder-length 64 155 \
  --pdb-id 4z18
```

Resulting YAML entry (appended to `configs/targets/targets_dict.yaml`):

```yaml
  02_PDL1:
    target_input: "A1-115"
    source: bindcraft_targets
    target_filename: "PD-L1"
    hotspot_residues: ["A37", "A39", "A49", "A98"]
    binder_length: [64, 155]
    pdb_id: 4z18
```

### 2. Ligand target with SMILES (FAD pocket)

User wants binders for the FAD-binding pocket of 7BKC, fixed length 100.

```bash
complexa target add 41_7BKC_LIGAND \
  --source ligand_targets \
  --target-filename 7BKC_ligand_centered \
  --pdb-id 7BKC \
  --ligand FAD \
  --ligand-only \
  --use-bonds-from-file \
  --smiles "O=C2C3=Nc1cc(c(cc1N(C3=NC(=O)N2)CC(O)C(O)C(O)COP(=O)(O)OP(=O)(O)OCC6OC(n5cnc4c(ncnc45)N)C(O)C6O)C)C" \
  --binder-length 100
```

Resulting YAML entry (appended to `configs/targets/ligand_targets_dict.yaml`):

```yaml
  41_7BKC_LIGAND:
    source: ligand_targets
    target_filename: "7BKC_ligand_centered"
    hotspot_residues: []
    binder_length: [100]
    pdb_id: 7BKC
    ligand: FAD
    ligand_only: True
    SMILES: "O=C2C3=Nc1cc(c(cc1N(C3=NC(=O)N2)CC(O)C(O)C(O)COP(=O)(O)OP(=O)(O)OCC6OC(n5cnc4c(ncnc45)N)C(O)C6O)C)C"
    use_bonds_from_file: True
```

### 3. Ligand target with `--use-bonds-from-file` and no SMILES

User has a curated `_centered` PDB whose bonds are authoritative, and does not want to provide a SMILES (e.g. uncommon ligand). They want a length range 40-88.

```bash
complexa target add Cambridge_bloodsugar_A_4D71_pdb \
  --source ligand_targets \
  --target-filename Cambridge_bloodsugar_A_4D71_pdb \
  --pdb-id 4D71 \
  --ligand UNK \
  --ligand-only \
  --use-bonds-from-file \
  --binder-length 40 88
```

(Note: `--ligand` requires a value; pass a placeholder like `UNK` or the actual 3-letter code if available. The `complexa target add` parser interprets the presence of `--ligand` as marking the target ligand-typed.)

Resulting entry mirrors `Cambridge_bloodsugar_A_4D71_pdb` from `configs/targets/ligand_targets_dict.yaml` — `SMILES: null`, `use_bonds_from_file: True`, `binder_length: [40, 88]`.

## Cross-references

- CLI: `src/proteinfoundation/cli/target_cli.py`
- Manager + schema: `src/proteinfoundation/cli/target_manager.py` (see `TARGET_FIELDS`)
- Validator: `src/proteinfoundation/cli/validate.py::validate_target`
- Protein dict: `configs/targets/targets_dict.yaml`
- Ligand dict: `configs/targets/ligand_targets_dict.yaml`
- AME dict: `configs/design_tasks/ame_dict_v2.yaml`
