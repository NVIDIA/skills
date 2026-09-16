# Boltz2 NIM — Full API Reference

## Endpoints

| Mode | Prediction | Health check |
|------|-----------|--------------|
| Hosted | `POST https://health.api.nvidia.com/v1/biology/mit/boltz2/predict` | N/A |
| Local | `POST http://localhost:8000/biology/mit/boltz2/predict` | `GET http://localhost:8000/v1/health/ready` |

---

## Request body schema

### Top-level fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `polymers` | list[Polymer] | yes | — | 1–12 polymer chains |
| `ligands` | list[Ligand] | no | [] | 0–20 small molecules |
| `constraints` | list[Pocket\|Bond] | no | [] | Structural constraints |
| `recycling_steps` | int | no | 3 | Model recycles (1–10); more = better quality |
| `sampling_steps` | int | no | 50 | Diffusion steps (10–1000) |
| `diffusion_samples` | int | no | 1 | Structures to generate (1–25) |
| `step_scale` | float | no | 1.638 | Sampling temperature (0.5–5.0) |
| `sampling_steps_affinity` | int | no | 200 | Affinity diffusion steps (10–1000) |
| `diffusion_samples_affinity` | int | no | 5 | Affinity samples (1–10; use 1 for speed) |
| `output_format` | string | no | "mmcif" | Only "mmcif" currently supported |
| `without_potentials` | bool | no | false | Exclude potentials from output |
| `concatenate_msas` | bool | no | false | Merge MSAs across chains |
| `affinity_mw_correction` | bool | no | false | Molecular weight correction (for metals) |
| `write_full_pae` | bool | no | false | Include full PAE matrix in output |

---

### Polymer object

| Field | Type | Required | Description |
|---|---|---|---|
| `molecule_type` | string | yes | `"protein"`, `"dna"`, or `"rna"` |
| `sequence` | string | yes | Amino acid or nucleotide sequence (max 4,096 chars) |
| `id` | string | no | Chain ID: single letter A–Z or 4-char alphanumeric |
| `cyclic` | bool | no | Whether the chain is cyclic (default false) |
| `msa` | dict | no | Custom MSA (proteins only); see MSA section below |
| `modifications` | list | no | Chemical modifications per residue |
| `structural_templates` | list | no | Template structures to guide prediction |

**Modifications entry:**
```json
{"ccd": "SEP", "position": 15}
```
`ccd`: 1–3 char CCD code; `position`: 1-based residue index.

**Structural template entry:**
```json
{"structure": "<cif or pdb content>", "format": "cif", "name": "optional_name"}
```

**MSA format:**
```json
{"alignment": "<msa content>", "format": "a3m", "rank": 0}
```
Accepted formats: `csv`, `fasta`, `a3m`, `sto`.

MSA records are nested under an arbitrary database/source key and format key:

```json
{
  "msa_search": {
    "a3m": {
      "alignment": ">query\nMTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPT",
      "format": "a3m",
      "rank": 0
    }
  }
}
```

Use `alignment`, not `data`. This shape was validated locally with A3M output
from the MSA-Search NIM. Older examples that use `data` fail with HTTP 422 on
the validated Boltz2 NIM.

---

### Ligand object

Provide exactly one of `ccd` or `smiles`, not both.

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | no | Chain ID for this ligand |
| `smiles` | string | no | SMILES string |
| `ccd` | string | no | 1–3 char CCD code (e.g. `"ATP"`, `"HEM"`) |
| `predict_affinity` | bool | no | Enable affinity prediction (max 1 per request) |

---

### Constraint objects

**Pocket constraint** — guides a ligand toward a known binding site:
```json
{
  "constraint_type": "pocket",
  "binder": "L1",
  "contacts": [
    {"id": "A", "residue_index": 25},
    {"id": "A", "residue_index": 47}
  ]
}
```

**Bond constraint** — enforces a covalent bond between atoms:
```json
{
  "constraint_type": "bond",
  "atoms": [
    {"id": "A", "residue_index": 15, "atom_name": "SG"},
    {"id": "L1", "residue_index": 1, "atom_name": "C1"}
  ]
}
```

---

## Response schema

```json
{
  "structures": [
    {
      "structure": "<mmcif content as string>",
      "format": "mmcif",
      "name": "optional_identifier",
      "source": "optional_source_ref"
    }
  ],
  "confidence_scores": [0.87],
  "affinities": {
    "L1": {
      "affinity_pic50": [6.2],
      "affinity_pred_value": [2.1],
      "affinity_probability_binary": [0.94]
    }
  },
  "metrics": {}
}
```

- `structures`: one entry per `diffusion_samples`
- `confidence_scores`: one float per structure; higher is better
- `affinities`: only present when a ligand has `predict_affinity: true`; keyed by ligand ID
  - `affinity_pic50`: `list[float]` — predicted pIC50 values (one per `diffusion_samples_affinity`); index `[0]` for the consensus
  - `affinity_pred_value`: `list[float]` — affinity prediction values (same length as `affinity_pic50`)
  - `affinity_probability_binary`: `list[float]` — probability of binding (0–1), one per sample
  - `model_1_*` / `model_2_*`: per-ensemble-model breakdowns (same keys, also lists)
- `metrics`: runtime diagnostics, useful for debugging

---

## Example: full-featured request

```json
{
  "polymers": [
    {
      "id": "A",
      "molecule_type": "protein",
      "sequence": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT",
      "modifications": [{"ccd": "SEP", "position": 10}]
    },
    {
      "id": "B",
      "molecule_type": "dna",
      "sequence": "ATCGATCGATCG"
    }
  ],
  "ligands": [
    {
      "id": "L1",
      "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
      "predict_affinity": true
    }
  ],
  "constraints": [
    {
      "constraint_type": "pocket",
      "binder": "L1",
      "contacts": [{"id": "A", "residue_index": 25}]
    }
  ],
  "recycling_steps": 3,
  "sampling_steps": 50,
  "diffusion_samples": 1,
  "step_scale": 1.638,
  "sampling_steps_affinity": 200,
  "diffusion_samples_affinity": 5,
  "output_format": "mmcif"
}
```
