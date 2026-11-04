# OpenFold3 NIM — API Reference

## Endpoints

| Mode | Method | URL |
|---|---|---|
| Hosted | POST | `https://health.api.nvidia.com/v1/biology/openfold/openfold3/predict` |
| Local Docker | POST | `http://localhost:8000/biology/openfold/openfold3/predict` |
| Health (local) | GET | `http://localhost:8000/v1/health/ready` |
| Liveness (local) | GET | `http://localhost:8000/v1/health/live` |

**IMPORTANT**: Local path has no `/v1/` prefix; hosted path does.

---

## Request Body Schema

```json
{
  "request_id": "<string, optional, max 128 chars>",
  "inputs": [ <InputObject> ]
}
```

`inputs` is required, must contain exactly 1 item.

### InputObject

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `input_id` | string | No | — | Echoed in response; max 128 chars |
| `output_format` | string | No | `"cif"` | `"cif"` or `"pdb"` |
| `molecules` | array | Yes | — | 1–32 molecule objects |

### MoleculeObject

| Field | Type | Required | Condition | Notes |
|---|---|---|---|---|
| `type` | string | Yes | — | `"protein"`, `"rna"`, `"dna"`, `"ligand"` |
| `id` | string or array | No | — | Chain ID: single letter A–Z, or 4 alphanumeric chars |
| `sequence` | string | Conditional | protein/dna/rna | Amino acids (protein), ATCG (DNA), AUCG (RNA); 2–4096 chars |
| `smiles` | string | Conditional | ligand | SMILES string; mutually exclusive with `ccd_codes` |
| `ccd_codes` | string | Conditional | ligand | CCD code e.g. `"ATP"`, `"NAG"`; 1–5 chars |
| `msa` | object | No | protein/rna | MSA alignment data (see below) |
| `paired_msa` | object | No | protein | Paired MSA for complex predictions |
| `diffusion_samples` | integer | No | 1 | 1–5 structure samples per molecule |
| `structural_templates` | array | No | — | Template CIFs for guided prediction (v1.1.0+) |

### MSA Object (nested 3 levels)

```json
"msa": {
  "<database_name>": {
    "<format_name>": {
      "alignment": ">query\n<SEQUENCE>",
      "format": "a3m",
      "rank": -1
    }
  }
}
```

- `database_name`: arbitrary string key (e.g. `"main"`, `"uniref30"`)
- `format_name`: arbitrary string key (e.g. `"a3m"`)
- `alignment`: A3M or CSV string; must start with FASTA header `>query\n`
- `format`: `"a3m"` or `"csv"`
- `rank`: integer, default `-1`

### StructuralTemplate Object

| Field | Type | Required | Notes |
|---|---|---|---|
| `structure` | string | Yes | Full CIF file content |
| `format` | string | Yes | Always `"cif"` |
| `name` | string | No | Identifier |
| `chain_id` | string | No | 1–10 alphanumeric chars |

---

## Response Schema

```json
{
  "request_id": "<echoed>",
  "outputs": [
    {
      "input_id": "<echoed>",
      "structures_with_scores": [
        {
          "structure": "<PDB or CIF file content>",
          "format": "cif | pdb",
          "confidence_score": 0.85,
          "complex_plddt_score": 0.82,
          "complex_pde_score": 1.2,
          "ptm_score": 0.78,
          "iptm_score": 0.71
        }
      ],
      "runtime_metrics": {}
    }
  ]
}
```

### Score Meanings

| Score | Description |
|---|---|
| `confidence_score` | Overall ranking score for this sample |
| `complex_plddt_score` | Average predicted local distance difference test (0–1; higher = better) |
| `complex_pde_score` | Average predicted distance error (Å; lower = better) |
| `ptm_score` | Predicted TM-score for monomer quality (0–1) |
| `iptm_score` | Predicted interface TM-score (0–1; relevant for complexes) |

---

## Docker Reference

Provide credentials and the cache path through the environment; this example
does not load credential files. Keep shell tracing disabled. Registry login
uses the key on stdin, and the container starts only if login succeeds.
The container uses the key for entitlement checks and first-run model downloads
of about 10–15 GB. Local inference is unauthenticated, so the published host
port binds to loopback.

```bash
set +x

if [ -z "${NGC_API_KEY:-}" ] && [ -n "${NVIDIA_API_KEY:-}" ]; then
  NGC_API_KEY="$NVIDIA_API_KEY"
fi
: "${NGC_API_KEY:?Set NGC_API_KEY or NVIDIA_API_KEY}"
export NGC_API_KEY
: "${LOCAL_NIM_CACHE:?Set LOCAL_NIM_CACHE}"

mkdir -p "${LOCAL_NIM_CACHE}"
chmod 755 "${LOCAL_NIM_CACHE}"

printf '%s\n' "$NGC_API_KEY" | \
  docker login nvcr.io --username '$oauthtoken' --password-stdin && \
docker run --rm --name openfold3 \
  --runtime=nvidia \
  --gpus "device=0" \
  --shm-size=16g \
  -e NGC_API_KEY \
  -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache" \
  -p 127.0.0.1:8000:8000 \
  nvcr.io/nim/openfold/openfold3:latest
```

| Flag | Value | Notes |
|---|---|---|
| `--gpus` | `device=0` | Single GPU only; choose another device only when required |
| `--shm-size` | `16g` | Required |
| `-p` | `127.0.0.1:8000:8000` | Bind the unauthenticated API to the host's loopback address |
| Cache mount | `/opt/nim/.cache` | ~10–15 GB model weights |
| Image | `nvcr.io/nim/openfold/openfold3:latest` | v1.4.0 as of 2025 |

---

## Annotated Example Request

```json
{
  "inputs": [{
    "input_id": "my_prediction",
    "output_format": "pdb",
    "molecules": [
      {
        "type": "protein",
        "id": "A",
        "sequence": "MKTVRQERLKSIVR",
        "diffusion_samples": 1,
        "msa": {
          "main": {
            "a3m": {
              "alignment": ">query\nMKTVRQERLKSIVR",
              "format": "a3m"
            }
          }
        }
      },
      {
        "type": "ligand",
        "id": "L",
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
      }
    ]
  }]
}
```

---

## Hardware Requirements

| GPU | VRAM | Architecture |
|---|---|---|
| H100 | 80 GB | Hopper (optimal) |
| A100 | 80 GB | Ampere |
| L40S | 48 GB | Ada Lovelace |
| RTX 6000 Ada | 48 GB | Ada Lovelace |

- Sequences >1800 residues require ≥80 GB GPU VRAM
- Driver: ≥590.44; Container Toolkit: ≥1.13.5
