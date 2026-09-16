# MSA-Search NIM — API Reference

## Endpoints

| Mode | Method | URL |
|---|---|---|
| Hosted — standard MSA | POST | `https://health.api.nvidia.com/v1/biology/colabfold/msa-search/predict` |
| Hosted — paired MSA | POST | `https://health.api.nvidia.com/v1/biology/colabfold/msa-search/paired/predict` |
| Hosted — structure templates | — | Not exposed on `health.api` during May 2026 validation; use local Docker |
| Local — standard MSA | POST | `http://localhost:8000/biology/colabfold/msa-search/predict` |
| Local — paired MSA | POST | `http://localhost:8000/biology/colabfold/msa-search/paired/predict` |
| Local — structure templates | POST | `http://localhost:8000/biology/colabfold/msa-search/structure-templates/predict` |
| Local — DB config | GET | `http://localhost:8000/biology/colabfold/msa-search/config/msa-database-configs` |
| Health (local) | GET | `http://localhost:8000/v1/health/ready` |

**IMPORTANT**: Local paths have no `/v1/` prefix; hosted paths do.

---

## Standard MSA Search — `/predict`

### Request Body

| Field | Type | Required | Default | Range | Notes |
|---|---|---|---|---|---|
| `sequence` | string | Yes | — | 1–4096 chars | Valid amino acids: ARNDCQEGHILKMFPSTWYVX |
| `databases` | list[string] | No | `["all"]` | 1–5 items | `"Uniref30_2302"`, `"colabfold_envdb_202108"`, `"all"` — **case-sensitive** |
| `search_type` | string | No | `"colabfold"` | — | `"colabfold"` (cascaded, more sensitive) or `"alphafold2"` (single-pass) |
| `e_value` | float | No | `0.0001` | 0.0–1.0 | Hit filtering threshold |
| `iterations` | integer | No | `1` | 1–6 | Ignored for cascaded search |
| `max_msa_sequences` | integer | No | `500` | 1–500 | Must equal `NIM_GLOBAL_MAX_MSA_DEPTH` when GPU Server enabled |
| `NIM_MODEL_PROFILE` | (auto: `databases:all`) | Set to a hash from `list-model-profiles` to download only that profile's databases — cuts storage & startup time. `databases:uniref30` (~500 GB) is the paired-search profile; `databases:pdb70` (~100 MB) is a smoke test. |
| `NIM_MODEL_NAME` | (unset) | Path to a mounted directory of custom/manually-downloaded MMSeqs2 databases. Completely replaces profile databases; NIM uses only DBs found under this path. |
| `output_alignment_formats` | list[string] | No | `["a3m"]` | — | `"a3m"`, `"fasta"` |

### Response Body

```json
{
  "alignments": {
    "<database_name>": {
      "<format>": {
        "alignment": "<a3m or fasta string>",
        "format": "a3m"
      }
    }
  },
  "metrics": {}
}
```

---

## Paired MSA Search — `/paired/predict`

### Request Body

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `sequences` | list[string] or dict | Yes | — | Minimum 2 chains (for complex) |
| `databases` | list[string] | No | `["all"]` | Same options as standard |
| `e_value` | float | No | `0.0001` | 0.0–1.0 |
| `max_msa_sequences` | integer | No | `500` | |
| `pairing_strategy` | string | No | `"greedy"` | `"greedy"` (max coverage rows) or `"complete"` (full coverage only) |

### Response Body

```json
{
  "alignments_by_chain": {
    "<chain_id>": {
      "<database_name>": {
        "<format>": {
          "alignment": "<string>",
          "format": "a3m"
        }
      }
    }
  },
  "metrics": {}
}
```

Chains are keyed by their order (`"A"`, `"B"`, etc.). All chains have equal numbers of MSA sequences, paired by species.

---

## Structure Template Search — `/structure-templates/predict`

This endpoint is documented for running NIM instances and validated through
local Docker. The hosted `health.api` path returned HTTP 404 during May 2026
validation.

### Request Body

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `sequence` | string | Yes | — | 1–4096 chars |
| `structural_template_databases` | list[string] | No | `["pdb70_220313"]` | Use canonical names such as `"pdb70_220313"` or `"pdb100_230517"` |
| `msa_databases` | list[string] | No | `["all"]` | Same as standard MSA |
| `e_value` | float | No | `0.0001` | 0.0–1.0 |
| `max_structures` | integer | No | `20` | |
| `max_msa_sequences` | integer | No | `500` | |

### Response Body

```json
{
  "alignments": { ... },
  "search_hits": {
    "<database_name>": {
      "m8": {
        "hits": "<M8 tabular string: query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,cigar>",
        "format": "m8"
      }
    }
  },
  "structures": {
    "<pdb_id>": {
      "structure": "<mmCIF file content>",
      "format": "mmcif"
    }
  },
  "metrics": {}
}
```

---

## Docker Reference

```bash
set -a
[ -f .env ] && . ./.env
set +a

if [ -z "${NGC_API_KEY:-}" ] && [ -n "${NVIDIA_API_KEY:-}" ]; then
  export NGC_API_KEY="$NVIDIA_API_KEY"
fi
: "${NGC_API_KEY:?Set NGC_API_KEY or NVIDIA_API_KEY in the environment or repo-root .env}"

: "${LOCAL_NIM_CACHE:?Set LOCAL_NIM_CACHE in the environment or repo-root .env}"
mkdir -p "${LOCAL_NIM_CACHE}"
chmod 755 "${LOCAL_NIM_CACHE}"

docker run --rm --name msa-search \
  --runtime=nvidia \
  --gpus all \
  -e NGC_API_KEY \
  -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache" \
  -p 8000:8000 \
  nvcr.io/nim/colabfold/msa-search:2
```

| Flag | Value | Notes |
|---|---|---|
| `--gpus` | `all` | Multi-GPU supported; 48 GB GPUs need ≥2 units |
| Cache mount | `/opt/nim/.cache` | ~1.4 TB for `databases:all`; far less with a task-specific `NIM_MODEL_PROFILE` (~500 GB UniRef30, ~100 MB PDB70) |
| Image | `nvcr.io/nim/colabfold/msa-search:2` | v2.3.0 as of 2025; `:2` is the major version tag |

### Key Environment Variables

| Variable | Default | Notes |
|---|---|---|
| `NIM_GLOBAL_MAX_MSA_DEPTH` | `500` | Must match `max_msa_sequences` in requests (GPU Server mode) |
| `NIM_MODEL_PROFILE` | (auto: `databases:all`) | Set to a hash from `list-model-profiles` to download only that profile's databases — cuts storage & startup time. `databases:uniref30` (~500 GB) is the paired-search profile; `databases:pdb70` (~100 MB) is a smoke test. |
| `NIM_MODEL_NAME` | (unset) | Path to a mounted directory of custom/manually-downloaded MMSeqs2 databases. Completely replaces profile databases; NIM uses only DBs found under this path. |

---

## Hardware Requirements

| Component | Requirement |
|---|---|
| GPU | A100 80GB, H100, B200, L40S (≥2 units), RTX 6000 Ada (≥2 units) |
| Min VRAM | 48 GB |
| CPU | 24 cores minimum |
| RAM | 64 GB minimum |
| Storage | 1660 GB NVMe SSD |
