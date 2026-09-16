---
name: msa-structure-prediction-pipeline
description: >
  NOTE: your protein sequence and the retrieved MSA alignment are transmitted to
  external NVIDIA-hosted APIs (health.api.nvidia.com) on every call. Use local
  NIM containers for confidential or proprietary sequences.
  Run a complete protein structure prediction pipeline using NVIDIA BioNeMo NIMs:
  search for MSA alignments with MSA-Search (ColabFold), then predict the structure
  with OpenFold3 using the retrieved alignments. Use this skill whenever the user wants
  to predict a protein structure with maximum accuracy using MSA context, run the
  full AlphaFold3-style pipeline, generate MSA-informed structure predictions, or
  improve structure prediction accuracy by providing evolutionary information.
  Triggers on: MSA structure prediction pipeline, structure prediction pipeline, MSA-informed prediction, OpenFold3,
  ColabFold MSA, AlphaFold3 pipeline, protein structure, homology search, a3m alignment,
  UniRef30, NIM microservice. This pipeline chains MSA-Search and OpenFold3.
license: Apache-2.0 AND CC-BY-4.0
allowed-tools: Bash, Read, Write, AskUserQuestion
---

# MSA Structure Prediction Pipeline

Predict protein structures with high accuracy by chaining two BioNeMo NIMs:

```
Step 1: MSA-Search  →  Step 2: OpenFold3
(Search homologs)       (Predict structure with MSA)
```

---

## Overview

Why chain these NIMs?
- **MSA-Search** finds evolutionary homologs in UniRef30 and ColabFold databases using GPU-accelerated MMSeqs2. The resulting alignment provides crucial evolutionary information.
- **OpenFold3** uses the MSA to improve structure prediction accuracy — especially for sequences where no close homolog exists in PDB.
- Running MSA-Search first means OpenFold3 gets the full evolutionary context rather than a single-sequence prediction.

---

## Before you start

Confirm with the user:
1. **Query sequence**: amino acid sequence to predict
2. **MSA depth**: how many sequences to retrieve (default 500; more = slower but more context)
3. **API mode**: hosted or local Docker?

Note: local MSA-Search requires 1.4 TB of database storage — strongly recommend hosted unless the user has that infrastructure.

For local Docker, do not assume MSA-Search and OpenFold3 are both on
`localhost:8000` concurrently. Run one container at a time and hand off the A3M
file, or start each NIM on a distinct host port and set the URLs explicitly.

---

## Step 1: Search for MSA with MSA-Search

```python
import requests, json, os
from pathlib import Path

NGC_API_KEY = os.getenv("NGC_API_KEY")
HOSTED = True

query_sequence = "<YOUR_PROTEIN_SEQUENCE>"

if HOSTED:
    msa_url = "https://health.api.nvidia.com/v1/biology/colabfold/msa-search/predict"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {NGC_API_KEY}"}
else:
    msa_url = "http://localhost:8000/biology/colabfold/msa-search/predict"
    headers = {"Content-Type": "application/json"}

payload = {
    "sequence": query_sequence,
    "databases": ["Uniref30_2302", "colabfold_envdb_202108"],
    "e_value": 0.0001,
    "output_alignment_formats": ["a3m"],
}

r = requests.post(msa_url, headers=headers, json=payload)
r.raise_for_status()
msa_result = r.json()

# Extract the A3M alignment
a3m_alignment = msa_result["alignments"]["Uniref30_2302"]["a3m"]["alignment"]

# Save for reference
with open("query_msa.a3m", "w") as f:
    f.write(a3m_alignment)

# Count sequences in alignment
n_seqs = a3m_alignment.count(">")
print(f"Step 1 complete: found {n_seqs} homologous sequences")
print(f"MSA saved to query_msa.a3m")
```

---

## Step 2: Predict structure with OpenFold3

Pass the MSA directly into OpenFold3's `msa` field:

```python
if HOSTED:
    of3_url = "https://health.api.nvidia.com/v1/biology/openfold/openfold3/predict"
else:
    of3_url = "http://localhost:8000/biology/openfold/openfold3/predict"

# Build the OpenFold3 MSA structure from the retrieved alignment
msa_data = {
    "uniref30": {
        "a3m": {
            "alignment": a3m_alignment,
            "format": "a3m"
        }
    }
}

# Optionally also include colabfold_envdb alignment if requested
# env_alignment = msa_result["alignments"]["colabfold_envdb"]["a3m"]["alignment"]
# msa_data["colabfold_env"] = {"a3m": {"alignment": env_alignment, "format": "a3m"}}

payload = {
    "inputs": [{
        "input_id": "prediction_with_msa",
        "output_format": "pdb",
        "molecules": [
            {
                "type": "protein",
                "sequence": query_sequence,
                "diffusion_samples": 1,
                "msa": msa_data
            }
        ]
    }]
}

r = requests.post(of3_url, headers=headers, json=payload, timeout=300)
r.raise_for_status()
result = r.json()

output = result["outputs"][0]
for i, sample in enumerate(output["structures_with_scores"]):
    fmt = sample["format"]
    filename = f"predicted_structure_{i+1}.{fmt}"
    with open(filename, "w") as f:
        f.write(sample["structure"])
    print(f"\nStep 2 complete: {filename} saved")
    print(f"  Confidence:  {sample['confidence_score']:.4f}")
    print(f"  pLDDT:       {sample['complex_plddt_score']:.4f}")
    print(f"  pTM:         {sample['ptm_score']:.4f}")
```

---

## Comparing single-sequence vs MSA-informed prediction

If the user wants to see the impact of MSA, run OpenFold3 twice — once with the full MSA and once with just the query sequence as a minimal alignment:

```python
# Minimal MSA (single sequence — same as no MSA context):
minimal_msa = {
    "main": {
        "a3m": {
            "alignment": f">query\n{query_sequence}",
            "format": "a3m"
        }
    }
}
```

A larger, higher-quality MSA typically yields higher pLDDT and lower pDE, especially for proteins with many known homologs.

---

## For protein complexes

Use the `/paired/predict` endpoint of MSA-Search to get paired alignments for multi-chain complexes, then pass each chain's alignment into the corresponding molecule's `msa` field and `paired_msa` fields:

```python
# Paired MSA search endpoint for complexes:
msa_paired_url = "https://health.api.nvidia.com/v1/biology/colabfold/msa-search/paired/predict"
paired_payload = {
    "sequences": [chain_A_sequence, chain_B_sequence],
    "e_value": 0.0001,
}
```

---

## Quick reference — skill dependencies

| Step | Skill | Key endpoint |
|---|---|---|
| MSA search | `msa-search-nim` | `/biology/colabfold/msa-search/predict` |
| Structure prediction | `openfold3-nim` | `/biology/openfold/openfold3/predict` |
