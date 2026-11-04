---
name: drug-discovery-pipeline
description: >
  NOTE: molecule and target inputs and your NGC_API_KEY are transmitted to
  external NVIDIA-hosted API endpoints on every call. Use local NIM containers
  for confidential or proprietary data.
  Run a complete computational drug discovery pipeline using NVIDIA BioNeMo NIMs:
  generate drug-like molecules with GenMol, dock them to a protein target with DiffDock,
  then predict binding affinity with Boltz2. Use this skill whenever the user wants to
  generate and screen small molecule drug candidates, perform hit discovery, optimize
  leads against a protein target, or do virtual screening combining molecule generation,
  docking, and affinity prediction. Triggers on: drug discovery pipeline, hit discovery,
  lead optimization, virtual screening, molecule generation, molecular docking, binding
  affinity, GenMol, DiffDock, Boltz2, SMILES, SAFE notation, NIM microservice. This is
  a multi-step pipeline composing three BioNeMo NIMs.
license: Apache-2.0 AND CC-BY-4.0
allowed-tools: Bash, Read, Write, AskUserQuestion
---

# Drug Discovery Pipeline

Screen drug candidates end-to-end using three BioNeMo NIMs in sequence:

```
Step 1: GenMol    →  Step 2: DiffDock  →  Step 3: Boltz2
(Generate mols)      (Dock to target)      (Predict affinity)
```

---

## Overview

This pipeline is used for:
- **De novo hit discovery**: generate drug-like molecules and screen them against a target
- **Lead optimization**: start from a known scaffold and generate improved analogs, then dock and score
- **Virtual screening**: dock a library of candidates and filter by docking confidence + affinity

---

## Before you start

Confirm with the user:
1. **Target protein**: PDB file or sequence of the binding target
2. **Starting point**: de novo (no scaffold) or scaffold decoration (known core)?
3. **Scoring**: drug-likeness (QED) or lipophilicity (LogP)?
4. **API mode**: hosted or local Docker?

For local Docker, do not assume all NIMs are running on `localhost:8000` at the
same time. Either run one container at a time and hand files/results between
steps, or start each NIM on a distinct host port and set the per-step URLs.

---

## Step 1: Generate molecules with GenMol

GenMol requires SAFE notation input (not raw SMILES). Use the `safe-mol` package.

```python
import requests, json, os
import safe as sf                          # pip install safe-mol
from pathlib import Path

NGC_API_KEY = os.getenv("NGC_API_KEY")
HOSTED = True

if HOSTED:
    genmol_url = "https://health.api.nvidia.com/v1/biology/nvidia/genmol/generate"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {NGC_API_KEY}"}
else:
    genmol_url = "http://localhost:8000/generate"
    headers = {"Content-Type": "application/json"}

# De novo generation (no scaffold):
safe_input = "[*{20-30}]"

# Scaffold decoration (known core):
# scaffold_smiles = "c1ccccc1"
# safe_input = sf.encode(scaffold_smiles) + ".[*{5-10}]"

payload = {
    "smiles": safe_input,              # field is named 'smiles' but takes SAFE notation
    "num_molecules": 30,               # request more to compensate for post-generation filtering
    "scoring": "QED",                  # QED or LogP
    "unique": True,
    "temperature": "1.0",             # NOTE: must be string, not float
    "noise": "1.0",                   # NOTE: must be string, not float
}

r = requests.post(genmol_url, headers=headers, json=payload)
r.raise_for_status()
molecules = r.json()["molecules"]
molecules_sorted = sorted(molecules, key=lambda x: x["score"], reverse=True)
top_20 = molecules_sorted[:20]

print(f"Generated {len(molecules)} valid molecules (requested 30)")
print("Top 5 by QED score:")
for m in top_20[:5]:
    print(f"  {m['smiles'][:50]}  score={m['score']:.4f}")
```

---

## Step 2: Dock molecules with DiffDock

Prepare the protein and dock each candidate:

```python
# Load protein (ATOM records only)
receptor_pdb_raw = Path("target.pdb").read_text()
receptor_pdb = "\n".join(line for line in receptor_pdb_raw.splitlines()
                          if line.startswith("ATOM"))

if HOSTED:
    diffdock_url = "https://health.api.nvidia.com/v1/biology/mit/diffdock"
else:
    diffdock_url = "http://localhost:8000/molecular-docking/diffdock/generate"

docking_results = []

for i, mol in enumerate(top_20):
    payload = {
        "protein": receptor_pdb,
        "ligand": mol["smiles"],
        "ligand_file_type": "txt",     # "txt" for SMILES input
        "num_poses": 5,
        "time_divisions": 20,
        "steps": 18,
        "save_trajectory": False,
    }

    r = requests.post(diffdock_url, headers=headers, json=payload)
    r.raise_for_status()
    result = r.json()

    best_conf = result["position_confidence"][0]  # rank 1 pose
    best_pose = result["ligand_positions"][0]

    docking_results.append({
        "smiles": mol["smiles"],
        "qed_score": mol["score"],
        "docking_confidence": best_conf,
        "best_pose_sdf": best_pose,
    })
    print(f"  Mol {i+1:2d}: QED={mol['score']:.3f}  docking_conf={best_conf:.4f}")

# Rank by docking confidence
docking_results.sort(key=lambda x: x["docking_confidence"], reverse=True)
print(f"\nTop 3 by docking confidence:")
for d in docking_results[:3]:
    print(f"  {d['smiles'][:50]}  conf={d['docking_confidence']:.4f}")
```

---

## Step 3: Predict binding affinity with Boltz2

For the top docking candidates, predict structure-based binding affinity:

```python
if HOSTED:
    boltz_url = "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict"
else:
    boltz_url = "http://localhost:8000/biology/mit/boltz2/predict"

# Use the target protein sequence (not PDB)
target_sequence = "<YOUR_TARGET_PROTEIN_SEQUENCE>"

affinity_results = []
for d in docking_results[:5]:  # score top 5 docking hits
    payload = {
        "polymers": [
            {"id": "A", "molecule_type": "protein", "sequence": target_sequence}
        ],
        "ligands": [
            {"id": "L1", "smiles": d["smiles"], "predict_affinity": True}
        ],
        "recycling_steps": 3,
        "sampling_steps": 50,
        "diffusion_samples": 1,
        "output_format": "mmcif",
    }

    r = requests.post(boltz_url, headers=headers, json=payload)
    r.raise_for_status()
    result = r.json()

    aff = result["affinities"]["L1"]
    pic50 = aff["affinity_pic50"][0]
    prob_binding = aff["affinity_probability_binary"][0]

    affinity_results.append({
        **d,
        "pic50": pic50,
        "probability_binding": prob_binding,
    })
    print(f"  {d['smiles'][:40]}  pIC50={pic50:.2f}  P(bind)={prob_binding:.3f}")

# Final ranking by pIC50
affinity_results.sort(key=lambda x: x["pic50"], reverse=True)
```

---

## Interpreting results

- **GenMol QED score**: 0–1; >0.5 is drug-like
- **DiffDock confidence**: higher = more reliable binding pose prediction
- **Boltz2 pIC50**: predicted -log10(IC50); >6 = sub-micromolar, >8 = very potent
- **P(bind)**: probability of binary binding; >0.7 = likely binder

---

## Quick reference — skill dependencies

| Step | Skill | Key endpoint |
|---|---|---|
| Molecule generation | `genmol-nim` | `/biology/nvidia/genmol/generate` |
| Docking | `diffdock-nim` | `/molecular-docking/diffdock/generate` |
| Affinity prediction | `boltz2-nim` | `/biology/mit/boltz2/predict` |
