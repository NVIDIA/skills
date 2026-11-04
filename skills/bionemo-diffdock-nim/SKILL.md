---
name: diffdock-nim
description: >
  Run DiffDock molecular docking via NVIDIA NIM to predict small-molecule binding poses against protein targets. Use for DiffDock, molecular docking, ligand docking, blind docking, SMILES or SDF ligands, ranked poses, confidence scores, hosted NVIDIA API, or local Docker deployment.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "requests>=2.28"
allowed-tools: Bash, Read, Write, AskUserQuestion
---

# DiffDock NIM

Predict protein-ligand binding poses with blind docking. Use this guide for
first-pass hosted/local usage; load supplemental files only when needed:

- `references/api.md`: exact hosted/local endpoints, schemas, Docker flags.
- `references/science.md`: docking use cases, limits, and handoffs.
- `references/parameters.md`: ligand formats, pose counts, diffusion controls.
- `references/validation.md`: receptor, ligand, pose, and confidence checks.
- `references/examples.md`: compact hosted/local and pose-saving patterns.

## Choose Mode

Ask only when context is unclear:

> Hosted NVIDIA API or local Docker NIM?

- Hosted: `https://health.api.nvidia.com/v1/biology/mit/diffdock`
- Local: `http://localhost:8000/molecular-docking/diffdock/generate`

The hosted and local paths differ. Local has no `/v1/` prefix and uses the
`/molecular-docking/` route. Hosted requests use `Authorization: Bearer $NGC_API_KEY`. Supported local Docker
startup uses `NGC_API_KEY` (or `NVIDIA_API_KEY` via the preflight) for
registry login, entitlement checks, and first-run model downloads; pass it
into the container with `-e NGC_API_KEY`. Local inference requests use no
auth header after readiness. Warm-cache key-free startup varies by
image/version and should not be assumed.

## Local Docker

For the exact local preflight (`.env` load, `NVIDIA_API_KEY` fallback,
`LOCAL_NIM_CACHE`, `NVIDIA_VISIBLE_DEVICES=0`, `--shm-size=2G`, both `--ulimit`
flags, `docker login`, and the `docker run` for `nvcr.io/nim/mit/diffdock:2.2.0`),
copy the command block in [`references/api.md`](references/api.md) under
**Docker Reference** verbatim.

Readiness:

```bash
until curl -sf http://localhost:8000/v1/health/ready; do sleep 5; done
```

## Prepare Inputs

Protein receptor must be ATOM records only. Strip headers, water, and HETATM.

```python
from pathlib import Path
raw_pdb = Path("protein.pdb").read_text()
protein = "\n".join(line for line in raw_pdb.splitlines() if line.startswith("ATOM"))
if not protein:
    raise ValueError("protein.pdb has no ATOM records")
```

Ligand options:

- SMILES: `ligand = "CC(=O)OC1=CC=CC=C1C(=O)O"`; `ligand_file_type = "txt"`.
- SDF: `ligand = Path("ligand.sdf").read_text()`; `ligand_file_type = "sdf"`.
- MOL2: `ligand_file_type = "mol2"`.

Do not use `"smiles"` as `ligand_file_type`; SMILES is `"txt"`.

## Request Pattern

```python
import os
import requests

HOSTED = True
url = (
    "https://health.api.nvidia.com/v1/biology/mit/diffdock"
    if HOSTED else "http://localhost:8000/molecular-docking/diffdock/generate"
)
headers = {"Content-Type": "application/json"}
if HOSTED:
    headers["Authorization"] = f"Bearer {os.getenv('NGC_API_KEY')}"

payload = {
    "protein": protein,
    "ligand": ligand,
    "ligand_file_type": ligand_file_type,
    "num_poses": 10,
    "time_divisions": 20,
    "steps": 18,
    "save_trajectory": False,
}
response = requests.post(url, headers=headers, json=payload, timeout=300)
response.raise_for_status()
result = response.json()
```

## Save And Report Output

`ligand_positions` and `position_confidence` are parallel ranked lists.
`position_confidence[0]` is the rank-1 pose confidence.

Save the ranked pose SDFs using the snippet in
[`references/examples.md`](references/examples.md) under **Save Ranked Poses**.

View pose SDF files with the receptor in PyMOL, ChimeraX, or UCSF Chimera. For
pose sanity checks and confidence caveats, read `references/validation.md`.

## Limits And Troubleshooting

- Max `num_poses`: 100. Max `time_divisions`: 20. Max `steps`: 18.
- Single GPU; local minimum is about 24 GB VRAM.
- `422`: invalid `ligand_file_type`, invalid SMILES/SDF, or no ATOM records.
- Empty poses: validate receptor ATOM records and ligand parseability.
- Local URL 404 usually means the wrong hosted path or an accidental `/v1/`.
