# DiffDock Examples

## Hosted SMILES Request

```python
import os
from pathlib import Path

import requests

protein = "\n".join(
    line for line in Path("protein.pdb").read_text().splitlines()
    if line.startswith("ATOM")
)
if not protein:
    raise ValueError("protein.pdb did not contain ATOM records")

payload = {
    "protein": protein,
    "ligand": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "ligand_file_type": "txt",
    "num_poses": 10,
    "time_divisions": 20,
    "steps": 18,
    "save_trajectory": False,
}
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('NGC_API_KEY')}",
}
r = requests.post(
    "https://health.api.nvidia.com/v1/biology/mit/diffdock",
    headers=headers,
    json=payload,
    timeout=300,
)
r.raise_for_status()
result = r.json()
```

## Local SMILES Request

```python
import requests
from pathlib import Path

protein = "\n".join(
    line for line in Path("protein.pdb").read_text().splitlines()
    if line.startswith("ATOM")
)
payload = {
    "protein": protein,
    "ligand": "c1ccccc1",
    "ligand_file_type": "txt",
    "num_poses": 10,
}
r = requests.post(
    "http://localhost:8000/molecular-docking/diffdock/generate",
    headers={"Content-Type": "application/json"},
    json=payload,
    timeout=300,
)
r.raise_for_status()
result = r.json()
```

## Save Ranked Poses

```python
from pathlib import Path

poses = result["ligand_positions"]
scores = result["position_confidence"]
if len(poses) != len(scores):
    raise ValueError("ligand_positions and position_confidence length mismatch")

out_dir = Path("diffdock_poses")
out_dir.mkdir(exist_ok=True)
for rank, (pose_sdf, score) in enumerate(zip(poses, scores), start=1):
    path = out_dir / f"pose_{rank:02d}_conf_{float(score):.3f}.sdf"
    path.write_text(pose_sdf)
    print(f"rank={rank} confidence={float(score):.4f} path={path}")
```

## Hosted SDF Request

```python
payload = {
    "protein": protein,
    "ligand": Path("ligand.sdf").read_text(),
    "ligand_file_type": "sdf",
    "num_poses": 20,
    "save_trajectory": False,
}
```
