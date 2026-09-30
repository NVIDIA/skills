# MolMIM Examples

## Hosted CMA-ES Generation

```python
import os
from pathlib import Path
import requests

url = "https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate"
headers = {
    "Authorization": f"Bearer {os.getenv('NGC_API_KEY')}",
    "Content-Type": "application/json",
}
payload = {
    "smi": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "algorithm": "CMA-ES",
    "num_molecules": 5,
    "property_name": "QED",
    "minimize": False,
    "min_similarity": 0.4,
    "particles": 8,
    "iterations": 3,
}
r = requests.post(url, headers=headers, json=payload, timeout=180)
r.raise_for_status()
result = r.json()
molecules = result.get("molecules")
if isinstance(molecules, str):
    import json
    molecules = json.loads(molecules)
generated = [item["sample"] for item in molecules or [] if isinstance(item, dict) and item.get("sample")]
if not generated and isinstance(result.get("generated"), list):
    generated = result["generated"]
Path("molmim_generated.smi").write_text("\n".join(generated) + "\n")
```

## Hosted Unguided Sampling Through Generate

```python
payload = {
    "smi": "CC(Cc1ccc(cc1)C(C(=O)O)C)C",
    "algorithm": "none",
    "num_molecules": 10,
    "particles": 20,
    "scaled_radius": 1.0,
}
```

## Local Embedding

```python
import requests

base = "http://localhost:8000"
payload = {"sequences": ["CC(Cc1ccc(cc1)C(C(=O)O)C)C"]}
r = requests.post(f"{base}/embedding", json=payload, timeout=60)
r.raise_for_status()
print(len(r.json()["embeddings"]))
```

## Local Hidden To Decode

```python
base = "http://localhost:8000"
seed_payload = {"sequences": ["CC(Cc1ccc(cc1)C(C(=O)O)C)C"]}

hidden = requests.post(f"{base}/hidden", json=seed_payload, timeout=60)
hidden.raise_for_status()

decoded = requests.post(f"{base}/decode", json=hidden.json(), timeout=60)
decoded.raise_for_status()
print(decoded.json()["generated"])
```

## Local Sampling

```python
payload = {
    "sequences": ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
    "beam_size": 1,
    "num_molecules": 5,
    "scaled_radius": 0.7,
}
r = requests.post("http://localhost:8000/sampling", json=payload, timeout=60)
r.raise_for_status()
print(r.json()["generated"])
```
