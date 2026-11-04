# OpenFold3 Examples

## Hosted Protein Smoke Test

```python
import os
import requests

seq = "MKTVRQERLKSIVR"
payload = {
    "inputs": [{
        "input_id": "protein_smoke",
        "output_format": "pdb",
        "molecules": [{
            "type": "protein",
            "id": "A",
            "sequence": seq,
            "diffusion_samples": 1,
            "msa": {"main": {"a3m": {"alignment": f">query\n{seq}", "format": "a3m"}}}
        }]
    }]
}

response = requests.post(
    "https://health.api.nvidia.com/v1/biology/openfold/openfold3/predict",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('NGC_API_KEY')}",
    },
    json=payload,
    timeout=300,
)
response.raise_for_status()
result = response.json()
```

## Local Protein Smoke Test

```python
response = requests.post(
    "http://localhost:8000/biology/openfold/openfold3/predict",
    headers={"Content-Type": "application/json"},
    json=payload,
    timeout=300,
)
response.raise_for_status()
```

## Protein Plus CCD Ligand

```python
payload["inputs"][0]["molecules"] = [
    {
        "type": "protein",
        "id": "A",
        "sequence": seq,
        "msa": {"main": {"a3m": {"alignment": f">query\n{seq}", "format": "a3m"}}},
    },
    {"type": "ligand", "id": "L", "ccd_codes": "ATP"},
]
```

## Protein Plus SMILES Ligand

```python
payload["inputs"][0]["molecules"] = [
    {
        "type": "protein",
        "id": "A",
        "sequence": seq,
        "msa": {"main": {"a3m": {"alignment": f">query\n{seq}", "format": "a3m"}}},
    },
    {"type": "ligand", "id": "L", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
]
```

## Protein-DNA Complex

```python
payload["inputs"][0]["molecules"] = [
    {
        "type": "protein",
        "id": "A",
        "sequence": seq,
        "msa": {"main": {"a3m": {"alignment": f">query\n{seq}", "format": "a3m"}}},
    },
    {"type": "dna", "id": "B", "sequence": "ATCGATCG"},
    {"type": "dna", "id": "C", "sequence": "CGATCGAT"},
]
```

## Save Structures

```python
for idx, sample in enumerate(result["outputs"][0]["structures_with_scores"], start=1):
    path = f"openfold3_structure_{idx}.{sample['format']}"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(sample["structure"])
```
