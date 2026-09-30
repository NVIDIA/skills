# GenMol Examples

Use these compact patterns when the activated skill needs a little more detail.

## Hosted De Novo Generation

```python
payload = {
    "smiles": "[*{20-30}]",
    "num_molecules": 30,
    "temperature": "1.0",
    "noise": "1.0",
    "step_size": 1,
    "scoring": "QED",
    "unique": False,
}
```

## Scaffold Decoration

```python
import safe as sf

def scaffold_to_safe(smiles: str, frag_min: int, frag_max: int) -> str:
    try:
        safe_str = sf.encode(smiles)
    except sf.SAFEFragmentationError:
        safe_str = smiles
    return f"{safe_str}.[*{{{frag_min}-{frag_max}}}]"

safe_input = scaffold_to_safe("C1CC(=O)NC1", 10, 15)
```

## Lead Optimization Ranked By LogP

```python
hit_smiles = "CC1=CC=C(C=C1)NC(=O)C"
safe_hit = sf.encode(hit_smiles)
safe_input = safe_hit.rsplit(".", 1)[0] + ".[*{5-12}]"

payload = {
    "smiles": safe_input,
    "num_molecules": 30,
    "temperature": "1.0",
    "noise": "1.0",
    "scoring": "LogP",
    "unique": True,
}
```

## Save Ranked Results

```python
if result["status"] != "success":
    raise RuntimeError(result.get("error", "GenMol failed"))

molecules = sorted(result["molecules"], key=lambda mol: mol["score"], reverse=True)
with open("generated_molecules.smi", "w", encoding="utf-8") as handle:
    handle.write("smiles\tscore\n")
    for molecule in molecules:
        handle.write(f"{molecule['smiles']}\t{molecule['score']:.4f}\n")
```
