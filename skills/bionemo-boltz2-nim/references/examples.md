# Boltz2 Examples

These examples are compact patterns to adapt inside answers. Use the exact
endpoints and auth rules from `api.md`.

## Hosted Protein Structure

```python
payload = {
    "polymers": [{
        "id": "A",
        "molecule_type": "protein",
        "sequence": sequence,
    }],
    "recycling_steps": 3,
    "sampling_steps": 50,
    "diffusion_samples": 1,
    "step_scale": 1.638,
    "output_format": "mmcif",
}
```

## Protein-Ligand Affinity

```python
payload = {
    "polymers": [{
        "id": "A",
        "molecule_type": "protein",
        "sequence": protein_sequence,
    }],
    "ligands": [{
        "id": "L1",
        "smiles": ligand_smiles,
        "predict_affinity": True,
    }],
    "sampling_steps_affinity": 200,
    "diffusion_samples_affinity": 5,
    "output_format": "mmcif",
}
```

## Protein Plus DNA With A3M

```python
payload = {
    "polymers": [
        {
            "id": "A",
            "molecule_type": "protein",
            "sequence": protein_sequence,
            "msa": {
                "msa_search": {
                    "a3m": {
                        "alignment": f">query\n{protein_sequence}",
                        "format": "a3m",
                        "rank": 0,
                    }
                }
            },
        },
        {
            "id": "B",
            "molecule_type": "dna",
            "sequence": dna_sequence,
        },
    ],
    "output_format": "mmcif",
}
```

## Save Structures And Affinity

```python
for i, structure in enumerate(result["structures"], start=1):
    path = f"boltz2_structure_{i}.cif"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(structure["structure"])

for i, score in enumerate(result.get("confidence_scores", []), start=1):
    print(f"structure {i} confidence_score={score}")

for ligand_id, affinity in result.get("affinities", {}).items():
    print(ligand_id, affinity["affinity_pic50"][0])
    print(ligand_id, affinity["affinity_pred_value"][0])
    print(ligand_id, affinity["affinity_probability_binary"][0])
```
