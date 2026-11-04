# RFDiffusion Examples

## Hosted De Novo

```python
DUMMY_PDB = (
    "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    "END\n"
)
payload = {
    "input_pdb": DUMMY_PDB,
    "contigs": "80-120",
    "diffusion_steps": 50,
}
```

## Motif Scaffolding

```python
payload = {
    "input_pdb": Path("target.pdb").read_text(),
    "contigs": "A25-35/0 50-80",
    "diffusion_steps": 50,
}
```

## Binder Design

```python
payload = {
    "input_pdb": Path("target.pdb").read_text(),
    "contigs": "A1-100/0 50-100",
    "hotspot_res": ["A50", "A51", "A52", "A53", "A54"],
    "diffusion_steps": 50,
}
```

## Save PDB

```python
with open("rfdiffusion_backbone.pdb", "w", encoding="utf-8") as handle:
    handle.write(result["output_pdb"])
```
