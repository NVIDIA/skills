# ProteinMPNN Examples

## Hosted Basic Design

```python
payload = {
    "input_pdb": pdb_content,
    "num_seq_per_target": 10,
    "sampling_temp": [0.1],
    "use_soluble_model": False,
    "ca_only": False,
}
```

## Redesign Chain A Only

```python
payload = {
    "input_pdb": Path("structure.pdb").read_text(),
    "input_pdb_chains": ["A"],
    "num_seq_per_target": 5,
    "sampling_temp": [0.2],
    "omit_AAs": ["C"],
}
```

## Diverse Soluble Designs

```python
payload = {
    "input_pdb": Path("protein.pdb").read_text(),
    "num_seq_per_target": 10,
    "sampling_temp": [0.1, 0.3, 0.5],
    "use_soluble_model": True,
    "omit_AAs": ["M"],
}
```

## Save Multi-FASTA and Report Scores

For common requests, use `scripts/design.py`; it performs the request, saves the
raw response and FASTA, and prints every designed sequence with its score and
artifact paths. For custom request code, apply the same result parser after a
successful response. Run this example from the skill directory and set
`expected_count` to the requested number of designs for that request:

```python
import json
from pathlib import Path
from scripts.design import design_results

summary = design_results(result, expected_count)
output = Path("design-run")  # choose a new directory for each run
output.mkdir(parents=True, exist_ok=False)
(output / "response.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
fasta_path = output / "designed_sequences.fa"
fasta_path.write_text(result["mfasta"])
summary["fasta_path"] = str(fasta_path.resolve())
(output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
print(json.dumps(summary, indent=2, allow_nan=False))
```

Keep the native/WT row in the saved raw FASTA, but exclude it from the design
count and score table. An array with one score per design maps to designed rows
only; an array that includes the native row must lose that row's score too.
Never silently truncate mismatched arrays with `zip`. If JSON scores are absent,
the helper can use each design's exact `score=` header field; it does not confuse
`global_score=` with `score=` or assign the native header score to a design.

Report the generated count, design identifiers, returned scores, sequence text
or labelled previews, and the saved paths. Report failure if the request or
result validation failed; example sequences are not execution evidence.
