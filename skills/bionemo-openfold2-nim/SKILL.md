---
name: openfold2-nim
description: >
  Use this skill for OpenFold2, NVIDIA's BioNeMo NIM microservice for monomer protein structure prediction. Invoke whenever the user mentions OpenFold2, AlphaFold2-like monomer folding, protein sequence-to-structure prediction, A3M MSAs, mmCIF templates, hosted NVIDIA API calls, or local Docker deployment.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "requests>=2.28"
allowed-tools: Bash, Read, Write, AskUserQuestion
---

# OpenFold2 NIM

Predict a single protein-chain structure from an amino-acid sequence, with
optional A3M multiple sequence alignments and mmCIF templates. Use this
guide for basic hosted/local NIM use; load supplemental files only when
the task needs deeper context:

- `references/api.md`: exact endpoints, schemas, Docker flags, response fields.
- `references/science.md`: model scope, strengths, limitations, and handoffs.
- `references/parameters.md`: MSA, template, model-selection, and relax effects.
- `references/validation.md`: artifact and scientific sanity checks.
- `references/examples.md`: compact hosted/local payload patterns.

## Choose Mode

Ask only when context is unclear:

> Hosted NVIDIA API or local Docker NIM?

- Hosted URL: `https://health.api.nvidia.com/v1/biology/openfold/openfold2/predict-structure-from-msa-and-template`
- Local URL: `http://localhost:8000/biology/openfold/openfold2/predict-structure-from-msa-and-template`
- Local readiness: `http://localhost:8000/v1/health/ready`

Mode difference: hosted and local use the same prediction path except local
does not include `/v1/`. Hosted requests use `Authorization: Bearer
$NGC_API_KEY`; local inference requests use no auth header after readiness.

## Auth And Environment

Do not print API keys. Confirm they exist with shell tests, not echoes.

Hosted needs `NGC_API_KEY` in the request header. Supported local Docker
startup uses `NGC_API_KEY`, or `NVIDIA_API_KEY` as a fallback, plus
`LOCAL_NIM_CACHE`. A repo-root `.env` file may be sourced as a local override.

## Local Docker

Use the official OpenFold2 NIM image and mount `LOCAL_NIM_CACHE` at
`/opt/nim/.cache`. Current docs recommend at least 80 GB disk, 64 GB system
RAM, 8 CPU cores, and one supported GPU; the container is roughly 55 GB and
first startup downloads about 10 GB of model parameters.

For the exact startup preflight (`.env` sourcing, `NGC_API_KEY`/`NVIDIA_API_KEY`
handling, `docker login`, and the `docker run` for
`nvcr.io/nim/openfold/openfold2:latest`), copy the command block in
[`references/api.md`](references/api.md) under **Local Docker** verbatim — do
not drop `.env`, `NGC_API_KEY`, `LOCAL_NIM_CACHE`, or the no-auth local request.

Readiness check:

```bash
until curl -sf http://localhost:8000/v1/health/ready; do sleep 5; done
```

## Request Pattern

Use Python `requests`; curl escaping is fragile for A3M/mmCIF text. The
`sequence` field is required. `input_id`, `alignments`, `selected_models`,
`relax_prediction`, `use_templates`, and `explicit_templates` are optional.

```python
import os
import requests

hosted = True
url = (
    "https://health.api.nvidia.com/v1/biology/openfold/openfold2/predict-structure-from-msa-and-template"
    if hosted
    else "http://localhost:8000/biology/openfold/openfold2/predict-structure-from-msa-and-template"
)
headers = {"Content-Type": "application/json"}
if hosted:
    headers["Authorization"] = f"Bearer {os.getenv('NGC_API_KEY')}"

seq = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPT"
payload = {
    "sequence": seq,
    "input_id": "kras_fragment",
    "selected_models": [1],
    "relax_prediction": False,
    "alignments": {
        "uniref90": {
            "a3m": {
                "alignment": f">query\n{seq}",
                "format": "a3m",
            }
        }
    },
}

response = requests.post(url, headers=headers, json=payload, timeout=300)
response.raise_for_status()
result = response.json()
```

Payload gotchas:

- OpenFold2 is monomer-only. For protein-ligand, protein-DNA/RNA, or
  multi-chain complexes, use OpenFold3 or Boltz2 instead.
- `sequence` must use valid amino-acid IUPAC symbols.
- Hosted API docs list sequence length 1-1000; local docs say current NIM
  supports sequences up to 2048 residues on supported hardware.
- A3M alignments go under `alignments` by database name, then `a3m` with
  `alignment` and `format`. When the user needs to create or deepen an MSA,
  hand off to `msa-search-nim` / MSA Search and map its A3M output into this
  `alignments` shape.
- Starting with OpenFold2 2.0.0, use `explicit_templates` with mmCIF content;
  do not write new HHR-template examples.
- `selected_models` chooses AlphaFold2/OpenFold parameter sets 1-5. Select one
  or two models for smoke tests; use all five for stronger production runs.

## Save And Interpret Output

The response includes one prediction per selected model, ordered by confidence.
Save every returned structure-like text field and the full JSON response so
field-shape differences are auditable. Production answers should explicitly
write `.pdb` or `.cif` artifacts, preserve the response JSON, and print any
confidence/ranking fields the service returns.

```python
from pathlib import Path
import json

Path("openfold2_response.json").write_text(json.dumps(result, indent=2))

def save_strings(obj, prefix="openfold2"):
    i = 0
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and ("ATOM" in value or value.lstrip().startswith("data_")):
                i += 1
                ext = "cif" if value.lstrip().startswith("data_") else "pdb"
                Path(f"{prefix}_{key}_{i}.{ext}").write_text(value)
            elif isinstance(value, (dict, list)):
                i += save_strings(value, f"{prefix}_{key}")
    elif isinstance(obj, list):
        for idx, value in enumerate(obj, start=1):
            if isinstance(value, (dict, list)):
                i += save_strings(value, f"{prefix}_{idx}")
    return i

saved = save_strings(result)
print(f"saved {saved} structure artifact(s)")
```

For production monomer runs:

- Use `selected_models: [1, 2, 3, 4, 5]` unless the user requests a smoke test.
- Use `relax_prediction: True` in Python payloads when relaxation is desired;
  JSON examples may show `true`.
- State the sequence length caveat: hosted API docs list 1-1000 residues, while
  local support-matrix docs list up to 2048 residues on supported hardware.
- If the task is a complex rather than a monomer, redirect to OpenFold3 or
  Boltz2.

Treat tiny toy sequences and single-sequence MSAs as API smoke tests, not
quality evidence. For scientific interpretation and validation, read
`references/science.md` and `references/validation.md`.

## Troubleshooting

- `401`: missing, expired, or unauthorized NGC API key.
- `422`: invalid amino-acid characters, sequence too long, malformed A3M, bad
  `selected_models`, or malformed mmCIF template object.
- Local `404`: remove `/v1/` from the prediction URL.
- Weak structures: use MSA Search to generate deeper A3M alignments and add
  biologically relevant mmCIF templates when appropriate.
- Local startup stalls: first run downloads parameters into `LOCAL_NIM_CACHE`.
