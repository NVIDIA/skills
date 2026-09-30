---
name: molmim-nim
description: >
  Use this skill for MolMIM, NVIDIA's BioNeMo NIM microservice for small-molecule latent-space generation and optimization. Invoke for MolMIM, molecular embeddings, hidden states, latent decoding, sampling around a seed SMILES, CMA-ES guided molecule generation, QED or plogP optimization, hosted NVIDIA API calls, or local Docker deployment.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "requests>=2.28; rdkit"
allowed-tools: Bash, Read, Write, AskUserQuestion
---

# MolMIM NIM

Generate, sample, embed, and decode small molecules with MolMIM. Use this
guide for first-pass hosted/local usage; load supplemental files only when
needed:

- `references/api.md`: endpoints, schema, Docker flags, response fields.
- `references/science.md`: use cases, strengths, limits, and handoffs.
- `references/parameters.md`: generation, sampling, and optimization effects.
- `references/validation.md`: SMILES/property/artifact checks.
- `references/examples.md`: compact hosted/local request patterns.

## Choose Mode

Ask only when context is unclear:

> Hosted NVIDIA API or local Docker NIM?

See [`references/api.md`](references/api.md) under **Endpoints** for the full
hosted/local endpoint list.

Mode difference: the hosted API reference exposes `/generate`; the local
container exposes the broader latent-space workflow (`/embedding`, `/hidden`,
`/decode`, `/sampling`, `/generate`). Do not invent hosted latent endpoints.

Hosted requests use `Authorization: Bearer $NGC_API_KEY`. Local inference uses
no auth header after readiness.

## Local Docker

Use shell env first; source repo-root `.env` only if present. Do not print keys.
MolMIM docs use `NGC_CLI_API_KEY` for the local container; this repo accepts
`NGC_API_KEY` or `NVIDIA_API_KEY` and maps to `NGC_CLI_API_KEY` for startup.
Mount `LOCAL_NIM_CACHE` at `/home/nvs/.cache/nim`.

For the exact startup preflight (the `NGC_API_KEY`/`NVIDIA_API_KEY` →
`NGC_CLI_API_KEY` mapping, `docker login`, and the `docker run` for
`nvcr.io/nim/nvidia/molmim:1.0.0`), copy the command block in
[`references/api.md`](references/api.md) under **Local Docker** verbatim.

Readiness check:

```bash
until curl -sf http://localhost:8000/v1/health/ready; do sleep 5; done
```

Local embedding smoke test after readiness. Local inference uses no
`Authorization` header:

```python
import requests

seed = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
response = requests.post(
    "http://localhost:8000/embedding",
    headers={"Content-Type": "application/json"},
    json={"sequences": [seed]},
    timeout=60,
)
response.raise_for_status()
embedding_data = response.json()
embeddings = embedding_data["embeddings"]
print(f"received {len(embeddings)} embedding vector(s)")
```

## Hosted Generation Pattern

Use hosted `/generate` for seed-SMILES generation or optimization. Use
`algorithm: "CMA-ES"` for guided property optimization and `algorithm: "none"`
for unguided sampling around the seed.

```python
import os
import requests

hosted = True
url = (
    "https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate"
    if hosted else "http://localhost:8000/generate"
)
headers = {"Content-Type": "application/json"}
if hosted:
    headers["Authorization"] = f"Bearer {os.getenv('NGC_API_KEY')}"

payload = {
    "smi": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "algorithm": "CMA-ES",
    "num_molecules": 10,
    "property_name": "QED",
    "minimize": False,
    "min_similarity": 0.4,
    "particles": 8,
    "iterations": 3,
}

response = requests.post(url, headers=headers, json=payload, timeout=180)
response.raise_for_status()
result = response.json()
```

Generation gotchas:

- Field name is `smi`, not `smiles`.
- `algorithm` is `"CMA-ES"` or `"none"`.
- `property_name` is `"QED"` or `"plogP"`.
- `num_molecules` is 1-100. `iterations` is 1-1000. `particles` is 2-1000.
- `min_similarity` is 0-1 in the hosted API reference; local docs emphasize
  common values up to 0.7 for constrained optimization.
- `scaled_radius` is 0-2 and is mainly used with `algorithm: "none"` or local
  `/sampling`.

## Local Latent Workflow

Use local-only endpoints for embedding, hidden-state manipulation, and decode.
This is also the surface used by the guided optimization example package.
For local latent workflows, state explicitly that the hosted API reference
exposes `/generate`; `/embedding`, `/hidden`, `/decode`, and `/sampling` are
local-only in the current docs.

```python
seed = "CC(Cc1ccc(cc1)C(C(=O)O)C)C"
base = "http://localhost:8000"
headers = {"Content-Type": "application/json"}

embedding = requests.post(
    f"{base}/embedding",
    headers=headers,
    json={"sequences": [seed]},
    timeout=60,
)
embedding.raise_for_status()
embedding_data = embedding.json()
embeddings = embedding_data["embeddings"]
print(f"received {len(embeddings)} embedding vector(s)")

hidden = requests.post(
    f"{base}/hidden",
    headers=headers,
    json={"sequences": [seed]},
    timeout=60,
)
hidden.raise_for_status()
hidden_data = hidden.json()
hiddens = hidden_data["hiddens"]
mask = hidden_data["mask"]

decoded = requests.post(
    f"{base}/decode",
    headers=headers,
    json={"hiddens": hiddens, "mask": mask},
    timeout=60,
)
decoded.raise_for_status()

sampled = requests.post(
    f"{base}/sampling",
    headers=headers,
    json={"sequences": [seed], "num_molecules": 10, "scaled_radius": 0.7},
    timeout=60,
)
sampled.raise_for_status()
```

## Save And Validate Output

Save generated SMILES and validate before using them downstream.

```python
from pathlib import Path
import json

def molmim_smiles(result):
    values = []
    if isinstance(result.get("generated"), list):
        for item in result["generated"]:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, list):
                values.extend(x for x in item if isinstance(x, str))
    molecules = result.get("molecules")
    if isinstance(molecules, str):
        molecules = json.loads(molecules)
    if isinstance(molecules, list):
        for item in molecules:
            if isinstance(item, dict) and isinstance(item.get("sample"), str):
                values.append(item["sample"])
    return values

generated = molmim_smiles(result)
if not generated:
    raise RuntimeError(f"MolMIM returned no generated molecules: {result}")

Path("molmim_response.json").write_text(json.dumps(result, indent=2))
Path("molmim_generated.smi").write_text("\n".join(generated) + "\n")
for i, smiles in enumerate(generated, start=1):
    print(i, smiles)
```

Use RDKit when available to check parseability, uniqueness, simple property
ranges, and whether seed similarity constraints are plausible. Generated
molecules are candidates, not validated hits; use downstream property, docking,
affinity, toxicity, and synthetic-feasibility checks before prioritization.

## Troubleshooting

- Hosted `404` on `/embedding`, `/hidden`, `/decode`, or `/sampling`: those
  endpoints are local-only in the docs.
- `401`: missing or unauthorized NGC key for hosted requests.
- Hosted response parsing: live hosted `/generate` may return `molecules` as
  a JSON string of `{sample, score}` objects, while local endpoints may return
  `generated`; parse both.
- `422`: invalid SMILES, unsupported `algorithm`, invalid `property_name`, or
  parameter outside documented ranges.
- Local startup auth: set `NGC_CLI_API_KEY`, or set `NGC_API_KEY`/`NVIDIA_API_KEY`
  and map it as shown above.
- Local startup cache misses: mount `LOCAL_NIM_CACHE` to `/home/nvs/.cache/nim`,
  not `/opt/nim/.cache`.
