---
name: genmol-nim
description: >
  Generate novel drug-like molecules using the GenMol NIM microservice. Use for de novo generation, scaffold decoration, motif extension, lead optimization, SAFE notation, QED or LogP ranking, hosted NVIDIA API calls, or local Docker deployment. GenMol takes SAFE notation in the smiles field, not ordinary SMILES.
license: Apache-2.0 AND CC-BY-4.0
compatibility: "safe-mol>=0.1.14; requests>=2.28"
allowed-tools: Bash, Read, Write, AskUserQuestion
---

# GenMol NIM

Generate drug-like molecules with GenMol. Use this guide for first-pass
hosted/local usage; load supplemental files only when needed:

- `references/api.md`: endpoints, schema, Docker flags, response fields.
- `references/science.md`: use cases, strengths, limits, and handoffs.
- `references/parameters.md`: SAFE patterns and tuning effects.
- `references/validation.md`: chemical and artifact checks.
- `references/examples.md`: compact request patterns.

## Choose Mode

Ask only when context is unclear:

> Hosted NVIDIA API or local Docker NIM?

- Hosted: `https://health.api.nvidia.com/v1/biology/nvidia/genmol/generate`
- Local: `http://localhost:8000/generate`

Hosted requests use `Authorization: Bearer $NGC_API_KEY`. Supported local Docker
startup uses `NGC_API_KEY` (or `NVIDIA_API_KEY` via the preflight) for
registry login, entitlement checks, and first-run model downloads; pass it
into the container with `-e NGC_API_KEY`. Local inference requests use no
auth header after readiness. Warm-cache key-free startup varies by
image/version and should not be assumed.

## Local Docker

Use shell env first; source repo-root `.env` only if present. Do not print keys.
For local setup answers, include this sequence: env preflight, `docker login`,
`docker run`, readiness loop, then a no-auth localhost request. Do not invent a
cache default or drop the `NVIDIA_API_KEY` fallback.

For the exact startup preflight (`.env` sourcing, `NVIDIA_API_KEY` fallback,
`--shm-size=2G`, both `--ulimit` flags, `docker login`, and the `docker run`
for `nvcr.io/nim/nvidia/genmol:1.0.1`), copy the command block in
[`references/api.md`](references/api.md) under **Docker run reference** verbatim.

GenMol is single-GPU; `NIM_TEST_GPU` defaults to `0`. Wait for readiness:

```bash
until curl -sf http://localhost:8000/v1/health/ready; do sleep 5; done
```

## SAFE Input

The API field is named `smiles`, but GenMol expects SAFE notation. Masked
positions use `[*{min-max}]`.

- De novo: `safe_input = "[*{20-30}]"`
- Scaffold decoration: `safe_input = scaffold_to_safe("C1CC(=O)NC1", 10, 15)`
- Motif extension: `safe_input = f"[*{{5-10}}].{motif_safe}.[*{{5-10}}]"`
- Lead optimization: encode the hit, then replace a fragment with `.[*{5-12}]`

Use `safe-mol` for conditioned generation. Simple ring scaffolds may raise
`SAFEFragmentationError`; fall back to the original SMILES plus a SAFE mask.

See the `scaffold_to_safe` helper in
[`references/examples.md`](references/examples.md) under **Scaffold Decoration**.

Wider masks increase diversity; tight masks keep analog size more predictable.

## Request Pattern

```python
import os
import requests

HOSTED = True
url = (
    "https://health.api.nvidia.com/v1/biology/nvidia/genmol/generate"
    if HOSTED else "http://localhost:8000/generate"
)
headers = {"Content-Type": "application/json"}
if HOSTED:
    headers["Authorization"] = f"Bearer {os.getenv('NGC_API_KEY')}"

payload = {
    "smiles": "[*{20-30}]",  # SAFE notation
    "num_molecules": 30,
    "temperature": "1.0",    # string, not float
    "noise": "1.0",          # string, not float
    "step_size": 1,
    "scoring": "QED",        # or "LogP"
    "unique": False,
}

response = requests.post(url, headers=headers, json=payload, timeout=180)
response.raise_for_status()
result = response.json()
```

Gotchas:

- `temperature` and `noise` are strings.
- `num_molecules` is 1-1000; invalid/duplicate molecules may be filtered, so
  request extra when the user needs a minimum count.
- `scoring` is `"QED"` for drug-likeness or `"LogP"` for lipophilicity.
- Set `unique=True` for deduplicated analog lists.

## Save And Report Output

Sort molecules by score, print the top ranks, and write a `.smi` file as shown
in [`references/examples.md`](references/examples.md) under **Save Ranked
Results**. For chemical validity, uniqueness, PAINS/alerts, and visualization
with RDKit, read `references/validation.md`.

## Limits And Troubleshooting

- Fewer molecules than requested is expected after filtering.
- Invalid SAFE strings cause `status: "failed"` or validation errors.
- Install `safe-mol` only for scaffold, motif, or lead-optimization workflows;
  de novo masks work without conversion.
- Local startup downloads about 20 GB into `LOCAL_NIM_CACHE`.
- Container issues: confirm `nvidia-smi`, NVIDIA Container Toolkit, and
  `--runtime=nvidia`; use `NIM_TEST_GPU` to choose the single visible GPU.
