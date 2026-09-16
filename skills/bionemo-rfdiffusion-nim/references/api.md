# RFDiffusion NIM — API Reference

## Endpoints

| Mode | Method | URL |
|---|---|---|
| Hosted | POST | `https://health.api.nvidia.com/v1/biology/ipd/rfdiffusion/generate` |
| Local Docker | POST | `http://localhost:8000/biology/ipd/rfdiffusion/generate` |
| Health (local) | GET | `http://localhost:8000/v1/health/ready` |

**IMPORTANT**: Local path has no `/v1/` prefix; hosted uses `health.api.nvidia.com`.

---

## Request Body Schema

| Field | Type | Required | Default | Range | Notes |
|---|---|---|---|---|---|
| `contigs` | string | Yes | — | — | Design specification DSL (see Contigs Language below) |
| `input_pdb` | string | Conditional | — | — | PDB file content as inline string. Live hosted validation returns 422 if both `input_pdb` and `input_pdb_asset` are omitted. For de novo inline requests, pass a minimal dummy PDB (single ATOM record). |
| `input_pdb_asset` | string | Conditional | null | — | NVCF Asset ID for large PDB files. NVIDIA docs describe this as an alternative asset path; when using it, provide the original filename via `input_pdb`. |
| `hotspot_res` | array[string] | No | null | — | Residues the binder must contact; format: `["A50", "A51"]` |
| `diffusion_steps` | integer | No | 50 | 1–50 | Denoising iterations; more = better quality |
| `random_seed` | integer | No | null | — | Fixed seed for reproducibility |

**No string-typed numeric fields** — all numeric parameters use proper integer types.

---

## Response Schema

| Field | Type | Description |
|---|---|---|
| `output_pdb` | string | Generated protein backbone in PDB format |
| `elapsed_ms` | integer | Server-side processing time (milliseconds) |

---

## Contigs Language Reference

The `contigs` field is a DSL defining what to preserve and what to generate:

| Pattern | Meaning |
|---|---|
| `"100"` | Generate exactly 100 new residues |
| `"50-150"` | Generate 50–150 new residues (length sampled from range) |
| `"A20-60"` | Keep chain A residues 20–60 from `input_pdb` |
| `"A20-60/0 50-100"` | Keep A20–60, chain break, then generate 50–100 residues |
| `"A1-100/0 B1-100"` | Keep chains A and B from input (as a complex target) |

### Design mode examples

**De novo inline request (minimal dummy PDB required by live hosted validation):**
```json
{
  "input_pdb": "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\nATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
  "contigs": "80-120",
  "diffusion_steps": 50
}
```

**Motif scaffolding (preserve functional loop):**
```json
{
  "input_pdb": "<pdb content>",
  "contigs": "A25-35/0 40-80",
  "diffusion_steps": 50
}
```

**Binder design (design protein to bind target at hotspots):**
```json
{
  "input_pdb": "<target pdb content>",
  "contigs": "A1-100/0 50-100",
  "hotspot_res": ["A50", "A51", "A52", "A53"],
  "diffusion_steps": 50
}
```

---

## Docker Reference

The full preflight + `docker run` block lives in `SKILL.md` under **Local
Docker**; the flags below annotate it.

| Flag | Value | Notes |
|---|---|---|
| `--gpus` | `device=0` | Single GPU default; change only when the user asks |
| Cache mount | `/opt/nim/.cache` | ~15 GB model weights |
| Image | `nvcr.io/nim/ipd/rfdiffusion:2` | v2.3.0 as of 2025; `:2` is major version tag |
| No `--shm-size` | — | Not required for this NIM |

---

## Hardware Requirements

| Component | Requirement |
|---|---|
| Minimum GPU VRAM | 12 GB |
| Compute capability | >7.0 |
| CPU | 4 cores |
| RAM | 16 GB |
| Storage | 15 GB NVMe SSD |

Supported GPU configs with pre-compiled TensorRT: H100, A100, L40S, A10G, GB200.
