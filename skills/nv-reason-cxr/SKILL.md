---
name: nv-reason-cxr
description: Used for command-shape or live NV-Reason-CXR chest X-ray reasoning smoke tests. Not for diagnosis or clinical reporting.
license: Apache-2.0
allowed-tools: Bash
metadata:
  author: NVIDIA MedTech Team
  tags:
    - MedTech
    - CXR
    - reasoning
---

# NV-Reason-CXR

## Purpose
- Used for command-shape or live NV-Reason-CXR chest X-ray reasoning smoke tests. Not for diagnosis or clinical reporting.
- Use the wrapper exactly as documented; do not replace the upstream entrypoint with a handwritten implementation.
- Manifest I/O: inputs are `chest_xray_image_or_fixture`; outputs are `result_json`.

## Instructions
- Read `skill_manifest.yaml` before changing arguments, side effects, or validation gates.
- Run `scripts/run_nv_reason_cxr.py` through the documented command below; keep outputs under a caller-provided run directory.
- If a host agent exposes `run_script`, use `run_script("scripts/run_nv_reason_cxr.py", args=[...])`; otherwise run the Bash/Python command shown below.
- Check the emitted JSON and paired verifier guidance before treating the run as evidence.

## Available Scripts
| Script | Purpose | Arguments |
|---|---|---|
| `scripts/run_nv_reason_cxr.py` | Primary entrypoint declared by skill_manifest.yaml. | `PATH_TO_CXR_OR_FIXTURE --out-dir OUT_DIR [--mock] [--check-setup]` |

## Prerequisites
- Runtime requirements: GPU/CUDA when declared by the manifest; Python packages listed in `runtime.side_effects.pip_packages`.
- Side effects: writes JSON outputs under the caller's `--out-dir`, may cache model assets under `~/.cache/huggingface/`, and may contact `https://huggingface.co` or `https://github.com` outside `--mock` mode.
- Run commands from the repository root unless an existing section below says otherwise.

## Limitations
- This is a thin wrapper. Image preprocessing, model inference, and decoding are delegated to Hugging Face Transformers and the NV-Reason-CXR-3B model.
- Output is not a diagnosis, clinical report, treatment recommendation, or triage decision. It is engineering evidence and must be reviewed by a qualified professional before any medical use.
- The model may hallucinate findings, miss subtle abnormalities, misread support devices, or produce overconfident prose.
- The committed fixture uses a generated synthetic PNG and deterministic mock response so CI can verify wrapper behavior without downloading model weights. Mock mode is not a substitute for model inference.
- Not for clinical deployment, clinical interpretation, autonomous diagnosis, treatment decisions.

## Troubleshooting
| Error | Cause | Fix |
|---|---|---|
| Missing dependency or import error | Runtime package drift from `skill_manifest.yaml`. | Install the packages declared in the manifest or use the documented setup command. |
| Empty or schema-invalid output | Wrong input path, unsupported modality, or upstream failure. | Re-run with a known fixture and inspect the wrapper JSON plus stderr. |
| Validation gate failure | Output violated a declared engineering invariant. | Keep the failed evidence pack and use the gate message to repair inputs or wrapper code. |

Runs NVIDIA-Medtech [`NV-Reason-CXR-3B`](https://github.com/NVIDIA-Medtech/NV-Reason-CXR)
for chest X-ray image interpretation through the documented Hugging Face
Transformers inference path. The wrapper does not reimplement the model,
image preprocessing, or decoding.


## Exact Runnable Surface

For command-shape smoke tests and JSON fixtures, use this repo-root wrapper path exactly:

```bash
python skills/nv-reason-cxr/scripts/run_nv_reason_cxr.py PATH_TO_CXR_OR_FIXTURE --mock --out-dir OUT_DIR
```

For live image inference, omit `--mock` only when the user asks for live model inference. Do not invent `Medical AI Skills run`, `eval_engine/run.py`, `infer.py`, or `python -m nv_reason_cxr` commands for ordinary user runs.

## Preconditions

Install the inference dependencies in the environment that will run the
skill:

```bash
pip install torch==2.7.1 torchvision==0.22.1 transformers==4.56.1 Pillow
```

The model weights are loaded from `nvidia/NV-Reason-CXR-3B` through
Transformers. They may download to the Hugging Face cache on first use.
Set `TRANSFORMERS_OFFLINE=1` or pass `--local-files-only` only after the
weights are already cached.

CUDA is expected for practical inference. CPU execution may work for small
tests but is slow and must be requested explicitly.

Check the local environment before downloading weights or running inference:

```bash
python skills/nv-reason-cxr/scripts/run_nv_reason_cxr.py --check-setup
```

The setup report checks importable dependencies, CUDA visibility, Hugging Face
cache state, and the recommended next step.

Operational environment variables:

| Variable | When to use |
|---|---|
| `MOCK_NV_REASON_CXR` | Set to `1` for deterministic command-shape smoke tests without model inference. |
| `NV_REASON_CXR_MODEL` | Override the Hugging Face model id only for compatibility probes. |
| `HF_HOME` | Point at a pre-populated Hugging Face cache. |
| `HF_TOKEN` | Authenticate model downloads when required by the local environment. |
| `TRANSFORMERS_OFFLINE` | Set to `1` only after weights are already cached. |
| `HF_HUB_OFFLINE` | Set to `1` only after Hugging Face assets are already cached. |

## License

The upstream repository code is Apache-2.0. The model weights are released
under the NVIDIA OneWay Noncommercial License Agreement. Users are responsible
for complying with the model-weight terms before live inference.

## Usage

From Medical AI Skills repo root:

```bash
python skills/nv-reason-cxr/scripts/run_nv_reason_cxr.py PATH_TO_CXR.png \
  --prompt "Find abnormalities and support devices." \
  --out-dir runs/nv_reason_cxr_case
```

Use the wrapper script directly for agent-generated commands. Do not replace
it with `eval_engine/run.py` unless the user explicitly asks to run the eval
harness. Do not redirect stdout with `>` in generated commands: callers and
the eval harness read the wrapper's stdout JSON, including
`output.response_text`, to verify the run. The direct runnable surface is:

```bash
python skills/nv-reason-cxr/scripts/run_nv_reason_cxr.py PATH_TO_CXR_OR_FIXTURE \
  --mock \
  --out-dir runs/nv_reason_cxr_case
```

`PATH_TO_CXR_OR_FIXTURE` may be a PNG/JPEG image or a JSON fixture. If the
user provides a JSON request such as
`runs/.../synthetic_cxr_input.json`, pass that exact JSON path as the first
argument. The script will load `generated://synthetic_chest_xray` fixtures,
create the temporary PNG under the output directory, and emit JSON with
`output.response_text`. Use `--mock` only for command-shape smoke tests or
fixtures that request mock mode; omit `--mock` for live model inference.

For JPEG input:

```bash
python skills/nv-reason-cxr/scripts/run_nv_reason_cxr.py PATH_TO_CXR.jpg \
  --prompt "Describe the chest X-ray findings." \
  --out-dir runs/nv_reason_cxr_case
```

Flags:

- `--model-id` — Hugging Face model id, default `nvidia/NV-Reason-CXR-3B`.
- `--device auto|cuda|cpu` — default `auto`, using CUDA when available.
- `--allow-cpu` — required for live CPU inference; CPU runs can be very slow.
- `--torch-dtype auto|float16|bfloat16|float32` — default `auto`, using
  bfloat16 on CUDA and float32 on CPU, matching the published BF16 model.
- `--max-new-tokens` — generation cap, default 2048.
- `--local-files-only` — use only locally cached Hugging Face assets.
- `--mock` — deterministic dry-run response for CI and wiring checks.
- `--prompt-preset findings|comprehensive|educational|structured` — optional
  known-good prompt presets from the model card/demo behavior.

The tested local live path uses:

- `AutoModelForImageTextToText.from_pretrained(..., dtype=torch.bfloat16).eval().to("cuda")`
- `AutoProcessor.from_pretrained(..., use_fast=True)`
- PNG/JPEG image input plus one text prompt
- `max_new_tokens=2048` by default

The script emits JSON on stdout and writes no clinical report files. It records
input image metadata, prompt, model id, runtime mode, response text, and known
limitations. If `runtime.truncated_by_max_new_tokens` is `true`, rerun with a
higher `--max-new-tokens` value.

## Fixture Smoke Test

The committed fixture uses a generated synthetic PNG and mock mode so the
eval harness can verify the wrapper without downloading weights:

```bash
python eval_engine/run.py skills/nv-reason-cxr \
  --fixture skills/nv-reason-cxr/fixtures/synthetic_cxr_input.json \
  --out runs/nv_reason_cxr_smoke
```

## Limits

This is research and engineering tooling only. It is not validated for
clinical diagnosis, treatment decisions, triage, patient-facing reporting, or
regulatory use. Model outputs can hallucinate, miss subtle findings, or
overstate uncertainty. A qualified professional must review any use in a
medical workflow.
