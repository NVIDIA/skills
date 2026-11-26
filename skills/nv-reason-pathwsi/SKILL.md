---
name: nv-reason-pathwsi
description: Run NV-Reason-PathWSI inference on user-provided PNG/JPEG pathology patches or tiles for engineering and research workflows. Not for diagnosis, treatment, clinical reporting, or whole-slide files.
license: Apache-2.0
allowed-tools: Bash, Read, Write, Env
metadata:
  author: "NVIDIA MedTech <noreply@nvidia.com>"
  tags:
    - medtech
    - pathology
    - reasoning
---

# NV-Reason-PathWSI

## Purpose

Runs the documented NV-Reason-PathWSI single-image CLI (`inference.py`) on one
user-provided PNG/JPEG pathology patch or whole-slide image tile and a text
prompt, then records the input hash, response text, runtime mode, dependency
versions, and upstream logs as JSON. A mock mode checks wiring without loading
weights.
Inputs are `pathology_image_or_request`; the JSON output is `result_json`.
Not for diagnosis, treatment, triage, patient-facing output, or clinical
reporting.

## Instructions

Read `skill_manifest.yaml`, then run `scripts/run_nv_reason_pathwsi.py`.
Agents with a script helper can use
`run_script("scripts/run_nv_reason_pathwsi.py", args=[...])`.
Run from the repository root and choose a fresh `--out-dir` for each run.
Do not call upstream `inference.py` directly or write Transformers code for
normal runs; the wrapper owns input checks, offline caches, and provenance.
Stdout is JSON; keep it available to the caller. Inspect `runtime.mode` before
reporting success: `mock` proves wiring only; `upstream_cli` records upstream
execution. A live failure never falls back to mock.

```bash
# Read-only setup check; no model load or network access. Go live only when
# ready_for_live is true, and relay any warnings with the results.
python skills/nv-reason-pathwsi/scripts/run_nv_reason_pathwsi.py --check-setup \
  --upstream-dir PATH_TO_UPSTREAM --model PATH_TO_LOCAL_MODEL

# Offline synthetic fixture, with no model inference.
python skills/nv-reason-pathwsi/scripts/run_nv_reason_pathwsi.py \
  skills/nv-reason-pathwsi/fixtures/synthetic_pathology_input.json \
  --mock --out-dir runs/pathwsi_mock

# Live inference with a local upstream checkout and model snapshot.
python skills/nv-reason-pathwsi/scripts/run_nv_reason_pathwsi.py PATH_TO_TILE.png \
  --live --upstream-dir PATH_TO_UPSTREAM --model PATH_TO_LOCAL_MODEL \
  --prompt "Describe the histopathological features visible in this image." \
  --out-dir runs/pathwsi_live
```

The harness template is `${python} ${script} ${fixture} --out-dir ${out}/artifacts`.
The committed fixture explicitly requests mock mode. Use a real image or a
live request JSON when collecting model evidence; `--live` rejects mock requests.

Tool scope: `Bash` runs the committed wrapper and documented setup commands;
`Read` covers skill files, caller-selected inputs, the upstream checkout, and
model assets; `Write` covers the caller-selected `--out-dir` and its per-run
caches. `Env` is SkillSpector's environment-capability label for the variables
listed under Prerequisites, accessed through shell/Python. The wrapper makes no
network calls; Hugging Face downloads happen only in the upstream child after
`--allow-downloads`. These declarations do not authorize unrelated file access,
printing authentication values, or changing an existing development environment.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/run_nv_reason_pathwsi.py` | Call the upstream single-image CLI and emit JSON, or check setup. | `[IMAGE_OR_REQUEST] --out-dir DIR [--upstream-dir DIR] [--model ID_OR_DIR] [--prompt TEXT] [--live or --mock] [--think] [--max-new-tokens N] [--device-map auto or cuda or cpu] [--timeout SECONDS] [--hf-home DIR] [--allow-downloads] [--check-setup]` |

## Prerequisites

Python 3.11+ with Pillow for input checks. Live runs need an upstream checkout
containing `inference.py`, the model weights, and a suitable CUDA GPU
(upstream recommends at least 24 GB). The wrapper installs nothing, pins no
versions, and edits no upstream files. Use a separate environment.

The wrapper uses only upstream's Transformers path, so vLLM is not needed. The
processor still needs Torchvision on the single-image path. Install the tested
environment and run its Python without activating it:

```bash
python3 -m venv runs/pathwsi_env
runs/pathwsi_env/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 \
  transformers==5.8.1 accelerate==1.13.0 Pillow
```

Tested with Python 3.11 and 3.12 and Pillow 11.3 and 13.0 on one RTX 6000 Ada
(48 GB). Other versions may work; `--check-setup` reports installed versions,
and every result records them.

Clone the [upstream repository](https://github.com/NVIDIA-Medtech/NV-Reason-PathWSI)
and obtain the [model](https://huggingface.co/nvidia/NV-Reason-PathWSI), accepting
its license terms where required. A local snapshot needs configuration,
tokenizer, processor, chat template, and weights. No `trust_remote_code` is
required by upstream.

Upstream provides `examples/sample_1.png`, `sample_2.png`, `sample_3.png`, and
`examples/queries.jsonl`. The PNGs are Git LFS assets. With Git LFS already
installed, fetch them using `git -C PATH_TO_UPSTREAM lfs pull --include='examples/*'`.
Pass a sample's actual PNG path to the live command above. Do not treat the
small pointer file as an image or copy upstream examples into this skill.

| Variable | Behavior |
|---|---|
| `NV_REASON_PATHWSI_ROOT` | Default upstream checkout when `--upstream-dir` is omitted. |
| `NV_REASON_PATHWSI_MODEL` | Default model ID or local snapshot directory when `--model` is omitted; also selects local weights for harness runs. |
| `HF_TOKEN` | Forwarded to the upstream child only with `--allow-downloads` for gated model downloads; never put its value in arguments or logs. |
| `HF_HOME` | Parent-process cache setting; use `--hf-home` to explicitly select an existing cache for the child. |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | Respected by the child; both are set to `1` unless `--allow-downloads` is supplied. |

Downloads are disabled by default. For a model ID with an existing cache,
pass `--hf-home PATH_TO_CACHE`. To download the model, add
`--allow-downloads`; inherited offline settings still apply. Other caches
stay under `--out-dir/cache`, and the parent environment is unchanged.
The child receives an explicit set of OS, CUDA, thread, and offline settings.
Hugging Face credentials and proxy/TLS settings are forwarded only with
`--allow-downloads`. Unrelated environment variables and credentials are not
inherited by the upstream process.
Local writes are limited to `<caller-provided --out-dir>` and
`<explicit --hf-home, if supplied>`. Downloads may contact
`https://huggingface.co`, `https://*.huggingface.co`, and
`https://*.xethub.hf.co`; local-only runs require no network.

## Inputs and results

Direct input is a `.png`, `.jpg`, or `.jpeg`. A JSON request contains `image`,
optional `prompt`, optional `case_id`, and a boolean `mock` (default `false`).
Relative image paths resolve from the JSON file. `--prompt` overrides its
prompt. The generated fixture URI is allowed only in mock mode.

The wrapper writes `result.json` and `response.txt`; live runs also write
`upstream.stdout.txt` and `upstream.stderr.log`. Stdout repeats `result.json`.
Image and response SHA-256 hashes bind the artifacts to the result. Model
loading, preprocessing, greedy decoding, and thinking mode remain upstream.
Defaults match the single-image CLI: thinking off and 2048 new tokens.
Use `--think` or change `--max-new-tokens` explicitly when needed.

## Limitations

- Research and engineering use only. Model output can hallucinate, omit
  findings, or present unreliable reasoning; it is not validated for medical
  correctness, diagnosis, treatment, clinical reporting, or regulatory use.
- Accepts image patches/tiles; it does not read SVS/TIFF slides, tile a slide,
  or aggregate tile outputs into slide-level conclusions. vLLM batching is
  outside this wrapper's scope.
- The synthetic RGB fixture contains geometry, not tissue. Mock tests do not
  establish that weights load or that model inference succeeds.
- The CLI returns text without token counts or stop reasons. Truncation stays
  `unknown`; a successful exit does not establish a complete answer.
- Upstream `inference.py` stops only at the model package's generation EOS
  ids. When those omit the chat end-of-turn token, text continues past the
  first answer, up to `--max-new-tokens`. `--check-setup` warns about this;
  the wrapper returns the text unchanged.
- The wrapper records the upstream commit and entrypoint hash but does not
  authenticate caller-supplied weights. Use reviewed, immutable upstream and
  model revisions for evidence runs.
- Wrapper code is Apache-2.0; upstream code and weights have separate
  OpenMDW-1.1 terms. Review the upstream license when obtaining those assets.

## Troubleshooting

| Error | Action |
|---|---|
| Git LFS pointer | Fetch upstream `examples/*` with Git LFS, then retry with a fresh output directory. |
| Unsupported slide or unreadable image | Provide a readable PNG/JPEG patch or tile. |
| Missing package or import error | Install the tested environment above in an isolated environment, run that Python, and inspect `upstream.stderr.log`. |
| Missing model / access denied | Supply a complete local snapshot or an existing cache; enable downloads only if intended. |
| CUDA failure or timeout | Inspect `upstream.stderr.log`, GPU availability, and `--timeout`; preserve the failed run. |
| Empty response | Treat as failure even if upstream exited zero. Inspect the saved stderr. |
| Text continues after the first answer | Upstream did not stop at the end-of-turn token; see `--check-setup` warnings. Keep the raw response, flag later text as possible continuation, and lower `--max-new-tokens` for short-answer prompts. Do not patch upstream or trim output. |
| `fast path is not available` in stderr | Optional linear-attention kernels are absent; upstream falls back to PyTorch. No action is needed. |
| Output artifacts already exist | Choose a fresh `--out-dir`; prior evidence is preserved. |
