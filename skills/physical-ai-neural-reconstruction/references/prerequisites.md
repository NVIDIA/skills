# Prerequisites for the sibling skills

The router itself has no runtime prerequisites beyond `git` for
fetching the upstream. Everything below belongs to the downstream
sibling skills; each one's own Prerequisites section is authoritative.

- **Linux x86_64** — aarch64 is not supported by `nre`.
- **NVIDIA GPU + driver** — CUDA 12.8 capability and >= 24 GB VRAM
  (48 GB+ recommended). Ampere (A100/A10/A40/RTX A6000), Ada
  (L20/L40/L40S), Hopper (H100/H20): R550+ required, R570+
  recommended. Blackwell (RTX Pro 6000D): R580+.
  `asset-harvester` needs driver >= 570 and ~16 GB VRAM.
- **Docker >= 23.0.1 + NVIDIA Container Toolkit >= 1.13.5** — for the
  `nre`, `nre-tools`, and `nurec-fixer` containers
  (`nvcr.io/nvidia/nre/nre-ga:latest`,
  `nvcr.io/nvidia/nre/nre-tools-ga:latest`, and the locally-built
  `harmonizer-cosmos-env` image layered on
  `nvcr.io/nvidia/pytorch:25.10-py3`).
- **NGC API key** — for pulling `nvcr.io` containers. Resolution
  order is `$NGC_CLI_API_KEY` first, then `$NGC_API_KEY`, and only
  then prompt the user (see `nre`'s NGC and registry reference).
- **Hugging Face token** (`HF_TOKEN`) with the gated licenses
  **accepted in advance** on Hugging Face: `nvidia/PhysicalAI-*`
  datasets, `nvidia/Harmonizer`, and
  `nvidia/Cosmos-Predict2-0.6B-Text2Image`. The
  `nvidia/asset-harvester` checkpoints themselves are public; its
  optional DINOv3, Llama Guard and SAM 3D Body models are gated.
- **Python 3.10+** with `huggingface_hub` installed;
  `pip install nvidia-ncore` for `ncore`; conda (Miniconda /
  Miniforge) for `asset-harvester`; it needs a GCC that `nvcc` accepts
  (10–13 is the tested range, but `setup.sh` selects its own compiler).
- **(Optional)** CARLA, Isaac Sim 5.1, or AlpaSim for simulator
  integration over `serve-grpc`.

## Verifying setup

Prefer each sibling's `scripts/validate_setup.py` (present in `nre`,
`asset-harvester`, and `nurec-fixer`) over hand-written checks. For
skills without one (`ncore`, `physical-ai-datasets`, this router),
verify secrets without echoing values:

```bash
hf auth whoami
[ -n "${HF_TOKEN:-}" ]         && echo "HF_TOKEN length=${#HF_TOKEN}"                 || echo "HF_TOKEN unset"
[ -n "${NGC_CLI_API_KEY:-}" ]  && echo "NGC_CLI_API_KEY length=${#NGC_CLI_API_KEY}"   || echo "NGC_CLI_API_KEY unset"
[ -n "${NGC_API_KEY:-}" ]      && echo "NGC_API_KEY length=${#NGC_API_KEY}"           || echo "NGC_API_KEY unset"
```

See [`secrets-handling.md`](secrets-handling.md) for the bash
anti-patterns to avoid and the full NGC key resolution order.
