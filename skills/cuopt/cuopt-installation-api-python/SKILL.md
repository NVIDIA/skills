---
name: cuopt-installation-api-python
version: "26.06.00"
description: Install cuOpt for Python — pip, conda, Docker, verification. Use when the user is installing or verifying the Python API. Standalone; no common skill.
---

# cuOpt Installation — Python (user)

Install cuOpt to *use* it from Python. Standalone skill (no separate common).

## System requirements

- **GPU**: NVIDIA Compute Capability ≥ 7.0 (Volta+). CUDA 12.x or 13.x; match package (cuopt-cu12 / cuopt-cu13).
- **Driver**: Compatible NVIDIA driver.

## pip (Python)

**Choose one** — do not run both. The second install would override the first and can cause CUDA/package mismatch.

- **CUDA 13.x:**
  ```bash
  pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu13
  ```
- **CUDA 12.x:**
  ```bash
  pip install --extra-index-url=https://pypi.nvidia.com 'cuopt-cu12==26.2.*'
  ```

## pip: Server + Client

```bash
pip install --extra-index-url=https://pypi.nvidia.com cuopt-server-cu12 cuopt-sh-client
```

## conda

```bash
conda install -c rapidsai -c conda-forge -c nvidia cuopt
conda install -c rapidsai -c conda-forge -c nvidia cuopt-server cuopt-sh-client
```

## Docker

```bash
docker pull nvidia/cuopt:latest-cuda12.9-py3.13
docker run --gpus all -it --rm -p 8000:8000 nvidia/cuopt:latest-cuda12.9-py3.13
```

## Verify Python

```python
import cuopt
print(cuopt.__version__)
from cuopt import routing
dm = routing.DataModel(n_locations=3, n_fleet=1, n_orders=2)
```

## Verify Server

```bash
python -m cuopt_server.cuopt_service --ip 0.0.0.0 --port 8000 &
sleep 5
curl -s http://localhost:8000/cuopt/health | jq .
```

## Common Issues

- No module 'cuopt' → check `pip list | grep cuopt`, `which python`, reinstall with correct index.
- CUDA not available → `nvidia-smi`, `nvcc --version`, match cuopt-cu12 vs cuopt-cu13 to CUDA.

## Examples

- [verification_examples.md](resources/verification_examples.md) — Python and server verification
