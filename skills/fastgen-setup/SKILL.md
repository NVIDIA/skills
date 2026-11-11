---
name: fastgen-setup
description: >-
  Set up or repair an NVIDIA FastGen environment for a chosen task: Docker or Conda installation, credentials and model access, weight and dataset downloads, and an optional bounded smoke test. Use it before any FastGen command whenever the environment, access or inputs are not yet known to work.
license: Apache-2.0
metadata:
  author: "Julius Berner <jberner@nvidia.com>"
  domain: ml
---

# Set up FastGen

Decide for **each** step whether it is needed from the request, the checkout and the
existing local setup; ask only when a consequential choice stays open. Do not reinstall,
replace credentials, download unrelated data or launch a test merely because this skill
was selected.

1. **Environment.** Follow the checkout's [installation instructions](https://github.com/NVlabs/FastGen/blob/main/README.md#installation)
   and [development setup](https://github.com/NVlabs/FastGen/blob/main/CONTRIBUTING.md#development-setup); Docker is the reference
   environment. Dependencies and pins live in [requirements.txt](https://github.com/NVlabs/FastGen/blob/main/requirements.txt) and
   [setup.py](https://github.com/NVlabs/FastGen/blob/main/setup.py); install accelerator-enabled PyTorch for the host architecture
   first where the README says so. Reuse a compatible environment unless isolation was
   requested.
2. **Context.** Apply [fastgen-runtime](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-runtime/SKILL.md) for paths,
   roots and launch rules without restarting setup from there. Confirm that
   `import fastgen` resolves to the intended checkout and record dependency versions.
3. **Access.** Identify the services the task uses (W&B, Hugging Face, S3) and whether
   access already works. Configure missing credentials as described under
   [environment variables](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/README.md#environment-variables) and verify
   without printing secrets. Check the documented precedence of the shell, import-time
   `.env` and configured credential files before choosing which source to change.
   [set_env_vars](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/env.py) skips file loading for a falsy
   `credentials_dir`; use that opt-out when preserving injected secrets.
   Gated-model acceptance is distinct from authentication;
   report the missing piece precisely.
4. **Weights and data.** Inventory what the selected config needs. Follow
   [pretrained models](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/README.md#pretrained-models) for weights and
   [fastgen-dataset](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-dataset/SKILL.md) for data; the CIFAR-10 and
   ImageNet recipes use `scripts/download_data.py`. Download only the selected task's
   inputs after checking size, license and storage; inference needs no training data.
   Preparation scripts may import GPU libraries, so check where they must run. Keep the
   documented conversion and integrity checks when using a mirror.
5. **Smoke test (optional).** Use [fastgen-train](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-train/SKILL.md) for a
   bounded run of a supported recipe on the requested hardware and logging destination.
   A resolved config or a successful install is not a training test.

Report which steps were already satisfied, performed, unnecessary or blocked, with the
environment location and any test artifacts. For failed installs, diagnose architecture
and binary compatibility before changing versions; keep a failed optional test distinct
from an unusable installation.
