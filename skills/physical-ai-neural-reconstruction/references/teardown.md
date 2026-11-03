# Cross-Skill Teardown

A complete NuRec workflow can leave **150 GB+ on disk** between
container images, model weights, code clones, conda envs, and output
directories. Each sibling skill has its own dedicated "Teardown"
section — read them in this order when the user no longer needs the
workflow:

| Sibling skill | Approximate footprint | Where the cleanup lives |
|---------------|------------------------|--------------------------|
| `nurec` | ~120 GB images + caches + per-run outputs | [`SKILL.md#teardown`](https://github.com/NVIDIA/nurec/blob/main/skills/nurec/SKILL.md#teardown) + [`references/teardown.md`](https://github.com/NVIDIA/nurec/blob/main/skills/nurec/references/teardown.md) |
| `harmonizer` | 100 GB+ possible (Cosmos image / build cache, HF model + dataset, checkout, outputs) | [`SKILL.md#teardown`](https://github.com/NVIDIA/harmonizer/blob/main/skills/harmonizer/SKILL.md#teardown) + [`references/teardown.md`](https://github.com/NVIDIA/harmonizer/blob/main/skills/harmonizer/references/teardown.md) |
| `asset-harvester` | ~60 GB+ — checkpoints (~12.8 GB), two conda envs (~10–15 GB each), benchmark assets, outputs | [`skills/asset-harvester/references/troubleshooting.md#teardown`](https://github.com/NVIDIA/asset-harvester/blob/main/skills/asset-harvester/references/troubleshooting.md#teardown) |
| `ncore-data-conversion` | clip-dependent | NCore shards live under `<dataset_dir>/`; delete after `nurec` training is done. That skill has no teardown companion. |
| `physical-ai-datasets` | dataset-dependent | HF caches under `${HF_HOME:-$HOME/.cache/huggingface}/hub/`; remove the per-dataset directory |

Two practical rules that apply across every container-based sibling:

1. Pin `-u $(id -u):$(id -g)` on every `docker run` so outputs land
   owned by the user, not by `root`. Every documented `docker run` in
   `nurec` and `harmonizer` already does this. If outputs end up
   `root`-owned anyway, recover with
   `sudo chown -R "$(id -u):$(id -g)" <output_dir>` before deleting.
2. Do **not** revoke `NGC_CLI_API_KEY` / `NGC_API_KEY` / `HF_TOKEN`
   as part of teardown unless there is reason to believe they were
   leaked — they are per-user and shared across every NVIDIA workflow
   on the host.
