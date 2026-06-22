# Spec and args construction

Detailed construction strategy per `mode`, the recommended decision order, and
worked examples for spec-driven and path-keyed jobs. Read after determining the
action's `mode` from `skill_info.yaml`.

## How to construct, per `mode`

The skill's action declares its config mechanism in `skill_info.yaml`'s
`actions.<action>.mode` field. Treat missing `mode` as invalid metadata and fix
the skill instead of inferring a default. The agent's construction strategy
follows from that:

| `mode` | How to construct |
|---|---|
| `args` | Copy the `actions.<a>.args` block from `skill_info.yaml` as your template. Substitute placeholders (`{storage_root}`, `{split_id}`, `{num_gpus}`, etc.) with the user's runtime values. Pass to `build_entrypoint(args=...)`. |
| `config` + `references/spec_template_<a>.yaml` exists | Load the template via `yaml.safe_load(...)` as the base spec; apply user overrides on top. Pass to `build_entrypoint(specs=...)`. |
| `config`, no template | Follow the model's `SKILL.md` — typically a "Critical Overrides" section lists which keys must be set. Construct the spec accordingly. Pass to `build_entrypoint(specs=...)`. |
| `passthrough` | Bare command + path-keyed `inputs={container_path: uri}` / `outputs=[paths]`. Pass to `build_entrypoint(inputs=..., outputs=...)`. |

**Recommended decision order:**

1. Read `action_cfg = skill_info["actions"][action]`. Check `action_cfg["mode"]`.
2. For `config` mode: check `references/spec_template_<action>.yaml`. If it exists, **load it as your base** — don't rebuild from scratch.
3. Apply user overrides on top (plus any "Critical Overrides" rows from the model's `SKILL.md`).
4. For `args` mode: copy `action_cfg["args"]`, fill placeholders, hand to `build_entrypoint(args=...)`.

```python
import yaml
from pathlib import Path

skill_dir = Path(bank) / "skills/models/<model>"
skill_info = yaml.safe_load((skill_dir / "references/skill_info.yaml").read_text())
action_cfg = skill_info["actions"][action]
mode = action_cfg["mode"]

if mode == "args":
    args = dict(action_cfg["args"])
    args["weak-video-list"] = args["weak-video-list"].format(storage_root=user_storage)
    # ... substitute remaining placeholders
    ep = build_entrypoint(command=action_cfg["command"], args=args, ...)

elif mode == "config":
    template = skill_dir / f"references/spec_template_{action}.yaml"
    specs = yaml.safe_load(template.read_text()) if template.exists() else {}
    # apply user overrides on top
    specs.setdefault("policy", {})["model_name_or_path"] = user_model
    # ... etc
    ep = build_entrypoint(command=action_cfg["command"], specs=specs, ...)
```

## Spec-driven jobs

The skill's action declares a config file (`config_format`, `command: ... {config_path} ...`). Covers TAO models (DINO, BEVFusion, classification-pyt, …) and cosmos-rl — anything whose container reads a spec file and writes outputs to declared spec keys. Use whichever platform SDK fits the target backend; the `build_entrypoint` call is identical across platforms.

```python
import yaml
from tao_sdk.script_runner import build_entrypoint
from tao_sdk.versions import resolve_container_image
# pick the SDK matching your target platform:
from tao_sdk.platforms.slurm      import SlurmSDK      # or
from tao_sdk.platforms.kubernetes import KubernetesSDK # or
from tao_sdk.platforms.docker     import DockerSDK     # or
from tao_sdk.platforms.brev       import BrevSDK

skill_info = yaml.safe_load(open(f"{bank}/models/tao-train-dino/references/skill_info.yaml"))
action_cfg = skill_info["actions"]["train"]

specs = {
    "dataset": {
        "train_data_sources": [{
            "image_dir":  "s3://my-bucket/coco/train/images",
            "json_file":  "s3://my-bucket/coco/train/annotations.json",
        }],
        "val_data_sources": [{
            "image_dir":  "s3://my-bucket/coco/val/images",
            "json_file":  "s3://my-bucket/coco/val/annotations.json",
        }],
        "num_classes": 80,
    },
    "train": {"num_epochs": 10, "num_gpus": 8},
    # No results_dir — script_runner auto-fills at runtime.
}

ep = build_entrypoint(
    command=action_cfg["command"],                       # e.g. "dino train -e {config_path}"
    specs=specs,                                          # → infers config mode
    inputs=action_cfg["inputs"],                          # spec-keyed dict from skill_info.yaml
    outputs=action_cfg["outputs"],
    config_format=action_cfg["config_format"],            # "yaml" / "toml" / "json"
    upload_excludes=action_cfg.get("upload_excludes", []),
)

sdk = ...   # one of the SDKs above
job = sdk.create_job(
    image=resolve_container_image(skill_info["container_image"]),
    command=ep["command"],
    gpu_count=8,
    # Platform-specific kwargs go here — see each platform's SKILL.md:
    #   SLURM:      partition, account, num_nodes
    #   Kubernetes: namespace, node_selector, tolerations, num_nodes
    #   Docker:     mounts
    #   Brev:       instance_id, gpu_type, cloud_cred_id, workspace_group_id
)
print(f"Job submitted: {job.id}    Results: {job.results_dir}")
```

## Path-keyed jobs (no config file)

The skill's action does not write a spec file — inputs are passed as `{container_path: uri}` and outputs as a list of container paths. Covers HF inference scripts, custom commands, anything that takes its inputs via direct paths rather than a config file.

```python
ep = build_entrypoint(
    command="python infer.py --model /models/cosmos --input /data/in --output /results",
    inputs={                                              # path-keyed → infers passthrough mode
        "/models/cosmos": "hf_model://nvidia/Cosmos3-Nano",        # HF Hub
        "/data/in":       "s3://bucket/test/in",                    # S3
        # also supported: "ngc://..."
    },
    outputs=["/results/"],
)
sdk.create_job(image=img, command=ep["command"], gpu_count=1)
```

In passthrough mode the runtime dispatches each input URI by scheme — `s3://`, `hf_model://`, `ngc://` — to the right downloader. No spec rewriting, no `{config_path}`. After the command, listed output paths are uploaded per the same destination resolution rules (S3 if `S3_BUCKET_NAME`, else mount, else container-ephemeral with warning).

## Entrypoint shape follows declared `mode`

Read `actions.<action>.mode` from `skill_info.yaml` first, then pass the matching argument shape to `build_entrypoint`:

| Declared mode | What the agent passes |
|---|---|
| `config` | `specs=...` with spec-keyed `inputs` / `outputs`; the helper writes the spec file, rewrites URIs, and runs the command |
| `args` | `args=...` with optional spec-keyed `inputs` / `outputs`; the helper substitutes CLI args into the command template |
| `passthrough` | path-keyed `inputs=...` and/or `outputs=...`; the helper downloads to listed paths, runs the command, and uploads listed outputs |

Do not infer mode from missing metadata. Missing `mode` means the skill contract is stale.
