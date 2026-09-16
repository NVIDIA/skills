# Viewer Container Image Build

## Triggers

Use this reference when packaging an OVRTX/ovstream viewer for a GPU host,
NVCF self-hosted deployment, Kubernetes, CI, or a registry-backed orchestrator.

## Choose A Build Path

| Path | Use when | Main concern |
| --- | --- | --- |
| Local Docker | Iterating on a known-good image recipe | Native wheels and registry bandwidth |
| CI builder | Images must be reproducible on merge or release | Runner memory, timeout, secrets |
| Cluster/platform builder | Source and registry are near the target | Permissions, storage, platform policy |

Choose the least complex path that reproduces the target runtime. The build
system is not the deployment system: after pushing the image, follow the
selected upstream deployment workflow.

## Image Requirements

- Start from a supported NVIDIA runtime base for the selected package set.
- Install dependencies from centralized guidance or a pinned app manifest.
- Copy only server code, frontend output, stage assets, launch wrapper, and
  required runtime files.
- Use a strict `.dockerignore` for `node_modules`, caches, credentials, private
  scenes, and unrelated build output.
- Make the entrypoint explicit about environment, stage path, readiness, and
  signal handling.
- Keep credentials, storage strings, and deployment tokens out of image layers.

## Tagging And Verification

Use a unique immutable tag or digest for each candidate image:

```bash
docker build --tag <registry>/<namespace>/<viewer>:<immutable-tag> .
docker run --rm --gpus all <registry>/<namespace>/<viewer>:<immutable-tag> nvidia-smi
docker push <registry>/<namespace>/<viewer>:<immutable-tag>
```

Use the target registry's supported authentication and CLI. Verify the pushed
digest and test the exact pulled image, not only the local build cache. Run the
GPU-container preflight and sanitize captured output.

## Handoff

An image is ready only after package/native-library, renderer/first-frame,
encoder/readiness, browser signaling/ICE/decoded-frame, and required shutdown
or reconnect checks pass. Hand the immutable image reference to
`nvcf-self-hosted.md` or the selected cloud deployment path. Do not add NVCF
function payloads or Kubernetes manifests here; those belong upstream.
