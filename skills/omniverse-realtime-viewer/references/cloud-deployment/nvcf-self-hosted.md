# NVCF Self-Hosted Handoff

## Triggers

Use this reference for NVCF self-hosted streaming, AWS or Azure NVCF infrastructure, NVCF control plane, compute plane, LLS, TURN, streaming proxy, or a self-hosted Omniverse Streaming stack.

## Upstream Source Of Truth

For this path, use [NVIDIA-Omniverse/streaming-self-hosted](https://github.com/NVIDIA-Omniverse/streaming-self-hosted) as the deployment source of truth. Before planning or changing infrastructure, read its root README and its root orchestration skill under `.agents/skills/orchestrate-streaming-stack`. Then read only the selected component skills for the target cloud provider and enabled services.

Do not copy upstream commands, manifests, chart versions, image-mirroring procedures, credential handling, port matrices, cache setup, storage-service setup, or cloud-provider recipes into this viewer skill package. The upstream repository owns those details.

## Boundary

- Upstream owns cloud infrastructure, NVCF control and compute planes, media routing, caches, deployed storage services, and readiness gates.
- This package owns the generated viewer application: OVRTX and OVStage runtime behavior, the selected viewer transport, UI, and app-level validation.
- Brev and OKAS remain supported alternatives in `cloud-deployment/README.md`; select this handoff only when the request is specifically for self-hosted NVCF streaming.

Do not assume a standalone `ovstream` Direct browser configuration applies to an NVCF deployment. Follow the upstream workflow to select and validate the exposed media and browser connection model.

Before handing an image to the upstream workflow, read
`nvcf-viewer-container-contract.md`, `gpu-container-runtime.md`, and
`webrtc-network-diagnostics.md`. These references cover only the application
boundary and validation evidence; they do not replace the upstream deployment
skills.

## Return Contract

After the upstream workflow reaches its deployment and readiness gates, return to the selected viewer references with:

- the resolved viewer endpoint and connection model;
- deployment readiness status and the upstream validation evidence;
- any viewer-container requirements that affect the generated app; and
- sanitized app-level stream, input, reconnect, and shutdown evidence.
