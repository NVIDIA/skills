# NVCF Self-Hosted Handoff

## Triggers

Use this reference for NVCF self-hosted streaming, AWS, Azure, or customer-managed on-premises Kubernetes NVCF infrastructure, NVCF control plane, compute plane, LLS, TURN, streaming proxy, a self-hosted Omniverse Streaming stack, or an in-place stack upgrade.

## Upstream Source Of Truth

For this path, use [NVIDIA-Omniverse/streaming-self-hosted](https://github.com/NVIDIA-Omniverse/streaming-self-hosted) as the deployment source of truth. Before planning or changing infrastructure, read its root README and its root orchestration skill under `.agents/skills/orchestrate-streaming-stack`. Then read only the selected component skills for the target cloud provider and enabled services.

Do not copy upstream commands, manifests, chart versions, image-mirroring procedures, credential handling, port matrices, cache setup, storage-service setup, or cloud-provider recipes into this viewer skill package. The upstream repository owns those details.

## Build A New Deployment

Start from the public upstream repository even when only this published skill folder is available. Read or clone it into the generated project's dependency workspace; do not require this staging repository, its templates, or an existing cluster. Read its root README and `.agents/skills/orchestrate-streaming-stack/SKILL.md`, then the selected provider and component skills. Resolve runtime acquisition through `../dependencies/README.md`.

For a new deployment, start with the current supported upstream release unless the user specifies a version. Read the [latest release notes](https://docs.omniverse.nvidia.com/ov/ovs/latest/release-notes.html), [Artifact Manifest](https://docs.omniverse.nvidia.com/ov/ovs/latest/manifest.html), and the linked NVCF compatibility matrix before selecting components. Keep the upstream skills, manifest, and documentation aligned to the chosen release; record the resolved versions in generated deployment artifacts rather than pinning them in this skill. Apply these checks:

- AWS, Azure, and customer-managed on-premises Kubernetes are supported. For on-premises requests, use the upstream `generic-kubernetes-provider` skill and [On-Premises Deployment Guide](https://docs.omniverse.nvidia.com/ov/ovs/latest/on-prem-deployment-guide.html).
- LLS Streaming Proxy, TURN service, and the LLS Helm chart are publicly available on NGC. Do not require early-access approval for these components. Public availability does not remove the selected platform's authentication, registry, or infrastructure prerequisites; follow the [LLS deployment guide](https://docs.omniverse.nvidia.com/ov/ovs/latest/low-latency-streaming/lls-deployment.html).
- OVDC (Omniverse Derived Cache, formerly DDCS) replaces deprecated GXCache as the recommended shader cache; new deployments should not install GXCache. OVCC (Omniverse Content Cache, formerly UCC) is the current content-cache name. Resolve versions and any chart image overrides from the manifest, then verify running images against it. Do not follow stale cache names or chart defaults in older upstream examples.
- NVCF control-plane, compute-plane, and observability stacks are independently versioned. Select a compatible combination from the NVCF compatibility matrix rather than giving every chart the platform documentation version.
- Check whether the current upstream workflow deploys and enables the function autoscaler. If it does not, use the upstream NVCF guidance when autoscaling is requested. Do not promise capacity beyond the function's minimum instance count until scaling behavior is configured and validated.

Build and validate the viewer image using `nvcf-viewer-container-contract.md`, then let the upstream workflow provision the selected platform and register the streaming function. Require both platform readiness and decoded browser frames through the resolved endpoint before reporting the deployment ready.

For an explicitly requested existing-stack upgrade, use upstream `streaming-inplace-upgrade-orchestrator` and its component skills; that migration workflow is not a prerequisite for a fresh build.

Upstream platform support does not establish that Kit-specific cache settings or browser profiles apply to a standalone OVRTX viewer. Confirm the generated viewer's integration contract through the upstream workflow.

## Fresh Kubernetes Bring-Up Checkpoints

Use these checkpoints to organize the generated project's runbook. Execute the
platform phases through the upstream orchestration skill and its child skills;
these checkpoints do not replace their commands or readiness gates.

1. **Record the integration before writing code.** Capture the chosen public
   release and compatibility sources, upstream checkout revision, selected
   provider/component skills, browser connection profile, signaling route,
   credential source, lifecycle contract, and expected validation evidence.
   Resolve dependencies through `../dependencies/README.md`; do not combine
   individually newest components without checking compatibility. Record access
   prerequisites and unavailable artifacts explicitly rather than substituting
   private builds or declaring the whole platform unavailable.
2. **Establish Kubernetes capabilities.** Let the selected upstream provider
   skill verify GPU and encoder support, CNI, storage, registry trust/pulls,
   ingress/load balancing, and external media exposure. A raw VM running
   Kubernetes, including one hosted by Brev, follows the upstream generic
   Kubernetes provider path rather than inheriting managed EKS/AKS capabilities.
   Discover existing access and cluster state before provisioning replacements.
3. **Bring up the platform in upstream order.** Follow prerequisites, control
   and compute planes, LLS, and the workload-required services selected by the
   orchestrator. Save sanitized evidence at each gate. A healthy node alone does
   not establish registered, schedulable NVCF GPU capacity; use the upstream
   capacity checks before registering/deploying the viewer function.
4. **Prove minimal video through NVCF first.** Use a small OVRTX-rendered scene
   and the container contract below. Resolve the browser profile from upstream
   NVCF guidance and the released browser SDK's NVCF documentation/examples;
   the standalone Direct example is not an NVCF template. Validate the real
   signaling proxy, session credentials, and external media route with advancing
   decoded frames before adding the full viewer UI. Keep frontend/proxy changes
   independently deployable where practical so browser configuration fixes do
   not require rebuilding the GPU image and repeating renderer warmup.
5. **Prove session recovery before expanding the app.** Follow
   `nvcf-viewer-container-contract.md` for disconnect/resume, expiry, callback
   acknowledgement, and a stable fresh worker. Then validate viewer interaction
   and scene switching through the same deployed endpoint.

Keep four results distinct in the runbook: renderer ready, platform/function
ready, browser decoding video, and a fresh session available after cleanup.
Report the last completed checkpoint and its evidence when any later gate fails.

## Boundary

- Upstream owns cloud infrastructure, NVCF control and compute planes, media routing, caches, deployed storage services, and readiness gates.
- This package owns the generated viewer application: OVRTX and OVStage runtime behavior, the selected viewer transport, UI, and app-level validation.
- Standalone ovstream on Brev and OKAS remain alternatives in `cloud-deployment/README.md`. NVCF on a Brev-hosted Kubernetes cluster follows this handoff.

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
