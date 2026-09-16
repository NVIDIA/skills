# NVCF Viewer Container Contract

## Triggers

Use this reference when an Omniverse Realtime Viewer image must run inside an
NVCF self-hosted deployment, or when its image, readiness, lifecycle, or WebRTC
handoff is unclear under the deployment platform.

## Boundary

The upstream `streaming-self-hosted` repository owns NVCF infrastructure,
function and deployment schemas, media routing, TURN/ICE provisioning, storage,
and platform readiness gates. This reference defines only the generated
viewer application's contract at that boundary.

Do not copy NVCF API payloads, platform environment variables, manifests, or
port matrices into the viewer application. Read the upstream orchestration and
selected component skills for those details.

## Application Contract

The viewer image should:

1. Start one clearly owned process and return a non-zero exit code with an
   actionable log when renderer or encoder initialization fails.
2. Run the OVRTX/OVStage render path on the assigned NVIDIA GPU. Treat one GPU
   as the default capacity unit for one active streamed viewer unless the
   deployment contract explicitly provides another model.
3. Expose the readiness endpoint and signaling surface required by the selected
   upstream deployment. Keep readiness false until the renderer initializes,
   the stream service listens, and the first valid frame is ready for delivery.
4. Consume connection, ICE, TURN, and lifecycle values supplied by the
   deployment integration. Do not bake a cluster address, public IP, NVCF
   token, or guessed media port into the image.
5. Stop and reconnect cleanly according to the selected streaming lifecycle.
   Do not destroy shared renderer state from a request handler while the render
   loop is active.

## Image Handoff Checklist

- Use a supported NVIDIA runtime and compatible OVRTX/OVStage/ovstream set.
- Verify GPU visibility, encoder capability, and native library resolution.
- Keep assets, frontend output, and stage paths explicit.
- Inject credentials and storage secrets through the deployment system.
- Use an immutable image tag or digest for each deployment version.
- Capture startup, readiness, first frame, stream, reconnect, and shutdown
  evidence with secrets and tenant-specific endpoints removed.

## Readiness Evidence

Prove in the same image and an equivalent GPU environment that the process starts
with the deployment entrypoint, fails readiness before renderer warmup, reaches
readiness after the first valid frame, delivers decoded video through the
resolved endpoint, and stops without leaving a renderer or stale session.

If the platform changes the readiness path, lifecycle callbacks, or browser
connection model, update deployment configuration from upstream guidance; do
not make the viewer image emulate a different platform contract.

After this contract is proven, return to `nvcf-self-hosted.md` and upstream
`streaming-self-hosted`. Record resolved endpoints only in sanitized validation
artifacts, not reusable skill text.
