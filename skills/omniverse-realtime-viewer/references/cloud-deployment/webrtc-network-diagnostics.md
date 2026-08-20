# WebRTC Network Diagnostics

## Triggers

Use this reference when a remote viewer's frontend loads or signaling connects
but video does not arrive, ICE fails, or deployment works only inside a cluster.

## Model The Paths Separately

A streamed viewer has a control/signaling path for session and SDP exchange and
a media path used by ICE/WebRTC for encoded video. Health or WebSocket success
proves only the first part. The orchestrator and upstream streaming workflow
decide the endpoint, ICE servers, and media routing. Do not invent a port range
or replace that workflow with a direct public-IP setting.

## Verification Sequence

Capture evidence in this order:

1. The application reaches renderer and stream readiness.
2. The resolved signaling endpoint is reachable from the browser network.
3. Browser ICE reaches `connected` or `completed`.
4. The video element receives decoded frames and advances `currentTime`.
5. Reconnect and clean shutdown release the session as expected.

Use browser WebRTC diagnostics and server logs for signaling, ICE, encoder,
and disconnect events. Test from the intended user's network class; a
cluster-local test does not prove external media reachability.

Do not use TCP-only port forwarding as an end-to-end WebRTC test. It can prove
that an HTTP or signaling listener exists while leaving media untested.

## Failure Patterns

| Observation | Likely boundary | Next action |
| --- | --- | --- |
| Frontend cannot reach signaling | Endpoint, TLS, proxy, or session routing | Verify resolved endpoint and upstream session state |
| Signaling succeeds but ICE fails | TURN/STUN, firewall, NAT, media routing | Inspect upstream ICE/media configuration and browser diagnostics |
| ICE connects but no decoded frames | Encoder, forwarding, codec, or client decode | Inspect stream logs, negotiated codecs, and video state |
| Works inside cluster only | External endpoint or media exposure | Validate externally and follow upstream exposure workflow |
| Works once, reconnect fails | Session lifecycle or stale renderer ownership | Check end-session, resume, cleanup, and render-loop concurrency |

For NVCF self-hosted deployments, return to upstream
`streaming-self-hosted` after collecting observations. Keep platform-specific
routing fixes out of this package unless upstream makes them application
contract.
