# Docker workflow validation — 2026-11-01

The initial Docker conversion exposed an asynchronous control-timing bug. The
skill now synchronizes every 30 Hz policy step with its matching ROS response.
Both fresh-container verifications of the final code passed the saved <3 mm
insertion criterion. The original failures remain recorded below.

## Diagnosis and fix

The earlier reference run passed at 2.019 mm. Its checkpoint, exported ONNX,
LEAPP configuration, recurrent initial state, simulator build and ROS package
versions match the failed Docker run. PhysX engine and tensor-library hashes
also match. Although flattened robot file hashes differ because USD assigns
internal prototype names differently, all 2,354 composed robot properties match.
Plug and socket file hashes match directly.

The former loop advanced physics while ROS inference was still running. The scene
polled one queued callback per physics step and applied the latest available
command without checking that it belonged to the current policy boundary. Initial
commands could arrive after simulation advanced by up to 87.5 ms. Different
render/encoder speeds changed delivery relative to physics, altering the contact
trajectory; the plug then stalled near the socket entrance while the gripper moved.

A controlled Docker trial with a minimum 50 ms wall interval per render update
passed at 2.009 mm without changing the checkpoint, geometry, contact properties
or gains. That pacing option was diagnostic and has been removed.

The final controller publishes its observation and clock at each 30 Hz boundary,
holds physics until the matching command arrives, then applies that command before
physics advances. Stale messages are drained; a future timestamp or five-second
wall timeout invalidates the trial. The recorder validates consecutive matching
responses and saves `policy-sync.jsonl` plus a timing summary in `result.json`.
Physics remains at 240 Hz and policy inference remains in ROS Deploy/Triton.

## Recorded trials

| Test | Trial | Final position error | Result |
| --- | --- | --- | --- |
| Original asynchronous workflow | attempt-01 | 10.301 mm | Fail |
| Original asynchronous workflow | attempt-02 | 10.297 mm | Fail |
| Original asynchronous workflow | attempt-03 | 10.300 mm | Fail |
| Diagnostic: minimum 50 ms per render update | attempt-01 | 2.009 mm | Pass |
| Synchronization implementation | attempt-01 | 1.892 mm | Pass |
| Final code: official image, cold caches | attempt-01 | 1.892 mm | Pass |
| Final code: prepared image, warm caches | attempt-01 | 1.893 mm | Pass |

Seven Docker trials were recorded across five invocations: three original
failures, two diagnostic successes, and two successful verifications of final
code. All used the same checkpoint and prepared-grasp case. This is regression
and integration evidence, not an independent policy success-rate estimate.

Each synchronized trial recorded 420 matching policy responses and 450 video
frames at 30 fps over 15 simulated seconds. Every response matched its held
simulation boundary; the policy-to-decoder simulated-time delay was zero. The
first synchronized trial and the cold-cache verification have identical 900-sample
physics traces. The warm-cache final result differs by about 0.0014 mm, so do not
claim universal bit-for-bit determinism. Both final trials held the success
criterion for the final 13.33 simulated seconds; their maximum per-step wall waits
were approximately 125 ms and 115 ms. A final synchronized video frame was visually
inspected and every recording passed its in-container video decode check.

## Environment and evidence

- Checkpoint: `model_999.pt`, SHA-256
  `d3eb2cb220724d3fe89d0aa5ee2dcb305575422f2da93bb05f926361af208e84`.
- LEAPP 0.6.1: a fresh export and all five numerical validation examples passed
  for every invocation. ONNX SHA-256:
  `a80bf8a980531373c8e12440ea472056db1f97bfb2596b01cdfd412b0196c5e8`.
- Isaac Sim image: `nvcr.io/nvidia/isaac-sim:6.0.0`, actual build
  `6.0.0-rc.59+release.41464.5f2772bc.gl`; base image ID
  `sha256:17d31a6b38b63e650a7ac3d6cfc5da9fa05208d8b933f6f796c11984dd1a3507`.
- ROS base: documented NVIDIA image ID
  `sha256:9e31ec9d7194ca3c895d2024370b87a27550db466d819184c441c5d8a5f10c54`.
  Diagnostic runs reused installed dependencies through locally prepared images
  and created new containers for every invocation. The cold-cache verification
  used the official simulator image and installed ffmpeg inside the new container.
- Semantic simulator compatibility, H.264 preflight, Triton readiness and asset
  verification passed. Asset checks covered 211 robot, 36 plug and 41 socket
  properties with matching geometry. They compare saved URLs resolved now;
  historical training-time asset bytes were not recorded.
- Both final invocations' recorded script hashes match the final code. All ten
  containers owned by these five invocations were confirmed stopped, and no GPU
  compute processes remained after verification.

## Regression coverage

All 35 deployment tests passed, including matching responses, stale-message
handling, future-command rejection, bounded timeouts, duplicate boundaries,
missing synchronization evidence and container cleanup. Skill metadata and
whitespace validation passed. Caller-selected output directories retain reports,
commands, exact image/package identities, original failures, videos and timing logs.
