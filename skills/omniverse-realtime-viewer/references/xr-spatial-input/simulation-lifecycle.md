<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# XR simulation and input lifecycle

For latency-sensitive XR, preserve the same ordering on the simulation owner
but cross to the frame owner with immutable completed snapshots:

```text
simulation owner: fixed-step clock -> solve -> synchronize -> validate
                  -> replace pending completed snapshot (latest wins)

OVXR/frame owner: drain lifecycle commands first
                  -> take newest ready snapshot without waiting
                  -> compose camera/input and ready simulation state; publish one ordinal
                  -> retain the last simulation pose when none is ready
                  -> acquire poses, render, and submit the XR frame
```

Do not pass a live solver state or DLPack view across that boundary. Do not wait
on the simulation device from the XR frame path. If GPU sharing still stalls
submission, measure the contention and reduce bounded simulation work or choose
a supported stream/device arrangement; do not skip Newton steps, synthesize
poses, or make XR input wait for a grasp result.

**Finite is necessary, not sufficient.** For externally driven proxies and
objects they contact, bound position, orientation norm, linear velocity,
angular velocity, per-frame displacement, and per-frame rotation using measured
scene extent, fixed `dt`, and the interaction's declared speed envelope. Reject
or halt on an impossible jump before publishing it. A finite hand teleport can
inject enough energy to launch a body out of view while every ordinary health
counter remains green.

For a grasp-and-release claim, assert the defining phases independently: named
contact or native constraint state before lift; bounded relative slip while
held; release while the source is open or lost; free-body gravity response;
floor contact; and velocity/energy below a declared settling threshold. Do not
collapse these into “the body moved.”

### Transient input loss must not own the simulation lifecycle

A tracked controller or hand has a shorter lifetime than the stage attachment.
Give each input session its own epoch inside the active stage generation. Focus
loss, tracking loss, or client disconnect retires that epoch; it does **not**
justify tearing down or rapidly restarting the renderer, stage, simulation, or
shared transport processes.

Apply loss handling through the normal owner-thread drain:

1. Stop accepting pose samples for the retired input epoch.
2. Queue release of anything held by that input source.
3. Clear pending pose deltas and gesture state.
4. Park the source's kinematic proxies outside the contact region, or disable
   them through a supported runtime control.
5. Publish the release and subsequent free-body motion normally.

Do not mutate solver state from the disconnect, focus, or input callback. Do not
leave a held body attached to a missing source, and do not keep the last hand
pose active. Reacquisition begins a new input epoch and requires one complete,
current joint snapshot before proxies become collidable; interpolate only
within that epoch, never from the retired pose to the reacquired pose.

Fence pose commands by both stage generation and input epoch. A model rebuild
or stage transition invalidates all input epochs; an input reconnect invalidates
only that source's old samples. This two-level fence prevents stale tracking
from moving a replacement model without forcing an expensive stage reattach for
ordinary tracking interruptions.

### Quiesce before disconnect or Stop XR

A full XR-session disconnect or explicit Stop XR is different from one hand or
controller losing tracking. Drain that lifecycle command before acquiring
another XR frame or scheduling another simulation tick. On the owner thread:

1. mark the XR session quiesced so no new input, publish, or simulation work is
   accepted for its epoch;
2. release remote constraints, clear direct-hand ownership, park its proxies,
   and discard queued gestures and pending completed snapshots;
3. stop scheduling fixed-step work and drain already submitted device work;
4. finish or cancel the current publication according to the runtime contract;
   then detach/stop XR and its consumers.

Make quiesce idempotent. Preserve a user's prior simulation-running intent only
as ordinary app state; restore it after a successful reconnect, a new input
epoch, and the initial held-state publish—not from the disconnect callback and
not while old work can still complete.

### One frame owner, asynchronous producers

XR does not create a second stage lane. Keep the OVXR frame helper, OVStage
publication, and render submission on one owner thread even when Newton, sensor,
or controller work runs asynchronously. Those producers own their private
clock and state and hand the frame owner immutable, generation-tagged completed
snapshots through a bounded latest-wins channel. They never advance the write
floor, call the renderer, or expose live solver/DLPack buffers across the
boundary.

At the start of a frame, the owner composes the newest ready simulation snapshot
with current camera, input and animation changes in one bounded publication.
When physics is pending, retain its last accepted pose while publishing other
changes normally. Delegate XR rendering through OVXR. A
latency-sensitive XR frame must not wait for a solver merely to obtain a newer
pose; freshness is telemetry, not permission to block head pose, hands,
locomotion, composition, or frame submission. Drop snapshots from retired stage
generations rather than retrying them against the current stage.


Read [shared simulation integration](../realtime-sim-integration/README.md)
for the general composer and [upstream stage publication](../dependencies/ovxr.md)
for SDK calls. Co-writing uses one application allocator and supplied ordinals;
never invent independent floor-plus-one counters for each writer.
