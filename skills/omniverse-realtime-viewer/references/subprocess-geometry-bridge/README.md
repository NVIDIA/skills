<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Subprocess Geometry Bridge

Use when the modeling library needs a different interpreter or native runtime,
or when expensive geometry work must be isolated from the viewer. Read
`dependencies/README.md` → USD BRep first and follow upstream skills and docs
before writing worker-side modeling code. Decide isolation from the selected
artifacts' compatibility, not a fixed early-access Python mismatch.

## Ownership And Startup

The persistent worker owns modeling handles and exact asset files. The parent
owns OVStage, OVRTX, ordinals, selection, and renderer publication. Exchange
owned arrays, scalar metadata, and files; never native handles or OpenUSD objects.
Configure the worker interpreter explicitly, start it with an argument list,
and use the installed public package without hand-built extraction paths.
Follow upstream for native loader remedies. On Windows retain any DLL-directory
handles for the duration of library use; Linux loader remedies are separate.

Use binary pipes, reserve stdout for protocol frames, and drain stderr separately.
Require a versioned READY handshake and startup deadline. Detect exit, timeout,
malformed headers, and short reads; terminate and reap the process before restart.
A restart invalidates every worker-side handle and begins from the accepted spec.

## Bounded Protocol

Send a request ID and monotonically increasing geometry revision with each spec.
Frame replies as a bounded JSON header followed by exactly the declared binary
byte count. Declare dtype, shape, byte order, offset, and length for each array;
validate bounds, non-overlap, finite positions, polygon counts and index ranges
before allocation or publication. Bound header size, total payload, part count,
and execution time. Errors carry the request ID, revision, diagnostic, and zero
payload bytes. Never assume one pipe read returns a whole frame.

Use contiguous NumPy output where the current binding provides it. Its polygon
stream includes counts; decode to USD faceVertexCounts/faceVertexIndices with
bounds checks. Declare normals as vertex or faceVarying according to the actual
layout. Prefer source-face corner normals when supported rather than introducing
crease smoothing solely to repair older vertex-averaged output. Preserve backing
buffers through asynchronous transfer completion.

## Publication And Interaction

Read worker replies on a background thread and enqueue candidates. Apply stage
writes only on the runtime owner thread, serialized with renderer stepping.
Reject superseded revisions, validate before replacing the accepted geometry,
and update exact-asset provenance and selection mapping with the same revision.
Coalesce slider changes to the newest pending spec while one build runs. Keep
showing the last accepted frame and surface progress or failure in the UI.
Do not use a fixed sleep as proof that GPU publication completed; follow the
selected OVStage/OVRTX synchronization and frame-completion contract.

Shutdown closes input, waits with a deadline, then terminates and reaps if needed.
Test many successive builds, fragmented reads, oversized/malformed replies,
worker death, timeout, stale results, and shutdown. Confirm camera input remains
responsive and rejected candidates preserve the previous mesh and exact asset.

See also: `ovstage-live-geometry/README.md`, `exact-brep-in-usd/README.md`,
`smlib-brep-pitfalls/README.md`, `ovstage-ovrtx-integration/README.md`.
