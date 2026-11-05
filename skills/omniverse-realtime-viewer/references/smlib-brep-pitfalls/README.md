<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# BRep Modeling In A Live Viewer

Use for fillet/boolean failures, empty results, handle lifetime errors, or
interactive modeling controls. Read `dependencies/README.md` → USD BRep first,
then the upstream API-use skill, operation guide, and error documentation.
The upstream kernel skills own diagnosis; old early-access measurements are
not current limitations or safe universal radius thresholds.

## Transactional Rebuilds

Keep the last accepted model and render mesh until a candidate passes validation.
Copy before consuming or mutating operations when the original must survive;
check each operation's contract, including export side effects. Reacquire borrowed
faces, edges, surfaces, and curves after topology or representation changes.
Do not use Python wrapper identity to deduplicate native topology.

On failure, discard the candidate and report the operation diagnostic. Restart
from a deterministic model specification rather than reuse consumed handles.
Check finite bounds, expected component counts, and the invariants appropriate
for the requested solid or sheet. Volume alone cannot prove geometric validity;
expensive queries belong in worker validation, not the render loop.

Use upstream seam/adjacency guidance for edge eligibility. A two-face test alone
is not a proof that a fillet is valid. Derive UI limits from the model and current
operation contract, then validate the result. Do not blanket-ban multi-body
booleans or aspect-ratio controls based on an older binary's behavior.

## Tessellation Acceptance

Use strict complete tessellation for normal publication. If partial output is
explicitly requested, retain per-face failures and label the preview incomplete;
do not silently export it as a finished solid. Upstream reports distinguish
edge-incidence manifold checks from full solid validity.

Prefer binding-owned NumPy arrays where the selected artifact provides them.
Match normals to their declared vertex or corner interpolation. Measure controls
on deterministic curved fixtures: edge-length and aspect-ratio parameters are
subdivision targets, not guarantees on final triangle shapes.

Validate candidate failure recovery, repeated rebuilds, preservation of the
accepted body, and selection invalidation after topology changes.

See also: `subprocess-geometry-bridge/README.md`,
`exact-brep-in-usd/README.md`, `ovstage-live-geometry/README.md`.
