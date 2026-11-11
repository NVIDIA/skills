# Stylized AOV Guide Contracts

Use this reference before enabling any guide-driven preset. Its purpose is to
turn an installed OVRTX output into a typed capability, or reject it cleanly.
It does not assume that authoring a RenderVar makes the tensor available.

## Contents

- [Minimal Guide Set](#minimal-guide-set)
- [Capability Record](#capability-record)
- [Packed Normal Adapter](#packed-normal-adapter)
- [Plausibility And Alignment Probe](#plausibility-and-alignment-probe)
- [Capability And Fallback Matrix](#capability-and-fallback-matrix)
- [Lifetime Rules](#lifetime-rules)
- [Required Tests](#required-tests)

SPG nodes consume these guides through same-product graph connections. The
mapping probes below establish their observed contract; they are validation
and capability discovery, not a Python guide-mapping chain in the live style
pipeline. See [SPG processing](../spg-processing/README.md).

## Minimal Guide Set

The canonical style family needs only:

| Source | Required observed contract | Consumers |
|---|---|---|
| `LdrColor` | full-resolution `uint8 [H,W,4]`, source RGBA | every preset |
| `NormalSD` | full-resolution world-space normals: direct float, or verified IEEE-754 bit-packed `uint32` decoded exactly; convert to the view-space guide below | Rim, AOV Ink, Facets, Technical, Concept, X-Ray |
| `InstanceSegmentationSD` | full-resolution `uint32 [H,W,1]`, background ID `0` | same guide-driven presets |

Depth is not required for the canonical styles. Technical and X-Ray classify
line visibility with a private geometry depth pass. Do not add `DepthSD` merely
because the preset name suggests depth.

The optional regular-mesh profile in `mesh-xray.md` has a separate enhanced
tier: it requires a runtime-validated float `DepthSD [H,W,1]` only for its
local-depth visible-edge accent. Missing depth must remove that accent and
report a lower tier; it must not disable the normal/ID shell or invent hidden
topology.

Resolve the apparent request-minimal/request-all tension this way:

- a presentation visualizer authors the deterministic minimal superset above;
- a raw AOV inspector may author its larger candidate set from `aov-switching`;
- only runtime-validated outputs are exposed to either UI;
- measure the cost of each additional RenderVar and remove it from the
  presentation path if it exceeds the declared budget.

## Capability Record

For each guide, store source name, mapped tensor name if multi-tensor, device,
dtype, shape, component meaning, coordinate space, background encoding, first
successful frame, and last failure. A preset consumes a guide only after all
checks pass.

### Color Checks

- Resolution equals the active RenderProduct.
- Dtype is `uint8`, shape is `[H,W,4]`, and alpha is plausible.
- Preserve native RGBA in graph input/output resources.
- Copy the final selected output within its map lifetime when the delivery
  path needs an application-owned buffer; swap once only for a BGRA consumer.

### Normal Checks

Accept direct numeric data only after installed-runtime inspection proves the
mapping is a float tensor. Native `NormalSD` XYZ are world-space surface normals.
After any bit decode, transform nonbackground normals as directions (`w=0`)
using the rendered camera's world-to-view rotation, exclude translation, and
normalize. Refresh that rotation with the camera for each frame. The canonical
recipes consume this view-space guide:

- shape is `[H,W,4]` or `[H,W,3]`;
- XYZ are view-space components in `[-1,1]`;
- at least 95% of nonbackground samples have vector length in `[0.75,1.25]`;
- orientation changes continuously and remains pixel-aligned during orbit;
- the fourth component, if present, is ignored unless upstream documentation
  defines it.

If the mapped dtype is an integer container, apply the packed adapter below only
when runtime inspection proves that it carries IEEE-754 float bits. Never use a
numeric integer-to-float conversion.

### Instance-ID Checks

- dtype is `uint32`, shape is `[H,W,1]`;
- background is `0` on the canonical fixture;
- object interiors are stable during a stationary and moving camera test;
- IDs need not be stable across stage reloads because the recipes use equality,
  not semantic identity;
- use native OVRTX pick queries for picking, never this guide.

## Packed Normal Adapter

The canonical packed profile is `uint32 [H,W,4]`, with lanes 0-2 containing the
unchanged IEEE-754 binary32 bit patterns for source-space `Nx,Ny,Nz`; lane 3 is
ignored. Reinterpret each lane's 32 bits as float32 on the same device. Do not
normalize, scale, mask, shift, reorder, or numerically convert the integer.
This bit decode does not change coordinate space; apply the world-to-view
conversion above afterward. Background normal values are irrelevant because
instance ID `0` masks them.

Implement the verified bit-preserving decode inside the consuming SPG shader
or a preceding adapter node, using the installed CUDA/Slang contract. Preserve
all lane bits and keep the result on the same device. A Warp adapter can remain
an independent validation oracle, not a required external presentation pass.

Accept the adapter only when all of these bit fixtures pass exactly:

| Raw lane bits | Decoded float |
|---:|---:|
| `0x00000000` | `0.0` |
| `0x3f800000` | `1.0` |
| `0xbf800000` | `-1.0` |
| `0x3f000000` | `0.5` |

Then render known `+X`, `-Y`, and `+Z` facing surfaces and verify lane order,
sign, vector length, and camera-space behavior. If bit fixtures pass but scene
axes do not, the source contract is different: reject it. This packed profile
is a conditional public adapter, not permission to guess an unknown encoding.

## Plausibility And Alignment Probe

Warm the renderer before probing. Over at least 30 frames containing camera
motion:

1. confirm each key is present;
2. map on the target device inside the frame-output lifetime;
3. validate dtype and exact shape;
4. sample nonempty and finite values;
5. compare guide boundaries against `LdrColor` silhouettes;
6. reject a guide when boundary displacement exceeds `1.25 px` at 1080p;
7. log the accepted contract once, or the first failure once.

Shape-only validation is insufficient. A same-shaped tensor with unknown
encoding must not unlock a style.

## Capability And Fallback Matrix

| Normal | IDs | Lines | Effective behavior | Fidelity |
|---:|---:|---:|---|---|
| yes | yes | yes | requested preset | canonical |
| yes | no | yes | normal-only variant; explicit silhouettes replace ID boundaries | degraded |
| no | yes | yes | ID-masked color variant plus explicit lines | degraded |
| no | no | yes | manifest fallback plus explicit lines | degraded |
| yes/no | yes/no | no | manifest image fallback; report `line_path_unavailable` | degraded |
| no | no | no | unmodified `LdrColor` when no color-only fallback exists | baseline only |

Only the first row may be called canonical. The server returns requested preset,
effective preset or variant, `canonical|degraded|baseline` fidelity, available
guide names, line capability, and degradation reasons. Client state alone never
proves that a guide-driven look is active.

## Lifetime Rules

- Wire guide resources and immutable intermediates into the SPG graph; keep
  application callbacks out of renderer/resource mutation.
- Retain the frame/product owner and final-output mapping until the delivery
  consumer has completed, following its public synchronization contract.
- For validation readback, keep all mapped guides alive until the probe finishes.
- Do not retain mapped tensors across renderer steps or overwrite a source
  texture while a neighborhood node reads it.
- Revalidate and rebind capabilities after stage/resolution changes; a reused
  buffer descriptor is not proof that its data belongs to the current frame.

## Required Tests

- direct float normal fixtures for `+X`, `-Y`, `+Z`, and background;
- exact packed-bit fixtures plus axis surfaces, and rejection when either fails;
- rejection of integer-packed normals without the verified packed profile;
- ID equality boundaries and ID-0 background behavior;
- wrong dtype, wrong shape, empty tensor, NaN, and misalignment rejection;
- every fallback row above;
- stage reload invalidates the capability registry and reprobes guides.
