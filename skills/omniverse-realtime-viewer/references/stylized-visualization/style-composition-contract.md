# Stylized Composition Contract

Use this reference for a new stylized look that does not already have a named
canonical recipe. It turns an art-direction request into a small, testable
SPG render graph without pretending that one recipe fits every visual
language.

## Start With A Layer Plan

Before writing a shader or exposing UI, record these decisions. Ask for a
visual reference when the requested look has ambiguous art direction; otherwise
state the interpretation before implementation.

| Layer | Choose and declare |
|---|---|
| Surface | preserve `LdrColor`, flat palette, normal-form, material-like wash, or presentation shell |
| Guides | color only, normal, ID, depth, albedo, or another runtime-validated guide |
| Marks | none, screen-space marks, topology/curve lines, silhouettes, or both |
| Occlusion | visible only, hidden-line convention, all-feature continuation, or no depth claim |
| Scope | participating prims, inherited visibility/purpose, exclusions, and passive supports/occluders |
| Motion | static, screen-locked, object-locked, camera-dependent, or intentionally animated |
| Fallback | the live result when each guide or line path is unavailable |
| Proof | fixed views, motion path, temporal check, and performance counters |

Keep this plan in a server-owned `StylePreset` or equivalent data record. Do
not distribute visual policy across browser controls, per-frame branches, and
scene mutation.

```python
@dataclass(frozen=True)
class StyleLayerPlan:
    surface_mode: str
    required_guides: tuple[str, ...]
    optional_guides: tuple[str, ...]
    mark_modes: tuple[str, ...]       # screen | topology | silhouette
    occlusion_mode: str               # visible | hidden | continuation | none
    scope_policy: str
    temporal_mode: str                # static | camera | animated
    fallback_name: str
```

The server resolves this record against current runtime capabilities and returns
the requested/effective style plus fidelity and degradation reasons. A client
selection alone does not prove that the corresponding visual layers exist.

Read [SPG processing](../spg-processing/README.md) for public node authoring,
internal resources and publication. Image, marks, private geometry depth and
composition are GPU nodes; do not replace this plan with an external Warp/GL
presentation chain. Materials and optional lights remain viewer scene state.

## Choose The Smallest Sufficient Graph

Use the fewest layers that can satisfy the look:

| User intent | Typical composition |
|---|---|
| Toon, clay, watercolor-like wash | `LdrColor` or normal-form surface; optional stable screen marks; visible silhouettes only |
| Blueprint, ink, hatch, halftone | color/normal/ID image pass; deterministic screen/object marks; optional explicit features |
| Technical drawing | restrained surface; topology plus silhouettes; hidden strokes only when the user requests them |
| X-Ray | presentation shell; topology or continuation marks; explicitly bounded hidden-layer claim |
| Explicit full wireframe | no-fill or declared surface; every accepted triangle edge including diagonals; deliberate visible/rear policy |
| Diagnostic heatmap | guide-driven palette; no inferred feature lines unless they improve the stated task |

Do not add depth, segmentation, a line provider, or temporal history merely
because another preset uses it. Runtime-validate every requested guide and keep
the app live when it is absent.

## Shared Rules Across Looks

- Keep all USD and 3D rendering in OVRTX. The browser displays the final stream.
- Preserve user USD files; style state, temporary lights, and render products
  belong to viewer-owned state.
- Keep surface treatment, screen-space AOV accents, and explicit geometry
  marks as independent passes. This lets a style gain or lose one layer without
  corrupting the others.
- Anchor procedural marks to stable screen pixels, object/world coordinates, or
  a deliberate monotonic phase. Never use frame-random noise.
- Rebuild camera-dependent marks from the camera pose committed for the frame.
  A cached overlay from an earlier pose is a visible latency bug, not a
  performance optimization.
- Route BREP and ordinary-mesh topology through separate providers behind the
  same line contract. Do not downgrade a BREP path to inferred mesh edges, or
  require BREP for arbitrary USD meshes.

## Scope And Fallback

Define one explicit style-scope predicate. Include render-visible objects and
passive supports by default; never use selection or animation membership as the
line-scope predicate. Exclude helper, guide, environment, and backdrop meshes
only through an explicit viewer configuration or inherited opt-out metadata.
Report the resulting included/excluded counts once per stage revision.

For every style, test at least: valid guides, each guide missing, no usable line
geometry, a scene reload, and a viewport/camera change. Fallbacks must be
visually coherent and remain live; `LdrColor` is preferable to a blank frame.

## Evidence

Capture one fixed frame for every declared layer, a moving camera proof for
camera-dependent marks, and a baseline-versus-style performance comparison.
For a named canonical style, use its recipe and validation profile. For a new
look, label the result `style_recipe_conformant` only after its written layer
plan, fallback matrix, and evidence all agree; do not call it a faithful
reproduction without a supplied visual reference and matching goldens.
