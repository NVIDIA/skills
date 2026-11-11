# C++ Native OVRTX Viewer

Use this path for a new C++ desktop USD viewer. Choose a native UI toolkit
that suits the requested interaction, platform and available dependencies.
Keep an existing app's selected shell; honor an explicitly requested toolkit.
For a small new viewer, choose a simple sufficient shell and proceed without
asking the user to select one. GLFW/ImGui below is a worked example; Qt or
another native toolkit can implement the same contracts. OVRTX renders all
3D; the selected toolkit presents its pixels. The application owns an attached OVStage,
its publication clock and renderer on one runtime thread; the UI thread owns
window events and presentation. File layout, widget library and pixel upload
mechanism are application choices; SDK ownership and lifetimes are contracts.

## Build In Requested Increments

Read the row for the current increment, then implement and check it before
loading the next group. A full editor, generic property reflection, material
effects, animation and extra menus are not prerequisites for a small viewer.

| Increment | Read and establish |
|---|---|
| Build and first displayed frame | `project-build.md`, then `renderer-session.md` and `presentation-loop.md`: one USD, viewer-owned camera/product, attached-stage publication, map/copy/unmap and an actual native-window frame. Keep the window responsive and verify normal close before adding panels. |
| Requested navigation | Camera/input and live-write sections of `interaction-features.md`; follow its camera/input links for the math or gesture contract you need. Verify orbit, pan and zoom on the actual frame. |
| Requested click selection | Native picking and selection sections of `interaction-features.md`: decode the GPU hit, resolve its ID with the renderer dictionary, and synchronize selection. |
| Requested scene tree and part details | The Runtime Stage Query Path / Native Tree Construction sections of `references/stage-hierarchy/README.md` and the decoding contract in `references/stage-attribute-reads/README.md`. Use the matching release's [C query/read example](https://github.com/NVIDIA-Omniverse/ovstage/blob/main/examples/c/queries/main.cpp) and installed population mapping. Start with readable path/type and the requested numeric properties; expand reflection only when needed. |
| Requested scene switching | Replacement/cleanup sections of `renderer-session.md`, then `validation.md`. Clear stale scene data and selection; test the new scene and switching back. |
| Additional controls, effects or animation | `references/viewer-control-patterns/README.md` for nontrivial controls; the optional section of `interaction-features.md` links effect/animation guidance. Read these when implementing those features. |

For SDK acquisition, begin at `references/dependencies/README.md`. Read the selected
upstream OVRTX/OVStage agent guidance and relevant skill entry points, then use
one matching native minimal example and the installed headers for the current
increment. Look up the exact functions, types and layouts you use; parallel
Python/Rust/other-shell implementations are not prerequisite reading. Detailed
snippets in this package are validation targets, not a substitute for the
installed release's public C contract. If a complete build/wait-helper example
is useful, `build-details.md` preserves it.

## Contracts To Preserve

- Keep population, writes, ordinals/write floors, renderer updates, picks,
  stepping and output mapping on one runtime owner. The selected toolkit's window and presentation APIs stay on
  their required UI thread; exchange bounded intent, copied state and latest owned frames.
- Use the attached-stage C APIs and paired loader model. Native OVRTX still
  owns picks and selection groups. Keep schema registration and any optional
  OpenUSD ABI/isolation boundary consistent with the selected release.
- Copy complete mapped output with its actual offset/strides, then unmap and
  destroy results on success and failure paths. No SDK pointer crosses to UI.
- Inspector metadata is typed data: path IDs, token IDs and token arrays need
  their owning Stage dictionary; numeric values stay numeric. Use the linked
  property-specific mapping rather than treating every unsigned value as a token.
  The linked C query example resolves derived prim types and schema tokens; the
  installed [population header](https://github.com/NVIDIA-Omniverse/ovstage/blob/main/include/ovstage/ovstage_population.h)
  owns property mappings. Read the corresponding C sections, not parallel
  Python/React inspector implementations.
- Preserve source USD. Tag scene-dependent work by generation and clear stale
  state on replacement. Check input ownership, click versus drag, stationary
  selection and clean close through `validation.md`.

Use `references/routing.md` only to select an additional capability or reconsider an
unspecified delivery method. Keep the first implementation proportional to the
user's requested features.
