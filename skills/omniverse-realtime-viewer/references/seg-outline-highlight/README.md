# Segmentation Outline Highlight

## Selection Highlight Path

This skill covers a legacy/special-purpose custom post-process path for
selection outlines composed from `InstanceSegmentationSD` with Warp CUDA after
copying color into a BGRA stream buffer.

Current OVRTX viewers should use native selection outlines and groups for normal
selection feedback:

- Use `selection-feedback` for selected-object visuals.
- Use `native-picking-selection` for the combined pick/query/selection-group
  workflow.
- Use `object-selection` for selected-path state and click/marquee routing.

New generated apps should not create `seg_outline.py`, selected-instance-ID
outline kernels, or Warp outline systems for ordinary selected-object feedback.

## Custom Overlay Use

Use this pattern only when the user explicitly asks for a custom post-process
overlay that cannot be expressed with native selection groups. In that case:

1. Request the required segmentation render var for the custom overlay.
2. Keep the custom overlay independent from native picking.
3. Composite after the active color/AOV has been copied into the display buffer.
4. Treat segmentation IDs as frame-local and scene-local.
5. Clear all custom overlay state on scene switch.

Do not use this custom overlay path as a fallback for native OVRTX selection
outlines. Keep it only for explicit compatibility needs or custom visual
effects that native selection groups cannot express.

See also: `selection-feedback`, `native-picking-selection`, `object-selection`,
`aov-switching`, `streaming-server`.
