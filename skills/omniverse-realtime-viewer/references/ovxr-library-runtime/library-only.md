# Bootstrap an application-owned XR viewer

Resolve [dependencies and upstream samples](../dependencies/ovxr.md). Use the
selected language's minimal XR README and source for setup through first
submitted frame; use the upstream application-flow, context-creation and
stage-publication skills when adding viewer co-writers. Do not build another
copy of that SDK bootstrap in these references.

The viewer supplies source-safe composition, canonical settings and one
publication coordinator for navigation, tools and simulation. Choose those
features through [the recipe](../xr-viewer-recipe/README.md), preserving
long-lived owners where the resolved SDK allows. Keep runtime/client readiness
separate from the product's interactive-ready state.

For an editor exposing user scale, configure
`ContextDesc.floor_aligned_anchor=False` before creation and follow
[stage-space spawn and scale policy](../xr-spatial-input/navigation.md).
That application choice is essential: the default floor-aligned policy removes
scale. Resolve camera anchors to stage-space heading/translation before using
full-transform mode, and verify the scale feature in the rendered view.
