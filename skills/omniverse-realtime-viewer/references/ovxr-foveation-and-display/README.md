# XR display and preview policy

Read [library foveation-and-display and frame-readback](../dependencies/ovxr.md) for settings, enums, negotiated features and domains.

Headset quality and desktop preview are separate application choices. Mirror resize must not change eye products. Downscale valid capture with bounded aspect-preserving GPU work; do not claim arbitrary SDK desktop sizing.

Validate disabled and warped modes independently on the package/runtime/client. AUTO is a request. A working disabled preview does not prove warped capture; preserve requested headset mode and show unavailable when mapping/capture is missing.

Do not inversely warp SDK-unwarped captures. Warped capture requires matching source-frame/per-eye metadata. Mirror output does not establish the submission domain.

[Foveation and edge math](../ovxr-composition-ui/foveation-and-edge-math.md) owns compositor domain gates. Missing vendor mapping blocks warped composition; an explicitly selected rectilinear baseline proves only that baseline. Expose supported [update lanes](../ovxr-library-runtime/settings-and-lifetime.md) and verify effective output rather than queue acknowledgements.
