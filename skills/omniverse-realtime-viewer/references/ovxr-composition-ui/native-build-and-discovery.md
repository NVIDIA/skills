# Native compositor build and discovery

The application must implement the native layer; the supplied libraries do not
ship the custom compositor described here. This route specifies its build and
discovery seams, without requiring a sample checkout. Native algorithms live in
[library construction](library-only-compositor.md),
[projection math](projection-and-depth.md) and
[passes/synchronization](compositor-passes-and-sync.md).

## Inputs and outputs

Inputs are matching OVXR/OVRTX/OVStage headers and native libraries, a C++
compiler, OpenXR headers including loader negotiation, Vulkan headers/loader,
and a GLSL-to-SPIR-V compiler. Obtain runtime packages through
[dependencies](../dependencies/nvidia-runtime.md#ovxr). Keep platform-specific
compiler/runtime and loader paths in application build configuration, not the
published skill. Lock the raster worker separately through
[Skia/Yoga dependencies](../dependencies/skia-yoga.md).

Implement these application-owned compilation units (names are examples):

- `layer`: loader negotiation, downstream dispatch tables, session/swapchain
  bookkeeping, deferred release and `xrEndFrame` interception.
- `compositor`: Vulkan resources, bounded slots, attachment passes and teardown.
- `mesh`: per-eye clipping/projection, tube/analytic tool packing and scissor.
- `bridge`: copied/versioned surface and raster ABI, mapping/telemetry getters;
  no caller-owned OVRTX step-result bridge is required by current OVXR submission.
- Vertex/fragment shaders compiled to SPIR-V during the build; embedded arrays
  or installed shader files must derive from those exact sources.

Export C ABI functions explicitly on Windows and ELF platforms; hidden default
visibility must not hide loader negotiation or required application exports.
Use `extern "C"` and the calling conventions from the supplied headers. Compile
with strict warnings and test struct size/alignment against the Python/C consumer.
Use documented CMake package targets when supplied, not guessed library names.
The layer normally resolves downstream OpenXR functions through dispatch rather
than calling its own intercepted exports recursively.

## Manifest and process-local activation

Generate an explicit API-layer manifest beside the built library. Example app
identity (replace with your own; this is not an NVIDIA-provided layer):

```json
{
  "file_format_version": "1.0.0",
  "api_layer": {
    "name": "XR_APILAYER_APP_spatial_ui",
    "library_path": "./app_spatial_ui.dll",
    "api_version": "1.0",
    "implementation_version": "1",
    "description": "Application-owned spatial UI compositor"
  }
}
```

Generate the actual platform filename (`.dll` versus `.so`) from the build;
resolve it relative to the manifest, not the caller's working directory. Use
the loader's process-local `XR_API_LAYER_PATH` discovery path and
`XR_ENABLE_API_LAYERS` selection before creating the OpenXR instance. Preserve
other requested layers rather than silently replacing them. Avoid machine-wide
registry/runtime changes for a local test. The loader's required negotiation
export is `xrNegotiateLoaderApiLayerInterface`; implement it using the installed
`openxr_loader_negotiation.h`, including interface/version compatibility checks.

At instance creation advance the next-info chain exactly once and retain its
downstream functions. Keep dispatch keyed by owning instance/session/device,
not a single unqualified global pointer. Destroy only application-owned objects.

## Staged proof

1. Build and run CPU math/ABI tests without a headset. Confirm generated shader
   bytes, symbol visibility and manifest library resolution.
2. Enumerate the layer through the loader and create an instance with it enabled.
   Enumeration alone is not proof it intercepts the live session.
3. Report loaded library identity and an intercepted frame/session counter from
   the target process, then verify one unchanged forwarded projection with UI off.
4. Capture actual color/depth swapchain usage, format, samples, array extent
   and negotiated depth-extension/submission policy. Require color/depth
   attachment usage before enabling one fixed LOCAL panel, then stereo depth
   and input incrementally. Unsupported usage is a failed feature gate, not a
   passing compositor.
5. Exercise missing exports, malformed ABI records, failed image matching,
   busy slots and retirement before adding more surfaces.

Ship manifest, native binary, shaders (if not embedded), ABI declaration and
matching build provenance together. Keep the retained worker/font dependencies
separate but versioned. Do not distribute developer machine paths or substitute
placeholder runtime/shader-cache files to make packaging appear complete.
