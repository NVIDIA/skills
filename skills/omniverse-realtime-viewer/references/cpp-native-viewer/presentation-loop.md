# C++ Presentation And Render Loop

## Owned Frame Transfer

The runtime owner CPU-maps `LdrColor`, copies its pixels, unmaps, and destroys
the step result before publishing the owned frame. It never calls OpenGL.
Respect the DLPack byte offset and strides; mapped pointers never cross the
thread boundary.

```cpp
#include <cstdint>
#include <vector>

struct DisplayFrame {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgba;
};

static bool copyRgba(DisplayFrame& frame, const DLTensor& tensor)
{
    if (!tensor.data || tensor.device.device_type != kDLCPU ||
        tensor.ndim != 3 || !tensor.shape || tensor.dtype.code != kDLUInt ||
        tensor.dtype.bits != 8 || tensor.dtype.lanes != 1 ||
        tensor.shape[0] <= 0 || tensor.shape[1] <= 0 || tensor.shape[2] != 4)
        return false;
    const int height = static_cast<int>(tensor.shape[0]);
    const int width = static_cast<int>(tensor.shape[1]);
    const int64_t row = tensor.strides ? tensor.strides[0] : width * 4;
    const int64_t pixel = tensor.strides ? tensor.strides[1] : 4;
    const int64_t channel = tensor.strides ? tensor.strides[2] : 1;
    if (row <= 0 || pixel <= 0 || channel <= 0) return false;
    const auto* source = static_cast<const std::uint8_t*>(tensor.data) + tensor.byte_offset;
    frame.width = width;
    frame.height = height;
    frame.rgba.resize(static_cast<size_t>(width) * height * 4);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < 4; ++c)
                frame.rgba[(static_cast<size_t>(y) * width + x) * 4 + c] =
                    source[y * row + x * pixel + c * channel];
    return true;
}
```

Publish only a successfully copied complete frame, with its stage generation
and dimensions, into a single latest-frame slot. The UI takes shared ownership
or moves that buffer under a short lock, then releases the lock before upload.

## Native Frame Order

For each changed scene state, write through OVStage at ordinal `N`, wait for
writes, and advance/wait the write floor to `N`. Before rendering, call and wait
`ovrtx_update_from_stage(renderer, N)`. The C function
`ovrtx_step_with_stage()` requires this update; unlike Python `Renderer.step()`,
it does not supply the high-level wrapper's automatic update.

After the update, apply dirty native outline/pickability state, then enqueue a
pending pick. Population can recreate renderer-owned prim state, so retained
selection must be dirty after a same-scene replacement and applied here, once
the new population is visible. A pick decoded after this frame updates selection
intent for the next frame.

The helper below assumes the update has completed and optional renderer-native
commands have been enqueued. It performs one step and maps the display output;
decode `OVRTX_RENDER_VAR_PICK_HIT` in this same result iteration when a query was
queued (see `interaction-features.md`).

```cpp
static bool sameString(ovx_string_t value, const char* literal)
{
    const std::string expected(literal);
    return value.ptr &&
        value.length == expected.size() &&
        std::char_traits<char>::compare(value.ptr, expected.data(), expected.size()) == 0;
}

static const DLTensor* firstTensor(const ovrtx_render_var_output_t& output)
{
    if (!output.tensors || output.num_tensors == 0) return nullptr;
    return output.tensors[0].dl;
}

static bool stepRenderAndCopy(
    ovrtx_renderer_t* renderer,
    const std::string& renderProductPath,
    double deltaSeconds,
    ovstage_ordinal_t ordinal,
    DisplayFrame& frameCopy)
{
    frameCopy = {};
    bool copied = false;
    const ovx_string_t productPath = toOvxString(renderProductPath);
    const ovrtx_render_product_set_t products = {&productPath, 1};

    ovrtx_step_result_handle_t stepHandle = OVRTX_INVALID_HANDLE;
    const ovrtx_enqueue_result_t step =
        ovrtx_step_with_stage(renderer, products, deltaSeconds, ordinal, &stepHandle);
    if (!ok(step) || stepHandle == OVRTX_INVALID_HANDLE) {
        printLastOvrtxError("Failed to step OVRTX");
        return false;
    }

    ovrtx_render_product_set_outputs_t outputs = {};
    const ovrtx_result_t results =
        ovrtx_fetch_results(renderer, stepHandle, ovrtx_timeout_infinite, &outputs);
    if (!ok(results) || outputs.status != OVRTX_EVENT_COMPLETED) {
        printLastOvrtxError("Failed to fetch OVRTX results");
        ovrtx_destroy_results(renderer, stepHandle);
        return false;
    }

    for (size_t productIndex = 0; productIndex < outputs.output_count; ++productIndex) {
        const auto& product = outputs.outputs[productIndex];
        for (size_t frameIndex = 0; frameIndex < product.output_frame_count; ++frameIndex) {
            const auto& frame = product.output_frames[frameIndex];
            for (size_t varIndex = 0; varIndex < frame.render_var_count; ++varIndex) {
                const auto& var = frame.output_render_vars[varIndex];
                // Compare output identity with the full authored RenderVar path.
                // `source_name` classifies the AOV; it is not the map key.
                if (!sameString(var.render_var_path, (renderProductPath + "/LdrColor").c_str())) continue;

                ovrtx_map_output_description_t mapDesc = {};
                mapDesc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;
                ovrtx_render_var_output_t mapped = {};
                const ovrtx_result_t map =
                    ovrtx_map_render_var_output(renderer, var.output_handle,
                        &mapDesc, ovrtx_timeout_infinite, &mapped);
                if (ok(map) && mapped.status == OVRTX_EVENT_COMPLETED) {
                    if (const DLTensor* tensor = firstTensor(mapped)) {
                        copied = copyRgba(frameCopy, *tensor);
                    }
                }
                if (mapped.map_handle != OVRTX_INVALID_HANDLE) {
                    ovrtx_unmap_render_var_output(renderer, mapped.map_handle, {});
                }
            }
        }
    }

    ovrtx_destroy_results(renderer, stepHandle);
    return copied;
}
```

Use scope guards in the app to unmap and destroy results on error/exception
paths as well. Publish an error if fetch/map/copy fails instead of reporting a
new ready frame. Never step twice just to decode the pick for a displayed frame.

## UI Presentation And Input

Use the selected toolkit's event loop and pixel/image presentation API to show
the latest complete owned frame independently of SDK progress. Keep toolkit
objects on their required thread; copy pixels before releasing the SDK map.
The viewport's actual displayed image rectangle owns coordinate conversion,
including aspect ratio, letterboxing and display scaling. Toolkit choice does
not change native picking or renderer ownership.

For the GLFW/ImGui example, poll GLFW, build ImGui controls, and present the
latest frame. Create
and resize the GL texture on this thread, set `GL_UNPACK_ALIGNMENT` to `1`, and
upload the packed RGBA buffer with `glTexSubImage2D(..., GL_RGBA,
GL_UNSIGNED_BYTE, frame.rgba.data())`. `LdrColor` is RGBA; do not swap red/blue.

In that example, fit `ImGui::Image()` inside the available content rect while preserving the
rendered aspect ratio. Retain that actual image rectangle for picking and
camera input. Reject letterbox clicks; convert accepted image coordinates into
the render product's normalized coordinates. Use the same top-left orientation
for display and input, and account for GLFW window versus framebuffer scaling.

Route input through the viewport only when the image owns it. Text editing and
modal controls consume their events, and gizmo handles get first refusal before
camera or picking. Track a pressed drag until release even outside the image;
clear drag state when focus is lost. See `viewer-input-routing`.

Mark readiness only after the first successfully rendered owned frame has been uploaded and
swapped while the UI can accept input. During scene preparation continue to
present the active scene and loading/error status. An SDK wait remains on the
runtime owner; UI polling and normal-close intent remain responsive.
