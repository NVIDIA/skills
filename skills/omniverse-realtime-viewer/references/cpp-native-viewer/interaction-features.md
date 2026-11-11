# C++ Interaction Features

Apply these SDK contracts through the selected native toolkit. ImGui/GLFW
input and widget snippets are examples; adapt event ownership and coordinates
to the chosen shell without adding a second UI dependency.

## Camera And Input State

Use `camera-controls` for orbit/pan/zoom and live-bounds fitting. Initialize the
camera from the scene's bounds, units and up-axis; preserve that convention in
navigation and fitting. UI events become bounded owner-thread commands through
`viewer-input-routing`. The owner updates camera state and writes its local
`omni:xform` through OVStage before publication. For a root-level session camera,
that local matrix is also its world matrix.

## Live Stage Writes

Create a query for the target prim through OVStage's own path dictionary and
`ovstage_query_from_path_list()` (see `stage-queries`). Reuse valid queries within
a stage generation and release them before replacement. A single-prim query can
write a camera/local transform as one float64 matrix element:

```cpp
#include <array>
using Mat4 = std::array<double, 16>;

static bool writeLocalMatrix(
    ovstage_instance_t* stage, ovstage_query_handle_t query,
    const Mat4& matrix, ovstage_ordinal_t ordinal)
{
    int64_t shape = 1;
    DLTensor tensor{};
    tensor.data = const_cast<double*>(matrix.data());
    tensor.device = {kDLCPU, 0};
    tensor.ndim = 1;
    tensor.dtype = {kDLFloat, 64, 16};
    tensor.shape = &shape;
    ovstage_write_data_t data{};
    data.tensors = &tensor;
    data.tensor_count = 1;
    data.semantic = OVSTAGE_SEMANTIC_MATRIX;
    return waitStageOperation(stage, ovstage_write_attribute(
        stage, query, {0, literal_to_ovx_string("omni:xform")}, ordinal,
        data, OVSTAGE_PRIM_MODE_UPSERT));
}
```

The wait keeps the stack-backed tensor alive until the write finishes; publish
the batch's ordinal only after all writes succeed. Existing populated
`omni:xform` values are local transforms. For world edits, read the live ancestor
chain and reset-stack state, then compute the local result with the effective
parent inverse. Preserve accepted local transforms across session recomposition.
Use `prim-transform-safety` and `transform-manipulator` for this math; do not
flatten a subtree into world matrices or create competing transform owners.

## Native Picking

For click selection, enqueue a native pick query before the step that should
produce the pick result. Read `OVRTX_RENDER_VAR_PICK_HIT` after that same step.

```cpp
struct PendingPick {
    bool pending = false;
    int left = 0;
    int top = 0;
    int right = 0;
    int bottom = 0;
};

static bool enqueuePick(
    ovrtx_renderer_t* renderer,
    const std::string& renderProductPath,
    PendingPick& pendingPick,
    int renderWidth,
    int renderHeight)
{
    if (!pendingPick.pending || renderWidth <= 0 || renderHeight <= 0) return false;

    const ovrtx_pick_query_desc_t desc = {
        toOvxString(renderProductPath),
        static_cast<float>(pendingPick.left) / renderWidth,
        static_cast<float>(pendingPick.top) / renderHeight,
        static_cast<float>(pendingPick.right) / renderWidth,
        static_cast<float>(pendingPick.bottom) / renderHeight,
        0,
    };
    pendingPick = {};

    const ovrtx_enqueue_result_t pick = ovrtx_enqueue_pick_query(renderer, &desc);
    if (!ok(pick)) {
        printLastOvrtxError("Failed to enqueue pick query");
        return false;
    }
    return true;
}
```

Decode the pick-hit output by checking params and resolving `primPath` IDs
through the path dictionary.

```cpp
#include <ovx/path_dictionary/path_dictionary.h>
#include <ovx/path_dictionary/path_dictionary_helper.h>
#include <ovx/path_dictionary/path_dictionary_utils.h>

#include <algorithm>
#include <cstring>
#include <vector>

static const DLTensor* findTensor(const ovrtx_render_var_output_t& output, const char* name)
{
    for (size_t i = 0; output.tensors && i < output.num_tensors; ++i) {
        if (output.tensors[i].name &&
            sameString(*output.tensors[i].name, name)) {
            return output.tensors[i].dl;
        }
    }
    return nullptr;
}

static const DLTensor* findParam(const ovrtx_render_var_output_t& output, const char* name)
{
    for (size_t i = 0; output.params && i < output.num_params; ++i) {
        if (sameString(output.params[i].name, name)) return &output.params[i].dl;
    }
    return nullptr;
}

static bool readU64(const DLTensor& tensor, size_t index, std::uint64_t& value)
{
    if (!tensor.data || tensor.dtype.lanes != 1) return false;
    const auto* base = static_cast<const std::uint8_t*>(tensor.data) + tensor.byte_offset;
    const size_t bytes = tensor.dtype.bits / 8;
    const int64_t stride = tensor.strides ? tensor.strides[0] : 1;
    const auto* ptr = base + index * static_cast<size_t>(stride) * bytes;

    value = 0;
    if (tensor.dtype.code == static_cast<std::uint8_t>(kDLUInt)) {
        std::memcpy(&value, ptr, std::min(bytes, sizeof(value)));
        return true;
    }
    return false;
}

static std::string resolvePrimPathId(ovrtx_renderer_t* renderer, ovx_primpath_t pathId)
{
    path_dictionary_instance_t dictionary = {};
    if (!ok(ovrtx_get_path_dictionary(renderer, &dictionary))) {
        printLastOvrtxError("Failed to get OVRTX path dictionary");
        return {};
    }

    std::vector<ovx_token_t> tokenBuffer(256);
    ovx_token_t* tokensPerPath[] = {nullptr};
    size_t tokenCounts[] = {0};
    size_t pathsProcessed = 0;

    ovx_api_result_t tokenResult = path_dictionary_get_tokens_from_paths(
        &dictionary,
        &pathId,
        1,
        tokenBuffer.data(),
        tokenBuffer.size(),
        tokensPerPath,
        tokenCounts,
        &pathsProcessed);
    if (tokenResult.status != OVX_API_SUCCESS || pathsProcessed != 1 || !tokensPerPath[0]) {
        return {};
    }

    std::vector<ovx_string_t> tokenStrings(tokenCounts[0]);
    ovx_api_result_t stringResult = path_dictionary_get_strings_from_tokens(
        &dictionary,
        tokensPerPath[0],
        tokenCounts[0],
        tokenStrings.data());
    if (stringResult.status != OVX_API_SUCCESS) {
        return {};
    }

    std::string path;
    for (ovx_string_t token : tokenStrings) {
        std::string segment = fromOvxString(token);
        if (segment.empty()) continue;
        if (segment.front() != '/') path.push_back('/');
        path += segment;
    }
    return path.empty() ? "/" : path;
}

static std::vector<std::string> decodePickPaths(
    ovrtx_renderer_t* renderer,
    const ovrtx_render_var_output_t& pickOutput)
{
    std::uint64_t magic = 0;
    std::uint64_t version = 0;
    std::uint64_t hitCount = 0;
    const DLTensor* magicParam = findParam(pickOutput, "magic");
    const DLTensor* versionParam = findParam(pickOutput, "version");
    const DLTensor* hitCountParam = findParam(pickOutput, "hitCount");
    if (!magicParam || !versionParam || !hitCountParam ||
        !readU64(*magicParam, 0, magic) ||
        !readU64(*versionParam, 0, version) ||
        !readU64(*hitCountParam, 0, hitCount)) {
        return {};
    }
    if (magic != OVRTX_PICK_HIT_MAGIC || version != OVRTX_PICK_HIT_VERSION) {
        return {};
    }

    const DLTensor* primPathTensor = findTensor(pickOutput, "primPath");
    if (!primPathTensor || !primPathTensor->shape) return {};

    const size_t count =
        std::min(static_cast<size_t>(hitCount), static_cast<size_t>(primPathTensor->shape[0]));

    std::vector<std::string> paths;
    for (size_t i = 0; i < count; ++i) {
        std::uint64_t id = 0;
        if (!readU64(*primPathTensor, i, id) || id == 0) continue;
        std::string path = resolvePrimPathId(renderer, static_cast<ovx_primpath_t>(id));
        if (!path.empty() && std::find(paths.begin(), paths.end(), path) == paths.end()) {
            paths.push_back(path);
        }
    }
    return paths;
}
```

In the render loop, handle pick output beside `LdrColor`:

```cpp
const bool pickQueued = enqueuePick(renderer, renderProductPath, pendingPick, renderWidth, renderHeight);

// After ovrtx_step_with_stage() and ovrtx_fetch_results():
if (pickQueued && sameString(var.render_var_path, OVRTX_RENDER_VAR_PICK_HIT)) {
    ovrtx_map_output_description_t mapDesc = {};
    mapDesc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;
    ovrtx_render_var_output_t mapped = {};
    if (ok(ovrtx_map_render_var_output(renderer, var.output_handle,
            &mapDesc, ovrtx_timeout_infinite, &mapped)) &&
        mapped.status == OVRTX_EVENT_COMPLETED) {
        std::vector<std::string> picked = decodePickPaths(renderer, mapped);
        setSelectedPrim(picked.empty() ? std::string{} : picked.front());
    }
    if (mapped.map_handle != OVRTX_INVALID_HANDLE) {
        ovrtx_unmap_render_var_output(renderer, mapped.map_handle, {});
    }
}
```

Treat `left` and `top` as inclusive, `right` and `bottom` as exclusive. A single
click is a `1x1` pixel rectangle: `{x, y, x + 1, y + 1}`. Divide its
edges by the current render-product width and height before enqueueing; the
OVRTX query takes top-left-origin normalized coordinates in `[0, 1]`, not pixels.

## Selection Outline

Enable outlines in renderer config, then call
`ovrtx_set_selection_outline_group_strings()` with canonical prim paths. Group `0` clears
the outline; group `1` is primary selection. Queue this state on the owner after
the waited stage update and before the next render step. Expand a selected Xform
to its drawable descendants when the UI selects containers.

```cpp
static void setSelectionOutline(
    ovrtx_renderer_t* renderer,
    const std::string& previousPath,
    const std::string& nextPath)
{
    if (!previousPath.empty()) {
        const ovx_string_t path = toOvxString(previousPath);
        const uint8_t group = 0;
        waitForOperation(renderer,
            ovrtx_set_selection_outline_group_strings(renderer, &path, 1, &group),
            "Failed to clear selection outline");
    }
    if (!nextPath.empty()) {
        const ovx_string_t path = toOvxString(nextPath);
        const uint8_t group = 1;
        waitForOperation(renderer,
            ovrtx_set_selection_outline_group_strings(renderer, &path, 1, &group),
            "Failed to set selection outline");
    }
}
```

For bulk updates, pass parallel string-path and group arrays. When renderer
path IDs are already available, use `ovrtx_set_selection_outline_group()` with
IDs from that renderer's own dictionary.
Do not replace this renderer-owned selection state with a generic attribute
write.

## Optional Material Effects And Animation

Material effects write to OVStage at the owner's next ordinal. Resolve actual
EffectLayer targets and save their current `inputs:Fader` values before changing
them; do not hard-code scene-specific paths or create unknown shader inputs.
For a single existing target, use a scalar float32 tensor (`dtype = {kDLFloat,
32, 1}`, shape `{1}`) with `ovstage_write_attribute()` and
`OVSTAGE_PRIM_MODE_UPSERT`, after confirming the target and input exist; wait,
then publish. Restore the saved value
when no selected prim still uses that target. See `prim-pick-effects` for shared
materials and app-owned effect setup.

Selection animation uses the same OVStage local-transform writer and publication
clock. Capture a known local baseline, compose a reversible offset in parent
space, and restore it on completion/deselection. Coordinate animation with manual
editing so they never overwrite each other; do not substitute an identity matrix
for an unknown baseline. See `selection-animation` and `prim-transform-safety`.
