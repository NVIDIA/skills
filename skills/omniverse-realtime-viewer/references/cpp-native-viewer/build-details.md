# C++ Build And Wait-Helper Details

Optional implementation detail for `project-build.md`. Use this when setting up
the build or operation helpers; the selected release's upstream example and
installed headers remain authoritative. Resume the current recipe increment
after the build works.

## GLFW/ImGui Build Example

Use the current upstream CMake package helpers for the paired OVRTX/OVStage
runtime, not a hand-written `find_library()` search over an SDK layout. Copy the
current `ovrtx.cmake` and `ovstage.cmake` helper from the upstream OVRTX C
minimal example into the project's `cmake/` directory, following the dependency
reference before choosing package versions. The remaining UI targets below
are specific to this example. For another toolkit, use its supported build
setup and retain the matching SDK loaders/runtime staging; do not fetch
GLFW/ImGui merely to follow this example.

```cmake
cmake_minimum_required(VERSION 3.20)

project(cpp_imgui_ovrtx_viewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG QUIET)

if(NOT glfw3_FOUND)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
    )
    FetchContent_MakeAvailable(glfw)
endif()

if(TARGET glfw)
    set(GLFW_TARGET glfw)
elseif(TARGET glfw3)
    set(GLFW_TARGET glfw3)
else()
    message(FATAL_ERROR "GLFW target was not found")
endif()

if(NOT DEFINED IMGUI_DIR AND DEFINED ENV{IMGUI_DIR})
    set(IMGUI_DIR "$ENV{IMGUI_DIR}")
endif()

if(IMGUI_DIR)
    set(imgui_SOURCE_DIR "${IMGUI_DIR}")
else()
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.91.9b
    )
    FetchContent_MakeAvailable(imgui)
endif()

add_library(imgui STATIC
    "${imgui_SOURCE_DIR}/imgui.cpp"
    "${imgui_SOURCE_DIR}/imgui_draw.cpp"
    "${imgui_SOURCE_DIR}/imgui_tables.cpp"
    "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp"
)
target_include_directories(imgui PUBLIC
    "${imgui_SOURCE_DIR}"
    "${imgui_SOURCE_DIR}/backends"
)
target_link_libraries(imgui PUBLIC ${GLFW_TARGET} OpenGL::GL)

# Copy current upstream helpers from the C minimal example into cmake/.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ovrtx)
include(ovstage)
ovrtx_fetch()
ovstage_fetch()

add_executable(cpp_imgui_ovrtx_viewer
    src/main.cpp
    src/runtime.cpp
    src/runtime.h
    src/camera.cpp
    src/camera.h
    src/session_layer.h
    src/ovrtx_helpers.h
)

target_link_libraries(cpp_imgui_ovrtx_viewer PRIVATE
    # Attached-stage viewers use one matching loader model for both packages.
    ovrtx::ovrtx
    ovstage::ovstage
    imgui
    Threads::Threads
    ${GLFW_TARGET}
    OpenGL::GL
)

# Stage each package's runtime alongside the executable. A static-loader design
# must use both static targets and pass both package roots at runtime as shown by
# the current upstream C minimal example.
ovrtx_setup_runtime(cpp_imgui_ovrtx_viewer)
ovstage_setup_runtime(cpp_imgui_ovrtx_viewer)
```

The helper stages the selected loader/runtime layout. Build and run normally:

```bash
cmake -S . -B build
cmake --build build -j
./build/cpp_imgui_ovrtx_viewer /absolute/path/to/scene.usd
```

Do not copy an old binary, manually set a broad loader path, or combine static
and shared-loader package models. Rebuild all OVRTX/OVStage C/C++ translation
units when upgrading the paired runtime.

## Common C API Helpers

Keep small helpers for string conversion, result checks, and operation waits.

```cpp
#include <ovrtx/ovrtx.h>
#include <ovstage/ovstage.h>
#include <ovrtx/ovrtx_config.h>
#include <ovrtx/ovrtx_types.h>

#include <cstdlib>
#include <iostream>
#include <string>

static ovx_string_t toOvxString(const std::string& value)
{
    return {value.c_str(), value.size()};
}

static std::string fromOvxString(ovx_string_t value)
{
    if (!value.ptr || value.length == 0) return {};
    return {value.ptr, value.length};
}

static bool ok(ovrtx_result_t result)
{
    return result.status == OVRTX_API_SUCCESS;
}

static bool ok(ovrtx_enqueue_result_t result)
{
    return result.status == OVRTX_API_SUCCESS;
}

static void printLastOvrtxError(const char* context)
{
    std::cerr << context;
    const std::string message = fromOvxString(ovrtx_get_last_error());
    if (!message.empty()) std::cerr << ": " << message;
    std::cerr << "\n";
}

static bool waitForOperation(
    ovrtx_renderer_t* renderer,
    ovrtx_enqueue_result_t op,
    const char* context)
{
    if (!ok(op)) {
        printLastOvrtxError(context);
        return false;
    }
    if (op.op_index == OVRTX_INVALID_HANDLE) {
        return true;
    }

    ovrtx_op_wait_result_t waitResult = {};
    const ovrtx_result_t wait =
        ovrtx_wait_op(renderer, op.op_index, ovrtx_timeout_infinite, &waitResult);
    if (!ok(wait) || waitResult.num_error_ops > 0) {
        printLastOvrtxError(context);
        return false;
    }
    return true;
}
```
