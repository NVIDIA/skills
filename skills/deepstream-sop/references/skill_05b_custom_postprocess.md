# § 5b — Custom C++ Postprocess Plugin

> **Reference files to copy** (do NOT regenerate):
> - `nvds_custom_postprocess_tensor_reference.cpp` → `nvds_action_detector/custom_postprocess/nvds_custom_postprocess_tensor.cpp`
> - `Makefile_custom_postprocess_reference` → `nvds_action_detector/custom_postprocess/Makefile`
>
> **Pipeline stage**: Stage 1 post-processing — converts DDM raw output tensor into
> NvDsObjectMeta entries that InferenceOutputTensorParser (§ 3) can read

---

## Why This Must Be Copied Verbatim

`nvinferserver` requires `libnvds_custom_postprocess_tensor.so`,
compiled from C++ during Docker build (`RUN cd custom_postprocess/ && make`).
The `.cpp` and `Makefile` **must exist** in the source tree or Docker build fails.

The C++ plugin uses DeepStream's `nvdsinferserver::IOptions` API which has distinct
methods for different data types:
- `getObj()` for single-pointer types (`GstBuffer*`, `NvBufSurface*`, `NvDsBatchMeta*`)
- `getValueArray()` for vector types (`std::vector<NvDsFrameMeta*>`, `std::vector<uint64_t>`)
- `getInt()` for integer values (`unique_id`)

Using the wrong method (e.g. `getValueArray()` on a `GstBuffer*`) causes a template
deduction failure at compile time.

---

## Dockerfile Requirements (both stages needed)

```dockerfile
# Stage 1: Extract DeepStream headers (needed by Makefile -I flags)
FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch AS ds_dev

# Stage 2: main build
FROM ${BASE_IMAGE} AS base
...
# Copy DS headers so Makefile can find infer_custom_process.h, nvdsmeta.h, etc.
COPY --from=ds_dev /opt/nvidia/deepstream/deepstream/sources/includes \
    /opt/nvidia/deepstream/deepstream/sources/includes

# Copy application source (must include custom_postprocess/ with .cpp + Makefile)
RUN --mount=type=bind,source=.,target=/tmp/ds_sop \
    cd /tmp/ds_sop && docker/copy_sources.sh ./ /opt/nvidia/nvds_sop/

# Compile the plugin — MUST come after source copy
RUN cd /opt/nvidia/nvds_sop/nvds_action_detector/custom_postprocess/ && make
```

---

## C++ API Pattern (CRITICAL — build will fail if wrong)

**DO NOT** include `gstnvdsmeta.h` — it transitively pulls `<gst/gst.h>` which is
unavailable under the Makefile's include paths, causing a fatal compilation error.
Use only DeepStream inference headers + forward-declare `GstBuffer`:

```cpp
#include "infer_custom_process.h"   // IInferCustomProcessor, IBatchArray, IOptions
#include "nvbufsurface.h"           // NvBufSurface, NvBufSurfaceParams
#include "nvdsmeta.h"               // NvDsBatchMeta, NvDsFrameMeta, NvDsObjectMeta

typedef struct _GstBuffer GstBuffer;   // forward-declare — do NOT #include <gst/gst.h>
```

---

## IOptions API Method Selection

This is the #1 compilation failure cause:

| Data to retrieve | IOptions method | Type signature |
|-----------------|----------------|----------------|
| `GstBuffer*` | `getObj()` | `getObj(OPTION_NVDS_GST_BUFFER, gstBuf)` |
| `NvBufSurface*` | `getObj()` | `getObj(OPTION_NVDS_BUF_SURFACE, bufSurf)` |
| `NvDsBatchMeta*` | `getObj()` | `getObj(OPTION_NVDS_BATCH_META, batchMeta)` |
| `vector<NvDsFrameMeta*>` | `getValueArray()` | `getValueArray(OPTION_NVDS_FRAME_META_LIST, frameMetaList)` |
| `vector<uint64_t>` | `getValueArray()` | `getValueArray(OPTION_NVDS_SREAM_IDS, streamIds)` |
| `int64_t` | `getInt()` | `getInt(OPTION_NVDS_UNIQUE_ID, unique_id)` |

Using `getValueArray()` on a single-pointer type causes:
`template argument deduction/substitution failed: mismatched types 'std::vector<_Tp>' and 'GstBuffer*'`

---

## What the Plugin Does

`nvds_custom_postprocess_tensor.cpp` implements `IInferCustomProcessor`:

```
DDM output tensor (float scores)
        │
        ▼
inferenceDone() — for each score in the batch:
  ├── Creates NvDsObjectMeta:
  │     object_id = frame_num - OFFSET + i
  │         (OFFSET = FRAMES_PER_SIDE + SEQUENCE_BATCH - 1)
  │     confidence = boundary score (float)
  │     class_id = 0 ("boundary")
  │
  ├── Attaches object metas to NvDsFrameMeta
  │     → InferenceOutputTensorParser (§ 3) reads via frame_meta.object_items
  │
  ├── Uses hasValue() guard before each getObj()/getValueArray() call
  │
  └── Acquires/releases batchMeta lock around nvds_add_obj_meta_to_frame()
```
