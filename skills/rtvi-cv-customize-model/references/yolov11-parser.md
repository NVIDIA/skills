# YOLOv11 DeepStream Parser

Use this reference when creating or debugging the custom bbox parser for the
YOLOv11 example in the CV customization skill.

## Contents

- [Inspect the model output](#inspect-the-model-output)
- [Parser ABI and decode pattern](#parser-abi-and-decode-pattern)
- [Compile the parser](#compile-the-parser)

## Inspect the model output

Confirm the output blob name, shape, coordinate space, and whether the graph
already applies NMS before writing the parser. The Ultralytics YOLOv11 COCO
export used by this skill has:

- Blob name: `output0`
- Shape: `[batch, 84, 8400]`
- Layout: transposed; channel `v`, anchor `i` is
  `data[v * numAnchors + i]`
- Coordinates: `cx`, `cy`, `w`, and `h` in pixel space
- No objectness score and no in-graph NMS

This differs from darknet YOLO (`[batch, 255, H, W]`) and post-NMS
`[N, 6]` outputs. Inspect every replacement model instead of assuming it uses
the YOLOv11 layout.

For an instance-segmentation or detection-plus-mask model, complete
[segmentation-model-contract.md](segmentation-model-contract.md) before
implementing parser or handoff changes.

## Parser ABI and decode pattern

Create
`${VSS_PROFILE_DIR}/deepstream/custom_parser/nvdsparseyolov11.cpp`. Keep the
same exported symbol in the function declaration, prototype-check macro, and
the nvinfer `parse-bbox-func-name` setting:

```cpp
extern "C" bool NvDsInferParseCustomYoloE(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);
```

The core decode pattern for the transposed output is:

```cpp
for (int i = 0; i < numAnchors; i++) {
    float cx = data[0 * numAnchors + i];
    float cy = data[1 * numAnchors + i];
    float bw = data[2 * numAnchors + i];
    float bh = data[3 * numAnchors + i];

    float maxScore = 0.0f;
    int maxClass = 0;
    for (int c = 0; c < numClasses; c++) {
        float score = data[(4 + c) * numAnchors + i];
        if (score > maxScore) {
            maxScore = score;
            maxClass = c;
        }
    }

    float threshold =
        detectionParams.perClassPreclusterThreshold[maxClass];
    if (maxScore < threshold) {
        continue;
    }

    NvDsInferParseObjectInfo obj;
    obj.left = std::max(0.0f, cx - bw * 0.5f);
    obj.top = std::max(0.0f, cy - bh * 0.5f);
    obj.width = std::min(netW, cx + bw * 0.5f) - obj.left;
    obj.height = std::min(netH, cy + bh * 0.5f) - obj.top;
    obj.detectionConfidence = maxScore;
    obj.classId = static_cast<unsigned int>(maxClass);
    objectList.push_back(obj);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloE);
```

This is the decode core, not a complete translation unit. The source file
must also include the DeepStream parser headers, validate tensor dimensions
and types, obtain the output buffer, and define `numAnchors`, `numClasses`,
`netW`, and `netH`.

## Compile the parser

Build the parser in
`${VSS_PROFILE_DIR}/Dockerfiles/perception.Dockerfile`:

```dockerfile
FROM nvcr.io/nvidia/vss-core/vss-rt-cv:3.2.1
# vss-rt-cv:3.2.1 ends as a non-root user.
USER root
COPY ./deepstream/custom_parser/nvdsparseyolov11.cpp \
     /tmp/nvdsparseyolov11.cpp
RUN DS_INC=$(find /opt/nvidia/deepstream -maxdepth 3 -name "includes" \
        -path "*/sources/includes" -type d | sort | tail -1) \
    && CUDA_INC=$(dirname "$(find -L /usr/local/cuda/ -maxdepth 4 \
        -name cuda_runtime_api.h -type f 2>/dev/null | head -1)") \
    && { test -f "${DS_INC}/nvdsinfer_custom_impl.h" \
        || { echo "ERROR: DeepStream includes not found"; exit 1; }; } \
    && { test -f "${CUDA_INC}/cuda_runtime_api.h" \
        || { echo "ERROR: cuda_runtime_api.h not found"; exit 1; }; } \
    && mkdir -p /opt/deepstream-yolo \
    && g++ -std=c++11 -shared -fPIC -O2 \
        -I"${DS_INC}" \
        -I"${CUDA_INC}" \
        -o /opt/deepstream-yolo/libnvdsparseyolov11.so \
        /tmp/nvdsparseyolov11.cpp \
    && rm /tmp/nvdsparseyolov11.cpp
```

`nvdsinfer_custom_impl.h` transitively includes CUDA headers, so discover both
DeepStream and CUDA include directories at build time. Do not rely on the
unversioned DeepStream symlink or hardcode a versioned path. `USER root` must
precede writes to system paths.

After building, verify the symbol before deployment:

```bash
nm -D /opt/deepstream-yolo/libnvdsparseyolov11.so \
  | grep NvDsInferParseCustomYoloE
```
