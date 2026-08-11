# Segmentation Model Addendum

Use this addendum when a replacement detection model also emits masks, such as
an instance-segmentation model or a detection model with an auxiliary mask head.
The required output mask is a frame-level mask, not object-local masks. Keep the
main `rtvi-cv-customize-model` workflow as the source of truth for the detection
model swap: ONNX export, class IDs, label files, bbox parser, `nvinfer`, engine
build, and deployment wiring.

This addendum is not a semantic-segmentation guide. If the model does not emit
or pair with boxes, classes, and confidences, use a segmentation-specific
workflow instead of this detection customization skill.

## Required Invariants

Before writing parser code, make these decisions explicit:

- Output frames must preserve object data and frame-level mask data. Do not
  replace object output with mask-only output.
- Each kept object must still have bbox, class ID, confidence, and label data.
- The serialized or downstream mask must be a frame mask. Object-local or
  per-instance masks are intermediate data only; project or composite them into
  the frame mask before output.
- Masks come from model output tensors by default. A tracker may add object IDs
  or propagate metadata, but it is not the mask source unless the design says so.
- Boxes and the frame mask must use one declared canonical coordinate space,
  usually source-frame or mux-frame coordinates.
- The frame-mask encoding must be declared before implementation: dimensions,
  dtype, value semantics, overlap behavior, and serialized representation.
- The frame-mask handoff path must be named. A bbox parser alone does not carry
  masks to OSD, Kafka, or another service consumer.

## Expected Data Flow

```text
video frame
  -> DeepStream preprocessing / nvinfer
  -> model outputs
       -> boxes/classes/confidence -> object metadata
       -> mask tensors             -> frame mask metadata or preserved tensor metadata
  -> downstream frame output
       -> objects[] with bbox/class/confidence/label
       -> frameMask aligned to the same frame coordinate space
```

## Contract Decisions

Record these decisions before implementation:

| Area | Required answer |
|------|-----------------|
| Detection output | Tensor name, shape, layout, class count, confidence source, pre-NMS vs post-NMS. |
| Mask output | Tensor name, shape, mask type, threshold rule, whether masks are post-NMS, and how model masks become the frame mask. |
| Association | If the model emits object-local masks, how each mask maps to a kept object before frame-mask composition. |
| Labels | Whether class IDs are contiguous label indexes or need remapping before OSD/output. |
| Coordinates | Source coordinate space for boxes and model masks, plus the canonical output coordinate space and frame-mask dimensions. |
| Frame-mask encoding | Dtype, shape, value semantics, overlap policy, and serialized representation. |
| Handoff | Whether the frame mask is attached as frame metadata, preserved as tensor metadata, converted to segmentation metadata, or consumed by a custom wrapper. |

## Frame Mask Encoding

Declare the exact frame-mask format before parser or wrapper work:

- Dimensions and coordinate space: source frame, mux frame, model input, or
  another explicitly named surface.
- Dtype and layout: for example `uint8` HxW, class-index HxW, instance-id HxW,
  packed bits, or another documented layout.
- Value semantics: background value, foreground value, class IDs, instance IDs,
  confidence-scaled values, or a documented multi-channel meaning.
- Overlap policy when multiple object-local masks contribute to the same pixel:
  highest confidence, class priority, last writer, instance ID priority, or
  another deterministic rule.
- Serialization: raw bytes, RLE, PNG, base64, protobuf bytes, or a wrapper-owned
  representation.
- Empty-frame behavior when no detections are present.

Tests should decode one emitted frame mask and assert its dimensions, dtype,
value domain, empty-frame behavior, and overlap policy where overlap is possible.

## Model Mask Output Patterns

Use the model output pattern to choose parser behavior. These are model-output
patterns only; they do not change the output contract that the downstream
service mask is one frame-level mask.

| Model output pattern | Parser decision |
|----------------------|-----------------|
| Per-instance mask tensor | Keep the post-filtered object/mask mapping, then project or composite kept masks into one frame mask. |
| Prototype masks plus coefficients | Decode masks from prototypes after selecting kept detections, then project or composite them into one frame mask. |
| Polygon or RLE output | Rasterize or convert the encoded shapes into the frame mask, unless the service explicitly supports encoded frame-mask output. |
| Frame-level mask with separate detector | Preserve the frame mask and document how object boxes share the same coordinate space. |

## Parser Interface

For detection-style models, the normal DeepStream custom bbox parser signature
may still be needed:

```cpp
extern "C" bool YourFuncName(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);
```

That signature only returns detection objects. It does not define how a frame
mask is carried forward. If the model emits masks, define the separate frame-mask
handoff path before coding.

Parser-level flow:

```text
for each candidate detection:
  decode bbox, class, and confidence
  apply thresholds and NMS or consume post-NMS outputs
  map class ID to the label-file index if needed
  find the model mask associated with the kept detection, if masks are object-local
  decode and threshold the mask from model tensors
  project bbox into the declared canonical coordinate space
  project or composite the model mask into a frame-level mask
  emit object metadata
  attach frame mask metadata or preserve tensor metadata for the next stage
```

## Coordinate Rule

Do not scale boxes to mask resolution as the default. Choose one canonical
output coordinate space, usually source-frame or mux-frame coordinates, and
project both bbox data and the frame mask into that space.

For object-local model masks, decode the mask from the model output, resize or
crop it according to the associated bbox, then place or composite it into the
frame mask in the same output coordinate space as the bbox. Parser tests should
prove that bboxes and the frame mask remain spatially aligned after scaling,
padding, or cropping.

## Minimum Tests

Add focused tests for the contracts most likely to break:

- ONNX/config consistency: expected output tensor names, shapes, and class count
  match the parser and `nvinfer` config.
- Object output: detections still emit bbox, class ID, confidence, and label.
- Mask source: masks are decoded from model tensors, not fabricated from tracker
  state.
- Frame-mask output: the service emits a frame-level mask, not object masks.
- Object-to-frame-mask contribution: object-local model masks, when present,
  contribute to the expected frame-mask region.
- Coordinate alignment: bbox data and the frame mask land in the same declared
  coordinate space after scaling, padding, or cropping.
- Frame-mask encoding: decoded output has the expected dimensions, dtype, value
  domain, empty-frame behavior, and overlap behavior.
- Label mapping: emitted class IDs match the label file consumed by OSD and
  downstream object metadata.
- Negative path: missing or renamed mask outputs fail clearly.

## Acceptance Criteria

On a known sample with visible target objects:

- At least one expected object is detected.
- The output frame contains object data and frame-level mask data.
- The frame mask has foreground pixels when detections are present.
- Object labels are human-readable and match the model's label contract.
- Mask overlays, if rendered, align with their object boxes at a coarse visual
  level.

## Handoff

After this addendum is complete:

- Return to the main `rtvi-cv-customize-model` workflow for detection parser
  code, label files, `nvinfer`, engine build, and deployment wiring.
- Put msgconv wrapper type, payload schema, Kafka/protobuf field names, and
  wrapper-specific config in
  `../../rtvi-cv-scaffold-vss-service/references/integration-contract.md`.
  This addendum only records the frame-mask data contract needed by the parser.
- Use a DeepStream skill for detailed metadata API or OSD mechanics.
- Use an RTVI-CV service onboarding workflow for Kafka payloads, GPU validation,
  OSD MP4 evidence, and cleanup.
