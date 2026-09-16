# deepstream-import-vision-model — Reference Documents

Detailed phase guides for the `deepstream-import-vision-model` skill. Read the relevant file before starting each pipeline phase.

| Document | Pipeline Steps | When to read |
|---|---|---|
| [model-acquire.md](model-acquire.md) | Steps 1–3 | Downloading from HuggingFace or NGC; ONNX vs SafeTensors detection and export |
| [engine-build.md](engine-build.md) | Steps 4–5 | TensorRT dynamic engine build; `trtexec` BS=1 and BS=MAX\_BS benchmarks |
| [pipeline-run.md](pipeline-run.md) | Steps 6–7 | Custom `nvinfer` bbox parser; single-stream validation; multi-stream benchmark sweep |
| [report-generation.md](report-generation.md) | Step 8 | 5 benchmark charts; Markdown → HTML → PDF report generation |
