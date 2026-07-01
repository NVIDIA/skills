# Step 5 — Output Format: Worked Example & Anti-Patterns

Reference for **Step 5 — Present the Pipeline**. The enforcing rules (the 5-block
contract Section 5.1, the pre-flight Section 5.2, the self-check Section 5.5, and the failure variant
Section 5.6) live in SKILL.md. This file holds the **literal correct template (Section 5.3)** your
output MUST match, and the **forbidden anti-pattern gallery (Section 5.4)**. Read it before
composing the Step 5 response.

---

#### 5.3 — ✅ CORRECT example (3-stream infer + display, dGPU, defaults present)

This is exactly what your output should look like. Match it.

````markdown
✓ Validated · 11 elements · 0 warnings · live-parse skipped (multi-stream) · confidence: HIGH (matched 10 verified examples)

```bash
gst-launch-1.0 -e filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_1 filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_2 nvstreammux name=m batch-size=3 width=1920 height=1080 ! nvinfer config-file-path=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt batch-size=3 unique-id=1 ! nvmultistreamtiler rows=1 columns=3 width=1920 height=720 ! nvvideoconvert ! nvdsosd ! nveglglessink
```

**Pipeline breakdown (by stage):**

| Stage | Elements | Role |
| --- | --- | --- |
| Source (×3) | `filesrc → qtdemux → h264parse → nvv4l2decoder` | Decode each MP4 onto a separate `nvstreammux` sink pad |
| Mux | `nvstreammux batch-size=3` | Batch the 3 streams into a single batched buffer |
| Inference | `nvinfer` (PGIE) | Primary object detection on the batched frames |
| Composition | `nvmultistreamtiler rows=1 columns=3` | Tile 3 streams side-by-side into one output frame |
| Render | `nvdsosd → nveglglessink` | Overlay bounding boxes and render to display |

**Suggestions:**

- **Use your own video:** replace `/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4` with your file path (in all 3 `filesrc` lines).
- **Use your own model config:** replace `/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt` with your `config_infer_*.txt`.
- **Save it to a script?** Say *"save it"* and I'll write it to `/tmp/pipeline.sh` — then run with `bash /tmp/pipeline.sh`.
- **Other changes?** Want me to add a tracker (NvDCF/NvSORT/IOU/DeepSORT), change the tile layout (e.g. `2x2`), save the output to MP4 instead of display, or switch to Jetson?
````

#### 5.4 — ❌ ANTI-PATTERNS — DO NOT EVER PRODUCE OUTPUT IN THESE FORMS

The following formats are forbidden. They are listed here so you have a concrete reference of the wrong shapes to avoid. **If your draft response resembles any of them, throw it away and re-emit in the Section 5.3 form instead.**

❌ **NEVER use a heredoc-to-script wrapper:**

````markdown
```bash
cat > /tmp/pipeline.sh <<'EOF'
#!/usr/bin/env bash
gst-launch-1.0 -e \
  filesrc location=… ! m.sink_0 \
  filesrc location=… ! m.sink_1 \
  …
EOF
bash /tmp/pipeline.sh
```
````

❌ **NEVER use shell-variable indirection or env-var overrides:**

````markdown
```bash
DS=${DS:-/opt/nvidia/deepstream/deepstream}
SRC=${SRC:-$DS/samples/streams/sample_1080p_h264.mp4}
gst-launch-1.0 -e filesrc location=$SRC ! …
```
````

❌ **NEVER add a "Run it:" or "bash /tmp/pipeline.sh" instruction.** The pipeline in the Section 5.3 code block IS the runnable thing — pasting it in a terminal runs it.

❌ **NEVER split the pipeline across multiple lines with `\` continuations**, even for readability. Long lines are correct; the bash code block preserves them faithfully on copy.

❌ **NEVER call the `Write` tool to create `/tmp/pipeline.sh` (or any other script file) as part of Step 5.** Saving to a script is opt-in via Step 6.5 — only when the user explicitly asks "save it" or names a path. The default delivery is the inline single-line command, period.
