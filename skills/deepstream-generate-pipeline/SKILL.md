---
name: deepstream-generate-pipeline
description: Build DeepStream GStreamer pipelines interactively. Use when the user asks about pipelines for video/image inference, detection, tracking, or streaming — including natural phrases like 'pipeline to infer on image', 'run inference on video', 'detect objects in stream', 'save inference output', 'deepstream pipeline', 'gst-launch pipeline', 'process video with detection', 'build a pipeline', or any request involving GStreamer/DeepStream elements (nvinfer, nvstreammux, nvtracker, etc.).
owner: NVIDIA CORPORATION
service: deepstream
version: 1.0.0
reviewed: 2026-04-27
license: CC-BY-4.0 AND Apache-2.0
---

# DeepStream Pipeline Builder

Generate ready-to-run `gst-launch-1.0` pipelines for NVIDIA DeepStream SDK by collecting pipeline requirements through an interactive questionnaire, then assembling the pipeline using a standalone BM25 retrieval backend with structural metadata boosting (similarity search over 270+ verified pipelines, zero external dependencies).

## Prerequisites

- **Python:** 3.8+ (stdlib only — no pip packages required)
- **DeepStream SDK:** Installed at `/opt/nvidia/deepstream/deepstream/` (for `gst-inspect-1.0` validation and element verification)
- **GStreamer:** `gst-launch-1.0` and `gst-inspect-1.0` on `PATH` (installed with DeepStream)
- **Platform:** x86 dGPU (T4, A100, L40, RTX, etc.) or aarch64 — Jetson (Orin, Xavier, Nano) / SBSA (Grace, GH200)

## Usage Examples

```text
# Fully specified — skips most questions
detect and track on 4 rtsp streams and display on jetson

# Partially specified — asks remaining questions
give me a pipeline to infer on an image

# Minimal — asks all 7 questions
build a pipeline
```

## Supported Configurations

| Parameter | Options |
| --- | --- |
| **Input** | Local video (.mp4/.h264/.h265), local image (.jpg/.png), RTSP stream, USB camera, test pattern |
| **Inference** | None, primary (nvinfer), primary+secondary, with preprocessor, Triton (nvinferserver) |
| **Tracker** | None, NvDCF, IOU, NvSORT, DeepSORT |
| **Sink** | Display (dGPU/Jetson), save (JPG/PNG/MP4/H264), RTSP out, fakesink |
| **Platform** | x86 dGPU (T4, A100, L40, RTX, etc.) or aarch64 — Jetson (Orin, Xavier, Nano) / SBSA (Grace, GH200) |
| **Extras** | Resize, rotate/flip, crop, color format conversion |

## Scripts

| Script | Purpose |
| --- | --- |
| `scripts/generate_pipeline.py` | BM25 retrieval engine — scores and ranks pipelines from `data/data.csv`. Supports `--format {json,compact,summary}` (default `json`) |
| `scripts/validate_pipeline.py` | 4-stage validator: syntax, elements, properties, live parse. Supports `--format {json,summary}` (default `json`) |
| `scripts/lint_data.py` | Data quality linter for the pipeline CSV (`--fix` to auto-repair) |

## Workflow

### Step 1 — Collect Pipeline Requirements

> **You MUST `Read references/requirement-extraction.md` before doing this step.**
> It contains the query-inference table, compound-extraction examples, the full
> `AskUserQuestion` question bank (with the default-first ordering contract), the
> automatic-OSD and extras/flip-method rules, and the dynamic question-reduction
> examples that this step depends on. Apply them exactly.

**Order of operations:**

1. **Infer everything you can from the query** using the inference table in `references/requirement-extraction.md`. The goal is to identify which of the 7 parameters (input source, num sources, inference, tracker, sink, platform, extras) the user has already specified.
2. **Ask the user about the unknowns via `AskUserQuestion` in a single call.** Do **not** silently default tracker/sink/platform/extras — these are real choices the user should make explicitly (display vs save, no tracker vs NvDCF, x86 dGPU vs aarch64 Jetson/SBSA, etc.). Skip only the questions whose answer is already clear from the query.
3. **Quote the inferred parameters back to the user** in the lead-in to the question call so they can see what you already extracted. Example: *"From your query I have: 3 mp4 videos, primary inference. Just need a few more details:"*

Follow the inference table, question bank, and OSD/extras rules in
`references/requirement-extraction.md` to decide which questions to ask and how to
place transform elements, then proceed to Step 2.

### Step 2 — Build the Natural Language Query

From the user's answers, construct a single descriptive query string. Follow this pattern:

```text
Please provide a GStreamer pipeline that [operation] on [num_sources] [input_type] [input_detail] [tracker_detail] and [output_action] [platform_detail]
```

**Examples of constructed queries:**

| User Selections | Constructed Query |
| --- | --- |
| Local video, 1 source, Primary detector, No tracker, Display, dGPU | "Please provide a GStreamer pipeline that performs primary inference on a single mp4 video and displays the output" |
| RTSP, 4 sources, Primary+Secondary, NvDCF, Save MP4, dGPU | "Please provide a GStreamer pipeline that performs primary and secondary inference with NvDCF tracker on 4 RTSP streams and saves output to MP4 file" |
| Local video, 2 sources, Primary with preprocessor, IOU, Display, Jetson | "Please provide a GStreamer pipeline that performs preprocessing before primary inference with IOU tracker on 2 mp4 streams and displays the output on Jetson" |
| Local image, 1 source, None, No tracker, Save file, dGPU, Rotate 90° cw | "Please provide a GStreamer pipeline that rotates a single jpg image 90° clockwise before processing and saves it to a file" |
| Local video, 3 sources, Primary detector, NvDCF, Save MP4, dGPU, Rotate 180° | "Please provide a GStreamer pipeline that rotates 3 mp4 videos 180° before primary inference with NvDCF tracker and saves output to MP4 file" |

### Step 3 — Run the Pipeline Generator Script

Execute the backend script with the constructed query and user parameters:

```bash
python3 <skill-path>/scripts/generate_pipeline.py \
  --query "<constructed_query>" \
  --source-type "<Local video file|Local image file|RTSP stream|USB camera|Test pattern>" \
  --num-sources <N> \
  --inference "<None|primary|primary+secondary|primary+preprocess|primary+secondary+preprocess|primary-triton|primary+secondary-triton>" \
  --tracker "<none|NvDCF|IOU|NvSORT|DeepSORT>" \
  --sink "<display|display-jetson|save-jpg|save-png|save-mp4|save-h264|rtsp-out|fakesink>" \
  --platform "<dGPU|Jetson|SBSA>" \
  --extras "<none|resize|rotate|crop|color-convert|osd>" \
  --format compact
```

> **Always pass `--format compact`.** The `compact` mode returns only confidence + the top retrieved pipeline (~25 lines), instead of dumping all 10 retrievals as ~150 lines of JSON in the chat. The `json` mode (default for backward compat) is only useful when debugging the retriever directly. A `summary` mode (single human-readable line) also exists for non-Claude callers.

The script will (zero external dependencies — pure Python stdlib):

1. Load the pipeline dataset (270+ verified DeepStream pipelines)
2. Extract structural metadata from each pipeline (platform, source type, sink type, inference mode, tracker, stream count)
3. Score with BM25 (document-length-normalized) + domain-specific synonym expansion on both queries and documents
4. Apply structural boosting — results matching the user's platform/source/sink/inference get boosted, mismatches get penalized
5. Return the top-K results as JSON with a `confidence` field (`high`/`medium`/`low`) based on the top score
6. Claude uses these retrieved examples + the assembly rules below to construct the final pipeline

When `confidence` is `low`, rely more heavily on the assembly rules below rather than the retrieved examples.

### Step 4 — Validate the Pipeline

Before presenting, run the validation script to catch syntax errors, unknown elements, and linking issues:

```bash
python3 <skill-path>/scripts/validate_pipeline.py "<assembled_pipeline>" --format summary
```

> **Always pass `--format summary`.** Summary prints a single status line (e.g. `valid · 11 elements · 0 warnings · live-parse skipped (multi-stream)`), with errors/warnings indented underneath only if present. The default `json` mode emits ~40 lines of structured output and is only useful for programmatic callers.

The validator performs 4 checks:

1. **Syntax check** — unbalanced quotes, empty pipe segments, missing source/sink
2. **Element check** — verifies each element exists via `gst-inspect-1.0`
3. **Property check** — validates known properties for DeepStream elements
4. **Live parse check** — uses `gst-launch-1.0` itself to construct the pipeline graph (with fakesrc/fakesink substituted), catching linking errors and pad mismatches. **Automatically skipped for multi-stream pipelines** (those with named pad refs like `m.sink_0`) since fakesrc cannot negotiate caps through named pads.

If validation fails (`"valid": false`), fix the errors and re-validate before presenting. **Limit validation retries to a maximum of 2 attempts** — if the pipeline still fails after 2 fixes, present it as-is (the remaining checks already cover syntax, element, property, and structural correctness). If there are only warnings, present the pipeline but mention the warnings to the user.

### Step 5 — Present the Pipeline

#### 5.1 — Output format (THE ONLY ACCEPTED FORM)

Your response **must** be exactly five blocks, in this order:

1. One-line **status badge** (validation + confidence)
2. **Single bash code block** containing the full `gst-launch-1.0 -e …` command with concrete absolute paths, on **one line** (no `\` continuations, no shell variables, no shell wrapper)
3. **Breakdown table** grouped by stage
4. **Suggestions** bullet list
5. (only if pre-flight failed) a `⚠` line above the status badge stating which default path is missing

That is the ONLY accepted output shape for this step. The Section 5.3 template in `references/output-format.md` is the literal template — match it.

#### 5.2 — Pre-flight check (run before composing the response)

Run one `Bash` `ls` over the default paths the pipeline will reference (sample video, PGIE config, tracker lib/config). The result tells you whether to mark the badge with `⚠ default path not found: <path>` and bump the matching "Use your own …" suggestion to the top.

```bash
ls /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 \
   /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt \
   2>&1
```

#### 5.3 / 5.4 — Worked example & forbidden anti-patterns

> **You MUST `Read references/output-format.md` before composing this response.** It contains the literal Section 5.3 template your output must match exactly, and the Section 5.4 gallery of forbidden output shapes (heredoc wrappers, shell-var indirection, `\` line-continuations, stray "Run it" lines, `Write`-to-script). Mirror Section 5.3; never emit any Section 5.4 form.

#### 5.5 — Self-check before sending the response

Before you emit your reply, mentally tick each box. If any check fails, rewrite the response.

- [ ] The pipeline is on **exactly one line** inside a single ```` ```bash ```` code block.
- [ ] The pipeline begins with `gst-launch-1.0 -e` and contains only literal absolute paths (e.g. `/opt/nvidia/deepstream/...`) — no `$VAR`, no `${VAR:-default}`, no `cat >`, no `EOF`, no `\` line continuations.
- [ ] The response does **not** contain any of: `cat > /tmp/pipeline.sh`, `bash /tmp/pipeline.sh`, `<<'EOF'`, `${VAR:-`.
- [ ] The response does **not** call the `Write` tool. (Save-to-file is offered as a *suggestion bullet*, not an action.)
- [ ] The breakdown table is grouped by stage (Source / Mux / Inference / Tracking / Composition / Render — adapt names to the pipeline's actual stages, e.g. add an `Encode/Mux` row for file sinks).
- [ ] The "Save it to a script?" line appears in the Suggestions list — never as a primary action.

#### 5.6 — Pre-flight failure variant

If the Section 5.2 `ls` reported one or more missing default paths, prepend a `⚠` line above the status badge and bump the matching "Use your own …" suggestion to the top:

````markdown
⚠ default path not found: `/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4` — substitute your own video path before running
✓ Validated · 11 elements · 0 warnings · confidence: HIGH

```bash
gst-launch-1.0 -e filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! …
```

[breakdown + suggestions as in Section 5.3, with the "Use your own video" suggestion bumped to the top]
````

> **On length:** 5–8 stream pipelines run long when on a single line. That is correct and intended — chat clients render bash code blocks faithfully and copy reproduces them correctly. Long ≠ split.

### Step 6 — Offer Refinement

After presenting the pipeline, ask the user if they want to adjust anything:

> Want me to modify anything? For example:
>
> - Change the number of streams
> - Add/remove tracker or secondary inference
> - Switch between display and file output
> - Change the platform (x86 dGPU / aarch64 Jetson / SBSA)

If the user requests changes, go back to **Step 2** with updated parameters — do NOT re-ask all 7 questions. Only ask about the specific parameter that changed, or just apply the change directly if it's clear.

### Step 6.5 — Optional: Save Pipeline to a Script

Only do this step when the user explicitly asks (e.g. *"save it"*, *"save to pipeline.sh"*, *"write it to a file"*, *"put it in ~/run.sh"*). Do **not** create the file proactively — Step 5 always shows the concrete pipeline in chat for direct copy-paste; saving is a follow-up convenience.

1. **Filename:** Default to `/tmp/pipeline.sh` if the user just says *"save it"*. Use the exact path the user named otherwise (e.g. `~/run.sh`, `scripts/demo.sh`).
2. **File contents:** Two lines — shebang + the same single-line pipeline shown in chat (concrete absolute paths, no shell vars). Keep them in sync — what the user runs from the file is bit-for-bit identical to what they could have copy-pasted.

   ```bash
   #!/usr/bin/env bash
   gst-launch-1.0 -e filesrc location=/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 … ! nvdsosd ! nveglglessink
   ```

   Use the `Write` tool to create the file.
3. **Confirm to user** with the run command:

   > Saved to `<path>`. Run it with:
   >
   > ```bash
   > bash <path>
   > ```

---

## Pipeline Assembly Rules

When the script is not available or fails, assemble the pipeline using the rules in [references/assembly-rules.md](references/assembly-rules.md). These rules cover source elements, multi-stream patterns, inference chains, tracker configs, sink elements, and extra operations. They also serve as validation for script output.

---

## Error Handling

| Failure | Cause | Recovery |
| --- | --- | --- |
| `generate_pipeline.py` returns `confidence: low` | Query doesn't match any pipeline in the dataset closely | Rely on the assembly rules in this skill instead of retrieved examples |
| `validate_pipeline.py` reports unknown element | GStreamer/DeepStream not installed or not on `PATH` | Install DeepStream SDK; confirm `gst-inspect-1.0 nvinfer` works |
| Validation fails after 2 retries | Unusual element combination or linking issue | Present the pipeline as-is with a warning — syntax/element/property checks still passed |
| Script not found at `<skill-path>/scripts/` | Skill not installed correctly or path misconfigured | Verify the skill directory is symlinked into `.claude/skills/` or `.cursor/skills/` |

## Testing

Run the test suite to verify retrieval quality and validator correctness:

```bash
python3 -m unittest discover -s <skill-path>/tests -v
```

The suite includes:

- **Unit tests** for the BM25 retriever (tokenizer, synonym expansion, metadata extraction, scoring)
- **Unit tests** for the validator (syntax, structure, property, named-pad checks)
- **Golden regression tests** — 20+ query→expected-result pairs ensuring retrieval quality doesn't regress
- **Data quality linter** — checks the CSV for duplicates, syntax issues, and structural bugs:

```bash
python3 <skill-path>/scripts/lint_data.py          # report issues
python3 <skill-path>/scripts/lint_data.py --fix     # auto-fix and overwrite
```

---

## Security, Limitations & Notes

Security posture, known limitations, and operational notes are documented in `references/security-and-limitations.md`. Read that file when you need details on subprocess safety, input validation, platform/SDK requirements, the multi-stream dry-run caveat, or sample-path/config-file reminders.

