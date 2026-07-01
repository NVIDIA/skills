# NVTX Coverage

Which DeepStream plugins emit NVTX ranges that `nsys profile --trace=nvtx` will capture.

> **Always verify with a real probe.** This skill assumes DS 9.0 from the
> `nvcr.io/nvidia/deepstream:9.0-triton-multiarch` container. Avoid the `9.0-samples-multiarch`
> variant — it strips the CUPTI NVTX injector, producing empty per-plugin NVTX traces even
> though the plugin source code clearly contains `nvtxDomainRangePushEx` calls.

## The verification one-liner

Always run this against the user's exact pipeline before relying on any NVTX-derived
diagnosis:

```bash
nsys profile --trace=cuda,nvtx --duration=10 --output=/tmp/nvtx_probe <pipeline-launch>
nsys stats --force-export=true --report nvtx_sum --format csv /tmp/nvtx_probe.nsys-rep \
  | tail -n +6 | awk -F, '$2+0 > 0 {print $NF}' \
  | sed -E 's/\(Frame=[0-9]+\)//g; s/\(Batch=[0-9]+\)//g; s/batch_num=[0-9]+//g; s/UID=[0-9]+/UID=N/g' \
  | sort -u
```

Any DS plugin element in the user's pipeline that does NOT appear in that output is
**uninstrumented for THIS build** — even if the table below labels it COVERED.

## What to look for in the verification output

When DS plugin NVTX is working in DS 9.0 `triton-multiarch`, you should see ranges like:

| Range pattern | Emitted by | Verified in 9.0-triton-multiarch |
|---|---|---|
| `GstNvInfer: UID=N:buffer_process / queueInput / convert_buf / dequeueOutputAndAttachMeta` | `nvinfer` (named domain `GstNvInfer: UID=N`) | ✅ |
| `TensorRT:<layer-name>` (50+ ranges) | TensorRT runtime inside nvinfer | ✅ |
| `:nvdsosdN_(Frame=N)` | `nvdsosd` / `nvosdbin` | ✅ |
| `:m_collectingBuffers(Batch=N)`, `:m_acquireBufferFromPool(Batch=N)` | `nvstreammux` | ❌ **not emitted** |
| (anything from `nvurisrcbin`, `nvv4l2decoder`, `nvvideoconvert`) | closed-source binary plugins | ❌ **not emitted** |
| `NvDsTracker*` | `nvtracker` | not yet verified — probe to confirm |
| `gst_nvdspreprocess_*` | `nvdspreprocess` | not yet verified — probe to confirm |

If the verification probe returns nothing from your pipeline's actual hot-path plugins,
fall back to non-NVTX signals (see [boundedness-rules.md](boundedness-rules.md)).

## Practical implication

**Decode/source-side bottleneck diagnosis on DS 9.0 cannot rely on NVTX.** Binary
plugins (decoder, urisrcbin, videoconvert) emit nothing, and the streammux
`m_collectingBuffers` range is not reliably emitted. Use the substitute signals listed
below in [Diagnosing the gap when decoder/source NVTX is missing](#diagnosing-the-gap-when-decodersource-nvtx-is-missing).

For inference-side diagnosis (what nvinfer is doing per batch, which layers dominate, OSD
overhead) NVTX is plenty informative on DS 9.0.

## DS plugin NVTX status — observed variability

NVIDIA-shipped plugins fall into three buckets in practice. The bucket a plugin lands in
depends on the container variant (always use `9.0-triton-multiarch`; `samples-multiarch`
strips the injector and breaks NVTX entirely) and on whether NVIDIA chose to compile NVTX
calls into closed-source plugin binaries.

| Plugin | Source ships with NVTX calls? | Range visible in trace (varies by build) |
|---|---|---|
| `nvinfer` | Yes (`nvtxDomainRangePushEx` in shipped source) | Often visible; sometimes hidden when nsys can't intercept named-domain calls in dlopen'd libs |
| `nvinferserver` | Yes (similar pattern) | Same — varies |
| `nvstreammux` (binary) | Source not shipped, ranges have appeared in some builds | Often visible |
| `nvdsosd` (source shipped) | Yes | Often visible |
| `nvtracker` (binary) | Source not shipped | Sometimes visible (older builds emitted `NvDsTracker*`) |
| `nvdspreprocess` (source shipped) | Yes | Often visible |
| `nvdsanalytics` (source shipped) | Yes | Often visible |
| `nvurisrcbin` / `nvmultiurisrcbin` (binary) | Source not shipped | **Often NOT visible** in our probes |
| `nvv4l2decoder` / `nvv4l2h264enc` / `nvv4l2h265enc` (binary) | Source not shipped | **Often NOT visible** in our probes |
| `nvvideoconvert` (binary) | Source not shipped | **Often NOT visible** in our probes |
| `nvmultistreamtiler` (binary) | Source not shipped | Sometimes visible |
| `nvmsgconv` / `nvmsgbroker` (source shipped) | Yes | Often visible |

The shipped-source set lives under `/opt/nvidia/deepstream/deepstream/sources/gst-plugins/`
inside any DS container — that directory is the ground truth for what NVIDIA chose to
open-source. The closed-source binary plugins (`nvv4l2decoder`, `nvv4l2*enc`,
`nvurisrcbin`/`nvmultiurisrcbin`, `nvvideoconvert`, `nvtracker`, `nvstreammux`/the new
`nvmultistream2` library) require trusting NVIDIA's NVTX choices in each release.

## GStreamer-core elements — never instrumented

These do NOT emit DS-style NVTX:

`filesrc`, `filesink`, `fakesink`, `fakesrc`, `appsink`, `appsrc`, `qtdemux`, `qtmux`,
`h264parse`, `h265parse`, `aacparse`, `queue`, `queue2`, `tee`, `capsfilter`,
`videoconvert`/`videoscale` (CPU variants — avoid in DS pipelines).

The skill never relies on these for diagnosis.

## Diagnosing the gap when decoder/source NVTX is missing

If the verification probe shows the inference-path plugins emit NVTX but
decoder/source-bin/converter do NOT, you cannot read decoder time directly. Use these
substitute signals (all encoded as decision rules in
[boundedness-rules.md](boundedness-rules.md)):

1. **`nvstreammux:m_collectingBuffers` share** (when streammux NVTX IS captured) — > 40% of
   NVTX time means streammux is *waiting for upstream*, i.e. source/decoder bound.

2. **Microbench scaling shape (Stage 3)** — *always works*, no NVTX needed. Per-batch FPS
   doubling from B=1 to B=2 with parallel sources is the gold-standard signal that the
   decoder was the limit at B=1.

3. **`nvidia-smi dmon -s u` during the run** — `dec` column near 100% confirms the NVDEC
   engines are saturated. Sample at a higher rate than 1 Hz (`-d 1` with `-c N`) for short
   bursty workloads — single-frame decode at 1080p H264 takes only ~1 ms, so low-frequency
   sampling underreports.

These three substitute for the missing per-plugin NVTX coverage on the source/decoder side.

## When the user's pipeline includes a custom plugin

Any element whose name starts with something other than the standard prefixes (`nv*`,
`queue`, `tee`, `caps`, `h264`, `h265`, `file`, `fake`, `app`, `qt`, `video`) is third-party
and almost certainly UNINSTRUMENTED. The verification probe will confirm. The skill cannot
diagnose its internal cost from NVTX. Either:

1. **Auto-inject NVTX** by wrapping the element with a buffer probe that calls
   `nvtxRangePush/Pop` on each `chain` callback (out of scope for the MVP — listed as future
   work in `SKILL.md`).
2. **Replace with an instrumented equivalent** if one exists.
3. **Use indirect signals** (microbench scaling, nvidia-smi dmon, CUDA kernel mix) and
   report the custom plugin as "not directly measurable" in the final report.

## Bottom line

NVTX coverage is **partial and version-specific**. The skill's measurement strategy must
NOT depend on it. Per-plugin NVTX is a bonus when present (gives sharper bottleneck
attribution); the skill's primary signals are CUDA kernel summaries, memcpy summaries,
microbench scaling shape, and `nvidia-smi dmon` — all of which work regardless of NVTX
coverage.
