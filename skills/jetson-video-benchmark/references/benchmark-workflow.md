# Benchmark workflow

Run compact, authenticated Video Codec SDK performance measurements on
user-supplied representative content. This workflow owns live FPS, multi-stream
concurrency, and P4/P5 comparisons. It also owns a separate
documentation-derived planning answer when content is unavailable. It never
substitutes an external codec, and it does not provide PSNR/SSIM quality
measurement. For a quality-only request, apply the terminal scope rule in
`SKILL.md` and stop.

## Two answer paths

- **Live measurement:** requires one exact user-selected path or URL, no
  catalog or synthetic substitution, and preserved provenance and identity.
  When setup is installed, apply its shared
  [video content policy](../../jetson-video-setup/references/video-content.md).
  The setup skill is not otherwise required for this media rule. An explicit
  run, measure, or real-benchmark request remains media-gated.
- **Documented estimate:** applies only to a no-media planning, expected,
  indicative, achievable-FPS, or theoretical codec-stream-capacity question at
  any positive resolution. Follow
  [documented-performance-estimates.md](documented-performance-estimates.md).
  Non-1080p results add a disclosed pixel-area heuristic to the documented row
  and clock scaling. This is a calculation-only answer path, not a controller
  route or benchmark result.

## Owner and exact CLI

One controller owns live measurements in this domain:
**`jetson-video-benchmark/scripts/benchmark_controller.py`**. Launch it
directly:

```text
python3 -I scripts/benchmark_controller.py \
  --request nvcodec-benchmark-request.json \
  --workspace fresh-workspace-dir \
  --output nvcodec-benchmark-result.json
```

All three flags are required (`--help` supported). Exit `0` on success, `2` on any
request or contract failure.

## Benchmark request schema

`--request` is a `schema_version: "1.0"`, `kind: "nvcodec-benchmark-request"` object.
`mode` is exactly `dry_run` (plan only) or `execute` (default `execute`). `route` is
exactly one of:

| `route` | Purpose |
|---|---|
| `encode` | Encode throughput / achievable encode FPS |
| `decode` | Decode throughput / achievable decode FPS |
| `compare` | P4 vs P5 performance, holding every non-preset control identical |
| `camera_capacity` | Multi-stream codec-worker concurrency capacity |

`environment` is optional. Setup emits the raw schema-1.2
`nvcodec-environment` JSON; the benchmark request carries its exact portable
identity, not the embedded JSON or a private workspace object:

```json
{
  "schema_version": "1.2",
  "kind": "nvcodec-environment",
  "path": "/canonical/absolute/fresh-environment.json",
  "size_bytes": 1234,
  "sha256": "64-lowercase-hex-digest"
}
```

Derive `path`, `size_bytes`, and `sha256` from the fresh file after the probe
finishes. The controller verifies the current bytes and never falls back if
that supplied identity is malformed, stale, or unsuitable.

After the media gate, an agent that needs `pynvc` or `both` and has no supplied
environment or exact interpreter first checks the installed skill catalog. If
`jetson-video-setup` is present, invoke its public `probe_nvcodec.py` with a
fresh output path, `--runtime pynvc` for Python-only or `--runtime both` for
`both`/`auto`, and no `--setup-candidate`. This is an artifact handoff, not a
consumer import: setup alone resolves and reauthenticates its fixed registry.
Inspect the emitted JSON before constructing the identity above. It is usable
for Py only when `mode=live`, the requested GPU matches,
`pynvc.installed=true`, and `pynvc.identity.status=verified`; the benchmark
controller then independently validates it again. Ask for an exact interpreter
for explicit `pynvc`/`both` only when setup is unavailable or returns any
absent, stale, unreadable, invalid-binding, or launch-failure result. For
`auto`, record Py as `not_evaluated` and continue only an eligible native
surface. Never pass a blocked probe as authority, scan for a venv, or trust an
older artifact.

When `environment` is omitted, the controller performs bounded read-only local
authentication. An explicit `native` route inspects only the fixed APT package
`nvidia-video-codec-sdk`, verifies its public 13.0.x version and unmodified
package-owned official sample tree, binds the existing build toolchain, and
later verifies non-stub codec linkage. An explicit `pynvc` route requires
`pynvc_interpreter` as one exact absolute path and authenticates the isolated
import, exact loaded extension, wheel RECORD, and wheel-owned sample. It never
searches for a venv. Explicit single-surface routes do not inspect their peer.
An explicit `pynvc` or `both` request without `pynvc_interpreter` returns
`input_required`. For local `auto`, an absent selector is reported as Python
`not_evaluated`, not silently omitted, and an eligible native branch may continue.
The controller's local binding is derived in-process and may be emitted as
evidence, but is never accepted as a request input or as a substitute setup
artifact.

For encode, pass `recipe` as the exact portable absolute identity of a schema-2
`nvcodec-recipe`; embedded recipe objects are invalid. Its codec, width, height,
format, FPS, optional frame count, and GPU must exactly match the input/request
workload before the documented PyNvVideoCodec frame cap is applied. Decode does
not consume a recipe. If the sibling recipe validator is absent, encode returns
structured `dependency_required` with the dependency, reason, and exact
install-and-retry action rather than a generic controller error.

If bounded local authentication proves that a selected SDK surface needs
installation or repair, preserve any healthy peer and return a structured
`jetson-video-setup` dependency with its installed state and retry action. A
missing exact Python interpreter selector is still `input_required` (or
`not_evaluated` for local `auto`), never inferred to be a broken installation.

`input` (encode) or `encoded_artifact` (decode) carries the verified artifact
identity, exact width/height/frames/FPS/codec/format metadata, and preserved
`source_url` (null for local input), `license`, and `attribution`.
Encode request metadata accepts H.264, HEVC, or AV1; decode additionally accepts
VP9. An accepted request codec is not itself a target-support claim.

`compare` requires two encode recipes that differ only by preset. Only released
sample routes that carry an authenticated FPS value (see the shared contract)
are eligible for throughput; runner process duration is never relabeled as
sample-reported throughput.

## Workflow

1. Classify the request into exactly one route above.
2. Apply the shared video-content input, resolution, provenance, and synthetic-
   fixture rules. If input is absent for an explicit live measurement, return
   its `input_required` result and pause before probing, retrieval, dry run, or
   execution. Request only the missing media at that gate; interpreter,
   environment, recipe, and surface-authority resolution happen afterward. A
   no-media planning question follows the separate documented-estimate path
   above.
3. For preset comparisons, produce recipes with the recipes workflow that hold everything but the
   compared control constant, and bind every variant to the same user-selected content and frame
   range.
4. Run the benchmark controller in `dry_run` first to review the planned invocation,
   then `execute` with a fresh output path and a bounded timeout (300 s per repetition).
5. Report the measured target and exact workload only — not a portable product ceiling.
   Resolve camera data direction before choosing the controller operation.
   Measured concurrency without matching capture/transport/latency evidence is
   an encode- or decode-stage capacity bound, not a verified camera count. A
   no-media capacity calculation is weaker still: label it a theoretical
   codec-stream bound and follow the estimate reference's shared-budget formula.

See [benchmark-output-contract.md](benchmark-output-contract.md) for the result JSON
contract. The controller authenticates the required official performance
sample from either the exact setup-produced live environment identity or the
bounded local authority described above.
