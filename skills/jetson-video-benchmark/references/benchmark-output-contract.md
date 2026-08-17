# Benchmark output contract

`benchmark_controller.py` returns strict JSON. Planned or attempted throughput routes return
`schema_version: "2.0"`, `kind: "nvcodec-benchmark-result"`, the requested
`route`/`mode`, a `status`, the resolved `surface_plan`, and per-variant results.
When required user media is absent, preflight instead returns `status: "input_required"`,
`gate: "media_input"`, `accepted_inputs`, `next_action: "provide_media_path_or_url"`,
`synthetic_input_allowed: false`, the required request field, and the missing variant names.
It has no `surface_plan` or attempted variant records because it stops before live-environment
resolution, workspace creation, authentication, or launch.
The CLI prints this preflight result to standard output and does not resolve or create
`--workspace` or write its workspace-relative `--output` for this status.

When a setup-produced environment identity is supplied, the result preserves it
under `environment`. When it is omitted, `runtime_authority` labels the bounded
local discovery, selected GPU, evaluated/not-evaluated surfaces, diagnostics,
and the controller-owned binding identity. The binding contains raw live facts
used by the existing normalizer; it never serializes normalized `surfaces` and
is never accepted in a request.

An encode whose sibling recipe public validator is absent returns
`status: "dependency_required"`, `gate: "skill_dependency"`, and a `dependency`
object naming `jetson-video-recipe`, why it is needed, and the install-and-retry
action. This is not an SDK-readiness verdict. Decode has no recipe dependency.

When an inspected SDK surface is not locally authenticatable, the result keeps
the independent surface verdict and adds `dependency` plus `dependencies`
objects naming `jetson-video-setup`, whether it is installed, why the selected
surface needs installation or repair, and the exact use-or-install-and-retry
action. An encode-capable PyNvVideoCodec environment that lacks the official
sample dependencies additionally reports `required_profile: "full-samples"`
and the missing dependency names. An omitted `pynvc_interpreter` remains an
input/selection condition, not a claim that setup is broken.

This controller contract applies only to live measurement requests. A no-media
planning question may use the separate
[documented performance estimate](documented-performance-estimates.md) answer
path. That path does not invoke the controller, does not emit an
`nvcodec-benchmark-result`, and never relabels a documentation calculation as a
measured result. Its prose evidence class is
`documented_clock_scaled_estimate` for an exact 1080p clock-scaled row,
`documented_clock_resolution_scaled_estimate` when the disclosed pixel-area
heuristic is added, or `documented_theoretical_capacity_estimate` when the
estimate is converted to a stream budget. The latter two always set
`measurement_performed: false`; any answer that applies the pixel-area heuristic
also sets `resolution_scaling_documented: false`, and capacity also sets
`capacity_verified: false`.

## Throughput lifecycle

Live encode/decode, comparison, and camera-capacity measurements require one whole-process
warmup followed by at least three whole-process measured repetitions for every selected
variant and surface. The warmup is retained but excluded from statistics. Every measured
command uses phase `measure`; no controller invocation emits `-loop`.

A throughput `dry_run` records the candidate official sample and argument vector for each
selected branch without authenticating or launching it. Native encode options remain labeled
as pending validation against the installed AppEncPerf `-h` and `-A` advertisements; the plan
does not claim executable or operation readiness. An explicit `both` dry run records both
requested branches: each is independently `planned` or `blocked`. The top level is `planned`
only when every requested branch is planned, `partial` when at least one is planned, and
`blocked` when none is planned.

For native encode, the controller authenticates `AppEncPerf`, runs both `-h` and `-A`,
unions the advertised option names, and rejects any constructed option not in that union.
Native decode uses authenticated `AppDecPerf`. Python encode/decode uses the corresponding
wheel-owned `encode_perf.py`/`decode_perf.py` route. Surface results remain independent and
are never pooled or ranked.

Each completed variant/surface result contains, at the appropriate level:

- authenticated sample and provenance identities;
- exact benchmark arguments and, for native encode, help-command evidence;
- for PyNvVideoCodec encode, the schema-1 `pynvc-encoder-config` identity whose
  workspace artifact is the exact `-json` argument;
- input path/size/SHA-256 plus geometry, frames, FPS, codec, format, source URL
  (or null), license, and attribution;
- for encode, the exact recipe artifact identity retained at variant level and
  re-verified immediately before authentication;
- the warmup command record;
- every measured command record, parsed FPS, MP/s when authenticated dimensions
  exist, and repetition number;
- finite mean, minimum, and maximum FPS and megapixels/second when present.

The authenticated PyNvVideoCodec 2.1 `decode_perf.py` output does not report stream
dimensions. Its decode records therefore set MP/s to null and carry an explicit
`metric_omissions.megapixels_per_second` reason in every repetition and summary;
caller-supplied dimensions are never used to publish that derived metric. Native
decode may publish MP/s only after AppDecPerf reports matching stream metadata.

The final customer response, not only the result artifact, must list every measured
repetition's FPS and MP/s when present (or the explicit MP/s omission reason), followed
by the mean, minimum, and maximum of each present metric. A mean/range-only summary or
artifact link is incomplete. For `both`, also disclose the exact native-CLI versus
PyNvVideoCodec-config representation differences and whether they changed the requested
semantic intent.

The expected processed-frame count is the effective frames per worker multiplied by
workers. A missing, contradictory, non-finite, or wrong-count sample marker fails that
branch. Runner-process duration is command evidence but never substitutes for
sample-reported FPS.

### PyNvVideoCodec performance frame cap

The authenticated PyNvVideoCodec 2.1 `encode_perf.py` helper contract caps every
performance encode at 1000 frames per worker; the released sample silently limits longer
inputs. The controller therefore derives one effective per-worker frame count for each
variant:

- Encode with an exact, participating `pynvc` projection uses
  `min(source frames, 1000)` per worker. When native participates in the same `both`
  variant, native receives the identical effective count (`AppEncPerf -frame` equals
  `encode_perf.py -f`), so both surfaces measure the same first frames and each parser
  requires exactly `effective x workers` processed frames.
- Native-only encode keeps the full requested count, including when PyNvVideoCodec
  is ineligible or its recipe projection is unrepresentable. Decode is never capped.
  `compare` (P4/P5) and `camera_capacity` variants inherit the same per-worker rule.
- The original input identity and its source frame count are preserved unchanged in the
  request and in each surface record's `input` block; the controller never trims or
  copies the raw file, rewrites the recipe, or claims that capped-away frames ran.

Every dry-run and executed surface record carries a `frame_accounting` object stating
`source_frames_per_worker`, `effective_frames_per_worker`, `workers`,
`expected_aggregate_frames`, `cap_applied`, `cap_limit_frames_per_worker` (exactly 1000),
and `cap_authority` (the authenticated PyNvVideoCodec 2.1 `encode_perf.py` helper
contract). If a request sets `require_source_frames` to true and the cap would apply, the
controller fails before any launch and directs the caller to a native-only surface or an
input at or below the limit; it never silently reinterprets that explicit intent.

## Route-specific fields

- `compare` accepts two variants only when their recipe/workload/execution facts are equal
  except for preset. Its `comparison` retains each completed surface and variant summary.
- `camera_capacity` requires strictly increasing worker counts beginning at one. It records
  every tested point, the explicit safety margin, the maximum passing tested count, and the
  limitation that capture, transport, and end-to-end latency remain unverified.
- `both` is accepted only when the user explicitly requests a dual-surface run or
  comparison. It reports native and Python branches independently. A completed branch is
  retained when the peer branch fails, is ineligible, or has no exact recipe projection,
  and the top-level status becomes `partial`. Projection losses block only the affected
  surface.
- “Whichever”, “best available”, “choose for me”, and other unspecified-surface
  wording remain `auto`; they are never rewritten to `both`. `auto` runs exactly one
  eligible surface, blocks with zero, and returns `selection_required` with two before
  searching prior results, dry run, authentication, or launch. The two-surface gate is
  unconditional and asks for exactly `native`, `pynvc`, or `both`.

## Status and exit code

`completed` and `planned` are successful and the CLI exits 0. `input_required`,
`dependency_required`, `partial`, `failed`, `blocked`, and `selection_required`
are preserved as structured results and the CLI exits 2. `input_required` is
emitted on stdout without creating `--output` or a workspace.
Malformed input, stale outputs, authentication failure, bad markers, and metric mismatch are
never rewritten as a successful result.
