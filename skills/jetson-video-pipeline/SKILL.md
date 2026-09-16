---
name: jetson-video-pipeline
license: "Apache-2.0"
description: >-
  Use when executing and verifying Jetson Video Codec SDK or PyNvVideoCodec
  encode/decode, transcode, segmentation, container decode, AV1, or acceptance
  workflows with exact artifact handoffs.
metadata:
  author: "Vinit Bansal <vinitkumarb@nvidia.com>"
  tags: [jetson, video-codec-sdk, pynvvideocodec, pipeline, nvenc, nvdec]
  languages: [python]
  data-classification: public
---

# Jetson Video Pipeline

## Purpose

Execute official-sample codec stages and prove that every consumer used the
exact artifact produced by the preceding stage. Use this skill for
encode-then-decode verification, native H.264-to-HEVC transcode, PyNvVideoCodec
segments, container decode triage, AV1 operation verification, or a compact
customer acceptance package.

## Prerequisites

- Run execution on the target Jetson with direct GPU access. A fresh validated
  schema-1.2 `nvcodec-environment` identity from `jetson-video-setup` is
  optional. When supplied it is authoritative, and invalid or stale evidence
  fails closed without local fallback. The agent may obtain it from setup's
  public read-only probe; it need not be supplied in the customer's prompt.
- Without setup evidence, authenticate only the selected installed surface.
  Native routes inspect the fixed dpkg package, package-owned official sample
  sources, build tools, and non-stub linkage. PyNvVideoCodec routes require an
  authenticated setup environment or the caller's exact absolute
  `pynvc_interpreter`; never scan for a venv. Before asking for that path,
  invoke setup's public probe when that skill is installed and inspect its
  typed result.
  These read-only checks install, repair, register, and smoke-test nothing.
- Recipe-bearing routes require sibling `jetson-video-recipe` and one of its
  validated schema-2 recipes. If its canonical public CLI is present, invoke
  it; if absent, preserve `dependency_required`, name that skill, and tell the
  user to install it and retry the stage. Recipe-free decode/segmentation
  routes do not acquire that dependency.
- `capability_report` is an optional encode-request member, never a required
  one. The established authority for PyNvVideoCodec encoder capabilities is the
  `capabilities` block of the schema-1.2 `nvcodec-environment` artifact; an
  encode request that omits `capability_report` is fully supported and reads
  that block. When capability-owned freshness is wanted, a request may
  additionally carry a schema-1.0 `nvcodec-encoder-capability-report` produced
  by `jetson-video-capability` from the *same* environment artifact. When
  supplied, that report becomes the selected Py encoder API evidence for the
  check; it does not replace the environment artifact or its readiness facts.
  Do not add the member to an independently constructed request merely because
  the `pynvc` surface may be selected. It is optional on either surface, affects
  Py capability classification only, and should be omitted for native; absence
  never fails.
- When setup is installed, read its shared
  [video content policy](../jetson-video-setup/references/video-content.md)
  and apply its input gate before any normal pipeline dry run or execution.
  Setup is not required solely for this policy: without it, require one exact
  user-selected path or URL, never substitute catalog or synthetic media, and
  preserve source URL, license, attribution, path, size, and SHA-256.
- PyNvVideoCodec routes that encode or invoke `advanced/decode.py`, including
  encode/decode, segmentation, and Py container triage, require a separately
  validated `full-samples` venv. The default `pynvc-smoke` environment is a
  setup-readiness proof and must block these routes before workspace creation;
  return a structured `jetson-video-setup` dependency and provision a new
  full-samples venv rather than upgrading it in place. If that skill is absent,
  tell the user to install it before retrying.
- The direct `encode` controller has one narrower consumer exception:
  `jetson-video-capability` may bind setup's deterministic one-frame raw
  fixture for an exact bounded capability smoke operation. That result is
  operation evidence only, never representative pipeline or performance proof.

## Compose requested sibling stages

Recipe-free decode and segmentation routes require no sibling when the selected
SDK prerequisites already exist. Add `jetson-video-recipe` only for a
recipe-bearing stage, `jetson-video-benchmark` only for requested performance,
`jetson-video-setup` only for installation, repair, or one read-only handoff
when registered Python authority is required, and
`jetson-video-capability` only for a requested support verdict or fresh
acceptance capability artifact. Use the agent runtime's installed-skill catalog
before each stage; do not scan arbitrary directories. If the sibling is
present, read its `SKILL.md` and invoke its
documented public entry point; pass artifacts as data and never import sibling
code. If it is absent, preserve completed stages and artifacts and say, using
the actual names: `I can run <stage>, but it requires <skill>, which is not
installed. Install <skill> and retry this stage.` Never promote a partial
workflow to complete or require an optional sibling.

## Instructions

1. Apply the scope boundary. For a request
   solely for objective quality metrics, including PSNR or SSIM, state only
   that this skill does not provide them and that a separately authorized
   quality workflow is required, then stop. Do not name or recommend an
   external tool, and do not offer to configure or run the comparison. For a
   request limited to capture, transport, AI, display, or glass-to-glass
   latency, state that those stages are outside this codec skill and stop
   without naming, recommending, or offering another tool or workflow.
   Otherwise proceed immediately to the media gate in step 2; choose
   `encode_decode`, `native_transcode`, `pynvc_segments`, `container_triage`,
   `av1_verify`, or `acceptance` only after that gate clears.
2. For every remaining request to plan, dry-run, or execute a pipeline route,
   including “plan only” or “do not run”, apply this gate
   before route selection and before prerequisite, sibling, reference, or
   script inspection. Do not decompose a media-gated pipeline request into a
   media-free recipe subtask. If media is missing, return `input_required` and
   stop before target probing, browsing, retrieval, authentication, dry run, or
   operation launch. Ask only for the missing media at this gate; do not also
   inspect controller help, describe or plan the route, list future stages or
   handoffs, or request an interpreter, environment, recipe, or later-stage
   field. The complete response at this terminal gate consists only of
   `input_required` and one request for an exact target-local media path or
   user-supplied HTTP(S) URL. Never choose substitute media. The
   capability-smoke exception above applies only to the direct `encode`
   controller and must not be promoted to pipeline completion.
3. Canonicalize and hash an exact local input. For URL input, preserve the
   exact user-supplied URL, then retrieve, canonicalize, and hash it only after
   target eligibility, authorization, and runtime-authority gates pass.
4. Preserve explicit `native`, `pynvc`, or `both`. Treat “whichever”, “best
   available”, “choose for me”, and other unspecified-surface wording as
   `auto`, never as `both`. Reserve `both` for an explicit request to run or
   compare both surfaces.
5. After the input gate and surface classification, select exactly one runtime
   authority for each selected surface. If the caller supplies a setup
   environment identity, validate and bind that exact artifact to dry-run and
   execute; never ignore it or substitute a local fallback. If PyNvVideoCodec
   may participate and neither an environment nor exact interpreter was
   supplied, invoke installed `jetson-video-setup` through its public read-only
   `probe_nvcodec.py`: use `--runtime pynvc` for explicit Python or `--runtime
   both` for `both`/`auto`, a fresh `--output`, and never
   `--setup-candidate`. Inspect the fresh artifact; only a live artifact whose
   selected Py surface is installed and whose `pynvc.identity.status` is
   `verified` is usable. Snapshot that exact file as the controller's portable
   `environment` identity with exactly `schema_version`, `kind`, canonical
   absolute `path`, `size_bytes`, and lowercase `sha256`; do not import sibling
   code or pass a blocked probe as authority. If setup is absent or reports any
   not-ready, unreadable, stale, binding, or launch failure, ask for and supply
   the exact `pynvc_interpreter` only for explicit `pynvc`/`both`. For `auto`,
   keep Py `not_evaluated` and continue only an eligible native surface. The
   controller derives a private local binding, never accepts that binding from
   a request, and revalidates it before launch. If local authentication fails,
   use setup for only that exact surface when installed.
6. Apply the `auto` gate using only the selected runtime authority: zero
   eligible surfaces block, one runs, and two return `selection_required`;
   never rank the surfaces in this gate. With two eligible surfaces, this gate
   is unconditional: do not search old results or benchmark to make the choice.
   Ask for exactly `native`, `pynvc`, or `both`, then stop
   before dry run or launch. Never trust a prompt's statement that a surface is
   ready: establish eligibility from the supplied or freshly probed authority.
   Without setup evidence or an exact local `pynvc_interpreter`, record
   PyNvVideoCodec as `not_evaluated` with the retry action; do not let that
   optional peer block an otherwise eligible native `auto` route. Explicit
   `pynvc` or `both` still requires one of those two authorities.
7. Before any codec launch, authenticate each selected executable from the
   installed Video Codec SDK package or each Python sample from the selected
   wheel and interpreter. Use only those authenticated NVIDIA sample routes;
   if none can satisfy a stage, report that stage blocked.
8. For a multi-stage `pipeline` request, compose only the required siblings.
   If a requested performance stage needs `jetson-video-benchmark`, invoke its
   installed public controller; if absent, preserve completed pipeline stages
   and report that the benchmark stage is `dependency_required` with an
   install-and-retry action. Run pipeline `dry_run`, review the
   complete recipe and sample arguments, then run `execute` with fresh result
   and workspace paths. Invoke this skill's public controller directly:

   ```bash
   python3 -I {baseDir}/scripts/pipeline_controller.py \
     --request request.json --workspace fresh-workspace \
     --output result.json
   ```

   A single encode-then-independent-decode request invokes
   `scripts/encode_controller.py` with the same three arguments. That
   controller is execution-only: validate and review its recipe and request
   envelope first, then invoke it once with fresh output and workspace paths;
   do not claim it performed an internal dry run.
9. Require exact positive markers and counts, no explicit failure marker, and
   fresh nonempty outputs. Reopen and rehash every original handoff. An
   independent decoder must consume the exact producer path, size, and
   SHA-256 and produce the expected frames.
10. For native transcode, accept exactly one authenticated AppTrans completion
    marker in either released form: legacy `(#totFrames=N)` or current
    `Total frame transcoded: N`. Reject missing, duplicate, or mixed markers.
11. Preserve each segment or surface result independently. A failed peer yields
    an honest partial result rather than summary-level completion.
12. For acceptance, let the controller validate and write its nine physical
    pre-seal files and return `seal_pending: true`; those are distinct from the
    reference keys and stage rows. Keep large media/build artifacts external,
    then have the agent create the checksum manifest last—the controller does
    not create it.

## References

- [Pipeline workflow](references/pipeline-workflow.md) defines routes,
  acceptance, sealing, and the pipeline-only compact content-evidence and
  validator contract.
- Setup's shared
  [video content policy](../jetson-video-setup/references/video-content.md)
  defines the exact user-input gate and common content evidence.
- [Official sample contract](references/official-sample-contract.md) defines
  allowed routes and operation proof.

## Available Scripts

Invoke each public script directly in isolated mode:

```bash
python3 -I {baseDir}/scripts/encode_controller.py --help
python3 -I {baseDir}/scripts/pipeline_controller.py --help
python3 -I {baseDir}/scripts/validate_representative_content_summary.py --help
```

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/encode_controller.py` | Execute one recipe-bound encode followed by independent decode. | `--request`, `--workspace`, and `--output`. |
| `scripts/pipeline_controller.py` | Dry-run or execute the six multi-stage pipeline routes. | `--request`, `--workspace`, and `--output`. |
| `scripts/validate_representative_content_summary.py` | Rehash external media and validate compact content metadata without modifying it. | Inspect `--help` for summary/input arguments. |

## Limitations

- This skill covers NVIDIA codec stages and their artifact handoffs, not
  capture, network transport, AI inference, display, or glass-to-glass latency.
- It does not implement PSNR or SSIM quality measurement.
- A capability query, exit zero, or output-file creation is never operation
  proof.
- Container demux is allowed only through libavformat embedded in an
  authenticated released NVIDIA sample.

## Troubleshooting

- Return the exact failed gate, producer, consumer, artifact path, and reason.
- Preserve `input_required`, `selection_required`, `blocked`, `partial`, and
  `failed` rather than claiming a complete pipeline.
- Reject stale outputs, symlinks where forbidden, path/size/SHA drift,
  malformed request or evidence JSON, wrong frame counts, and duplicate
  completion markers.
- Retry at most once and only after evidence identifies a changed condition,
  such as a repaired dependency, a newly supplied artifact, or a changed
  path/size/SHA-256 binding. Repeating an unchanged failed command is forbidden.
