---
name: jetson-video-pipeline
license: "Apache-2.0"
description: >-
  Use when planning, executing, and independently validating Jetson Video
  Codec SDK or PyNvVideoCodec encode/decode, transcode, segmentation,
  container decode, AV1, or concise acceptance workflows.
metadata:
  author: "Vinit Bansal <vinitkumarb@nvidia.com>"
  tags: [jetson, video-codec-sdk, pynvvideocodec, pipeline, nvenc, nvdec]
  languages: [markdown]
  data-classification: public
---

# Jetson Video Pipeline

## Purpose

Build and run direct NVIDIA sample commands, then prove each consumer used the
exact bytes produced by the preceding stage. This skill owns codec workflow and
evidence policy; it does not own environment installation or product-support
claims.

## Terminal gates

Apply these before inspecting the target, another skill, or a command:

1. For a PSNR/SSIM-only request, state that objective quality measurement is
   outside this skill and requires a separately authorized workflow, return
   `not_evaluated` with reason `out_of_scope`, then stop. Do not name a tool or
   request media. Do the same for a request limited to capture, transport, AI,
   display, or glass-to-glass latency. For a mixed codec-plus-quality request,
   continue only the codec portion and report the quality portion as
   `not_evaluated` with reason `out_of_scope` rather than silently omitting it.
2. Classify every other request before applying the media gate:
   - Return `planned` for architecture and route questions; they need no media.
     State assumptions and what execution would verify, then stop before target
     inspection, retrieval, authentication, recipe dispatch, or workspace
     creation.
   - A dry run with concrete commands needs a user-named path or placeholder
     and the metadata required by those commands. The media need not exist.
     Obtain recipe options through `jetson-video-recipe`, report `planned`, and
     do not inspect, authenticate, or launch anything.
   - Execution, verification, and performance require one exact target-local
     media path or one user-supplied HTTP(S) URL. Without it, return
     `input_required`, identify the intended route briefly using only stages
     expressible by this skill's allowlisted samples, ask for that one item,
     and stop before target inspection, retrieval, authentication, recipe
     dispatch, or workspace creation. Label every other requested transform as
     unresolved rather than inventing an executable route.

   Never choose catalog or synthetic media. The deterministic setup smoke
   fixture is allowed only for a bounded capability operation and is never
   representative pipeline or performance evidence.

## Select and authenticate the execution surface

Steps 3–4 apply to execution; step 5 also applies to recipe-bearing dry runs.
For other planning, preserve an explicit surface without claiming live
eligibility; delegated surface-specific dry runs return `selection_required`
without ranking the choices.

3. Preserve explicit `native`, `pynvc`, and `both`. Map “whichever”, “best
   available”, “choose for me”, or otherwise delegated selection to `auto`, not
   `both`. For `auto`: zero eligible surfaces is `blocked`, one runs, and two is
   `selection_required`; do not rank them or consult old results.
4. Obtain a fresh read-only readiness result from `jetson-video-setup` through
   public skill dispatch. For Python, pass any exact user-supplied or
   current-conversation interpreter. Otherwise select the profile before
   dispatch: decode-performance may use `pynvc-smoke`; encode, segmentation,
   `advanced/decode.py`, pipeline, and encode-benchmark work require
   `full-samples`. Setup checks that profile's conventional path; never scan
   for a venv. If only the smoke profile is ready for full-samples work, return
   `dependency_required` before workspace creation and direct the user to
   provision a separate full-samples venv; never upgrade the smoke venv in
   place. Native eligibility requires one
   installed, package-verified SDK and one package-owned Samples root. Python
   eligibility requires that exact interpreter, an importable PyNvVideoCodec
   distribution loaded from its environment, and a clean `pip check`. The
   result must match the requested Jetson and GPU. Preserve the setup reason
   when a candidate is ineligible. The requested pipeline supplies its own
   operation proof.
5. Recipe-bearing encode and transcode work requires `jetson-video-recipe`;
   an acceptance request containing a performance stage also requires
   `jetson-video-benchmark`. Invoke either through public skill dispatch and
   pass its result as data. If a needed sibling is absent, preserve completed
   stages and say: `I can run <stage>, but it requires <skill>, which is not
   installed. Install <skill> and retry this stage.`

   “Preserve” means retain each completed stage's status and current
   path/size/SHA-256 identities in the user-facing partial result; it does not
   create hidden resumable state. On retry in the same request, reopen and
   rehash those artifacts and skip only unchanged `complete` stages. With a
   changed/missing identity, or a later request that does not supply the prior
   evidence, use a new workspace and rerun the stage.

## Execute

1. Choose the smallest route: `encode_decode`, `native_transcode`,
   `pynvc_segments`, `container_triage`, `av1_verify`, or `acceptance`.
2. Read [pipeline-workflow.md](references/pipeline-workflow.md) for stage order,
   content handling, retry policy, and reporting. Read
   [official-sample-contract.md](references/official-sample-contract.md)
   completely once; it covers sample allowlists, direct arguments, marker
   grammar, and frame-layout checks. Read
   [buffer-sharing-and-synchronization.md](references/buffer-sharing-and-synchronization.md)
   completely once for filesystem boundaries. For a custom in-process
   CUVID/CUDA/NVENC graph, also read
   [in-process-codec-boundaries.md](references/in-process-codec-boundaries.md)
   completely once.
3. For execution, canonicalize the exact media, record its provenance and
   identity, and use a new mode-0700 workspace with fresh outputs. For a dry
   run, use only the supplied identity and metadata, state assumptions, and
   show planned commands and handoffs without opening the media or
   authenticating launchers.
4. For execution, authenticate each selected native sample or wheel member,
   launch its literal argument list directly, and retain unedited logs. Require
   the reference's exit, marker, count, error, freshness, and layout checks.
5. Reopen and rehash every producer output before and after its independent
   consumer. Preserve successful branches, report a failed peer as `partial`,
   and apply the reference's single-retry rule.
6. Emit the applicable `io_contract` directly in every plan, result, and
   producer/consumer boundary. Do not depend on a contract module or infer
   external sharing from a device-memory mode.

## Route requirements

| Route | Required proof |
|---|---|
| `encode_decode` | One validated recipe; direct AppEncCuda→AppDec or wheel-owned basic encode→advanced decode; exact raw and decoded byte counts. |
| `native_transcode` | H.264 input, exact HEVC native projection, AppTrans output, exactly one accepted transcode marker, then AppDec over the same hash; require the AppTrans and AppDec frame counts to be equal and positive, and to equal the known input count when available. |
| `pynvc_segments` | Wheel-owned schedule/config; every declared segment is fresh and nonempty; decode and rehash every segment independently. |
| `container_triage` | Eligible Jetson, exact local/retrieved container, intrinsic libavformat demux in AppDec or wheel-owned advanced decode, fresh decoded output. |
| `av1_verify` | Exact AV1 native recipe; host and video-memory modes; AppDec consumes each exact IVF output and reports the expected frame count. |
| `acceptance` | A concise reproducible report covering requested readiness, capability, recipe, codec, and benchmark stages with per-stage status and identities. |

## Report

Use the statuses and concise report defined in
[pipeline-workflow.md](references/pipeline-workflow.md). Include the selected
runtime, exact media and recipe identities, literal commands, producer and
consumer results, decoded layout/size validation, logs, limitations, and retry
reason. Add compact JSON or a checksum manifest when useful or requested.

## Limitations

- Codec work does not prove capture, transport, inference, display, quality, or
  end-to-end latency.
- API fields and inventory are not operation proof or product support.
- Container demux is allowed only inside an authenticated released NVIDIA
  sample.
- Apply every deterministic acceptance check in the references.
- Codec/API capability and benchmark results do not prove buffer
  interoperability, zero copy, or synchronization compatibility.
