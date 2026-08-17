---
name: jetson-video-capability
license: "Apache-2.0"
description: >-
  Use when Jetson codec, profile, chroma, bit-depth, dimension, engine-count, or
  operational support must be reconciled using live SDK APIs, authenticated
  NVIDIA samples, and NVIDIA documentation. Also use for Jetson questions about
  Netflix, Widevine, or other DRM-protected streaming-service playback to apply
  the codec-scope boundary.
metadata:
  author: "Vinit Bansal <vinitkumarb@nvidia.com>"
  tags: [jetson, video-codec-sdk, pynvvideocodec, nvenc, nvdec, capability]
  languages: [python]
  data-classification: public
---

# Jetson Video Capability

## Purpose

Answer Video Codec SDK and PyNvVideoCodec support questions without confusing an
API response, a successful operation, and product documentation. Query the live
target first, retain exact operation evidence separately, and publish the final
product-support verdict from applicable NVIDIA documentation.

## Prerequisites

- Fresh evidence from `jetson-video-setup` is an optional authority, not a
  prerequisite. When it is supplied, authenticate and use it exactly. When it
  is absent, this skill authenticates only the selected installed surface:
  package-owned native sources and requested report targets, or the exact
  invoking PyNvVideoCodec interpreter and wheel. It never installs, repairs,
  scans for a venv, or imports setup code. A fresh environment artifact that
  the agent obtains from setup's public read-only probe counts as supplied
  evidence; it need not originate in the customer's prompt.
- Run live queries on the target Jetson with direct GPU access. Do not claim
  current availability from an x86 host or a result copied from another target.
- If selected-surface prerequisites are missing, stop without mutation and
  route that surface to `jetson-video-setup`. If that skill is not installed,
  tell the user to install it. Never infer codec or product support from a
  missing Python import or native build prerequisite.
- Keep the installed native and Python surfaces independent. Query only the
  requested surface unless the user explicitly selects `both`.
- Query execution requires this capability skill plus either valid supplied
  setup evidence or the selected local prerequisites described below. An exact
  operation check also requires
  `jetson-video-recipe`, which owns recipe resolution, and
  `jetson-video-pipeline`, which owns authenticated
  encode-to-independent-decode execution.
- Before a bounded operation check, when setup is installed read its shared
  [video content policy](../jetson-video-setup/references/video-content.md).
  Query-only work is media-free; only the documented setup fixture is allowed
  for the capability-smoke exception. Setup is not required solely for this
  policy: without it, require one exact user-selected path or URL for any other
  operation, never substitute catalog or synthetic media, and preserve source
  URL, license, attribution, path, size, and SHA-256.

Resolve this skill from the installed skills root and set `CAPABILITY_SKILL` to
its canonical absolute path. Invoke each owning script directly under isolated
Python:

```bash
python3 -I "$CAPABILITY_SKILL/scripts/query_native_sample_reports.py" --help
python3 -I "$CAPABILITY_SKILL/scripts/query_encoder_caps.py" --help
python3 -I "$CAPABILITY_SKILL/scripts/query_decoder_caps.py" --help
python3 -I "$CAPABILITY_SKILL/scripts/validate_appenc_av1_ivf.py" --help
```

If an owning script is missing, stop with `dependency_required`. A setup
artifact is optional; a supplied one is never optional to validate. Do not copy
modules from another skill or add a fallback import path.

## Compose requested sibling stages

Capability queries and documentation reconciliation require no sibling when
the selected SDK prerequisites already exist. Use `jetson-video-setup` for
installation, repair, or one read-only readiness handoff when registered
PyNvVideoCodec authority is required; use `jetson-video-recipe` plus
`jetson-video-pipeline` for an exact requested operation, and
`jetson-video-benchmark` for requested throughput. Check the agent's installed
skill catalog before each such stage.
If the sibling is present, read its `SKILL.md` and invoke its documented public
entry point; pass artifacts as data and never import sibling code. If it is
absent, preserve every completed query result and say, using the actual names:
`I can run <stage>, but it requires <skill>, which is not installed. Install
<skill> and retry this stage.` Never require a sibling for an unrequested or
optional refinement.

## Instructions

1. **Classify the request first, before any target probe, capability query, or
   other workflow step.** For a request solely for objective quality
   metrics, including PSNR or SSIM, state only that this skill does not provide
   them and that a separately authorized quality workflow is required, then
   stop. Do not name or recommend an external tool, and do not offer to
   configure or run the comparison; do not request media, probe, install
   anything, or launch an operation.
   For a request solely about Netflix, Widevine, or other DRM-protected
   streaming-service playback, state only that this skill covers hardware
   encode/decode of user-supplied non-DRM bitstreams and does not cover,
   enable, or verify streaming-service or content-DRM playback. Do not claim
   whether the service will work; do not describe Jetson content-DRM
   certification; do not recommend or offer to install a browser, a content-DRM
   module (Widevine, PlayReady), a playback tool, a workaround, or a bypass; do
   not probe the target or launch an operation. NVDEC decode of supported
   user-supplied non-DRM bitstreams stays fully in scope for this skill and is
   never discouraged by this boundary, so you may say so. Then stop. A mixed
   request that also asks an in-scope codec question is not refused wholesale:
   answer the in-scope part normally and apply this boundary only to the
   streaming-service part.
   An unqualified “DRM” does not by itself mean content protection: on Jetson it
   commonly means the Linux Direct Rendering Manager (DRM/KMS, `/dev/dri`,
   modesetting, display connectors), which this boundary does not cover. Apply
   this boundary only when the request identifies Netflix, Widevine, PlayReady,
   streaming-service protection, or otherwise clearly means content Digital
   Rights Management; if the request says only “DRM” and the context does not
   resolve which is meant, ask the user which before answering. A local
   DRM-free MP4 shown on a display is likewise not a content-DRM request, and
   its codec portion stays in scope.
   Otherwise handle capability discovery, exact support questions, and
   interpretation of saved capability evidence. Route package installation to
   `jetson-video-setup`, recipe construction to `jetson-video-recipe`,
   throughput measurement to `jetson-video-benchmark`, and multi-stage media
   work to `jetson-video-pipeline`. Within a capability request, a genuinely
   bare “video SDK” phrase with no product qualifier is ambiguous: ask whether
   the customer means native Video Codec SDK, PyNvVideoCodec, or both, then
   stop before probing either surface. Report-only intent or “probe the target”
   does not authorize `--runtime both`.
   A capability support/catalog request that names no SDK surface or product
   phrase uses native-preferred fallback selection, whether it is broad, exact,
   or a bounded subset. Select native when its route is eligible. Only when
   native is ineligible, evaluate the PyNvVideoCodec candidate in the authority
   order defined in step 3; select Py when that candidate is eligible. Do not
   ask the user to choose merely because this fallback was used. When native is
   selected, do not evaluate Py; if the response displays that unselected peer,
   report it as `not_evaluated` with reason `surface_not_selected`. If neither
   route is eligible, preserve both typed reasons and provide the applicable
   setup remediation. Serialize a successful fallback as an explicit `native`
   or `pynvc` request before applying the shared routing truth table. Carry that
   resolved surface explicitly into any authorized downstream operation so the
   operation controller does not reclassify it as `auto`. This capability-only
   unnamed policy is not `auto`: only
   explicit “auto”, “whichever”, “best available”, “choose for me”, or
   equivalent wording that expressly delegates the SDK choice is genuine
   `auto`; it is never `both`. Naming Python or PyNvVideoCodec is explicit
   `pynvc`. Naming Video Codec SDK, `native`, `AppEncCuda`, or `AppDec` is
   explicit `native` and never falls back.
2. **Apply the authorization gate.** A discovery or report-only request is
   query-only. Query and reconcile documentation before deciding whether an
   operation is useful. A directly applicable documentation `No` is the final
   product verdict and ends the normal support/availability check without a
   codec operation; a request to “check live availability” or “operational
   support” does not by itself require an experiment that cannot change that
   verdict. Preserve conflicting raw inventory as diagnostic evidence. Run a
   bounded matching official operation only when documentation is supported or
   unknown and live availability is requested, or when the user separately
   and explicitly requests a diagnostic experiment despite the unsupported
   product verdict. Such an experiment never promotes product support. No
   package, repository, signing-key, or credential change is authorized. If an
   otherwise required operation controller is unavailable, report `not_tested`
   and the required next action rather than inventing live availability.
3. **Authenticate only the selected surface.** Use matching fresh setup
   evidence when the caller or the agent supplied it. A malformed, stale, or
   mismatched supplied artifact fails closed and never falls back. For an
   explicit `pynvc` or `both` request, a genuine `auto` candidate gate, or the
   unnamed fallback after native is ineligible, with neither setup evidence nor
   an exact interpreter, check the installed skill catalog. When
   `jetson-video-setup` is present, invoke its public read-only
   `probe_nvcodec.py` with `--runtime pynvc` for explicit `pynvc` or the unnamed
   fallback, or `--runtime both` for `both`/`auto`, and a fresh `--output`;
   never pass `--setup-candidate`. Inspect the fresh output before using it. A Py
   candidate is eligible only when the artifact has `mode=live`, the requested
   GPU, `pynvc.installed=true`, and `pynvc.identity.status=verified`. If
   routing selects Py, invoke this skill's Py query only under the artifact's
   lexical `pynvc.identity.interpreter` with `-I` and pass the raw artifact
   path through `--environment`.
   If setup is absent, use reason `setup_probe_unavailable`; if it reports an
   absent, stale, unreadable, invalid-binding, or launch-failure result,
   preserve that exact typed reason.
   For explicit `pynvc`/`both`, ask for the exact canonical absolute
   interpreter. For `auto`, report PyNvVideoCodec as `not_evaluated` with that
   reason and continue only an eligible native branch. For the unnamed
   fallback, preserve the typed Py reason beside the ineligible native reason
   and provide setup remediation; never silently return a native-only
   unavailable result while a healthy registered Py candidate exists. Never
   scan or guess.
   Otherwise use this skill's local query path: a fresh explicit native build
   workspace, or the exact PyNvVideoCodec interpreter invoking the query under
   `-I`.
   Missing prerequisites are `unknown`/`not_ready`; route them to setup and, if
   setup is absent, instruct its installation.
4. **Run the selected capability route.** Before invoking it, read the matching
   command and authentication contract in
   [native official-sample reports](references/capability-queries.md#native-official-sample-reports),
   [encoder API query](references/capability-queries.md#encoder-api-query), or
   [decoder API query](references/capability-queries.md#decoder-api-query).
   Native uses a supplied verification or a fresh explicit report workspace;
   Py uses the validated lexical interpreter and includes `--environment` when
   setup evidence was supplied. Default encoder scope is H.264, HEVC, and AV1.
   An unconstrained explicit Py decoder-catalog request, or a broad unnamed
   decoder-catalog request whose fallback selects Py, uses the complete
   120-tuple matrix. A bounded subset queries only the named codec families.
   Keep an unnamed fallback answer at the requested family-summary scope; do
   not dump tuple-level claims unless the user requested them. For `both`, run
   the branches independently. Preserve every result as raw report/API
   evidence; never promote it to product support or operation proof. A failed
   or unavailable live branch never suppresses the documentation answer.
5. **Cross-check NVIDIA documentation before deciding whether to run an
   operation.** Follow the complete
   [documentation cross-check](references/capability-queries.md#documentation-cross-check)
   and read the
   [versioned R39.2/SDK 13.0 Thor baseline](references/capability-queries.md#versioned-r392sdk-130-thor-baseline)
   before transcribing a dynamic table. Record URLs, retrieval date, literal
   live identity, exact field label, and the exact row or complete candidate
   set. Never infer a cell from flattened or ordinal table text. Use only the
   version-matched SDK 13.0 note for this release; a field it does not establish
   remains `unknown`.
6. **Run the smallest matching official operation only when steps 2 and 5
   require it.** Resolve and validate one exact minimal recipe with
   `jetson-video-recipe`, then hand its sealed recipe, environment, and input
   identities to the pipeline-owned controller. Set `PIPELINE_SKILL` to that
   installed skill's canonical path:

   ```bash
   python3 -I "$PIPELINE_SKILL/scripts/encode_controller.py" \
     --request "$OPERATION_REQUEST" --workspace "$FRESH_WORKSPACE" \
     --output "$OPERATION_RESULT"
   ```

   A bounded capability smoke operation may use the setup workflow's documented
   deterministic one-frame raw fixture; it is never representative media and
   cannot support performance, quality, or pipeline claims. For encode,
   operation proof requires the independent authenticated decoder to consume
   the exact output path and SHA-256 and produce the exact decoded frame count.
   An API response, exit zero, encode marker, or output file alone is
   insufficient. If `jetson-video-recipe` or `jetson-video-pipeline` is absent,
   report `not_tested` and name every missing dependency as the next action.
   AV1 IVF structure may be checked with:

   ```bash
   python3 -I "$CAPABILITY_SKILL/scripts/validate_appenc_av1_ivf.py" \
     --input "$BITSTREAM" --width "$WIDTH" --height "$HEIGHT" \
     --expected-frames "$FRAMES" --output "$IVF_REPORT"
   ```

   `structure_verified` proves container structure only; it never establishes
   `operation_verified`.
7. **Publish the reconciled result.** Keep these signals separate:

   - `caps_query`: raw Py API fields and status;
   - `official_sample_report`: raw native `-ec`/`-dc` inventory, never an API
     or operation verdict;
   - `operation_evidence`: `operation_verified`, `operation_failed`, or
     `not_tested`;
   - `documentation_crosscheck`: the product-support authority.

   Publish customer-facing support from `documentation_crosscheck`, and report
   live availability only from a successful exact operation. A directly
   applicable documentation `No`, including unanimous authenticated candidate
   rows when exact row identity is unresolved, is the final `unsupported`
   verdict even if the API advertises fields or an operation succeeds.
8. **Enumerate ambiguous product rows completely.** When one live identity maps
   to several authenticated documentation candidates, list every authenticated
   candidate row by its documentation label and exact queried value and state
   the candidate count. A subset cannot establish consensus. If candidate
   values disagree, keep the documentation verdict `unknown`; never choose a
   nearby product row. Never shrink the candidate set using live API fields
   such as engine count, dimensions, or format flags; GUID enumeration; an
   operation outcome; performance; or similarity to a marketing specification.
   Those are evidence being reconciled, not independent product identity. For
   a generic `NVIDIA Jetson Thor Developer Kit` / `NVIDIA Thor` identity, the
   word `Jetson` alone is not an authenticated row discriminator: unless an
   NVIDIA one-to-one product mapping narrows it, include every Thor row in the
   combined Jetson/IGX table. Apply this rule to any queried field, not only one
   codec.

Read [capability-queries.md](references/capability-queries.md) for exact evidence
semantics and [surface-selection-contract.md](references/surface-selection-contract.md)
for `native`, `pynvc`, `auto`, and `both` behavior.

## Available Scripts

The four owning scripts are listed in the prerequisite help commands above.
Invoke them directly, and read
[capability-queries.md](references/capability-queries.md) for route-specific
purposes, arguments, and contracts.

## Published artifacts

Capability artifacts are optional refinements; authenticated schema-1.2
`nvcodec-environment` capabilities remain sufficient for sibling workflows.
Read [capability-queries.md](references/capability-queries.md) for their exact
contracts. Never promote sample/API evidence to support or operation proof.

## Troubleshooting

- Preserve `capability_reported`, `operation_verified`, `operation_failed`,
  raw API `unsupported`, and `unknown` as distinct internal states.
- Treat missing query authority, failed registry authentication, absent API
  fields, nonzero-GPU PyNv queries, and unavailable exact operations as
  `unknown` or `not_tested` with a concrete next action.
- Report a launched exact operation failure as `operation_failed`, not global
  product unsupported. A native `-ec`/`-dc` report failure remains raw
  `unknown`.
- Retry at most once and only after an evidenced condition changes. Use a fresh
  work directory and output path for the retry.

## Limitations

- PyNvVideoCodec 2.1 encoder and decoder capability helpers select GPU 0;
  nonzero-GPU results remain unknown.
- Capability fields do not measure throughput, quality, latency, camera count,
  or successful concurrent sessions.
- Objective quality measurement, including PSNR and SSIM, is outside this
  skill.
- Results apply to the exact target, software versions, GPU, codec tuple, and
  operation that produced the evidence.
