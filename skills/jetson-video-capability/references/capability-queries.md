# Capability query guidance

## Classification

| Evidence | Meaning | Never means |
|---|---|---|
| `api_query_helper` | A live SDK API returned raw fields. | Operation success or product support. |
| `official_sample_report` | An authenticated NVIDIA sample emitted its report grammar. | API truth, exact operation success, or product support. |
| `official_sample_operation` | The exact authenticated tuple passed observable encode/decode checks. | Other tuples or documented product support. |
| `documentation_reference` | An applicable NVIDIA row/field was captured. | Current installation readiness. |

Keep missing, malformed, unavailable, or failed authority `unknown`. Use
`unsupported` as a product verdict only for applicable documentation `No`; a
raw decoder API false remains explicitly an API result.

## PyNvVideoCodec API query

Use the exact interpreter returned by the setup readiness check. Before and
after querying, record the interpreter path, `PyNvVideoCodec` distribution
version, imported module path, and requested GPU. Require the module path to be
inside that interpreter's environment. Do not set import paths, scan for a
venv, or switch to system Python.

Create a short task-local query from the installed public API and this
reference. Inspect the installed API signature when necessary rather than
guessing enum or argument names. For a broad encoder catalog, call
`GetEncoderCaps` for `h264`, `hevc`, and `av1`; a request naming specific
members of that set queries only those. Call `GetDecoderCaps` for the scoped
tuple set below. Preserve the raw return values and per-call exceptions. Keep
the program task-local and apply the user's normal workspace cleanup policy.

A successful `GetEncoderCaps` record is `capability_reported`,
`supported=null`, and `operation_status=not_tested`. Fields absent from the
installed extension are unknown rather than unsupported.

A broad decoder catalog covers these 120 unique tuples:

- codecs: `mpeg1`, `mpeg2`, `mpeg4`, `vc1`, `h264`, `hevc`, `vp8`, `vp9`,
  `av1`, `jpeg`;
- chroma: `monochrome`, `420`, `422`, `444`;
- bit depths: `8`, `10`, `12`.

Require exactly 120 unique attempted tuples for a broad catalog. A request
naming specific decoder families queries only those families across the same
four chroma formats and three bit depths, states the attempted count, and skips
nothing within that scope.
`bIsSupported=1` makes the raw record `capability_reported`, `supported=true`,
and its limits applicable. Zero makes the raw record `supported=false`; this
is a namespaced API result, never the product verdict, and the remaining
zeroed output fields are inapplicable. A missing enum, call, or flag is
`unknown`.

If the public helper selects only GPU 0, retain every requested nonzero-GPU
record as unknown rather than relabeling GPU-0 facts.

## Native official-sample reports

For a query-only request, do not run setup's encode/decode smoke merely to
obtain reports. Reuse current authenticated package-owned report binaries when
available, or create a fresh user-owned report build from the single
package-owned `nvidia-video-codec-sdk` 13.0.x Samples tree. Follow the
pipeline [native build contract](../../jetson-video-pipeline/references/official-sample-contract.md#build-package-owned-native-samples),
building only `AppEncCuda` and/or `AppDec`; building report binaries is the
only mutation and does not authorize package installation, encode, or decode.
Recheck the binary SHA-256 before and after, require real `ldd` resolution of
`libcuda.so.1` plus `libnvidia-encode.so.1` for AppEncCuda or
`libnvcuvid.so.1` for AppDec, reject stub paths, use a clean bounded
environment, and invoke exactly:

```bash
"$APPENC" -ec
"$APPDEC" -dc
```

If package, source, tool, or library authentication fails, report unknown and
request setup repair. Never execute an encode/decode operation under
query-only authorization.

Accept `AppEncCuda -ec` only when exit is zero, timeout is false, no explicit
CUDA/NVENC/error/failure marker appears, and one recognized grammar is exact:

- legacy: one `Encoder Capability Summary`, one detail hint, unique
  `GPU <ordinal> - <name>` blocks including the selected GPU, one codec-support
  and capability-summary section per GPU, and exactly one H264/HEVC/AV1 row;
- SDK-13 compact: one `Encoder Capability`, unique GPU blocks including the
  selected GPU, and one basic H264/HEVC/AV1 `yes|no` row per GPU.

Accept `AppDec -dc` under the same process rules and either:

- legacy: unique GPU blocks including the selected GPU, one
  `GPU Decoder Capabilities` and `Codec Support Summary` per GPU, and one
  detail hint;
- SDK-13 compact: one `Decoder Capability`, one `GPU in use: <name>`, at least
  one `Codec ... BitDepth ... ChromaFormat ... Supported ...` row, and every
  `Codec`-prefixed line matching that strict grammar. Because this form has no
  ordinal, it applies only to requested GPU 0.

Preserve raw row values and recognized format variant. Native reports are
independent: one failure must not erase the other. Never synthesize the Py
decoder matrix from `-dc` output or translate a sample `yes`, `no`, or numeric
`Supported` field into a product verdict.

## Documentation reconciliation

Identify the live product independently from capability values. Prefer exact
immutable device-tree/product evidence; never infer a SKU from GPU name,
memory, engine count, queried limits, operation behavior, or similarity. A
generic Thor identity remains a family identity.

Read the NUL-terminated device-tree nodes with `tr '\0' '\n'`.
`/proc/device-tree/compatible`, or its
`/sys/firmware/devicetree/base/compatible` mirror, supplies exact board/module
identifiers when present; `/proc/device-tree/model` supplies only the family
label. Record the exact path and literal values beside the documentation row.

Open each applicable official URL at execution time with the agent's web
retrieval tool. Use the live page, not a cached excerpt or model memory. If a
page cannot be retrieved or its row-to-column binding cannot be verified,
record the attempted URL, UTC time, failure reason, and affected fields as
unknown; do not reconstruct the values.

Use the official links in `SKILL.md`; do not preserve a copied product table in
the skill. Record retrieval date, URL, table title, all header levels, exact row
label, exact field label, and literal cell. If extraction loses the
row-to-column binding, the field is unknown. Never use an application note from
another SDK release.

Select an HTML table by its own exact caption or section anchor, not the first
raw-text occurrence of its title, which may be a navigation link. Parse only
that closed table element and require one ordered header binding for each cell.
Repeated headers, duplicate matching tables, a row-width mismatch, extraction
that crosses table boundaries, or conflicting cells makes the field unknown;
never choose one value. For codec-specific tables, do not reuse rows from a
preceding H.264 or HEVC table when evaluating AV1.

Capture documentation evidence manually as part of the agent workflow; there
is no scraper or copied table to maintain. Retain one compact record per claimed
field with: UTC retrieval time, source URL and title, SDK release, ordered
header path, exact row label, exact field label, literal cell text, the live
product-identity evidence used for applicability, and applicability as
`exact`, `candidate_consensus`, or `unknown`. A missing value makes that field
unknown rather than permission to reconstruct it from memory.

An authenticated candidate row is a current-document row whose product label
is consistent with every fresh immutable device-tree or product identifier
observed from the live target. Record those identifiers and their source paths
beside the row. GPU name, memory, engine count, API fields, and test outcomes
cannot authenticate or exclude a row. When immutable evidence identifies one
exact product, use only its exact row and do not invoke consensus.

For a generic `NVIDIA Jetson Thor Developer Kit` or `NVIDIA Thor` identity,
authenticate every matching Jetson/IGX Thor row. Publish candidate-row
consensus only when all candidates agree for the exact field; do not narrow
candidates using the query or test outcome.

The support matrix and application note may differ or omit an exact tuple.
Preserve disagreement instead of choosing silently, and never expand a
family-level statement into an undocumented chroma/bit-depth/profile claim.

## Exact operation handoff

Documentation `No` ends the normal availability check. When documentation is
positive or unknown and the user asks for live availability, obtain a schema-2
recipe and follow the pipeline official-sample contract. Require exact input
geometry/format/frame count, fresh nonempty output, anchored success markers,
and independent authenticated decode of the same path/size/SHA-256. Container
structure, exit zero, and an encode marker alone are insufficient.

If the user explicitly requests a diagnostic despite documentation `No`, run
only the exact requested tuple and label its result separately as
`operation_verified` or `operation_failed`. The documentation-derived product
verdict remains unsupported regardless of that diagnostic result.

For AppEncCuda AV1 output, independently decode the exact produced path with
the package-owned decoder as the pipeline reference requires. Container
structure alone deliberately carries `operation_verified=false`; only the
independent decode, with a positive frame count matching the request, supports
`operation_verified=true`.
