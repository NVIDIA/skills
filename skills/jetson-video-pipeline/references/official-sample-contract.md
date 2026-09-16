# Official sample routes and provenance

Only released NVIDIA samples may perform codec work. This skill keeps its exact
pipeline subset beside its authenticator:

- `scripts/_pipeline_provenance.py:NATIVE_ROUTES` owns the native package
  sample paths, source/CMake paths, and required runtime libraries used here.
- `scripts/_pipeline_provenance.py:PYNVC_ROUTES` owns the PyNvVideoCodec 2.1
  wheel sample paths and required `RECORD` members used here.

The allowlists are static routing facts, not readiness or operation proof. Every selected
file is still authenticated from the current installed package or wheel before launch,
and its path, size, and SHA-256 are rechecked immediately before execution.
Authenticating a route never selects or acquires non-smoke media. Recipe execution,
performance/benchmarking, content-sensitive comparison, and pipeline routes require an absolute
target-local path or exact user-supplied HTTP(S) URL, never catalog or synthetic substitution, and
preserved provenance and identity. When setup is installed, also apply its shared
[video content policy](../../jetson-video-setup/references/video-content.md).
Missing input is `input_required` and pauses before route authentication or
launch. Only setup/readiness and minimal capability-operation smoke tests may
use the documented deterministic synthetic fixture.

## Active routes

Pipeline-native routes are `AppEncCuda`, `AppDec`, and `AppTrans`.

Pipeline PyNvVideoCodec routes are `samples/basic/encode.py`,
`samples/advanced/decode.py`, and `samples/basic/create_video_segments.py`.

The current public controllers intentionally use smaller route subsets:

- pipeline encode: `AppEncCuda` + `AppDec`, or basic encode + advanced decode;
- multi-stage pipeline: `AppEncCuda`, `AppDec`, `AppTrans`, basic segments, and
  advanced decode.

Do not route a sample merely because it appears in an SDK source tree. It must be present
in the installed package or selected wheel and implemented by the chosen controller.

## Evidence classifications

Keep these observations separate:

- `api_query_helper`: API fields only; never operation proof or live availability;
- `official_sample_report`: an authenticated official sample report mode;
- `official_sample_operation`: an authenticated sample actually executed;
- `documentation_reference`: product-support authority for the exact tuple, but never live
  readiness, availability, or operation proof;
- `absent_release_sample`: not shipped by the authenticated release payload;
- `source_tree_candidate`: visible in source, but not yet an installed runnable route.

An operation is verified only when its exact positive markers and counts are present, no
failure marker appears, the requested output is fresh and nonempty, and the independent
consumer verifies the same path and SHA-256. Exit zero, import success, or output creation
alone is insufficient.

## Updating a route

Changing either allowlist requires matching controller support and focused tests for its
package/wheel identity, exact argv, help-advertised options, markers, fresh outputs, and
producer-to-consumer handoff. Never add a compatibility alias or fallback that silently
changes the requested codec, surface, format, or operation.
