# DOCA AES-GCM workflows

**Where to start:** The verbs run `configure → build → modify → run
→ test → debug`. Skip ahead only when the user is already past a
verb. The `## test` verb is an iterative loop (cap check →
known-vector round-trip → small-bulk → full-bulk → loop back if the
key size or buffer sizing changed), not a one-shot pass — see the
eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the underlying capability surface, task
types, key-size surface, capability-query rules, AEAD semantics,
error taxonomy, observability, and safety / path-selection policy,
see [CAPABILITIES.md](CAPABILITIES.md). For the cross-library DOCA
patterns layered under everything below (the universal lifecycle,
the cross-library `DOCA_ERROR_*` taxonomy, the modify-a-shipped-
sample workflow), see
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through the
steps in order, verifying preconditions before recommending the
next call.

## configure

Goal: bring up a DOCA AES-GCM context on a host or BlueField and
confirm the device's accelerator supports the task types and key
sizes the user actually intends to use.

Steps the agent should walk the user through:

1. **Confirm the installed DOCA version.** Use the procedure in
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).
   Quote the version observed (`pkg-config --modversion
   doca-aes-gcm`, then `doca_caps --version`); do not assume
   "latest".
2. **Discover the device capability surface for AES-GCM.** Run
   `doca_caps --list-devs` to see which devices have AES-GCM
   capability, then run the per-`doca_devinfo` `doca_aes_gcm_cap_*`
   queries against the candidate device. Record at minimum:
   `doca_aes_gcm_cap_task_encrypt_is_supported(devinfo)`,
   `doca_aes_gcm_cap_task_decrypt_is_supported(devinfo)`, and for
   each key type the user is considering
   (`DOCA_AES_GCM_KEY_128` and/or `DOCA_AES_GCM_KEY_256` —
   AES-192 is not in the enum and is not supported),
   `doca_aes_gcm_cap_task_encrypt_is_key_type_supported(devinfo, key_type)`
   and the matching `_decrypt_is_key_type_supported`.
   Also record
   `doca_aes_gcm_cap_task_encrypt_get_max_buf_size(devinfo)`. The
   capability surface to compare against lives in
   [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes).
3. **Decide whether to offload at all.** Per the path-selection
   bullets in [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy),
   doca-aes-gcm is the right answer only when the input is bulk,
   sustained, or already pinned in `doca_mmap` memory. For a tiny
   one-shot encryption, recommend CPU + AES-NI; for a non-GCM AES
   mode, recommend CPU + OpenSSL — do not invent a doca-aes-gcm
   use case the user did not ask for.
4. **Pick the task types per side.** Encrypt-only, decrypt-only, or
   both (the symmetric peer case). The pick decides the next-step
   code shape; the trade-off table lives in
   [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes).
   Do not pick *for* the user when their intent is ambiguous —
   ask whether the program is a producer, a consumer, or both.
5. **Surface the AEAD shape to the user.** Before any code, the
   agent must surface the AEAD-input contract: encrypt takes key +
   IV + AAD (optional) + plaintext, produces ciphertext + auth tag;
   decrypt takes key + IV + AAD (optional) + ciphertext + expected
   tag, verifies the tag, produces plaintext. The user must know
   where each of these comes from (key from their KMS, IV per the
   protocol, AAD per the protocol, plaintext / ciphertext from the
   data flow) before any task config is meaningful.
6. **Configure the AES-GCM instance.** Mandatory before
   `doca_ctx_start()`: enable at least one task type
   (`doca_aes_gcm_task_encrypt_set_conf` and/or
   `doca_aes_gcm_task_decrypt_set_conf`); set source mmap
   permissions (`doca_mmap_set_permissions` to include
   `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY`); set destination mmap
   permissions (`DOCA_ACCESS_FLAG_LOCAL_READ_WRITE`); size the
   per-submission plaintext to at most
   `doca_aes_gcm_cap_task_encrypt_get_max_buf_size(devinfo)`. Per
   the matrix in [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy).
7. **Sanity check before any task submission.** Confirm with the
   user: which task type(s), which key size, source-buffer size,
   destination-buffer size, where the key + IV + AAD come from.
   Run a known-vector round-trip (a published AES-GCM test vector
   from NIST GCMVS or RFC 5288) and verify the ciphertext + tag
   match on encrypt and the plaintext recovers on decrypt before
   any user data flows. If any step fails with a `DOCA_ERROR_*`,
   route through the error taxonomy in
   [CAPABILITIES.md ## Error taxonomy](CAPABILITIES.md#error-taxonomy)
   before retrying.

For the canonical DOCA universal lifecycle that underlies steps 4-7,
see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).
This skill adds the AES-GCM overlay; do not re-explain the lifecycle
here.

## build

Goal: produce a binary that links DOCA AES-GCM against the user's
installed DOCA, using the canonical cross-library build pattern.

The build pattern for any DOCA C/C++ consumer is **identical**
across libraries — `pkg-config` for include + link flags, meson or
CMake as the build system — and is fully documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the AES-GCM-specific overlay:

| Slot | Value for AES-GCM | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-aes-gcm` | The library's `.pc` file installed by the DOCA host packages |
| Required runtime libs | `libdoca-common`, `libdoca-aes-gcm`, plus whatever `pkg-config --libs doca-aes-gcm` resolves to | AES-GCM depends on Core; the link line should not pull in unrelated DOCA libraries |
| Header check | The public header that `pkg-config --cflags` for this artifact resolves to actually exists on disk at the path pkg-config reports (do not hardcode the include path) | If `pkg-config --cflags doca-aes-gcm` resolves but the include is missing, the install is partial |
| Minimum required DOCA version | Query with `pkg-config --modversion doca-aes-gcm`; never hardcode in build files | Cross-version build/runtime mixing breaks per [CAPABILITIES.md ## Version compatibility](CAPABILITIES.md#version-compatibility) |

For non-C consumers (Rust, Go, Python), the link surface is the
same `*.so` files; the FFI wrapper layer is the language-specific
binding and is out of scope for this skill — but the four slots
above are still the load-bearing inputs the wrapper needs.

## modify

Goal: take a shipped DOCA AES-GCM sample as the verified starting
point and apply a minimum-diff modification to express the user's
intent.

The universal modify-a-shipped-sample workflow lives in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify).
Use it as-is. The AES-GCM-specific overlay is the *modify-from-sample
schema fill* — the five slots the agent must elicit from the user
before recommending any code-level edit:

| Slot | What the agent asks the user | AES-GCM-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which sample under `/opt/mellanox/doca/samples/doca_aes_gcm/`? | Pick the closest in *task direction* (encrypt vs decrypt vs both) to the user's intent. Do NOT bridge across both axes — a smaller diff is always safer than a re-architecture |
| 2. Task type added or removed | Which task type from the two? | Each added type needs its own `doca_aes_gcm_task_*_set_conf` call before `doca_ctx_start()`, plus its matching cap-query in [`## configure`](#configure) step 2 |
| 3. Key size change | Switching from AES-128-GCM to AES-256-GCM? (AES-192-GCM is not in the library — `enum doca_aes_gcm_key_type` only has `DOCA_AES_GCM_KEY_128` and `DOCA_AES_GCM_KEY_256`. If the user actually needs AES-192, route to a CPU library.) | Re-run `doca_aes_gcm_cap_task_encrypt_is_key_type_supported(devinfo, new_key_type)` (and the matching `_decrypt_is_key_type_supported`); the key buffer length the user provides MUST match the new key type in bytes (16 for `_KEY_128`, 32 for `_KEY_256`). The agent must remind the user that hard-coding key bytes in the diff is a leak; the key comes from the user's KMS, not the source file |
| 4. Buffer-size changes | Plaintext size per submission changing? | Per-submission plaintext must be ≤ `doca_aes_gcm_cap_task_encrypt_get_max_buf_size(devinfo)`; if the user has larger records, the application layer must fragment (this library does not auto-fragment). Over-broad mmap permissions are a silent security regression |
| 5. AAD shape change | Adding / removing / resizing the additional-authenticated-data field? | AAD must match exactly between encrypt and decrypt for the tag to verify. If the protocol the user is implementing binds a sequence number or header into the AAD, both ends must agree byte-for-byte. AAD mismatches surface only as decrypt tag-verification failures, not as configure-time errors — they are silent until decrypt |

The agent emits an *intent description + the five filled slots*;
the *actual* unified diff against the sample source is produced by
the modify-from-sample renderer (deferred to a future round). Until
the renderer ships, the agent must walk the user through the diff
line-by-line against the sample source they read on disk, and have
the user paste back the result for validation. **Special hazard for
AES-GCM modifications:** the agent must explicitly flag if any
proposed diff includes hard-coded key bytes — that diff is unsafe
to commit regardless of whether it builds.

## run

Goal: actually execute the built binary against the user's
installed DOCA on a host or BlueField, with a real input.

Steps the agent should walk the user through:

1. **Confirm the device is reachable.** AES-GCM runs on a single
   side (no peer); the only env-side requirement is that the
   `doca_dev` the binary opens corresponds to a device whose
   accelerator the user expects. Mismatched `doca_dev` selection
   (opening a NIC without AES-GCM accelerator support) returns
   `DOCA_ERROR_NOT_SUPPORTED` at task submit, not at open.
2. **Run the known-vector round-trip first.** A binary that encrypts
   + decrypts one short fixed input with a published key + IV + AAD
   and compares both directions against the published expected
   output is the cheapest correctness signal. Do not bulk-encrypt
   or bulk-decrypt before this passes.
3. **Capture the structured log.** Set `DOCA_LOG_LEVEL=trace` for
   the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability)).
   This is the cheapest way to make the lifecycle and task-submit
   transitions visible on first failure. **Do not log the key bytes
   themselves** — the trace flag is for the library, not the user's
   key buffer.
4. **Capture the completion events on the PE.** A run that produces
   no completion events but doesn't error is almost always a missed
   `doca_pe_progress()` call. Confirm the progress engine is being
   driven on the main thread. For decrypt completions specifically,
   inspect the per-completion result for the tag-verification
   status before reading the plaintext.

## test

Goal: prove the configured AES-GCM context can actually produce
correct ciphertexts (on encrypt) and verify + recover plaintexts
(on decrypt) at the user's intended throughput, on the user's
hardware, and that the key size and buffer sizing were right.

> **Performance harness routing.** For *throughput / latency
> measurement* on the configured AES-GCM context (or for cross-library
> comparison against the other DOCA crypto primitives), route the
> user to [`doca-bench TASKS.md ## test`](../../tools/doca-bench/TASKS.md#test)
> — `doca-bench` is the cross-library performance harness with
> documented warm-up / steady-state / outlier semantics, and it
> explicitly supports AES-GCM. The iteration loop below stays the
> *correctness* harness; `doca-bench` is the *performance* harness.
> Hand-rolling a benchmark loop inside this skill's `## test`
> would re-invent what `doca-bench` already does and would not be
> reviewable against published accelerator headroom.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the task type, the key size, the buffer sizing, the AAD
shape, or the permission set. The loop terminates when either (a)
the user's intended AES-GCM workload encrypts + decrypts correctly
end-to-end with acceptable throughput, or (b) the agent has narrowed
the failure cause to a layer outside DOCA AES-GCM itself (driver /
firmware / device) and escalated to the matching skill.

Iteration shape:

1. **Capability re-check.** Re-run
   `doca_aes_gcm_cap_task_encrypt_is_supported`,
   `_task_decrypt_is_supported`,
   `_task_encrypt_is_key_type_supported(devinfo, key_type)` for
   each key type in use (`DOCA_AES_GCM_KEY_128` /
   `DOCA_AES_GCM_KEY_256`), and
   `_task_encrypt_get_max_buf_size` against the active
   `doca_devinfo`. If any return false / unexpected → that's the
   answer; the user's device or DOCA version does not support the
   requested config. Update the intent or update the install.
2. **Permission cross-check.** Compare the configured source +
   destination mmap permissions against the matrix in
   [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy).
   Mismatches surface as `DOCA_ERROR_NOT_PERMITTED` on the first
   task submission, not at configure time.
3. **Known-vector round-trip.** Encrypt one short fixed plaintext
   with a published key + IV + AAD and compare the accelerator's
   ciphertext + tag against the published expected ciphertext + tag
   byte-for-byte. Then decrypt the published ciphertext + tag with
   the same key + IV + AAD and verify the plaintext recovers
   byte-for-byte. If either direction differs, the configuration is
   wrong — do not proceed to bulk input.
4. **Completion drain.** Confirm completion events arrive on the PE
   for every submitted task. *Submitted but no completion* is the
   most expensive class of bug to discover late; confirm it on the
   known-vector round-trip before bulk submissions.
5. **Bulk encrypt test.** If the user intends bulk encryption,
   submit a series of encrypt tasks (small inputs first, then sizes
   approaching `_task_encrypt_get_max_buf_size`) and verify the
   ciphertexts + tags against a CPU AES-GCM reference (OpenSSL
   `EVP_aes_*_gcm`). Throughput numbers come from this step;
   correctness comes from step 3.
6. **Bulk decrypt test (with tampering negative).** If the user
   intends bulk decryption, submit the matching ciphertexts + tags
   and verify the plaintexts recover. Then submit ONE deliberately
   tampered ciphertext (flip a single bit) and verify the
   completion reports a tag-verification failure. This validates
   that the agent's tag-check observability is wired up correctly —
   silently accepting a tampered ciphertext is the worst failure
   mode this library exposes.
7. **Negative test for unsupported config.** Once the positive path
   works, intentionally request a key size the device should NOT
   support (per step 1) and confirm the failure is the expected
   `DOCA_ERROR_NOT_SUPPORTED`. This validates the agent's
   capability-discovery is itself correct.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_NOT_SUPPORTED` on a key size we expected to work | The docs list the key size but the cap query returns false | The agent quoted the *library* surface; the *device* capability per `doca_devinfo` is the real gate. Re-narrow to the device-level query. |
| `DOCA_ERROR_INVALID_VALUE` on first submit | Plaintext is larger than `_task_encrypt_get_max_buf_size`, OR the auth-tag length the user passed does not match the AES-GCM spec, OR the key length does not match the declared key size | Re-size the buffer using the cap-query output, or correct the tag / key length. The error is sizing-vs-cap mismatch, not corruption. |
| Known-vector encrypt produces a wrong ciphertext or tag | Configuration accepted but output mismatches the published vector | Key-size mis-selection (asked for AES-256-GCM, configured AES-128-GCM), wrong IV length, or AAD bytes not matching the vector. Re-check the key length, IV length, and AAD bytes against the published vector before any other diagnosis. |
| Known-vector decrypt fails to recover the plaintext | Encrypt round-trips fine but decrypt with the published ciphertext + tag fails verification | Either the AAD passed to decrypt does not match the AAD used at encrypt, or the expected-tag pointer is wrong, or the key bytes differ between the two sides. AAD mismatch is the most common cause. |
| Submitted task produces no completion | `doca_task_submit()` returned `DOCA_SUCCESS`; the PE produces nothing | The PE is not being progressed. Add a `doca_pe_progress()` call in the main loop. |
| Decrypt completion reports `IO_FAILED` (or equivalent tag-verification-failure status) on real user ciphertext | Tag verification failed on data the user thought was good | This is a **security signal**, not a code bug: the ciphertext was tampered with or corrupted in transit / at rest. Do not consume the plaintext, do not retry the decrypt — treat the input as untrusted and route to the user's application-layer policy for tampered input. |
| Bulk submit returns `DOCA_ERROR_AGAIN` | First N submissions succeed, then `AGAIN` | The task queue is full. Drain completions between bursts via `doca_pe_progress()`, or raise the configured queue depth at configure time. |

Loop termination: stop iterating once two consecutive iterations of
the same kind do not change anything — that means the cause is
below DOCA AES-GCM. Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
with the captured cap-query snapshot + known-vector diff as
evidence.

## debug

Goal: when a DOCA AES-GCM call returns a `DOCA_ERROR_*` (or the
program produces no completion event, or a decrypt completion
reports a tag-verification failure), narrow the cause to a specific
layer and act on it.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending AES-GCM-specific
fixes. This skill's overlay names the AES-GCM-specific manifestation
at layers 5 (runtime) and 6 (program):

**Layer 5 (runtime) — AES-GCM overlay.**

- Walk the lifecycle: was the context started? Was the task enabled
  before start (`doca_aes_gcm_task_*_set_conf` before
  `doca_ctx_start()`)? Submitting before the task is enabled
  returns `DOCA_ERROR_BAD_STATE`, not a clear symptom.
- Confirm the PE is being progressed. *No completion events* is
  almost always a missing `doca_pe_progress()` in the user's main
  loop.
- Confirm both the source mmap and the destination mmap are still
  alive at submit time. Destroying either before `doca_ctx_destroy`
  is a use-after-free that surfaces as `DOCA_ERROR_BAD_STATE` from
  subsequent calls.

**Layer 6 (program) — AES-GCM overlay.**

- Key-type discipline: a `_set_conf` or submit call that quotes a
  key type the cap query returns false for returns
  `DOCA_ERROR_NOT_SUPPORTED`. Re-run the cap query against the
  active `doca_devinfo`; do not assume from prior installs. The key
  buffer length in bytes (16 / 32) MUST match the declared key
  type (128-bit `DOCA_AES_GCM_KEY_128` or 256-bit
  `DOCA_AES_GCM_KEY_256`; AES-192 is not in the enum).
- Buffer-sizing matrix: the most common AES-GCM program-layer bug
  is a plaintext larger than
  `doca_aes_gcm_cap_task_encrypt_get_max_buf_size(devinfo)` —
  surfaces as `DOCA_ERROR_INVALID_VALUE` at submit. The fix is to
  fragment at the application layer; this library does not
  auto-fragment.
- AAD-binding mismatch (decrypt-only symptom): if encrypt + decrypt
  round-trips fail on the user's own data but pass on a published
  vector, the AAD bytes the user passes to decrypt do not match the
  AAD bytes used at encrypt. AAD mismatches are silent at configure
  time and only surface as tag-verification failure on decrypt
  completion. Walk the user's AAD construction on both sides
  byte-for-byte.
- IV / nonce discipline: AES-GCM tag verification depends on the
  exact same IV being passed to encrypt and decrypt. An IV
  mismatch between the two sides surfaces as a tag-verification
  failure, identical to a tampered ciphertext. Walk the user's IV
  source on both sides (per-record counter, random per record, …)
  to disambiguate.
- Tag-verification failure on real data — security-critical path.
  When a decrypt completion reports the tag-verification-failure
  status on data the user expected to be good, the agent's response
  must be:
  (a) treat the plaintext output as poisoned — do not consume it;
  (b) treat the input ciphertext as untrusted — log a security
  event with the IV, AAD, and the upstream source the ciphertext
  came from (NOT the key);
  (c) do NOT silently retry — retrying the same ciphertext + key +
  IV + AAD will fail the same way and hides the security signal;
  (d) route the user to their application-layer policy for
  tampered input (rotate keys? drop the connection? quarantine the
  storage object?).
- Known-vector mismatch: if a published-vector encrypt produces a
  wrong ciphertext, the key bytes, IV bytes, or AAD bytes the
  program is passing do not match the vector. If the tag is the
  wrong length, the user has misconfigured the tag-length field.

Once the layer is identified, route to the matching debug verb on
the matching skill: install / build / link / driver to
[`doca-setup ## debug`](../../doca-setup/TASKS.md#debug); version to
[`doca-version ## debug`](../../doca-version/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug);
program-layer Core-context patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug).

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows so
the agent does not invent guidance:

- **install.** Installing DOCA, choosing packages, post-install
  verification, `pkg-config` wiring — defer to
  [`doca-setup`](../../doca-setup/SKILL.md) and to the install-tree
  layout in
  [doca-public-knowledge-map ## Layout of an installed DOCA package](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill assumes DOCA is already installed.
- **deploy.** Deploying AES-GCM-using applications at scale
  (TLS-record encryption workers across many hosts, encrypted
  storage daemons, Kubernetes operator workflows) — out of scope
  for Phase 1 and reserved for a future platform skill. For
  single-host first-run testing, the right verb in this skill is
  `## run`; do not invent a "deploy" workflow.
- **rollback.** Coordinated rollback of AES-GCM-using applications
  across many hosts — out of scope. For a single in-session
  AES-GCM configuration rollback, the right verb in this skill is
  destroying the context (`doca_ctx_stop` → `doca_ctx_destroy`)
  and re-running [`## configure`](#configure) with corrected
  parameters.
- **key management.** Key generation, rotation, escrow, KMS / HSM
  integration — outside this skill. The skill's only key-handling
  contribution is the operational discipline in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  (do not log keys, do not commit keys, zero key buffers before
  free). The actual key-management substance belongs to the user's
  KMS or HSM documentation.
- **kernel-level driver install / firmware burn.** AES-GCM depends
  on the underlying ConnectX firmware and BlueField BFB; if the
  debug ladder lands on a driver-layer issue, the fix is via
  `mlxconfig` / `mlxfwreset` / re-imaging the BFB, all of which
  belong to
  [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) layer 5.

## Command appendix

Every command below is **cross-cutting on DOCA AES-GCM** — it
answers a recurring class of question that comes up in the verbs
above. The agent should treat the *class* as load-bearing; the
worked example is a single instance. Run-as user is the
unprivileged user unless noted. Sudo is called out per row.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST (`doca-env --json`
   for version + devices + libraries + drivers + hugepages in one
   shot; `doca-capability-snapshot` for per-device capability flags;
   `version-matrix.json` for *"available since"* lookups).
2. If the probe succeeds, the structured tool's output is the
   authoritative answer and the agent SHOULD NOT also run the
   manual command in the row below. Report *"using structured
   `<tool>`"*.
3. If the probe fails, fall back to the manual command in the row.
   Report *"falling back to manual chain"*.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics (four-way match, NGC,
   headers-win) are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Command (worked example) | Owning step | Class of question it answers | What healthy output looks like |
| --- | --- | --- | --- |
| `pkg-config --modversion doca-aes-gcm` | `## configure` step 1; `## build` slot 4 | What is the build-time DOCA AES-GCM version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --cflags --libs doca-aes-gcm` | `## build` | What include + link flags does the linker need? | Trust whatever `pkg-config --cflags --libs` produces on this install. Do not hardcode either the `-I` include path or the `-l<name>` flag form — both can drift between DOCA install profiles and DOCA majors; the on-disk `.so` basenames use underscores on every release where we have ground truth, while the `.pc` package names use hyphens, and `pkg-config` is the only thing that resolves both correctly. Hand-crafted `-l` lines silently break when DOCA upgrades. |
| `doca_caps --list-devs` | `## configure` step 2 | Which devices on this host can be used as a `doca_dev` for AES-GCM? | One row per visible device with PCIe address and capability flags; the agent must still run `doca_aes_gcm_cap_*` per-device to confirm key-size support |
| `doca_caps --version` | `## configure` step 1; `## test` step 1 | What is the *runtime* DOCA version on this host? | A semver string matching `pkg-config --modversion doca-aes-gcm` |
| `ls /opt/mellanox/doca/samples/doca_aes_gcm/` | `## modify` slot 1 | Which AES-GCM samples ship in this install, and which is the closest starting point? | A list of sample directories named after the task pattern they demonstrate |
| `cat /opt/mellanox/doca/applications/VERSION` | `## configure` step 1; `## debug` layer 1 | What does the install tree itself claim its version is? | A semver string matching the other two version sources |
| `openssl enc -aes-256-gcm -K <hex_key> -iv <hex_iv> -in plaintext.bin -out ciphertext.bin` (illustrative — the real reference uses `EVP_*` programmatically) | `## test` step 5 | What is the CPU-reference ciphertext for the bulk-encrypt comparison? | Bytes matching the doca-aes-gcm output for the same key + IV + AAD + plaintext. Differences = configuration bug, not a CPU-vs-accelerator difference |
| `dmesg | tail -n 40` (sudo) | `## debug` layer 7 | What did the kernel / driver log around the last AES-GCM call? | Empty or recent benign messages. Repeated mlx5 / accelerator errors → driver-layer bug; route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 3 | What did the structured DOCA logger emit for the first failing call? | A trace-level line on every lifecycle transition and every task submission. Silence after submission = PE not progressed. **Do not log the key buffer** — only the library's own trace lines |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the AES-GCM-specific rows on top.
