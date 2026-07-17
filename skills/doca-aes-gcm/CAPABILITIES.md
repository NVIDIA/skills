# DOCA AES-GCM capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(capabilities / version / errors / observability / safety) and read
that section end-to-end. The tables in each section are the
load-bearing content; the prose around them is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern (the
verbs `configure / build / modify / run / test / debug`), jump to
[TASKS.md](TASKS.md). For the canonical DOCA version-handling rules
that this skill layers an AES-GCM overlay on top of, see
[`doca-version`](../../doca-version/SKILL.md).

## Pattern overview

Every DOCA AES-GCM question this skill teaches resolves into one of
SIX patterns. The patterns are CLASSES — they apply across every
AES-GCM release and every BlueField / ConnectX device the accelerator
runs on.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Decide whether to offload | Compare the per-call setup cost of doca-aes-gcm against a CPU AES-GCM (OpenSSL `EVP_aes_*_gcm`) on the same input; choose offload only when the input is bulk, sustained, or already pinned in `doca_mmap` memory | [`## Capabilities and modes`](#capabilities-and-modes) path-selection table + [`## Safety policy`](#safety-policy) *"when NOT to use doca-aes-gcm"* bullets |
| 2. Pick the task type | Encrypt (`doca_aes_gcm_task_encrypt`) to produce ciphertext + auth tag from plaintext + key + IV + AAD; decrypt (`doca_aes_gcm_task_decrypt`) to verify the tag and produce plaintext from ciphertext + key + IV + AAD + expected tag | [`## Capabilities and modes`](#capabilities-and-modes) encrypt-vs-decrypt table + [TASKS.md ## modify](TASKS.md#modify) |
| 3. Discover capabilities | Query `doca_aes_gcm_cap_*` for task-type presence, per-key-size support, and max plaintext size per submission, against the active `doca_devinfo` BEFORE choosing the key size or sizing buffers | [`## Capabilities and modes`](#capabilities-and-modes) capability-query rule + [TASKS.md ## configure](TASKS.md#configure) step 2 |
| 4. Honor source / destination permissions | Source mmap = `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` at minimum; destination mmap = `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE`; both must be set BEFORE the first task submission | [`## Safety policy`](#safety-policy) permission matrix + [TASKS.md ## test](TASKS.md#test) |
| 5. Treat auth-tag verification as security-critical | On decrypt, the auth tag MUST verify before the plaintext output is consumed. A failed verification is a security-relevant event — the ciphertext was tampered with or corrupted, and the plaintext output of that task is poisoned | [`## Error taxonomy`](#error-taxonomy) decrypt-failure row + [`## Safety policy`](#safety-policy) AEAD-discipline section |
| 6. Diagnose an AES-GCM error | Map symptom (`DOCA_ERROR_BAD_STATE`, `_INVALID_VALUE`, `_NOT_SUPPORTED`, `_NOT_PERMITTED`, `_AGAIN`, `_IO_FAILED`) to root cause — lifecycle, key-size, oversized input, permission, queue pressure, or tag verification failure — without leaving the AES-GCM layer prematurely | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Three cross-cutting rules that apply to *every* pattern above:

- **AES-GCM is AEAD, not raw AES.** AES-GCM provides confidentiality
  AND integrity in one operation; the auth tag on encrypt and the
  tag verification on decrypt are not optional add-ons. Treating
  AES-GCM like AES-CTR (where there is no tag) or AES-CBC (where
  authentication is a separate HMAC) is the single most common
  baseline-agent error. If the user wants a non-AEAD AES mode, the
  answer is CPU + OpenSSL — not this library.
- **Discover the version-installed surface, do not assume.** Every
  pattern above gates on `pkg-config --modversion doca-aes-gcm` and
  on the `doca_aes_gcm_cap_*` capability queries against the active
  `doca_devinfo`. Quoting a key-size, a max plaintext size, or even
  the presence of the encrypt or decrypt task without checking is
  the most common hallucination failure mode.
- **Validate against a published test vector before bulk.** The
  cheapest way to confirm the configured AES-GCM context produces a
  correct ciphertext + tag is to encrypt + decrypt a small fixed
  input (e.g. a NIST GCMVS vector or an RFC 5288 AES-GCM example)
  with a known key + IV + AAD and verify both directions match the
  published output, before submitting any user data.

## Capabilities and modes

DOCA AES-GCM is a **DOCA Core Context**. Every AES-GCM instance
follows the universal `cfg-create → cfg-set-* → init → start → use
→ stop → destroy` lifecycle (see
[`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes)).
On top of that lifecycle, DOCA AES-GCM layers its own task model and
key-size-selection surface.

**The two task types.** DOCA AES-GCM exposes two task types; each has
its own `doca_aes_gcm_task_*_set_conf` (enable the task) and its own
matching `doca_aes_gcm_cap_task_*_is_supported` (capability query)
entry point. The agent must call the capability query before assuming
the task type is available on the user's device.

| Task type | Class shape | Notes |
| --- | --- | --- |
| `doca_aes_gcm_task_encrypt` | Take a single source `doca_buf` (plaintext) + key + IV + optional AAD, produce ciphertext in a destination `doca_buf` plus an authentication tag | Asynchronous; completion arrives via `doca_pe_progress`. The output is ciphertext + a tag the receiver must check. The plaintext-per-submission ceiling is `doca_aes_gcm_cap_task_encrypt_get_max_buf_size(devinfo)`. |
| `doca_aes_gcm_task_decrypt` | Take a single source `doca_buf` (ciphertext) + key + IV + optional AAD + expected auth tag, verify the tag, produce plaintext in a destination `doca_buf` | Asynchronous; completion arrives via `doca_pe_progress`. **Tag verification is mandatory** — if the completion reports a verification failure, the plaintext output is *not* safe to consume (see [`## Safety policy`](#safety-policy)). |

**Path selection — encrypt vs decrypt.** The two task types are
independent surfaces; a single AES-GCM context can have both enabled
(typical) or just one (e.g. a producer side only encrypts; a
consumer side only decrypts). Choose at least one before
`doca_ctx_start()`.

| Path | What it is | Right shape for | Wrong shape for |
| --- | --- | --- | --- |
| Encrypt only | Configure only `doca_aes_gcm_task_encrypt`; the context produces ciphertext + tag for downstream consumers | A producer-only pipeline (log-shipper that encrypts before send, a backup writer encrypting before storing) | Any flow that also needs to consume incoming AES-GCM-protected data on the same context |
| Decrypt only | Configure only `doca_aes_gcm_task_decrypt`; the context consumes ciphertext + tag and produces verified plaintext | A consumer-only pipeline (a TLS record receiver, a backup reader decrypting before processing) | Any flow that also needs to produce outgoing AES-GCM-protected data on the same context |
| Both | Both tasks enabled on the same context | A symmetric peer that both encrypts outbound and decrypts inbound (the canonical case for an end-to-end encrypted channel) | A read-only or write-only flow — enabling the unused side adds setup overhead without benefit |

**Path-selection — when to use DOCA AES-GCM at all.** The agent's
rule:

- **Use doca-aes-gcm when** the input is bulk (rule of thumb: ≥ a
  few KiB per record and sustained), the throughput requirement is
  high enough that freeing the CPU for other work is valuable, or
  the input is already pinned in `doca_mmap` memory because another
  DOCA library produced it. TLS record encryption at line rate,
  encrypted storage (data-at-rest pipelines), and high-throughput
  backup encryption are the canonical fits.
- **Do NOT use doca-aes-gcm when** the input is a single tiny
  message (the per-call DMA-to-accelerator setup cost dominates;
  CPU + AES-NI is faster); when the user needs a non-GCM AES mode
  (CBC, CTR, XTS — use CPU + OpenSSL, not this library); or when
  the user needs *only* a keyed hash for authentication without
  encryption (use [`doca-sha`](../doca-sha/SKILL.md) for hashing,
  or a CPU HMAC).

**AES-GCM key sizes — only 128 and 256.** AES-GCM is defined in
the spec for 128 / 192 / 256-bit keys, but the DOCA AES-GCM
library's `enum doca_aes_gcm_key_type` exposes only
`DOCA_AES_GCM_KEY_128` and `DOCA_AES_GCM_KEY_256` — **AES-192-GCM
is NOT expressible through this library at all**. The agent must
not invent an AES-192 path. Per-device support for each of the
two real key types is still cap-gated: call
`doca_aes_gcm_cap_task_encrypt_is_key_type_supported(devinfo, key_type)`
and the matching `_decrypt_is_key_type_supported` query before
sizing key material.

| Key size (bits) | `doca_aes_gcm_key_type` enum value | Key length (bytes) | Notes |
| --- | --- | --- | --- |
| 128 | `DOCA_AES_GCM_KEY_128 = 1` | 16 | Often supported; the smallest AES-GCM key the library accepts |
| 256 | `DOCA_AES_GCM_KEY_256 = 2` | 32 | Often supported; the most common choice for new deployments |
| 192 | *(not in the enum)* | *(n/a)* | NOT supported by the DOCA AES-GCM library — route to a CPU library if AES-192 is a hard requirement |

The agent must NOT quote either of the two real key types as
universally available — the cap query against the active
`doca_devinfo` is the runtime authority. A response that
recommends a key size without first naming the cap query — or
that quotes AES-192 as a per-device-cap-queried option — is the
baseline-agent failure mode this skill exists to prevent.

**Capability discovery — the only rule.** Before choosing a key size
or sizing any plaintext buffer, call the matching
`doca_aes_gcm_cap_*` query against the active `doca_devinfo`:

| Capability | Query | Why the agent must ask |
| --- | --- | --- |
| Encrypt task supported | `doca_aes_gcm_cap_task_encrypt_is_supported(devinfo)` | If false, the device has no AES-GCM encrypt accelerator and the user must fall back to CPU |
| Decrypt task supported | `doca_aes_gcm_cap_task_decrypt_is_supported(devinfo)` | If false, the device has no AES-GCM decrypt accelerator; the user must fall back to CPU for inbound |
| Key type supported for encrypt | `doca_aes_gcm_cap_task_encrypt_is_key_type_supported(devinfo, key_type)` | Per-key-type boolean against the real `enum doca_aes_gcm_key_type` (`DOCA_AES_GCM_KEY_128` or `DOCA_AES_GCM_KEY_256`); AES-192-GCM is NOT in the enum and cannot be queried |
| Key type supported for decrypt | `doca_aes_gcm_cap_task_decrypt_is_key_type_supported(devinfo, key_type)` | Same surface for the decrypt task |
| Maximum encrypt plaintext size | `doca_aes_gcm_cap_task_encrypt_get_max_buf_size(devinfo)` | Hardware-bound ceiling on plaintext size per submission; inputs larger than this require the user to fragment at the application layer |

**Configuration shape.** *Mandatory* configurations before
`doca_ctx_start()`: at least one task type enabled via
`doca_aes_gcm_task_encrypt_set_conf` or
`doca_aes_gcm_task_decrypt_set_conf`, and the matching mmap
permissions set on both source and destination buffers per the
matrix in [`## Safety policy`](#safety-policy). Key material is
typically passed as part of the per-task setup; key bytes never live
in the skill or in the configured context for longer than the user's
own program needs them.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match
rule, NGC container semantics, and the headers-win-over-docs rule,
see [`doca-version`](../../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The AES-GCM-specific overlay** is:

- **Per-key-type membership is device-bound, not version-bound.**
  The agent must not infer *"DOCA version X includes a third
  AES-GCM key size"* from release notes alone — the public
  `enum doca_aes_gcm_key_type` has two values
  (`DOCA_AES_GCM_KEY_128`, `DOCA_AES_GCM_KEY_256`); AES-192-GCM
  is NOT in the enum and is not expressible through this library.
  For the two real key types, the runtime authority is
  `doca_aes_gcm_cap_task_encrypt_is_key_type_supported(devinfo, key_type)`
  / `_decrypt_is_key_type_supported(devinfo, key_type)` against
  the active device. Per the cross-cutting cap-query rule in
  [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability),
  the cap query is the runtime authority — never quote a key
  type as available from agent memory.
- **Per-device task presence is independent of the library install.**
  An install that ships the `doca-aes-gcm` shared object (verify with `ldconfig -p | grep -i doca_aes_gcm`; the on-disk basename uses underscores on every DOCA release where we have ground truth) does not guarantee a
  particular device exposes both encrypt and decrypt; some device
  generations advertise only one. Confirm via
  `doca_aes_gcm_cap_task_encrypt_is_supported(devinfo)` and
  `_task_decrypt_is_supported(devinfo)` per the cross-cutting
  cap-query rule above; if one returns false on a device the user
  expected to support it, route to
  [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
  before invoking the missing codepath.
- **`doca-aes-gcm.pc` plus `doca-common.pc` must both match
  `doca_caps --version`** at the four-way-match check (per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)).
  A common partial-install pattern on hosts where the AES-GCM
  package was installed separately from the rest of DOCA is that
  `doca-aes-gcm.pc` reports release *X* while `doca-common.pc`
  reports release *Y*; route to
  [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
  layer 2 before any AES-GCM-layer diagnosis.

## Error taxonomy

AES-GCM-specific overlays on the cross-library `DOCA_ERROR_*`
taxonomy. The cross-library taxonomy itself lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *AES-GCM surface* meaning that the agent must
disambiguate before falling back to the cross-library response.

| Error | AES-GCM context where it shows up | AES-GCM-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | Any call after `doca_ctx_stop()` or before `doca_ctx_start()`; task submit before the matching `_set_conf` ran | Lifecycle violation. Walk the call sequence against the lifecycle in [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes); confirm the task type the user is submitting was enabled via `doca_aes_gcm_task_*_set_conf` before `doca_ctx_start()`. |
| `DOCA_ERROR_INVALID_VALUE` | `doca_aes_gcm_task_*_alloc_init`; submit time | The configured key size is not in the supported set, OR the plaintext exceeds `doca_aes_gcm_cap_task_encrypt_get_max_buf_size(devinfo)`, OR the auth tag length the user passed does not match the AES-GCM spec. Re-run the matching cap query, then resize or correct the tag length. |
| `DOCA_ERROR_NOT_SUPPORTED` | `_task_*_set_conf`; submit with an unsupported key type | The task type is not in this device's accelerator, OR the configured key type is not in this device's supported set. Re-run `doca_aes_gcm_cap_task_*_is_supported` / `_is_key_type_supported` against the active `doca_devinfo`; if false, that is the answer — fall back to CPU or change the key type. |
| `DOCA_ERROR_NOT_PERMITTED` | First task submission | The source mmap does not have `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` (or a stronger flag that supersedes it), OR the destination mmap is missing `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE`, OR the program lacks the access the user's key-handling layer requires. Re-check the matrix in [`## Safety policy`](#safety-policy). |
| `DOCA_ERROR_AGAIN` | `doca_task_submit` on either AES-GCM task type | The task queue is full. This is *not* a hardware error; the program must drain completions via `doca_pe_progress()` before re-submitting. Same as the cross-library *"would-block, retry after progress"* pattern. |
| `DOCA_ERROR_IO_FAILED` (or equivalent device-reported error on the decrypt completion) | `doca_aes_gcm_task_decrypt` completion | **Auth tag verification FAILED on decrypt.** This is the security-relevant outcome: the ciphertext was tampered with or corrupted in transit / at rest. The plaintext output of this task is *poisoned* — do not consume it. Treat the input ciphertext as untrusted, log a security event, and route to the user's application-layer policy for tampered input. Do NOT silently retry. |
| `DOCA_ERROR_DRIVER` | Any submit / completion call | The layer below DOCA reported failure. Capture state and route to env-class debug ([`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)) — the layer below DOCA is the suspect, not the AES-GCM program. |

The agent's rule: **never recommend a retry loop on `DOCA_ERROR_*`
without first identifying which of the rows above is the cause**.
`_AGAIN` is the only one that wants a retry (after
`doca_pe_progress()`); the others want investigation, not retry.
**Tag-verification failure (the `IO_FAILED`-class row above) MUST NOT
be retried** — a retry of the same ciphertext + key + IV + AAD will
fail the same way, and silently re-submitting hides the security
signal.

## Observability

DOCA AES-GCM observability is **event-driven, not poll-driven**.
Every submitted task produces a completion event on the DOCA Core
progress engine; there are no AES-GCM-specific counters the way Flow
has per-pipe counters.

Four primary signals the agent should reach for:

1. **Task completion events on the PE.** Every submitted AES-GCM
   task (encrypt or decrypt) produces a completion event when it
   finishes (or errors). The completion carries the `doca_error_t`
   if it failed; the agent must inspect the per-task completion,
   not the `doca_task_submit()` return value alone. *Submitted but
   no completion* is almost always a missed `doca_pe_progress()`
   call in the user's main loop.
2. **Decrypt completion status — the security-critical signal.** On
   `doca_aes_gcm_task_decrypt` completions specifically, the
   per-task completion result distinguishes "tag verified, plaintext
   is good" from "tag did NOT verify, plaintext is poisoned". The
   agent's observability rule: every decrypt completion must be
   inspected for this status before the plaintext is read; treating
   the absence of a system-level error as "success" misses the
   tampered-ciphertext case.
3. **Capability snapshot at configure time.** The output of every
   `doca_aes_gcm_cap_*` query is a snapshot of *what the device's
   accelerator said was possible* before any task was submitted.
   Save it as the baseline; if a task later returns
   `DOCA_ERROR_NOT_SUPPORTED` or `DOCA_ERROR_INVALID_VALUE` the diff
   against this snapshot is the bug.
4. **Known-vector round-trip.** The cheapest correctness signal is
   to encrypt + decrypt a small fixed input (a published AES-GCM
   test vector — NIST GCMVS or RFC 5288 example) with a known key +
   IV + AAD and compare against the published ciphertext + tag (on
   encrypt) and recovered plaintext (on decrypt). If the
   accelerator produces a different ciphertext or fails to recover
   the plaintext, the configuration is wrong; do not move on to
   bulk input.

For cross-cutting observability primitives (`--sdk-log-level`, the
`doca-<lib>-trace` build flavor, the `DOCA_LOG_LEVEL` env var) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the install-tree observability (logger names, package layout)
defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

DOCA AES-GCM's safety surface is **buffer-permission discipline plus
AEAD discipline plus key-handling discipline plus path-selection
discipline**. AES-GCM is authenticated encryption; an incorrect
permission flag, a skipped tag check on decrypt, a leaked key, or
the wrong choice of accelerator-vs-CPU each produces a distinct
class of failure, and the agent's job is to verify all four before
any submission.

The **permission matrix** the agent must walk for any new AES-GCM
setup:

| Buffer | Minimum mmap permission | Class shape | Common over-broad mistake |
| --- | --- | --- | --- |
| Source (plaintext on encrypt / ciphertext on decrypt) | `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` | The accelerator only needs to read the input; granting write is unnecessary and a small attack-surface widening | Granting `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE` "to keep things simple" — works, but the read-only flag is the safe minimum |
| Destination (ciphertext on encrypt / plaintext on decrypt) | `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE` | The accelerator writes the output; the application reads it back | Reusing the source `doca_mmap` as the destination — fails at submit if the mmap is read-only; if both buffers share an mmap, the mmap must be read-write |

**AEAD discipline — the rules the agent must surface every time.**

- **AES-GCM is authenticated encryption.** The output of encrypt is
  ciphertext PLUS an auth tag; the input to decrypt is ciphertext
  PLUS an expected auth tag. The agent must never describe AES-GCM
  as if it were AES-CTR (where there is no tag) or AES-CBC (where
  authentication is a separate HMAC). A response that surfaces only
  the ciphertext path and ignores the tag is wrong by construction.
- **AAD (additional authenticated data) is authenticated but not
  encrypted.** When the user passes AAD on encrypt, the tag covers
  both the ciphertext and the AAD; on decrypt, the same AAD must be
  passed alongside the ciphertext and tag, or verification will
  fail. AAD is the slot for protocol headers, sequence numbers, or
  any context the receiver must be able to bind to the ciphertext.
- **Tag verification failure on decrypt is poisoned output.** If
  `doca_aes_gcm_task_decrypt` reports a tag-verification failure on
  its completion (the `IO_FAILED`-class row in
  [`## Error taxonomy`](#error-taxonomy) above), the plaintext
  output of that task is *not* safe to consume. The ciphertext was
  tampered with or corrupted in transit / at rest; the right
  response is to discard the plaintext, treat the input as
  untrusted, log a security event, and route to the user's
  application-layer policy. A silent retry on the same ciphertext
  is wrong by construction — the failure is deterministic.

**Key-handling discipline — operational rules, not key-management
substance.** Key management itself (generation, storage, rotation,
distribution) is out of scope for this skill; it belongs to the
user's KMS / HSM / sealed-storage system. The skill's only
operational rules:

- **Do not log key bytes.** Trace-level DOCA logging (`DOCA_LOG_LEVEL=trace`)
  can be verbose; the agent must not recommend dumping the key
  buffer in user-side prints. If the agent is helping the user
  instrument their program, the *length* of the key is observable;
  the *contents* are not.
- **Do not commit key bytes.** Hard-coded keys in a sample diff are
  a leak waiting to happen; the agent's `## modify` workflow MUST
  surface this as a hazard if the user pastes a key into a code
  edit.
- **Treat the key buffer as sensitive memory.** When the program is
  done with the key, zero it before free. The library does not own
  the key bytes; the user does.

**Path-selection — when NOT to use doca-aes-gcm.** Equally important:

- **Tiny one-shot encryption.** A single 64-byte AES-GCM encrypt on
  the CPU (AES-NI) is faster than the DMA-to-accelerator setup. The
  agent should recommend doca-aes-gcm only when the input is ≥ a
  few KiB *and* sustained, *or* the input is already pinned in
  `doca_mmap` memory because another DOCA library produced it.
- **Non-GCM AES modes.** If the user wants AES-CBC, AES-CTR, or
  AES-XTS, the answer is CPU + OpenSSL (or another CPU library) —
  not this library. The agent must not invent
  `doca_aes_*_cbc` / `_ctr` / `_xts` symbols; this library covers
  GCM specifically.
- **Authentication without encryption.** If the user needs only a
  keyed hash for integrity (e.g. an HMAC over a plaintext payload),
  AES-GCM is the wrong tool. Route to
  [`doca-sha`](../doca-sha/SKILL.md) for hardware-accelerated SHA
  hashing, or to CPU HMAC.

**The mmap must stay valid until the AES-GCM context is destroyed.**
Destroying either the source or destination mmap before
`doca_ctx_destroy()` is a use-after-free on the library's
bookkeeping; symptoms include `DOCA_ERROR_BAD_STATE` from subsequent
calls and undefined behavior on outstanding tasks.

**Validate against a known vector BEFORE bulk submission.** A single
known-vector smoke (one short input with a published key + IV + AAD
+ ciphertext + tag) catches key-size mis-selection, IV-length
assumptions, AAD-binding bugs, and tag-length errors before they
silently corrupt or poison a multi-GiB result.

## Deferred topic boundaries

This skill scopes itself to the DOCA AES-GCM library. Adjacent
topics the agent will get asked but should route elsewhere:

- **General AEAD / cryptography background** (why GCM is preferred
  over CBC-then-HMAC, GCM nonce-reuse pitfalls beyond the API
  surface, the underlying GHASH construction, when to choose AES-GCM
  vs ChaCha20-Poly1305) — outside this skill. Route to upstream NIST
  / IETF cryptography documentation; this skill assumes the user
  already knows AES-GCM is the right primitive and is asking *how to
  express it through the DOCA AES-GCM API*.
- **Key management** (key derivation functions, key rotation
  schedules, HSM integration, key escrow) — owned by the user's
  KMS / HSM stack, not this library. The skill's only contribution
  is the operational key-handling discipline in
  [`## Safety policy`](#safety-policy).
- **DOCA Core context and progress engine internals** — owned by
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  This skill *uses* the Core context lifecycle; it does not
  redefine it.
- **Cross-cutting `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the AES-GCM overlay, not the taxonomy itself.
- **Other DOCA crypto-acceleration libraries (DOCA SHA, DOCA
  Compress, DOCA DMA)** — separate libraries with their own skills
  (when they ship). Path-selection guidance in
  [`## Capabilities and modes`](#capabilities-and-modes) names them;
  the deep per-library substance lives in the matching skill.
- **Cross-library `doca_caps` invocation patterns** — owned by the
  cross-library `doca-caps` tool skill (when it ships). This skill
  references the *AES-GCM capability query family*
  (`doca_aes_gcm_cap_*`), which is per-library; the *cross-library
  capability snapshot tool* (`doca_caps --list-devs`) is a separate
  surface.
