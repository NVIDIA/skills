---
license: Apache-2.0
name: doca-aes-gcm
description: >
  Use this skill when the user is doing hands-on DOCA AES-GCM
  work on a BlueField DPU or ConnectX NIC — configuring
  `doca_aes_gcm_task_encrypt` / `_task_decrypt`, querying
  `doca_aes_gcm_cap_*` for per-key-type (only
  `DOCA_AES_GCM_KEY_128` / `_256` — AES-192 not supported) and
  per-task support, sizing plaintext against the max-buf cap,
  setting source / destination mmap permissions, validating
  with a NIST GCMVS or RFC 5288 vector, or debugging
  DOCA_ERROR_* including the security-critical
  tag-verification-failed outcome on decrypt. Trigger even
  when the user does not explicitly mention "DOCA AES-GCM" or
  "AEAD" — typical implicit phrasings: "decrypt completion
  IO_FAILED", "auth tag isn't verifying",
  "NOT_PERMITTED on my encrypt buffer", "is AES-192-GCM on
  this BlueField" (no), or "encrypted record came back
  tampered". Refuse and route elsewhere for non-GCM AES modes
  (CBC / CTR / XTS — CPU OpenSSL), key management
  (KMS / HSM / rotation), SHA (doca-sha), or general AEAD
  background.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux
  (Ubuntu 22.04/24.04 or RHEL/SLES) with a BlueField DPU or
  ConnectX NIC attached. Reads the local install via
  `pkg-config doca-aes-gcm` and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}; the
  accelerator must advertise the desired key type at runtime
  via `doca_aes_gcm_cap_task_{encrypt,decrypt}_is_key_type_supported`
  (only `DOCA_AES_GCM_KEY_128` / `_256`; AES-192 unsupported).
---

# DOCA AES-GCM

**Where to start:** This skill assumes DOCA is already installed and
the user is doing **hands-on AES-GCM-acceleration work** on a
BlueField / ConnectX / host with DOCA. Open [`TASKS.md`](TASKS.md) if
the user wants to *do* something (configure / build / modify / run /
test / debug); open [`CAPABILITIES.md`](CAPABILITIES.md) when the
question is *what can DOCA AES-GCM express* on this version. If the
user has not installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first. If the user is
asking *"should I even use the accelerator for this encryption?"*,
the path-selection rule in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
is the first stop. If the user is treating AES-GCM as a confidentiality-only
primitive (raw AES-CTR / AES-CBC style), stop and read the AEAD note
in [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
first — AES-GCM is authenticated encryption, and confusing the two is
the most expensive failure mode this skill exists to prevent.

## Example questions this skill answers well

The CLASSES of DOCA AES-GCM questions this skill is built to answer,
each with one worked example. The agent should treat the *class* as
the load-bearing piece — the worked example is a single instance.

- **"Should I offload this AES-GCM encryption to DOCA AES-GCM, or
  just do it on the CPU with OpenSSL?"** — worked example: *"I am
  encrypting 4 KiB TLS records at line rate; is doca-aes-gcm worth
  the setup vs OpenSSL `EVP_aes_256_gcm` on the CPU?"*. Answered by
  the path-selection table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the *"when NOT to use doca-aes-gcm"* bullets in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- **"Does my device support the AES-GCM key size I want?"** — worked
  example: *"is AES-256-GCM in the accelerator on this BlueField?
  And while we're here, is AES-192-GCM available?"* (Answer: the
  library exposes only `DOCA_AES_GCM_KEY_128` / `DOCA_AES_GCM_KEY_256`;
  AES-192 is not in the enum and is not supported. For the two
  real key types, gate on
  `doca_aes_gcm_cap_task_encrypt_is_key_type_supported(devinfo, key_type)`
  and the matching `_decrypt_is_key_type_supported`. AES-192 is
  not available — route to a CPU library.) Answered by the
  per-key-type capability queries and the per-task
  `doca_aes_gcm_cap_task_*_is_supported` queries in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the discovery step in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"How do I correctly decrypt an AES-GCM message and verify the
  auth tag?"** — worked example: *"my `doca_aes_gcm_task_decrypt`
  completion reports an error — is the plaintext output safe to
  use?"*. Answered by the auth-tag verification rule in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  (*do not use the plaintext if the auth tag did not verify*) +
  the decrypt completion-handling workflow in
  [`TASKS.md ## test`](TASKS.md#test) and
  [`TASKS.md ## debug`](TASKS.md#debug).
- **"What permissions does the source / destination mmap need?"** —
  worked example: *"my `doca_aes_gcm_task_encrypt` returns
  `DOCA_ERROR_NOT_PERMITTED`"*. Answered by the permission matrix
  in [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the mmap-set-permissions checklist in
  [`TASKS.md ## test`](TASKS.md#test).
- **"Is this DOCA AES-GCM API available on my installed DOCA
  version?"** — worked example: *"is AES-192-GCM in the DOCA I have
  installed, on this device?"*. Answered by the version-compatibility
  overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
  which cross-links the canonical detection chain in
  [`doca-version`](../../doca-version/SKILL.md) and adds the
  AES-GCM-specific *"discover key sizes via cap query"* bullets.
- **"What does this `DOCA_ERROR_*` from an AES-GCM call mean and
  which layer caused it?"** — worked example: *"`DOCA_ERROR_IO_FAILED`
  on the decrypt completion — is this a hardware bug or a tag
  mismatch?"*. Answered by the AES-GCM overlay on the cross-library
  taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md).

## Audience

This skill serves **external developers building applications that
consume the DOCA AES-GCM library** — i.e., users whose code calls
`doca_aes_gcm_*` (directly in C/C++, or through FFI/bindings from
another language) to offload AES-GCM authenticated encryption /
decryption onto a BlueField DPU or ConnectX accelerator. It is *not*
for NVIDIA developers contributing to DOCA AES-GCM itself.

**Language scope.** DOCA AES-GCM ships as a C library with
`pkg-config` module name `doca-aes-gcm`. The shipped samples are
written in C. C and C++ consumers are the canonical case and the
worked examples in `TASKS.md` assume that path. Other-language
consumers (Rust, Go, Python, …) consume the same `*.so` through FFI
or language-specific bindings; the skill's contribution in that case
is to keep the lifecycle, capability-discovery, permission,
error-taxonomy, AEAD-semantics, and encrypt-vs-decrypt guidance
language-neutral, and to route the agent to the public C ABI as the
authoritative surface that any wrapper will eventually call.

**Key handling is out of scope.** This skill teaches the agent how to
*use* the DOCA AES-GCM library; it does not teach the user how to
generate, store, rotate, or distribute AES-GCM keys. Key-management
is the user's responsibility (a KMS, an HSM, a sealed file, an env
var the user trusts). The skill's only key-handling rule is the
operational one in
[`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy):
do not log keys, do not commit them to source, and treat any key
buffer the program holds as sensitive memory.

## When to load this skill

Load this skill when the user is doing hands-on DOCA AES-GCM work,
in any language. Concretely:

- Initializing a `doca_aes_gcm` context on a `doca_dev` and
  configuring at least one task type (`doca_aes_gcm_task_encrypt`
  and/or `doca_aes_gcm_task_decrypt`) before `doca_ctx_start()`.
- Choosing between **encrypt** (`doca_aes_gcm_task_encrypt` —
  takes key + IV + AAD + plaintext, produces ciphertext + auth
  tag) and **decrypt** (`doca_aes_gcm_task_decrypt` — takes key +
  IV + AAD + ciphertext + expected auth tag, produces plaintext
  *and verifies the tag*) for the user's data shape.
- Setting permissions on `doca_mmap` correctly for the source buffer
  (`DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` at minimum — the plaintext on
  encrypt or the ciphertext on decrypt) and the destination buffer
  (`DOCA_ACCESS_FLAG_LOCAL_READ_WRITE`).
- Checking which AES-GCM key types (`DOCA_AES_GCM_KEY_128` /
  `DOCA_AES_GCM_KEY_256` — AES-192 is not in the library) the
  active device's accelerator advertises via
  `doca_aes_gcm_cap_task_encrypt_is_key_type_supported` /
  `doca_aes_gcm_cap_task_decrypt_is_key_type_supported`, and which task types via
  `doca_aes_gcm_cap_task_encrypt_is_supported` /
  `_task_decrypt_is_supported`.
- Sizing the per-submission plaintext against
  `doca_aes_gcm_cap_task_encrypt_get_max_buf_size(devinfo)`.
- Validating an encrypt + decrypt round-trip against a published
  AES-GCM test vector (NIST GCMVS, or RFC 5288 examples) before
  pushing any user data through the accelerator.
- Handling the auth-tag verification result on decrypt completions
  *as a security-critical signal* — a tag-mismatch completion means
  the ciphertext was tampered with or corrupted, and the plaintext
  output of that task is poisoned and must not be consumed.
- Debugging a `DOCA_ERROR_*` returned from an AES-GCM call
  (lifecycle vs. unsupported key size vs. permission vs. tag
  verification failure on decrypt) and the task-completion event on
  the progress engine.
- Designing or extending non-C bindings (Rust, Go, Python, …) that
  wrap the AES-GCM C ABI — for the lifecycle, permission,
  capability, AEAD-semantics, and encrypt-vs-decrypt rules the
  wrapper must honor.

Do **not** load this skill for general DOCA orientation, install of
DOCA itself, AES modes that are not GCM (CBC / CTR / XTS — those are
not in this library and CPU + OpenSSL is the right answer), SHA
hashing on the same accelerator family (use
[`doca-sha`](../doca-sha/SKILL.md)), or other DOCA libraries. For
those, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive AES-GCM-specific
material lives in two companion files:

- `CAPABILITIES.md` — what DOCA AES-GCM can express on this version:
  the two task types (encrypt and decrypt), the AEAD output shape
  (ciphertext + auth tag on encrypt; verified plaintext on decrypt),
  the AES-GCM key-type surface (only 128-bit and 256-bit — AES-192
  is not in the enum, both cap-queried), the capability-query
  surface (`doca_aes_gcm_cap_*` for task presence, key-type
  support, and buffer sizing), the
  AES-GCM error taxonomy (mapped onto the cross-library
  `DOCA_ERROR_*` set, with explicit treatment of the
  tag-verification-failure outcome as security-critical), the
  observability surface (per-task completion events on the progress
  engine), the safety policy that gates source / destination mmap
  permission decisions and key-handling cautions, and the
  path-selection rule (when to use doca-aes-gcm versus CPU OpenSSL
  or a different DOCA crypto library).
- `TASKS.md` — step-by-step workflows for the six in-scope AES-GCM
  verbs: `configure`, `build`, `modify`, `run`, `test`, `debug`.
  Plus a `Deferred task verbs` block that points out-of-scope
  questions at the right next skill.

The skill assumes a host or BlueField where DOCA is already
installed at the standard location and the user has the privileges
their public install profile expects. It does not cover installing
DOCA — that path goes through
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA AES-GCM application source code, in any
  language.** The verified AES-GCM source code is the shipped C
  samples at `/opt/mellanox/doca/samples/doca_aes_gcm/`. The agent's
  job is to route the user to those files and prescribe a
  minimum-diff modification on them via the universal
  modify-a-sample workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the AES-GCM-specific overrides in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **Pre-computed AES-GCM test vectors.** The skill tells the agent
  to use a *published* test vector (e.g. NIST GCMVS, RFC 5288
  AES-GCM examples) as the known-vector smoke; it does not ship a
  vector bank of its own. The agent must cite the vector source so
  the user can audit it.
- **Pre-baked AES-GCM keys, IVs, or AAD strings.** A key in this
  repo is a key in every customer's repo — by construction, it must
  not be there. The skill teaches the *shape* of the inputs; the
  user supplies the actual bytes from their own key-management
  system.
- **Standalone build manifests** (`meson.build`, `CMakeLists.txt`,
  `Cargo.toml`, …) parked inside the skill. The agent constructs
  the build manifest *in the user's project directory* against the
  user's installed DOCA, where `pkg-config --modversion
  doca-aes-gcm` is the source of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even
  one labeled "reference", is misleading: users will read it as
  buildable.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is in
   scope.
2. **For the AES-GCM capability matrix, task types, key-size
   surface, capability-query rules, permission matrix, AEAD
   semantics, error taxonomy, observability, and safety /
   path-selection policy, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify, run,
   test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
version-handling rules, and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public docs or the
installed package layout" rather than "AES-GCM-specific guidance".

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation source and
  the on-disk layout of an installed DOCA package. The DOCA AES-GCM
  page lives at `docs.nvidia.com/doca/sdk/DOCA-AES-GCM/`; it is a
  member of the DOCA Crypto Acceleration family alongside
  [`doca-sha`](../doca-sha/SKILL.md).
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, and the *I have no install yet* path with
  the public NGC DOCA container. This skill assumes its
  preconditions are satisfied.
- [`doca-version`](../../doca-version/SKILL.md) — canonical DOCA
  version-handling rules. This skill's
  [`## Version compatibility`](CAPABILITIES.md#version-compatibility)
  cross-links the four-way match rule and adds only the
  AES-GCM-specific *"discover key sizes + task presence via cap
  query"* overlay.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect / prefer /
  fall back / report). The Command appendix in
  [TASKS.md](TASKS.md) honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library: the
  canonical `pkg-config` + meson build pattern, the universal
  modify-a-shipped-sample first-app workflow, the universal
  lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, and the
  program-side debug order. This skill layers AES-GCM specifics on
  top.
- [`doca-sha`](../doca-sha/SKILL.md) — the sibling library in the
  DOCA Crypto Acceleration family for hardware-accelerated SHA
  hashing. Load alongside this skill when the user's flow is
  authenticated-encryption *with a separate keyed hash* (rare —
  AES-GCM already provides authentication via its tag) or when the
  user is comparing offload paths between the two.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). AES-GCM-specific debug (key-size-not-supported,
  oversized input, tag-verification failure on decrypt) overlays on
  top of that ladder.
