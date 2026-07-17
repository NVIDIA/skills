# DOCA Flow gRPC Server — Capabilities

**Where to start:** `doca_flow_grpc` is a remote-control
gRPC surface in front of `doca-flow`. The pattern overview below
names the recurring server-side questions. Pick the pattern
first, then drill into the H2 that owns the substance. For the
*how* of executing each pattern, jump to [TASKS.md](TASKS.md).
For the underlying `doca-flow` API the server's RPCs program, see
[`doca-flow CAPABILITIES.md`](../../libs/doca-flow/CAPABILITIES.md).

This file is loaded by [`SKILL.md`](SKILL.md). It documents *what
state the server exposes*, *how the gRPC contract is defined and
where the authoritative `.proto` files live*, *what auth /
transport posture decisions the operator owns*, *which versions
it ships in*, *the layered error and observability surfaces*,
and *the safety policy that treats the endpoint as an admin
attack surface*.

## Pattern overview

Every `doca_flow_grpc` question this skill teaches
resolves into one of FIVE patterns. The patterns are CLASSES —
they apply across every doca-flow control-plane deployment, not
one specific application.

| `doca_flow_grpc` pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Decide remote-vs-direct | Is a remote gRPC control plane the right answer, or should the client just link `libdoca_flow.so` directly? The agent must surface the trade-off (network boundary, language barrier, deployment topology) instead of defaulting to gRPC because it sounds modern. | [`## Capabilities and modes`](#capabilities-and-modes) remote-vs-direct decision + [TASKS.md ## configure](TASKS.md#configure) |
| 2. Locate the `.proto` contract | The `.proto` files are the AUTHORITATIVE gRPC contract. **Monorepo layout:** the Flow gRPC `.proto` files live under `doca/libs/doca_flow/grpc/` (`common.proto`, `doca_flow.proto`, and `packet_buffering/packet_buffering.proto`) — NOT under `doca/tools/flow_grpc_server/`. **Binary install layout:** they are shipped via the doca-flow include / share path on the installed tree (`pkg-config doca-flow --variable=prefix` for the prefix; agent should confirm via `find <prefix> -name '*.proto'` on the user's install rather than assume a hard-coded path). Inventing RPC names or message field shapes from generic gRPC intuition is the canonical hallucination failure. | [`## Capabilities and modes`](#capabilities-and-modes) `.proto`-as-contract bullet + [TASKS.md ## configure](TASKS.md#configure) |
| 3. Pick auth / TLS / network segment | The transport posture is the operator's call: mTLS / TLS / plaintext-on-trusted-segment, token-based auth or none, the network segment the endpoint binds on. Per the [`## Safety policy`](#safety-policy), this is an admin attack surface. | [`## Capabilities and modes`](#capabilities-and-modes) auth-transport bullet + [`## Safety policy`](#safety-policy) |
| 4. Smoke-before-bulk | Start → bind → confirm one client (in the client language the operator actually plans to use) can dial the endpoint, complete the TLS / auth handshake, issue one read-only RPC, and confirm the underlying Flow application is the one being programmed. THEN, and only then, expose the endpoint to additional clients or mutating RPCs. | [TASKS.md ## test](TASKS.md#test) + [`## Safety policy`](#safety-policy) smoke-before-bulk rule |
| 5. Diagnose connect / TLS / RPC failures | Walk the layered error taxonomy in [`## Error taxonomy`](#error-taxonomy) — server-not-started / server-binding-failed / TLS-or-auth-rejected / RPC-call-error / Flow-precondition-failed / version / cross-cutting — instead of guessing at causes from a generic gRPC status code. | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **The `.proto` file is the contract.** Every concrete RPC
  name, message field name, and field type the agent quotes
  must come from the shipped `.proto` files on the user's
  installed version (or, equivalently, from the public DOCA
  Flow gRPC Server page on `docs.nvidia.com` for the same
  release). Inventing RPC names from generic gRPC patterns
  (*"`ListPipes`"*, *"`AddRule`"*) is the canonical
  hallucination failure for this skill.
- **State-changing RPCs are dataplane-affecting.** An RPC that
  creates / mutates / deletes pipes or entries does to Flow
  state exactly what a direct `libdoca_flow.so` call would
  do. The agent must label every RPC as read-only or
  state-changing and gate every state-changing RPC on a
  clean smoke per [`## Safety policy`](#safety-policy).

## Capabilities and modes

`doca_flow_grpc` is a **single CLI binary built from the DOCA
source tree** (`executable('doca_flow_grpc', ..., install: false)`,
gated by `flag_enable_grpc_support`), plus the `.proto` files
under `libs/doca_flow/grpc/` that define its
gRPC contract and (per the shipped `packet_buffering/` and
`dpa_device/` source subtrees) optional companion components
for configurations that need packet buffering or DPA-side
helpers. The interaction model is *operator starts the server
with appropriate device + Flow + transport configuration;
clients dial the gRPC endpoint and call the documented RPCs*.

### Remote-vs-direct decision

The first question the agent must surface before recommending
this tool at all.

| Surface | When to reach for it |
| --- | --- |
| Direct link to `libdoca_flow.so` in the controlling process (see [`doca-flow`](../../libs/doca-flow/SKILL.md)) | The controlling process is C / C++ and runs in the same address space (or the same host with shared libraries available); a network boundary is not required. |
| **`doca_flow_grpc`** (this skill) | The controlling process is in a different language than C / C++ (and language bindings are not available), OR runs on a different host / network segment from the BlueField / DPU, OR the deployment topology requires a centralized control plane addressing multiple BlueFields. |

The downstream rule: do not default to gRPC because it sounds
modern. A direct library link in the same process is simpler,
faster, and has a smaller attack surface; gRPC is the right
answer only when the deployment topology genuinely requires a
network boundary.

### The `.proto` file is the authoritative gRPC contract

The shipped DOCA install contains the `.proto` files that
define every RPC method, every request / response message, and
every field on the gRPC surface. Those files are the source of
truth on the user's installed version; nothing else (not a
public docs page snapshot, not agent memory, not a generic gRPC
pattern) is.

The agent's rule:

- Locate the `.proto` files on the user's install (under the
  tool's source tree shipped with DOCA; the exact path is
  install-specific and the agent should not invent it).
- Generate client stubs with the standard gRPC tooling for
  the client language — `protoc` plus the language-specific
  gRPC plugin per the [gRPC quickstart](https://grpc.io/docs/)
  index on `grpc.io`.
- Cite the `.proto` file when naming an RPC method or a
  message field; never quote a name from prose or memory.

### Auth / TLS / network-segment decision axes

> **CRITICAL (Run-12 + R13).** The shipped `doca_flow_grpc`
> binary hard-codes `grpc::InsecureServerCredentials()` (gRPC
> C++ **server-side** API); the C++ `doca_flow_grpc_client`
> binary hard-codes `grpc::InsecureChannelCredentials()` (gRPC
> C++ **client-side** API); the Python client uses
> `grpc.aio.insecure_channel(...)`. The substantive "no TLS / no
> mTLS / no token-auth" posture is identical across all three;
> the correct symbol name on the **server** side is
> `InsecureServerCredentials`, NOT `InsecureChannelCredentials`
> (a Grep against `tools/flow_grpc_server/server/` will return
> the server-side symbol). The "Auth" and
> "TLS" axes below are **NOT** in-binary knobs — there is no
> shipped flag, config file, env var, or build option that
> turns on TLS or mTLS or token-auth on this server today. The
> only sound posture is **plaintext-on-a-trusted-segment behind
> an external TLS/identity layer** (a TLS-terminating reverse
> proxy, a service mesh sidecar, a WireGuard tunnel, etc.).
> The decision the operator makes is therefore: *which external
> hardening layer* will gate this plaintext endpoint — NOT which
> in-binary auth/TLS knob to flip. Any prose below that suggests
> the operator picks an in-binary auth/TLS mode is the bundle's
> previous aspirational framing and is wrong against the shipped
> source.

The operator must commit to three independent transport
decisions before exposing the endpoint.

| Axis | Operator's call | Public-source guidance |
| --- | --- | --- |
| **Auth** | mTLS-with-client-cert, token-based, or plaintext-on-trusted-segment | The [gRPC auth concepts](https://grpc.io/docs/guides/auth/) page on `grpc.io` documents the standard gRPC auth mechanisms; the agent routes to it for the auth-design discussion |
| **TLS** | TLS-encrypted, mTLS, or plaintext (only on a trusted segment behind a hardened firewall) | Same as above plus the public DOCA Flow gRPC Server guide on `docs.nvidia.com` for any DOCA-specific notes |
| **Network segment** | Loopback, an internal management VLAN, a control-plane-only subnet, or (rarely) a broader network | This is operator-specific; the agent never invents an IP address or interface and never asserts an endpoint is *"safe to expose on `0.0.0.0`"* |

### Language bindings

gRPC is a multi-language ecosystem. The languages the standard
gRPC tooling covers include — per the
[gRPC language support](https://grpc.io/docs/languages/) index on
`grpc.io` — C++, Java, Python, Go, Ruby, C#, Node.js, Android,
Objective-C, PHP, Dart, Kotlin, and (via community plugins) Rust.
For any of those languages, the client is generated by
`protoc` + the language-specific gRPC plugin from the shipped
`.proto` files; this skill does not pin a "supported language"
list of its own — the gRPC ecosystem's coverage is the contract.

### Optional companion components

Per the shipped source tree, the tool can be paired with two
optional companions:

- **packet_buffering/** — a packet-buffering helper for
  configurations that need it. Whether to enable it is a
  deployment-shape decision that depends on the surrounding
  Flow application's traffic pattern.
- **dpa_device/** — a DPA-side helper for configurations
  that involve DPA-offload paths on a DPA-capable device.
  Cross-references
  `doca-dpa` and
  [`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md) for
  the underlying DPA programming model and performance
  surface.

The exact configuration field names and CLI shape live on the
user's installed binary's `--help` plus the public DOCA Flow
gRPC Server page on `docs.nvidia.com`.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the headers-win-over-
docs rule, see [`doca-version`](../../doca-version/SKILL.md).
The body lives there; this skill does not duplicate it.

**The `doca_flow_grpc`-specific overlay** is:

- **The server rides the `doca-flow` library version it
  links against.** A `doca-flow` library `*.so` from one
  DOCA train paired with a gRPC server binary from another
  is a partial-install hazard per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
  The four-way match per
  [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test)
  applies without modification.
- **The `.proto` contract is versioned with the DOCA install.**
  RPC method names, message fields, and field types can shift
  across DOCA releases. Clients generated from one release's
  `.proto` files cannot be assumed to work against a server
  binary from another release; if the contract changed, the
  client must be regenerated.
- **Where it runs.** The server runs on whichever side has the
  `doca-flow` library installed (host x86 / Arm, BlueField
  Arm, or NGC container with the Flow trace flavor present).
  The client can run anywhere with gRPC tooling for the
  client language plus network reachability to the server's
  endpoint.
- **gRPC ecosystem versioning is independent of DOCA's
  versioning.** Per the
  [gRPC versioning policy](https://grpc.io/docs/what-is-grpc/core-concepts/)
  on `grpc.io`, the gRPC libraries themselves track their own
  versions; the agent's rule is *"use the gRPC library
  version your language ecosystem currently recommends, plus
  the `.proto` files from the user's DOCA install"*.

## Error taxonomy

`doca_flow_grpc`'s error surface is broader than a
local-only tool because the tool both serves remote clients and
exposes Flow state. The error layers the agent should
distinguish, in escalating order:

1. **Server-not-started.** The Flow setup is in place but
   the gRPC server process is not running. Cause: the
   operator did not start the binary, the binary crashed,
   or a precondition (Flow library missing, device not
   bound) blocked startup. Routing: confirm DOCA is
   installed via
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure)
   and the Flow library is present per
   [`doca-flow TASKS.md ## test`](../../libs/doca-flow/TASKS.md#test);
   confirm the binary's own logs.
2. **Server-binding-failed.** The init ran but the server
   could not bind the configured network address / port.
   Cause: another process holds the port, the configured
   address does not exist on the host, a TLS cert is
   missing / unreadable, an mTLS CA file is unreachable.
   The server's own error log is ground truth; do not
   guess. Routing: confirm the address / port / TLS cert
   paths against the operator's deployment config; route
   env-side issues to
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug).
3. **TLS-or-auth-rejected.** The client reached the server
   but the TLS handshake or the auth check failed. Cause:
   client cert not trusted by the server's CA, token
   missing / expired, mTLS expected but not configured on
   the client side, ALPN / cipher mismatch. The standard
   gRPC status codes (`UNAUTHENTICATED`,
   `PERMISSION_DENIED`) plus the server's own logs are
   ground truth. Route the client-side gRPC auth fix
   through the
   [gRPC auth concepts](https://grpc.io/docs/guides/auth/)
   page on `grpc.io`; the server-side configuration belongs
   to the operator's deployment config.
4. **RPC-call-error.** The TLS / auth handshake succeeded
   and the client issued an RPC, but the server returned a
   gRPC status code other than `OK`. The right move is to
   match the status code to the documented RPC contract in
   the `.proto` files — `INVALID_ARGUMENT` means the
   request message is malformed; `NOT_FOUND` means the
   requested pipe / entry does not exist; `FAILED_PRECONDITION`
   means a Flow-side precondition (pipe not created,
   validate-before-commit not performed) blocked the
   operation. Routing for the Flow-side preconditions:
   [`doca-flow TASKS.md ## modify`](../../libs/doca-flow/TASKS.md#modify)
   plus
   [`doca-flow CAPABILITIES.md ## Error taxonomy`](../../libs/doca-flow/CAPABILITIES.md#error-taxonomy).
5. **Flow-precondition-failed.** The RPC was syntactically
   valid but the underlying Flow application is not in a
   state to accept it (port not started, pipe not created,
   another mutation in flight). The right move is the
   doca-flow side, not the gRPC server side.
6. **Version.** Cross-cutting partial-install / mixed-
   version layer per
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
   Symptoms: the server's `.proto` contract does not match
   the client-side generated stubs (the client was
   generated from a different DOCA release's `.proto`);
   the server binary version disagrees with the Flow
   library version it links. Walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   end-to-end.
7. **Cross-cutting.** All layers above are clean and the
   client still cannot use the server. The cause is below
   DOCA — driver, firmware, BlueField mode, network
   reachability, kernel-level firewall. Hand off to
   [`doca-debug ## debug`](../../doca-debug/SKILL.md) and
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug).

The gRPC server itself uses the cross-library
`DOCA_ERROR_*` values when calling into `doca-flow`; the agent
maps those into the layers above before quoting them.

## Observability

`doca_flow_grpc` exposes three observability surfaces
the agent should consult, in order:

- **Server logs.** The binary's own log output (per
  `DOCA_LOG_LEVEL` and the standard DOCA logging surface; see
  [`doca-programming-guide CAPABILITIES.md ## Observability`](../../doca-programming-guide/CAPABILITIES.md#observability))
  is the first source of truth for server-side errors.
  Bind start / bind failure / per-RPC accept and reject
  lines live here.
- **Client-side gRPC status codes.** Every failed RPC the
  client sees carries a standard gRPC status code per the
  [gRPC status codes](https://grpc.io/docs/guides/status-codes/)
  reference on `grpc.io`. The agent must quote the code
  (`UNAVAILABLE` / `UNAUTHENTICATED` / `INVALID_ARGUMENT` /
  `NOT_FOUND` / `FAILED_PRECONDITION` / etc.) verbatim;
  paraphrasing the code is the canonical lost-fidelity
  failure.
- **The live Flow application's observability surface.** Per
  [`doca-flow CAPABILITIES.md ## Observability`](../../libs/doca-flow/CAPABILITIES.md#observability),
  the Flow library exposes pipe / entry / counter / inspector
  state programmatically. When the gRPC RPC succeeds but the
  user's downstream behavior is wrong, this is where the
  diagnosis continues.

For the env-side counters that bound the deployment (link
state, PCIe, IB), reach for
[`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

Exposing `doca_flow_grpc` in production is a
**deliberately high-stakes posture** — it is an admin attack
surface attached to the dataplane:

- **Treat the endpoint as a privileged surface.** Bind it on
  a trusted segment (loopback, an internal management VLAN, a
  control-plane-only subnet) and require auth + TLS before
  any client beyond a local smoke client. Recommend the
  operator confirm *who can reach the endpoint* before
  declaring it exposed.
- **Auth / TLS are not optional.** Per the
  [gRPC auth concepts](https://grpc.io/docs/guides/auth/)
  page on `grpc.io`, gRPC supports mTLS and token-based
  auth out of the box. Plaintext-without-auth is acceptable
  only on a genuinely trusted segment (e.g. loopback within
  one process boundary) — and even there the agent must
  surface the trade-off rather than default to it.
- **Smoke-before-bulk is mandatory.** Before pointing a fleet
  of clients (or any production client) at the server, the
  agent runs the start → bind → one-client-smoke → read-only-
  RPC sequence in [`TASKS.md ## test`](TASKS.md#test). A
  server exposed without that sequence is a guess against a
  possibly-unbound endpoint, a possibly-mis-configured TLS
  posture, or a possibly-mismatched contract.
- **State-changing RPCs are dataplane-affecting.** Every RPC
  that creates / mutates / deletes pipes or entries does to
  Flow state exactly what a direct `libdoca_flow.so` call
  would do. The agent must label every RPC as read-only or
  state-changing and must NOT issue a state-changing RPC
  before a clean read-only smoke.
- **Never widen access as a workaround.** If a client cannot
  reach the server or an RPC is rejected, walk
  [`## Error taxonomy`](#error-taxonomy) — do not disable
  TLS, widen the bind address to `0.0.0.0`, or grant a token
  unconditionally so the symptom goes away.
- **Do not invent RPC names, message field names, or
  endpoint paths.** The shipped `.proto` files on the user's
  installed version are the contract; prose-derived names
  are the most common hallucination failure for this skill.

## Public-source pointer

The single canonical public source for `doca_flow_grpc`
is the **DOCA Flow gRPC Server** page on `docs.nvidia.com`,
reachable through
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
The second authoritative source on the user's install is the
shipped `.proto` file set under the tool's source tree. For
generic gRPC ecosystem documentation (auth, status codes,
language bindings, versioning), the canonical source is
[`grpc.io`](https://grpc.io/) — the public DOCA Flow gRPC
Server page does not duplicate that ecosystem documentation.
For the `doca-flow` API the server programs, see
[`doca-flow`](../../libs/doca-flow/SKILL.md) plus the public
DOCA Flow guide reached the same way.
