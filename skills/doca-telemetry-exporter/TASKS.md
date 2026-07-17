# DOCA Telemetry Exporter workflows

**Where to start:** The verbs run `configure → build → modify →
run → test → debug`. Skip ahead only when the user is already past a
verb. The `## test` verb is an iterative loop (single-event smoke →
receiver-side reception check → multi-event smoke → load behavior →
loop back if the receiver staging or transport changes), not a
one-shot pass — see the eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the exporter capability surface, the
exporter-vs-service role split, the object family, the
schema-register-before-emit lifecycle, the capability-query rule,
the error taxonomy (including the `AGAIN`-means-drop-not-block
rule), observability, and safety policy, see
[CAPABILITIES.md](CAPABILITIES.md). For where to find docs, the
installed DOCA layout, or release notes, route through
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through the
steps in order, verifying preconditions before recommending the next
call.

## configure

Goal: stand up a `doca_telemetry_exporter` context inside the user's
application, with at least one schema registered and a reachable
receiving telemetry consumer, before any event is emitted.

Steps the agent should walk the user through:

1. **Confirm the role: this is the PUBLISHER side.** Before any
   code change, surface the exporter-vs-service distinction per the
   role-split table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
   `doca-telemetry-exporter` is what the user's application links
   to publish events; the aggregating / receiving side is the DOCA
   Telemetry Service, a separate DOCA service with its own public
   guide reachable via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
   An agent that walks the user toward the service guide when they
   wanted to emit from the app is wrong; an agent that recommends
   linking the exporter when the user actually wanted the
   aggregating service is wrong. State the role first.
2. **Verify the receiver is up and reachable BEFORE the
   exporter.** Walk the permission + staging matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy):
   (a) the receiving telemetry consumer is started and reachable
   on the host where the application will run; (b) the
   application's user can write to the transport endpoint the
   receiver listens on (no sudo required as a rule — that is a
   common first-app misconception); (c) the user knows
   independently how to confirm reception on the receiver side
   (its own log / status). If the receiver is not up first, route
   to its own public guide via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
   or to [`doca-setup TASKS.md ## configure`](../../doca-setup/TASKS.md#configure)
   for the env-side path.
3. **Confirm the installed DOCA version and run capability
   discovery.** Use the procedure in
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).
   Quote the version observed (`pkg-config --modversion
   doca-telemetry-exporter`, then `doca_caps --version`); do not
   assume "latest". Then run the matching
   `doca_telemetry_exporter_*_get_*` queries (max schema fields,
   max event size, supported event types) — per the capability-query
   rule in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   the queried value is the runtime authority, not the agent's
   memory. Quote the values back to the user.
4. **Define and REGISTER the schema BEFORE any emit.** Sketch the
   `doca_telemetry_exporter_schema` for the events the app will
   publish: field names + matching `doca_telemetry_exporter_type`
   (counter / gauge / event) per field. Register every schema with
   the exporter context BEFORE the first emit that references it
   — per [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   schema-register-before-emit rule. Emitting against an
   unregistered schema returns `DOCA_ERROR_NOT_FOUND`; emitting a
   value whose type does not match the registered schema returns
   `DOCA_ERROR_INVALID_VALUE`.
5. **Create the `doca_telemetry_exporter_source`(s).** One source
   per logical reporter (per-worker / per-pipeline-stage / per-
   tenant), per the object table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
   The agent should not default to *one* source for the whole app
   without asking — multi-source aggregation is one of the things
   the receiver fans out on, and the user's choice here is
   workload-bound.
6. **Confirm the exporter is the right tool.** Walk the
   path-selection rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes):
   if the user really wants stdout / structured-log shipping, use
   `doca_log` instead (covered in
   [`doca-programming-guide`](../../doca-programming-guide/SKILL.md));
   if the user wants a non-DOCA-aware Prometheus scrape endpoint,
   use a Prometheus client library directly; if the user actually
   needs real-time event subscription back INTO the app, route to
   [`doca-comch`](../doca-comch/SKILL.md). Picking the exporter
   *for* the user when the path-selection rule rules it out is a
   wrong answer regardless of how cleanly the rest of the
   configure step goes.
7. **Start the exporter context.** `doca_ctx_start()` on the
   exporter; per the universal lifecycle in
   [`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure),
   nothing emits cleanly until the context is in `RUNNING`. If
   start fails, route through the error taxonomy in
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   before retrying.

If any step fails with a `DOCA_ERROR_*`, route through the error
taxonomy in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
before retrying. In particular, `DOCA_ERROR_AGAIN` is *not* a
configure-time failure; it shows up at emit time under transport
load and is governed by the hot-path drop-not-block rule.

## build

Goal: produce an application binary that links DOCA Telemetry
Exporter against the user's installed DOCA, using the canonical
cross-library build pattern.

The build pattern for any DOCA C/C++ consumer is **identical**
across libraries — `pkg-config` for include + link flags, meson or
CMake as the build system — and is fully documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the exporter-specific overlay:

| Slot | Value for Telemetry Exporter | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-telemetry-exporter` | The library's `.pc` file installed by the DOCA host packages. Wrong module name = `pkg-config: Package 'doca-telemetry-exporter' was not found` (most often the user typed `doca-telemetry` and was reading the receiving service's guide by mistake — re-check role) |
| Include flags | `pkg-config --cflags doca-telemetry-exporter` | Resolves to exporter headers under $(pkg-config --variable=includedir doca-common) for the exporter subset |
| Link flags | `pkg-config --libs doca-telemetry-exporter` | Pulls in whatever `pkg-config --libs` resolves on this install (do not predict the `-l<name>` form by hand — `.so` basenames use underscores, `.pc` names use hyphens, and `pkg-config` is the only correct translator) plus the transitive set the resolver computes against this install |
| Header check | the artifact's public header resolvable under whichever include directory `pkg-config --cflags` reports (do not hardcode the include path — the install layout can move) on the host | If `pkg-config --cflags doca-telemetry-exporter` resolves but the include is missing, the install is partial — route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2 |
| Minimum required DOCA version | Query with `pkg-config --modversion doca-telemetry-exporter`; never hardcode in build files | Cross-version build/runtime mixing breaks per [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility) |

For non-C consumers (Rust, Go, Python), the link surface is the
same `*.so` files; the FFI wrapper layer is the language-specific
binding and is out of scope for this skill — but the slots above
are still the load-bearing inputs the wrapper needs.

## modify

Goal: take a shipped DOCA Telemetry Exporter sample as the verified
starting point and apply a **minimum-diff modification** to express
the user's intent.

The universal modify-a-shipped-sample workflow lives in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify).
Use it as-is. The exporter-specific overlay is the *modify-from-
sample schema fill* — the slots the agent must elicit from the user
before recommending any code-level edit:

| Slot | What the agent asks the user | Exporter-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which sample under `/opt/mellanox/doca/samples/doca_telemetry_exporter/`? | Pick the closest in *event shape* (counter / gauge / event) and *source layout* (single source / multi source) to the user's intent. A smaller diff is always safer than a re-architecture |
| 2. Event schema | Which fields, with which `doca_telemetry_exporter_type` each, does the user want to emit? | Re-validate the proposed schema against `doca_telemetry_exporter_*_get_*` per [`## configure`](#configure) step 3; an event shape that works on one install may exceed `max event size` on another |
| 3. Schema registration site | Where in the user's app flow does the schema register relative to the first emit? | Per [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) schema-register-before-emit rule, registration MUST happen before the first emit; if the sample emits before registering (some samples register implicitly on context-start), keep that order intact when modifying |
| 4. Emit-site behavior on `AGAIN` | What does the modified app do when an emit returns `DOCA_ERROR_AGAIN`? | Per [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy) hot-path drop-not-block rule, the correct behavior is to DROP the event or buffer it bounded — never to block the data path. If the sample's existing emit-site does the wrong thing under load (sleep / retry / busy-wait), that is a sample bug that needs to be edited out before the modify lands |
| 5. Receiver staging | Is the receiver the user is targeting started before the modified app starts? | Per [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy) consumer-up-first rule; not a code change but a runtime-staging concern the modify must surface |
| 6. Build manifest | Keep the sample's existing `meson.build` (which already wires `pkg-config doca-telemetry-exporter`)? | Yes. Do not switch to a hand-rolled Makefile for *"simplicity"* — it removes the version-check rail |

The agent emits an *intent description + the filled slots*; the
*actual* unified diff against the sample source is produced by the
modify-from-sample renderer (deferred to a future round, per
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify)).
Until the renderer ships, the agent must walk the user through the
diff line-by-line against the sample source they read on disk, and
have the user paste back the result for validation.

## run

Goal: actually execute the built application against the user's
installed DOCA, with the receiver up first, the schema registered,
and the application's user able to write to the transport.

Steps the agent should walk the user through:

1. **Start the receiver first.** The receiving telemetry consumer
   (the DOCA Telemetry Service or whichever consumer the exporter
   is configured to write to) must be running and reachable BEFORE
   the application starts the exporter context — per the
   permission + staging matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
   Starting the exporter first risks queued or dropped events with
   no application-side error.
2. **Run the application as its normal user (no sudo as a rule).**
   Per [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   the exporter runs in the application's process as the
   application's user. A `DOCA_ERROR_NOT_PERMITTED` from the first
   emit is almost always the receiver's transport endpoint
   permissions, not a missing capability on the exporter process —
   resist the reflex to add `sudo`; fix the receiver-side
   permission instead.
3. **Smoke-emit one event and confirm reception on the receiver
   side.** Before any bulk emit, emit ONE event from the
   application and have the user independently confirm the
   receiver logged it. A successful emit-and-receive confirms the
   permission + receiver staging + transport path are all correct;
   only then expand to bulk emits. Skipping this step is the most
   common reason *"emit returns success and we see nothing on the
   receiver"*.
4. **Capture the structured log.** Set `DOCA_LOG_LEVEL=trace` for
   the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability)).
   This is the cheapest way to make the exporter lifecycle
   transitions and per-emit calls visible on first failure.
5. **Watch for `DOCA_ERROR_AGAIN` only when the bulk emit starts.**
   `AGAIN` shows up under transport load, not at the first emit.
   When it appears, the application's correct response is to DROP
   the event (or push to a bounded app-side buffer) — never to
   block the hot path. See [`## debug`](#debug) layer 6 for the
   per-emit pattern.

## test

Goal: prove the configured exporter context can actually deliver
structured telemetry to the receiver, end-to-end, before claiming
the *"build a first telemetry-emitting app"* journey is done.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the receiver staging, the schema shape, the source layout,
or the under-load drop behavior. The loop terminates when either
(a) the user's intended emit cadence runs end-to-end with the
expected events reaching the receiver and the under-load behavior
is drop-not-block, or (b) the agent has narrowed the failure cause
to a layer outside the exporter itself (receiver / transport /
driver) and escalated to the matching skill.

Iteration shape:

1. **Receiver-up smoke first.** Confirm the receiving consumer is
   running on the host and the application's user can write to
   the transport. If `doca_ctx_start()` on the exporter context
   returns `DOCA_ERROR_NOT_PERMITTED`, the receiver-side endpoint
   permissions are the suspect, not the exporter — fix per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
   permission row before advancing.
2. **Capability re-check.** Re-run the
   `doca_telemetry_exporter_*_get_*` queries. If the proposed
   schema exceeds a queried cap (`max schema fields`, `max event
   size`), that *is* the answer for this install; update the
   schema (or update the install) before adding emit code.
3. **Single-event smoke.** Emit ONE event from the application
   and have the user independently confirm the receiver logged
   it. If the smoke returns success on the publisher side but
   nothing on the receiver, the consumer-up-first staging is the
   prime suspect, not the exporter call. Per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   skipping this step is the most common reason *"my emit
   succeeded but the dashboard is empty"*.
4. **Schema-register order pass.** Confirm that schema
   registration happens BEFORE the first emit in every code path
   that emits. A `DOCA_ERROR_NOT_FOUND` from an emit is the
   canonical *"forgot to register the schema first"* symptom,
   per [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
5. **Multi-event smoke.** Loop a small N (say, 100) emits with
   the receiver reading concurrently; confirm the receiver count
   matches the publisher count. Catches lost events that the
   per-emit return value alone would not surface.
6. **Under-load `AGAIN` behavior.** Push the emit rate up until
   the transport queue saturates and the publisher starts
   returning `DOCA_ERROR_AGAIN`. The application MUST drop (or
   bounded-buffer) — confirm the data path's latency does NOT
   rise. If the data path's latency rises with the emit rate,
   the modify violated the hot-path drop-not-block rule per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
   and the fix is at the emit-site in the app, not on the
   exporter side.

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_NOT_PERMITTED` on first emit | Receiver-side transport endpoint permissions wrong, OR user is trying to run with sudo unnecessarily | Re-walk the permission row in [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy); fix on the receiver side, not by adding sudo to the app |
| `DOCA_ERROR_NOT_FOUND` on first emit | Schema was never registered with this exporter context, OR the source the emit references was never created | Walk the schema-register-before-emit rule in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes); fix the registration order in the modify before re-running |
| Emit returns success but receiver log is empty | Receiver is not up, OR the exporter is configured against a transport the receiver does not listen on | Re-walk the consumer-up-first staging in [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy); verify the receiver independently before suspecting the exporter |
| `DOCA_ERROR_AGAIN` appears under bulk load | Receiver is slower than the publisher; transport queue is full | This is by design — the app must DROP / bounded-buffer at the emit site, not block. If the drop rate is unacceptable, widen the receiver side (separate problem on the receiver guide), do NOT back-pressure the data path |
| Same code drops events on host A, doesn't on host B | Receiver throughput / transport differs between hosts | Re-narrow to the receiver side; the exporter's behavior is the same on both, the variance is at the consumer / transport layer |

Loop termination: stop iterating once two consecutive iterations of
the same kind don't change anything — that means the cause is below
the exporter (transport, receiver, driver). Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug) with
the captured layer-1-through-5 evidence and the receiver-side log
state.

## debug

Goal: when a DOCA Telemetry Exporter call returns a `DOCA_ERROR_*`
(or events do not show up on the receiver), narrow the cause to a
specific layer and act on it.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk through it in order — install → version → build → link →
runtime → program → driver — *before* recommending exporter-
specific fixes. This skill's overlay names the exporter-specific
manifestation at layers 5 (runtime) and 6 (program):

**Layer 5 (runtime) — exporter overlay.**

- Walk the role rule: did the user actually want the exporter
  (publisher) and not the receiving DOCA Telemetry Service? If
  the user is reading the service's guide, the answer is to route
  them to the service skill via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
  not to debug the exporter further.
- Walk the receiver-up-first staging: is the receiving consumer
  running and reachable BEFORE the exporter starts? A passing
  build + a missing receiver + an empty dashboard is the
  canonical *"emit succeeded into a transport with no reader"*
  symptom — fix at the receiver side, not in the exporter code.
- Walk the permission state: the exporter does NOT need sudo as
  a rule. A `DOCA_ERROR_NOT_PERMITTED` on first emit means the
  receiver's transport endpoint permissions are wrong — fix on
  the receiver side; resist adding `sudo` to the application.

**Layer 6 (program) — exporter overlay.**

- The schema-register-before-emit trap: an emit that returns
  `DOCA_ERROR_NOT_FOUND` is misinterpreting the API surface — the
  schema (or the source) the emit names was never registered
  with this exporter context. Walk the configure-time
  registration order against the app's startup path, per
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
- Lifecycle order: configure (register schemas + create sources)
  → start → emit → stop → destroy. Out-of-order returns
  `DOCA_ERROR_BAD_STATE`. The most common case is emitting
  before the exporter context reached `RUNNING`.
- Hot-path drop-not-block: an emit-site that retries / sleeps /
  busy-waits on `DOCA_ERROR_AGAIN` is the canonical first-app
  bug — it back-pressures the data path through telemetry, which
  is exactly what the exporter is designed to prevent. The fix
  is at the emit-site in the app: DROP or push to a bounded
  app-side buffer; never block. Per
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- Value-vs-schema mismatch: a `DOCA_ERROR_INVALID_VALUE` on emit
  is a type mismatch against the registered schema, a missing
  required field, or an oversized payload. Re-read the registered
  schema against the matching `doca_telemetry_exporter_*_get_*`
  caps; fix in the emit, not by widening the cap (it is
  install-bound).

Once the layer is identified, route to the matching debug verb on
the matching skill: install / build / link / driver to
[`doca-setup ## debug`](../../doca-setup/TASKS.md#debug); version
to [`doca-version ## debug`](../../doca-version/TASKS.md#debug);
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
- **receiver / collector / aggregating service.** Setting up the
  DOCA Telemetry Service (the receiver) or any downstream sink
  (NetFlow / IPFIX / Prometheus / Grafana) — out of scope for this
  skill. Route to the service's own public guide via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
  This skill is publisher-side only.
- **deploy.** Deploying telemetry-emitting applications at scale
  across many hosts, Kubernetes operator workflows with telemetry
  sidecars — out of scope for Phase 1 and reserved for a future
  platform skill.
- **firmware burn / reset.** The exporter does not depend on
  firmware-layer state directly; if the debug ladder lands on a
  driver-layer issue (`DOCA_ERROR_DRIVER` from an exporter call),
  the fix is via the env-side skill:
  [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) layer 5,
  then upstream documentation reachable through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Command appendix

Every command below is **cross-cutting on DOCA Telemetry
Exporter** — it answers a recurring class of question that comes
up in the verbs above. The agent should treat the *class* as
load-bearing; the worked example is a single instance. Run-as
user is the application's normal unprivileged user unless noted. Rows that need elevated privileges call that out explicitly. (and is rarely needed for the exporter
itself, per [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)).

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
3. If the probe fails, fall back to the manual command in the
   row. Report *"falling back to manual chain"*.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics (four-way match, NGC,
   headers-win) are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Command (worked example) | Owning step | Class of question it answers | What healthy output looks like |
| --- | --- | --- | --- |
| `pkg-config --modversion doca-telemetry-exporter` | `## configure` step 3; `## build` slot 1 | What is the build-time DOCA Telemetry Exporter version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --cflags --libs doca-telemetry-exporter` | `## build` | What include + link flags does the linker need? | Trust whatever `pkg-config --cflags --libs` produces on this install. Do not hardcode either the `-I` include path or the `-l<name>` flag form — both can drift between DOCA install profiles and DOCA majors; the on-disk `.so` basenames use underscores on every release where we have ground truth, while the `.pc` package names use hyphens, and `pkg-config` is the only thing that resolves both correctly. Hand-crafted `-l` lines silently break when DOCA upgrades. |
| `ls /opt/mellanox/doca/samples/doca_telemetry_exporter/` | `## modify` slot 1 | Which exporter samples ship in this install, and which is the closest starting point? | A list of sample directories named after the event shape / source layout they demonstrate |
| `doca_caps --version` | `## configure` step 3; `## test` step 2 | What is the *runtime* DOCA version? | A semver string matching `pkg-config --modversion doca-telemetry-exporter` |
| `id` | `## configure` step 2; `## run` step 2 | Is the application user the one the receiver's transport endpoint expects to allow? | The user's id matches what owns / can write the receiver's transport endpoint. Mismatch = `DOCA_ERROR_NOT_PERMITTED` on first emit — fix on the receiver side, not by adding sudo |
| `cat /opt/mellanox/doca/applications/VERSION` | `## configure` step 3; `## debug` layer 1 | What does the install tree itself claim its version is? | A semver string matching the other two version sources |
| `dmesg \| tail -n 40` (sudo) | `## debug` layer 7 | What did the kernel / driver log around the last exporter call? | Empty or recent benign messages. Repeated mlx5 / network / socket errors → driver / env-layer bug; route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 4 | What did the structured DOCA logger emit for the first failing emit? | A trace-level line on every lifecycle transition and every emit. Per-emit `AGAIN` traces under load = transport-full / drop-or-buffer (NOT an error to retry on the hot path) |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the exporter-specific rows on top.
