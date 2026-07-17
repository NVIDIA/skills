# DOCA Flow workflows

**Where to start:** The verbs run `configure → build → modify → run →
test → debug`; skip ahead only when the user is already past a verb.
`## test` is an iterative loop (validate → cross-check → counter wiring →
negative test → loop back if the spec changed), not a one-shot pass.

For the capability matrix, version compatibility, error taxonomy,
observability, and safety policy these workflows assume, see
[CAPABILITIES.md](CAPABILITIES.md). For docs, installed-DOCA layout, or
release notes, route through
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
Each verb describes the **shape of the workflow**, not a copy-paste
recipe: walk the user through the steps in order, verifying preconditions
before recommending the next call.

## Key samples to read first

The installed samples are the canonical, compile-tested reference for THIS
install — read the relevant one before writing or quoting code (`ls` and
read it on the user's machine; layouts change between releases). These few
cover the bulk of customer patterns:

| Sample (under `/opt/mellanox/doca/samples/doca_flow/`) | Pattern it teaches |
| --- | --- |
| `flow_port_fwd/` | Match → `FWD_PORT`: the simplest VNF forward |
| `flow_switch_single/` | Switch mode + representor (multi-VF eswitch) |
| `flow_control_pipe/` | Root **control / classifier** pipe branching to non-root pipes |
| `flow_lpm/` | Longest-prefix-match routing pipe |
| `flow_acl/` | ACL (5-tuple allow / deny) pipe |
| `flow_hash_pipe/` | Hash-based fan-out / load balancing (slots = power of 2) |
| `flow_vxlan_encap/` | Tunnel encap action |
| `flow_shared_meter/` | Shared meter / rate limiting + `parser_meta.meter_color` |
| `flow_drop/` | Explicit drop pipe (pairs with `fwd_miss`) |
| `flow_aging/` | Entry aging / time-based eviction |

For connection tracking and NAT see [`## flow-ct`](#flow-ct); for larger
end-to-end apps (`doca_flow_ct`, `doca_simple_fwd_vnf`, …) see
`/opt/mellanox/doca/applications/`.

## configure

Goal: bring up a DOCA Flow port on a supported NVIDIA NIC/DPU and confirm
the environment is ready for pipe construction.

1. **Confirm the installed DOCA version.** Use the procedure in
   `doca-public-knowledge-map`; quote the version observed, do not assume
   "latest".
2. **Check device placement BEFORE anything else — host-side vs DPU
   Arm.** On a BlueField the hardware-steering plane is owned by only one
   side of the card. Read `INTERNAL_CPU_MODEL` from `mlxconfig -d <pcie> q`
   (the `mlxconfig` command is owned by
   [`doca-debug`](../../doca-debug/TASKS.md#command-appendix)):
   - `SEPARATED_HOST` → NIC mode; the steering plane lives on the **DPU
     Arm**, and the **x86 host function cannot start a Flow port** (it
     fails with the placement signature in
     [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy):
     `Failed to get hws cap` / `dest action ROOT … err -121`). On the host
     in this mode, STOP and route Flow work to the DPU Arm side, or change
     the card's mode through
     [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) — do
     not proceed to pipe construction.
   - `EMBEDDED_CPU` (DPU mode), or a NIC whose opened function advertises
     a usable steering plane → placement is fine; continue.
   Being listed by `doca_caps --list-devs` is NOT proof the opened
   function has a steering plane — placement is the gate, enumeration is
   not.
3. **Discover device capabilities.** Run the installed `doca_caps` tool
   and the Flow capability-query API; record the supported match kinds,
   action kinds, and pipe/entry budgets. The matrix to compare against lives in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
4. **Enumerate ports and representors.** Confirm the port the user wants
   to program is visible (`devlink dev show` plus the installed Flow
   port-enumeration sample) and that the expected representors are
   present.
5. **Runtime prerequisites — hugepages + HWS devargs, before any Flow
   program runs.** A DOCA Flow program needs hugepages allocated and
   mounted first, and hardware steering (HWS) must be requested through
   DPDK devargs:

```bash
echo '1024' | sudo tee -a /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
sudo mkdir -p /mnt/huge
sudo mount -t hugetlbfs -o pagesize=2M nodev /mnt/huge
```

   - **HWS** is enabled via DPDK devarg `dv_flow_en=2` AND a `mode_args`
     value containing `hws` (e.g. `"vnf,hws"` or `"switch,hws"`). Without
     HWS, many features (shared encap/decap, shared meter across entries,
     pipe resize, …) are unavailable.
   - **Switch mode** additionally requires the DPDK devargs
     `fdb_def_rule_en=0,vport_match=1,repr_matching_en=0,dv_xmeta_en=4`
     (the DOCA samples hide these behind `-r <pci>,<reps>`).
6. **Create the DPDK↔DOCA device mapping (the #1 runtime failure), then
   bring up the Flow port.** Three runtime signatures — `mlx5_common:
   Probe again parameters aren't compatible`, `doca_dpdk_port_as_dev(...):
   Requested Resource Not Found`, `doca_flow_port_start(...): Unknown
   error` — all trace to this mapping not being created correctly. The
   shipped `dpdk_init` helper, wired via
   `doca_argp_set_dpdk_program(dpdk_init)`, does the whole sequence for
   you; the load-bearing rules if you roll your own:
   - **EAL must NOT auto-probe the mlx5 PFs.** Pass a non-existent-device
     sentinel (`-a 00:0.0`) to `rte_eal_init` so EAL scans PCI but
     allowlists nothing, then let DOCA do the real probe. Skipping this →
     `Probe again parameters aren't compatible`.
   - **`doca_dpdk_port_probe(dev, "dv_flow_en=2")` creates the mapping;**
     `doca_dpdk_port_as_dev()` only *reads* it. Open each NIC by PCI BDF
     first (`open_doca_device_with_pci(bdf, NULL, &dev)`), then probe. The
     NIC has **no** DPDK port until this probe runs: `rte_eth_dev_count_avail()`
     stays 0 and any "need N ports" check (`dpdk_utils.c`) aborts if you
     count or queue-setup before probing *every* device — collecting BDFs from
     argv is not the same as probing them. **Required strict order — gate
     the bridge here, not at port-start:**
     1. `rte_eal_init` with the no-auto-probe sentinel above.
     2. Open each `doca_dev` from argv (one per requested device).
     3. Call `doca_dpdk_port_probe(dev, "dv_flow_en=2")` for **every**
        opened device — this is the call that creates the DPDK port from
        the `doca_dev` (`samples/doca_flow/flow_common.c`).
     4. Only **after** every device has been probed, call
        `rte_eth_dev_count_avail()` and verify it returns ≥ the expected
        port count (`samples/doca_flow/flow_common.c` precedes
        `applications/common/dpdk_utils.c`, where a count below
        `nb_ports` returns `DOCA_ERROR_DRIVER`).
     5. **If count == 0, or count < the expected port count, propagate the
        error out of `main()` with a non-zero exit code immediately —
        log the failure verbatim and abort.** Counting before probe always
        reports `0`, so a binary that reaches the keep-alive loop in
        [`## run`](#run) step 6 with `count == 0` is forwarding nothing
        while *appearing* to run — the watchdog cannot tell the difference
        from a healthy program. It is a hard error here, not a warning,
        and not something to "log and continue" past.
     6. Then call `doca_dpdk_get_first_port_id(dev, &port_id)` /
        `RTE_ETH_FOREACH_DEV` + `doca_dpdk_port_as_dev` to derive the
        numeric port id, and proceed to `rte_eth_dev_configure` /
        `rte_eth_dev_start`.
   - **Hardware-neutral `port_id` *and* BDFs — never hard-code `0`/`1` or a
     PCI address.** The DOCA-probed NIC can land at `port_id 2/3` when the
     PCI auto-scan grabs other devices first, and its BDF differs per host —
     a hard-coded `0000:04:00.0` typically lands on an unrelated/virtio
     function (`PCI_BUS: Requested device … cannot be used` →
     `doca_flow_port_start … Unknown error`). Take the device(s) from the
     **program args after `--`** with the shipped `register_flow_device_params()`
     (`flow_common.c`, `flow_common.h`): it registers the canonical
     *repeatable* `-a` / `--device pci/<bdf>,<devargs>` param
     (`DOCA_ARGP_TYPE_DEVICE`) and opens each `doca_dev` for you — this is the
     exact shape the samples are driven with (`-- -a pci/08:00.0,dv_flow_en=2`).
     Do **not** invent your own (`register_argp_device_param()` does *not*
     exist) and do **not** hard-reject when no BDF is supplied — fall back to
     discovery (`doca_devinfo_create_list` / `lspci -d 15b3:`). Then probe and
     derive the numeric id from the `doca_dev` with
     `doca_dpdk_get_first_port_id(dev, &port_id)` or `RTE_ETH_FOREACH_DEV`
     + `doca_dpdk_port_as_dev`. The same source then runs on any host.
   - **Bridging two physical PFs:** a naive `DOCA_FLOW_FWD_PORT` across two
     un-peered PFs fails at pipe-create with `... are not hairpin peers`.
     On ConnectX-7+ the simplest correct bridge is the `dv_flow_en=2`
     E-Switch domain (what `doca_simple_fwd_vnf` uses); DPDK hairpin queues
     (`rte_eth_*_hairpin_queue_setup`, with `peer_count=1`, `manual_bind=1`,
     `tx_explicit=1`) are the alternative when you need hardware hairpin
     semantics.

   Then the per-port Flow bring-up — **two setters are MANDATORY, both
   enforced in `engine_port.c`**: `doca_flow_port_cfg_set_port_id()` (else
   `doca_flow_port_start()` fails *"port ID is mandatory"*, `engine_port.c`)
   **and** `doca_flow_port_cfg_set_dev()` with the matching probed `doca_dev`
   (else *"either doca_dev or doca_dev_rep must be provided"*,
   `engine_port.c`; a `dev_rep` is rejected in VNF mode,
   `engine_port.c`). Required per-port sequence:

```c
struct doca_flow_port_cfg *port_cfg;
doca_flow_port_cfg_create(&port_cfg);
doca_flow_port_cfg_set_port_id(port_cfg, port_id);   /* MANDATORY: else "port ID is mandatory" */
doca_flow_port_cfg_set_dev(port_cfg, dpdk_dev);      /* MANDATORY: else "doca_dev or doca_dev_rep must be provided" */
struct doca_flow_port *port;
doca_flow_port_start(port_cfg, &port);               /* takes 2 args */
doca_flow_port_cfg_destroy(port_cfg);
```

   **Bind the port with `set_port_id()`, not `set_devargs()`** — the
   trap behind a clean compile that dies the instant the port starts.
   `doca_flow_port_cfg_set_devargs()` carries *other* per-port options
   (the shipped `flow_common.c` passes `"th_win_us=0"`), not
   identity; a numeric devargs string like `"0"` leaves the `port_id`
   unset and `doca_flow_port_start()` aborts with *"port ID is
   mandatory"*. Only `doca_flow_port_cfg_set_port_id()` binds the port
   (`flow_common.c`), and it is a *different* "devargs" from the
   EAL probe devarg `dv_flow_en=2` in step 5 — three unrelated uses of
   the word. Pair it with the *matching* probed `doca_dev` via
   `set_dev()` — **mandatory, not HWS-only**: a missing dev aborts with
   *"either doca_dev or doca_dev_rep must be provided"*
   (`engine_port.c`, sometimes surfaced as a bare *"Unknown
   error"*). The `port_id` MUST be the DOCA-probed id from above (per
   the hardware-neutral rule earlier in this step); a hard-coded `0`
   surfaces as `doca_dpdk_port_as_dev(0): Requested Resource Not
   Found`.

   Do this for **every** DPDK port the app uses, in the order the DPDK port
   IDs come out (`rte_eth_dev_count_avail()` / `RTE_ETH_FOREACH_DEV()`; IDs
   are 0..N-1 in `-a <BDF>` attach order). The lifecycle is *cfg created →
   `port_id` set → dev set → port started → cfg destroyed*; do not create
   pipes before the port reports started, and any `DOCA_FLOW_FWD_PORT`
   destination port must be started before a pipe forwards to it.
7. **Sanity check before any pipe work.** Confirm with the user: ingress
   port, egress representor(s), traffic class. If unclear, stop and ask —
   do not invent.

If any step fails with a `DOCA_ERROR_*`, route through
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
before retrying.

## build

Goal: construct a pipe specification that expresses the user's intent
without committing to hardware yet.

1. Restate the user's intent in match/action terms ("match dst MAC = X,
   forward to representor Y, count").
2. **Verify each match kind and action kind against the active
   capability set** from
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
   If the device does not support a kind, stop and offer alternatives; do
   not generate a spec that will fail at validate.
3. Allocate the pipe spec via the Flow pipe-create API with explicit
   match-mask and action-mask declarations. Implicit-anything is the
   leading cause of misprogrammed steering.
4. Attach a counter to every entry the user wants to *observe* in
   production; `## debug` assumes counters exist. Per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
   counters are a **two-step setup**: BOTH the per-entry monitor
   attachment AND a global pool reservation are required, or the
   counter silently produces no telemetry. Reserve the pool **once**
   at `doca_flow_init` time — for NON_SHARED per-entry counters via
   `doca_flow_cfg_set_nr_counters(cfg, N)` (size `N` to the sum of
   `NON_SHARED` monitor entries across every pipe plus ~10% headroom);
   for shared counters / meters via the shared-pool path in
   [`## shared-resources`](#shared-resources):

```c
struct doca_flow_cfg *cfg;
doca_flow_cfg_create(&cfg);
doca_flow_cfg_set_nr_counters(cfg, N);   /* MANDATORY for NON_SHARED per-entry counters */
doca_flow_init(cfg);
doca_flow_cfg_destroy(cfg);
```

   The pool reservation above is only step one. The **per-entry** step
   is to pass the *same* populated `&monitor` (with `monitor.counter_type
   = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED`) as the `monitor` argument of the
   pipe-type add-entry call — it is slot 6 of the 10-argument
   `doca_flow_pipe_basic_add_entry(queue, pipe, match, action_idx,
   actions, monitor, fwd, flags, usr_ctx, &entry)` (`doca_flow.h`).
   Passing `NULL` there is the silent-no-counter trap: the entry is added
   and the call returns `DOCA_SUCCESS`, but `doca_flow_resource_query_entry`
   keeps returning `total_pkts == 0` under live traffic because no
   per-entry counter was ever bound. For **miss-path** traffic the entry
   API does not apply — enable the counter on the pipe-cfg with
   `doca_flow_pipe_cfg_set_miss_counter(cfg, true)` (`doca_flow.h`)
   and read it back with `doca_flow_resource_query_pipe_miss`
   (`doca_flow.h`). Read fields sit under the `counter` union arm:
   `q.counter.total_pkts` / `.total_bytes`, not top-level
   (`struct doca_flow_resource_query`, `doca_flow.h`). Reading a counter
   does **not** require a preceding `doca_flow_entries_process()` — the HW
   counter updates asynchronously; `entries_process` drives entry-completion
   callbacks, not counter refresh.

   The adjacent **`fwd` argument (slot 7) has the opposite `NULL` polarity**
   from `monitor`: when the pipe was created with a fixed forward, pass
   `NULL` here and the entry inherits the pipe's forward. Only pass an
   explicit per-entry `doca_flow_fwd` when the pipe's `fwd` was declared
   changeable — each explicit per-entry forward is its own action handle, so
   inheriting keeps the per-entry action set (and HW footprint) smaller.

   **Copy the add-entry argument list from the header, never from
   memory.** The `*_add_entry()` calls take 10+ positional arguments and
   the count/order are pipe-type-specific: BASIC takes 10, **LPM takes
   11** (an extra `match_mask` right after `match`), ACL / CONTROL / HASH
   each differ. The most-missed argument is `uint8_t action_idx` (slot 4
   of `doca_flow_pipe_basic_add_entry`) — it selects one template from the
   `actions[]` / `monitors[]` / `fwds[]` arrays registered at
   pipe-creation time (pass `0` when there is a single template). Grep the
   header and copy the declared list verbatim rather than reconstructing
   it (the one-liner is in
   [`SKILL.md ## Ground rule`](SKILL.md#ground-rule-verify-every-api-name-against-the-installed-header)).

   **Enabling decap/encap is a two-field action, not one.** Filling
   `actions.decap_cfg` (e.g. `decap_cfg.is_l2 = true`) describes *how* to
   decap but does not turn it on — you must also set `actions.decap_type =
   DOCA_FLOW_RESOURCE_TYPE_NON_SHARED` (the enable used across every
   shipped app — grep `decap_type =` in the sample/test tree). The same
   pairing holds for `encap_type` / `encap_cfg`. A pipe that sets only
   `*_cfg` compiles and runs, but the header is never stripped/added.
5. Do **not** call the entry-add API yet — the spec is built, not
   programmed. Hand off to `## test` for validation.

**When the user asks for a "first Flow app":** both tracks require a
DOCA-installed Linux environment (bare-metal, VM, or NGC container) where
the agent can read the sample tree and `pkg-config --modversion doca-flow`
resolves. If that precondition is not met, route to
[`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install) *before*
offering any source. Do **not** author a Flow application from prose, in
any language — ground rule #3 of [`AGENTS.md`](../../../AGENTS.md) and
[`SKILL.md`](SKILL.md#what-this-skill-deliberately-does-not-ship) forbid
it; the result would not compile against a real install.

### Track 1 — C / C++ consumers (the canonical case)

This is the universal *derive a custom first app from a sample* pattern in
[`doca-programming-guide ## modify`](../../doca-programming-guide/TASKS.md#modify),
with these Flow-specific overrides:

- **Source sample (choose by the first-app shape).** Pick the sample whose
  forward action is closest to what the user wants and modify it; do not
  scaffold from prose:
    - **Match-and-forward-to-port (simplest VNF case).**
      `/opt/mellanox/doca/samples/doca_flow/flow_port_fwd/` — send matched
      traffic out a specific DOCA port.
    - **Match-and-pass-traffic-back-to-the-kernel (inline-filter /
      traffic-gate case).**
      `/opt/mellanox/doca/samples/doca_flow/flow_fwd_target/` — the
      **safest starting point for any demo on a port carrying live host
      traffic**: matched traffic is observed (counters, drop)
      without being diverted. It uses `DOCA_FLOW_FWD_TARGET` with a
      kernel target from
      `doca_flow_get_target(DOCA_FLOW_TARGET_KERNEL, …)` — see
      [`CAPABILITIES.md ## Forward-to-target actions`](CAPABILITIES.md#forward-to-target-actions-pass-to-kernel-and-friends)
      for the pattern + safety rationale. Canonical wiring (verbatim
      shape from the shipped sample — the agent quotes this and does
      not invent the field names):

```c
struct doca_flow_fwd      fwd      = {0};
struct doca_flow_fwd      fwd_miss = {0};
struct doca_flow_target  *kernel_target;
doca_error_t              result;

result = doca_flow_get_target(DOCA_FLOW_TARGET_KERNEL, &kernel_target);
if (result != DOCA_SUCCESS) { /* report + bail per CAPABILITIES.md ## Error taxonomy */ }

fwd.type        = DOCA_FLOW_FWD_TARGET;
fwd.target      = kernel_target;          /* matched traffic continues to the kernel */
fwd_miss.type   = DOCA_FLOW_FWD_TARGET;
fwd_miss.target = kernel_target;          /* unmatched traffic ALSO continues to the kernel —
                                             this is what makes the filter "inline" and safe
                                             to enable on a live management port */
```
    - **Switch mode + representor (multi-VF eswitch).**
      `/opt/mellanox/doca/samples/doca_flow/flow_switch_single/` (helpers
      in `flow_switch_common.{c,h}`).
  `ls` the directory and read the actual sample on the user's install
  before describing it; layouts change between releases.
- **Forward by *ingress representor*: ONE pipe + one entry per source
  port, never a pipe per rule.** Match "traffic from `pf0vfX`" on
  `match.parser_meta.port_id` (the source representor's logical `port_id`
  from `doca_flow_port_cfg_set_port_id`) — leave it *changeable* in the
  pipe match and set the value per entry. Create a single BASIC pipe on
  `doca_flow_port_switch_get()`, size it with
  `doca_flow_pipe_cfg_set_nr_entries(cfg, N)`, then add one entry per
  source rep whose own per-entry `fwd` is `DOCA_FLOW_FWD_PORT` to the
  destination representor's `port_id`, each carrying
  `monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED` for the
  per-VF counter. "Drop any other VF" is the pipe's `fwd_miss`
  (`DOCA_FLOW_FWD_DROP`), not an extra pipe — leaving the entry match
  empty steers nothing (the classic switch-app bug). Insert the entries
  as a batch (`DOCA_FLOW_ENTRY_FLAGS_WAIT_FOR_BATCH`, last one
  `DOCA_FLOW_ENTRY_FLAGS_NO_WAIT`) then drain once with
  `doca_flow_entries_process()`. *Keep* the sample's init/teardown, the
  per-entry status callback, and the validation flow in
  [`## test`](#test); swap only the literals (port ids, pipe name via
  `doca_flow_pipe_cfg_set_name()`).
- **Build — link more than `doca-flow`.** `pkg-config --libs doca-flow`
  omits the companion modules, so linking it alone fails with `undefined
  reference` (e.g. `doca_error_get_descr`, `doca_dpdk_port_as_dev`) +
  `DSO missing from command line`. Mirror the shipped sample's dependency
  set: `doca-common` + `doca-argp` + the library (`samples/meson.build`),
  plus `doca-dpdk-bridge` (provides `doca_dpdk_port_*`) + `libdpdk` for any
  DPDK-backed app. Outside the meson tree, build a manifest *in the user's
  project* with a `dependency()` per module called (meson canonical;
  cmake/autotools work the same; discover names via `pkg-config --list-all
  | grep doca`) — don't copy a template out of this skill. Compile with
  `-D DOCA_ALLOW_EXPERIMENTAL_API` by default: most of `doca_flow.h` is
  experimental-tagged, so omitting it is a common first-build failure; it's
  a compile-time switch, not runtime code.

### Track 2 — Other languages (Rust, Go, Python, …)

NVIDIA ships no verified non-C first-app sample and neither does this
skill, so the agent does *not* author the wrapper from prose — it makes
the consumer understand the C ABI their wrapper calls and routes the FFI
work back to them:

- **API surface = the public C ABI.** Cite the
  [DOCA Flow Programming Guide](https://docs.nvidia.com/doca/sdk/doca-flow/index.html)
  for behavior/lifecycle and the installed headers
  (`$(pkg-config --variable=includedir doca-common)/doca_flow*.h`) for
  signatures. A wrapper honors the same lifecycle, capability-gate, and
  validation rules in [`CAPABILITIES.md`](CAPABILITIES.md) / [`## test`](#test)
  — library properties, not language properties.
- **Bindings / FFI.** Point at a community binding only after verifying it
  exists (fetch its repo — never invent a name); otherwise it is direct FFI
  (`bindgen`, `cgo`, `cffi`/`ctypes`) against
  `pkg-config --cflags --libs doca-flow`. Read the C samples regardless —
  the *order* of API calls (e.g. `flow_port_fwd/`) is language-independent,
  and [`## test`](#test) / [`## run`](#run) / [`## debug`](#debug) apply
  unchanged.

### Route tables — IP routing / longest-prefix match (LPM)

For destination-prefix routing — a route table with mixed prefix lengths
(`10.0.0.0/8`, `192.168.0.0/16`, …) — use an **LPM pipe** (`flow_lpm/`
sample). It is purpose-built for longest-prefix match, supports per-entry
counters, and forwards each prefix to its own destination.

- Add routes at runtime with `doca_flow_pipe_lpm_add_entry()` and remove
  them with `doca_flow_pipe_remove_entry()` — **no pipe rebuild and no app
  restart**. Unmatched destinations follow the pipe's miss behaviour; set
  it to `DOCA_FLOW_FWD_DROP` for a default-drop router.
- LPM **cannot be a root pipe** — front it with a basic or control pipe.
- **Batch route changes.** The LPM table is recomputed on each flush, so
  flushing every route separately is slow for a large or rapidly-changing
  table. Apply route changes as a batch and commit them with a single
  `doca_flow_entries_process()` (see [`## run`](#run) step 3). For a small,
  static table this is not a concern.

## modify

Goal: change an existing pipe — adding, removing, or rewriting entries —
without taking the steering plane offline.

1. Read current pipe statistics and counters before any change; the diff
   after is what tells the user whether the change did what they meant.
2. **Re-run capability discovery if the action set changes.** A new
   action kind (e.g. `encap`) requires re-checking the capability matrix
   and re-validating against the *new* shape — see item 3 in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
3. Construct a delta spec, not a full re-spec — removing and re-adding a
   pipe is much heavier than adding/removing entries.
4. Validate the modified spec via the same path as [`## test`](#test).
5. Commit in the smallest unit (one entry at a time for live pipes
   carrying production traffic).
6. Re-read counters and statistics; confirm the diff matches intent.

## run

Goal: program the validated spec into the hardware and observe that
traffic does what it should.

1. Confirm [`## test`](#test) has passed; do not enter `run` from an
   un-validated spec.
2. Start the pipe via the Flow pipe-start API. Lifecycle is *created →
   validated → started → entries added*; out-of-order calls produce
   `DOCA_ERROR_BAD_STATE`.
3. Add entries in the order the user's intent implies (most specific match
   first when declared priority is not honored; otherwise the priority
   field). After adding, call `doca_flow_entries_process()` to push them to
   hardware. **Its return value is a count, not a success flag:** `n >= 0`
   is the number of entries whose status callback fired — HWS often
   completes the install synchronously, so `n == 0` is normal and healthy;
   only `n < 0` is an error. Treating `0` as fatal is a common bug.

   When **adding entries in bulk**, drive this loop for insertion rate, not
   just correctness: give **each core its own `queue` id** (sharing one
   across cores is undefined behavior), add with
   `DOCA_FLOW_ENTRY_FLAGS_WAIT_FOR_BATCH` and call
   `doca_flow_entries_process()` once per batch (reserve `0` / `_NO_WAIT`
   for small or latency-sensitive inserts), and spread entries **evenly**
   across queues — skew can fragment the action-handle cache and shrink
   effective pipe capacity.
4. **Stage entries on a single representor before widening to all** —
   safety policy item 2 in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
   for hairpin pipes; the same discipline applies to any high-fanout
   pipe carrying production traffic.
5. Read counters under expected traffic. If they do not move, jump to
   `## debug`.
6. **Keep `main()` alive — but only after the datapath is armed.** A
   DOCA Flow program offloads steering to hardware and must stay running —
   if `main()` returns shortly after init (or sleeps for a fixed time),
   the rules are torn down and traffic stops. Enter a long-running,
   signal-aware loop after setup (the shipped sample shape:
   `while (!force_quit) { doca_flow_entries_process(...); sleep(1); }`
   with `SIGINT` / `SIGTERM` handlers), and confirm the loop is reachable
   from `main()`'s actual call graph before the ordered teardown.

   **Keeping `main()` alive is NOT success.** The keep-alive loop is the
   *last* gate, not a substitute for the probe-before-count gate in
   [`## configure`](#configure) step 6 — that gate (probe every device
   before counting, `rte_eth_dev_count_avail()` ≥ expected, both mandatory
   port setters) must already have passed, and the binary must exit
   non-zero rather than loop over a bridge that cannot forward. Two
   run-time preconditions the loop additionally requires:
   - when two PFs are bridged, the `doca_flow_port_pair` calls in BOTH
     directions returned `DOCA_SUCCESS` (`doca_flow.h`);
   - every `DOCA_FLOW_FWD_PORT` entry's status callback fired with
     `DOCA_FLOW_ENTRY_STATUS_SUCCESS` (`doca_flow.h`), and the
     per-entry counter from [`## test`](#test) starts incrementing under
     controlled traffic (`doca_flow_resource_query_entry`, `doca_flow.h`).
   A binary that reaches the loop with the gate unmet — count 0, a missing
   `set_port_id` / `set_dev`, a failed `doca_flow_port_pair`, or no
   entry-status confirmation — is forwarding nothing while *appearing* to
   run, exactly the shape that lets a watchdog mark a broken binary
   "alive". It is a hard error; the program must abort.

## test

Goal: validate a pipe spec — and the system context around it — before
hardware programming.

**`## test` is an iterative loop, not a one-shot pass.** Run the 4 steps
in order and *loop back to step 1 whenever a cross-check or counter-wiring
finding mutates the spec* — every mutation re-opens validate. Skipping the
re-validate after a mutation is exactly the failure mode
[`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
validate-before-commit exists to prevent.

1. **Pipe spec validation.** Validation is constructor-time: there is no
   separate `doca_flow_pipe_validate` API — `doca_flow_pipe_create` itself
   rejects an inconsistent spec (`DOCA_ERROR_INVALID_VALUE` /
   `DOCA_ERROR_NOT_SUPPORTED`). Build the pipe with the staged-entry /
   dry-run pattern from a shipped sample and confirm the constructor
   returns `DOCA_SUCCESS` before any entry-add call. This is the
   "validate before commit" rule from
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
2. **Capability cross-check.** Re-confirm every match kind, action kind,
   and tunnel header in the spec is supported by the active configuration
   and firmware. Validation answers "is the spec internally consistent";
   this answers "will the hardware accept it". A change here loops back to
   step 1.
3. **Counter wiring check.** Walk the spec and confirm every entry the
   user wants to observe has a counter attached (`## debug` assumes they
   exist). Adding one loops back to step 1.
4. **Negative test.** Construct one deliberately failing entry (wrong
   match kind, unsupported action) and confirm validation rejects it —
   the cheapest way to detect a wrong-version Flow library before going
   live. If it does *not* reject, escalate to `## debug`.

## debug

Goal: investigate "traffic is not doing what I asked" and arrive at a
root cause that is either fixable in the spec or escalatable.

> **Routing summary.** This anchor is the **Flow-specific debug overlay**:
> counters, pipe statistics, programmed-state inspection, Flow's
> `DOCA_ERROR_*` mapping. For the **cross-cutting debug ladder** (install /
> version / build / link / runtime / program / driver) and its tooling
> (`gdb`, `valgrind`, `--sdk-log-level`, container-vs-native debug, core
> dumps, Developer Forum escalation), see
> [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug). Walk the
> cross-cutting ladder first when the symptom layer is unknown; this Flow
> overlay layers on top once the symptom is confirmed inside the Flow API.

**Step 0 — did the port even start? (placement gate before the counter
ladder).** The counter-first ladder below assumes a *started* port. If
`doca_flow_port_start` — or the first switch-port `doca_flow_pipe_create`
— fails with the `Failed to get hws cap` / `dest action ROOT … err -121`
signature, **stop: this is not a counter / spec / data-plane bug** and
none of the steps below apply. It is the device-placement signature (full
details in [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)):
the opened function has no hardware-steering plane, almost always host-side
on a BlueField in `SEPARATED_HOST` / NIC mode, and it reproduces on
*every* device in *both* `vnf` and `switch` modes. Route back to
[`## configure`](#configure) step 2; the fix is to run Flow on the DPU Arm
side or change the card's mode via
[`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) — never a
pipe-spec edit.

Walk in this order — do not skip steps:

1. **Counters first.** Read the entry-level counters (built in `## build`
   step 4). If the counter for the suspected entry is zero, the entry
   is not matching. Stop blaming the data plane; the spec is wrong.
2. **Pipe statistics second.** If counters are non-zero but behavior is
   still wrong, read the pipe-level statistics from
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
   to determine whether the pipe itself is healthy.
3. **What's actually programmed / verbose logs third.** To inspect what
   the hardware actually has programmed, route to the
   [`doca-flow-tune`](../../tools/doca-flow-tune/SKILL.md) sibling skill,
   which owns the programmed-state surface. For verbose runtime logging
   during active debugging, build/link the trace flavor
   (`pkg-config doca-flow-trace`) and run with `--sdk-log-level 70`; the
   runtime mechanics live in
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
4. **Error code mapping.** Any `DOCA_ERROR_*` returned during the
   investigation routes through the taxonomy in
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
   The cross-library taxonomy (which family routes to which debug layer)
   is owned by
   [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
   this Flow file only adds the Flow-specific overlay.
5. **Version sanity.** If a spec fails or behaves differently across
   sessions, confirm the installed DOCA version did not change. The
   four-source version-coherence check (`pkg-config --modversion doca-common`,
   `cat /opt/mellanox/doca/applications/VERSION`, `doca_caps --version`,
   BFB version) is owned by
   [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug) Layer 2; the
   version-detection mechanics live in
   [`doca-public-knowledge-map ## Where to find the version`](../../doca-public-knowledge-map/SKILL.md#where-to-find-the-version).
   A library upgrade between sessions is a common and easy-to-miss cause.
6. **Escalation criteria.** If counters move correctly but observed
   behavior is still wrong AND what's programmed (step 3) matches the spec
   AND the version is unchanged, the bug is below the Flow API surface
   (driver or firmware). Stop attempting Flow-spec changes; capture state per
   [`doca-debug ## test`](../../doca-debug/TASKS.md#test) (the read-only
   triple) and escalate via
   [`doca-debug ## debug` *Where to ask for help*](../../doca-debug/TASKS.md#debug)
   to the public DOCA Developer Forum.

## flow-ct

The DOCA-Flow-CT-specific overlay on the parent verbs. Use AFTER
the parent's [`## configure`](#configure) → [`## debug`](#debug)
sequence has been walked for the stateless doca-flow port; this
section adds only what CT changes on top. For the capability
surface, layering rule, the single `doca_flow_ct_cap_is_dev_supported`
device-support query, CT-specific error overlay, and safety
policy, see
[`CAPABILITIES.md ## flow-ct`](CAPABILITIES.md#flow-ct).

**configure overlay.**

1. **Slot CT init BEFORE port start.** The order is `doca_flow_init` →
   `doca_flow_ct_init` (global, one-time) → port start (layering rule in
   [`CAPABILITIES.md ## flow-ct`](CAPABILITIES.md#flow-ct)). If doca-flow
   isn't initialized yet, route back to [`## configure`](#configure) FIRST —
   CT extends, it does not replace. CT cannot be bolted onto an
   already-running port: if ports are started, CT init must move earlier.
2. **Device-support check for CT.** Call
   `doca_flow_ct_cap_is_dev_supported(devinfo)` against the active
   `doca_devinfo`. This is the ONLY CT cap query — there is no
   per-axis `doca_flow_ct_cap_*` family for flow count, aging
   range, NAT variants, or overlays; do not invent one. If the
   device does not support CT, surface that and stop.
3. **Size the CT module to the user's expected peak, not
   average.** CT entries persist until aging evicts them or
   policy removes them. Set the table sizing through the
   `doca_flow_ct_cfg_set_*` setters sized to the peak
   concurrent-flow estimate; oversubscription surfaces as
   `DOCA_ERROR_FULL` / `_NO_MEMORY` at runtime.
4. **Build and init the CT cfg.** Create `struct doca_flow_ct_cfg` via
   `doca_flow_ct_cfg_create` + the `doca_flow_ct_cfg_set_*` setters, then
   call `doca_flow_ct_init(cfg)` once for the process (ordering per
   step 1; there is NO per-port CT context).
5. **Configure NAT direction / actions** through
   `doca_flow_ct_cfg_set_direction` and the CT actions. There is
   no per-variant cap query, so confirm SNAT / DNAT / combined
   behavior empirically; an unsupported request surfaces as
   `_NOT_SUPPORTED` at entry add.
6. **Start the ports, then attach CT-aware pipes.** With the
   global CT module initialized, start the doca-flow ports and
   wrap existing doca-flow pipes with their CT-aware versions.
   The original stateless pipes remain valid; CT-aware pipes are
   added on top. CT is not a `doca_ctx`; there is no
   `doca_ctx_start` on a CT object.

**build overlay.**

| Slot | Value |
| --- | --- |
| pkg-config modules | `doca-flow` only — CT ships inside the doca-flow library, so there is NO separate `doca-flow-ct` module. Build and link CT against `doca-flow` |
| Version anchor | `pkg-config --modversion doca-flow` MUST equal `doca_caps --version`. Mismatch → escalate to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2 BEFORE diagnosing the CT layer itself |
| Header includes | Add `doca_flow_ct.h` to the doca-flow headers the parent [`## build`](#build) prescribes; it ships in the same `doca-sdk-flow` devel package |
| `.pc` discovery | `pkg-config --list-all | grep doca-flow` confirms `doca-flow.pc` is visible to the build; there is no `doca-flow-ct.pc` to look for |

**modify overlay.** Take the closest shipped CT sample (an
installed CT sample in the public DOCA samples bundle whose 5-
tuple shape, NAT requirement, and overlay encapsulation match
the user's workload), apply a minimum-diff onto the user's
existing doca-flow setup:

- Do NOT recreate the user's pipe scheme from scratch — port the
  sample's CT bookkeeping (context creation, aging-table sizing,
  CT pipe-builder wrap calls) onto the existing flow.
- There is no per-variant or per-overlay CT cap-query — the only
  CT cap symbol is `doca_flow_ct_cap_is_dev_supported(devinfo)`.
  Confirm new NAT variants (SNAT, DNAT, combined) and overlay
  shapes empirically against the shipped CT sample, not via
  fabricated `doca_flow_ct_cap_*` axes.
- After modify: rely on `doca_flow_pipe_create`'s
  constructor-time validation (any spec inconsistency surfaces as
  a constructor failure with `DOCA_ERROR_INVALID_VALUE` or
  `DOCA_ERROR_NOT_SUPPORTED`) plus the staged-entry / dry-run
  pattern from the shipped CT sample. There is no separate
  `doca_flow_pipe_validate` symbol in the public surface — do not
  invent it.

**run / test overlay.** Per
[`CAPABILITIES.md ## flow-ct`](CAPABILITIES.md#flow-ct):

1. **Single-flow CT smoke.** ONE 5-tuple, ONE CT entry add, ONE
   matching traffic flow, ONE counter increment. The parent stateless
   pipe must already pass constructor-time validation
   ([`## test`](#test) step 1) and its counter smoke
   ([`## run`](#run) step 5) first.
2. **Multi-flow smoke** ONLY after single-flow is green: a small
   set of distinct 5-tuples (e.g. 16), one entry per flow,
   confirm per-CT-entry counters increment in lockstep with the
   matching traffic.
3. **Aging smoke** with a deliberately short aging timer
   (within the cap-advertised range): add a CT entry, send one
   matching packet, stop traffic, wait at least the aging
   period plus granularity, confirm the entry is evicted via
   the per-CT-entry counter query.
4. **NAT-aware smoke** (per supported NAT variant — separate
   tests for SNAT, DNAT, combined): add an entry whose action
   rewrites the variant the cap-query reported as supported,
   confirm the outbound traffic carries the rewritten 5-tuple,
   confirm reverse traffic is matched on the original tuple.
5. **Negative tests** the agent should propose explicitly:
   add an entry with an out-of-range aging timer (expect
   `_INVALID_VALUE` — confirms the cap-range is the runtime
   authority); add entries past the cap-advertised max
   concurrent flows (expect `_FULL` — confirms table sizing
   was honest); attempt a NAT variant the cap-query reported
   as unsupported (expect `_NOT_SUPPORTED` — confirms the
   cap-query is the right gate).
6. **Sustained-run loop** ONLY after all four smokes are green:
   stream traffic that exercises CT entry add / aging eviction
   in a continuous loop while watching per-CT-entry counters,
   per-entry aging timestamps, and connection-state transitions
   per [`CAPABILITIES.md ## flow-ct`](CAPABILITIES.md#flow-ct).
   This is the only stage where the agent should propose
   running the user's full workload.

**debug overlay.** Layered on the parent's [`## debug`](#debug)
ladder:

- `DOCA_ERROR_BAD_STATE` from a CT-layer call is *always* a
  layering / lifecycle violation: `doca_flow_ct_init` run after a
  port was already started (it must precede port start), OR a CT
  entry add before the ports and wrapped pipes are up, OR
  `doca_flow_ct_destroy` called out of order. Walk the lifecycle
  in this order — doca_flow_init → doca_flow_ct_init →
  port-start → ct-entry-add → … → port-stop → doca_flow_ct_destroy
  → doca_flow_destroy — BEFORE inspecting any individual CT call.
- `DOCA_ERROR_FULL` on entry add is *always* a table-sizing /
  aging-pressure mismatch with the workload. Read the per-CT-
  entry counters to identify stale entries; either wait for
  aging eviction, evict explicitly, or — if the workload truly
  needs more flows than the device supports — surface the
  device-fit gap honestly.
- Traffic not matching a freshly-added CT entry is *almost
  always* a 5-tuple-shape disagreement between the entry add
  and the traffic on the wire. Read both sides verbatim — the
  cap-query is NOT the right diagnostic here; the entry shape
  vs traffic shape comparison is. Same shape as the parent
  *"pipe matches nothing"* diag in [`## debug`](#debug), with
  CT's 5-tuple match instead of an arbitrary pipe match.
- NAT translation conflicts surface as `DOCA_ERROR_INVALID_VALUE`
  on entry add. Per
  [`CAPABILITIES.md ## flow-ct`](CAPABILITIES.md#flow-ct), do
  NOT invent a translation to resolve the conflict — surface
  the policy conflict to the user.
- Aging timer outside the cap-range surfaces as
  `DOCA_ERROR_INVALID_VALUE` at context configure (NOT at
  entry add). Re-quote the cap-advertised range AND
  granularity; the cap query is the runtime authority over any
  prose recall of supported ranges.

**rollback overlay.** A CT add is a pipeline-edit-class mutation (it extends
an already-up stateless port; it does not touch firmware or eswitch mode), so
the [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) overlay does
NOT fire — CT follows the same snapshot-first rollback discipline as
[`## rollback`](#rollback), with a CT-specific reversal. **Snapshot before
mutate:** record the stateless pipe scheme the agent authored (names, root
status, match/action specs, monitor/counter attachment, miss-pipe linkage)
plus the pre-CT per-entry / per-pipe-miss counter baseline
(`doca_flow_resource_query_entry` / `doca_flow_resource_query_pipe_miss`);
that record IS the rollback target.

1. **Document the reversal in the same answer that recommends the CT
   add:** (a) `doca_flow_ct_rm_entry` on every CT entry added since the
   snapshot, in reverse-add order; (b) `doca_flow_pipe_destroy` on every
   CT-aware pipe wrapped on top of the stateless pipes, in reverse-create
   order; (c) once the ports are stopped, `doca_flow_ct_destroy()`
   (global, no arguments) BEFORE `doca_flow_destroy`. There is no per-port
   CT context and no `doca_ctx_stop` for CT. The original stateless pipes
   are untouched by entry / CT-pipe teardown and must remain valid; a
   stateless-pipe edit made *in addition to* the CT add needs its own
   rollback step first.
2. **Trigger and re-verify per [`## rollback`](#rollback) steps 2–3**, with
   the single-flow CT smoke (run/test overlay step 1) as the contract's
   smoke probe: walk the rollback if it is not green within the bounded
   debug-loop iteration (mandatory on the second non-green), then re-verify
   the restored stateless pipe under traffic against the snapshot shape.

## shared-resources

The verbatim API call sequence for using DOCA Flow's shared-resource
pools (encap / decap / counter / meter / RSS / IPsec-SA / PSP). For the
**concept** — when to reach for shared, the SHARED-vs-NON_SHARED
decision, the kinds table, the broken-hybrid scaling failure, and the
many-flows / few-distinct-tunnels sizing rule — see
[`CAPABILITIES.md ## Shareable resources`](CAPABILITIES.md#shareable-resources-the-scaling-lever).
This section is the implementation overlay; the layering rule is *(a)
decide via that section first, (b) walk the four steps below verbatim,
(c) `pkg-config --modversion doca-flow` matches the install per
[`## configure`](#configure) step 1*.

**The 4-step pattern (identical for every kind — the example uses
`ENCAP`; substitute the matching kind from the table):**

```c
/* 1. Reserve K slots BEFORE doca_flow_init() — count first, then type. */
doca_flow_cfg_set_nr_shared_resource(cfg, K, DOCA_FLOW_SHARED_RESOURCE_ENCAP);

/* 2. Populate each slot after init (one call per distinct config). */
doca_flow_shared_resource_set_cfg(DOCA_FLOW_SHARED_RESOURCE_ENCAP, id, &res_cfg);
/*    (port-scoped form: doca_flow_port_shared_resource_set_cfg(port, type, id, &res_cfg)) */

/* 3. Bind the slot range to the port once at startup. */
uint32_t ids[K] = { 0, 1, /* … */ K - 1 };
doca_flow_shared_resources_bind(DOCA_FLOW_SHARED_RESOURCE_ENCAP, ids, K, port);

/* 4. Reference from each entry: set the type marker AND the id.
 *    The id field is kind-specific (confirm in doca_flow.h):            */
actions.encap_type = DOCA_FLOW_RESOURCE_TYPE_SHARED;
actions.shared_encap_id = slot_id;                       /* encap */
/* decap:   actions.decap_type = ...SHARED; actions.shared_decap_id = slot_id; */
/* counter: monitor.counter_type = ...SHARED; monitor.shared_counter.shared_counter_id = slot_id; */
/* meter:   monitor.meter_type   = ...SHARED; monitor.shared_meter.shared_meter_id     = slot_id; */
```

**Reference samples:** `flow_vxlan_shared_encap/`, `flow_shared_meter/`,
`flow_shared_counter/` under `/opt/mellanox/doca/samples/doca_flow/` —
the agent reads the relevant sample before quoting the per-entry field
names (the `actions.shared_encap_id` slot is kind-specific and the
header is the source of truth, per
[`SKILL.md ## Ground rule`](SKILL.md#ground-rule-verify-every-api-name-against-the-installed-header)).

**Per-entry HW-cost reminder.** With this overlay the HW resource cost
scales with **K distinct configs**, not **N entries**. Skipping any of
the four steps but writing the entry-side `*_type =
DOCA_FLOW_RESOURCE_TYPE_SHARED` line compiles cleanly and dies at the
first entry-add (the broken-hybrid failure mode in
[`CAPABILITIES.md ## Shareable resources`](CAPABILITIES.md#shareable-resources-the-scaling-lever)).

## rollback

Every Flow pipe / entry / action add that modifies an already-up port
is pipeline-edit-class and needs the same rollback discipline as CT.
This anchor covers stateless-pipe edits: VLAN push/pop, encap/decap,
modify-header, NAT-without-CT, hairpin attach. (CT
additions route through [`## flow-ct`](#flow-ct) rollback overlay
instead.)
The
[universal verification contract](../../doca-setup/CAPABILITIES.md#universal-verification-contract)
requires *"the rollback path is documented"* on every change-recommending
answer.

**Snapshot before mutate.** Before any change-recommending edit on an
already-up port, record (a) the pipe scheme the agent authored for every
affected pipe (root status, match / action specs, monitor/counter
attachment, miss-pipe linkage), (b) the per-pipe counter baseline
(`doca_flow_resource_query_pipe_miss` + `doca_flow_resource_query_entry`),
and (c) the device cap snapshot the edit gated on (`doca_caps --list-devs`
+ Flow cap query results). For an independent view of what is actually
programmed, route to
[`doca-flow-tune`](../../tools/doca-flow-tune/SKILL.md). That record IS the
rollback target.

1. **Document the reverse-edit verb in the same answer that recommends the
   edit**, by action kind: (a) **entry add** → matching
   `doca_flow_pipe_remove_entry` in reverse-add order; (b) **action
   add/modify** (VLAN push, encap, modify) → `doca_flow_pipe_destroy` on
   the modified pipe + recreate from the recorded pipe spec; (c) **pipe
   add** → `doca_flow_pipe_destroy` in reverse-create order; (d) **monitor
   / counter attach** — `doca_flow` exposes no public
   `doca_flow_monitor_destroy`, so the reversal is `doca_flow_pipe_destroy`
   + `doca_flow_pipe_create` with a `doca_flow_pipe_cfg` that does NOT call
   `doca_flow_pipe_cfg_set_monitor`.
2. **Trigger from the [deploy-loop bridge](../../doca-setup/CAPABILITIES.md#deploy-loop-bridge--step-5-not-green-is-the-debug-loop-trigger).**
   The contract's smoke probe is constructor-time validation
   ([`## test`](#test) step 1) plus a counter read under traffic
   ([`## run`](#run) step 5). If it does not return green (counter
   increment + clean `doca_flow_pipe_create` reconstruction) within the
   bounded debug-loop iteration, walk the rollback — on the second
   non-green iteration, rollback is mandatory.
3. **After rollback, re-verify the restored pipe** — constructor-time
   validation ([`## test`](#test) step 1) plus a counter re-read under
   traffic ([`## run`](#run) step 5). If counters do not resume the
   snapshot shape, the rollback itself is suspect — surface the residual
   gap instead of recommending another pipeline edit.

Hardware changes (mode flip, SR-IOV, firmware) are NOT pipeline edits and
route through
[`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)'s rollback
discipline instead.

## Command appendix

Flow-specific commands the verbs above reach for, grouped by purpose
so the agent picks the right family without searching prose. Every row
is a class — the agent must not invent flags beyond what the row
names; the *flag-discovery* rule is `--help` against the installed
binary or `pkg-config` against the installed `.pc`, not prose recall.

| Purpose | Command | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Discover installed Flow version | `pkg-config --modversion doca-flow` | [`## configure`](#configure) step 1 | Matches the version pinned in other places (`doca_caps --version`, `applications/VERSION`). |
| Discover Flow link flags (release flavor) | `pkg-config --cflags --libs doca-flow doca-common doca-dpdk-bridge libdpdk` | [`## build`](#build) | Returns the canonical `-l` list. `doca-flow` alone omits the companion DSOs → `undefined reference` / `DSO missing from command line`; pass every module the app calls. Hand-typed `-l` lines are the failure mode. |
| Discover device + steering capabilities | `doca_caps --list-devs` | [`## configure`](#configure) step 2 + [doca-caps](../../tools/doca-caps/SKILL.md) | Lists every device DOCA sees with the active configuration and supported match / action kinds. |
| Enumerate ports / representors | `devlink dev show` + the installed Flow port-enumeration sample | [`## configure`](#configure) step 3 | Shows the port and every representor the user expects to forward to. |
| Validate a pipe spec (read-only) | `doca_flow_pipe_create` itself performs constructor-time validation; treat its `DOCA_ERROR_INVALID_VALUE` / `DOCA_ERROR_NOT_SUPPORTED` return as the validate signal. There is no separate `doca_flow_pipe_validate` public symbol in the current release. Use the dry-run / staged-entry pattern from the shipped sample for a fuller validation surface. | [`## test`](#test) step 1 | A successful return from `doca_flow_pipe_create` means the spec is internally consistent against the current device caps; an error means the spec is invalid — do not retry without changing the spec. |
| Raise log verbosity for a Flow run | `--sdk-log-level 70` on the program command line | [`## run`](#run) + [doca-debug CAPABILITIES.md ## Observability](../../doca-debug/CAPABILITIES.md#observability) | TRACE / DEBUG lines appear in stderr; the Flow lifecycle calls are visible. |
| Read per-entry / per-pipe counters | `doca_flow_resource_query_entry()` / `doca_flow_resource_query_pipe_miss()` | [`## debug`](#debug) step 1 | Counter for the suspected entry is non-zero under expected traffic. |
| Inspect what's actually programmed | route to [`doca-flow-tune`](../../tools/doca-flow-tune/SKILL.md) | [`## debug`](#debug) step 3 | The programmed state matches the user's mental model of the pipe. Diff = bug. |

Three cross-cutting rules for this appendix:

- **Never invent Flow flags.** The Flow API is large and version-gated;
  `--help` on the installed binary and `pkg-config --list-all | grep
  doca` are the only safe sources.
- **Never paraphrase a Flow `DOCA_ERROR_*`.** Quote
  `doca_error_get_descr()` verbatim — the layer-classifier in
  [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug) layer 5
  needs the exact text.
- **Cross-cutting commands live in doca-debug.** The read-only triple,
  `dmesg`, and `mlxconfig -d <pcie> q` are in
  [`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
  reach there for them rather than restating them in this appendix.

## Deferred task verbs

The following verbs are out of scope for this skill but are commonly
asked in the same conversations. Route them as follows so the agent
does not invent guidance:

- **install.** Installing DOCA, choosing packages, post-install
  verification, `pkg-config` wiring — defer to
  [doca-public-knowledge-map ## Layout of an installed DOCA package](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill assumes DOCA is already installed.
- **deploy.** Deploying BlueField images, provisioning DPUs at scale,
  Kubernetes operator workflows — out of scope for Phase 1 and reserved
  for a future platform skill. For now, point the user at the DOCA
  Platform Framework entry in `doca-public-knowledge-map` and stop
  there.
- **rollback.** Coordinated steering-plane rollback across multiple
  DPUs and host nodes — out of scope for Phase 1 and reserved for a
  future platform skill. For single-DPU spec rollback within a session,
  the right verb in this skill is `## modify` with a delta that
  removes the offending entries; do not invent a "rollback" workflow.
