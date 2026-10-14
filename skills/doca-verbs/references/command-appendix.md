## Command appendix

Every command below is **cross-cutting on DOCA Verbs** — it answers
a recurring class of question that comes up in the verbs above. The
agent should treat the *class* as load-bearing; the worked example
is a single instance. Run-as user is the unprivileged user unless
noted. Rows that need elevated privileges call that out explicitly.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST (`doca-env
   --json` for version + devices + libraries + drivers in one
   shot; `doca-capability-snapshot` for per-device capability
   flags; `version-matrix.json` for *"available since"* lookups).
2. If the probe succeeds, the structured tool's output is the
   authoritative answer and the agent SHOULD NOT also run the
   manual command in the row below. Report *"using structured
   `<tool>`"*.
3. If the probe fails, fall back to the manual command in the row.
   Report *"falling back to manual chain"*.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics (four-way match, NGC,
   headers-win) are owned by
   [`doca-version`](../../../doca-version/SKILL.md).

| Command (worked example) | Owning step | Class of question it answers | What healthy output looks like |
| --- | --- | --- | --- |
| `pkg-config --modversion doca-verbs` | [`## configure`](../TASKS.md#configure) step 3; [`## build`](../TASKS.md#build) | What is the build-time raw-verbs version? | A semver string matching `pkg-config --modversion doca-common` AND `doca_caps --version`. Disagreement = verbs-only partial install (route to [`doca-version TASKS.md ## debug`](../../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --cflags --libs doca-verbs` | [`## build`](../TASKS.md#build) | What include + link flags does the linker need? | Includes resolve under $(pkg-config --variable=includedir doca-common); libs include the verbs runtime plus `-ldoca_common`; should NOT auto-pull `-ldoca-rdma` or `-ldoca-eth` |
| `ls /opt/mellanox/doca/samples/` | [`## modify`](../TASKS.md#modify) slot 1 | Which DOCA samples ship in this install — and which one is the closest starting point for the user's QP shape / completion-handling choice? | A list of sample subdirectories; the user picks the closest in QP shape + completion path |
| `ls $(pkg-config --variable=includedir doca-common)  \| grep doca_verbs` | [`## configure`](../TASKS.md#configure) step 2 | Which `doca_verbs_*.h` headers are installed (the headers-win-over-docs anchor)? | A list including `doca_verbs.h` and any adjacent `doca_verbs_*.h` files the release ships |
| `doca_caps --list-devs` | [`## configure`](../TASKS.md#configure) step 4 | Which devices on this host can be used as a `doca_dev`? | One row per visible device with PCIe address and capability flags; raw verbs needs at least one row, same as the higher-level RDMA / Eth / Rmax libraries |
| `doca_caps --version` | [`## configure`](../TASKS.md#configure) step 3; [`## test`](../TASKS.md#test) step 2 | What is the *runtime* DOCA version on this host? | A semver string matching `pkg-config --modversion doca-verbs` |
| `cat /opt/mellanox/doca/applications/VERSION` | [`## configure`](../TASKS.md#configure) step 3; [`## debug`](../TASKS.md#debug) layer 1 | What does the install tree itself claim its version is? | A semver string matching the other two version sources |
| `dmesg \| tail -n 40` (sudo) | [`## debug`](../TASKS.md#debug) layer 7 | What did the kernel / driver log around the last raw-verbs call? | Empty or recent benign messages. Repeated mlx5 / IB errors → driver-layer bug; route to [`doca-setup ## debug`](../../../doca-setup/TASKS.md#debug) |
| `ibv_devinfo` (sudo) | [`## configure`](../TASKS.md#configure) step 5; [`## debug`](../TASKS.md#debug) layer 7 | What does the underlying `libibverbs` see for this device? | One device row with `state: PORT_ACTIVE` and a sane MTU. **NOTE:** running `ibv_devinfo` is a diagnostic; it does NOT license the program to also call `ibv_*` against a `doca-verbs`-owned object (per the no-mixing rule in [CAPABILITIES.md ## Safety policy](../CAPABILITIES.md#safety-policy)) |
| `DOCA_LOG_LEVEL=trace ./<binary>` | [`## run`](../TASKS.md#run) step 3 | What did the structured DOCA logger emit for the first failing call? | A trace-level line on every lifecycle transition and every WR submission. Silence after submit = chosen completion-handling path not driven |
| `<binary> --sdk-log-level DEBUG 2> doca.log` | [`## run`](../TASKS.md#run) step 3 | Same as above via SDK-tier setter | A flood of DOCA library internal trace; the user's own DEBUG lines remain controlled by the app tier per the two-tier model in [`doca-common CAPABILITIES.md ## log`](../../doca-common/CAPABILITIES.md#log) |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL` / `DOCA_LOG_LEVEL_SDK`) the cross-library overlay
is in
[`doca-debug TASKS.md ## Command appendix`](../../../doca-debug/TASKS.md#command-appendix);
this table adds the raw-verbs-specific rows on top.

Three cross-cutting rules for this appendix:

- **Never invent verbs symbols or flags.** The verbs ABI is large,
  version-gated, and shipped under the experimental tier on this
  release; `ls
  $(pkg-config --variable=includedir doca-common)  | grep doca_verbs`
  on the user's install is the only safe source.
- **Never paraphrase a verbs `DOCA_ERROR_*`.** Quote
  `doca_error_get_descr()` verbatim — the layer-classifier in
  [`doca-debug ## debug`](../../../doca-debug/TASKS.md#debug) layer 5
  needs the exact text.
- **Cross-link instead of duplicate.** Cross-cutting commands (the
  read-only triple, `dmesg`, `mlxconfig -d <pcie> q`) live in
  [`doca-debug TASKS.md ## Command appendix`](../../../doca-debug/TASKS.md#command-appendix);
  this appendix names only the raw-verbs-specific ones.
