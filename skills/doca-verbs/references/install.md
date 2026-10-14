## install

DOCA Verbs is part of the standard DOCA host install — installing
`doca-verbs` in isolation is not a thing. The install verb routes
to [`doca-setup`](../../../doca-setup/SKILL.md) for the env / install
chain, and to
[`doca-public-knowledge-map ## Layout of an installed DOCA package`](../../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package)
for the on-disk layout the rest of this file assumes.

The agent's checks before declaring the install ready for any
raw-verbs work:

1. **`pkg-config --modversion doca-verbs` returns a semver
   string** — if the query fails, the verbs library is not
   installed on this host. Route to
   [`doca-setup ## debug`](../../../doca-setup/TASKS.md#debug) layer 1
   (install).
2. **`pkg-config --modversion doca-common` agrees** with the
   `doca-verbs` version per the four-way match rule in
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../../doca-version/CAPABILITIES.md#version-compatibility).
   A verbs-only upgrade against an older `doca-common` install is
   a partial-install hazard.
3. **The verbs headers are resolvable** under
   $(pkg-config --variable=includedir doca-common) (`doca_verbs.h`
   plus the adjacent `doca_verbs_*.h` family) per the
   headers-win-over-docs rule in
   [`doca-version`](../../../doca-version/SKILL.md).
4. **The selected workload has a usable device.** For RDMA-class
   QP workloads, `ibv_devinfo` (with sudo) returns at least one
   device row with `state: PORT_ACTIVE`, the same precondition as
   [`doca-rdma`](../../doca-rdma/SKILL.md). For Ethernet-SQ/RQ
   workloads, `doca_caps --list-devs` returns the selected
   Ethernet-capable `doca_dev`; its exact queue feature is then
   gated by the device-level query in `## configure` step 5.

If any of the four checks fails, **stop** — this skill's workflows
assume the install is healthy, and a partial install is a
[`doca-setup`](../../../doca-setup/SKILL.md) concern.

