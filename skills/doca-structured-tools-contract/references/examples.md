# Structured-tools routing examples

These are question shapes this skill routes. The shape is
load-bearing; each worked example is one instance.

## Environment inventory

**Question:** "Is there a single command that tells me everything
about my DOCA install?"

**Example:** "Version, devices, installed libraries, sample paths,
drivers, and hugepages — all at once."

Use the `doca-env --json` schema and its ordered manual fallback chain
in [`SKILL.md ## Schemas`](../SKILL.md#schemas).

## Capability version

**Question:** "How do I look up which DOCA version a capability
requires without fetching a doc page?"

**Example:** "Is symmetric RSS hash mode in DOCA 2.6.0?"

Use the `version-matrix.json` schema. Its manual fallback fetches the
matching per-library page through
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Hardware topology

**Question:** "What does my hardware actually look like to DOCA —
every PCIe function and representor?"

**Example:** "Show every PF, VF, and SF visible on this BlueField,
including PCIe address and link state."

Use the `collect-host-state` / `collect-dpu-state` schema. The manual
fallback is `devlink dev show`, `lspci | grep Mellanox`, and
`ip -j link`.

## Validate before commit

**Question:** "How do I know my Flow, RDMA, or DMS specification is
valid before I program hardware?"

**Example:** "Will this Flow pipe pass validation before commit?"

Use the `validate-before-commit` schema. The fallback uses the
library's documented validation surface. For DOCA Flow,
`doca_flow_pipe_create` is not a read-only probe and the public header
does not provide `doca_flow_pipe_validate`; never invent one.

## Different behavior across hosts

**Question:** "My agent behaves differently on two hosts — is one
missing the helpers?"

**Example:** "Host A produced one line while host B walked five manual
commands."

Use the report-which-path rule in
[`SKILL.md ## The agent behavior contract`](../SKILL.md#the-agent-behavior-contract).
If the answer did not identify the selected structured or fallback
path, ask it to do so.

Questions about writing code, debugging crashes, or setting up an
environment route through the matching skill in
[`AGENTS.md`](../../../AGENTS.md).
