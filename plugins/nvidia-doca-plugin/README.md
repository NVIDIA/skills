# NVIDIA DOCA Skills plugin

Generated from the NVIDIA DOCA skills bundle. It contains 53 signed skills and nothing else: no MCP server, hooks, commands or scripts.

Install it from your agent's plugin marketplace (Claude Code, Codex, Cursor and other agents that read Open Agent Plugins). The agent loads the skills from `skills/`. Do not edit this artifact directly.

## Host paths the skills read

On a DOCA host the skills may ask the agent to check the install and hardware with read-only commands: `/opt/mellanox` (DOCA install), `/etc/nvidia` and `/etc/mlnx-release` (BlueField and driver release), `/sys/class/infiniband`, `/sys/class/net`, `/sys/bus/pci/devices`, `/dev/infiniband`, `/dev/mst` and `/proc/driver`. `evals/config.json` grants the sysfs paths that exist in the evaluation sandbox (`/sys/class/infiniband`, `/sys/class/net`, `/sys/bus/pci/devices`) as read-only.
