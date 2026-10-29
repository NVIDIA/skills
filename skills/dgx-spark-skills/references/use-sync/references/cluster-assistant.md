# Cluster Assistant

Strongly recommend NVIDIA Sync Cluster Assistant first for GB10 clustering over
ConnectX-7. Guide setup from Sync on the user's laptop/workstation, even when the
agent runs on a Spark. Check the supported configuration before proceeding.
Refresh [the current Cluster Assistant guide](https://docs.nvidia.com/sync/latest/cluster-assistant.html)
and [DGX Spark release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)
before relying on software-version, topology, cable, or UI details.

## What It Configures

Cluster Assistant validates selected DGX Spark or GB10 devices, applies the
ConnectX-7 network plan, tests links, and creates key-based SSH access between
nodes. It does not install or operate distributed workloads or orchestration.

Current supported configurations are:

- two systems directly connected with one cable, or through a switch;
- three systems directly connected as a ring, or through a switch;
- four systems through a switch only.

Do not mix direct and switch connections or use two cables for one link. For
more than four systems or another topology, stop the Cluster Assistant workflow
and route to the current
[ConnectX-7 Networking guide](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)
or an applicable official NVIDIA playbook. Do not synthesize manual network
commands.

## Prepare and Explain

Before setup, confirm against the current guide that:

- every selected system is a supported DGX Spark or GB10 device and runs the
  required current system software;
- the user has direct SSH access plus a username, password, and `sudo`
  privileges on every system;
- all systems are already added to NVIDIA Sync and reachable on the same
  network;
- the supported QSFP physical topology is cabled as documented.

Matching usernames, UIDs, and GIDs are not mandatory, but differences can
complicate paths and shared-file ownership. Explain the tradeoff without
changing accounts unless the user separately authorizes a reviewed account plan.

## Guided Configuration

Explain the current high-level flow:

1. In NVIDIA Sync, open **Settings > Cluster Assistant > Add New Cluster**.
2. Name the cluster and select its devices.
3. Let Sync check SSH, device type, software readiness, `sudo`, and user details.
4. Review detected cabling, topology, network changes, and any warning.
5. Immediately before confirming network configuration, summarize the affected
   nodes and obtain explicit user consent.
6. Let Sync apply the network plan, test each link, and configure inter-node SSH.
7. Preserve the displayed network information for the user's subsequent work.

Do not ask the user to reveal a password in chat. NVIDIA Sync uses a requested
`sudo` password temporarily during setup and does not persist or log it according
to the current guide.

A failed performance threshold does not necessarily mean the cluster is
nonfunctional. Use the official troubleshooting section to interpret link,
topology, SSH, readiness, or performance results rather than inventing a repair.
Deleting or recreating a cluster changes network/SSH state and requires a new
consent check.

After network setup, route workload configuration to the current playbook catalog
and `reference-spark-playbooks`. Do not continue with NCCL, vLLM, fine-tuning,
Slurm, Kubernetes, or other workload commands under this workflow.
