# NVSHMEM Issue or Bug Report

Use this template only after troubleshooting has collected the available evidence. It mirrors the `NVSHMEM issue or bug` GitHub issue form. Start the title with `[Issue]:`; the issue form applies the `triage` label automatically.

Preserve exact error text and commands. Replace every placeholder; use `Not provided` rather than guessing. Under **How is this issue impacting you?**, select exactly one value from the issue form.

````markdown
# [Issue]: <concise symptom and affected operation>

## How is this issue impacting you?

<Select exactly one: Lower performance than expected | Application crash | Data corruption | Application hang>

## Share Your Debug Logs

- Complete stdout/stderr and logs: <attachments, links, or Not provided>
- `NVSHMEM_DEBUG=INFO`: <confirmed/not confirmed>
- `NVSHMEM_DEBUG_SUBSYS=ALL`: <confirmed/not confirmed>
- One log per rank: <file pattern, attachments, links, or Not provided>

### First causal log lines

```text
<First error or divergent log lines, including preceding context and PE/rank identity.>
```

### Required debug run when INFO-level logs are not attached

Reproduce the failure before submitting. Keep the executable, launcher, rank and node counts, binding, transport, container, and workload arguments unchanged. Configure the launcher to propagate these settings to every rank and capture one log per rank.

```bash
export NVSHMEM_DEBUG=INFO
export NVSHMEM_DEBUG_SUBSYS=ALL
export NVSHMEM_DEBUG_FILE=nvshmem-%h-%p.log

<original launcher command>
```

Attach the exact repeated command, complete stdout/stderr from every rank, and every generated `nvshmem-<host>-<pid>.log` file. Record whether INFO logging reproduces the failure, changes it, or succeeds. Until these artifacts are attached, label the draft **Incomplete — INFO-level NVSHMEM logs required before support submission**.

- [ ] Logs and attachments were checked for credentials, tokens, internal hostnames/IPs, and user-identifying data.

## Steps to Reproduce the Issue

### Minimal steps

1. <Setup step>
2. <Run step>
3. <Expected failure signal>

### Exact command or batch stanza

```bash
<Module loads, exported variables, container command, launcher, launcher options, executable, and arguments.>
```

- Minimal reproducer source or attachment: <link/path/Not provided>
- Relevant software versions and settings: <CUDA, driver, NCCL, MPI/PMIx/PMI, launcher, scheduler, OS/kernel, container, and NVSHMEM_* settings>
- Intermittency: <always | intermittent, N failures in M runs | unknown>
- Previous success: <last known-good NVSHMEM version or configuration/unknown>
- Smallest failing scale: <ranks, nodes, GPUs/Not tested>
- Smallest passing scale: <ranks, nodes, GPUs/Not tested>

## NVSHMEM Version

<Complete version from the debug logs, including the CUDA suffix; for example, 3.5.6+cuda12.8>

## Your platform details

- GPU architecture, model, and count per node: <details>
- GPU topology: <summary or `nvidia-smi topo -m` attachment>
- Network/NIC: <architecture, link details, and `ibstatus` attachment when applicable>
- Environment: <bare metal | container | cloud>
- Nodes, ranks, and rank-to-GPU mapping: <details>
- Scale-specific behavior: <specific rank/node counts that pass or fail>
- NVSHMEM bootstrap and transports: <values>
- IMEX/fabric-manager context, if applicable: <details/not applicable>

## Error Message & Behavior

### First error

```text
<Initial error message exactly as it appears in the logs, with preceding context and PE/rank identity.>
```

### Expected behavior

<What should happen.>

### Actual behavior

<What happens instead, including where the operation fails or hangs.>

### Additional context

<Relevant troubleshooting results, documentation or knowledge-base matches, suspected NVSHMEM involvement, workarounds attempted, or Not provided.>
````

Before returning the report, list every placeholder still marked `Not provided` and explain how the user can collect it safely. The issue form requires an impact choice and the NVSHMEM version. The troubleshooting workflow additionally requires the exact command, complete output from every PE/rank, and INFO-level per-rank logs for a support-ready submission. Do not block a clearly labeled draft when the user wants to preserve an incomplete investigation.
