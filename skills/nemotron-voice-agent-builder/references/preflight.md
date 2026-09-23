# Preflight

Sections 1-3 are step 0 of `intake.md` and run before the proposal. They establish host
class and base credentials, and they stay silent unless they find something that changes
the conversation, which is a missing credential, a detection failure, or memory held by
another process. Section 4 runs during proposal, after pipeline, framework, and candidate
models are known.

Per-slot footprints and knobs live in `models/llm.md`, `models/asr.md`, and
`models/tts.md`. Section 4 combines them into self-hosted, cloud, or hybrid placement.

| Output | Feeds |
| --- | --- |
| Host class | which platform guide applies |
| Usable VRAM | later deployment fit |
| Compute capability | LLM precision planning (`models/llm.md`) |
| Deployment row | intake proposal |
| A blocking credential, if any | halts everything before the table |

## 1. Probe the Host

Collect the operating system, GPU inventory, machine identity, system RAM, and free disk.

Detect the operating system first, for example with `uname -sm`. Self-hosted NVIDIA model
containers require Linux, so on macOS or Windows the deployment is cloud. The Linux
identity paths below do not exist on other systems, and their absence there is expected.

Run the GPU probe as the ordered stages below. Do not collapse them into one command. A
single combined query is the reported cause of a host with a working GPU being classified
as cloud only, because one unsupported field makes the whole query return nothing.

### Stage 1. Locate `nvidia-smi`

```bash
command -v nvidia-smi
```

When that returns nothing, check the paths a driver install can use before concluding
anything: `/usr/bin/nvidia-smi`, `/usr/local/nvidia/bin/nvidia-smi`,
`/opt/nvidia/bin/nvidia-smi`, and `/usr/lib/wsl/lib/nvidia-smi` on WSL. Use the path that
exists for every later stage.

### Stage 2. Read the Inventory

```bash
nvidia-smi -L
```

`nvidia-smi -L` is the inventory step because it takes no field arguments, so no driver
version can reject part of it. One line per GPU proves a usable device exists. Treat a
failure here as a driver or permission problem to be reported, never as proof of no GPU.

### Stage 3. Read the Fields in Two Queries

Query the long-standing fields first, then compute capability on its own:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

`compute_cap` is a newer field than the rest. Asking for it in the same query as the
memory fields means a driver that does not know it discards the memory and name output
too. Keeping it separate loses only the field the driver cannot answer.

Treat `[N/A]`, `[Not Supported]`, and `Insufficient Permissions` as unread values rather
than as zero. A blank or `[N/A]` memory figure on an integrated device is normal, and
§Unified memory hosts explains what to read instead.

### Stage 4. Confirm Machine Identity

On Linux, read both identity sources, because they are the only reliable way to separate
DGX Spark and Jetson Thor from a generic host:

```bash
cat /sys/class/dmi/id/product_name 2>/dev/null
cat /proc/device-tree/model 2>/dev/null
```

Either file can be absent on a given machine, and that alone is not an error. Match the
strings exactly as §2 lists them.

### Stage 5. Look for a GPU Independently

Reach this stage only when Stages 1 to 3 produced no GPU. Before recording `no_gpu`,
confirm the absence against evidence that does not depend on `nvidia-smi`:

```bash
cat /proc/driver/nvidia/version 2>/dev/null
ls /dev/nvidia* /dev/nvgpu 2>/dev/null
lspci -nn -d 10de: 2>/dev/null
```

A loaded kernel module, a device node, or an NVIDIA PCI vendor id means the machine has a
GPU that the probe could not read. That is a detection failure to report, not a cloud-only
host. On Tegra platforms such as Jetson there is no discrete PCI entry and no
`/dev/nvidia0`, so the device-tree model from Stage 4 is the signal there.

### Unified Memory Hosts

DGX Spark and Jetson Thor share one memory pool between the CPU and the GPU, so
`memory.total` describes that pool rather than dedicated VRAM. Read host memory as well
and plan against the lower of the two figures:

```bash
awk '/MemTotal|MemAvailable/ {print}' /proc/meminfo
free -h
```

### Find Out What Is Holding the Memory

When free memory is well below total, identify what is resident before that number changes
the deployment. A stale container looks exactly like a small GPU, and the two deserve
opposite responses.

```bash
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv
docker ps --format '{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}'
```

Map a pid to a container with `docker inspect -f '{{.State.Pid}}'` per running id. On Tegra
platforms per-process memory is often unavailable, so attribute by container and say so.

| What is resident | Action |
| --- | --- |
| A slot this project owns that matches the lock | keep it (`platforms/deployment.md` §Per-slot reuse) |
| A slot this project owns that is stale, wrong, or unhealthy | offer to stop that named container, with the memory it returns |
| Anything else, including notebooks, training jobs, and desktop sessions | report and name it. Let the user decide |

Report the finding with numbers, for example that stopping a named stale container returns
about 40 GiB and makes the deployment fully local. Then wait.

Three rules:

1. **Stop only named containers the user approved.** Never `docker system prune`, never
   `docker kill $(docker ps -q)`, never kill a bare pid.
2. **Re-measure after reclaiming**, because memory returns when the process exits rather
   than when the command returns.
3. **Do not downgrade around reclaimable memory.** Confirm the shortfall is real before
   proposing a smaller model, a lighter quantization, a shorter context, or a cloud slot. If
   the user declines or the memory is someone else's, treat what remains as the budget and
   record in the proposal that the GPU was partially occupied.

### What to Report

Report usable VRAM as `memory.free`, not nameplate `memory.total`, and alongside whatever is
holding the rest. Add VRAM across GPUs only when the user intends to pin slots to separate
devices. Name the GPU exactly as `nvidia-smi` reports it, for example
`NVIDIA RTX 6000 Ada Generation`.

Warn when system RAM is under 32 GiB or free disk is under 50 GiB, because model images
and weights are large.

Re-probe when the user says the agent will run on a different machine from the one being
probed.

### Detection Failure Is Not `no_gpu`

`no_gpu` is a finding. A probe that could not answer is a different outcome, and the two
have opposite next steps. Read the actual output before choosing between them.

| Observation | What it means | Do this |
| --- | --- | --- |
| `Field "<name>" is not a valid field to query.` with exit 2 | the driver predates that field, and the entire query returned nothing | rerun without that field, then resolve compute capability through §Compute capability fallback |
| `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver` | the driver is not loaded, or this is a container without the container toolkit | detection failure. Report the layer and stop |
| `Failed to initialize NVML: Driver/library version mismatch` | the loaded module and the user-space library differ | detection failure. The host needs a module reload or reboot |
| `Failed to initialize NVML: Insufficient Permissions` | the account cannot reach the device nodes | detection failure. Report it rather than adding `sudo` |
| `nvidia-smi` is absent but `/proc/driver/nvidia/version` exists | the driver is present and the utility is not on this PATH | retry with the Stage 1 paths |
| `nvidia-smi -L` lists GPUs but the field query is empty | partial field support | keep the Stage 2 inventory and continue |
| Stages 1 to 3 empty and Stage 5 finds a module, device node, or PCI id | a GPU exists and the probe cannot read it | detection failure. Report it |
| Stages 1 to 3 empty and Stage 5 finds nothing | genuinely no NVIDIA GPU | record `no_gpu` |

On a detection failure, show the command, its exit code, and its error output, then say
what you were unable to determine. Ask the user to confirm the hardware or fix the named
layer. Never present cloud as the only option on the strength of a probe that failed, and
never claim a machine has no GPU without the Stage 5 evidence.

### Compute Capability Fallback

Compute capability decides LLM precision, so it cannot be guessed silently. When Stage 3
cannot return it, use the GPU name to identify the architecture, state that the value was
derived from the name rather than read from the driver, and treat the resulting precision
as a candidate only. Confirm it against the authoritative source for the locked model,
which is the support matrix for a NIM path and the model card for a vLLM path, before
anything is generated.

## 2. Classify the Host

| Signal | Host class |
| --- | --- |
| OS is macOS or Windows | `no_gpu`, cloud only, because self-hosted model containers need Linux |
| Stage 5 confirms no NVIDIA GPU | `no_gpu`, cloud is the only option |
| GPU name contains `NVIDIA GB10`, or DMI product name contains `DGX_Spark` | `dgx_spark`, see `platforms/dgx-spark.md` |
| Device-tree model or DMI product name contains both `Jetson` and `Thor` | `jetson_thor`, see `platforms/jetson-thor.md` |
| x86_64 with an NVIDIA GPU | `workstation`, see `platforms/deployment.md` |
| Jetson Orin or any other Tegra platform that is not Thor | `unsupported_edge`, cloud only |
| aarch64 with a GPU that is neither GB10 nor Thor | `unsupported_edge`, cloud only |

Match `DGX_Spark` with the underscore as the file reports it. Check DGX Spark before the
generic aarch64 row, because GB10 is an aarch64 platform and would otherwise fall through
to `unsupported_edge`.

Jetson Orin is not a smaller Thor. The models in this skill do not fit it, so it routes to
cloud rather than to a reduced local stack.

A probe that ended in detection failure has no host class yet. Resolve it with the user
before continuing.

## 3. Check the Credentials

Each path needs one model credential. Check that one, not every key.

| Path | Credential | Authenticates |
| --- | --- | --- |
| Cloud inference | `NVIDIA_API_KEY` | the API calls |
| NIM on a workstation | `NVIDIA_API_KEY` | the `nvcr.io` image pull |
| vLLM plus NeMo-Speech.cpp, any host | `HF_TOKEN` | the LLM weights and the speech GGUFs |
| LiveKit, cloud or self-hosted | `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | the room connection |

The single-GPU stack needs no `NVIDIA_API_KEY`, because everything it downloads comes from
Hugging Face and its speech container pulls from `nvcr.io` without NGC authentication.
Confirm that on the container's current NGC page, and stop on a registry authentication
failure rather than retrying.

A hybrid layout needs both model keys, one per placement. Ask for them together.

Check the shell environment first, then `.env`. A key exported in the shell counts even
when there is no `.env` file at all. Parse `.env`, never source it. Count a key as missing
when it is absent, empty, or still carries a placeholder such as `your-key-here`,
`changeme`, or `API_KEY_REQUIRED`.

When a key is missing, stop. Name it, say how to get it, and wait for the user to confirm
it is set. Ask for every missing key in one message rather than one at a time.

| Key | Where the user gets it |
| --- | --- |
| `NVIDIA_API_KEY` | [NVIDIA API key settings](https://build.nvidia.com/settings/api-keys). Ask the user to set it locally and confirm only |
| `HF_TOKEN` | Hugging Face settings. A read-scoped token is enough |
| `LIVEKIT_*` | the `lk cloud auth` browser flow, or project settings, API keys, create key |

Fetch the LiveKit steps from the LiveKit documentation MCP before answering, because that
console flow changes. Self-hosted LiveKit is an override the user has to ask for by name.

After the user approves file generation, write required keys to `.env.example` as
placeholders as specified by `output-contract.md`. Never write a real value to disk, and
never put model ids or endpoints in `.env`, which holds secrets only.

## 4. Deployment Fit

During proposal, set the intake **Deployment** row from usable VRAM. Model weights are
planning hints, not service footprints. Runtime memory also includes KV cache,
activations, CUDA graphs, and engine overhead.

Host class chooses the local stack. On a workstation, expected concurrency chooses between
two:

| Host class | Local stack | Fit method |
| --- | --- | --- |
| `workstation`, one user or a few concurrent sessions | vLLM plus NeMo-Speech.cpp | `platforms/single-gpu.md` §Memory |
| `workstation`, many users and higher concurrency | NIM for LLM, ASR, and TTS | §NIM budget below |
| `dgx_spark` | vLLM plus NeMo-Speech.cpp | `platforms/dgx-spark.md` and `platforms/single-gpu.md` §Memory |
| `jetson_thor` | vLLM plus NeMo-Speech.cpp | `platforms/jetson-thor.md` and `platforms/single-gpu.md` §Memory |
| `no_gpu`, `unsupported_edge` | none | cloud, skip local fit |

NIM is the workstation path for serving many users, because it is built for throughput and
scale-out. DGX Spark and Jetson Thor have no NIM path and no VRAM threshold, because both
run one shared unified-memory pool.

Infer concurrency from the use case and state the assumption in the Deployment row, so the
user amends one row rather than answering another question. A personal assistant, a demo, a
developer workstation, or an on-device agent is low concurrency. A contact centre, a shared
service, or a stated user count is high. When the use case does not say, propose the
single-GPU stack and name the scale at which NIM becomes the better choice.

### NIM Budget

Authoritative speech VRAM comes from the selected Speech matrix profiles. The model service
uses the LLM NIM matrix for a Cascaded LLM and the VLM NIM matrix for Omni. Its
post-approval `list-model-profiles` confirms the exact profile. Build the budget before
approving co-location:

1. select a candidate Compatible model profile from the correct NIM support matrix.
   Post-approval `list-model-profiles` from that NIM image pins the exact profile
2. reserve the selected ASR matrix GPU memory
3. reserve the selected TTS batch profile GPU memory
4. reserve GPU memory for the framework, CUDA, service startup, and engine build
5. give only the remainder to the model runtime, including weights and KV cache

That budget is a hard gate, and the arithmetic must pass before anything starts:

```text
model_weights + model_kv_and_activations + tts_vram + asr_vram + startup_overhead
    <= free VRAM measured now
```

Omni omits the ASR term, so its stack is Omni plus TTS. Hybrid counts only the slots
assigned to this GPU, because cloud slots consume no local VRAM. If the sum does not fit,
change the profile, the TTS batch profile, or the placement. Do not start the stack and
let the last service discover the shortfall as an OOM.

Use a Compatible NVFP4 profile when the selected service's matrix offers one. Otherwise,
use the next compatible precision from that matrix. Do not transfer a precision or memory
estimate between the LLM NIM and VLM NIM families.

For every shared workstation NIM GPU, set the controls in `models/llm.md` (profile,
context, runtime memory cap) and `models/tts.md` (explicit batch profile). Require the
Shared-GPU memory gate in `platforms/deployment.md`.

Higher concurrency raises the reserve on every slot, because it adds LLM KV cache and
needs a larger ASR and TTS batch profile. Size those profiles for the concurrency the
Deployment row assumes, and recalculate the budget when that assumption changes.

### Shared Rules

Recalculate whenever profile, precision, context length, TTS batch size, or slot placement
changes. Re-read this section whenever a slot moves GPU or to cloud, or the user asks why
something will not fit.

Estimates alone can support only `provisional co-location`, never the final
`self-hosted, co-located` status. That status requires the measured gate for the routed
stack, which is `platforms/deployment.md` §Shared-GPU memory gate on the workstation NIM
path and `platforms/single-gpu.md` §Start and verify on the single-GPU path.

Before recording anything below `self-hosted`, confirm the shortfall is real rather than
reclaimable (§Find out what is holding the memory).

| Fit result | Deployment row |
| --- | --- |
| Runtime budgets fit one GPU with startup headroom | self-hosted, provisional co-location |
| Slots fit only when pinned across GPUs | self-hosted, show placement (`platforms/deployment.md` §Scaling across GPUs) |
| Only some slots fit | hybrid, name each cloud slot |
| No local layout fits | cloud |

Host GPU detection is not container readiness. After self-hosting is approved and before
pulling an image, require Docker, Compose, NVIDIA Container Toolkit, in-container GPU
visibility, and writable cache paths through `platforms/readiness.md`.

## Anti-Patterns

- Asking whether to use cloud or self-hosted before probing.
- Combining `compute_cap` with the memory fields in one query, which discards the whole
  result on a driver that does not know the field.
- Recording `no_gpu` from a failed probe without the Stage 5 evidence.
- Presenting cloud as the only option to hide a detection failure.
- Reading a derived compute capability as a confirmed one.
- Offering a deployment choice on a machine with no GPU.
- Routing DGX Spark or Jetson Thor to NIM images, or unsupported edge hardware to any
  local stack.
- Proposing NIM for a single-user workstation agent, or the single-GPU stack for a
  high-concurrency service, without saying which assumption drove it.
- Reporting nameplate VRAM instead of what is actually free, or treating unified memory as
  dedicated VRAM.
- Reporting low free VRAM without saying what is holding the rest.
- Downgrading the model, the quantization, the context, or the placement to fit around a
  stale container the user would have stopped.
- Stopping, killing, or pruning anything the user did not approve by name.
- Building the budget from the pre-reclaim number after a container was stopped.
- Approving co-location from model weights without budgeting KV cache and service startup.
- Leaving the LLM runtime memory control or TTS batch profile implicit on a shared GPU.
- Generating files before checking credentials.
- Treating an `.env.example` placeholder as a real key.
