# Fit rules

How `nimselect.py fit` and `select` decide. It follows the NIM Metadata API procedure "Choose a
Profile for Your Hardware" plus the NIM runtime's precision rules. Each profile goes through the
checks below in order and gets one verdict. Every reason is kept in the profile's `reasons` list, and
`why` says it in plain words.

## Verdicts
| verdict | meaning |
|---|---|
| `fits` | every check was made and passed |
| `unknown` | nothing failed, but at least one check could not be made (the field is missing, an environment value is unknown, or the profile was not tested on this GPU) |
| `rejected` | at least one check failed |

The overall verdict is `fits` when any profile fits, else `unknown` when any profile is unknown, else
`does_not_fit`. `recommended` is the best `fits` profile; `best_unconfirmed` is the best `unknown`
profile when nothing fits. With `fits`, `alternative` can name a profile with a faster precision
(more accurate with `--prefer accuracy`) for the user to choose: one that fits in another image
select checked, else an untested one of the same image (Ranking, The alternative). A missing record
that a check needs (VRAM, CPU architecture, LoRA, backend) is never a pass: `unknown` stays `unknown`
until the user decides. An absent driver, CPU, memory or disk requirement (section 5) is not checked;
`not_recorded` lists those that no candidate records.

## 1. GPU match (tier)
The scripts match on PCI ids (`pciDeviceId:pciVendorId`, e.g. `2321:10de`), never on GPU names.
The first row that holds gives the tier.

| tier | rule | evidence |
|---|---|---|
| `direct` | the profile's `gpu_device` contains one of our GPU's ids | built for this GPU |
| `tested` | `tested_gpu_devices` contains one of our ids | validated on this GPU |
| `tested`, record of another image | select, fit and the list: this image's record of the profile names our GPU in neither field, but another image of the same repository that select checks records the same `profile_id`, with the same nonempty `workspace_hash` in both images, as built for or tested on it | built for or validated on this GPU in that image: note `tested_in_other_image`, and a why line that names the image |
| `agnostic` | the profile has neither `gpu` nor `gpu_device`, and neither this image nor another image of the same repository that select checks records it as built for or tested on our GPU | `gpu_not_tested`: `unknown` at best |
| rejected `gpu_mismatch` | `gpu_device` names another GPU, and no row above holds | |
| rejected `gpu_name_only` | only the human `gpu` name is set, and no row above holds; it cannot be matched safely | |
| `unmatched` or `agnostic` with `pci_unknown` | our GPU's PCI id is unknown (a Kubernetes product name not in the PCI table), so built-for and tested-on cannot be checked | verdict `unknown` |

A build's metadata can leave out a test that another build of the same repository records for the
same profile: on 2026-11-19, nemotron-3.5-lightning-30b-a3b:2.1.6 records no GB10 test of its two
GPU-agnostic 1-GPU nvfp4 profiles, `920207996e4e...` and `15766f03e594...`, and 2.1.5 lists the
GB10 for both. So select reads every image it checks first, then counts such a profile as `tested`
in the image it judges, with the note `tested_in_other_image` in its `reasons`, a line in
`facts.notes`, and a why line that names the image the record comes from, the newest when several
do: "Tested on this GPU (2e12:10de) in 2.1.5 (same profile id); this image's metadata records no
test" ("Built for" when that image records it as built for this GPU). The record counts only for the
exact `profile_id` with the same `workspace_hash`, recorded and not empty in both images (the id
hashes the profile's configuration, the hash its files: a record of other files under the same id,
or with no hash on either side, never counts), and only from an image of the same
repository that select checks: never across repositories (a hardware variant in another
repository, such as `<repo>-dgx-spark`, lends none and takes none) or ids, however alike the
profiles. The repository's own tag for a GPU (`1.8.0-RTX`) is an image of that repository: it
lends and takes records as the others do. A select of one image (a tag named) and `fit` read the
other images a select without a tag checks, for their records alone, so an image gets the same
verdict however it is checked; `--max-images` counts the image named among them (`--max-images
1`: that image alone). A CND select and `fit --metadata-file` read none. An image whose metadata
could not be read lends nothing, and a note says so; when NGC stops answering (`ngc_unreachable`,
`ngc_rate_limited`) while they are looked for or read, no more is asked of it, and the image named
is still judged, with the records read, and a note says which were not. A record is a test of
that profile on this GPU, not of this image's runtime: the NIM's start is still its first real
check.

Only `direct` and `tested` profiles can be `fits`. The metadata is a screen before the pull; only
the image can confirm a profile, with its `list-model-profiles` after the pull. The hand-off asks for
that check (`confirm_with`). nim-deploy can list the image's profiles with the read-only
`list-model-profiles` command of its Docker reference, but neither nim-deploy nor nim-tune checks
the handed-off profile against that list by itself, so the NIM's start is the first real check.

Names mislead: profiles marked `H100` usually target H100 SXM (`2330:10de`), not H100 NVL (`2321:10de`).
On Kubernetes the GPU Feature Discovery labels carry a product name, not a PCI id; the ids come from a
product table (`pci_inferred`), and a match on them says so in the notes.

## 2. Precision (agnostic tier only)
A direct or tested match (a record of another image too) is validation evidence and skips this
check.

| precision | needs compute capability |
|---|---|
| `fp8` | 8.9 or newer |
| `nvfp4` | 10.0 or newer. The NIM runtime can emulate NVFP4 on 8.0-9.x with vLLM or SGLang only when the profile opts in, which the metadata does not record, so fit treats it as unsupported |
| `int4_awq` | not supported on 10.0 up to (not including) 12.0 |
| others | not gated |

Unknown compute capability: `capability_unknown` (unknown). Failure: `precision_unsupported`.

## 3. VRAM
`min_vram_per_device_gb` is per GPU, not a total, in decimal GB. The environment's `memory_gb` is
MiB x 1048576 / 1e9, cut (not rounded) to 2 decimals, so it is never overstated: an H100 NVL with
95830 MiB has 100.48 GB, an L40S with 46068 MiB has 48.30 GB and passes a 48.0 floor.

| case | result |
|---|---|
| GPU memory >= floor | pass |
| GPU memory < floor | `vram_too_small` |
| field absent | `vram_not_recorded` (unknown) |
| GPU memory unknown (a Kubernetes node without GPU Feature Discovery labels, or a discrete GPU whose `nvidia-smi` shows no memory size) | `vram_unknown` (unknown) |

Unified memory: a GPU built into the system (the GB10 of DGX Spark, Jetson) has no memory of its own
and shares the host's, so inspect judges VRAM against the host's memory and adds the condition
`unified_memory`. On a host `nvidia-smi` reports no memory size for it, and inspect takes a GPU with
no memory size for one only on evidence: its PCI id (`2e12:10de`, the GB10 of DGX Spark) or a
compute capability only such GPUs have (8.7 Jetson Orin, 11.0 Jetson Thor, 12.1 GB10). It then uses
the host's total memory and counts the GPU as idle. Any other GPU with no memory size has unknown
memory (`vram_unknown`). On Kubernetes inspect reads GPU memory from the node's
`nvidia.com/gpu.memory` label. Only the GB10 (`2e12:10de`) is known there to share memory: inspect
ignores its label (missing, 0, or the node's total memory), which then splits no nodes into groups,
and uses the node's allocatable memory, what a pod can have (the smallest of the group's nodes).
This is the one place where the smallest node counts: host memory and CPUs take the largest node,
with `handoff.placement` for the smaller ones (section 5), but a GB10's VRAM is its node's memory,
and the hand-off cannot yet steer the pod to the larger node for it, so a profile that only the
larger GB10 node can hold is `does_not_fit` rather than a fit that may land on the smaller one. The
Grace superchips (GH200, GB200, GB300) have memory of their own and report it.

## 4. GPU count
Needed GPUs = `tensor_parallel_size` x `pipeline_parallel_size` (each 1 when absent). Available GPUs =
the smallest of: idle GPUs on one node (host: memory used <= 5%, NIM's own rule; Kubernetes: the
allocatable GPUs of one node, because usage is unknown), `--max-gpus`, and the namespace GPU quota left
(the most constraining quota). Profiles span one node only. A host GPU whose memory use `nvidia-smi`
does not give is neither idle nor known to be busy: it is not available, but it can make a profile
`unknown`. Possible GPUs = the smallest of: the idle GPUs plus the GPUs of unknown use (at most the
node's GPUs), `--max-gpus`, and the quota left; on Kubernetes they are the available GPUs. With 6
idle GPUs, 2 of unknown use and `--max-gpus 6`, 6 are possible, so a profile that needs 8 is
`not_enough_gpus`.

| case | result |
|---|---|
| needed > possible | `not_enough_gpus` |
| needed > available, but not more than possible | `gpu_use_unknown` (unknown) |
| `max_gpu_count` < needed | `max_gpu_count` |

## 5. Host requirements (only when the field is present)
| field | compared with | failure | env value unknown |
|---|---|---|---|
| `min_driver_version` | driver version, dotted integers | `driver_too_old` | `driver_unknown` |
| `min_cpu_count` | CPU cores | `cpu_count_too_low` | `cpu_unknown` |
| `system_memory_per_device_gb` x GPUs | host RAM or node allocatable | `memory_too_small` | `memory_unknown` |
| both of the two above, on Kubernetes | the CPU cores and memory of one node | `node_too_small` | as above |
| `disk_space_per_device_gb` x GPUs | free disk at the NIM cache folder | `disk_too_small` | `disk_unknown` |
| `release_details.architecture`, image architectures | CPU architecture | `cpu_arch_unsupported` | `arch_unknown` |

On Kubernetes, inspect records each GPU type's driver and CPU architecture from its own nodes, and
the largest CPU count and memory (`host_memory_gb`) among them: a NIM pod needs one node, and one
with room is enough. The two largest can be on different nodes, so inspect also
keeps each node's CPU count and memory as a pair (`node_sizes`; a node that another matches or beats
on both is left out). When the largest of each is enough for a profile but no one node has both,
the profile is `node_too_small`: nodes with 64 CPUs and 128 GB, and 16 CPUs and 512 GB, do not fit
a profile that needs 32 CPUs and 256 GB. inspect also records the smallest of each
(`smallest_cpu_count`, `smallest_host_memory_gb`): a profile that only some nodes can hold still
`fits`, but a pod that requests only GPUs can be placed on any node with enough free GPUs, of this
group or another, so the hand-off says so (`placement`, The hand-off). Nodes of one type that
differ only in CPUs or memory (often by a few MB) are one group. fit judges each type with those
values: one cluster can mix Grace (arm64) and x86 GPU nodes. Only a type whose own nodes have more
than one architecture, or no `kubernetes.io/arch` label, is `arch_unknown`. An environment file
without these values uses the cluster-wide ones.

`fit` and `select` also need the image's own architectures (both set `arch_required`). They come
from NGC's image list, which has each tag's own (`GET /repos/{id}/images`). When the list cannot be
read (a NIM that is not public answers 403 there; also an error, a timeout or an answer that is not
an image list), a note says why and the repository record gives them, but only for the image tagged
latestTag, and for the newest version tag when latestTag is `latest` or an alias of it (a note says
this is taken to be the same image; no digest can confirm it). An image no source covers, whose
profiles record none either, gets `arch_not_recorded` (unknown), so an unverified image is never
`fits` by default. A China NIM distributor (CND) image is checked with the architectures it has and
its NGC build has too (`references/china-distributors.md`).

Fields absent on every candidate are listed in `not_recorded`. Tell the user these were not checked
(the driver especially).

## 6. Capabilities (only when asked)
| option | field | absent | mismatch |
|---|---|---|---|
| `--need-lora` | `feat_lora` true | `lora_not_recorded` | `lora_unsupported` |
| `--backend NAME` | `inference_backend` equal, any case | `backend_not_recorded` | `backend_mismatch` |

## Ranking
Among `fits` (and separately among `unknown`), the smallest key wins:
1. tier: direct, tested (a record of another image of the same repository too, GPU match), agnostic
2. precision, `--prefer throughput` (default): nvfp4, mxfp4, fp4, int4_awq, fp8, int8, bf16, fp16, fp32, other.
   `--prefer accuracy`: fp32, bf16, fp16, fp8, int8, int4_awq, mxfp4, fp4, nvfp4, other
3. backend: tensorrt_llm, trtllm, vllm, sglang, other
4. LoRA: without `--need-lora`, profiles without LoRA first
5. evidence: the image's own record (built for or tested on this GPU) before a record of another
   image (`tested_in_other_image`)
6. more GPUs first, within the limit (the NIM runtime also prefers higher tensor parallelism)
7. profile id, ascending

`throughput` is informational only; its units are not standardised, so it never ranks.

### The alternative
Tier ranks before precision on purpose. A profile built for or tested on this GPU (`direct`,
`tested`, here or in another image of the same repository select checks, by the same profile id)
has evidence that it runs here; a GPU-agnostic profile not tested on it has none, however
good its precision, and only the image's `list-model-profiles` after the pull, and the NIM's start,
can confirm it. So an untested profile is never `fits` and never recommended over a tested one,
and select keeps the image it chose (the first that fits, Image choice) even when another image it
checked has a faster profile that fits. Instead, with a `fits` verdict, select and fit name one
such profile as `alternative`, a choice for the user. Its precision must have fewer bits than that
of the recommended profile and of every other profile that fits in the chosen image (`throughput`,
the default: 4-bit `nvfp4`, `mxfp4`, `fp4` and `int4_awq`, then 8-bit `fp8` and `int8`, then
16-bit `bf16` and `fp16`, then `fp32`), so it is faster; with `accuracy`, more bits, so it is more
accurate. Ranking ahead in the order is not enough: `bf16` is no faster than `fp16`, nor `nvfp4`
than `mxfp4`. When a profile that fits in the chosen image already has as few bits, it is that
faster choice, with a test: nothing is named. A precision the order does not list (such as `int4`)
ranks last but says nothing of speed or accuracy: a profile with one is never named, and a
recommended profile with one gets no alternative. Rule 1 leaves the chosen image's other profiles
with such a precision out of the comparison; rule 2 names nothing when any profile that fits has
one. `--max-gpus`, `--need-lora` and `--backend` apply as to any profile: one that fails them, or
leaves them unknown, is never named. Of the two rules below, the first that finds a profile wins:
1. Another image (select only; `confirmed` true): a profile that fits (tier `direct` or `tested`,
   no check left open) in another image select checked (`images_considered`) by that image's own
   records, never by the record of a third image (`tested_in_other_image`: the offer says it was
   built for or tested on this GPU in that image), on the GPU type select picked for the chosen
   image, which must also be the one it picks for that image, so the select of that image hands
   off the same GPUs. The images are taken in this order: hardware variants in other
   repositories, as select checks them (their versions do not compare with the repository's),
   then the repository's own images, newest version first, its tag for a GPU (`1.8.0-RTX`) placed
   by its version numbers. The first image with such a profile wins, and in it the first such
   profile in the ranking above. An image without a fixed version tag (a moving alias such as
   `1.7`, or `latest`) is never named, nor is one select did not check (`--max-images`). Select
   checks one image on a CND and with `--tag`, and one profile with `--profile`: none of them
   gives one.
2. The same image (`confirmed` false): a profile of the chosen image and GPU group whose only open
   check is `gpu_not_tested`: no failure, and no other unknown (`vram_not_recorded`,
   `arch_not_recorded`, `driver_unknown`, `capability_unknown`, `pci_unknown` and the like rule it
   out). Notes such as `pci_inferred` are fine. The first such profile in the ranking is named.
   It is not offered (`facts.alternative` null) when its select could not check the same image
   and profiles: an NGC image without a fixed version tag (`latest`, an alias with no build), or
   fit with `--metadata-file` (select reads the metadata NGC publishes, not the file).

`alternative` has the profile's record (`profile_id`, `verdict`, `tier`, `precision`, `backend`,
`gpus_required`, `reasons`, `why`), its `image` with the version tag (null for a CND image listed
only as `latest`), that image's `kind` as `images_considered` names it (null from fit),
`confirmed`, and an `offer` sentence: what it gives over the recommended profile (and its GPU
count, when that differs); for rule 1, its evidence and its image ("faster (nvfp4) and tested on
this GPU, but in the older image 2.1.5"): older or newer than the chosen image by the version
numbers when both are of one repository (on an RTX PRO 6000, beside 1.8.0-RTX, 2.0.13 is the
newer image), else a hardware variant, or an older version of the repository; for rule 2, that
this image's NGC metadata records no test of it on this GPU. With none, `alternative` is null. The
user who takes it gets a select of that profile alone (`--tag` of its image, `--profile`), on the
GPU type the first select chose, whose hand-off is `fits` for rule 1 and `unknown` for rule 2
(The alternative profile, below).
On 2026-11-19, on a DGX Spark (GB10, `2e12:10de`), nemotron-3.5-lightning-30b-a3b:2.1.6 records
the GB10 as tested only for its bf16 profile `aa2e5b4037ab...`; of its GPU-agnostic 1-GPU nvfp4
profiles, `920207996e4e...` records no tested GPU and `15766f03e594...` only the B200 (`2901:10de`).
2.1.5 and 2.1.4 list the GB10 as tested for both ids, so select counts both as `tested` in 2.1.6
(GPU match), takes 2.1.6, the newest image that fits, and recommends `920207996e4e...` (without
LoRA it ranks first; `15766f03e594...` supports LoRA), with the why line "Tested on this GPU
(2e12:10de) in 2.1.5 (same profile id); this image's metadata records no test". The bf16 profile
stays a fit below them, and nothing is faster than nvfp4: no alternative. `--tag 2.1.6` and `fit`
read 2.1.5 and 2.1.4 for their records too, and give the same. Only with `--max-images 1` is 2.1.6
read alone: select then recommends the bf16 profile and names its own `920207996e4e...` as the
alternative, unconfirmed.

## Environment conditions
| condition | meaning | what to do |
|---|---|---|
| `mig_enabled`, `gpus_partitioned` | MIG or GPU sharing hides whole GPUs (`gpus_known` false) | host: ask which whole GPUs to use (`--gpus`), or to disable MIG. Kubernetes: ask for a context with whole GPUs, or to disable MIG or sharing |
| `mig_mixed` | a Kubernetes GPU node advertises MIG slices next to whole GPUs (the mixed MIG strategy) | only its whole GPUs are counted; say so |
| `gpus_busy` | some GPUs use more than 5% of their memory, or `nvidia-smi` gives no memory use for them | neither counts as available; a profile that needs a GPU whose use is unknown is `unknown` (`gpu_use_unknown`): ask the user whether it is free |
| `mixed_gpus` | several GPU types | a NIM uses one type; fit reports the best group and lists all groups |
| `unified_memory` | the GPU shares the host's memory (the GB10 of DGX Spark, Jetson Orin, Jetson Thor; section 3) | the VRAM check uses the host's total memory (Kubernetes, GB10: the node's allocatable memory); say so |
| `pci_inferred`, `pci_unknown` | Kubernetes PCI ids from the product table, or none | say the match is inferred, or that it could not be made |
| `nodes_unschedulable`, `nodes_not_ready` | cordoned or not-ready nodes | they are skipped or unknown; when they hide every GPU, ask whether the nodes can be fixed |
| `gpus_not_advertised` | a GPU node advertises no allocatable `nvidia.com/gpu` | its GPUs are not counted; ask the user to check the NVIDIA device plugin |
| `gpu_labels_missing` | a GPU node has no GPU Feature Discovery labels | its GPU model, memory and PCI id are unknown, so its profiles stay `unknown`; ask for GPU Feature Discovery |
| `gpu_quota_scoped` | a scoped ResourceQuota limits GPUs for some pods only | it is not counted; tell the user it may still limit the NIM |
| `docker_missing`, `docker_remote` | no Docker, or a remote daemon (`DOCKER_HOST` or the Docker context) | not a fit failure; `nim-deploy` needs Docker on the GPU host |
| `docker_unreachable`, `docker_no_gpu_access` | `docker info` failed, or Docker shows no nvidia runtime or CDI device | not a fit failure; tell the user `nim-deploy` needs Docker with GPU access |

A cluster with no schedulable GPU node stops inspect with `no_gpus` (exit 3), as a host without GPUs does.

## Image choice (`select`)
On a China NIM distributor, select checks one image: the one named (`requested`), or the newest
image of the repository with a version tag and a known NGC build (`newest`), with the profiles of
its NGC build; it checks no older version or hardware variant (`references/china-distributors.md`).
On NGC, each candidate image goes through the checks above. Candidates, in this order, at most
`--max-images` (6):

| kind | which image |
|---|---|
| `requested` | the tag the user named (in MODEL or `--tag`); then it is the only candidate. A moving alias (`1.7`) is replaced by the build it stands for (`1.7.0-variant`; tag_source `alias_replaced`). Judged by the version numbers, an alias without a suffix never stands for a build for named hardware: `1` on deepseek-r1-distill-llama-8b is 1.5.2, not 1.8.0-RTX |
| `newest` | the newest version tag of the repository: plain tags first, then suffixed builds for any GPU (`-variant`); a prerelease (`-rc1`) or a build for named hardware (`-RTX`) only when nothing else is listed. A moving alias gives way to the build it stands for. A major.minor alias of that line that stands for a newer build wins (`1.8` points to `1.8.3-variant`, not the plain `1.8.1`). What an alias stands for comes from the image digests of the public image list, else from the version numbers |
| `variant` | a hardware build of the same model: `<repo>-<suffix>` in the same team, where the suffix is or starts with `dgx-spark`, `spark`, `rtx`, `rtx-pro`, `gh200`, `gb200`, `gb300`, `gb10`, `jetson`, `igx` or `thor`, at its newest tag; and the newest tag of the repo whose suffix names a GPU found (`1.8.0-RTX` for an RTX 6000 Ada) |
| `older` | first the earlier builds of the newest's major.minor line, newest first (after 2.1.6: 2.1.5, 2.1.4; after the arm64-only `1.7.1-variant` of qwen3.6-35b-a3b: `1.7.0-variant`, amd64 and arm64). A newest build can lose a GPU from its tested list: on 2026-11-11 no profile of llama-3.1-8b-instruct:2.1.6 lists the H100 NVL (`2321:10de`), while 2.1.5 lists four; 2.1.6 has the same four ids, so with 2.1.5 checked they count as tested in 2.1.6 too (GPU match). They take only the places `--max-images` leaves after the newest, the variants and the older lines, and at least one: on an RTX 6000 Ada, where 1.8.0-RTX takes a place, llama-3.1-8b-instruct checks 2.1.5 but not 2.1.4. A tag that is the same image as the newest or as a build taken is skipped: the same digest in the image list or, without digests, a build for any GPU with the same numbers (`1.7.0-variant` beside `1.7.0`; on 2026-11-11 such twins were one image on every NIM that listed both). Then the newest build of each of the next two older major.minor lines (after 2.1.6: 2.0.13, 1.15.5), which keep their places. A build has three or more numbers, may carry a suffix such as `-variant`, and is not a prerelease (`-rc1`) or a build for named hardware (`-RTX`). A major.minor alias counts as the build it stands for (`1.7` as `1.7.0-variant`), and as itself only when no build is listed for it |

| case | what select does |
|---|---|
| production branch (`-pb6`, `-pb25h2`) | never checked; listed in `notes` |
| other repo with the same prefix (`deepseek-r1-0528`, `-distill-*`, `gpt-oss-120b-turbo`) | another model or build: never checked; listed in `notes` |
| `-base` model family | ignored |
| more candidates than `--max-images` | the rest are listed in `notes` ("Not checked"); one older version keeps a place when variants would fill the limit |
| an older image chosen | `image_choice.reason` names the newer images checked and why each was not taken ("the newer images checked are not confirmed to fit": each does not fit or is `unknown`); a note names the newer version tags of the repository that were not checked (other builds of its line and of the older lines), whose fit is not known. A hardware build of the repository chosen (`1.8.0-RTX`) gets no such note: it was taken for the GPU |
| 404, an empty profile list, or a metadata error for one image | recorded in `images_considered`; the search goes on |

The choice: the first candidate that `fits`; else the first `unknown`, unless a later `unknown`
image's best profile was built for or tested on this GPU (tier `direct` or `tested`) and leaves no
check open that the first one's leaves closed (its `reasons` are among the first one's, notes such
as `pci_inferred` and `tested_in_other_image` aside): it then confirms more, so it is taken. A
build for this GPU that records no VRAM need does not replace an earlier image that records one.
Else the first `does_not_fit`.
An `unknown` ranks above `does_not_fit` whatever the reasons, so an older build whose metadata
records less (no VRAM need: `vram_not_recorded`; CPU architectures not listed:
`arch_not_recorded`) is taken over a newer image whose own records rule it out
(`not_enough_gpus`, `vram_too_small`, `cpu_arch_unsupported`); `images_considered` and the
reason show both, and the user decides. On 2026-11-11, on one L40S, the L40S profiles of
qwen3.8-27b:2.1.4-variant need more GPUs, and 2.1.1-variant, whose 1-GPU profile records no
VRAM, is `unknown`; on GB10 (arm64), llama-3.1-nemotron-nano-vl-8b-v1:1.3.1 is amd64 only, and
1.3.0, whose CPU architectures NGC does not show, is `unknown`.
`image_choice.reason` says why in one sentence, with the checks each unknown image left open
(`images_considered[].reasons`). No candidate with published metadata: exit 3 `no_metadata`. Every
candidate's metadata failed: exit 4 with that error. `ngc_unreachable` stops the command (exit 4)
and keeps the images already checked in `images_considered`, but not while other images are read
only for their records (GPU match).

## The hand-off
`facts.handoff` is written for the profile that fits, or for `best_unconfirmed` when the verdict is
`unknown` (its `verdict` says which). It names the image with its version tag (a CND image: the CND
image, never `facts.ngc_image`; null when the CND lists it only as `latest`), `model_profile`, the
profile's `backend` (nim-tune tunes `vllm` profiles only) and the GPUs. On a host these are the
first idle GPUs of the group the profile was checked on: `gpu_ids`, and `gpus` in Docker's
`device=0,1` form, because a bare `1` reads as a count (nim-deploy takes `gpus`; nim-tune's
`docker.gpus` takes `gpu_ids` joined with commas). When fewer are idle than the profile needs
(`gpu_use_unknown`), both are null and a warning says to ask the user which GPUs to use. On
Kubernetes the scheduler picks the GPUs, so `gpus` is the count and `gpu_ids` is null. SKILL.md
(Hand-off) says which skill takes it.
A Kubernetes hand-off also carries the CPU cores and host memory the profile records, in the units
of a pod's resource requests: `cpu` (`min_cpu_count`, in cores) and `memory`
(`system_memory_per_device_gb` x GPUs, in decimal GB: `64G`), each null when not recorded. A pod
that requests only GPUs can be placed on any node with enough free GPUs, and nim-deploy sets only
the GPU count. So when a GPU node the pod can be placed on is not known to have them, `placement`
says the deployment must request them or be pinned to a node that has them, and the user confirms
how it is placed; else it is null. Those are the nodes of every group with at least the GPUs the
profile uses, whatever its GPU, not only of the group it was checked on: a node of the same GPU
with another GPU count or driver is a group of its own. Each group gives its smallest values
(`smallest_cpu_count`, `smallest_host_memory_gb`); an older environment file, the cluster's
smallest. With nodes of 16 CPUs and 128 GB, and 4 CPUs and 32 GB, a profile that needs 8 CPUs and
64 GB `fits`, with `placement` set, whether the two nodes are one group or two. A host hand-off
has none of the three.
`confirm_with` is `list-model-profiles`: the check to make after the pull, which neither nim-deploy
nor nim-tune makes by itself yet.

## Why the profile is pinned
A profile id is an opaque id that belongs to one image version: the same model at another tag can have
other ids. Without a pinned profile the NIM runtime picks one at start-up, which may differ from the one
checked here. So the hand-off always carries both the version-tagged image and `model_profile`. A
test of the same id recorded in another image of the repository (GPU match) changes neither: the
hand-off still names the image select chose, with that id.

## The alternative profile (`facts.alternative`)
With a `fits` verdict, select can also name `facts.alternative`: a profile whose precision is
faster than that of the recommended profile and of the other profiles that fit in the chosen image
(`throughput`, the default; `accuracy`: more accurate), on the same GPU type, for the user to
choose (Ranking, The alternative, above). Its `confirmed` says which kind it
is:
- `confirmed` true: it fits (built for or tested on this GPU, by that image's own records) in
  another image select checked: `image`, with its version tag, and `kind` (`older`, `variant`) as
  `facts.images_considered` names it. Its `offer` says how that image stands to the chosen one:
  older or newer (by version, in one repository), or a hardware variant. Select still recommends
  the image it chose (the first that fits, in the order it checks them).
- `confirmed` false: a profile of the chosen image that passed every check except one: no test of
  it on this GPU is recorded, in this image's NGC metadata or in another image of its repository
  that select checks (`gpu_not_tested`). So it is `unknown`, never `fits`.

A note in `facts.notes` says how to offer it and the select to run on a yes. `facts.alternative`
null: offer no other profile this way. With more than one GPU type, no profile of a hardware
variant in another repository is named, from another image or the chosen one: the select a note
gives reads that repository's other images for their records, which this select did not read,
so it could hand the profile off on another GPU type than the one the user confirmed.
- Report the recommended profile as select gives it. The alternative never replaces it and is
  never counted with the `fits` profiles of the chosen image.
- Offer it in the confirm question, after the question for the recommended profile, with its
  `offer` sentence; name its `profile_id`, and ask which profile to use. A reply that does not say
  which: ask which.
  - `confirmed` true ("faster (nvfp4) and tested on this GPU, but in the older image 2.1.5"): name
    its image with the version tag, as the offer places it (older, newer, a hardware variant), and
    ask whether that image is acceptable. It fits in that image only: never say it fits the chosen
    image, which lacks that profile or does not fit it.
  - `confirmed` false ("faster (nvfp4) than the recommended bf16, but this image's NGC metadata
    records no test of it on this GPU"; it names the GPU count when that differs): say it is
    unconfirmed, and never call it compatible, confirmed or a fit.
- Never choose it for the user. Asked to choose, say the recommended profile is the one select
  recommends (built for or tested on this GPU, as its why lines say, in the image it chose) and
  that taking the other is the user's call.
- Offer no other profile as a faster or more accurate choice on your own (one of
  `facts.unconfirmed`, or one of another image in `facts.images_considered`): only
  `facts.alternative`, which select names only when every rule of Ranking, The alternative
  (above) holds.
- The user takes it: run the select command the note gives, as written: `select MODEL --tag TAG
  --profile=ID`. MODEL and TAG are those of the alternative's image: for `confirmed` true, its own
  repository and version tag (a hardware variant's repository, `...-dgx-spark`, is not the MODEL
  first given); for `confirmed` false, the chosen image's (`facts.resource_id` and `facts.tag`; on a
  CND, `facts.cnd_repository` and `facts.cnd_tag`). ID is the alternative's full `profile_id`. Keep
  `--env` and the other options of the first select, but write to its own file, `--out
  ~/.cache/nim-select/select-alternative.json`, so the first select's file stays as it was (the user
  may still take the recommended profile). That select checks only that profile
  of that image, on the GPU type the first select chose, and gives `fits` for `confirmed` true
  (`next: confirm_with_user`) or `unknown` for false (`next: ask_user`). The user has already
  chosen, so ask nothing more: hand off its `facts.handoff`, and its file, as SKILL.md's Hand-off
  says (for
  `unknown`, saying the profile is unconfirmed and that the NIM's start is its first real check).
  When its GPUs differ from the ones the user confirmed (the alternative needs another count), or
  on Kubernetes its `facts.handoff.placement` is set, first ask the confirm question for its
  `facts.handoff.gpu_ids` (Kubernetes: its count, and that placement). A CND image listed only as
  `latest` (`facts.pinned` false): first ask for its own version tag, as for any hand-off. Another
  verdict, or an error such as `profile_not_found` (exit 2, which leaves the `--out` file as it
  was): report it and ask again before any hand-off.
- The user takes the recommended profile: hand off the first select's `facts.handoff` and its file
  (`select.json`), as usual.
