# NIM Metadata API

The NGC service that says, for one NIM image tag, which profiles exist and what each needs. Docs:
https://docs.nvidia.com/nim/metadata-api/latest/ . `nimselect.py` calls it for you; this page explains
what comes back. A China NIM distributor image is checked with the document of the NGC build it maps
to (`references/china-distributors.md`).

## Endpoints the scripts use
Fixed base URL `https://api.ngc.nvidia.com/v2` (`references/security.md`). The scripts send no
API key. The metadata, repository and search endpoints answer without one; the image list does too,
but only for a public NIM.

| purpose | request |
|---|---|
| profiles of one tag | `GET /nim/metadata?resourceId=nim/meta/llama-3.1-8b-instruct&tag=2.1.4` |
| repository, tags, and the CPU architectures and size of the image tagged latestTag | `GET /repos/nim/meta/llama-3.1-8b-instruct` |
| digest, CPU architectures and size of every tag | `GET /repos/nim/meta/llama-3.1-8b-instruct/images` |
| find a NIM from free text, or its hardware variants (select) | `GET /search/catalog/resources/CONTAINER?q=...` filtered to NIMs |

resolve and fit read the image list once, and select once for each repository it checks. It answers
403 for a NIM that is not public (for example `nim/nvidia/llama-3.3-nemotron-super-49b-v1`). When the
list cannot be read (that 403, another error, a timeout, or a 200 that is not an image list), the
scripts add a note, take what an alias stands for from the version numbers (an alias without a
suffix, such as `1`, never stands for a build for named hardware such as `1.8.0-RTX`), and use the
repository record for the image tagged latestTag (and for the newest version tag when latestTag is
`latest` or an alias of it); other tags of that NIM get `arch_not_recorded` (unknown).

The script retries 429, 5xx, timeouts and connection errors (3 attempts, 1 s and 2 s backoff), except
a 500 from the metadata endpoint, which is reported at once. It never retries 400 or 404, and branches
on the status code, never on the description text. A 429 left after the retries is
`ngc_rate_limited` (exit 4): NGC asks for fewer requests, so the command stops sending them.

## Status codes
| status | meaning | script result |
|---|---|---|
| 200 | metadata returned | fit runs. A body that is not JSON (a proxy page) or has no `profiles` list: `ngc_bad_response`, exit 4. An empty `profiles` list is like a 404: `no_metadata` (select tries the next image), never `does_not_fit` |
| 404 | no metadata for this NIM and tag. This is normal, not a failure | `no_metadata`, exit 3 (select: the next image is checked) |
| 400 | malformed request, or stored metadata that fails validation | `metadata_service_error`, exit 4 |
| 500 | internal error, or metadata that cannot be parsed | `metadata_service_error`, exit 4, without a retry |

Error bodies carry `requestStatus.requestId`; the script puts it in the message. Report it with that
id; do not retry with other parameters.

## Coverage
Coverage is partial, and an image's age does not predict it. Older NIMs were backfilled: 110 of the
136 documents found were for images pushed before the API launched on 2026-09-23. On 2026-11-03,
136 of 167 LLM and VLM repositories had metadata at their newest tag. Most documents carry no
`min_vram_per_device_gb` (32 of the 136 have it on any profile), so `unknown` with
`vram_not_recorded` is common.

## No metadata: the fallback
`no_metadata` (exit 3) means the API publishes no profiles for the image: for fit, the tag checked;
for select, every image it checked (`facts.images_considered`). select still writes its `--out`
file (`~/.cache/nim-select/select.json`) with `images_considered` and, on NGC, `tags_to_try`; fit
writes none. A select that an error stops before every image is checked leaves the previous file.
That file has no `image`, `tag` or verdict: match it by its `model` and `images_considered[].image`.
Each `images_considered[].metadata` is `not_published` (no metadata) or `error` (NGC failed to
answer for that image, not a sign that no metadata exists); a select whose every image failed exits
4, not 3, and writes the same file.
1. Say plainly that select found no published metadata (NIM Metadata API) for the images it checked
   (fit: for that tag), also when the user reports such a select that you cannot read. Never
   estimate VRAM or GPU count from the parameter count, and never invent a profile id. A guessed
   id cannot work: an id that is not in the image stops the NIM at start-up, before it serves.
   nim-deploy's troubleshooting reference lists that error as `profile-not-found`
   (`NIM_MODEL_PROFILE ... not valid`: the named profile is not in this image). The NIM does not
   fall back to a profile of its own, and nim-deploy never drops a pinned profile to let it pick one.
2. Offer the tags in `facts.tags_to_try` (other builds of the repo, newest first: a moving alias is
   given as the build it stands for, and aliases of the image without metadata and prereleases are
   left out; select also leaves out the ones it already checked) and rerun select with `--tag` on the
   one the user picks. Without that select's file, offer a new select with `--tag` on a tag the user
   names.
3. Or, after the user decides to pull the image, list its profiles with the image's read-only
   `list-model-profiles` command. `nim-deploy` (its Docker reference) owns that step; this skill pulls
   nothing.
4. If the user has a metadata JSON file for the exact tag, rerun fit with `--metadata-file FILE --tag TAG`.
   Always pass both: the file does not record its tag, and fit warns in `notes` when `--tag` is missing.
   fit still reads the repository and its image list from NGC first, and uses the file alone only
   when NGC cannot be reached (`ngc_unreachable`; NGC images only): the file replaces the Metadata
   API's answer, not the requests to NGC.

## Fields fit reads
Parse rules: unknown keys are ignored; `gpu` and `gpu_device` may be a string or a list; an absent field
means unknown, never false or 0. Only `profile_id` and `tested_gpu_devices` are always present.

| field | meaning |
|---|---|
| `profile_id` | opaque id; compare for equality only, never parse it. Becomes `model_profile` |
| `gpu` | human GPU name such as `H100`; never used to match |
| `gpu_device` | PCI id(s) of the target GPU, `pciDeviceId:pciVendorId` |
| `tested_gpu_devices` | PCI ids the profile was validated on; may be empty |
| `precision` | e.g. `fp8`, `bf16`, `nvfp4` |
| `inference_backend` | e.g. `tensorrt_llm`, `vllm`, `sglang` |
| `tensor_parallel_size`, `pipeline_parallel_size` | their product is the GPU count |
| `min_vram_per_device_gb` | per-GPU floor in decimal GB; the main fit field |
| `max_gpu_count` | most GPUs the profile can use |
| `min_driver_version` | minimum NVIDIA driver |
| `min_cpu_count`, `system_memory_per_device_gb`, `disk_space_per_device_gb` | host requirements |
| `feat_lora` | LoRA adapter support |
| `release_details.architecture` | CPU architectures, e.g. `["AMD 64", "ARM 64"]` |
| `throughput` | informational; units not standardised, never used to rank |

`facts.metadata` says where the profiles came from: `published` (the API) or `file` (`--metadata-file`).

## What resolve reports (NGC)
resolve reads the repository record and its image list, never the metadata. Besides `notes` and
`warnings`, its facts are:

| fact | meaning | what to tell the user |
|---|---|---|
| `resource_id`, `model`, `display_name` | the NIM found (`nim/TEAM/REPO`), its `TEAM/REPO` and its catalog name | which NIM it is |
| `mapping` | `exact` (the name was the repository), `search` (a catalog search found it), `by_name` (an image on a registry mirror, checked as the `nvcr.io/nim` image with the same name) | `search`, `by_name`: which NIM was found for the user's words or image |
| `tag`, `tag_source`, `image` | the version tag and why: `newest_version` (no tag asked for), `requested`, `alias_replaced` (a moving alias asked for, replaced by its build), `latest_replaced` (`latest` asked for, replaced by the newest version tag), `latest_tag_field` (no version tag listed: the repository's latestTag) | the image with its tag, and what was replaced |
| `tags`, `latest_tag_field` | the first 15 tags of the repository record, and its latestTag | only when asked |
| `architectures`, `image_size_gb` | the CPU architectures and size of that image; null when NGC does not show them | only when asked |
| `experimental` | the repository's labels include `Experimental` | true: NGC labels this NIM experimental |
| `nvaie_supported` | the repository carries the label `nvaie_supported` | true: NGC lists it as supported by NVIDIA AI Enterprise; false: nothing, unless asked |
| `can_guest_pull` | the repository's `canGuestPull` | true: the image can be pulled without an NGC key (starting the NIM can still need one; `nim-deploy` checks). false or null: pulling it needs an NGC key, which nim-select never uses or asks for (`nim-deploy` pulls; the list marks such rows `needs_ngc_key`) |
| `source_registry` | the registry of a mirror image the user named, else null | the mirror must carry the image handed off (SKILL.md, Inputs) |

These are NGC's labels and records, shown as data: none of them changes a verdict. The candidates
of `model_ambiguous` carry `experimental` and `nvaie_supported` from the catalog search too.
