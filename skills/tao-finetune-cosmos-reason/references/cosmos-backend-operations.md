# Cosmos backend operations and recovery

## Runtime image selection and optional source builds

Runtime selection is independent of source construction. A user-supplied
SLURM `.sqsh` is the authoritative container artifact and is used directly
after compute-frame readability checks. Its filename does not need to match a
packaged image tag, and repository/build provenance and SQSH SHA256 are not
runtime gates. If no SQSH is supplied, use the exact image in the selected
backend contract, derive its target under `sqsh_cache_dir`, reuse that target
when readable, or convert the exact image once through the SLURM platform
before GPU submit. Docker uses an explicit image tag when supplied and the
packaged backend image otherwise.

The remaining source-equivalence rules apply only when the user explicitly
selects `source-build`.

For every repository, record branch, commit, tree, and dirty state. Refuse a
reproducibility build when any packaged source is dirty. The Framework path
first builds its native Dockerfile, then builds the TAO action layer using that
exact base. The Cosmos-RL path builds `Dockerfile.cosmos_rl` with exact native
RL, TAO actions, DAFT, and TAO Core commits.

Inspect `/opt/tao/image-provenance.json` after build. Verify repository commits,
source-manifest checksum, dependency inputs, Python/package locations, and
non-root imports. Resolve and record image ID/digest. SLURM then converts that
exact digest to a newly named SQSH in the runtime-supplied cache directory and
records its SHA256. A source change invalidates both image and SQSH.

For Cosmos-RL, image verification must import both `deep_ep` and
`deep_ep_cpp`, inspect the compiled extension for the internode mask-buffer
symbols referenced by the Python bindings, and verify that vLLM uses the
linear-equivalent Qwen3-VL Conv3D path for every PyTorch version at or above
2.9. A successful `nvidia-smi` or `cosmos-rl --help` check does not cover these
ABI and dispatch contracts.

## Model preparation

The logical base model is always supplied. A local Qwen3-VL safetensors model
can be used directly after config, tensor-index, tokenizer, and processor
validation. For a Hugging Face ID/URL, the planner resolves `main` or the
user’s optional branch/tag to an immutable commit and snapshots that commit to the
runtime checkpoint area. Cosmos3 Nano Omni inputs use the packaged
`Qwen/Qwen3-VL-8B-Instruct` architecture mapping and the already selected
backend runtime; the planner resolves both Hub identities to immutable commits.
The selected image invokes the TAO-owned
`cosmos_rl.model_preparation.vlm_safetensors` entrypoint. Cosmos-RL packages an
isolated Framework converter environment pinned by the Framework repository's
`uv.lock`; Cosmos Framework uses its native environment behind the same
entrypoint. Its baked `/opt/tao/framework-converter-runtime.json` must attest
`validation_mode=imported_converter_module`, which proves that the converter
and its transitive dependencies imported in the isolated interpreter during
the image build. Reject an existing SQSH before submit if either implementation
or this import-level attestation is absent.
Do not ask a Cosmos-RL user for a donor checkpoint, a second backend image, or
a second SQSH. The conversion manifest proves the common source model and
fingerprints the prepared representation. Framework DCP evaluation uses the
native exact-key VLM exporter; PEFT adapters are reconstructed and merged
before shared evaluation.

For every Framework evaluate, inference, or inference-microservice request,
the skill runs `scripts/framework_checkpoint_action.py plan` and then
`prepare` before constructing the action command. The helper skips native HF
safetensors inputs, but a Framework DCP is exported with the Framework
repository's `cosmos_framework.scripts.export_vlm_dcp` entry point. The
default output name includes the DCP-metadata and saved-config fingerprint.
Reuse requires a matching export manifest, checkpoint record, DCP metadata
hash, config path/hash, base-model identity/revision, and complete indexed
weights. The returned `action_model_path` replaces the model field in the
evaluate/inference request. Export failure is an action failure, not a reason
to load an older export or checkpoint.

## SLURM preflight

Validate SSH configuration without reading key contents, scheduler reachability,
partition/account/QOS/reservation, shared paths, free space, Pyxis, Enroot,
the selected or derived SQSH, mount mapping, work directory, and non-root Python imports.
On a short allocation validate the allocated GPU count/type/memory, driver,
CUDA, PyTorch CUDA, architecture, NCCL initialization, decoder/library, and
the model/data paths through the container.

Jobs explicitly use Bash. Use one launcher task per node and preserve the
training child code. Record scheduler state/reason/exit independently. Requeue
is off unless separately validated. Scheduler `COMPLETED` never overrides a
nonzero child exit or missing/failed TAO terminal state.

Framework topology is shard degree equal to GPUs per node and replicate degree
equal to nodes. Cosmos-RL uses one controller on node zero and policy workers
with the declared shard/replica topology. Multi-node asynchronous checkpointing
is rejected. The environment records deterministic seed/hash settings, NCCL
diagnostics/error handling, CUDA allocator settings, driver capabilities, and
resource limits without credentials.

## Decoder and cache recovery

Framework uses its native CUDA TorchCodec path. On A100, the pinned Cosmos-RL
image maps the qwen-vl-utils torchvision name to its sparse software System-
PyAV reader. The canonical image build downloads the exact official PyAV wheel,
verifies its SHA256, and requires `h264 -> h264` and `hevc -> hevc`; generic
codec names resolving to CUVID are rejected because A100 has no NVDEC engine.
Positive DataLoader worker counts require the runtime's `spawn` context, a
picklable cache, and worker initialization that registers the reader without
creating or selecting a CUDA context. Worker count zero requires prefetch to be
absent or null. Cosmos-RL defaults to direct on-demand processing and starts
model training without a dataset-cache prewarm phase. Prewarming is an explicit
opt-in; when selected for conversation-style data, separate train and validation
cache keys combine dataset, model, and processor fingerprints, and completeness
manifests plus every entry are validated before training. An explicitly
selected NVDEC/PyNvVideoCodec task-aware run uses the packaged
`cosmos_rl.utils.video_override_artifacts` builder and validator; the A100
software path decodes the original paired media directly and does not create a
GPU override artifact.

Do not change decoder semantics silently. The generated configuration must
record the selected software or hardware contract, and a decoder, cache, or
media failure is a failed gate.

## Failure classes

- request/input: missing runtime model, revision, dataset, media, path, or
  scheduler value;
- source/provenance (`source-build` only): dirty checkout, mismatched image
  source, or host source import;
- runtime image: inaccessible explicit SQSH, failed packaged-image conversion,
  or unreadable derived SQSH;
- platform: SSH, scheduler, Pyxis/Enroot, mount, permission, GPU, CUDA, NCCL,
  decoder, or storage failure;
- model/data: incompatible checkpoint keys/config, missing media, duplicate or
  overlapping records, incompatible structural family, or missing task metadata;
- experiment parity: model, dataset, optimization, prompt/preprocessing, or
  evaluator mismatch;
- runtime: OOM, distributed timeout, decoder error, checkpoint failure,
  nonfinite metric, child nonzero, or missing terminal status.

A source/image defect is fixed only in its owning repository, with a test and a
new clean build/SQSH. Never patch an existing container or rely on a generated
script as the sole fix.
