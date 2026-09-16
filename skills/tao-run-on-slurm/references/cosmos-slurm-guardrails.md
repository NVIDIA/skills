# Cosmos SLURM guardrails

When `tao-finetune-cosmos-reason` resolves a backend, read that backend
contract before rendering the SLURM command. Cosmos jobs require a prebuilt,
compute-node-readable `.sqsh`; convert the selected image before the GPU
allocation and do not substitute a direct registry reference in the training
job. Use the model planner's explicit post-review `materialize` verb to write
generated TOMLs and any merged/smoke manifests atomically on user-supplied
shared storage, then verify their checksums before rendering or submitting the
job. Run the expensive remote model/dataset inspection once during `plan`, save
it with an explicit controller-side `--plan-artifact`, and reuse that sealed
artifact for `preflight`, `materialize`, and `render-slurm`; those later verbs
must not rebuild the plan. Planning and preflight must remain read-only with
respect to the target compute frame and must not require the
shared filesystem to be mounted on the SSH launch host. Stage the Framework
status bridge in the repository-derived image, not as an ad hoc source patch.
Evaluation consumes the model skill's validated spec-bundle and its declarative
`execution` lifecycle. The Cosmos producer owns runtime attestation, evaluator
selection, and metric extraction; this platform owns the standard SBATCH/Pyxis
envelope, torchrun mapping, mounts, timeout, and child exit preservation. Do
not re-encode model semantics in a Cosmos-only SLURM renderer or improvise a
command outside the sealed bundle.

For a single-node exclusive job, request the user's CPU count, resolve granted
`NumCPUs` inside the allocation, and pass it to the training `srun`. Record
requested, allocated, and step values; never infer multi-node CPUs this way.
The platform validates failed or comment-quarantined nodes against current
eligible-node inventory and owns `#SBATCH --exclude`; the model planner seals
the validated set in its fresh retry plan. Never edit a rendered directive.

- Cosmos Framework: one Pyxis task/container per node; inside each task set
  `NODE_RANK=$SLURM_PROCID` and launch native torchrun with node count,
  GPUs-per-node, master address/port. Set Framework HSDP shard degree to GPUs
  per node and replicate degree to nodes. Use `--no-container-mount-home`,
  `/workspace/.venv/bin/python`, `ulimit -n 65536`, and disable asynchronous DCP
  for multi-node shared-SLURM runs.
- Cosmos-RL: single-node uses its normal CLI. Validated policy-only multi-node
  SFT starts the controller only on node zero and one policy replica on every
  node; its spec uses global GPU count as shard size and one replicate. Enable
  the decoder explicitly. On A100, use the image's checksum-pinned official
  PyAV wheel and sparse software System-PyAV reader; require generic H.264/HEVC
  names to resolve to software decoders and ensure spawned workers register the
  reader without creating a CUDA context. An explicit PyNvVideoCodec/NVDEC path
  retains the video driver capability and decoder-artifact gates.

For both backends, preserve the real `srun`/torchrun/policy exit code through
cleanup and any requeue footer. A zero exit from a later shell command must not
mask a failed training process. Treat SLURM `COMPLETED` as provisional until
the Cosmos structured status contains terminal `SUCCESS`; then extract train
and epoch validation metrics with the model skill's packaged helper.
