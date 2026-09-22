# Failure analysis & retry — classify before resubmitting

When `status` reaches `ERROR`, **read the tail of the logs and classify the
failure before deciding to retry** — never blindly resubmit (a bad spec would
burn the whole retry budget on GPU-hours). This is a judgment you make from the
log; the signals below are guidance, not an exhaustive table.

- **Infrastructure → retriable** (a node / hardware / transport fault a resubmit
  may escape): SLURM `NODE_FAIL` / `BOOT_FAIL` / preemption; GPU `Xid` / `ECC` /
  "fell off the bus"; driver-too-old; genuine `NCCL` / InfiniBand / RDMA transport
  failures; `ImagePullBackOff`; pod `Evicted`. Resubmit under a NEW job-record
  with `--retry-of <id>` (mark the old one `ERROR --err-class ERR_INFRA`), up to
  **10**. SLURM's `#SBATCH --requeue` can handle node-fault/preemption at the
  scheduler level when the workload contract enables it; Cosmos training
  explicitly disables requeue unless it has been separately validated.
- **Program → never retry** (a code / data / config fault a resubmit will just
  repeat): `OOMKilled` / CUDA out-of-memory (reduce batch); CUDA **device-side
  assert** / illegal-memory-access / CUBLAS/CUDNN status (a code or label-range
  bug); missing files; Python tracebacks; timeout / deadline (raise the limit).
  Surface the cause to the user.

Two context calls to apply with judgment: a **device-side assert** is a program
bug *even when it cascades into a CUDA/NCCL error* — the assert is the root cause;
and a bare traceback that is *downstream* of a real `Xid` / node fault is a
symptom — retry the node fault. (These are exactly what a regex table gets wrong,
which is why classification lives here as agent judgment, not in a script.)

**In-turn** monitoring is the agent polling `status` / `logs` at the interval.
**Post-turn** (the turn may end before a long job finishes): background a poller
with the harness (`run_in_background` wait-loop, `CronCreate`, or `/loop`). The
poller auto-resubmits only the **unambiguous state-based** infra cases
(`NODE_FAIL` / `BOOT_FAIL` / `PREEMPTED`) and, for auto-cleanup backends (K8s
`ttlSecondsAfterFinished`, docker `--rm`), writes the **daemon-independent
terminal record** (`tao_job_record mark --state <terminal> --source poller`)
*before* the backend object is deleted; anything nuanced re-wakes the agent to
classify. Pollers are idempotent — on re-attach, just re-establish one.

## Ambiguous submission is reconciliation, not a retry

An empty `sbatch --parsable` response, SSH timeout, or lost connection does not
prove that submission failed. Every sbatch script must carry the unique TAO
job-record id as its exact SLURM job name. Before any resubmit, query both
`squeue` and `sacct` by user and exact job name on every configured login host,
allowing bounded accounting-propagation time:

- exactly one matching SLURM id: adopt it as the record's backend reference;
- more than one: stop, preserve the evidence, and cancel/disqualify duplicates
  before any job can share a result or checkpoint directory;
- none after the bounded reconciliation window: mark the attempt as an
  ambiguous infrastructure failure and submit once under a **new** job record
  with `--retry-of <id>` and fresh output paths.

Never issue repeated blind `sbatch` calls because stdout was blank. A scheduler
id is not considered accepted until it is persisted in the job record.

Before carrying a bad-node exclusion list into a retry, resolve the live node
inventory with `scontrol show nodes`. Remove retired or nonexistent names that
make `sbatch` reject the request, retain exclusions supported by failure
evidence, and record the validated exclusion set in the new attempt.

## Retry ownership boundary

The retry is a normal new `submit`, not a workflow-specific fifth platform
verb. Core launch owns classification, retry budget, the new job record, and
`retry_of`. The platform owns live backend reconciliation and infrastructure
inventory. The producer owns restoration of its sealed semantic request and
must rebase every writable path under the new record before resealing. Reuse
immutable inspection evidence; never reuse a prior result/checkpoint/cache
root, mutate an old plan, or patch a rendered platform artifact.
