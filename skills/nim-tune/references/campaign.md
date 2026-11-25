# Campaign file

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
-->

`scripts/run_campaign.py` reads one JSON file that describes a whole measured series: where to
deploy, what to measure, the latency target, and the candidates. Paths are relative to the file.
Keep the file in the run folder and run from there with `--out .`.

## Fields

| field | required | meaning |
|---|---|---|
| `name` | yes | Campaign id; labels every NIMService or container it creates (`nim-tune/campaign`) |
| `target` | no | `nim-operator` (default) or `docker` |
| `context`, `namespace` | `nim-operator` | The kube context and namespace the user authorized |
| `node` | no | Node name; candidates are pinned to it (`nodeSelector`) and one scheduled elsewhere fails; GPU allocation is recorded at the start and end |
| `gpu_budget` | yes | Total GPUs the user authorized; a candidate above it is rejected |
| `base_manifest` | `nim-operator` | A working NIMService (YAML or JSON): image, pull secrets, auth secret, model env, storage, probes, node selector. Secrets are referenced, never created. Any other kind is refused |
| `client_manifest` | `nim-operator` | CPU-only pod on the same node with `aiperf==0.13.0` installed, kept alive with `sleep infinity`; reused if it already runs. Must define one `v1` Pod with `metadata.name`, and no namespace or the campaign's. `--dry-run` checks its create with a server-side dry run when no pod of that name exists |
| `docker` | `docker` | The Docker target, below |
| `name_prefix` | no | NIMService or container name is `PREFIX-ID` (default `nimtune`); include the user's name |
| `model` | for the result file | The NIM model as nim-select names it (`facts.handoff.model`), not the served name. Read by `scripts/write_recommended.py` only |
| `model_profile` | for the result file | The profile id the NIM was measured on (`facts.handoff.model_profile`). Required together with the pin: `NIM_MODEL_PROFILE` in `docker.env` (or `spec.env` of the base manifest) with the same id, so the measured NIM really ran that profile; the runner does not use `model_profile` itself. `write_recommended.py` refuses an unpinned NIM (`profile_not_pinned`) and never guesses it |
| `image_digest` | for the result file | `sha256:...` of the measured image, optional. It only confirms the digest the measured replicas show (Docker `RepoDigests` of the measured container's image; the NIM container's `imageID` on the NIM Operator): a different one is refused (`image_digest_conflict`), and it never stands in when they show none (`image_digest_incomplete`) |
| `gpu_name` | for the result file | The GPU model the campaign ran on, as `nvidia-smi --query-gpu=name --format=csv,noheader` prints it (or nim-select `facts.handoff.gpu_name`), for example `NVIDIA H100 NVL`. A non-empty string. `write_recommended.py` copies it into the result so nim-deploy can refuse a result tuned on other hardware. When absent the result has `gpu_name: null` and a warning, and nim-deploy asks the user before using it |
| `baseline` | yes | Candidate id of the untuned or current-best NIM |
| `workload.model`, `workload.tokenizer` | yes | Served model name and the Hugging Face tokenizer id |
| `workload.isl`, `workload.osl` | no | Input and output tokens (default 1024 and 256) |
| `workload.concurrency` | no | Fixed loads measured for every candidate (default 8, 32, 128) |
| `workload.repeats` | no | Repeats per load, at least 3 (default 3); seed is 100 x load + repeat |
| `workload.dataset` | no | `{"file": PATH, "type": "multi_turn"}`: a real-text AIPerf JSONL written by `scripts/prep_dataset.py` (its `facts.campaign_field` gives the absolute path); a relative path is relative to the campaign file. The runner copies it into the client pod once (checked by SHA-256) and records the hash in every manifest, so runs on other data are never compared. Required for speculative decoding |
| `workload.extra_inputs` | no | Passed to AIPerf `--extra-inputs`. Default `ignore_eos:true` for synthetic prompts (every request returns exactly `osl` tokens); `temperature:0.0` with a dataset (greedy, so every candidate generates the same tokens and each turn keeps its real length) |
| `workload.dataset_args` | no | Extra AIPerf dataset arguments; without `workload.dataset` they replace the synthetic `--isl/--osl` (check names with `aiperf profile --help`). Options the runner sets on every run (`--model`, `--tokenizer`, `--url`, `--endpoint-type`, `--streaming`, `--concurrency`, `--request-count`, `--warmup-request-count`, `--random-seed`, `--artifact-dir`, `--output-artifact-dir`, `--profile-export-prefix`, `--profile-export-file`, and the aliases `-m`, `--model-names`, `-u`, `--num-requests`, `--num-warmup-requests`, `--no-streaming`, `--endpoint`/`--custom-endpoint`, which would send requests to another path than the smoke test checked, and `--num-profile-runs`, which the plan's run count would not include) are refused (`bad_dataset_args`) |
| `workload.cache_salt` | no | A fresh `cache_salt` per run (default true); dropped automatically if the server rejects it |
| `sla.metric`, `sla.max_ms` | no | Latency target, e.g. `ttft_p99` at most 1000 ms. A load passes only when its worst repeat meets it and no request failed |
| `sla.search` | no | `{start, max, resolution_pct}`: per candidate, find the highest concurrency that meets the target (defaults 8, 512, 10) |
| `candidates[]` | yes | `id` (`[a-z0-9-]`), `gpus` per replica, `replicas`, `args` (engine arguments, replaces the base list), `env` (added or overriding variables), `why` (the hypothesis and evidence), `engagement_pattern` (a regex for the startup log, required when only `env` changes something the launch line cannot show, such as a `NIM_*` variable) |

## Docker

With `"target": "docker"`, each candidate is one container on one GPU host, and AIPerf runs in a
client container (`PREFIX-aiperf`, `python:3.12-slim` with `aiperf`, `sleep infinity`) on a Docker
network (`PREFIX-net`) with them, so it reaches each NIM by name on port 8000. Every candidate has
one replica; its `gpus` are the first of `docker.gpus`. The candidate id `aiperf` is refused in Docker
mode: `PREFIX-aiperf` is the client's name.

| `docker` field | required | meaning |
|---|---|---|
| `image` | yes | The NIM image with its tag |
| `gpus` | yes | GPU indices the user authorized, e.g. `"0,1,2,3"` (`--gpus "device=..."`) |
| `env` | no | Non-secret variables, e.g. `{"NIM_MODEL_PATH": "hf://Qwen/Qwen2.5-7B-Instruct", "NIM_SERVED_MODEL_NAME": "qwen7b"}` |
| `env_from_host` | no | Secret names passed as `-e NAME`, so the value comes from the host's environment, e.g. `["NGC_API_KEY", "HF_TOKEN"]` |
| `env_file` | no | An env file on the host (`--env-file`), for secrets the ssh session's environment lacks; `~/` is the host's home. The dry run checks that it exists |
| `client_env_from_host` | no | Names passed to the client container (default `["HF_TOKEN"]`, for gated tokenizers) |
| `cache_dir` | no | Host folder mounted at `/opt/nim/.cache`, so each candidate reuses the downloaded model; `~/` is the host's home, and the dry run checks it exists |
| `shm_size`, `user`, `extra_run_args` | no | `--shm-size` (default `16g`), `-u` (e.g. `"1000:0"` for cache permissions), more `docker run` arguments |
| `ssh`, `sudo` | no | `user@host` to run every Docker command on a remote GPU host over ssh; `sudo` when that host needs it |
| `client_image`, `aiperf_version`, `network` | no | Defaults `python:3.12-slim`, `0.13.0`, `PREFIX-net`. `aiperf_version` must be one exact PEP 440 version that pip accepts in `aiperf==VERSION`, e.g. `0.13.0`, `v0.13.0`, `0.14.0rc1`, `0.14.0-rc.1`, `0.13.0.post1`, `0.13.0+local` (pre-releases spelled `a`, `b`, `c`, `rc` or `pre`; ASCII, no spaces, no `*` wildcard); other forms are refused before anything is started. It is checked and passed to `pip install` as an argument, never as shell source |

```json
{
  "name": "qwen7b-h100-docker",
  "target": "docker",
  "gpu_budget": 2,
  "name_prefix": "nimtune-alice",
  "baseline": "before",
  "docker": {
    "image": "nvcr.io/nim/nvidia/vllm-model-free-nim:2.1.4",
    "gpus": "2,3",
    "ssh": "alice@gpu-host-01",
    "env": {"NIM_MODEL_PATH": "hf://Qwen/Qwen2.5-7B-Instruct", "NIM_SERVED_MODEL_NAME": "qwen7b"},
    "env_from_host": ["NGC_API_KEY", "HF_TOKEN"],
    "cache_dir": "~/nim-cache",
    "user": "1000:0"
  },
  "workload": {"model": "qwen7b", "tokenizer": "Qwen/Qwen2.5-7B-Instruct",
               "concurrency": [8, 32], "repeats": 3},
  "candidates": [
    {"id": "before", "gpus": 2},
    {"id": "mnt4096", "gpus": 2, "args": ["--max-num-batched-tokens", "4096"]}
  ]
}
```

## Example

Phase 1 runtime tuning of Qwen2.5-7B on NIM 2.1.4, keeping the untuned NIM's profile (TP4 on 4
GPUs). Every candidate changes one engine setting:

```json
{
  "name": "qwen7b-h200-rt1",
  "context": "my-cluster",
  "namespace": "nim-tuning",
  "node": "gpu-node-01",
  "gpu_budget": 4,
  "base_manifest": "nimservice.base.yaml",
  "client_manifest": "aiperf-client.yaml",
  "name_prefix": "nimtune-alice",
  "baseline": "before",
  "workload": {"model": "qwen7b", "tokenizer": "Qwen/Qwen2.5-7B-Instruct",
               "isl": 1024, "osl": 256, "concurrency": [8, 32, 128], "repeats": 3},
  "sla": {"metric": "ttft_p99", "max_ms": 1000,
          "search": {"start": 8, "max": 512, "resolution_pct": 10}},
  "candidates": [
    {"id": "before", "gpus": 4, "args": [], "why": "untuned NIM, its own profile"},
    {"id": "mnt4096", "gpus": 4, "args": ["--max-num-batched-tokens", "4096"],
     "why": "smaller step budget: less prefill per step, lower ITL under load"},
    {"id": "mnt16384", "gpus": 4, "args": ["--max-num-batched-tokens", "16384"],
     "why": "larger step budget: prefill drains faster, lower TTFT under load"},
    {"id": "seqs64", "gpus": 4, "args": ["--max-num-seqs", "64"],
     "why": "cap the batch near the target load to protect latency"},
    {"id": "kvfp8", "gpus": 4, "args": ["--kv-cache-dtype", "fp8"],
     "why": "halve KV bandwidth in decode; needs an accuracy check"}
  ]
}
```

## What a run writes

| path | holds |
|---|---|
| `OUT/campaign_state.json` | Status per candidate (`measured`, `failed` with `kind` config or intermittent, `pruned` or `rejected` before deploy, `stopped`), `needs_accuracy_check`, attempts, engagement, `failed_runs` and `overloaded_runs` (a measured overload is kept in both), `engine_flags_source` (`file` or `probed`), `repairs` (repair passes used, at most 2), `unrepaired` (runs that still do not count after them), counters (`after` is empty when a replica was replaced after the last run), counters, SLA search, stop condition, active time. `cleanup_pending` on a candidate means its removal failed: a rerun removes it first, before any candidate deploys, whatever `--only` selects and even when a stop gate ends the run at once; its results are kept. It holds the resource's name, and `cleanup_uid` which resource that was: `deployed_uid`, recorded by this run's deploy and saved before it waits for readiness (the NIMService's `metadata.uid` from the API server's answer to its create, or to the replace when this campaign's NIMService was already there: one without this campaign's label is refused (`foreign_resource`); the replace sets the spec to exactly the approved one and carries the uid and resourceVersion read, so the API server refuses one replaced since (uid precondition, version Conflict) and never configures it; `--dry-run` rehearses the same operation server-side (the create, or this replace), so a missing create or update permission shows before approval; the container Id `docker run -d` printed, or that of the container carrying that run's `nim-tune/create-nonce` label; never read back by name later), `""` when the deploy made nothing, `null` when none was recorded: the removal and a rerun remove only that resource, never a same-named one made since. If `name_prefix` changed since, the rerun never removes the new name: it stops with `cleanup_pending` while that resource still exists, and clears the mark once it is gone. If the name now holds another resource (another run's, possibly another user's with the same campaign name), it is never removed: the mark is cleared once the recorded one is gone and no pod labelled `app=NAME` runs; otherwise, or when the recorded container still exists under another name, the run stops with `cleanup_pending`. A same-named resource this campaign did not create is never removed and never taken for a removal: the run stops with `foreign_resource` and the mark stays. A mark without `cleanup_uid` (an earlier runner, `true` from before names were recorded, or `null`) is cleared only when nothing has that name and no replica runs; otherwise the run stops with `cleanup_pending`: check the resource's `nim-tune/campaign` label and who created it, remove it if it is this campaign's and the user agrees, then delete `cleanup_pending`, `cleanup_target` and `cleanup_uid` of that candidate in the state. `cleanup_target` records the context/namespace, or the Docker host and its daemon's ID (`docker info`; `DOCKER_HOST` or the docker context picks the daemon), the resource was left on; a rerun against another target or daemon stops with `cleanup_pending` instead of clearing it. A Docker mark without a daemon ID (an earlier runner, or the daemon did not answer) is never cleared by the runner: check every daemon the campaign may have used, remove the container once the user agrees, then delete `cleanup_pending`, `cleanup_target` and `cleanup_uid` of that candidate in the state; a daemon that does not answer on a resume stops the run with `docker_failed`. `image_id` records the image a candidate's runs were measured on (Docker's image ID, or the digest of the NIM container's `imageID`); when a redeploy runs another image, or the state has none, its saved run folders are set aside whole as `cC-rR.superseded` (`.superseded-2`, … when taken; never read as runs, never deleted) and measured again with the search, and records of replicas no longer serving become `*.pod.json.superseded`. A `rejected` candidate records `flags_sha256`, the engine help that rejected it; when the help changes it is checked again; without any help it stays rejected |
| `OUT/campaign.log` | Progress log |
| `OUT/engine_flags.txt` | `vllm serve --help=all` from the baseline pod (or `--engine-flags FILE`); later candidates are checked against it. A rerun or its `--dry-run` without the `--engine-flags` file earlier runs used, or after a probed copy was deleted, is refused with `engine_help_missing` before anything deploys: pass the same file, restore the probed copy, or use a new `--out`. A copy whose origin the state does not record is set aside as `engine_flags.unrecorded.txt` (or `engine_flags.unrecorded.N.txt` when that name is taken; never deleted or overwritten) and the help is probed again. `--engine-flags OUT/engine_flags.txt` is used as is, never rewritten |
| `OUT/ID/deploy.json` | The exact NIMService or `docker run` arguments used (no secret values) |
| `OUT/ID/POD.startup.log`, `POD.failure.log` | Engine logs, kept before removal |
| `OUT/ID/POD.pod.json` | The pod or container spec the engagement check read. For Docker it keeps only the variables named in `docker.env` or the candidate's `env` (never a credential-named one), and so does the check's `effective_nim_env` |
| `OUT/ID/cC-rR/aiperf.log`, `profile_export_aiperf.csv` | AIPerf console log and CSV (when read back), moved there from `cC-rR/.attempt/` only once the attempt counts or shows failed requests; both scripts read the CSV when it holds rows, else the console (CSV missing, empty or header-only) |
| `OUT/ID/manifest.json` | Workload, AIPerf version, node, client and candidate; `compare_aiperf.py` refuses to compare cases whose measurement setup differs, including a case whose only evidence is an overload |
| `OUT/final/before_after.md` | The before-and-after table, best case per load and the highest load within the target |

A rerun with the same `--out` resumes: finished runs are skipped, a candidate that failed on its
configuration is not retried, and active time keeps counting toward `--max-hours`.

## Running on a shared cluster

Lessons from the first campaigns on a busy shared cluster (about 3,400 pods in the namespace, October 2026):

- **Slow API.** Listing every pod in a busy namespace took 1 to 2 minutes and could time out. The
  runner lists only the candidate's pods (`-l app=NAME`, the NIM Operator's label), which takes
  about 1 second. Expect the first `kubectl` calls of a run to be slow anyway.
- **An empty pod list never passes.** If the runner sees fewer pods than replicas after the
  Deployment reports ready, it fails the candidate as intermittent (retried once) rather than
  skipping the engagement check.
- **Engagement evidence is the current container's.** For Docker the live log is read by
  container id, only since the container's latest start, so a restart or a same-named replacement
  never lends it an earlier launch line; if the container restarted while its log was read, or its
  name no longer resolves to the same container at the same start (renamed or replaced), the check
  stops with `target_restarted` (a container spec without `State.StartedAt` is refused with
  `no_target`); a live pod log is likewise refused with `target_restarted` when
  the pod's uid or the container's id or restart count changed while it was read. On Kubernetes the pod's
  `kubectl.kubernetes.io/default-container` annotation (it may name a regular, init or sidecar, or
  ephemeral container, as `kubectl logs` finds it; else, or when it names a missing container, the
  first container) picks the container for its spec, its status and the log. Container args are
  compared as Kubernetes delivers them, with `$(NAME)` expanded from the container's `env` values; a
  name declared more than once takes its last declaration. A reference Kubernetes fills from a value
  not readable here (`valueFrom`, a variable built from one, any name when the container has
  `envFrom`, or a service variable the kubelet injects such as `KUBERNETES_SERVICE_HOST`, or
  `NAME_SERVICE_HOST` unless `enableServiceLinks: false`) is unconfirmed, never engaged; it is a
  mismatch only when the engine runs another value. An escaped `$$(NAME)` or an unset name is
  literal text.
- **Start it once, supervised, from the campaign folder.** `nohup python3 SKILL_DIR/scripts/run_campaign.py
  campaign.json --out . --max-hours H --approved PLAN_ID --supervise >> campaign.nohup.log 2>&1 &`.
  Paths in the campaign file are relative to the file itself. Add `--keep-awake` on a laptop
  (`caffeinate` on macOS, `systemd-inhibit` on Linux). `--supervise` is not part of the plan id.
- **Transient stops resume by themselves.** A lost VPN or API server ended a campaign with
  `kubectl_failed` after about 40 minutes of retries, and another ended with `client_lost`; each
  needed a person to rerun it. Under `--supervise` the runner waits until the target answers
  (`kubectl get --raw /version`, or `docker version`) and reruns the same approved command, up to
  `--max-resumes` times, and gives up with `target_unreachable` after `--max-wait-hours` (default 4)
  without an answer. A failed baseline, a refused plan, `resume_mismatch`, the same unexpected
  error twice in a row and an attempt killed by a signal (out of memory, or `kill`) stay final.
  A target that does not answer at the start is a setup problem (context, CLI, network): the
  supervisor stops at once with `target_unreachable` instead of waiting.
- **One runner per folder.** Two runners started on the same folder by mistake stopped each
  other's AIPerf in the shared client pod and wrote the same state files. A live run now holds
  `OUT/.run_campaign.lock` (process id, host, plan id); a second one is refused with
  `already_running`. The supervisor holds the lock between attempts. The claim is an OS file
  lock (`flock`): it is released when its process ends, so a crashed runner, a reboot or a
  reused process id never blocks a rerun, and two runners started at once cannot both win.
  A lock file written on another host (a shared folder) is never taken over.
- **Restart to change code.** A running campaign keeps the script version it started with. Stop
  it with `kill PID` (for a supervised run, the supervisor's PID: it stops its attempt, frees the
  folder and reports `stopped`) and rerun the same command: finished runs stay, a run cut off mid-benchmark
  is measured again (its AIPerf, still running in the client pod, is stopped first: one load
  generator at a time), and a NIMService labelled with this campaign is reused, not redeployed.
- **The client pod.** The runner removes the client pod only if it created it, and records that
  in `campaign_state.json` so a resumed run still removes it. It creates the pod with `kubectl
  create` and records the uid the API server returned for it; a pod of that name that appears first
  is reused and never counted as created. It removes it only while it is still
  the one it created (same pod uid, or same container id); a same-named replacement is left alone.
  Docker removes the client by its container id. A client pod that already existed
  (or one created by a campaign started before that record existed) is left running; ask the
  user before deleting it.
- **Keep the client alive for the whole campaign.** Its command must be `sleep infinity`, never a
  fixed sleep: a 6-hour sleep ended in the middle of the first runtime campaign. The runner checks
  the client before each smoke test and benchmark; when it has stopped, the runner removes the
  candidate being measured, leaves it pending (not failed) and ends with `client_lost`. Delete the
  finished client pod (ask first if the campaign did not create it), then rerun the same command:
  finished runs are kept and only the interrupted runs are measured again.
- **A dropped exec stream.** The API server reset `kubectl exec` streams several times a day, often
  just as AIPerf finished. The runner therefore starts AIPerf detached in the client pod and polls
  for its exit code and console output, so a dropped stream never loses or repeats a run.
- **A terminating client still reads as Running.** A campaign that ends removes the client it
  created; deletion takes a while, and a campaign started right after must not reuse that pod.
  The runner treats a pod with a deletion timestamp as gone and waits for it before creating one.
- **Rerunning repairs gaps.** A rerun measures again any run that failed twice in a measured
  candidate, or any saved run of it that does not count (no error rate, no throughput or target
  metric, export gone); it is redeployed for that, and runs that count are reused. It re-checks the
  stop gates instead of keeping the last run's stop condition. A load where every repeat failed
  requests is a measured overload: while its exports still show it, it is kept, never redeployed for
  and never benchmarked again when other runs are repaired, nor by the SLA search (a saved run with
  failed requests is a miss there). A candidate gets at most 2 repair passes; after that the gap
  stays, reported as `candidate_unrepaired` (`unrepaired` in the state): the comparison leaves a
  tuned case out so the others can still be compared; for the untuned NIM a stray failed-request
  run leaves the comparison provisional, and a run without a metric fails it as `baseline_missing`
  (measure again in a new folder without the loads it cannot complete). `--max-candidates` counts
  candidates, not repair passes: a rerun at the cap still repairs a measured one and deploys no new
  candidate; once a candidate is admitted, the cap is not checked again at each load or search
  point (the time and failure gates are).
- **A kubectl error is not an answer.** Reads are retried for about a minute; only NotFound means
  gone. If the API server stays unreachable the campaign stops with `kubectl_failed`, leaves the
  candidate running (listed in `left_running`) and a rerun picks it up.
- **The baseline comes first and must succeed.** If the untuned NIM is not measured, the campaign
  stops there: nothing else could be compared. A failed, stopped or partly measured candidate is
  left out of the comparison, so it can never be recommended. A load where requests failed
  (overload, common for an untuned NIM at the top load) is listed in `overloaded_runs` and is not
  "missing": the case stays in the comparison, which reports that run as `run_had_request_errors`; a
  comparison that is complete apart from such overloads ends the run ok (`decide`), listing them in
  `facts.compare.notes` and `facts.compare.rejected_runs`.
- **A run counts only when every request succeeded.** A run with failed requests, or with an error
  rate row that is present but unreadable, is not counted and is measured again on a rerun. AIPerf
  writes that row on every run (0.00 when all requests succeeded), so a run whose export lacks it
  is not counted either; `compare_aiperf.py` lists it in `runs_without_error_rate`. A run must also have measured output throughput (`otps_avg`) and, with a target, the target
  metric (the metrics `compare_aiperf.py` needs from every counted repeat: one rule for both); a
  saved run without one is set aside as `*.incomplete` and measured again. Any other metric, such as
  ITL (AIPerf has none for 1-token outputs), never stops a run from counting.
- **An incomplete comparison is provisional.** A load meets the target only when every counted
  repeat measured the target metric. Throughput ranks every case, so it must be in every counted repeat too. Any other latency metric
  measured in fewer than `--min-repeats` repeats gets no verdict and is listed in `partly_measured`;
  it never blocks. ITL p99 settles a throughput tie only when every tied case measured it in every
  counted repeat; otherwise the tie is not settled (`recommendation.unsettled_tie`, error
  `tie_unsettled`) and the recommendation stays provisional. With too few repeats, a case without counted runs at a
  requested load (unless its runs there all had failed requests), or a run without an error rate,
  `recommendation.provisional` is true and `provisional_reasons` lists the gaps: measure them, never
  promote it. A load where every run of a case had failed requests (at least `--min-repeats` runs, which `run_campaign.py` sets to `workload.repeats`, unless it
  is an SLA-search point) is a measured overload: the comparison is still
  complete (`ok`, listed in `errors` and `rejected_runs`). At a `--loads` load, a failed run next to
  counted runs is not, and is repeated first. An SLA-search point (a load not in `--loads`, only with `--sla`;
  without both no load is one, so pass the campaign's loads and target when comparing by hand) stops at its first
  repeat with
  failed requests, so any failed repeat there is that point's measured miss, even after counted
  repeats. `sla_boundary.CASE.unmeasured` lists loads above the boundary that lack the metric in
  some repeat, or that have run folders but no counted run and no measured overload (neither passing
  nor failing); a case whose every load failed requests is listed too, with no passing load. A
  percentage change too large to state is `change_pct: null`. When no baseline repeat counts because
  none has a readable error rate, `baseline_missing` names them (`facts.runs_without_error_rate`). An
  export that cannot be read is an unfinished run (`run_unfinished`), not a crash. `picks[].best.otps_per_gpu` ranks cases with different
  GPU counts by tokens/s per GPU.
- **Approve the exact plan.** `--dry-run` prints `plan_id`, a hash of the campaign file and of
  every local file it deploys or sends (base and client manifests, dataset), and of the options
  that decide what runs: `--only`, `--max-candidates`, `--max-hours`, `--max-consecutive-failures`,
  `--ready-timeout`, and the bytes of the `--engine-flags` file (`plan_options.engine_flags_sha256`).
  A live run needs `--approved PLAN_ID` and the same options as the dry run;
  if any of them changed after the review, it is refused (`not_approved`).
  The run reads each file once, so the bytes it hashed are the bytes it deploys. Refused before
  any GPU time too: wrong field types, `sla.search` values that are not whole numbers with
  `1 <= start <= max` and `0 < resolution_pct < 100`, an `env_from_host` entry that is not a
  variable name (`NAME=value` is `literal_secret`), an invalid
  `engagement_pattern` regex, a `docker.ssh` value that is not `user@host` or starts with `-`, `docker.extra_run_args`
  that set what the runner owns (`--gpus`, `--rm`, `--network`, `--name`, `--label`, `-e`, `--device`, `--cap-add`, `--privileged`, also in the attached form
  `-eNAME=x`) or carry a credential (`literal_secret`), and
  `docker.sudo` with `env_from_host` (`sudo` resets the environment, so the NIM would start without
  its key): with `sudo`, put secrets in `docker.env_file`.
- **Containers carry their settings.** Each Docker candidate is labelled with a hash of its exact
  `docker run` line; a labelled container from an older run with other settings is recreated, never
  measured. AIPerf's process id is kept per campaign, so two campaigns sharing a client never stop
  each other's runs.
- **Secrets by reference only.** A candidate `env` entry, `docker.env` entry or argument that
  looks like a credential (`*_KEY`, `*_TOKEN`, `--api-key X`) is refused (`literal_secret`): it
  would sit in the NIMService spec or on a `docker run` command line. So is `--api-key` or a
  credential header (`-H`/`--header` with `Authorization`, a cookie or a credential-named header,
  also inside a comma-separated list or a JSON object of headers) in `workload.dataset_args`.
  For a credential the NIM needs, use a `valueFrom` secret in the base manifest,
  `docker.env_from_host` or `docker.env_file`. The client needs none: the runner benchmarks each
  candidate directly at `http://NAME:8000` inside the cluster or Docker network, not through a
  gateway, and has no option to send a client credential.
  Everything the runner writes
  (results, logs, state, its output) also passes a scrubber, before and after JSON encoding, that
  masks the values of
  the well-known credential variables (`NGC_API_KEY`, `NVIDIA_API_KEY`, `HF_TOKEN`, ...) and of the
  ones the campaign passes from the host (`env_from_host`, `client_env_from_host`), read by name
  (the environment is never listed or copied), known token shapes and secret flags. A named credential under 6 characters is masked wherever no ASCII letter or digit touches it and no "." joins it to a digit (a version); the last character of an escape before it (a JSON escape such as `\n` or `\u00e9`, a URL escape such as `%20`, a terminal escape sequence, raw or written as `\x1b`/`\033`) does not count as touching it.
- **A rerun only resumes the same campaign.** Each candidate's fingerprint (its settings, the
  workload and the deploy spec) is kept with its results; if it changed, the rerun stops with
  `resume_mismatch` instead of mixing old and new results. Use a new folder for a changed campaign.
- **Unexpected errors.** `kubectl exec` output can end mid-character; the runner decodes it
  leniently. Any other unexpected error ends with a `crashed` JSON result (never a traceback)
  that lists what is still running in `left_running`, including a candidate or client that could
  not be removed; rerun the same command to resume. A failed AIPerf attempt's log and export are
  kept as `aiperf.attempt1.log` and `profile_export_aiperf.attempt1.csv` next to the retry. A run
  with failed requests keeps its export, so `compare_aiperf.py` reads the overload from it. An
  attempt during which a replica restarted keeps its console only as `aiperf.restarted.log`. An attempt that does not count for another reason (no error rate, no throughput or target metric) is set aside as `*.incomplete`, so a rerun measures it again rather than reading its console. Each attempt is written into `cC-rR/.attempt/` first; a runner stopped before the attempt is judged leaves nothing under the names a rerun reuses.
- **Progress.** `campaign.log` shows each candidate, readiness, every run and the search result.
  `campaign_state.json` is written once the client is ready, then after every engagement check,
  load and candidate.

## The result file

`scripts/write_recommended.py campaign.json --out .` writes `final/recommended_config.json` from the
final campaign (the untuned NIM and the winner, back to back). The winner is the case
`compare_aiperf.py` recommends; it is promoted only when it is `better` than the untuned NIM on the
target metric or tokens/s, never `worse` on the target metric, fully measured, engaged, and
accuracy-checked when `needs_accuracy_check`. Otherwise `baseline_won` is true and `args` is `[]`;
a baseline win is written only when the untuned NIM itself meets the target at some measured load,
else the writer refuses with `no_recommendation` (as it does when no case meets the target). `env` always describes what was measured: the
shared base env (`docker.env`; on the NIM Operator the whole `spec.env` that was deployed) plus the
winner's own `env`; for a baseline win that is the base env alone, so settings such as
`NIM_MAX_MODEL_LEN` are not dropped. `baseline_won` with a non-empty `env` still means no tuned
change. Secrets stay names in `env_from_host`, never values. Image, env and secret names come from what was deployed (`docker` block, or the saved
`ID/deploy.json` on the NIM Operator). `gpu_name` comes from `campaign.json` (null with a warning when absent). It is checked against `recommended-config.schema.json`
before it is written, and refuses with a fix when the model, profile or image digest is unknown
(`--allow-unpinned-image` writes a null digest). The digest comes only from the measured replicas: campaign `image_digest` or a digest in the image reference only confirms it (`image_digest_conflict` when it differs) and never replaces it (`image_digest_incomplete` when no replica shows one). A measured digest counts only from the deployed
image's repository: Docker's `RepoDigests` of that repository, and on the NIM Operator the
`imageID` of the NIM container, never a sidecar's; that `imageID` must itself name the repository
(a measured NIM container whose `image` names another repository, with a pullable digest of it, is
refused, `image_mismatch`; an `imageID` of another name next to an `image` of the campaign's
repository, as containerd reports an image stored under two names, is a missing digest; the NIM container is found by kubectl's rules, an
annotated init or ephemeral container included), and a pod record that does not name its NIM container counts only
when one container status is of that repository. Every measured replica (pod or Docker container)
must show the digest: one without it while another has one is refused (`image_digest_incomplete`); measure again in a new `--out` folder. Every replica in the state's `engagement` needs its saved `ID/REPLICA.pod.json`; a missing one is refused (`image_digest_incomplete`): restore it, never delete it. Every measured case must record the same `image_id`, and the winner's saved records must still show it; otherwise the writer refuses with `image_changed` (the tag moved during the campaign: pin the image by digest and measure again in a new `--out` folder). A state without `image_id` is `measured_config_unrecorded`. A pod record whose `metadata` or `annotations` is not an object is `pod_record_malformed`. A run without `--dry-run` first removes an earlier
`final/recommended_config.json`, so a refusal never leaves a stale result for nim-deploy to read.

The runner records each candidate's deployed settings (`gpus`, `replicas`, `args`, `env` and, for
Docker, `image`, `docker.env`, `env_from_host`, `env_file`) and the `workload` as `measured` in `campaign_state.json`.
The writer refuses with `campaign_changed` when `campaign.json` differs from that record (it names
the settings), including the `gpus` or `replicas` of any measured case (they set tokens/s per
GPU), and with `measured_config_unrecorded` when the record of any measured case is missing. Restore the file, or
measure again in a new `--out` folder; the result is never written from edited settings.

A `docker.env_file` is refused (`env_file_unrecorded`): the writer never reads secret values, so it
cannot know which settings or secrets the file gave the measured NIM, and the result could differ
from what was measured. Move the non-secret settings into `docker.env` and the secret names into
`docker.env_from_host`, then measure again. With `docker.sudo` the names cannot be passed by
`env_from_host`; if the file holds only secrets (such as `HF_TOKEN`), pass
`--env-file-secrets-only`: the result is written with a warning that the file's contents are not
recorded and the deploy must get those secrets another way.

`recommended-config.schema.json` here is the producer's canonical copy. nim-deploy holds the
consumer copy, `references/tune-result.schema.json`. Keep the two files equal.
