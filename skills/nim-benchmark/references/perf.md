# Performance mode

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
-->

Read this for `scripts/nimbench.py perf`: what it sends, where AIPerf runs, what can go wrong.

## What a run does
For every load (concurrency) and every repeat, the script starts AIPerf 0.13.0 once, against the
endpoint, with a fixed seed, and reads the export file `profile_export_aiperf.json` it wrote.

| setting | value |
|---|---|
| input and output tokens | `--isl 1024`, `--osl 256`, output length fixed (`ignore_eos`) |
| loads | `--loads 8,32,128` |
| repeats | `--repeats 3`; fewer is refused |
| requests per run | 10 per concurrent slot, at least 40 |
| seed | 100 x load + repeat, so every run is repeatable and different from the others |
| cache salt | a new random value per run, so no run reuses another run's cached prompts |
| images | none for a text model; `--images-per-request N` for a VLM |

## Endpoint checks in Kubernetes
With `--client kubernetes --namespace NS --pod POD`, endpoint checks run through
`kubectl exec` in that client pod. Use a URL the pod can reach, such as an in-cluster Service
name. The host only needs access to the Kubernetes API. The pod needs `python3` and AIPerf.
Baseline endpoint checks and the model lists saved in both manifests use the same pod.
`--api-key-env` and `--baseline-api-key-env` name variables inside the pod for these checks,
just as they do for AIPerf; their values are not copied from the host.
The key is sent only to the endpoint URL itself: a redirect is followed without it, so it never
reaches another origin or a plain `http` URL.

`check` and the preconditions first look at the pod: it must exist (`client_pod_not_found`), be
`Running` (`client_pod_not_running`, with the phase) and `aiperf --version` must run in it
(`client_missing`: aiperf not installed, kubectl exec refused, or aiperf failing). With this client the `aiperf` fact is the
pod's result and `aiperf_in` says so; it is not about this machine. A request that fails inside the
pod is reported with its type and a short reason (`URLError: Name or service not known`); a pod
that cannot run python3 is `client_missing`, never a bad URL.

For a check without load, use the same client flags:

```bash
python3 scripts/nimbench.py check --endpoint http://my-nim.nim:8000/v1 \
  --model qwen/qwen3.5-122b-a10b --client kubernetes \
  --namespace nim --pod bench-client
```

Local and Docker clients keep host-side endpoint checks. Accuracy mode uses Docker;
its endpoint, baseline and judge URLs must remain reachable from that setup.

## Vision-language models
`--images-per-request N` sends N synthetic images with every request. `--image-width` and
`--image-height` default to 512, `--image-format` to `jpeg`. The images are random noise that AIPerf
makes on the fly (`--image-source noise`), a new one per request, so a server's image cache cannot
make a repeated image look fast.

- Image size and count change the result a lot: encoding the image dominates TTFT, and every image adds
  prompt tokens. Always say them with the numbers; two runs with different images are refused.
- Images use more memory per request. If the default loads 8, 32 and 128 fail on a VLM, lower them
  with `--loads`.
- Noise images measure speed, not quality. Use accuracy mode for what the model sees correctly.
- `--isl` still counts the text tokens only.

## Where AIPerf runs
| `--client` | what it does | needs |
|---|---|---|
| `docker` (default) | `docker run` of `--client-image` (default `python:3.12-slim`) with AIPerf 0.13.0 installed at start, on the host network, results in the run folder | Docker on this host; the endpoint reachable from it |
| `local` | runs `aiperf` on this host | `pip install aiperf==0.13.0` |
| `kubernetes` | `kubectl exec` into the pod you name with `--namespace` and `--pod`; AIPerf writes under `/tmp/nim-benchmark/<run folder hash>/` in the pod, so the baseline endpoint has its own folder; the script reads the export back | a running pod that has `aiperf` (and so `python3`); it never creates or deletes a pod |

For an endpoint that needs a key, `--api-key-env NAME` passes the name. The key is read where AIPerf
runs: from this host's environment for `docker` and `local`, from the pod's environment for
`kubernetes`. The value of `NAME` is masked in every file and log the skill writes.

Warning: the shell expands the key into AIPerf's command line (`--api-key "$NAME"`). Anyone who can
list processes where AIPerf runs (this host, or the pod) can read it with `ps` while the run lasts.
Use a short-lived key, or run AIPerf where no one else can list processes.

## Clean runs
- Before each run the script deletes the old `profile_export_aiperf.json`, in the run folder and in
  the pod, so a run that writes no export is refused instead of reusing an older one.
- On a timeout the script stops everything the run started: the Docker container, the local process
  group, or the `aiperf` of this run in the pod (a Python helper, so the pod needs no `pkill`). The helper signals the selected process and
  its workers, then waits up to five seconds to confirm they exited. If they remain alive, the
  error says so; stop it by hand. The script keeps that warning even if cleanup also fails,
  and starts no further repeat, load, or baseline benchmark until the client is stopped.

## What counts
- Local and Kubernetes clients must report AIPerf 0.13.0 before the plan is accepted. The export must report the same version.
- A failed raw-artifact cleanup fails the attempt; fix access to the named run folder or pod before retrying.
- A startup failure names the client executable and permissions to check. It never points to a missing or stale log. Output-file errors name the folder or client export access to repair.
- A failed pod export read reports its exit status and names the pod and export access to restore. It is not evidence that AIPerf wrote no export; that message is reserved for a confirmed missing file.
- A run counts only when the export file passes every check: a supported `schema_version` (1.x), a
  finished run, all requested requests answered, no failed request and the five metrics present.
- A run with failed requests is repeated once, then not counted (`run_had_request_errors`).
- A load needs every planned repeat counted (`--repeats`, at least 3). Otherwise it is listed as
  incomplete and gets no verdict, so both sides are always compared on the same number of repeats.
- AIPerf's raw per-request files (`inputs.json`, `profile_export.jsonl`) hold prompts and answers.
  The script deletes them after each run, also on a timeout, in the run folder and in the pod.

## Known AIPerf problems (0.13.0)
| problem | how the skill handles it |
|---|---|
| Repeated prompts inflate TTFT and throughput on servers with prefix caching | a new seed and a new cache salt per run |
| Throughput divided by the wrong time window in some runs | compare runs of one tool version only; the manifest records it |
| It can exit 0 with a short export, or exit 1 on a valid run | the exit code is never used |
| The server may reject `cache_salt` | the script drops it, says so in the manifest, and goes on |

## Errors
| code | exit | fix |
|---|---|---|
| `endpoint_unreachable`, `model_not_served` | 3 | check the URL, the port, the network path and the served names |
| `docker_missing`, `client_missing` | 3 | install Docker, `kubectl`, or `pip install aiperf==0.13.0`; with `--client kubernetes` the message says which pod, whether it is unreadable, has no python3 or has no aiperf |
| `client_version_mismatch` | 3 | install `aiperf==0.13.0` on this host for `--client local`, or inside the selected pod for `--client kubernetes`, then run `--dry-run` again |
| `client_pod_not_found` | 3 | check `--namespace` and `--pod` (`kubectl get pod -n NS`) |
| `client_pod_not_running` | 3 | wait for the pod to be Running, or name one that is |
| `tool_failed` | 4 | follow `errors[].fix`: stop an unstopped client before rerunning, restore folder or pod access after cleanup failures, or investigate a client timeout; read the saved log when the fix names one |
| `run_had_request_errors` | 0 | the run is not counted; lower the load or ask the endpoint owner |
| `bad_input`, `literal_secret` | 2 | fix the flag the message names; a key goes in a variable |
| `not_approved` | 5 | run `--dry-run` again, show the plan, use the plan id the user approves |
| `baseline_mismatch` | 5 | run the baseline again with this setup, or report without a verdict |
