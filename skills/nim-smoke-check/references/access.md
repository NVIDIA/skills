# Access: reaching a running NIM, read-only

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
-->

Read this when choosing `--url`, `--docker` or `--k8s`, when access fails, or when the user asks
what the script runs on their host or cluster. The checks need one base URL.
`scripts/nsc_access.py` works it out, changes nothing, and reports how in `facts.access`.

## Choose the target
| the NIM runs | pass | what the script does |
|---|---|---|
| anywhere a URL reaches | `--url URL` | uses the URL as given |
| on the NVIDIA API catalog | `--url https://integrate.api.nvidia.com/v1 --model ID --api-key-env NVIDIA_API_KEY` | the same; the key comes from the named variable (`NVIDIA_API_KEY` is also the default for this host) |
| in Docker on this machine | `--docker NAME` | reads the container's published port |
| as a NIM Operator NIMService | `--k8s nimservice/NAME --context CTX --namespace NS` | its own port-forward to one Ready pod |
| from the NIM Helm chart | `--k8s helm/RELEASE --context CTX --namespace NS` | the same |
| behind any Service | `--k8s svc/NAME --context CTX --namespace NS` | the same |
| in one pod | `--k8s pod/NAME --context CTX --namespace NS` | the same |
| behind an Ingress, a LoadBalancer or an API gateway | `--url URL` | tests the real path the users take |

Never guess the context or the namespace; ask. The script never uses the current context.

`facts.access`:

| field | meaning |
|---|---|
| `place` | `url`, `hosted`, `docker` or `kubernetes` |
| `url`, `health_url` | the base URLs; `health_url` is null when health answers on a port this machine cannot reach (a note says why) |
| `how` | `direct`, `published-port` or `port-forward` |
| `image`, `image_id` | what the container or pod runs |
| `passthrough` | whether the deployment sets `NIM_PASSTHROUGH_ARGS`: `set`, `not set` or `unknown` |
| `port_forward` | `command`, `argv`, `pod`, `remote_port`, `local_port`, `restarts` |
| `notes` | plain sentences: a bind address, TLS, a moved health port, a rollout |

## A URL
- `http://` or `https://` only. A trailing `/` and one trailing `/v1` are removed.
- A URL with `user:pass@` is refused (`bad_input`). Put a gateway token in a variable and pass
  `--api-key-env NAME`.
- `--health-url` overrides the health base, for a NIM that sets `NIM_HEALTH_PORT`.
- The key goes only to the scheme, host and port of the URL. A `--health-url` (or a published
  `NIM_HEALTH_PORT`) on another host, port or scheme gets its health `GET`s without it; a 401
  there says so in the health result. The host counts as written: `gw.example.com.`, with a
  final dot, is another host than `gw.example.com`.
- No redirect is followed. A 3xx is reported, and the host it names gets no request.
- Over plain `http://`, which anyone on the network path can read, a key goes only to this
  machine: the name `localhost`, 127.x (also as `127.1` and the other forms the resolver
  takes), `::1`, a port-forward, and `0.0.0.0` or `::`, which connect to this machine. A
  `*.localhost` name or a host with a final dot (`127.0.0.1.`) is not one: the resolver may
  look it up in DNS. To any other host the run stops before any request with
  `key_over_plain_http` (exit 2), also for a `--docker` port published on one address that is
  not loopback (the URL names that address). Use an `https://` URL, or run the check on the
  machine that serves the endpoint, with a loopback URL. For a private gateway that answers
  only plain http, the user can agree to `--allow-plain-http-key`: the key is sent, and the
  warning is in `facts.warnings` however the run ends, and in `results.json` and `report.md`
  when it writes them. A run with no key sends nothing secret and needs no flag.
- A host with a `%` (percent-encoding) is refused (`bad_input`): the connection decodes it, so
  `127.0.0.1%2Egw.example.com` would reach the name `127.0.0.1.gw.example.com`. Only an IPv6
  zone keeps its `%` (`[fe80::1%25eth0]`).
- A loopback URL (`localhost`, `127.0.0.1` and the rest of 127.x, also in the short or numeric
  forms the resolver takes, such as `127.1` or `2130706433`, `::1`, a port-forward) never goes
  through `HTTP_PROXY`. Other URLs use the environment's proxy settings. Each URL decides for
  itself: a `--health-url` on another host uses the proxy settings even when `--url` is loopback,
  so a host it reaches directly, such as a container IP, belongs in `NO_PROXY`.

A NIM does not check client keys: its API reference sets `api_key="not-used"` and says NIM does
not validate the `x-api-key` header. So a bearer token belongs to a gateway in front of the NIM:
a bare NIM ignores one and needs no `--api-key-env`. A gateway can also:
- route only the inference paths, so a 404 on `/v1/health/*` does not prove the NIM is down;
- cut long requests: "Also check any proxy timeout between the client and the service".

## Hosted: the NVIDIA API catalog
| item | value |
|---|---|
| base URL | `https://integrate.api.nvidia.com/v1`; `http://` is refused (`bad_input`, exit 2) before the key is read, as the key would cross the network in clear text |
| key | `Authorization: Bearer $NVIDIA_API_KEY`; a key from build.nvidia.com or an NGC Personal Key with the service "AI Foundation Models and Endpoints". The API answers a chat request without a key with 401, so a run of any chat check with no key (`NVIDIA_API_KEY` unset or empty, and no `--api-key-env`) is `bad_input` (exit 2) before any request: ask the user to export the key. A `--dry-run`, or `--checks health,models`, needs none |
| health | no `/v1/health/*` routes (404), so health is `not_applicable` (`hosted_no_health_route`) |
| `/v1/models` | lists the whole catalog, not one served model; `GET /v1/models/ID` answers 200, 410 (end of life) or 404 |
| rate limits | "The rate of requests will vary per model queried"; the concurrency check stays at 4 (`--concurrency` above 4 is `concurrency_too_high`, exit 2, before any request); a 429 is `fail` (`rate_limited`), never retried in a loop |

The 404s and the 200/410/404 answers come from keyless probes on 2026-11-11.

## TLS
- `NIM_SSL_MODE=TLS` or `MTLS` makes the NIM serve https. The script reads it from a container or
  pod and uses `https://`.
- Verification is always on. Nothing turns it off.
- Pass `--ca-file` with the CA that signed the server certificate. For `MTLS`, also pass
  `--client-cert` and `--client-key`. A key with a passphrase is not supported.
- A port-forward, and most published ports, give a `127.0.0.1` URL. If the certificate does not
  name that address, verification fails (`tls_error`): pass `--url` with a host name it names.

## Keys, by variable name only
| variable | used for | rule |
|---|---|---|
| `NVIDIA_API_KEY` | the hosted catalog | the default only when the host is `integrate.api.nvidia.com`, over https |
| any `NAME` given with `--api-key-env NAME` | a gateway's bearer token | the name only; a key given as the value is refused (`literal_secret`, exit 2) and never echoed, also a key whose short prefix ends in `_` (`gsk_...`, `sk_live_...`); a variable that is not set is `bad_input` (exit 2), and the error does not repeat the name. Its fix asks the user to export the key; only for an endpoint other than the hosted API does it also offer to leave the flag out. A key shorter than 6 characters is `bad_input` too, as it is too short to be masked, and its value is not shown |
| `NGC_API_KEY`, `HF_TOKEN` | model downloads, not inference | never read. A container's environment can hold them: `-e NGC_API_KEY` with no value passes the local value into the container |
| `KUBECONFIG`, `DOCKER_HOST`, `DOCKER_CONTEXT`, proxy and CA variables | read by kubectl, docker and Python; `nsc_access.py` also reads `DOCKER_HOST`, only to spot a remote daemon | never set or exported by the skill |

Every printed or saved text masks the key, `Authorization: Bearer ***`, known token shapes
(`nvapi-`, `hf_`, `glpat-`, `ghp_`, `sk-`, `AKIA`), the values of credential flags and of
credential names (`x-api-key: X`, `NGC_API_KEY=X`, `["--api-key", "X"]`; the full list and the
known gap are in `references/security.md`). Text is masked before it is cut, so a key that an
endpoint, `kubectl`, `docker` or a credential plugin echoes is never kept in part. A flag counts
only when written in full: `--api-key=KEY` is an unknown flag (`bad_input`), never a short form
of `--api-key-env`.

## Docker
`--docker NAME` checks that the daemon is local (`docker context inspect`), then reads the
container with one `docker container inspect NAME`. From the container's environment it reads
only `NIM_SERVER_PORT` (default 8000), `NIM_HEALTH_PORT`, `NIM_SSL_MODE` and
`NIM_PASSTHROUGH_ARGS`; every other value is dropped unread and never printed.

| published as | URL | note |
|---|---|---|
| all addresses: `0.0.0.0` and `::` (`-p 8000:8000`; newer daemons list both) | `http://127.0.0.1:PORT` (`[::1]` when only `::` is listed) | none |
| `127.0.0.1` or `::1` (`-p 127.0.0.1:8000:8000`) | that address | answers on the Docker host only |
| one address (`-p 192.168.1.100:8000:8000`) | that address | `localhost` does not answer |
| host networking (network mode `host`) | `http://127.0.0.1:NIM_SERVER_PORT` | `-p` does not apply |
| not published | none: `port_not_published` | pass `--url`, or republish with `nim-deploy` |

- **Health port.** With `NIM_HEALTH_PORT` set, `/v1/health/*` moves to that port, which needs
  its own `-p` mapping. Unpublished, `health_url` is null: health is `not_applicable`
  (`health_port_not_reached`) and nothing is probed, and `--wait-ready` stops the run with
  `health_port_not_reached` (exit 3). Pass `--health-url` with an address that reaches the port.
- **Remote daemon.** When `DOCKER_HOST`, or the active context (`docker context inspect`), names
  anything other than a local `unix://` or `npipe://` socket, the port is on that host. The
  script stops with `docker_remote` (exit 3) before it reads the container, with or without
  `--allow-plain-http-key`: pass `--url http://HOST:PORT`, or run the check on the Docker host.
  With `--api-key-env`, the fix also says that this `http://` URL sends the key to another host,
  so it stops with `key_over_plain_http` unless the user agrees to the flag; on the Docker host
  the key stays on loopback.
- **Older inspect output.** When the record has no port table, `docker port NAME` is read
  (`8000/tcp -> 0.0.0.0:8005`, `8000/tcp -> [::]:8005`). If docker fails or times out there, the
  run stops with `docker_failed` (exit 4), not `port_not_published`.
- **State.** Only a running container is checked. One that is restarting (its restart policy
  starts a NIM that keeps exiting; Docker still says Running then, and empties the published
  ports) or paused is `container_not_running` (exit 3), never `port_not_published`.
- **Image default.** `docker image inspect --format '{{json .Config.Env}}' ID` gives the
  image's own `NIM_PASSTHROUGH_ARGS`, as the NIM docs check it. `ID` is the `sha256:` ID of the
  image the container runs, never its tag: a pull after the start can move a tag such as
  `:latest` to an image with another default. No output, or output that is not JSON, leaves the
  default unknown. A deployment value replaces the image value; it does not append. The script
  compares the two (`parser_dropped_by_override`). When the default is unknown, the parser flags
  in the container's `NIM_PASSTHROUGH_ARGS` still count as what the NIM runs with (so an opt-in
  parser such as Kimi K3.1 Pro's `--reasoning-parser` is seen), but nothing is said about the
  default or whether `-e` replaced it.

## Kubernetes
| target | how the pod and the port are found |
|---|---|
| `nimservice/NAME` | reads the NIMService (image, platform). The Operator names the Service after the NIMService; its port is `api`. On the KServe platform, the pods labelled `serving.kserve.io/inferenceservice=NAME` |
| `helm/RELEASE` | the Services labelled `app.kubernetes.io/name=nim-llm,app.kubernetes.io/instance=RELEASE`, skipping the headless one (`clusterIP: None`, the `-sts` Service); port `http-openai`. The name is found, not built, so `fullnameOverride` and release names that contain the chart name work |
| `svc/NAME` | the Service's selector; its port `api`, then `http-openai`, else the first TCP port (a UDP or SCTP port is skipped: a port-forward carries TCP only) |
| `pod/NAME` | that pod; its port `api` or `http-openai`, else its first port, else `NIM_SERVER_PORT` or 8000 |

- **One pod.** The pod must be Running, Ready and not terminating. A pod labelled
  `nim-llm-role=leader` (multi-node) comes first, then by name. The report names the pod, and
  every request goes to it: the Service's load balancing does not apply.
- **A pod still loading.** With no such pod, the run stops with `pod_not_ready` (exit 3), unless
  `--wait-ready S` is given: then the script reads the pods again every 5 s, with the same
  read-only `kubectl get`, until one is Ready (its readiness probe passes once the model has
  loaded, which can take many minutes) or S seconds have passed, and the health check gets
  what is left of S. A `pod/NAME` that does not exist yet is waited for too (a StatefulSet pod
  that is created again keeps its name). A missing NIMService, Service, release or context, or
  a refused read, is never waited for.
- **Numeric port.** A named Service port maps to its `targetPort`. A named `targetPort` maps to
  the pod's `containerPort` with that name. The forward uses that number. A numeric
  `targetPort` picks the one container that declares it, so a sidecar listed first is not read
  as the NIM.
- **Configuration read.** The container's image and `args`, and the literal values of
  `NIM_PASSTHROUGH_ARGS`, `NIM_HEALTH_PORT`, `NIM_SSL_MODE` and `NIM_SERVER_PORT`. No ConfigMap
  or Secret is ever read (the NGC key comes from a `secretKeyRef`), so a value taken from one
  (`valueFrom`) is unknown:
  - `NIM_SSL_MODE` stops the run with `setting_not_read` (exit 3) before the port-forward, as
    http or https cannot be told. So does `NIM_SERVER_PORT`, when the container declares no
    port. Ask the user for an address that reaches the NIM and pass `--url`.
  - `NIM_PASSTHROUGH_ARGS` is `unknown`, and `NIM_HEALTH_PORT` gets a note.
- **envFrom.** It is not read either. A note names the settings above that the pod spec does
  not set, as `envFrom` may set them: `NIM_PASSTHROUGH_ARGS` is then `unknown`, and a
  `NIM_SSL_MODE` set only there is not seen, so the URL is http.
- **Image default.** Read on this machine's Docker daemon only by the pod's `imageID` digest
  (`REPO@sha256:...`), which the daemon answers only when it holds that very image; a local copy
  of the tag can be another image. Otherwise the image default is unknown, and nothing is
  claimed from it.
- **Health port.** With `NIM_HEALTH_PORT` set, health is on a pod port the forward does not
  reach, so `health_url` is null, with the result and the `--wait-ready` stop of Docker above.
- **Ingress or LoadBalancer.** Pass that address with `--url` to test the users' path.

## The port-forward
The command is `kubectl port-forward pod/POD :PORT --context CTX -n NS`. The empty local part lets
kubectl pick a free local port, so nothing collides with port 8000.

| step | what the script does |
|---|---|
| start | runs kubectl with its output going to files in a private temporary folder (kubectl prints a line per connection, so an unread pipe would block it); waits up to 30 s for `Forwarding from 127.0.0.1:N`, then for a TCP connect |
| before each check | checks that kubectl still runs and accepts a connection on its local port. kubectl ends the whole forward when the pod restarts or the NIM refuses one connection, so a stopped forward is restarted once, on the same local port when it is free; `restarts` counts it. A forward whose first start failed never listened: it is started, not restarted |
| after a check whose requests got no answer | waits up to 2 s for kubectl to exit: it closes the connection first and exits a moment later, so the next check would still find it running. When it exits, the check gets a warning and the next check restarts the forward. This holds for the first chat request too: the run then does not stop with `endpoint_unreachable` or `first_request_timeout`, and the next check's request counts as the first |
| a second stop | `port_forward_lost` (exit 4). A forward that stops mid-run and cannot be restarted (also `pod_not_ready` when the pod is gone, or `kubectl_failed`) ends the requests, not the files: the checks that ran keep `results.json`, `report.md` and their evidence, the rest are listed in `not_run`, and `report DIR` reads them back |
| end | SIGINT (terminate off POSIX), wait 10 s, then kill. Also on an error, at exit, and on SIGTERM or SIGHUP |

| kubectl says | code | what to do |
|---|---|---|
| `Forbidden ... pods/portforward` | `rbac_missing` (exit 3) | ask for the role below |
| `pod is not running`, or the pod is gone | `pod_not_ready` (exit 3) | wait until the pod is Ready, or rerun with `--wait-ready S`; a pod that does not start is `nim-deploy`'s |
| `unable to listen` on the old local port | none | the restart takes a new port |
| no `Forwarding from` line in 30 s (for example, kubectl waits for a login) | `port_forward_failed` (exit 4) | log in to the cluster first; stdin is closed, so a prompt cannot be answered |
| `getting credentials: exec: executable aws not found` (a missing or failing credential plugin), or `Unable to connect to the server` | `kubectl_failed` (exit 4), on any kubectl call | the cluster was never asked: the user installs the plugin the message names or logs in |
| `lost connection to pod` twice | `port_forward_lost` (exit 4) | check the pod's restarts with the user |

## RBAC needed
| step | permission | built-in `view` | built-in `edit` |
|---|---|---|---|
| pre-check | `kubectl auth can-i create pods --subresource=portforward` | yes (every user may ask) | yes |
| read a NIMService | `get nimservices.apps.nvidia.com` | no | no |
| find Services and pods | `get`, `list` on `services` and `pods` | yes | yes |
| port-forward | `create` on `pods/portforward` | **no** | yes |

- The pre-check runs first. A "no" stops the run with `rbac_missing` (exit 3) before anything
  else runs. can-i judges the request's attributes only, and does not check that the
  namespace exists: a user whose role is bound in one namespace gets "no" for a mistyped one,
  and a read there is forbidden too. So first confirm the context and the namespace with the
  user; if both are right, ask the cluster admin for a role in the namespace with `create` on
  `pods/portforward` plus the reads above, or pass `--url`.
- A forbidden NIMService read is `rbac_missing` too. `--k8s svc/NAME` needs only the `view`
  reads, plus the port-forward.
- Port-forward creates no object. It is a `create` on the `pods/portforward` subresource.
  Kubernetes warns that it "may bypass network-level controls".

## What the script runs
Only `kubectl` and `docker`, found in the absolute folders of `PATH` (an empty or relative entry
is skipped). No shell, stdin closed, a timeout on every call, never `sudo`. Every kubectl command
but `config get-contexts` names `--context CTX -n NS`. Anything else is refused before it starts
(`refused_command`, exit 5).

| command | why |
|---|---|
| `kubectl auth can-i create pods --subresource=portforward` | the RBAC pre-check |
| `kubectl get nimservice NAME -o json` | image and platform of a NIMService |
| `kubectl get service NAME -o json`, `kubectl get services -l SELECTOR -o json` | the Service, its selector and port |
| `kubectl get pods -l SELECTOR -o json`, `kubectl get pod NAME -o json` | a Ready pod, its image, args and ports |
| `kubectl port-forward pod/POD [LOCAL]:PORT` | the forward |
| `kubectl config get-contexts -o name` | lists context names when `--context` is not in the kubeconfig |
| `docker context inspect --format '{{.Endpoints.docker.Host}}'` | local or remote daemon |
| `docker container inspect NAME`, `docker port NAME` | image, environment subset, published ports |
| `docker image inspect --format '{{json .Config.Env}}' ID` | the default `NIM_PASSTHROUGH_ARGS` of the image the container runs (its ID) or the pod runs (`REPO@sha256:...`) |

## What it never runs
| command | why not |
|---|---|
| `kubectl run` (a curl pod, as the Operator docs verify) | creates a Pod: a write to the cluster |
| `kubectl exec`, `debug`, `cp`, `attach` | runs a process in, or changes, the serving container |
| `kubectl apply`, `patch`, `edit`, `scale`, `delete`, `rollout` | changes the deployment |
| `kubectl get secrets`, `kubectl get --raw` | reads Secrets |
| `helm test` | creates test Pods |
| `helm status`, `helm get values` | read Secrets: Helm keeps release data in Secrets |
| `docker exec`, `run`, `start`, `stop`, `restart`, `rm`, `pull` | runs or changes containers |
| `sudo` | the script runs as the user |

## Errors
| code | exit | when | what to do |
|---|---|---|---|
| `bad_input` | 2 | a bad container name, target, context, namespace or URL; `http://` to the hosted API; a key variable that is not set or holds a key shorter than 6 characters, or a hosted run of chat checks with no key | fix the flag; for a key, ask the user to export it |
| `concurrency_too_high` | 2 | `--concurrency` above 8, or above 4 on the hosted API | a lower value; a load test is `nim-benchmark`'s |
| `key_over_plain_http` | 2 | a key would go over plain `http://` to a host that is not this machine, without `--allow-plain-http-key`; nothing is sent | tell the user the risk; an `https://` URL, or a run on the machine that serves the endpoint; the flag only when the user agrees |
| `tool_missing` | 3 | no `kubectl` or `docker` in an absolute `PATH` folder | install it, or pass `--url` |
| `docker_remote` | 3 | the Docker daemon is on another host | `--url http://HOST:PORT` |
| `target_not_found` | 3 | no such container, NIMService, Service, release or pod (for `pod/NAME`, after `--wait-ready S`, when given); a Service with no selector or no TCP port | check the name and namespace with the user |
| `container_not_running` | 3 | the container is stopped, restarting (a NIM that keeps exiting) or paused | ask the user; finding why a NIM exits, and starting it, is `nim-deploy`'s job |
| `port_not_published` | 3 | the NIM port has no `-p` mapping | `--url`, or republish with `nim-deploy` |
| `context_not_found` | 3 | `--context` is not in the kubeconfig | ask the user; the fix lists the context names |
| `rbac_missing` | 3 | no `create pods/portforward`, or a read is forbidden (also what a mistyped namespace gives) | confirm the context and namespace with the user first; then ask the admin for the role above, or `--url` |
| `pod_not_ready` | 3 | no Running and Ready pod (after `--wait-ready S`, when given), or the forward's pod is gone mid-run | wait, or rerun with `--wait-ready S`; a pod that does not start is `nim-deploy`'s |
| `health_port_not_reached` | 3 | `--wait-ready` with `NIM_HEALTH_PORT` on a port this run cannot reach | ask for a `--health-url` that reaches it, or rerun without `--wait-ready` |
| `setting_not_read` | 3 | the pod takes `NIM_SSL_MODE`, or `NIM_SERVER_PORT` with no port declared, from a ConfigMap or a Secret (`valueFrom`), which is never read | ask the user for an address that reaches the NIM; pass `--url` (`https://` with `--ca-file` for TLS or MTLS, plus `--client-cert` and `--client-key` for MTLS) |
| `docker_failed` | 4 | the daemon is down, its socket is missing, or it denied access; also when `docker port` fails | start it, or get access (the `docker` group) |
| `kubectl_failed` | 4 | the cluster does not answer, a login is needed, or a credential plugin is missing (`exec: executable ... not found`) | the user logs in or installs the plugin, then rerun |
| `port_forward_failed` | 4 | the forward did not start | as the message says, then rerun |
| `port_forward_lost` | 4 | the forward stopped twice | check the pod with the user; the checks that ran are in `--out` |
| `refused_command` | 5 | a command off the read-only list | a bug: report it, never run the command another way |

## Sources
- NIM for LLMs 2.1.5: [environment variables](https://docs.nvidia.com/nim/large-language-models/2.1.5/reference/environment-variables.html) (`NIM_SERVER_PORT`, `NIM_HEALTH_PORT`, `NIM_SSL_MODE`), [architecture](https://docs.nvidia.com/nim/large-language-models/2.1.5/reference/architecture.html) (port 8000, health port), [passthrough arguments](https://docs.nvidia.com/nim/large-language-models/2.1.5/reference/advanced-configuration.html#passthrough-arguments-nim-passthrough-args) (replace, not append; the image check), [API reference](https://docs.nvidia.com/nim/large-language-models/2.1.5/reference/api-reference.html) (keys not validated), [request troubleshooting](https://docs.nvidia.com/nim/large-language-models/2.1.5/troubleshooting/requests.html) (proxy timeouts), [NIM Operator deployment](https://docs.nvidia.com/nim/large-language-models/2.1.5/deployment/kubernetes-deployment/nim-operator-deployment.html) (Service and `app=` label).
- NIM Helm and platforms: [Helm deployment](https://docs.nvidia.com/nim/large-language-models/latest/deployment/kubernetes-deployment/helm-k8s.html) (`kubectl get svc -l app.kubernetes.io/instance=...`), [Azure](https://docs.nvidia.com/nim/large-language-models/latest/deployment/csp-deployment/azure.html) (`app.kubernetes.io/name` label), [KServe](https://docs.nvidia.com/nim/large-language-models/latest/deployment/kubernetes-deployment/kserve.html) (predictor label), [VLM Helm](https://docs.nvidia.com/nim/vision-language-models/latest/deploy-helm.html) (port `http-openai`), [Helm chart on NGC](https://catalog.ngc.nvidia.com/orgs/nim/helm-charts/nim-llm).
- NIM Operator: [NIMService docs](https://docs.nvidia.com/nim-operator/latest/service.html), and [k8s-nim-operator](https://github.com/NVIDIA/k8s-nim-operator) `api/apps/v1alpha1/nimservice_types.go` (Service name, port `api`, selector `app=NAME`, `nim-llm-role=leader`).
- Docker: [docker container port](https://docs.docker.com/reference/cli/docker/container/port/), [port publishing](https://docs.docker.com/engine/network/port-publishing/), [host network](https://docs.docker.com/engine/network/drivers/host/), [docker run, environment](https://docs.docker.com/reference/cli/docker/container/run/), [contexts](https://docs.docker.com/engine/manage-resources/contexts/).
- Kubernetes: [kubectl port-forward](https://kubernetes.io/docs/reference/kubectl/generated/kubectl_port-forward/), [port-forward task](https://kubernetes.io/docs/tasks/access-application-cluster/port-forward-access-application-cluster/), [kubectl auth can-i](https://kubernetes.io/docs/reference/kubectl/generated/kubectl_auth/kubectl_auth_can-i/), [RBAC user-facing roles](https://kubernetes.io/docs/reference/access-authn-authz/rbac/#user-facing-roles), [request verbs](https://kubernetes.io/docs/reference/access-authn-authz/authorization/), [bootstrap policy](https://github.com/kubernetes/kubernetes/blob/master/plugin/pkg/auth/authorizer/rbac/bootstrappolicy/policy.go) (`pods/portforward` in `edit`, not `view`), [client-go port-forward](https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/client-go/tools/portforward/portforward.go) (a line per connection; one error ends the forward).
- Helm: [storage backends](https://helm.sh/docs/topics/advanced/) (release data in Secrets).
- Hosted: [LLM APIs](https://docs.api.nvidia.com/nim/reference/llm-apis), [authentication](https://docs.api.nvidia.com/nim/docs/authentication), [NVIDIA NIM FAQ](https://forums.developer.nvidia.com/t/nvidia-nim-faq/300317) (rate limits).
