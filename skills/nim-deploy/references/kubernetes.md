# Kubernetes targets: Helm chart nim-llm and NIM Operator

Read this when the target is `helm` or `nim-operator`, or a pod does not become ready. Every
command carries the approved scope: `kubectl --context CTX -n NS`, `helm --kube-context CTX -n NS`.
Never use the current context, and never `helm --set`: values live in `values.yaml`.

## Prerequisites the user owns
Kubernetes 1.26 or later with GPU nodes, the NVIDIA GPU Operator, a storage class, and for the
Operator path NIM Operator 3.0.2 or later. Secrets are created by the user and passed to the run by
NAME only:
- an image pull secret for `nvcr.io` (`pull_secret`, type docker-registry);
- a generic secret holding `NGC_API_KEY` (`ngc_secret`);
- optionally a secret holding `HF_TOKEN` (`hf_token_secret`).
Ask the user for the secret names. Never ask for the key values.

## Helm chart `nim-llm` (v2.x)
Repository `https://helm.ngc.nvidia.com/nim`, chart `nim/nim-llm`. The renderer writes
`values.yaml` and one command: `helm upgrade --install NAME nim/nim-llm --kube-context CTX -n NS
-f values.yaml`.

| value | set from |
|---|---|
| `image.repository`, `image.tag` | `image` (an explicit tag is required) |
| `model.ngcAPISecret` | `ngc_secret` |
| `imagePullSecrets` | `pull_secret` |
| `resources.limits.nvidia.com/gpu` | `gpus` (a count, not `all`) |
| `nodeSelector` | `node_selector`: node labels that select the GPU's nodes (nim-select's inspection of the cluster, or labels the user verified) |
| `affinity` | `node_hostnames`: actual `kubernetes.io/hostname` label values, which can differ from node names; fallback when no placement key qualifies |
| `persistence` (`enabled`, `size`, `storageClass`) | `cache_size`, `storage_class` |
| `model.name` | `served_model_name`, else `model` |
| `model.modelPath`, `model.hfTokenSecret` | `model_path`, `hf_token_secret` (model-free) |
| `env` | `env`, and `NIM_MODEL_PROFILE` from `model_profile` |
| `customArgs` | `extra_args`: backend flags, one list item per option and value |
| `replicaCount` | `replicas` |

- `model.profile` only feeds multi-node; a single node reads `NIM_MODEL_PROFILE`.
- A StatefulSet is the default. With no spare GPU for a rolling update the user can pick a
  Deployment with `Recreate`; that is a values change, so set it through a new run.
- Never put `NIM_PASSTHROUGH_ARGS` in `env`: Kubernetes replaces the image value. Use `customArgs`.
- The service is `<release>-nim-llm` on port 8000.

## NIM Operator (`apps.nvidia.com/v1alpha1`)
Two objects in `nimservice.yaml`:
- `NIMCache` caches the model on a PVC: `spec.source.ngc` with `modelPuller` (the image),
  `pullSecret`, `authSecret`, and a `model` filter (`engine` of `vllm` or `sglang`,
  `tensorParallelism`, or an explicit `profiles` list). Wait for status `Ready`.
- `NIMService` runs the pods: `spec.image`, `spec.authSecret`, `spec.storage.nimCache.name` (the
  NIMCache name) with `profile`, `spec.resources.limits`, `spec.expose.service`, `spec.env`, and
  `spec.args` for backend flags (each option and value its own item; unset when multi-node).
- The default startup probe is `/v1/health/ready` with `failureThreshold` 120 and `periodSeconds`
  10: 20 minutes. Large models need more; set `startup_failure_threshold` (for example 240).
  Readiness is `/v1/health/ready`, liveness `/v1/health/live`, both on port 8000.
- `spec.expose.service.port` must match the container port; the Operator sets `NIM_SERVER_PORT`
  from it. The service is named after the NIMService, port 8000.
- Without a NIMCache (`use_nimcache=false`, used for model-free images) the NIMService gets its own
  PVC.

## Reaching the endpoint
`nimdeploy.py status RUN` prints `facts.port_forward` in `wait_ready` and `verify`. Run it in the
background, then use `facts.next_command`. To use a real URL (Ingress, LoadBalancer) instead, ask
for it before approval and run `nimdeploy.py set RUN url=...` while the run is in plan or validate:
`set` is refused after approval. If the URL is only known later, pass it to the checks directly
(`verify.py ready --url URL --run RUN`, then `verify.py smoke --url URL --run RUN`).

## When a pod is not ready
Run `facts.next_command` in `diagnose` (`verify.py triage --k8s NS/POD --context CTX`); it reads
`kubectl logs` and `kubectl describe pod` and matches known failures
(`references/troubleshooting.md`). Pod name: `kubectl get pods --context CTX -n NS`. A NIMCache that
stays pending: check the storage class and the cache Job `<nimcache>-job`.
