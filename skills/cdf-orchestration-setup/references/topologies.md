# Controller and model-service topologies

Controller placement and model-service placement are independent. A local controller does not
imply local inference services, and external inference endpoints do not imply a remote controller.

Use the checkout, kubeconfig, namespace, storage class, endpoint inventory, and capacity findings
from `$cdf-orchestration-preflight`. Do not repeat prerequisite checks here. If preflight reports
an unsupported Kubernetes distribution, including K3s, stop before setup.

| Controller | Workflow compute | Model services | Typical use |
| --- | --- | --- | --- |
| Existing controller | Remote Kubernetes | External | Lowest-footprint trial |
| Same Kubernetes cluster | Same cluster | External | One-GPU Image Attribute Augmentation deployment; Event Video Generation still needs more GPU capacity |
| Existing or same-cluster controller | Kubernetes | Internal | Multi-GPU deployment verified from the current manifests by preflight |
| Existing controller | OSMO | External only | DAGs that ship an OSMO manifest |

## Existing controller with remote workflow compute

Use this topology when an Airflow controller already exists and the verified Kubernetes cluster or
OSMO environment will run workflow tasks. Kubernetes workflows use the `kubernetes_remote` Airflow
connection. Its embedded kubeconfig API endpoint must be reachable from the controller pod, not
only from the workstation.

This path does not require installing another controller. Continue with the post-install connection
checks in the setup skill against the existing controller.

## Controller and workflow compute in the same Kubernetes cluster

Use this topology when the controller should be installed into the verified workflow cluster.
External model endpoints are compatible with this layout and reduce in-cluster GPU demand.

`make setup` embeds the preflight-verified kubeconfig into generated Helm values for the
`kubernetes_remote` Airflow connection. The supported make flow uses `in_cluster: false` even when
the controller and workflow pods share a cluster.

## External model services

Choose external mode when the necessary endpoints are available and preflight verifies their
workflow-specific credentials and configuration. Controller placement remains an independent
choice. A one-GPU Kubernetes node can support Image Attribute Augmentation in external mode because
its current task profiles are CPU-only; do not generalize that result to other workflows.

OSMO workflows are external-service-only because their manifests define batch tasks but no managed
endpoint components. Use OSMO only for DAGs with a checked-in OSMO manifest, and carry forward the
endpoint and token findings from preflight.

## Internal model services

Choose internal mode only when preflight reports that every selected workflow's aggregate GPU,
storage, image, and credential requirements are satisfied. Derive capacity from current manifests,
not remembered constants. At the time of this update, the manifests resolve to two GPUs for Image
Attribute Augmentation internal services and seven for Event Video Generation internal execution,
or nine when both evaluator checks are enabled.

## Local versus remote invocation

The repository make targets run on the machine containing the checkout. They use the explicit
kubeconfig verified by preflight and may target either a local or remote Kubernetes API. The target
cluster does not need a second copy of the repository.

For a controller install, continue with [deploy-controller.md](deploy-controller.md). After the
install, return to the setup skill for controller connection and post-install verification.
