<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NGC prebuilds and documentation

Checked **2026-10-20**. Use these links to find supported distribution artifacts
and owning documentation. Read the [release record](releases.md) before selecting
versions; source repositories, libraries, charts, images and API specifications
have separate release identities. Fetch and deployment instructions belong to the
[workspace and cache experts](repositories.md), not this router.

## Services

Start with the [NGC service collection](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/omniverse/collections/storage_apis)
and [Developer Guide](https://docs.omniverse.nvidia.com/ovstorage/ovstorage-guide/latest/index.html).
The collection retains the catalog label “Omniverse Storage APIs.” It distributes
service artifacts; the broad workflow expert here is Omniverse Content Services.
The `ovstorage` client library is a separate component, not the Storage Service image.

| Capability | Prebuilt entrypoints | Owning documentation |
| --- | --- | --- |
| Storage access through service adapters | [Storage image](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/containers/storage-service/-), [chart](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/helm-charts/storage-service/-) | [Storage configuration](https://docs.omniverse.nvidia.com/ovstorage/ovstorage-guide/latest/dev-journey/full-deployment/storage-service.html); follow its service reference for configuration authority |
| Service discovery | [Legacy Discovery chart](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/helm-charts/discovery-service/-); locate Auto Discovery in the service collection | [Discovery guide](https://docs.omniverse.nvidia.com/ovstorage/ovstorage-guide/latest/dev-journey/full-deployment/discovery-service.html); ask the workspace expert to distinguish the static catalog from the operator |
| Notifications and subscriptions | [Aggregation image](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/containers/event-aggregation-service/-), [chart](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/helm-charts/event-aggregation-service/-); [Consumer image](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/containers/event-consumer-service/-), [chart](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/helm-charts/event-consumer-service/-) | [Notifications guide](https://docs.omniverse.nvidia.com/ovstorage/ovstorage-guide/latest/dev-journey/full-deployment/notifications-services.html), including broker dependencies |
| Permission policies | [Permission image](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/containers/permission-service/-), [chart](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/helm-charts/permission-service/-) | [Permissions guide](https://docs.omniverse.nvidia.com/ovstorage/ovstorage-guide/latest/dev-journey/full-deployment/permissions-service.html) |
| Identity and User Info | Service collection: `microsoft-graph-userinfo-service` and authentication dependencies; verify exact coordinates using the workspace inventory | [User Info guide](https://docs.omniverse.nvidia.com/ovstorage/ovstorage-guide/latest/dev-journey/full-deployment/userinfo-service.html) |
| Live USD sessions | [Live Edit image](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/containers/live-edit-service/-), [chart](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/helm-charts/live-edit-service/-) | [Live Edit deployment](https://docs.omniverse.nvidia.com/ovstorage/ovstorage-guide/latest/dev-journey/full-deployment/live-edit-service.html) and [ovlive client integration](https://github.com/nvidia-omniverse/ovlive/blob/v0.1.0/client/README.md) |
| Thumbnail generation and retrieval | [Thumbnail image](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/containers/thumbnail-service/-), [chart](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/helm-charts/thumbnail-service/-) | Developer Guide's Thumbnail Service sections; verify the worker image as well as the service |

Navigator, permission UI, API specification resources and supporting test artifacts
are also in the workspace inventory. Load that inventory for those tasks instead
of treating every artifact as a mandatory service. An API resource is not a
running service. Custom adapters still require implementation and conformance
checks; a prebuilt service does not prove support for an arbitrary backend.

## Caches

| Cache and purpose | Prebuilt distribution | Owning documentation |
| --- | --- | --- |
| Hub Workstation Cache: local reuse of content and derived data through supported clients | [Resource packages](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/resources/hub_workstation_cache), [container](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/containers/hub_workstation_cache/-) | [Hub guide](https://docs.omniverse.nvidia.com/utilities/latest/cache/hub-workstation.html) |
| Omniverse Content Cache (OVCC): shared source-content caching | [Image `usd-content-cache`](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/containers/usd-content-cache/-), [chart `ovcontentcache`](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/helm-charts/ovcontentcache/-) | [OVCC guide](https://docs.omniverse.nvidia.com/ovcaches/ovcontentcache/latest/index.html) |
| Omniverse Derived Cache (OVDC): shared reuse of computed/derived data | [Image](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/containers/ovderivedcache/-), [chart](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/helm-charts/ovderivedcache/-) | [OVDC guide](https://docs.omniverse.nvidia.com/ovcaches/ovderivedcache/latest/) |

Ask the cache expert to select placement and compatible client integrations from
the user's locality, reuse and retention constraints. These caches do not turn
an arbitrary SMB/NFS client into a cache-aware application. Establish permission
for managed local caching before recommending workstation persistence.

## Artifact selection and access

Catalog visibility, entitlement, download access and runtime behavior are separate
checks. Some NGC entries require sign-in or subscription access. Container images
and Helm charts use different registries and authentication; an NGC acquisition
key is not a runtime service identity. Use the workspace expert's
[artifact-access reference on release/v0.1.0](https://github.com/nvidia-omniverse/usd-content-workspace/blob/release/v0.1.0/skills/usd-content-workspace-expert/references/deploy/artifact-access.md)
and the [NGC catalog guide](https://docs.nvidia.com/ngc/latest/ngc-catalog-user-guide.html)
for the actual retrieval procedure. Never put credentials in a report.

For a concrete deployment handoff, identify the selected chart version and every
required image tag (including workers and test images), the relevant API/client
versions and host support, then have the owning expert verify the rendered
configuration and a representative workflow. Record immutable digests when
available. If a source or catalog listing disagrees with a release inventory,
report the discrepancy and verify the exact artifact instead of silently choosing
a substitute. Documentation review alone does not verify a download or deployment.
