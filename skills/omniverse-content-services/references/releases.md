<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Release record

Checked **2026-10-20**. Router: **omniverse-content-services@0.1.0**.
This is a dated source review, not a tested compatibility matrix or deployment
lockfile. Use [repository routes](repositories.md) for expert entrypoints and
[NGC prebuilds](prebuilt-artifacts.md) for downloads and service documentation.

## Repository releases and source revisions

GitHub releases below are ordered by publication date, excluding drafts and
retaining prerelease labels. A source tag does not imply a GitHub Release or a
matching NGC artifact. No second release is invented when only one tag exists.

| Repository | Most recent published release | Previous published release | Interpretation |
| --- | --- | --- | --- |
| ovstorage | [v0.4.0](https://github.com/nvidia-omniverse/ovstorage/releases/tag/v0.4.0), 2026-10-15 | [v0.3.0](https://github.com/nvidia-omniverse/ovstorage/releases/tag/v0.3.0), 2026-09-22 | v0.4.0 adds the preview USD resolver and reorganizes the repository. v0.3.0 adds cloud URL aliases, Hugging Face access and experimental Hub integration. |
| ovpackage | [0.2.1](https://github.com/nvidia-omniverse/ovpackage/releases/tag/0.2.1), 2026-08-14, **prerelease** | [0.1.0](https://github.com/nvidia-omniverse/ovpackage/releases/tag/0.1.0), 2026-06-02, **prerelease** | 0.2.1 expands connections and transfer handling; 0.1.0 provides the initial package CLI/API release. |
| usd-content-workspace | No GitHub Releases or source tags found | None found | Selected source: [release/v0.1.0 branch](https://github.com/nvidia-omniverse/usd-content-workspace/tree/release/v0.1.0). A release branch is distinct from published GitHub Releases and the service release sets below. |
| ovlive | No GitHub Releases; one [source tag v0.1.0](https://github.com/nvidia-omniverse/ovlive/tree/v0.1.0) | No second tag found | Client 0.1.0 is experimental. Live Edit Service 0.1.1 is a separate NGC artifact; neither number is its wire protocol version. |
| usd-caches | No GitHub Releases; one [source tag v0.1.0](https://github.com/nvidia-omniverse/usd-caches/tree/v0.1.0) | No second tag found | Hub, OVCC and OVDC have independent artifact versions below. |

The [ovstorage version manifest](https://github.com/nvidia-omniverse/ovstorage/blob/v0.4.0/VERSION.md)
identifies repository 0.4.0, library 0.3.0 and preview resolver 0.1.0. Do not tell
users to install an `ovstorage==0.4.0` wheel based on the repository release.

| Inspected source | Exact revision | Declared expert version |
| --- | --- | --- |
| ovstorage v0.4.0 | `42270c5236008e76c0e356c8e5f350305138d7c2` | `ovstorage-expert`: **unversioned** |
| ovpackage 0.2.1 | `fe7299fc206f30565a7a48ff100a9e4e98625f6b` | `ovpackage-cli` and `ovpackage-api`: **unversioned**; task-skill fallback |
| usd-content-workspace `release/v0.1.0` branch | `748f472fe4c00a74368e416fa36469dbe5d82af4` | `usd-content-workspace-expert`: **unversioned** |
| ovlive v0.1.0 | `1a7dfe65c1b5662c70d8a9320b32d8b08eb2ae4c` | **unknown**: no expert entrypoint found; documentation fallback |
| usd-caches v0.1.0 | `019b30238a886026610748acc01e143390995c53` | `usd-caches-expert`: **unversioned** |

Previous release revisions: ovstorage v0.3.0
`8a8689c0f497ae9755052cce9ceb4cbb13378234`; ovpackage 0.1.0
`7cd965abdeda9d6c014413604ae1865cda3dc9c7`. Read the expert at the version
actually selected; the metadata above describes the inspected current revisions.
An attribution to an older skill package is not a version declaration for the
current expert.

## Service release sets

The [Developer Guide changelog](https://docs.omniverse.nvidia.com/ovstorage/ovstorage-guide/latest/changelog.html)
lists **1.3.0** and **1.2.0** as its latest two release sets, reviewed here on
2026-10-20; the page was updated 2026-10-07. These are guide/service release sets,
not usd-content-workspace Git tags or a version of the whole Content Services portfolio.
1.3.0 adds Live Edit, Thumbnails and migration tooling, and updates notifications
and Navigator. 1.2.0 adds manifest/metadata support and User Info integration.

For exact charts, images, API resources and upgrade boundaries, read the workspace
expert's [released inventory on release/v0.1.0](https://github.com/nvidia-omniverse/usd-content-workspace/blob/release/v0.1.0/skills/usd-content-workspace-expert/references/versions/released-inventory.md).
It records an NGC observation on **2026-10-07**, with a current 1.3.0 set and a
previous `published` profile. The following is a routing summary of those profiles,
not proof of the two most recent individual NGC uploads:

| Service | Current inventory chart / image | Previous inventory chart / image |
| --- | --- | --- |
| Storage | 1.2.3 / 1.2.3 | 1.2.0 / 1.2.0 |
| Auto Discovery | 1.1.1 / 1.1.1 | Absent; previous profile uses legacy Discovery |
| Legacy Discovery | 2.3.11 / simple-nginx 1.1.0 | 2.3.11 / simple-nginx 1.1.0 |
| Event Aggregation | 1.6.0 / 1.6.0 | 1.5.6 / 1.5.6 |
| Event Consumer | 1.11.0 / 1.11.0 | 1.10.0 / 1.10.0 |
| Permission | 0.0.53 / 0.0.53 | 0.0.52 / 0.0.52 |
| Microsoft Graph User Info | 0.0.8 / 0.0.8 | 0.0.7 / 0.0.7 |
| Live Edit | 0.1.1 / 0.1.1 | Absent |
| Thumbnail | 0.1.0 / 0.1.0 | Absent |

Evidence conflicts remain explicit: the guide lists Auto Discovery 1.1.0 while
the inventory lists 1.1.1. Several anonymously visible NGC listings show older
versions than the inventory. Verify the selected artifact through authorized NGC
access before use; neither a successful catalog page load nor a source chart's
`appVersion` proves that an image can be fetched or works with the host.

## Cache artifact versions

| Artifact | Latest evidenced version | Earlier evidenced version | Evidence and limits |
| --- | --- | --- | --- |
| Hub workstation resource packages | 2.3.0, 2026-08-31 | 2.2.0, 2026-07-07 | Latest two entries in [NGC resource history](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/resources/hub_workstation_cache/2.3.0/version-history). |
| Hub container | 2.2.0 displayed | Previous container tag unverified | [Container listing](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/containers/hub_workstation_cache/-); do not infer a 2.3.0 image from the resource packages. |
| OVCC chart `ovcontentcache` | 4.0.4 | Previous chart version unverified | [Chart listing](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/helm-charts/ovcontentcache/-). |
| OVCC container `usd-content-cache` | 3.0.2 | 3.0.1 | [Container listing](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/containers/usd-content-cache/-) and [July manifest](https://docs.omniverse.nvidia.com/ov/ovs/26.07.0/manifest.html); the earlier manifest is not a complete tag history. |
| OVDC chart `ovderivedcache` | 6.0.1 | 6.0.0 baseline; earlier history unverified | [Chart listing](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/helm-charts/ovderivedcache/-) identifies the naming change at 6.0.0. |
| OVDC container `ovderivedcache` | 6.0.1 in October manifest | 6.0.0 displayed in catalog | [October manifest](https://docs.omniverse.nvidia.com/ov/ovs/latest/manifest.html) and [container listing](https://catalog.ngc.nvidia.com/orgs/nvidia/omniverse/containers/ovderivedcache/-) disagree on latest; verify the chosen tag through NGC. |

The October manifest validates a particular reference architecture set; it is not
a portfolio-wide latest-version index. It pairs OVCC chart 4.0.4 with image 3.0.2,
although that chart defaults to 3.0.1. Follow the owning deployment guide's image
override. Do not equate a chart version, its default image and its `appVersion`.

## Refresh and handoff

When refreshing this record, check all five GitHub release histories and tags,
the guide changelog, workspace inventory and NGC histories. Retain up to two
published releases for each component and state explicitly when a previous
version or history is unavailable. Include publication dates when provided,
prerelease status, exact source revision, expert version, artifact type and source
link. Advance the checked date only after reviewing the sources; retain upstream
observation dates. A moving `latest` page must not silently change an existing
workflow's selected versions. Do not promote unreleased source changes into the
release record. Record any fetch, compatibility or runtime checks separately.
