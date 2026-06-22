---
name: jetson-download-bsp
description: >-
  Download NVIDIA Jetson Linux BSP artifacts (BSP tarball, sample
  rootfs, public_sources, x-tools, guides) for the active target. Use
  for Auto Setup; not for extraction or profile edits.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags: [setup, bootstrap, download]
  domain: setup
---

# Download BSP Artifacts

`jetson-download-bsp` owns network fetches for the Auto Setup path.
It does not extract archives, initialize source trees, register documents,
or edit the active target profile. Those remain owned by
`jetson-init-image`, `jetson-init-source`, and `jetson-link-docs`.

## When to invoke

- `/jetson-quick-start` selected `Auto Setup`.
- The user wants NVIDIA BSP artifacts downloaded into the current
  workspace before running setup skills.

## Inputs and outputs

Resolve the active profile per
[`../../context/target-platform-contract.md`](../../context/target-platform-contract.md).
Refuse if no active profile exists.

Follow the shared
[`quick_start_prefill` contract](../../context/bsp-customization-workflow.md#quick_start_prefill-contract).
When invoked by `/jetson-quick-start`, this skill consumes
`quick_start_prefill.mode` and `quick_start_prefill.download.bsp_release`
as the requested release when present. Missing or invalid prefilled values
are not fatal, but a concrete release is required before any download
starts.

Use the workspace root as the parent of `target-platform/`. Place files
where downstream setup skills already search:

| Artifact | Destination |
|---|---|
| Jetson Linux BSP tarball | `<bsp_image.root_path or workspace/Image>/Jetson_Linux_R<ver>_aarch64.tbz2` |
| Sample rootfs tarball | `<bsp_image.root_path or workspace/Image>/Tegra_Linux_Sample-Root-Filesystem_R<ver>_aarch64.tbz2` |
| `public_sources.tbz2` | `<workspace>/Downloads/public_sources.tbz2` |
| `x-tools.tbz2` | `<workspace>/Downloads/x-tools.tbz2` |
| Developer Guide | `<documents.root_path or workspace/Documents>/` |
| User Guide | `<documents.root_path or workspace/Documents>/` |

Do not write `bsp_image:`, `source:`, or `documents:` blocks. The next
setup skills will register derived metadata after extraction or user
confirmation.

## Procedure

### Resolve target release

Fetch the release list from:
<https://developer.nvidia.com/embedded/jetson-linux-archive>

From the archive and linked release pages, list concrete releases that
explicitly support the active platform. Let the user choose a concrete
Jetson Linux / L4T release token (`R36.4.4`, `36.4.4`, `38.2`,
`38.2.1`, `38.4`, etc.); do not proceed with a family placeholder such
as `R38.x`. A concrete release token is an optional leading `R` followed
by two or three numeric dotted components. Preserve the token exactly as
the official page presents it; do not add `.0`, remove a patch component,
or otherwise normalize it. Sort release choices by dotted numeric order
descending (newest first), comparing the components that are present; keep
source-provided recommendations in descriptions.

Resolve `quick_start_prefill.download.bsp_release` as follows:

- Concrete and supported: continue with that release.
- Missing, `skip`, blank, or a family placeholder (`R38.x`, `38.x`): show
  supported concrete releases and ask the user to choose one.
- Concrete but unsupported or ambiguous: stop before artifact discovery and
  show the active platform, requested BSP release, supported concrete
  releases, and source archive or release page.

Never auto-substitute the requested release with a supported release, even
when there is only one supported option.

For unsupported or ambiguous prefill, ask with the shared
[`User input prompt style`](../../context/bsp-customization-workflow.md#user-input-prompt-style),
to choose exactly one click-to-select action:

- `pick_supported_release` — user selects one supported concrete release.
- `confirm_override` — user confirms the incompatible release and enters
  an override reason.
- `cancel` — stop with no downloads.

Do not continue to "Locate release artifacts" until the selected release is concrete and
supported, or the user explicitly confirms override. Record any override
reason in the final summary. If the user cancels, stop with no downloads.
Do not claim support unless the archive or linked release page explicitly
lists the active platform.

### Discovery retries

For every artifact WebFetch in "Resolve target release" and "Locate release artifacts" (BSP, rootfs,
`public_sources.tbz2`, `x-tools.tbz2`, Developer Guide, User Guide),
try 3 attempts before marking it missing:

1. Focused prompt — artifact name + file type.
2. Verbatim link dump on the same URL — *"list every hyperlink, text
   and href, no filtering"*. Accept product-specific names and
   `.html` indexes.
3. Verbatim dump on a sibling page — Jetson Linux archive, JetPack
   downloads, or `https://docs.nvidia.com/jetson/archives/r<ver>/`.

Only after all 3 fail, route to user-paste / skip. Never fabricate
URLs.

### Locate release artifacts

Follow links from the selected release page. Never construct NVIDIA
download URLs by string concatenation.

Required artifacts:

- Jetson Linux BSP image tarball.
- Tegra Linux sample root filesystem tarball.
- `public_sources.tbz2`.
- `x-tools.tbz2` / **Crosstool-NG Toolchain gcc** matching the selected
  BSP release.
- Developer Guide.
- User Guide.

If a guide is not available as a direct file link, accept the release
documentation URL as the artifact and record it in the final summary.
If the toolchain is listed on a JetPack downloads page instead of the
release page, follow the official page link and match the selected BSP
release; do not synthesize the URL from a release number.
Do not proceed until each required artifact is downloaded, already
present at its destination, represented by a URL, or explicitly skipped
by the user.

### Show the fetch plan

Before changing the workspace, print a plan with:

- selected release and release page URL;
- active platform summary (`reference_devkit.name`, optional
  `custom_carrier.name`, and `flash_config`);
- source URL and destination path, or documentation URL, for every artifact;
- any existing destination file that will be reused;
- any artifact that requires browser/EULA handling instead of `curl`.

Default to **Download all**. Do not ask an extra confirmation when every
required artifact is downloadable, already present for reuse, or a
documentation URL to record. Print the plan and continue to disk-space
verification.

Still stop and ask before side effects in any of these cases:

- the release required an incompatibility override;
- a required artifact is missing and needs user-paste / skip;
- a destination file would be overwritten rather than reused;
- the artifact requires browser/EULA handling;
- disk-space verification fails or cannot be computed safely.

### Verify disk space

Sum `Content-Length` across artifacts to download (`curl -sI <url>`);
refuse if `df -B1 --output=avail <workspace>` headroom < sum × 1.2.
Skip URL-only artifacts.

### Download safely

Create destination directories as needed. For direct file links, use a
temporary `.part` path and rename only after success:

```bash
mkdir -p "<destination-dir>"
curl -fL --retry 3 --retry-all-errors --retry-delay 5 -o "<destination>.part" "<artifact-url>"
mv "<destination>.part" "<destination>"
```

If a destination file already exists and is non-empty, reuse it by
default after showing its path. Ask before overwriting. If an artifact
download fails due to 404, 401/403, EULA/browser authentication, or a
network error, stop and show the artifact URL plus expected destination
path. Ask the user to download the file manually, skip that artifact, or
cancel. Do not synthesize alternate NVIDIA URLs.

### Validate filenames

Before finishing, verify the BSP and rootfs tarball filenames contain
the exact `R<ver>` token from the official artifact URL or filename
discovered in Step 2. `<ver>` may have two or three numeric components
(`R38.2`, `R38.2.1`, `R38.4`, etc.). Do not infer `R38.2.0` from
`R38.2`, and do not assume that a selected two-component release maps to
a three-component tarball name:

- `Jetson_Linux_R<ver>_aarch64.tbz2`
- `Tegra_Linux_Sample-Root-Filesystem_R<ver>_aarch64.tbz2`

Verify `public_sources.tbz2` is a bzip2 tarball when it exists locally:

```bash
file -b "<workspace>/Downloads/public_sources.tbz2" | grep -q "bzip2 compressed"
```

Verify `x-tools.tbz2` the same way when it exists locally:

```bash
file -b "<workspace>/Downloads/x-tools.tbz2" | grep -q "bzip2 compressed"
```

Warn, but do not fail, if document filenames differ from the release
version; NVIDIA documentation naming is less consistent than tarballs.

## Finish

Report the selected release, downloaded/reused/skipped artifacts, and
exact local paths. Then hand back to the caller with:

1. `/jetson-init-image`
2. `/jetson-init-source`
3. `/jetson-link-docs`

## Gotchas

- Keep this skill download-only; extraction belongs to
  `jetson-init-image` and `jetson-init-source`.
- Put image tarballs in the image root, not `Downloads/`, because
  `jetson-init-image` searches the image root, workspace root, and
  current directory.
- Put `public_sources.tbz2` in `Downloads/` because
  `jetson-init-source` auto-discovers that path for Branch A.
- Put `x-tools.tbz2` in `Downloads/` because `jetson-init-source`
  auto-extracts that path and writes the resulting `source.toolchain`
  prefix.
- The User Guide maps to `documents.bsp_user_guide` in the profile
  schema (alongside `bsp_developer_guide`). `jetson-link-docs` writes
  it when the user confirms; this skill only stages the file or URL.

## Purpose

Stage the four NVIDIA-shipped artifacts (BSP, sample rootfs,
public_sources, x-tools) plus the Developer / User Guides at the exact
paths the Setup skills look in, so `/jetson-init-image`,
`/jetson-init-source`, and `/jetson-link-docs` can run offline once
this skill is done.

## Prerequisites

- Active target profile resolved per
  `../../context/target-platform-contract.md`.
- Network access to `developer.nvidia.com` and the Jetson Linux archive
  page.
- Workspace layout with the expected `Image/`, `Source/`,
  `Downloads/`, and `Documents/` roots (default workspace shape).

## Limitations

- Download-only; never extracts, runs `apply_binaries.sh`, mounts
  sources, registers documents, or mutates the active profile.
- Release selection is bounded to entries published on the official
  Jetson Linux archive; pre-release or internal builds are out of
  scope.
- No retry / mirror logic — failures are surfaced to the user, not
  silently retried.

## Troubleshooting

- **Release page not reachable** — check VPN / proxy; do not invent
  fallback URLs. Stop and report.
- **Archive name mismatch with active platform** — verify the release
  declares the active reference devkit; if not, ask the user to pick a
  supported release or switch target via `/jetson-set-target`.
- **`public_sources.tbz2` missing on archive** — some releases ship it
  as a separate page; follow the release-page link rather than
  guessing the URL.
- **Existing files in the destination** — refuse to overwrite without
  confirmation; downloads must be idempotent at the user's discretion.

## References

- [`../../context/target-platform-contract.md`](../../context/target-platform-contract.md)
- [`../jetson-quick-start/SKILL.md`](../jetson-quick-start/SKILL.md)
- [`../jetson-init-image/SKILL.md`](../jetson-init-image/SKILL.md)
- [`../jetson-init-source/SKILL.md`](../jetson-init-source/SKILL.md)
- [`../jetson-link-docs/SKILL.md`](../jetson-link-docs/SKILL.md)
- Jetson Linux archive: <https://developer.nvidia.com/embedded/jetson-linux-archive>
