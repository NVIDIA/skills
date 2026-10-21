# CDF NGC Containers

Use this reference when the user wants prebuilt Auto-Labeling stage images
instead of building them, asks where the containers are published, needs to
log in to `nvcr.io`, or a run fails because a `cdf-*` image is missing
locally.

Operator-facing detail lives in
[NGC Containers](../../../docs/user-guide/ngc-containers.md). This page is the
agent procedure.

## Facts

- Images are public in the `nvidia` org: `nvcr.io/nvidia/<image>:<tag>`.
  Catalog pages are
  `https://catalog.ngc.nvidia.com/orgs/nvidia/containers/<image>`.
- The v1.1.0 release tag is `1.1.0`. Each tag is multi-arch
  (`linux/amd64`, `linux/arm64`).
- Anonymous pulls work. `docker login nvcr.io` is optional.
- The workflow runner never pulls. It launches stages by bare local name (for
  example `cdf-captioning-service`) and checks for it with
  `docker image inspect`.

| Stage | Image |
| --- | --- |
| Workflow runner | `cdf-workflow-runner` |
| `super_resolution` | `cdf-super-resolution-service` |
| `detection_and_tracking` | `cdf-detection-and-tracking-rfdetr-service` (default) or `cdf-detection-and-tracking-sam3-service` |
| `referring_expressions` | `cdf-referring-expressions-service` |
| `captioning` | `cdf-captioning-service` |
| `grounding_2d` | `cdf-grounding-2d-service` (default) or `cdf-grounding-2d-sam3-service` |
| `visual_qa` | `cdf-visual-qa-service` |
| `reasoning` | `cdf-reasoning-service` |
| `training_export` | `cdf-training-export-service` |
| `person_attribute_search` | `cdf-event-and-person-attribute-search-service` |

## Workflow

1. **Find the images the run needs.** Dry-run the cookbook and read the
   image from each planned `docker run` command:
   `make run SCRIPT=workflow-runner:main ARGS='--cookbook-file <config> --container-dry-run'`.
   A cookbook `container.images` block or a `--tracker sam3` flag can select
   the SAM3 variants, so read the plan instead of assuming the defaults.

2. **Check what is already present.** Run `docker image ls 'cdf-*'`. Skip
   any image that is already present unless the user asks for a fresh pull.

3. **Is a login needed?** (single decision)
   - Default: no. Pull anonymously.
   - Log in only if the user asks for authenticated pulls, or a pull fails
     with an authentication or rate-limit error. Ask the user to export
     `NGC_API_KEY` in their own shell (a personal key from
     <https://org.ngc.nvidia.com/setup>). Then run:
     `echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin`.
     If `NGC_API_KEY` is unset, stop and ask. Do not request the key value in
     chat.

4. **Pull and tag.** Pull only the images from Step 1:
   `make pull NGC_IMAGES="<image> <image> ..."`.
   The target pulls `nvcr.io/nvidia/<image>:1.1.0` and tags it as `<image>`,
   so shipped cookbooks work unchanged. Use `CONTAINER_RUNTIME=podman` on
   Podman hosts and `NGC_TAG=<tag>` only when the user names another release.

5. **Verify.** Rerun the dry-run from Step 1 and confirm every stage command
   uses an image that `docker image ls 'cdf-*'` lists. Then continue with
   [workflow-runner-debugging](workflow-runner-debugging.md) for execution.

## Example

The user wants to run the video data augmentation cookbook without building
images:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/video_data_augmentation/configs/pipeline_video.yaml --container-dry-run'
make pull NGC_IMAGES="cdf-super-resolution-service cdf-detection-and-tracking-sam3-service cdf-captioning-service cdf-visual-qa-service cdf-reasoning-service"
docker image ls 'cdf-*'
```

## Guardrails

- Pull only from `nvcr.io/nvidia/cdf-*`. Do not substitute images from other
  registries or orgs.
- Never print, log, or write an NGC API key, and never put one in a cookbook,
  payload, command argument, or committed file. Always pass it with
  `--password-stdin`.
- Do not run `docker logout`, delete images, or prune the host unless the user
  asks.
- Do not pass `--container-build-images` after pulling, because it rebuilds
  every stage image from source and replaces the pulled ones.
- Pulled images do not change host requirements. GPU stages still need the
  NVIDIA driver, NVIDIA Container Toolkit, and mounted checkpoints.
