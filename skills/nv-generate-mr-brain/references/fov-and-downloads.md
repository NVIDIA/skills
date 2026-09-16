# FOV And Downloads

Use this reference when choosing NV-Generate-CTMR brain MR image-only settings.

## Field Of View

FOV is `dim * spacing` in millimeters. MR-Brain v1 ships an axial T1w
default of `[256, 256, 128]` at `[0.94, 0.94, 1.36]` mm, or approximately
`240 x 240 x 174` mm. Rounded axial starting points derived from the pinned
upstream v1 FOV table are:

| Modality | `dim` | `spacing` (mm) | Approximate FOV (mm) |
|---|---:|---:|---:|
| T1w | `[256, 256, 128]` | `[0.94, 0.94, 1.36]` | `241 x 241 x 174` |
| T2w | `[256, 256, 128]` | `[0.94, 0.94, 1.23]` | `241 x 241 x 157` |
| FLAIR | `[256, 256, 128]` | `[0.98, 0.98, 1.37]` | `251 x 251 x 175` |
| SWI | `[256, 256, 128]` | `[0.90, 0.90, 1.13]` | `230 x 230 x 145` |
| MRA | `[256, 256, 128]` | `[0.86, 0.86, 1.23]` | `220 x 220 x 157` |

Whole-brain and skull-stripped variants use the same FOV guidance. The pinned
upstream source also provides sagittal and coronal medians. MRA in every plane,
and sagittal/coronal SWI, have sparse training coverage; output quality is not
guaranteed. Keep dimensions as multiples of 32 and spacing positive. Use the
`nv-generate-mr` skill for non-brain body MR.

## Downloads

For reproducible brain MR generation, download the reused autoencoder and v1
diffusion checkpoint from the exact manifest revisions:

```bash
python -m huggingface_hub.commands.huggingface_cli download \
  nvidia/NV-Generate-CT models/autoencoder_v1.pt \
  --revision 75ac080fb1083c403793563477724c038e7d430c \
  --local-dir "$NV_GENERATE_ROOT"
python -m huggingface_hub.commands.huggingface_cli download \
  nvidia/NV-Generate-MR-Brain models/diff_unet_3d_rflow-mr-brain_v1.pt \
  --revision ef9759bf221265b2704569cdeeac20bbf03b62ee \
  --local-dir "$NV_GENERATE_ROOT"
```

This path does not use ControlNet, mask generation, or the CT mask database.
Cached model weights do not imply Python packages are installed. Fresh
benchmark environments should still run:

```bash
python -m pip install -r "$NV_GENERATE_ROOT/requirements.txt"
```
