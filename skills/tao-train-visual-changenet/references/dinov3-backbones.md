# Frozen DINOv3 backbones

Visual ChangeNet classify can use these frozen DINOv3 backbones:

| `model.backbone.type` | Hugging Face/timm model |
|---|---|
| `vit_small_dinov3` | `timm/vit_small_patch16_dinov3.lvd1689m` |
| `vit_small_plus_dinov3` | `timm/vit_small_plus_patch16_dinov3.lvd1689m` |
| `vit_base_dinov3` | `timm/vit_base_patch16_dinov3.lvd1689m` |
| `vit_large_dinov3` | `timm/vit_large_patch16_dinov3.lvd1689m` |
| `vit_huge_plus_dinov3` | `timm/vit_huge_plus_patch16_dinov3.lvd1689m` |
| `vit_7b_dinov3` | `timm/vit_7b_patch16_dinov3.lvd1689m` |

Accept the DINOv3 license and export `HF_TOKEN` before a gated download.
DINOv3 is supported only as a frozen Visual ChangeNet backbone:

```yaml
model:
  backbone:
    type: vit_base_dinov3
    freeze_backbone: true
    # Empty selects the matching timm/Hugging Face weights at container startup.
    pretrained_backbone_path: ''
```

Pass `HF_TOKEN` into the container when `pretrained_backbone_path` is empty.
This automatic fetch is DINOv3-specific; C-RADIO paths are always local files.

For a reproducible or air-gapped standalone run, use Hugging Face tooling to
stage `model.safetensors` from the mapped repository, then point the spec at
its in-container mount. When the full skill bank is installed, the DEFT helper
can stage any supported profile and reads `HF_TOKEN` only on the host:

```bash
python3 skills/applications/tao-run-deft-aoi/scripts/stage_backbone.py \
  --workspace <workspace> --backbone-type vit_base_dinov3
# -> <workspace>/augmentation/backbone/vit_base_dinov3.safetensors
```

Then mount that exact file and replace the empty value above with, for example,
`/data/pretrained_models/vit_base_dinov3.safetensors`. The backbone type,
staged filename, and checkpoint architecture must agree. In the full DEFT AOI
workflow, follow `tao-run-deft-aoi/references/visual-changenet.md`; its preflight
validates this pairing and rewrites the container path.
