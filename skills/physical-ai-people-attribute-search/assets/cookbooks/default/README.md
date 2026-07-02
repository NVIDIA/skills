# PAS Default Cookbook

Default attribute distribution and augmentation pipeline configuration for the
Person Attribute Search (PAS) image augmentation workflow.

## Files

| File | Purpose |
|---|---|
| `distribution_config.yaml` | Attribute sampling distributions: top color/type, bottom color/type, shoe color/type with near-uniform probabilities |
| `augmentation_config.yaml` | Pipeline config for `modules/cli.py`: captioning template, MCQ verification options, model endpoints, retry policy |

## Attribute Space

The default distribution covers:

- **Top outer color** (13 values): beige, black, blue, brown, green, grey, orange, pink, purple, red, white, yellow, maroon
- **Top outer type** (7 values): hoodie, cropped jacket, sweater, puffer jacket, denim jacket, blazer, vest
- **Bottom type** (5 values): jeans, leggings, shorts, skirt, cargo pants
- **Bottom color** (8 values): black, blue, brown, green, grey, khaki, navy, white
- **Shoe type** (5 values): sneakers, boots, sandals, loafers, barefoot
- **Shoe color** (conditional on shoe type): varies per shoe type

## Customization

To create a new cookbook (for example, for a specific attribute subset):

1. Copy this directory to `cookbooks/<your_cookbook>/`
2. Edit `distribution_config.yaml` to adjust attribute probabilities
3. Edit `augmentation_config.yaml` to tune verification options or prompt template
4. Pass `cookbook=<your_cookbook>` at submit time
