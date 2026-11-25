# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the synthetic imaging dataset for the image-statistics eval.

Creates ./imaging_data/site-{1,2,3}/ with 110 seeded 8-bit grayscale PNGs
per site (site-shifted intensity means so per-site histograms differ) plus
one deliberately corrupt file at site-1 to exercise failure_count. 110
images per site clears the bin-cap cleanser for a 10-bin histogram. No
real data; requires numpy and Pillow.
"""

from pathlib import Path

import numpy as np
from PIL import Image

SITE_MEANS = {"site-1": 90, "site-2": 150, "site-3": 118}
IMAGES_PER_SITE = 110
OUTPUT_ROOT = Path("imaging_data")


def main() -> None:
    rng = np.random.default_rng(21)
    for site, mean in SITE_MEANS.items():
        site_dir = OUTPUT_ROOT / site
        site_dir.mkdir(parents=True, exist_ok=True)
        for i in range(IMAGES_PER_SITE):
            pixels = np.clip(rng.normal(mean, 32, (8, 8)), 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(site_dir / f"img_{i:04d}.png")
    (OUTPUT_ROOT / "site-1" / "img_corrupt.png").write_bytes(b"this is not a png")
    print(f"generated {IMAGES_PER_SITE} images per site under {OUTPUT_ROOT}/ (plus one corrupt file at site-1)")


if __name__ == "__main__":
    main()
