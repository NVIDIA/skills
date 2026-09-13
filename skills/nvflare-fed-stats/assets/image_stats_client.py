# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Template for an image-intensity federated statistics client.

Adapt this template when generating the client:
- swap ``_load_image`` for the format the site actually holds: PIL covers
  PNG/JPEG/BMP/TIFF; DICOM needs ``pydicom`` (or MONAI's ITK reader), NIfTI
  needs ``nibabel`` — preflight the import for the chosen loader and fail
  closed on a missing dependency;
- ``count`` is the number of discovered image files; unreadable files are
  detected up front by a full-decode verification in ``initialize()`` so
  ``failure_count`` (which must be configured explicitly to appear in
  output) is correct from the first statistics round; verification uses
  ``_load_image`` itself, so it cannot disagree with the histogram pass,
  which remains a deduplicated backstop for transient read errors; if a
  file fails later, the histogram output is computed only from still-readable
  files and the updated ``failure_count`` lets the standard privacy filters
  withhold output based on the honest effective count;
- grayscale conversion is applied before histogramming; state that policy in
  the report (multi-channel per-channel statistics are not supported here);
- parameterize the data location by site identity; do not hardcode one
  site's absolute path.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.statistics_spec import Bin, DataType, Feature, Histogram, HistogramType, Statistics

# ADAPTATION POINT: extend with the site's actual formats (e.g. ".dcm",
# ".nii", ".nii.gz") when swapping _load_image for the matching loader; the
# dataset block's per-site image_formats says which formats are present.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class ImageIntensityStatistics(Statistics):
    def __init__(self, data_root_dir: str, dataset_name: str = "data"):
        """Local pixel-intensity statistics generator for image folders.

        Args:
            data_root_dir: root directory holding one subdirectory per site.
            dataset_name: dataset label used in the output.
        """
        super().__init__()
        self.data_root_dir = data_root_dir
        self.dataset_name = dataset_name
        self.image_paths: Optional[List[Path]] = None
        # failed paths are deduplicated so a retried histogram() call cannot
        # count the same unreadable file twice
        self.failed_paths: set = set()

    def initialize(self, fl_ctx: FLContext):
        site_name = fl_ctx.get_identity_name()
        site_dir = Path(self.data_root_dir) / site_name
        # ADAPTATION POINT: adjust discovery to the site's layout (recursive
        # globs, a datalist JSON, or format-specific extensions).
        self.image_paths = sorted(p for p in site_dir.iterdir() if p.name.lower().endswith(IMAGE_EXTENSIONS))
        if not self.image_paths:
            raise ValueError(f"no image files found under {site_dir}")
        # Verify readability up front with the SAME operation the histogram
        # pass uses, so failure_count is correct in the FIRST statistics
        # round (histogramming only happens in round 2, after the server has
        # collected round-1 failure counts) and cannot disagree with the
        # later read: header-only checks miss broken pixel data. Trade-off:
        # every image is decoded twice; for very large datasets an adapted
        # client may relax this and accept round-2-only failure discovery.
        for path in self.image_paths:
            try:
                self._load_image(path)
            except Exception:
                self.failed_paths.add(path)
        self.log_info(
            fl_ctx,
            f"site {site_name}: {len(self.image_paths)} image files, {len(self.failed_paths)} unreadable",
        )

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        # ADAPTATION POINT: replace with the loader for the site's format
        # (pydicom for DICOM, nibabel for NIfTI); keep the returned array as
        # raw intensities.
        from PIL import Image

        return np.asarray(Image.open(path).convert("L"))

    def features(self) -> Dict[str, List[Feature]]:
        return {self.dataset_name: [Feature("intensity", DataType.FLOAT)]}

    def count(self, dataset_name: str, feature_name: str) -> int:
        return len(self.image_paths)

    def failure_count(self, dataset_name: str, feature_name: str) -> int:
        return len(self.failed_paths)

    def histogram(
        self, dataset_name: str, feature_name: str, num_of_bins: int, global_min_value: float, global_max_value: float
    ) -> Histogram:
        totals = np.zeros((num_of_bins,), dtype=np.int64)
        edges = None
        for path in self.image_paths:
            if path in self.failed_paths:
                continue  # known-bad since initialize(); do not decode again
            try:
                arr = self._load_image(path)
                counts, edges = np.histogram(arr, bins=num_of_bins, range=(global_min_value, global_max_value))
                totals += counts
            except Exception:
                # passed verification but failed now: the data changed (or a
                # transient read error) AFTER count/failure_count were already
                # consumed by the controller. Do not raise here: the statistics
                # task handler catches per-statistic exceptions and can return
                # partial output. Instead update failure_count and let the
                # standard result filters screen this histogram against the
                # honest effective count in this second-round payload.
                self.failed_paths.add(path)
                self.log_warning(
                    None,
                    f"{path} failed after passing verification; histogram excludes it and "
                    "failure_count was updated for privacy filtering",
                    fire_event=False,
                )
        if edges is None:
            # every image at this site failed: round-1 count/failure_count
            # already surface the data-quality condition, so return
            # zero-count bins instead of aborting the federated job for the
            # healthy sites
            _, edges = np.histogram(np.empty(0), bins=num_of_bins, range=(global_min_value, global_max_value))
        bins = [
            Bin(low_value=float(edges[j]), high_value=float(edges[j + 1]), sample_count=int(totals[j]))
            for j in range(num_of_bins)
        ]
        return Histogram(HistogramType.STANDARD, bins)
