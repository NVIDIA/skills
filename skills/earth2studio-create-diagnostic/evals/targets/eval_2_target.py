# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Precipitation estimator diagnostic model with AutoModel support.

This diagnostic model estimates total precipitation from atmospheric variables
using a neural network loaded from a model package.
"""

from collections import OrderedDict

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem

INPUT_VARIABLES = ["t2m", "u10m", "v10m", "msl", "tcwv"]
OUTPUT_VARIABLES = ["tp"]


class PrecipEstimator(torch.nn.Module, AutoModelMixin):
    """Precipitation estimator diagnostic model.

    Estimates total precipitation from atmospheric variables using
    a neural network. Loads weights from a model package.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core pytorch model for precipitation estimation
    center : torch.Tensor
        Model input center normalization tensor of size [5, 1, 1]
    scale : torch.Tensor
        Model input scale normalization tensor of size [5, 1, 1]

    Note
    ----
    This model estimates 6-hourly total precipitation from atmospheric
    state variables on a 0.5-degree grid.
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.core_model = core_model
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of diagnostic model.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary.
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(INPUT_VARIABLES),
                "lat": np.linspace(90, -90, 361, endpoint=True),
                "lon": np.linspace(0, 360, 720, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of diagnostic model.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinates to validate and transform.

        Returns
        -------
        CoordSystem
            Output coordinates with tp variable.
        """
        target_input_coords = self.input_coords()

        # Validate dimensions
        handshake_dim(input_coords, "variable", 1)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "lon", 3)

        # Validate coordinate values
        handshake_coords(input_coords, target_input_coords, "variable")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "lon")

        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(OUTPUT_VARIABLES)
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained model package.

        Returns
        -------
        Package
            Model package with checkpoint files.
        """
        return Package(
            "hf://nvidia/precip-estimator@0123456789abcdef0123456789abcdef01234567",
            cache_options={
                "cache_storage": Package.default_cache("precip_estimator"),
                "same_names": True,
            },
        )

    @classmethod
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load diagnostic model from package.

        Parameters
        ----------
        package : Package
            Model package with checkpoint files.

        Returns
        -------
        DiagnosticModel
            Loaded model instance.
        """
        # Resolve checkpoint files
        model_path = package.resolve("model.pt")
        center_path = package.resolve("center.npy")
        scale_path = package.resolve("scale.npy")

        # Load model
        core_model = torch.load(model_path, map_location="cpu", weights_only=False)
        core_model.eval()
        core_model.requires_grad_(False)

        # Load normalization parameters
        center = torch.from_numpy(np.load(center_path)).view(5, 1, 1)
        scale = torch.from_numpy(np.load(scale_path)).view(5, 1, 1)

        return cls(core_model, center=center, scale=scale)

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, 5, lat, lon).
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Precipitation tensor (batch, 1, lat, lon) and coordinates.
        """
        output_coords = self.output_coords(coords)

        with torch.no_grad():
            # Move to device
            device = next(self.parameters()).device
            x = x.to(device)

            # Normalize
            x = (x - self.center) / self.scale

            # Forward pass
            out = self.core_model(x)

        return out, output_coords
