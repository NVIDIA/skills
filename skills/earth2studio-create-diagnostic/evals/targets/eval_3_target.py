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

"""Probabilistic super-resolution diagnostic model.

This diagnostic model performs probabilistic super-resolution similar to CorrDiff,
generating multiple samples from a coarse input grid to a fine output grid.
"""

from collections import OrderedDict

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem

INPUT_VARIABLES = ["t2m", "u10m", "v10m", "msl"]
OUTPUT_VARIABLES = ["t2m", "u10m", "v10m", "msl"]


class SuperResolution(torch.nn.Module, AutoModelMixin):
    """Probabilistic super-resolution diagnostic model.

    Performs probabilistic super-resolution from a coarse 2-degree grid
    to a fine 0.5-degree grid, generating multiple samples per input.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core model for super-resolution
    in_center : torch.Tensor
        Input center normalization tensor
    in_scale : torch.Tensor
        Input scale normalization tensor
    out_center : torch.Tensor
        Output center normalization tensor
    out_scale : torch.Tensor
        Output scale normalization tensor
    number_of_samples : int, optional
        Number of samples to generate per input, by default 1
    seed : int | None, optional
        Random seed for reproducibility, by default None

    Note
    ----
    Similar to CorrDiff, this model generates multiple high-resolution
    samples from a single low-resolution input using probabilistic methods.
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        in_center: torch.Tensor,
        in_scale: torch.Tensor,
        out_center: torch.Tensor,
        out_scale: torch.Tensor,
        number_of_samples: int = 1,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.core_model = core_model
        self.number_of_samples = number_of_samples
        self.seed = seed

        # Register normalization buffers
        self.register_buffer("in_center", in_center)
        self.register_buffer("in_scale", in_scale)
        self.register_buffer("out_center", out_center)
        self.register_buffer("out_scale", out_scale)

        # Output grid dimensions
        self._out_lat = np.linspace(90, -90, 361, endpoint=True)
        self._out_lon = np.linspace(0, 360, 720, endpoint=False)

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of diagnostic model.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary for coarse 2-degree grid.
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(INPUT_VARIABLES),
                "lat": np.linspace(90, -90, 91, endpoint=True),
                "lon": np.linspace(0, 360, 180, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system with sample dimension.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinates to validate and transform.

        Returns
        -------
        CoordSystem
            Output coordinates including sample dimension.
        """
        target_input_coords = self.input_coords()

        # Validate input dimensions
        handshake_dim(input_coords, "variable", 1)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "lon", 3)

        # Validate coordinate values
        handshake_coords(input_coords, target_input_coords, "variable")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "lon")

        # Output includes sample dimension
        output_coords = OrderedDict(
            {
                "batch": input_coords["batch"],
                "sample": np.arange(self.number_of_samples),
                "variable": np.array(OUTPUT_VARIABLES),
                "lat": self._out_lat,
                "lon": self._out_lon,
            }
        )
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
            "ngc://models/nvidia/modulus/super_resolution@v1.0",
            cache_options={
                "cache_storage": Package.default_cache("super_resolution"),
                "same_names": True,
            },
        )

    @classmethod
    def load_model(
        cls,
        package: Package,
        number_of_samples: int = 1,
        seed: int | None = None,
    ) -> DiagnosticModel:
        """Load generative diagnostic from package.

        Parameters
        ----------
        package : Package
            Model package with checkpoint files.
        number_of_samples : int, optional
            Number of samples to generate, by default 1
        seed : int | None, optional
            Random seed for reproducibility, by default None

        Returns
        -------
        DiagnosticModel
            Loaded model instance.
        """
        # Resolve checkpoint files
        model_path = package.resolve("model.pt")
        in_center_path = package.resolve("in_center.npy")
        in_scale_path = package.resolve("in_scale.npy")
        out_center_path = package.resolve("out_center.npy")
        out_scale_path = package.resolve("out_scale.npy")

        # Load model
        core_model = torch.load(model_path, map_location="cpu", weights_only=False)
        core_model.eval()
        core_model.requires_grad_(False)

        # Load normalization parameters
        in_center = torch.from_numpy(np.load(in_center_path)).view(4, 1, 1)
        in_scale = torch.from_numpy(np.load(in_scale_path)).view(4, 1, 1)
        out_center = torch.from_numpy(np.load(out_center_path)).view(4, 1, 1)
        out_scale = torch.from_numpy(np.load(out_scale_path)).view(4, 1, 1)

        return cls(
            core_model,
            in_center=in_center,
            in_scale=in_scale,
            out_center=out_center,
            out_scale=out_scale,
            number_of_samples=number_of_samples,
            seed=seed,
        )

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor."""
        return (x - self.in_center) / self.in_scale

    def _denormalize_output(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize output tensor."""
        return x * self.out_scale + self.out_center

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Internal forward pass generating samples.

        Parameters
        ----------
        x : torch.Tensor
            Single input tensor [var, lat, lon].

        Returns
        -------
        torch.Tensor
            Generated samples [samples, var, lat, lon].
        """
        device = x.device

        # Normalize input
        x_norm = self._normalize_input(x.unsqueeze(0))

        # Generate samples
        samples = []
        for i in range(self.number_of_samples):
            # Set seed for reproducibility if provided
            if self.seed is not None:
                torch.manual_seed(self.seed + i)

            # Generate noise for stochastic generation
            noise = torch.randn(1, 4, 361, 720, device=device)

            # Forward pass through model
            sample = self.core_model(x_norm, noise)
            samples.append(sample)

        # Stack samples: [samples, var, lat, lon]
        out = torch.cat(samples, dim=0)

        # Denormalize
        out = self._denormalize_output(out)

        return out

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass generating samples.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, variable, lat, lon).
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor (batch, sample, variable, lat, lon) and coordinates.
        """
        output_coords = self.output_coords(coords)

        with torch.no_grad():
            # Allocate output tensor
            out = torch.zeros(
                [len(v) for v in output_coords.values()],
                device=x.device,
                dtype=torch.float32,
            )

            # Generate samples for each batch element
            for i in range(out.shape[0]):
                out[i] = self._forward(x[i])

        return out, output_coords
