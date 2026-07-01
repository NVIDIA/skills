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

"""Additive prognostic model that adds a constant at each time step.

Demonstrates normalization buffers (center and scale) loaded from package.
"""

from collections import OrderedDict
from collections.abc import Iterator

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem

VARIABLES = ["t2m", "u10m", "v10m", "msl"]


class AdditiveModel(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Additive prognostic model.

    A simple model that adds a small constant (0.01) to each variable at
    each time step. Demonstrates normalization with center/scale buffers.

    Parameters
    ----------
    core_model : torch.nn.Module, optional
        Core model (not used, for interface compatibility).
    center : torch.Tensor, optional
        Normalization center values.
    scale : torch.Tensor, optional
        Normalization scale values.

    Note
    ----
    This model demonstrates the use of normalization buffers in E2S wrappers.
    """

    def __init__(
        self,
        core_model: torch.nn.Module | None = None,
        center: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = core_model
        if center is not None:
            self.register_buffer("center", center)
        else:
            self.register_buffer("center", torch.zeros(len(VARIABLES)))
        if scale is not None:
            self.register_buffer("scale", scale)
        else:
            self.register_buffer("scale", torch.ones(len(VARIABLES)))
        self.register_buffer("device_buffer", torch.empty(0))
        self._time_step = np.timedelta64(6, "h")
        self._increment = 0.01

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary.
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 361, endpoint=True),
                "lon": np.linspace(0, 360, 720, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinates to validate and transform.

        Returns
        -------
        CoordSystem
            Output coordinates with updated lead_time.
        """
        target_input_coords = self.input_coords()

        handshake_dim(input_coords, "lead_time", 1)
        handshake_dim(input_coords, "variable", 2)
        handshake_dim(input_coords, "lat", 3)
        handshake_dim(input_coords, "lon", 4)

        handshake_coords(input_coords, target_input_coords, "variable")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "lon")

        output_coords = input_coords.copy()
        output_coords["lead_time"] = input_coords["lead_time"] + np.array(
            [self._time_step]
        )
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained model package.

        Returns
        -------
        Package
            Model package.
        """
        return Package(".")

    @classmethod
    def load_model(cls, package: Package) -> PrognosticModel:
        """Load prognostic model from package.

        Parameters
        ----------
        package : Package
            Model package with model.pt, center.npy, scale.npy.

        Returns
        -------
        PrognosticModel
            Loaded model instance.
        """
        # For testing without files, use defaults
        try:
            model_path = package.resolve("model.pt")
            core_model = torch.load(model_path, map_location="cpu", weights_only=False)
            core_model.eval()
        except Exception:
            core_model = None

        try:
            center_path = package.resolve("center.npy")
            center = torch.from_numpy(np.load(center_path))
        except Exception:
            center = None

        try:
            scale_path = package.resolve("scale.npy")
            scale = torch.from_numpy(np.load(scale_path))
        except Exception:
            scale = None

        return cls(core_model, center=center, scale=scale)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor."""
        center = self.center.view(1, 1, 1, -1, 1, 1)
        scale = self.scale.view(1, 1, 1, -1, 1, 1)
        return (x - center) / scale

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize output tensor."""
        center = self.center.view(1, 1, 1, -1, 1, 1)
        scale = self.scale.view(1, 1, 1, -1, 1, 1)
        return x * scale + center

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Run prognostic model 1 step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor with added constant and coordinates one time step ahead.
        """
        target_input_coords = self.input_coords()
        handshake_coords(coords, target_input_coords, "variable")
        handshake_dim(coords, "variable", 2)

        device = self.device_buffer.device
        x = x.to(device)

        # Add small constant
        output = x + self._increment

        out_coords = self.output_coords(coords)
        return output, out_coords

    @batch_func()
    def _default_generator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Batch-decorated generator for time integration.

        Parameters
        ----------
        x : torch.Tensor
            Initial condition tensor.
        coords : CoordSystem
            Initial coordinate system.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Predicted state and coordinates at each time step.
        """
        yield x, coords

        current_x = x
        current_coords = coords
        while True:
            current_x, current_coords = self.front_hook(current_x, current_coords)
            current_x, current_coords = self.__call__(current_x, current_coords)
            current_x, current_coords = self.rear_hook(current_x, current_coords)
            yield current_x, current_coords

    def create_iterator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Create time-integration iterator.

        Parameters
        ----------
        x : torch.Tensor
            Initial condition tensor.
        coords : CoordSystem
            Initial coordinate system.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Predicted state and coordinates at each time step.
        """
        yield from self._default_generator(x, coords)
