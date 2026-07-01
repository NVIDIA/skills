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

"""Tests for WindSpeed diagnostic model."""

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.models.dx import WindSpeed
from earth2studio.utils import handshake_dim


@pytest.mark.parametrize(
    "x",
    [
        torch.randn(1, 2, 721, 1440),
        torch.randn(2, 2, 721, 1440),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_windspeed_call(x, device):
    """Test forward pass of WindSpeed diagnostic."""
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("No GPU available")

    dx = WindSpeed().to(device)
    x = x.to(device)

    coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )

    out, out_coords = dx(x, coords)

    # Validate output shape: [batch, 1, 721, 1440]
    assert out.shape == torch.Size([x.shape[0], 1, 721, 1440])

    # Validate output variable
    assert np.array_equal(out_coords["variable"], np.array(["ws10m"]))

    # Validate dimension ordering
    handshake_dim(out_coords, "lon", 3)
    handshake_dim(out_coords, "lat", 2)
    handshake_dim(out_coords, "variable", 1)
    handshake_dim(out_coords, "batch", 0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_windspeed_computation(device):
    """Test wind speed is computed correctly."""
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("No GPU available")

    dx = WindSpeed().to(device)

    # Create known input: u=3, v=4 -> ws=5
    x = torch.zeros(1, 2, 721, 1440, device=device)
    x[:, 0, :, :] = 3.0  # u10m
    x[:, 1, :, :] = 4.0  # v10m

    coords = OrderedDict(
        {
            "batch": np.ones(1),
            "variable": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )

    out, _ = dx(x, coords)

    # Wind speed should be 5.0 everywhere
    expected = torch.full((1, 1, 721, 1440), 5.0, device=device)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_windspeed_exceptions(device):
    """Test WindSpeed raises on invalid coordinates."""
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("No GPU available")

    dx = WindSpeed().to(device)
    x = torch.randn(1, 2, 721, 1440).to(device)

    # Test 1: Wrong variable name
    wrong_coords = OrderedDict(
        {
            "batch": np.ones(1),
            "variable": np.array(["wrong_var"]),
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )
    with pytest.raises((KeyError, ValueError)):
        dx(x, wrong_coords)

    # Test 2: Wrong dimension order
    wrong_coords = OrderedDict(
        {
            "batch": np.ones(1),
            "variable": dx.input_coords()["variable"],
            "lon": dx.input_coords()["lon"],
            "lat": dx.input_coords()["lat"],
        }
    )
    with pytest.raises(ValueError):
        dx(x, wrong_coords)

    # Test 3: Wrong lat values
    wrong_coords = OrderedDict(
        {
            "batch": np.ones(1),
            "variable": dx.input_coords()["variable"],
            "lat": np.linspace(-90, 90, 721),  # Wrong direction
            "lon": dx.input_coords()["lon"],
        }
    )
    with pytest.raises(ValueError):
        dx(x, wrong_coords)
