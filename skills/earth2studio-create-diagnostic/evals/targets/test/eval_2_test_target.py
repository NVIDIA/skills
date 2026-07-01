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

"""Tests for PrecipEstimator diagnostic model."""

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.models.auto import Package
from earth2studio.models.dx import PrecipEstimator
from earth2studio.utils import handshake_dim


class PhooPrecipEstimator(torch.nn.Module):
    """Dummy model for testing PrecipEstimator."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [batch, 5, lat, lon] -> Output: [batch, 1, lat, lon]
        return x[:, :1, :, :]


@pytest.fixture(scope="class")
def test_package(tmp_path_factory):
    """Create dummy package for testing."""
    tmp_path = tmp_path_factory.mktemp("model_data")

    # Save dummy model
    model = PhooPrecipEstimator()
    torch.save(model, tmp_path / "model.pt")

    # Save normalization parameters
    np.save(tmp_path / "center.npy", np.zeros(5))
    np.save(tmp_path / "scale.npy", np.ones(5))

    return Package(str(tmp_path))


class TestPrecipEstimatorMock:
    """Mock tests using dummy model weights."""

    @pytest.mark.parametrize(
        "x",
        [
            torch.randn(1, 5, 361, 720),
            torch.randn(2, 5, 361, 720),
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_precip_estimator_call(self, test_package, x, device):
        """Test forward pass of PrecipEstimator."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("No GPU available")

        dx = PrecipEstimator.load_model(test_package).to(device)
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

        # Validate output shape: [batch, 1, 361, 720]
        assert out.shape == torch.Size([x.shape[0], 1, 361, 720])

        # Validate output variable
        assert np.array_equal(out_coords["variable"], np.array(["tp"]))

        # Validate dimension ordering
        handshake_dim(out_coords, "lon", 3)
        handshake_dim(out_coords, "lat", 2)
        handshake_dim(out_coords, "variable", 1)
        handshake_dim(out_coords, "batch", 0)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_precip_estimator_exceptions(self, test_package, device):
        """Test PrecipEstimator raises on invalid coordinates."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("No GPU available")

        dx = PrecipEstimator.load_model(test_package).to(device)
        x = torch.randn(1, 5, 361, 720).to(device)

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
                "lat": np.linspace(-90, 90, 361),
                "lon": dx.input_coords()["lon"],
            }
        )
        with pytest.raises(ValueError):
            dx(x, wrong_coords)

    def test_precip_estimator_normalization(self, test_package):
        """Test that normalization is applied correctly."""
        dx = PrecipEstimator.load_model(test_package)

        # Check buffers exist
        assert hasattr(dx, "center")
        assert hasattr(dx, "scale")
        assert dx.center.shape == torch.Size([5, 1, 1])
        assert dx.scale.shape == torch.Size([5, 1, 1])
