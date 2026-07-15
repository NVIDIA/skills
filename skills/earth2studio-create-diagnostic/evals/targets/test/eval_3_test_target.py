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

"""Tests for SuperResolution generative diagnostic model."""

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.models.auto import Package
from earth2studio.models.dx import SuperResolution
from earth2studio.utils import handshake_dim


class PhooSuperResolution(torch.nn.Module):
    """Dummy super-resolution model for testing."""

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # Input: [batch, 4, 91, 180] + noise -> Output: [batch, 4, 361, 720]
        # Simple bilinear interpolation + noise
        out = torch.nn.functional.interpolate(
            x, size=(361, 720), mode="bilinear", align_corners=True
        )
        return out + 0.01 * noise


@pytest.fixture(scope="class")
def test_package(tmp_path_factory):
    """Create dummy package for testing."""
    tmp_path = tmp_path_factory.mktemp("model_data")

    # Save dummy model
    model = PhooSuperResolution()
    torch.save(model, tmp_path / "model.pt")

    # Save normalization parameters
    np.save(tmp_path / "in_center.npy", np.zeros(4))
    np.save(tmp_path / "in_scale.npy", np.ones(4))
    np.save(tmp_path / "out_center.npy", np.zeros(4))
    np.save(tmp_path / "out_scale.npy", np.ones(4))

    return Package(str(tmp_path))


class TestSuperResolutionMock:
    """Mock tests using dummy model weights."""

    @pytest.mark.parametrize("number_of_samples", [1, 3])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_super_resolution_samples(self, test_package, number_of_samples, device):
        """Test that correct number of samples are generated."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("No GPU available")

        dx = SuperResolution.load_model(
            test_package, number_of_samples=number_of_samples
        ).to(device)
        x = torch.randn(1, 4, 91, 180).to(device)

        coords = OrderedDict(
            {
                "batch": np.ones(1),
                "variable": dx.input_coords()["variable"],
                "lat": dx.input_coords()["lat"],
                "lon": dx.input_coords()["lon"],
            }
        )

        out, out_coords = dx(x, coords)

        # Validate sample dimension exists
        assert "sample" in out_coords
        assert len(out_coords["sample"]) == number_of_samples

        # Validate output shape: [batch, samples, var, lat, lon]
        assert out.shape == torch.Size([1, number_of_samples, 4, 361, 720])

        # Validate dimension ordering
        handshake_dim(out_coords, "lon", 4)
        handshake_dim(out_coords, "lat", 3)
        handshake_dim(out_coords, "variable", 2)
        handshake_dim(out_coords, "sample", 1)
        handshake_dim(out_coords, "batch", 0)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_super_resolution_deterministic_seed(self, test_package, device):
        """Test reproducibility with seed."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("No GPU available")

        # Two models with same seed
        dx1 = SuperResolution.load_model(test_package, number_of_samples=2, seed=42).to(
            device
        )
        dx2 = SuperResolution.load_model(test_package, number_of_samples=2, seed=42).to(
            device
        )

        x = torch.randn(1, 4, 91, 180).to(device)
        coords = OrderedDict(
            {
                "batch": np.ones(1),
                "variable": dx1.input_coords()["variable"],
                "lat": dx1.input_coords()["lat"],
                "lon": dx1.input_coords()["lon"],
            }
        )

        out1, _ = dx1(x, coords)
        out2, _ = dx2(x, coords)

        # Same seed should produce same output
        torch.testing.assert_close(out1, out2)

        # Different seed should produce different output
        dx3 = SuperResolution.load_model(
            test_package, number_of_samples=2, seed=123
        ).to(device)
        out3, _ = dx3(x, coords)
        assert not torch.allclose(out1, out3)

    @pytest.mark.parametrize(
        "x",
        [
            torch.randn(1, 4, 91, 180),
            torch.randn(2, 4, 91, 180),
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_super_resolution_call(self, test_package, x, device):
        """Test forward pass with different batch sizes."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("No GPU available")

        dx = SuperResolution.load_model(test_package, number_of_samples=1).to(device)
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

        # Validate output shape: [batch, 1, 4, 361, 720]
        assert out.shape == torch.Size([x.shape[0], 1, 4, 361, 720])

        # Validate output variables
        expected_vars = np.array(["t2m", "u10m", "v10m", "msl"])
        assert np.array_equal(out_coords["variable"], expected_vars)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_super_resolution_exceptions(self, test_package, device):
        """Test SuperResolution raises on invalid coordinates."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("No GPU available")

        dx = SuperResolution.load_model(test_package).to(device)
        x = torch.randn(1, 4, 91, 180).to(device)

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
