# Synthetic TensorRT-LLM module test prepared for an evaluation exercise.
import pytest
import torch
from utils.util import skip_pre_blackwell


@skip_pre_blackwell
class TestMoeEpScaling:

    @pytest.mark.skip_less_device(4)
    def test_ep4_nvfp4(self):
        assert torch.cuda.device_count() >= 4

    @pytest.mark.skip_less_device(16)
    def test_ep16_nvfp4(self):
        assert torch.cuda.device_count() >= 16
