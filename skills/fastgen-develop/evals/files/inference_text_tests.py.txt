# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for inference entry points and core student sampling."""

from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import pytest
import torch

from fastgen.configs.config import BaseCheckpointerConfig
from fastgen.datasets.crop_resize import transform_image
from fastgen.datasets.wds_dataloaders import transform_video
from fastgen.methods import CausalSFTModel, CausVidModel
from fastgen.methods.model import FastGenModel
from fastgen.networks.network import FastGenModule, FastGenNetwork, pixel_shape_from_factors
from fastgen.networks.network_causal import CausalFastGenNetwork
from fastgen.networks.noise_schedule import (
    AlphasNoiseSchedule,
    BaseNoiseSchedule,
    CompositeNoiseSchedule,
    EDMNoiseSchedule,
    RFNoiseSchedule,
    StudentGridConfig,
    get_noise_schedule,
)
from fastgen.utils.global_vars import set_temp_global_vars
from fastgen.utils.checkpointer import Checkpointer
from fastgen.utils.tree_utils import get_batch_size, get_device, get_dtype, per_modality, randn_like
from scripts.inference import inference_utils
from scripts.inference import flashdreams_inference as flashdreams
from scripts.inference import image_model_inference as image_inference
from scripts.inference import video_model_inference as video_inference
from scripts.inference import vlm_infer
from tests.test_configs import build_config


def test_inference_frontends_import_shared_preparation_helpers():
    assert image_inference.load_source_image_for_net is inference_utils.load_source_image_for_net
    assert video_inference.load_conditioning_image_for_net is inference_utils.load_conditioning_image_for_net
    assert video_inference.load_source_video_for_net is inference_utils.load_source_video_for_net
    assert video_inference.load_raw_depth_latent is inference_utils.load_raw_depth_latent
    assert video_inference.prepare_depth_latent is inference_utils.prepare_depth_latent
    assert video_inference.load_camera_poses is inference_utils.load_camera_poses
    assert video_inference.prepare_camera_poses is inference_utils.prepare_camera_poses
    assert video_inference.expand_prompts_with_qwen is inference_utils.expand_prompts_with_qwen


def test_load_checkpoint_returns_only_the_iteration(tmp_path):
    checkpoint_path = tmp_path / "0000017.pth"
    (tmp_path / "0000017.net_model").mkdir()
    checkpoint_path.touch()  # completion marker of the sharded checkpoint
    loaded_paths = []

    def load(model_dict, path, strict):
        if path != str(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {path}")
        assert list(model_dict) == ["net"]
        assert strict is False
        loaded_paths.append(path)
        return 17

    model = SimpleNamespace(net=torch.nn.Identity(), ema_dict={})
    checkpointer = SimpleNamespace(load=load)

    assert inference_utils.load_checkpoint(checkpointer, model, str(checkpoint_path)) == 17
    assert loaded_paths == [str(checkpoint_path)]
    assert inference_utils.load_checkpoint(checkpointer, model, "") is None
    # A missing explicit path must not fall back to the pretrained network silently.
    with pytest.raises(FileNotFoundError, match="not found"):
        inference_utils.load_checkpoint(checkpointer, model, str(tmp_path / "0000099.pth"))
    assert loaded_paths == [str(checkpoint_path)]


def test_load_checkpoint_rejects_legacy_prefixes(tmp_path):
    checkpointer = Checkpointer(BaseCheckpointerConfig(save_dir=str(tmp_path)))
    model = SimpleNamespace(net=torch.nn.Identity(), ema_dict={})
    with pytest.raises(ValueError, match=r"must end with \.pth"):
        inference_utils.load_checkpoint(checkpointer, model, str(tmp_path / "0000017"))


def test_load_checkpoint_leaves_format_dispatch_to_the_checkpointer(tmp_path):
    checkpoint_prefix = tmp_path / "0000017"
    marker_path = Path(f"{checkpoint_prefix}.pth")
    marker_path.touch()
    metadata_path = Path(f"{checkpoint_prefix}.net_model") / ".metadata"
    metadata_path.parent.mkdir()
    metadata_path.touch()
    loaded_paths = []

    class RecordingCheckpointer(Checkpointer):
        def load(self, model_dict, path, strict):
            assert list(model_dict) == ["net"]
            loaded_paths.append(path)
            return 17

    checkpointer = RecordingCheckpointer(BaseCheckpointerConfig(save_dir=str(tmp_path)))
    model = SimpleNamespace(net=torch.nn.Identity(), ema_dict={})

    assert inference_utils.load_checkpoint(checkpointer, model, str(marker_path)) == 17
    assert loaded_paths == [str(marker_path)]


@pytest.mark.parametrize("saved_logvar", [False, True], ids=["checkpoint-without-logvar", "checkpoint-with-logvar"])
def test_load_checkpoint_tolerates_a_logvar_head_mismatch_unless_strict(tmp_path, saved_logvar):
    class Net(torch.nn.Module):
        def __init__(self, logvar: bool):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)
            if logvar:
                self.logvar_linear = torch.nn.Linear(2, 1)

    checkpointer = Checkpointer(BaseCheckpointerConfig(save_dir=str(tmp_path)))
    source = Net(logvar=saved_logvar)
    path = checkpointer.save(torch.nn.ModuleDict({"net": source}), iteration=17)

    target = Net(logvar=not saved_logvar)
    assert inference_utils.load_checkpoint(checkpointer, SimpleNamespace(net=target, ema_dict={}), path) == 17
    torch.testing.assert_close(target.proj.state_dict(), source.proj.state_dict(), rtol=0, atol=0)

    strict_model = SimpleNamespace(net=Net(logvar=not saved_logvar), ema_dict={})
    with pytest.raises(RuntimeError, match="Unexpected key" if saved_logvar else "Missing key"):
        inference_utils.load_checkpoint(checkpointer, strict_model, path, strict=True)


def test_default_inference_output_dir_is_checkpoint_specific(tmp_path):
    assert inference_utils.default_inference_output_dir(tmp_path, "", None) == tmp_path / "inference"
    assert (
        inference_utils.default_inference_output_dir(tmp_path, "/runs/example/0000017.pth", 17) == tmp_path / "0000017"
    )
    assert inference_utils.default_inference_output_dir(tmp_path, "/runs/example/17.pth", 17) == tmp_path / "17"
    assert (
        inference_utils.default_inference_output_dir(tmp_path, "/runs/example/0000017.pth", 18)
        == tmp_path / "0000017_0000018"
    )
    assert (
        inference_utils.default_inference_output_dir(tmp_path, "/runs/example/latest_ar.pth", 23)
        == tmp_path / "latest_ar_0000023"
    )
    assert (
        inference_utils.default_inference_output_dir(tmp_path, "/runs/example/latest_ar.pth", 24)
        == tmp_path / "latest_ar_0000024"
    )
    assert (
        inference_utils.default_inference_output_dir(tmp_path, "/runs/example/latest_ar_0000023.pth", 23)
        == tmp_path / "latest_ar_0000023"
    )


def test_prepare_video_frames_matches_dataloader_temporal_and_spatial_processing():
    frames = (torch.arange(5 * 6 * 10 * 3).reshape(5, 6, 10, 3) % 256).to(torch.uint8)

    expected = transform_video(frames, sequence_length=3, img_size=(8, 4))
    actual = inference_utils.prepare_video_frames(frames, num_frames=3, height=4, width=8)

    torch.testing.assert_close(actual["real_raw"], expected["real_raw"].unsqueeze(0))
    assert actual["cropping_params"] == expected["cropping_params"]
    with pytest.raises(ValueError, match="video too short"):
        inference_utils.prepare_video_frames(frames[:2], num_frames=3, height=4, width=8)


def _geometry_net(factors, **attributes):
    """A stand-in network whose ``pixel_shape`` applies ``factors``."""
    return SimpleNamespace(
        pixel_shape=lambda latent_shape: pixel_shape_from_factors(latent_shape, factors), **attributes
    )


def test_decoded_image_preparation_accepts_comfyui_tensors(monkeypatch):
    source_net = _geometry_net((2, 2))
    image = torch.linspace(0, 1, 1 * 5 * 9 * 3).reshape(1, 5, 9, 3)

    source = inference_utils.prepare_source_image_for_net(image, source_net, [4, 2, 3])

    assert source.shape == (1, 3, 4, 6)
    assert source.dtype == torch.float32
    assert source.min() >= -1
    assert source.max() <= 1
    legacy = Image.fromarray((image[0] * 255).round().to(torch.uint8).numpy()).convert("RGB")
    assert torch.equal(source, transform_image(legacy, (6, 4)).unsqueeze(0))

    rgba = Image.fromarray(np.arange(5 * 9 * 4, dtype=np.uint8).reshape(5, 9, 4))
    assert torch.equal(
        inference_utils.prepare_source_image_for_net(rgba, source_net, [4, 2, 3]),
        transform_image(rgba, (6, 4)).unsqueeze(0),
    )

    captured = {}
    video_net = _geometry_net((4, 8, 16))

    def fake_prepare(frames, num_frames, height, width):
        captured.update(frames=frames, num_frames=num_frames, height=height, width=width)
        return {"real_raw": "IMAGE", "cropping_params": {"crop": "metadata"}}

    monkeypatch.setattr(inference_utils, "prepare_video_frames", fake_prepare)

    prepared = inference_utils.prepare_conditioning_image_for_net(
        image, video_net, {"video": [16, 5, 10, 20], "audio": None}
    )
    assert prepared == {"real_raw": "IMAGE", "cropping_params": {"crop": "metadata"}}
    torch.testing.assert_close(captured.pop("frames"), image)
    assert captured == {"num_frames": 1, "height": 80, "width": 320}


def test_supplied_image_paths_fail_at_the_loading_boundary(monkeypatch, tmp_path):
    import imageio.v3 as iio

    with pytest.raises(FileNotFoundError, match="Source image not found"):
        inference_utils.load_source_image_for_net(tmp_path / "missing.png", object(), [4, 2, 2])
    with pytest.raises(FileNotFoundError, match="Conditioning image not found"):
        inference_utils.load_conditioning_image_for_net(tmp_path / "missing.png", object(), [4, 2, 2, 2])

    invalid_path = tmp_path / "invalid.png"
    invalid_path.touch()

    def fail_decode(*_args, **_kwargs):
        raise OSError("invalid")

    monkeypatch.setattr(iio, "imread", fail_decode)
    with pytest.raises(RuntimeError, match="Failed to decode conditioning image"):
        inference_utils.load_conditioning_image_for_net(invalid_path, object(), [4, 2, 2, 2])


def test_shared_segment_and_depth_preparation():
    shape = {"video": [16, 5, 8, 8], "audio": None}
    assert inference_utils.total_latent_frames(shape, num_segments=3, overlap_frames=2) == 11

    depth = torch.arange(2 * 13 * 3 * 4).reshape(2, 13, 3, 4)
    prepared = inference_utils.prepare_depth_latent(depth, latent_frame_count=11)
    assert prepared.shape == (1, 2, 11, 3, 4)
    torch.testing.assert_close(prepared[0], depth[:, :11])


@pytest.mark.parametrize("depth_shape", [(2, 11, 3, 4), (1, 2, 11, 3, 4)], ids=["unbatched", "batched"])
def test_depth_preparation_requires_the_latent_to_cover_the_span(depth_shape):
    """A short depth latent would leave the uncovered frames without control instead of failing."""
    depth = torch.zeros(depth_shape)
    assert inference_utils.prepare_depth_latent(depth, latent_frame_count=11).shape == (1, 2, 11, 3, 4)
    with pytest.raises(ValueError, match="Depth latent has 11 latent frames but the generated video spans 12"):
        inference_utils.prepare_depth_latent(depth, latent_frame_count=12)


def test_load_raw_depth_latent_uses_tensor_only_cpu_loading(monkeypatch, tmp_path):
    """The depth latent path is CLI input, so it must not go through the general unpickler."""
    depth_path = tmp_path / "depth.pt"
    torch.save(torch.arange(2 * 4 * 3 * 3).reshape(2, 4, 3, 3), depth_path)
    seen = {}
    real_load = torch.load

    def recording_load(path, *args, **kwargs):
        seen.update(kwargs)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(inference_utils.torch, "load", recording_load)

    depth = inference_utils.load_raw_depth_latent(depth_path)

    assert seen == {"map_location": "cpu", "weights_only": True}
    assert depth.shape == (2, 4, 3, 3)


def test_shared_source_video_preparation_uses_latent_frame_count(monkeypatch):
    captured = {}

    def fake_prepare(frames, num_frames, height, width):
        captured.update(frames=frames, num_frames=num_frames, height=height, width=width)
        return {"real_raw": "VIDEO", "cropping_params": {"crop": "metadata"}}

    monkeypatch.setattr(inference_utils, "prepare_video_frames", fake_prepare)
    frames = torch.zeros(33, 2, 2, 3)
    net = _geometry_net((4, 8, 16), fps=16)

    prepared = inference_utils.prepare_source_video_for_net(
        frames,
        net,
        {"video": [16, 5, 10, 20], "audio": None},
        latent_frame_count=9,
    )

    assert prepared == {
        "real_raw": "VIDEO",
        "cropping_params": {"crop": "metadata"},
        "source_frame_indices": list(range(33)),
    }
    torch.testing.assert_close(captured.pop("frames"), frames)
    assert captured == {"num_frames": 33, "height": 80, "width": 320}


def test_source_video_preparation_follows_the_minimax_h3_frame_mapping(monkeypatch):
    """MiniMax-H3's 37 latent frames span 124 pixel frames, so FL2VA's last keyframe is the clip's last frame."""
    from fastgen.networks.MiniMaxH3.network import MiniMaxH3

    captured = {}

    def fake_prepare(frames, num_frames, height, width):
        captured.update(num_frames=num_frames, height=height, width=width)
        return {"real_raw": "VIDEO", "cropping_params": {}}

    monkeypatch.setattr(inference_utils, "prepare_video_frames", fake_prepare)

    prepared = inference_utils.prepare_source_video_for_net(
        torch.zeros(150, 2, 2, 3),
        object.__new__(MiniMaxH3),
        {"video": [24, 37, 34, 60], "audio": None},
        latent_frame_count=37,
    )

    assert captured == {"num_frames": 124, "height": 544, "width": 960}
    assert prepared["source_frame_indices"] == list(range(124))


def test_source_video_loader_uses_training_full_decoder(monkeypatch, tmp_path):
    from fastgen.datasets import decoders

    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"video")
    frames = torch.zeros(33, 2, 2, 3)
    net = SimpleNamespace(fps=16)
    shape = {"video": [16, 5, 10, 20], "audio": None}
    captured = {}
    source_frame_indices = list(range(10, 43))

    def fake_decode(key, data, **kwargs):
        captured.update(key=key, data=data, decode_kwargs=kwargs)
        return {"frames": frames, "source_frame_indices": source_frame_indices}

    monkeypatch.setattr(decoders, "decode_video_full", fake_decode)

    def fake_prepare(received_frames, received_net, input_shape, latent_frame_count, source_frame_indices=None):
        captured.update(
            frames=received_frames,
            net=received_net,
            input_shape=input_shape,
            latent_frame_count=latent_frame_count,
            source_frame_indices=source_frame_indices,
        )
        return {
            "real_raw": "VIDEO",
            "cropping_params": {"crop": "metadata"},
            "source_frame_indices": source_frame_indices,
        }

    monkeypatch.setattr(inference_utils, "prepare_source_video_for_net", fake_prepare)

    prepared = inference_utils.load_source_video_for_net(video_path, net, shape, 9)
    assert prepared == {
        "real_raw": "VIDEO",
        "cropping_params": {"crop": "metadata"},
        "source_frame_indices": source_frame_indices,
    }
    assert captured.pop("frames") is frames
    assert captured.pop("net") is net
    assert captured == {
        "key": "source.mp4",
        "data": b"video",
        "decode_kwargs": {
            "output_format": "torch",
            "target_fps": net.fps,
            "timeout_s": None,
            "return_source_frame_indices": True,
        },
        "input_shape": shape,
        "latent_frame_count": 9,
        "source_frame_indices": source_frame_indices,
    }


def test_source_video_and_camera_alignment_matches_training_full_loader(tmp_path):
    from fastgen.datasets.wds_dataloaders import CameraVideoWDSLoader
    from tests.test_dataloader import DummyWebDatasetBuilder

    source_fps, target_fps = 16, 8
    num_source_frames = 32
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (num_source_frames, 32, 48, 3), dtype=np.uint8)
    encoder = DummyWebDatasetBuilder(str(tmp_path / "unused"))
    mp4 = encoder._create_dummy_mp4(frames, fps=source_fps)
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(mp4)

    poses = np.tile(np.eye(4, dtype=np.float32), (num_source_frames, 1, 1))
    poses[:, 0, 3] = np.arange(num_source_frames)
    intrinsics = np.tile(np.array([40.0, 40.0, 24.0, 16.0], dtype=np.float32), (num_source_frames, 1))

    dataset_dir = tmp_path / "camera_dataset"
    DummyWebDatasetBuilder(str(dataset_dir), "00000.tar").add_sample(
        "sample_0000",
        {
            "mp4": mp4,
            "poses.npy": poses,
            "intrinsics.npy": intrinsics,
            "txt": "caption",
        },
    ).build()
    loader = CameraVideoWDSLoader(
        datatags=[f"WDS:{dataset_dir}"],
        batch_size=1,
        num_workers=0,
        key_map={
            "real_raw": "mp4",
            "camera_poses": "poses.npy",
            "camera_intrinsics": "intrinsics.npy",
            "condition_raw": "txt",
        },
        sequence_length=5,
        img_size=(48, 32),
        target_fps=target_fps,
        decoder_type="full",
        train=False,
    )
    training = next(iter(loader))

    net = _geometry_net((4, 1, 1), fps=target_fps)
    inference = inference_utils.load_source_video_for_net(
        video_path,
        net,
        [3, 2, 32, 48],
        latent_frame_count=2,
    )
    camera_poses, camera_intrinsics = inference_utils.prepare_camera_poses(
        poses,
        intrinsics,
        inference["cropping_params"],
        inference["source_frame_indices"],
    )

    assert inference["source_frame_indices"] == [0, 2, 4, 6, 8]
    torch.testing.assert_close(inference["real_raw"], training["real_raw"], rtol=0, atol=0)
    torch.testing.assert_close(camera_poses, training["camera_poses"], rtol=0, atol=0)
    torch.testing.assert_close(camera_intrinsics, training["camera_intrinsics"], rtol=0, atol=0)


def test_video_decoder_raises_at_the_input_boundary(monkeypatch, tmp_path):
    from fastgen.datasets import decoders

    video_path = tmp_path / "broken.mp4"
    video_path.write_bytes(b"broken")
    net = SimpleNamespace(fps=16)
    shape = {"video": [16, 5, 10, 20], "audio": None}

    monkeypatch.setattr(decoders, "decode_video_full", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match=r"Failed to decode source video"):
        inference_utils.load_source_video_for_net(video_path, net, shape, 5)


def test_camera_preparation_matches_dataloader_indices_and_crop():
    from fastgen.utils.camera_utils import transform_intrinsics

    frames = torch.zeros(4, 6, 10, 3, dtype=torch.uint8)
    prepared_video = inference_utils.prepare_video_frames(frames, num_frames=2, height=4, width=8)
    poses = torch.eye(4).repeat(4, 1, 1)
    poses[:, 0, 3] = torch.arange(4)
    intrinsics = torch.tensor([[8.0, 8.0, 5.0, 3.0]]).repeat(4, 1)
    indices = [1, 3]

    prepared_poses, prepared_intrinsics = inference_utils.prepare_camera_poses(
        poses,
        intrinsics,
        prepared_video["cropping_params"],
        indices,
    )

    torch.testing.assert_close(prepared_poses[0], poses[indices])
    torch.testing.assert_close(
        prepared_intrinsics[0],
        transform_intrinsics(intrinsics[indices], prepared_video["cropping_params"]),
    )


def test_shared_prompt_expansion_supports_images_and_system_prompt(monkeypatch):
    from fastgen.third_party.wan_prompt_expand import prompt_expand

    captured = {}

    class FakePromptExpander:
        def __init__(self, model_name, is_vl, device):
            captured.update(model_name=model_name, is_vl=is_vl, device=device)

        def __call__(self, prompt, **kwargs):
            captured.update(prompt=prompt, kwargs=kwargs)
            return SimpleNamespace(prompt="expanded prompt")

    monkeypatch.setattr(prompt_expand, "QwenPromptExpander", FakePromptExpander)
    image = object()
    device = torch.device("cpu")

    expanded = inference_utils.expand_prompts_with_qwen(
        ["short prompt"],
        "QwenVL2.5_3B",
        device,
        7,
        images=[image],
        system_prompt="system",
    )

    assert expanded == ["expanded prompt"]
    assert captured["model_name"] == "QwenVL2.5_3B"
    assert captured["is_vl"] is True
    assert captured["device"] == device
    assert captured["prompt"] == "short prompt"
    assert captured["kwargs"]["image"] is image
    assert captured["kwargs"]["system_prompt"] == "system"


def test_prompt_expansion_uses_rank0_broadcast_on_nonzero_rank(monkeypatch):
    import fastgen.utils.distributed.collectives as collectives
    from fastgen.third_party.wan_prompt_expand import prompt_expand

    class UnexpectedPromptExpander:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("nonzero rank must not instantiate the prompt expander")

    captured = {}

    def fake_broadcast(value, device=None):
        captured.update(value=value, device=device)
        return ["expanded on rank 0"]

    monkeypatch.setattr(prompt_expand, "QwenPromptExpander", UnexpectedPromptExpander)
    monkeypatch.setattr(collectives, "get_rank", lambda: 1)
    monkeypatch.setattr(collectives, "broadcast_object", fake_broadcast)

    assert inference_utils.expand_prompts_with_qwen(
        ["short prompt"],
        "Qwen2.5_3B",
        torch.device("cpu"),
        7,
    ) == ["expanded on rank 0"]
    assert captured == {"value": None, "device": None}


@pytest.mark.parametrize("rank", [0, 1])
def test_prompt_expansion_failure_reaches_all_ranks_before_broadcast(monkeypatch, rank):
    import fastgen.utils.distributed.collectives as collectives
    from fastgen.third_party.wan_prompt_expand import prompt_expand

    constructed = []
    broadcasts = []

    class FailingPromptExpander:
        def __init__(self, *_args, **_kwargs):
            constructed.append(rank)
            raise OSError("expander download failed")

    monkeypatch.setattr(prompt_expand, "QwenPromptExpander", FailingPromptExpander)
    # Exercise the actual error guard with its transport simulated: rank 0's failure reaches peers.
    monkeypatch.setattr(collectives, "world_size", lambda: 2)
    monkeypatch.setattr(collectives, "get_rank", lambda: rank)
    monkeypatch.setattr(collectives.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(collectives, "sync_min", lambda value, device: 0)
    monkeypatch.setattr(
        collectives,
        "broadcast_object",
        lambda value, **kwargs: broadcasts.append(value) or "OSError: expander download failed",
    )

    with pytest.raises(
        RuntimeError, match="prompt expansion: rank 0 failed: OSError: expander download failed"
    ) as error:
        inference_utils.expand_prompts_with_qwen(["short prompt"], "Qwen2.5_3B", torch.device("cpu"), 7)

    assert constructed == ([0] if rank == 0 else [])
    assert isinstance(error.value.__cause__, OSError) if rank == 0 else error.value.__cause__ is None
    assert len(broadcasts) == 1, "only the failure report may be broadcast after a failed expansion"


class _TinyInferenceNetwork(FastGenNetwork):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)

    def forward(self, x_t, t, **kwargs):
        del t, kwargs
        return self.proj(x_t.to(dtype=self.precision_model)).to(dtype=x_t.dtype)


class _TinyPreprocessor(FastGenModule):
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Linear(2, 2)


@pytest.mark.parametrize("checkpoint_parent", ["checkpoints", "0000007.pth/checkpoints"])
@pytest.mark.parametrize("mode", ["normal", "partial", "preview_rank0", "preview_rank1"])
def test_fid_checkpoint_sweep_prepares_once_before_loading(tmp_path, monkeypatch, checkpoint_parent, mode):
    from scripts.fid import compute_fid_from_ckpts as fid_sweep

    checkpointer_config = BaseCheckpointerConfig(save_dir=str(tmp_path / checkpoint_parent))
    checkpointer = Checkpointer(checkpointer_config, save_sharded=True)
    source = torch.nn.ModuleDict({"net": _TinyInferenceNetwork()})
    for iteration in (1, 2):
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(iteration)
        checkpointer.save(source, iteration=iteration)
    # A directory is never a checkpoint, even when its name looks like a marker.
    (tmp_path / checkpoint_parent / "0000009.pth").mkdir()
    if mode == "partial":
        # An interrupted sweep can leave directories without a complete set of images.
        (tmp_path / "samples/iter_1/000000").mkdir(parents=True)

    net = _TinyInferenceNetwork()
    with torch.no_grad():
        net.proj.weight.fill_(-1.0)
    net.label_dim = 0
    preparations = []
    sampled_weights = []

    def prepare():
        # Preparation still sees the initial values: every checkpoint loads into the prepared net.
        preparations.append(net.proj.weight[0, 0].item())
        net.prepare_module(device="cpu", precision_model=torch.float32)

    def generate(network, noise, **kwargs):
        assert network is net and net.is_prepared
        sampled_weights.append(net.proj.weight[0, 0].item())
        return torch.zeros_like(noise)

    model = SimpleNamespace(
        net=net,
        model_dict=torch.nn.ModuleDict({"net": net}),
        use_ema=[],
        net_preprocessors=net,
        device=torch.device("cpu"),
        precision=torch.float32,
        input_shape=(3, 2, 2),
        on_train_begin=prepare,
        prepare_batch=lambda data, **kwargs: data,
        generator_fn=generate,
    )
    config = SimpleNamespace(
        model_class=None,
        model=SimpleNamespace(inference=True),
        trainer=SimpleNamespace(seed=1, checkpointer=checkpointer_config),
        log_config=SimpleNamespace(save_path=str(tmp_path)),
        dataloader_train=SimpleNamespace(dataset_path="cifar10-32x32.zip", batch_size=1),
        eval=SimpleNamespace(
            samples_dir="samples", num_samples=2, min_ckpt=0, max_ckpt=10, save_images=mode.startswith("preview")
        ),
    )
    monkeypatch.setattr(fid_sweep, "instantiate", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(fid_sweep, "synchronize", lambda: None)
    monkeypatch.setattr(fid_sweep, "world_size", lambda: 2)
    monkeypatch.setattr(fid_sweep, "get_rank", lambda: int(mode == "preview_rank1"))
    monkeypatch.setattr(fid_sweep.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(fid_sweep, "calc", lambda *_args, **_kwargs: None)

    fid_sweep.main(config)

    assert preparations == [-1.0]
    assert sampled_weights == [1.0, 2.0]
    assert len(list((tmp_path / "samples").rglob("*.png"))) == (0 if mode == "preview_rank1" else 2)


def test_inference_loads_dcp_before_preparing_selected_ema(tmp_path):
    source = torch.nn.ModuleDict({"net": _TinyInferenceNetwork(), "ema": _TinyInferenceNetwork()})
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters()):
            parameter.fill_(index + 0.25)
    checkpointer = Checkpointer(BaseCheckpointerConfig(save_dir=str(tmp_path)), save_sharded=True)
    marker = checkpointer.save(source, iteration=17)
    # Single-process meta targets have no real source rank; two-rank meta loading is covered by the checkpointer tests.
    net = _TinyInferenceNetwork()
    ema = _TinyInferenceNetwork()
    events = []
    model = SimpleNamespace(
        net=net,
        ema=ema,
        teacher=None,
        ema_dict={"ema": ema},
        use_ema=["ema"],
        module_dict={"net": net, "ema": ema},
        init_preprocessors=lambda: ema,
        apply_torch_compile=lambda: None,
    )

    def prepare(module, name):
        assert events == ["loaded"]
        assert module is ema and name == "ema"
        module.prepare_module(device="cpu", precision_model=torch.float64, precision_opt=torch.float64, trainable=False)
        events.append("prepared")

    model.prepare_module = prepare

    assert inference_utils.load_checkpoint(checkpointer, model, marker) == 17
    for name, target in model.module_dict.items():
        assert not target.is_prepared
        for key, value in target.state_dict().items():
            assert value.device.type == "cpu"
            torch.testing.assert_close(value, source[name].state_dict()[key], rtol=0, atol=0)
    events.append("loaded")

    assert inference_utils.setup_inference_modules(model, False, True) == (None, ema, ema)
    assert events == ["loaded", "prepared"]
    assert model.module_dict == {"ema": ema} and model.net is None
    for key, value in ema.state_dict().items():
        assert value.dtype == torch.float64
        torch.testing.assert_close(value, source["ema"].state_dict()[key].double(), rtol=0, atol=0)


def test_inference_setup_prepares_each_selected_network_once():
    teacher = _TinyInferenceNetwork()
    student = _TinyInferenceNetwork()
    placements = []
    model = SimpleNamespace(
        teacher=teacher,
        net=student,
        use_ema=[],
        device=torch.device("cpu"),
        precision=torch.float32,
        config=SimpleNamespace(student_sample_steps=1),
        module_dict={"net": student, "teacher": teacher},
        init_preprocessors=lambda fsdp_context=None: student,
        apply_torch_compile=lambda: None,
        prepare_module=lambda module, name, fsdp_context=None: placements.append((name, module)),
    )

    result = inference_utils.setup_inference_modules(model, True, True)

    assert result == (teacher, student, student)
    # Both sampling networks are prepared, the student first.
    assert placements == [("net", student), ("teacher", teacher)]


def test_inference_setup_prunes_unused_roots_before_preprocessors_and_compile():
    net = _TinyInferenceNetwork()
    teacher = _TinyInferenceNetwork()
    student = _TinyInferenceNetwork()
    extra_ema = _TinyInferenceNetwork()
    pruned_vae = _TinyPreprocessor()
    kept_vae = _TinyPreprocessor()
    dropped_helper = _TinyPreprocessor()
    net.preprocessors["pruned_vae"] = pruned_vae
    student.preprocessors["vae"] = kept_vae
    placements = []
    model = SimpleNamespace(
        net=net,
        teacher=teacher,
        ema=student,
        ema_extra=extra_ema,
        use_ema=["ema", "ema_extra"],
        module_dict={
            "net": net,
            "teacher": teacher,
            "ema": student,
            "ema_extra": extra_ema,
            "pruned_vae": pruned_vae,
            "vae": kept_vae,
            "dropped_helper": dropped_helper,
        },
    )

    def prepare_module(module, name, fsdp_context=None):
        del fsdp_context
        placements.append((name, module))

    model.prepare_module = prepare_module

    def assert_pruned():
        assert model.net is None
        assert model.teacher is None
        assert model.ema is student
        assert model.ema_extra is None
        assert model.module_dict == {"ema": student, "vae": kept_vae}

    def init_preprocessors(fsdp_context=None):
        del fsdp_context
        assert_pruned()
        return student

    def apply_torch_compile():
        assert_pruned()

    model.init_preprocessors = init_preprocessors
    model.apply_torch_compile = apply_torch_compile

    result = inference_utils.setup_inference_modules(model, False, True)

    assert result == (None, student, student)
    assert placements == [("ema", student)]
    assert model.module_dict == {"ema": student, "vae": kept_vae}


def test_inference_setup_builds_a_missing_teacher_on_demand():
    """A method that never built a teacher gets a real one, not the base net under its name."""
    network = _TinyInferenceNetwork()
    student = _TinyInferenceNetwork()
    built = _TinyInferenceNetwork()
    placements = []
    model = SimpleNamespace(
        net=network,
        ema=student,
        use_ema=["ema"],
        module_dict={"net": network, "ema": student},
        init_preprocessors=lambda fsdp_context=None: student,
        apply_torch_compile=lambda: None,
        prepare_module=lambda module, name, fsdp_context=None: placements.append((name, module)),
    )

    def build_teacher():
        # Mirrors ``FastGenModel.build_teacher``: instantiate, then register as its own root.
        model.teacher = built
        model.module_dict["teacher"] = built

    model.build_teacher = build_teacher

    result = inference_utils.setup_inference_modules(model, True, True)

    assert model.teacher is built
    assert result == (built, student, student)
    assert placements == [("ema", student), ("teacher", built)]
    # The base net hosts neither sampled network once the teacher is its own root.
    assert model.module_dict == {"ema": student, "teacher": built}
    assert model.net is None


def test_inference_setup_builds_a_missing_teacher_without_a_student():
    """The built teacher is the only surviving root when student sampling is disabled."""
    network = _TinyInferenceNetwork()
    built = _TinyInferenceNetwork()
    placements = []
    model = SimpleNamespace(
        net=network,
        use_ema=[],
        module_dict={"net": network},
        init_preprocessors=lambda fsdp_context=None: built,
        apply_torch_compile=lambda: None,
        prepare_module=lambda module, name, fsdp_context=None: placements.append((name, module)),
    )

    def build_teacher():
        model.teacher = built
        model.module_dict["teacher"] = built

    model.build_teacher = build_teacher

    result = inference_utils.setup_inference_modules(model, True, False)

    assert result == (built, None, built)
    assert placements == [("teacher", built)]
    assert model.module_dict == {"teacher": built}
    assert model.net is None


def test_inference_setup_rejects_disabling_all_sampling_before_mutation():
    network = _TinyInferenceNetwork()
    calls = []
    model = SimpleNamespace(
        net=network,
        use_ema=[],
        module_dict={"net": network},
        init_preprocessors=lambda fsdp_context=None: calls.append("init_preprocessors"),
        apply_torch_compile=lambda: calls.append("apply_torch_compile"),
        prepare_module=lambda module, name, fsdp_context=None: calls.append((name, module)),
    )

    with pytest.raises(ValueError, match="At least one of teacher or student sampling must be enabled"):
        inference_utils.setup_inference_modules(model, False, False)

    assert model.net is network
    assert model.module_dict == {"net": network}
    assert calls == []


def test_inference_cli_defaults_to_student_only():
    parser = argparse.ArgumentParser()
    inference_utils.add_common_args(parser)

    args = parser.parse_args([])

    assert args.do_student_sampling is True
    assert args.do_teacher_sampling is False


@pytest.mark.parametrize("do_student_sampling", [False, True])
def test_inference_setup_prepares_shared_base_teacher_once(do_student_sampling):
    from fastgen.networks.lora_utils import SharedBaseTeacher

    student = _TinyInferenceNetwork()
    teacher = SharedBaseTeacher(student)
    placements = []
    model = SimpleNamespace(
        teacher=teacher,
        net=student,
        use_ema=[],
        module_dict={"net": student},
        init_preprocessors=lambda fsdp_context=None: student,
        apply_torch_compile=lambda: None,
        prepare_module=lambda module, name, fsdp_context=None: placements.append((name, module)),
    )

    result = inference_utils.setup_inference_modules(model, True, do_student_sampling)

    assert result == (teacher, student if do_student_sampling else None, student)
    assert placements == [("net", student)]
    assert model.net is student
    assert model.module_dict == {"net": student}


class _TextEncoder:
    """Small deterministic stand-in supporting tensor and tuple embeddings."""

    def __init__(self, output: str = "tensor"):
        self.output = output

    def encode(self, prompts):
        embeddings = torch.tensor(
            [[len(prompt), sum(map(ord, prompt)) % 997] for prompt in prompts],
            dtype=torch.float32,
        )
        if self.output == "tuple":
            return embeddings, torch.ones(embeddings.shape[0], 2, dtype=torch.int64)
        return embeddings


class _VAE(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def encode(self, value, **_kwargs):
        if isinstance(self.input_shape, Mapping):
            return {
                key: torch.zeros(value[key].shape[0], *shape, device=value[key].device, dtype=value[key].dtype)
                for key, shape in self.input_shape.items()
            }
        return torch.zeros(value.shape[0], *self.input_shape, device=value.device, dtype=value.dtype)


class _ConditioningNet(torch.nn.Module):
    """Makes every conditioning input observable in the final sample call."""

    encode_real = FastGenNetwork.encode_real
    pixel_shape = FastGenNetwork.pixel_shape
    latent_compression_factors = (1, 1, 1)

    def __init__(
        self,
        input_shape,
        required_fields=(),
        text_output: str | None = "tensor",
        is_image_edit: bool = False,
    ):
        super().__init__()
        self.required_fields = tuple(required_fields)
        self.is_image_edit = is_image_edit
        self.fps = 24.0
        self.preprocessing_modes = []
        self.preprocessors = {}
        if text_output is not None:
            self.preprocessors["text_encoder"] = _TextEncoder(text_output)
        if is_image_edit or any(field in self.required_fields for field in ("real_raw", "depth_latent")):
            self.preprocessors["vae"] = _VAE(input_shape)

    def sample(self, *_args, **_kwargs):  # only used by the CLI's capability check
        raise AssertionError("The wrapper model, rather than net.sample, should perform teacher sampling")

    def build_conditioning(self, data, input_shape=None, precision=None):
        self.preprocessing_modes.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
        missing = [field for field in self.required_fields if field not in data]
        if missing:
            raise AssertionError(f"Missing task conditioning fields: {missing}")

        shared = {}
        for field in self.required_fields:
            if field == "condition_image_raw":
                shared["condition_image"] = self.preprocessors["vae"].encode(data[field], mode="argmax")
            else:
                shared[field] = data[field]

        def resolve(encoded_key, raw_key):
            value = data.get(encoded_key)
            raw = data.get(raw_key)
            if value is None and raw is not None:
                encoder = self.preprocessors.get("text_encoder")
                if encoder is None:
                    value = torch.tensor([[len(prompt)] for prompt in raw], dtype=torch.long)
                else:
                    value = encoder.encode(raw)
            return value

        def branch(value, raw_key):
            result = {"text": value, **shared}
            if data.get(raw_key) is not None:
                result["text_raw"] = data[raw_key]
            return result

        condition = resolve("condition", "condition_raw")
        neg_condition = resolve("neg_condition", "neg_condition_raw")
        out = {}
        if condition is not None:
            out["condition"] = branch(condition, "condition_raw")
        if neg_condition is not None:
            out["neg_condition"] = branch(neg_condition, "neg_condition_raw")
        return out


class _InferenceModel:
    _SAMPLE_STEPS_UNSET = object()
    state_fields = FastGenModel.state_fields
    prepare_batch = FastGenModel.prepare_batch

    def __init__(self, net, input_shape):
        self.net = net
        self.net_preprocessors = net
        self.input_shape = input_shape
        self.device = torch.device("cpu")
        self.precision = torch.float32
        self.precision_amp_enc = None
        self.precision_amp_infer = None
        self.use_ema = []
        self.sample_calls = []
        self.generator_calls = []

    @staticmethod
    def reshard_fsdp_modules(preprocessors_only=False):
        del preprocessors_only

    def sample(
        self,
        teacher,
        noise,
        condition=None,
        neg_condition=None,
        sample_steps=_SAMPLE_STEPS_UNSET,
    ):
        call = {
            "teacher": teacher,
            "noise": noise,
            "condition": condition,
            "neg_condition": neg_condition,
        }
        if sample_steps is not self._SAMPLE_STEPS_UNSET:
            call["sample_steps"] = sample_steps
        self.sample_calls.append(call)
        return noise

    def generator_fn(self, student, noise, condition=None, neg_condition=None):
        self.generator_calls.append(
            {
                "student": student,
                "noise": noise,
                "condition": condition,
                "neg_condition": neg_condition,
            }
        )
        return noise


def _assert_tree_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, Mapping):
        assert isinstance(expected, Mapping)
        assert set(actual) == set(expected)
        for key in actual:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_tree_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _config(tmp_path, input_shape):
    return SimpleNamespace(
        model_class=object(),
        model=SimpleNamespace(input_shape=input_shape),
        trainer=SimpleNamespace(seed=123, checkpointer=object()),
        log_config=SimpleNamespace(save_path=str(tmp_path)),
    )


def _patch_inference_runtime(monkeypatch, module, model, tmp_path, teacher_override=None, save_calls=None):
    monkeypatch.setattr(module, "instantiate", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(module, "Checkpointer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(module, "load_checkpoint", lambda *_args, **_kwargs: 1)

    def setup_inference_modules(_model, do_teacher_sampling, do_student_sampling):
        teacher_net = model.net if teacher_override is None else teacher_override
        teacher = teacher_net if do_teacher_sampling else None
        student = model.net if do_student_sampling else None
        return teacher, student, model.net

    monkeypatch.setattr(module, "setup_inference_modules", setup_inference_modules)

    def save_media(*args, **kwargs):
        if save_calls is not None:
            save_calls.append((args, kwargs))

    monkeypatch.setattr(module, "save_media", save_media)
    monkeypatch.setattr(module, "cuda_timer", lambda *_args, **_kwargs: contextlib.nullcontext())
    monkeypatch.setattr(module.basic_utils, "set_random_seed", lambda *_args, **_kwargs: 123)


def _training_conditioning(model, data):
    return model.prepare_batch(data)


def _assert_default_or_explicit_steps(sample_call, sample_steps):
    if sample_steps is None:
        assert "sample_steps" not in sample_call
    else:
        assert sample_call["sample_steps"] == sample_steps


def _assert_sampling_received_no_conditioning(model):
    for call in (model.generator_calls[0], model.sample_calls[0]):
        assert call["condition"] is None
        assert call["neg_condition"] is None


@pytest.mark.parametrize(
    "task",
    [
        "unconditional",
        "unconditional-text",
        "unconditional-class",
        "class-conditional",
        "text-to-image",
        "image-editing",
    ],
)
def test_image_inference_conditioning_matches_training(task, tmp_path, monkeypatch):
    input_shape = [4, 2, 2]
    prompt = "turn the sky green" if task in ("text-to-image", "image-editing") else "2"
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(prompt + "\n")

    source = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)
    input_image_file = None
    required_fields = ()
    is_image_edit = task == "image-editing"
    if is_image_edit:
        source_path = tmp_path / "source.png"
        source_path.touch()
        input_image_file = tmp_path / "source_images.txt"
        input_image_file.write_text(str(source_path) + "\n")
        required_fields = ("condition_image_raw",)

    uses_text = task in ("unconditional-text", "text-to-image", "image-editing")
    net = _ConditioningNet(
        input_shape,
        required_fields=required_fields,
        text_output="tuple" if is_image_edit else ("tensor" if uses_text else None),
        is_image_edit=is_image_edit,
    )
    model = _InferenceModel(net, input_shape)
    _patch_inference_runtime(monkeypatch, image_inference, model, tmp_path)
    monkeypatch.setattr(image_inference, "load_source_image_for_net", lambda *_args, **_kwargs: source)

    classes = 4 if task in ("unconditional-class", "class-conditional") else None
    unconditional = task in ("unconditional", "unconditional-text", "unconditional-class")
    args = SimpleNamespace(
        ckpt_path="",
        classes=classes,
        do_student_sampling=True,
        do_teacher_sampling=True,
        image_save_dir=str(tmp_path / "images"),
        input_image_file=str(input_image_file) if input_image_file else None,
        neg_prompt_file=None,
        num_samples=1,
        prompt_file=str(prompt_file),
        sample_steps=None,
        unconditional=unconditional,
    )

    image_inference.main(args, _config(tmp_path, input_shape))
    assert len(model.sample_calls) == 1
    assert len(model.generator_calls) == 1
    assert net.preprocessing_modes == [(True, False)]
    sample_call = model.sample_calls[0]
    _assert_default_or_explicit_steps(sample_call, None)

    if task in ("unconditional", "unconditional-text", "unconditional-class"):
        # Unavailable branches stay absent rather than present-but-None.
        training_data = {}
    elif task == "class-conditional":
        # ``ImageDataset`` pairs every label with the zero null vector; the script spells out the same.
        condition = torch.zeros(1, classes)
        condition[0, int(prompt)] = 1
        training_data = {"condition": condition, "neg_condition": torch.zeros_like(condition)}
    else:
        # ``neg_prompt_file=None`` deactivates the negative branch.
        training_data = {"condition_raw": [prompt]}
        if is_image_edit:
            training_data["condition_image_raw"] = source

    training = _training_conditioning(model, training_data)
    assert net.preprocessing_modes[-1] == (False, False)
    # Unavailable branches are absent from the built conditioning, not present-but-None.
    _assert_tree_equal(sample_call["condition"], training.get("condition"))
    _assert_tree_equal(sample_call["neg_condition"], training.get("neg_condition"))
    _assert_tree_equal(model.generator_calls[0]["condition"], training.get("condition"))
    _assert_tree_equal(model.generator_calls[0]["neg_condition"], training.get("neg_condition"))
    if is_image_edit:
        assert "condition_image" in sample_call["condition"]
        assert "condition_image_raw" not in sample_call["condition"]


def test_image_inference_accepts_omitted_conditioning_branches(tmp_path, monkeypatch):
    """Conditioning builders may omit unavailable positive and negative branches."""

    input_shape = [4, 2, 2]
    net = _ConditioningNet(input_shape)
    net.build_conditioning = lambda _data, input_shape=None, precision=None: {}
    model = _InferenceModel(net, input_shape)
    save_calls = []
    _patch_inference_runtime(monkeypatch, image_inference, model, tmp_path, save_calls=save_calls)

    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("a paper boat crossing a pond\n")
    args = SimpleNamespace(
        ckpt_path="",
        classes=None,
        do_student_sampling=True,
        do_teacher_sampling=True,
        image_save_dir=None,
        input_image_file=None,
        neg_prompt_file=None,
        num_samples=1,
        prompt_file=str(prompt_file),
        sample_steps=None,
        unconditional=False,
    )

    image_inference.main(args, _config(tmp_path, input_shape))

    _assert_sampling_received_no_conditioning(model)
    assert {Path(call_args[1]).parent for call_args, _kwargs in save_calls} == {tmp_path / "inference" / "prompts"}


def _run_image_negative_prompt_inference(prompt_lines, neg_prompt_lines, tmp_path, monkeypatch):
    input_shape = [4, 2, 2]
    net = _ConditioningNet(input_shape)
    requests = []

    def build_conditioning(data, input_shape=None, precision=None):
        requests.append(data)
        return {}

    net.build_conditioning = build_conditioning
    model = _InferenceModel(net, input_shape)
    _patch_inference_runtime(monkeypatch, image_inference, model, tmp_path)

    args = SimpleNamespace(
        ckpt_path="",
        classes=None,
        do_student_sampling=True,
        do_teacher_sampling=True,
        image_save_dir=str(tmp_path / "images"),
        input_image_file=None,
        neg_prompt_file=(
            None if neg_prompt_lines is None else _write_lines(tmp_path / "negative.txt", *neg_prompt_lines)
        ),
        num_samples=1,
        prompt_file=_write_lines(tmp_path / "prompts.txt", *prompt_lines),
        sample_steps=None,
        unconditional=False,
    )

    image_inference.main(args, _config(tmp_path, input_shape))
    return requests


@pytest.mark.parametrize(
    ("prompt_lines", "neg_prompt_lines", "expected"),
    [
        (("a paper boat crossing a pond",), None, None),
        (("a paper boat crossing a pond",), ("",), [""]),
        (("a paper boat crossing a pond",), ("a negative prompt",), ["a negative prompt"]),
        (("a pond", "a river"), ("shared",), ["shared", "shared"]),
        (("a pond", "a river"), ("first", "second"), ["first", "second"]),
        (("a pond", "a river", "a lake"), ("first", "second"), ["first", "second", "second"]),
        (("a pond", "a river"), ("first", "second", "third"), ["first", "second"]),
    ],
    ids=["deactivated", "blank-line", "file-with-prompt", "shared", "one-per-prompt", "extended", "trimmed"],
)
def test_image_inference_negative_prompt_file_resolution(
    prompt_lines, neg_prompt_lines, expected, tmp_path, monkeypatch
):
    """The image entry point resolves ``--neg_prompt_file`` exactly like the video one."""

    requests = _run_image_negative_prompt_inference(prompt_lines, neg_prompt_lines, tmp_path, monkeypatch)

    assert len(requests) == len(prompt_lines)
    if expected is None:
        assert all("neg_condition_raw" not in request for request in requests)
    else:
        assert [request["neg_condition_raw"] for request in requests] == [[prompt] for prompt in expected]


def test_image_inference_spells_out_class_conditional_null_negative(tmp_path, monkeypatch):
    """Class-conditional runs spell out the zero null class for batched CFG, as ``ImageDataset`` does."""

    input_shape = [4, 2, 2]
    net = _ConditioningNet(input_shape, text_output=None)
    requests = []

    def build_conditioning(data, input_shape=None, precision=None):
        requests.append(data)
        return {key: data[key] for key in ("condition", "neg_condition") if key in data}

    net.build_conditioning = build_conditioning
    model = _InferenceModel(net, input_shape)
    _patch_inference_runtime(monkeypatch, image_inference, model, tmp_path)

    args = SimpleNamespace(
        ckpt_path="",
        classes=4,
        do_student_sampling=True,
        do_teacher_sampling=True,
        image_save_dir=str(tmp_path / "images"),
        input_image_file=None,
        neg_prompt_file=None,
        num_samples=1,
        prompt_file=_write_lines(tmp_path / "prompts.txt", "3"),
        sample_steps=None,
        unconditional=False,
    )

    image_inference.main(args, _config(tmp_path, input_shape))

    assert len(requests) == 1
    assert requests[0]["condition"].argmax().item() == 3
    null_class = torch.zeros_like(requests[0]["condition"])
    assert torch.equal(requests[0]["neg_condition"], null_class)
    assert torch.equal(model.sample_calls[0]["neg_condition"], null_class)
    assert torch.equal(model.generator_calls[0]["neg_condition"], null_class)


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (("shared",), ["shared"] * 3),
        (("first", "second"), ["first", "second", "second"]),
        (("first", "second", "third"), ["first", "second", "third"]),
        (("first", "second", "third", "fourth"), ["first", "second", "third"]),
    ],
    ids=["one-for-all", "extended", "exact", "trimmed"],
)
def test_load_file_entries_returns_the_requested_count(lines, expected, tmp_path):
    path = _write_lines(tmp_path / "prompts.txt", *lines)
    assert inference_utils.load_file_entries(path, num=3) == expected


def test_load_file_entries_keeps_blank_lines_as_empty_prompts(tmp_path):
    """Whitespace-only lines read as empty prompts, so a file can request several empty-prompt samples."""
    assert inference_utils.load_file_entries(_write_lines(tmp_path / "blanks.txt", "", "  ", "")) == ["", "", ""]
    assert inference_utils.load_file_entries(_write_lines(tmp_path / "mixed.txt", "a cat", "", "a dog")) == [
        "a cat",
        "",
        "a dog",
    ]


def test_load_file_entries_passes_through_an_unset_file():
    """Optional per-prompt inputs hand ``None`` straight through instead of resolving a path."""
    assert inference_utils.load_file_entries(None, num=3, drop_empty=True) is None


def test_load_file_entries_rejects_a_line_less_file(tmp_path):
    """A zero-byte file requests nothing, which a blank line would have made explicit."""
    path = tmp_path / "empty.txt"
    path.write_text("")
    with pytest.raises(ValueError, match="holds no entry"):
        inference_utils.load_file_entries(path)


def test_load_file_entries_rejects_a_path_list_without_a_path(tmp_path):
    """Dropping blank lines leaves nothing, which is the same empty request."""
    with pytest.raises(ValueError, match="holds no entry"):
        inference_utils.load_file_entries(_write_lines(tmp_path / "paths.txt", "", "  "), drop_empty=True)


def test_shipped_prompt_files_carry_no_accidental_empty_prompt():
    """Blank lines are prompts now, so only the deliberate empty-prompt file may hold one."""
    prompts_dir = Path(inference_utils.__file__).parent / "prompts"
    empty_prompt_file = prompts_dir / "empty_prompt.txt"
    assert inference_utils.load_file_entries(empty_prompt_file) == [""]

    for path in sorted(prompts_dir.glob("*.txt")):
        if path == empty_prompt_file:
            continue
        assert all(inference_utils.load_file_entries(path)), f"{path.name} holds an empty prompt"


@pytest.mark.parametrize("sample_steps", [None, 7])
def test_image_teacher_cli_only_overrides_explicit_sample_steps(sample_steps, tmp_path, monkeypatch):
    input_shape = [1, 2, 2]
    net = _ConditioningNet(input_shape, text_output=None)
    model = _InferenceModel(net, input_shape)
    model.precision_amp_infer = torch.bfloat16
    _patch_inference_runtime(monkeypatch, image_inference, model, tmp_path)
    args = SimpleNamespace(
        ckpt_path="",
        classes=None,
        do_student_sampling=False,
        do_teacher_sampling=True,
        image_save_dir=str(tmp_path / "images"),
        input_image_file=None,
        neg_prompt_file=None,
        num_samples=1,
        prompt_file="unused",
        sample_steps=sample_steps,
        unconditional=True,
    )

    image_inference.main(args, _config(tmp_path, input_shape))
    sample_call = model.sample_calls[0]
    _assert_default_or_explicit_steps(sample_call, sample_steps)


_VIDEO_TASKS = {
    "text-to-video": {},
    "text-to-audio-video": {"text_output": "tuple", "multimodal": True},
    "image-to-video": {"image": True, "required_fields": ("real_raw",)},
    "video-to-world": {"image": True, "required_fields": ("real_raw",)},
    "video-to-video": {"video": True, "required_fields": ("real_raw",)},
    "video-to-video-depth": {
        "video": True,
        "depth": True,
        "required_fields": ("real_raw", "depth_latent"),
    },
    "camera-image-to-video": {
        "image": True,
        "camera": True,
        "required_fields": ("real_raw", "camera_poses", "camera_intrinsics"),
    },
}


def _path_list(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(str(value) + "\n")
    return str(path)


def _run_video_inference(
    task,
    sample_steps,
    tmp_path,
    monkeypatch,
    teacher_fps=None,
    save_high_quality=False,
    precision_amp_enc=None,
    vid_format="mp4",
):
    spec = _VIDEO_TASKS[task]
    input_shape = {"video": [4, 2, 2, 2], "audio": [2, 2, 2]} if spec.get("multimodal") else [4, 2, 2, 2]
    net = _ConditioningNet(
        input_shape,
        required_fields=spec.get("required_fields", ()),
        text_output=spec.get("text_output", "tensor"),
    )
    # Video networks have a VAE even when their build_conditioning is text-only.
    if "vae" not in net.preprocessors:
        net.preprocessors["vae"] = _VAE(input_shape)
    model = _InferenceModel(net, input_shape)
    model.precision_amp_enc = precision_amp_enc
    teacher_override = None
    if teacher_fps is not None:
        teacher_override = _ConditioningNet(input_shape)
        teacher_override.fps = teacher_fps
    save_calls = []
    _patch_inference_runtime(
        monkeypatch,
        video_inference,
        model,
        tmp_path,
        teacher_override=teacher_override,
        save_calls=save_calls,
    )

    prompt = "a paper boat crossing a pond"
    prompt_file = _path_list(tmp_path, f"{task}-prompts.txt", prompt)
    raw_image = torch.arange(3 * 1 * 4 * 4, dtype=torch.float32).reshape(1, 3, 1, 4, 4)
    raw_video = torch.arange(3 * 3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 3, 4, 4)
    depth_latent = torch.arange(2 * 2 * 2 * 2, dtype=torch.float32).reshape(1, 2, 2, 2, 2)
    camera_poses = torch.eye(4).reshape(1, 1, 4, 4)
    camera_intrinsics = torch.tensor([[[1.0, 1.0, 0.5, 0.5]]])
    latent_frame_counts = {}
    camera_inputs = {}

    monkeypatch.setattr(
        video_inference,
        "load_conditioning_image_for_net",
        lambda *_args, **_kwargs: {"real_raw": raw_image, "cropping_params": {"crop": "metadata"}},
    )

    def load_source_video(_path, _net, _shape, latent_frame_count):
        latent_frame_counts["video"] = latent_frame_count
        return {
            "real_raw": raw_video,
            "cropping_params": {"crop": "metadata"},
            "source_frame_indices": [0, 1],
        }

    monkeypatch.setattr(video_inference, "load_source_video_for_net", load_source_video)
    monkeypatch.setattr(video_inference, "load_raw_depth_latent", lambda *_args, **_kwargs: depth_latent)

    def prepare_depth(_depth, latent_frame_count):
        latent_frame_counts["depth"] = latent_frame_count
        return depth_latent

    monkeypatch.setattr(video_inference, "prepare_depth_latent", prepare_depth)
    monkeypatch.setattr(
        video_inference,
        "load_camera_poses",
        lambda *_args, **_kwargs: ("RAW_POSES", "RAW_INTRINSICS"),
    )

    def prepare_camera(poses, intrinsics, cropping_params, source_frame_indices):
        camera_inputs.update(
            poses=poses,
            intrinsics=intrinsics,
            cropping_params=cropping_params,
            source_frame_indices=source_frame_indices,
        )
        return camera_poses, camera_intrinsics

    monkeypatch.setattr(video_inference, "prepare_camera_poses", prepare_camera)

    input_image_file = _path_list(tmp_path, f"{task}-images.txt", "source.png") if spec.get("image") else None
    source_video_file = _path_list(tmp_path, f"{task}-videos.txt", "source.mp4") if spec.get("video") else None
    depth_latent_file = _path_list(tmp_path, f"{task}-depth.txt", "depth.pth") if spec.get("depth") else None
    camera_poses_file = _path_list(tmp_path, f"{task}-camera.txt", "poses") if spec.get("camera") else None

    args = SimpleNamespace(
        camera_poses_file=camera_poses_file,
        camera_source_video_file=None,
        ckpt_path="",
        depth_latent_file=depth_latent_file,
        do_student_sampling=True,
        do_teacher_sampling=True,
        input_image_file=input_image_file,
        neg_prompt_file=None,
        num_segments=1,
        overlap_frames=0,
        prompt_expand_model=None,
        prompt_expand_model_seed=0,
        prompt_file=prompt_file,
        sample_steps=sample_steps,
        save_high_quality=save_high_quality,
        source_video_file=source_video_file,
        vid_format=vid_format,
        video_save_dir=None,
    )

    video_inference.main(args, _config(tmp_path, input_shape))
    assert latent_frame_counts == {key: 2 for key in ("video", "depth") if spec.get(key)}
    assert camera_inputs == (
        {
            "poses": "RAW_POSES",
            "intrinsics": "RAW_INTRINSICS",
            "cropping_params": {"crop": "metadata"},
            "source_frame_indices": None,
        }
        if spec.get("camera")
        else {}
    )
    assert len(model.sample_calls) == 1
    assert len(model.generator_calls) == 1
    assert "fps" not in model.sample_calls[0]
    teacher_net = net if teacher_override is None else teacher_override
    assert save_calls[0][1]["fps"] is net.fps
    assert save_calls[1][1]["fps"] is teacher_net.fps
    expected_save_dir = tmp_path / "inference" / f"{task}-prompts"
    if save_high_quality:
        expected_save_dir = expected_save_dir.with_name(f"{expected_save_dir.name}_hq")
    for call_args, save_kwargs in save_calls:
        assert Path(call_args[1]).parent == expected_save_dir
        assert save_kwargs["precision_amp"] is precision_amp_enc
        assert save_kwargs["vid_format"] == vid_format
        if save_high_quality:
            assert save_kwargs["quality"] == 18
            assert save_kwargs["preset"] == "medium"
    assert net.preprocessing_modes == [(True, False)]

    # ``neg_prompt_file=None`` deactivates the negative branch.
    training_data = {"condition_raw": [prompt]}
    if spec.get("image"):
        training_data["real_raw"] = raw_image
    if spec.get("video"):
        training_data["real_raw"] = raw_video
    if spec.get("depth"):
        training_data["depth_latent"] = depth_latent
    if spec.get("camera"):
        training_data["camera_poses"] = camera_poses
        training_data["camera_intrinsics"] = camera_intrinsics

    training = _training_conditioning(model, training_data)
    assert net.preprocessing_modes[-1] == (False, False)
    return model.sample_calls[0], model.generator_calls[0], training


@pytest.mark.parametrize("task", list(_VIDEO_TASKS))
def test_video_inference_conditioning_matches_training(task, tmp_path, monkeypatch):
    sample_call, generator_call, training = _run_video_inference(task, None, tmp_path, monkeypatch)

    _assert_default_or_explicit_steps(sample_call, None)
    _assert_tree_equal(sample_call["condition"], training["condition"])
    _assert_tree_equal(sample_call["neg_condition"], training.get("neg_condition"))
    _assert_tree_equal(generator_call["condition"], training["condition"])
    _assert_tree_equal(generator_call["neg_condition"], training.get("neg_condition"))


def test_video_inference_accepts_omitted_conditioning_branches(tmp_path, monkeypatch):
    """Conditioning builders may omit unavailable positive and negative branches."""

    input_shape = [4, 2, 2, 2]
    net = _ConditioningNet(input_shape)
    net.preprocessors["vae"] = _VAE(input_shape)

    net.build_conditioning = lambda _data, input_shape=None, precision=None: {}
    model = _InferenceModel(net, input_shape)
    _patch_inference_runtime(monkeypatch, video_inference, model, tmp_path)

    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("a paper boat crossing a pond\n")
    args = SimpleNamespace(
        camera_poses_file=None,
        camera_source_video_file=None,
        ckpt_path="",
        depth_latent_file=None,
        do_student_sampling=True,
        do_teacher_sampling=True,
        input_image_file=None,
        neg_prompt_file=None,
        num_segments=1,
        overlap_frames=0,
        prompt_expand_model=None,
        prompt_expand_model_seed=0,
        prompt_file=str(prompt_file),
        sample_steps=None,
        save_high_quality=False,
        source_video_file=None,
        vid_format="mp4",
        video_save_dir=str(tmp_path / "videos"),
    )

    video_inference.main(args, _config(tmp_path, input_shape))

    _assert_sampling_received_no_conditioning(model)


@pytest.mark.parametrize(
    ("prompt_lines", "neg_prompt_lines", "expected"),
    [
        (("a paper boat crossing a pond",), None, None),
        (("a paper boat crossing a pond",), ("",), [""]),
        (("a paper boat crossing a pond",), ("a negative prompt",), ["a negative prompt"]),
        (("a pond", "a river"), ("shared",), ["shared", "shared"]),
        (("a pond", "a river"), ("first", "second"), ["first", "second"]),
        (("a pond", "a river", "a lake"), ("first", "second"), ["first", "second", "second"]),
        (("a pond", "a river"), ("first", "second", "third"), ["first", "second"]),
    ],
    ids=["deactivated", "blank-line", "file-with-prompt", "shared", "one-per-prompt", "extended", "trimmed"],
)
def test_video_inference_negative_prompt_file_resolution(
    prompt_lines, neg_prompt_lines, expected, tmp_path, monkeypatch
):
    """A deactivated negative prompt carries no negative branch, so guidance-distilled networks
    never receive a prompt they cannot encode. Otherwise every positive prompt gets its own
    negative one, with a short file repeating its last one and extras dropped."""

    input_shape = [4, 2, 2, 2]
    net = _ConditioningNet(input_shape)
    net.preprocessors["vae"] = _VAE(input_shape)
    requests = []

    def build_conditioning(data, input_shape=None, precision=None):
        requests.append(data)
        return {}

    net.build_conditioning = build_conditioning
    model = _InferenceModel(net, input_shape)
    _patch_inference_runtime(monkeypatch, video_inference, model, tmp_path)

    args = SimpleNamespace(
        camera_poses_file=None,
        camera_source_video_file=None,
        ckpt_path="",
        depth_latent_file=None,
        do_student_sampling=True,
        do_teacher_sampling=True,
        input_image_file=None,
        neg_prompt_file=(
            None if neg_prompt_lines is None else _write_lines(tmp_path / "negative.txt", *neg_prompt_lines)
        ),
        num_segments=1,
        overlap_frames=0,
        prompt_expand_model=None,
        prompt_expand_model_seed=0,
        prompt_file=_write_lines(tmp_path / "prompts.txt", *prompt_lines),
        sample_steps=None,
        save_high_quality=False,
        source_video_file=None,
        vid_format="mp4",
        video_save_dir=str(tmp_path / "videos"),
    )

    video_inference.main(args, _config(tmp_path, input_shape))

    assert len(requests) == len(prompt_lines)
    if expected is None:
        assert all("neg_condition_raw" not in request for request in requests)
    else:
        assert [request["neg_condition_raw"] for request in requests] == [[prompt] for prompt in expected]


@pytest.mark.parametrize("sample_steps", [None, 7])
def test_video_teacher_cli_only_overrides_explicit_sample_steps(sample_steps, tmp_path, monkeypatch):
    sample_call, _generator_call, _training = _run_video_inference("text-to-video", sample_steps, tmp_path, monkeypatch)
    _assert_default_or_explicit_steps(sample_call, sample_steps)


def test_video_inference_saves_with_each_sampling_network_fps(tmp_path, monkeypatch):
    # A separate but numerically matching teacher has a distinct FPS object. Identity checks in
    # ``_run_video_inference`` prove each save reads its producer rather than a shared run value.
    _run_video_inference("text-to-video", None, tmp_path, monkeypatch, teacher_fps=float("24.0"))


def test_video_inference_high_quality_keeps_format_and_decoder_amp(tmp_path, monkeypatch):
    _run_video_inference(
        "text-to-video",
        None,
        tmp_path,
        monkeypatch,
        save_high_quality=True,
        precision_amp_enc=torch.bfloat16,
    )


@pytest.mark.parametrize("task", ["text-to-video", "text-to-audio-video"])
def test_video_inference_forwards_vid_format(task, tmp_path, monkeypatch):
    # ``save_media`` owns the MP4 fallback, so audio-video runs also receive the requested GIF.
    _run_video_inference(task, None, tmp_path, monkeypatch, vid_format="gif")


def test_video_inference_extrapolation_names_the_method_requirement(tmp_path, monkeypatch):
    input_shape = [4, 2, 2, 2]
    net = _ConditioningNet(input_shape)
    net.preprocessors["vae"] = _VAE(input_shape)
    model = _InferenceModel(net, input_shape)
    _patch_inference_runtime(monkeypatch, video_inference, model, tmp_path)
    args = SimpleNamespace(
        camera_poses_file=None,
        camera_source_video_file=None,
        ckpt_path="",
        depth_latent_file=None,
        do_student_sampling=True,
        do_teacher_sampling=False,
        input_image_file=None,
        neg_prompt_file=None,
        num_segments=2,
        overlap_frames=0,
        prompt_expand_model=None,
        prompt_expand_model_seed=0,
        prompt_file=_write_lines(tmp_path / "prompts.txt", "a paper boat crossing a pond"),
        sample_steps=None,
        save_high_quality=False,
        source_video_file=None,
        vid_format="mp4",
        video_save_dir=str(tmp_path / "videos"),
    )

    with pytest.raises(RuntimeError, match="generator_fn_extrapolation.*_InferenceModel has none"):
        video_inference.main(args, _config(tmp_path, input_shape))
    assert model.generator_calls == []


def test_flashdreams_transform_converts_the_requested_checkpoint_root():
    def wan_state(value):
        return {"transformer.proj_out.weight": torch.full((1,), value)}

    checkpoint = {"model": {"net": wan_state(0.0), "ema": wan_state(1.0)}, "optimizer": {"net": {}}, "iteration": 7}

    for kwargs, value in (({}, 0.0), ({"root": "net"}, 0.0), ({"root": "ema"}, 1.0)):
        converted = flashdreams.fastgen_state_dict_transform(checkpoint, **kwargs)
        assert list(converted) == ["head.head.weight"]
        assert converted["head.head.weight"].item() == value


@pytest.mark.parametrize(("use_ema", "root"), [(False, "net"), (True, "ema"), (["ema_1", "ema"], "ema_1")])
def test_flashdreams_reads_the_network_inference_samples(use_ema, root):
    config = build_config("fastgen/configs/experiments/WanT2V/config_sf.py")
    config.model.use_ema = use_ema

    assert flashdreams._settings_from_config(config).checkpoint_root == root


def test_flashdreams_sizes_frames_with_the_network_pixel_shape():
    settings = flashdreams._settings_from_config(build_config("fastgen/configs/experiments/WanT2V/config_sf.py"))

    assert (settings.height, settings.width) == (480, 832)


@pytest.mark.parametrize("truncated", [b'{"id": "b", "sco', b'{"id": "b", "raw_response": "\xc3'])
def test_vlm_infer_resume_rescores_a_truncated_record_on_its_own_line(tmp_path, monkeypatch, truncated):
    """A record that an interrupted run cut short is not counted as done, and the resumed records start on a line of
    their own instead of extending the truncated one."""
    jsonl = tmp_path / "prompts.jsonl"
    jsonl.write_text('{"id": "a"}\n{"id": "b"}\n')
    output = tmp_path / "scores.jsonl"
    completed = b'{"id": "a", "score": 7.0}'
    output.write_bytes(completed + b"\n" + truncated)
    assert vlm_infer.load_existing_ids(output) == {"a"}
    monkeypatch.setattr(vlm_infer, "set_env_vars", lambda: None)
    monkeypatch.setattr(vlm_infer, "load_vlm_client", lambda base_url: None)
    monkeypatch.setattr(vlm_infer, "score_one", lambda record, **kwargs: {**record, "score": 5.0})
    arguments = ["--jsonl", str(jsonl), "--output", str(output), "--model", "judge", "--base-url", "http://judge"]

    assert vlm_infer.main([*arguments, "--resume"]) == 0

    lines = output.read_bytes().splitlines()
    assert lines[:2] == [completed, truncated]
    assert [json.loads(line) for line in lines[2:]] == [{"id": "b", "score": 5.0}]
    assert vlm_infer.load_existing_ids(output) == {"a", "b"}


# ---------------------------------------------------------------------------
# End-to-end random-student smoke tests for the inference scripts
# ---------------------------------------------------------------------------
#
# The production configs name very large pretrained networks.  These tests keep each
# real config and method class, but replace its network with a tiny randomly initialized
# student.  This exercises script routing, conditioning, preprocessing, split-precision
# placement, the method's real ``generator_fn``, and saving without checkpoint downloads.


@dataclass(frozen=True)
class InferenceCase:
    config: str
    script: str
    conditioning: str = "text"
    multimodal: bool = False
    extrapolate: bool = False


CASES = [
    # Image script: every distinct conditioning path and each text-image family.
    InferenceCase("EDM/config_sft_edm_cifar10.py", "image", "unconditional"),
    InferenceCase("DiT/config_sft_sit_xl.py", "image", "class"),
    InferenceCase("SD15/config_dmd2.py", "image", "text_mask"),
    InferenceCase("SDXL/config_dmd2.py", "image"),
    InferenceCase("Flux/config_dmd2.py", "image"),
    InferenceCase("Flux2/config_dmd2.py", "image"),
    InferenceCase("Flux2/config_dmd2_image_edit.py", "image", "image_edit"),
    InferenceCase("QwenImage/config_dmd2.py", "image"),
    # Video script: every network family and conditioning/routing contract.
    InferenceCase("CogVideoX/config_dmd2.py", "video"),
    InferenceCase("WanT2V/config_dmd2.py", "video"),
    InferenceCase("WanT2V/config_dmd2_wan22_14b.py", "video"),
    InferenceCase("WanI2V/config_dmd2_wan22_5b.py", "video", "i2v"),
    InferenceCase("WanI2V/config_sft_lingbot_world.py", "video", "camera_i2v"),
    InferenceCase("WanV2V/config_dmd2.py", "video", "v2v"),
    InferenceCase("LTX2/config_sft.py", "video", multimodal=True),
    InferenceCase("LTX2I2V/config_dmd2.py", "video", "i2v", multimodal=True),
    InferenceCase("CosmosPredict2/config_dmd2.py", "video"),
    InferenceCase("CosmosPredict2/config_dmd2_v2w.py", "video", "i2v"),
    InferenceCase("Cosmos3/config_dmd2.py", "video", "tokenizer"),
    InferenceCase("MiniMaxH3/config_sft.py", "video", multimodal=True),
    InferenceCase("MiniMaxH3/config_dmd2.py", "video", multimodal=True),
    InferenceCase("MiniMaxH3/config_sft.py", "video", "contextual_text", multimodal=True),
    # CausVid owns the distilled multi-segment path; causal SFT uses its network sampler.
    InferenceCase("WanT2V/config_causvid.py", "video", extrapolate=True),
    InferenceCase("WanI2V/config_sft_causal_wan22_5b.py", "video", "i2v"),
    InferenceCase("WanV2V/config_sft_causal.py", "video", "v2v"),
]


def _case_id(case: InferenceCase) -> str:
    family, filename = case.config.split("/", 1)
    return f"{case.script}-{family}-{filename.removeprefix('config_').removesuffix('.py')}"


class _RandomTextEncoder(FastGenModule):
    def __init__(self, with_mask: bool = False):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        self.with_mask = with_mask
        self.encode_calls = 0

    def encode(self, prompts):
        self.encode_calls += 1
        values = torch.arange(2, device=self.proj.weight.device, dtype=self.proj.weight.dtype)
        values = self.proj(values).expand(len(prompts), 2)
        if self.with_mask:
            return values, torch.ones(len(prompts), 2, device=values.device, dtype=torch.long)
        return values


class _RandomTokenizer(FastGenModule):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.randn(()), requires_grad=False)
        self.encode_calls = 0

    def encode(self, prompts, **kwargs):
        del kwargs
        self.encode_calls += 1
        return {"input_ids": [torch.arange(3, device=self.anchor.device, dtype=torch.long) for _ in prompts]}


class _RandomVAE(FastGenModule):
    def __init__(self, latent_compression_factors):
        super().__init__()
        self.latent_compression_factors = latent_compression_factors
        self.scale = torch.nn.Parameter(torch.randn(()), requires_grad=False)

    def encode(self, value, **kwargs):
        del kwargs
        return (value.to(self.precision_model) * self.scale).to(value.dtype)

    def decode(self, value, **kwargs):
        del kwargs
        return (value.to(self.precision_model) / self.scale.abs().clamp_min(0.25)).to(value.dtype)

    def reencode_overlap(self, latents, num_frames):
        return self.encode(self.decode(latents)[:, :, -num_frames:])


class _RandomImageEncoder(FastGenModule):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.randn(()), requires_grad=False)


class _RandomStudentMixin:
    """Small random network implementing all script-facing network contracts."""

    def _init_random_student(self, case: InferenceCase):
        self.case = case
        self.latent_compression_factors = (1, 1) if case.script == "image" else (1, 1, 1)
        self.is_image_edit = case.conditioning == "image_edit"
        self.fps = 24.0
        self.scale = torch.nn.Parameter(torch.randn(()))
        self.forward_records: list[dict[str, Any]] = []
        self.conditioning_records: list[dict[str, Any]] = []
        self.clear_cache_calls = 0
        if case.multimodal:
            self.noise_scheduler = CompositeNoiseSchedule(
                schedules={
                    "video": {"name": "rf", "min_t": 0.0, "max_t": 0.999},
                    "audio": {"name": "rf", "min_t": 0.0, "max_t": 0.999},
                }
            )

    def init_preprocessors(self):
        if self.case.conditioning == "tokenizer":
            self.preprocessors["text_tokenizer"] = _RandomTokenizer()
        elif self.case.conditioning not in {"unconditional", "class"}:
            self.preprocessors["text_encoder"] = _RandomTextEncoder(with_mask=self.case.conditioning == "text_mask")

        if self.case.script == "video" or self.case.conditioning == "image_edit":
            self.preprocessors["vae"] = _RandomVAE(self.latent_compression_factors)
        if self.case.conditioning in {"i2v", "camera_i2v"}:
            self.preprocessors["image_encoder"] = _RandomImageEncoder()

    def _predict(self, value):
        return (value.to(self.precision_model) * torch.tanh(self.scale)).to(value.dtype)

    def forward(self, x_t, t, condition=None, **kwargs):
        del kwargs
        self.forward_records.append(
            {
                "state_dtype": get_dtype(x_t),
                "time_dtype": t.dtype,
                "condition": condition,
            }
        )
        return per_modality(self._predict, x_t)

    def sample(
        self,
        noise,
        condition=None,
        neg_condition=None,
        sample_steps=None,
        sample_type="ode",
        guidance_cfg=None,
    ):
        del neg_condition, sample_steps, sample_type, guidance_cfg
        t = torch.full(
            (get_batch_size(noise),),
            self.noise_scheduler.max_t,
            device=get_device(noise),
            dtype=self.noise_scheduler.t_precision,
        )
        return self(noise, t, condition=condition, fwd_pred_type="x0")

    def build_conditioning(self, data, input_shape=None, precision=None):
        # Snapshot the builder input: ``prepare_batch`` merges the result into this same mapping.
        self.conditioning_records.append(dict(data))
        mode = self.case.conditioning
        if mode == "image_edit":
            assert data["condition_image_raw"].dtype == torch.float32
        if mode in {"i2v", "camera_i2v", "v2v"}:
            assert data["real_raw"].dtype == torch.float32
        if mode == "camera_i2v":
            assert data["camera_poses"].dtype == torch.float32
            assert data["camera_intrinsics"].dtype == torch.float32
        if mode == "contextual_text":
            # Like MiniMax-H3 FL2VA, encode the raw prompts inside the builder together with task context.
            encoder = self.preprocessors["text_encoder"]
            out = {"condition": encoder.encode(data["condition_raw"])}
            if data.get("neg_condition_raw") is not None:
                out["neg_condition"] = encoder.encode(data["neg_condition_raw"])
            return out
        if mode == "tokenizer":
            # Like Cosmos3, tokenize raw prompts inside the complete builder.
            tokenizer = self.preprocessors["text_tokenizer"]
            out = {"condition": tokenizer.encode(data["condition_raw"])}
            neg_prompts = data.get("neg_condition_raw")
            if neg_prompts is not None:
                out["neg_condition"] = tokenizer.encode(neg_prompts)
            return out
        return super().build_conditioning(data, input_shape=input_shape, precision=precision)


class _RandomStudent(_RandomStudentMixin, FastGenNetwork):
    def __init__(self, case: InferenceCase):
        super().__init__(net_pred_type="x0", schedule_type="rf")
        self._init_random_student(case)


class _RandomCausalStudent(_RandomStudentMixin, CausalFastGenNetwork):
    def __init__(self, case: InferenceCase):
        super().__init__(net_pred_type="x0", schedule_type="rf", chunk_size=1, total_num_frames=2)
        self._init_random_student(case)

    def sample(
        self,
        noise,
        condition=None,
        neg_condition=None,
        sample_steps=None,
        sample_type="ode",
        guidance_cfg=None,
        context_noise=None,
    ):
        del context_noise
        return super().sample(
            noise,
            condition=condition,
            neg_condition=neg_condition,
            sample_steps=sample_steps,
            sample_type=sample_type,
            guidance_cfg=guidance_cfg,
        )

    def clear_caches(self):
        self.clear_cache_calls += 1


def _random_model(config, case: InferenceCase):
    """Create the config's real method class without constructing its huge network."""
    model_type = config.model_class["_target_"]
    model = model_type.__new__(model_type)
    torch.nn.Module.__init__(model)
    model.config = config.model
    model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model._resolve_precisions(
        precision="float32",
        precision_model="bfloat16",
        precision_opt="bfloat16",
        precision_amp=None,
        precision_amp_infer=None,
        precision_amp_enc=None,
        precision_fsdp_reduce="float32",
    )
    model.input_shape = config.model.input_shape
    model.use_ema = []
    is_causal = issubclass(model_type, (CausVidModel, CausalSFTModel))
    model.net = _RandomCausalStudent(case) if is_causal else _RandomStudent(case)
    model.module_dict = {"net": model.net}
    model.teacher = None
    return model


def _write_lines(path: Path, *lines: str) -> str:
    path.write_text("".join(f"{line}\n" for line in lines))
    return str(path)


def _image_args(case: InferenceCase, tmp_path: Path, prompt_file: str, image_list: str | None):
    return SimpleNamespace(
        unconditional=case.conditioning == "unconditional",
        num_samples=1,
        prompt_file=prompt_file,
        ckpt_path="",
        do_student_sampling=True,
        do_teacher_sampling=False,
        image_save_dir=str(tmp_path / "outputs"),
        classes=10 if case.conditioning in {"unconditional", "class"} else None,
        neg_prompt_file=None,
        input_image_file=image_list,
        sample_steps=None,
    )


def _video_args(
    case: InferenceCase,
    tmp_path: Path,
    prompt_file: str,
    neg_prompt_file: str,
    image_list: str | None,
    source_video_list: str | None,
    camera_list: str | None,
):
    return SimpleNamespace(
        prompt_file=prompt_file,
        prompt_expand_model=None,
        prompt_expand_model_seed=0,
        depth_latent_file=None,
        source_video_file=source_video_list,
        input_image_file=image_list,
        camera_poses_file=camera_list,
        camera_source_video_file=None,
        ckpt_path="",
        do_student_sampling=True,
        do_teacher_sampling=False,
        video_save_dir=str(tmp_path / "outputs"),
        vid_format="mp4",
        save_high_quality=False,
        num_segments=2 if case.extrapolate else 1,
        overlap_frames=1 if case.extrapolate else 0,
        sample_steps=None,
        neg_prompt_file=neg_prompt_file,
    )


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_random_student_inference_for_representative_config(case, tmp_path, monkeypatch):
    """Run one student sample through the matching inference script for each representative config."""
    config_path = f"fastgen/configs/experiments/{case.config}"
    config = build_config(config_path)
    config.model.input_shape = {"video": [2, 2, 4, 4], "audio": [2, 4, 3]} if case.multimodal else [2, 2, 4, 4]
    if case.script == "image":
        config.model.input_shape = [2, 4, 4]
    config.model.student_sample_type = "ode"
    config.model.enable_preprocessors = True
    config.model.torch_compile_mode = None
    config.model.inference = True

    prompt = "3" if case.conditioning == "class" else "a tiny random student"
    prompt_file = _write_lines(tmp_path / "prompts.txt", prompt)
    neg_prompt_file = _write_lines(tmp_path / "negative.txt", "negative")

    image_path = tmp_path / "source.png"
    Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(image_path)
    image_list = (
        _write_lines(tmp_path / "images.txt", str(image_path))
        if case.conditioning in {"image_edit", "i2v", "camera_i2v"}
        else None
    )
    source_video_list = (
        _write_lines(tmp_path / "videos.txt", str(tmp_path / "source.mp4")) if case.conditioning == "v2v" else None
    )

    camera_list = None
    if case.conditioning == "camera_i2v":
        camera_dir = tmp_path / "camera"
        camera_dir.mkdir()
        np.save(camera_dir / "poses.npy", np.stack([np.eye(4), np.eye(4)]).astype(np.float32))
        np.save(camera_dir / "intrinsics.npy", np.array([[2, 2, 2, 2], [2, 2, 2, 2]], dtype=np.float32))
        camera_list = _write_lines(tmp_path / "cameras.txt", str(camera_dir))

    model = _random_model(config, case)
    script = image_inference if case.script == "image" else video_inference
    saved = []
    monkeypatch.setattr(script, "instantiate", lambda *args, **kwargs: model)
    monkeypatch.setattr(script, "Checkpointer", lambda *args, **kwargs: object())
    monkeypatch.setattr(script, "load_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(script, "save_media", lambda value, path, **kwargs: saved.append((value, path, kwargs)))

    if case.conditioning == "v2v":

        def load_source_video(path, net, input_shape, latent_frame_count):
            del path, net, latent_frame_count
            shape = input_shape["video"] if isinstance(input_shape, dict) else input_shape
            return {
                "real_raw": torch.randn(1, 3, shape[1], shape[2], shape[3], dtype=torch.float32),
                "cropping_params": {},
                "source_frame_indices": list(range(shape[1])),
            }

        monkeypatch.setattr(video_inference, "load_source_video_for_net", load_source_video)

    if case.script == "image":
        args = _image_args(case, tmp_path, prompt_file, image_list)
    else:
        args = _video_args(
            case,
            tmp_path,
            prompt_file,
            neg_prompt_file,
            image_list,
            source_video_list,
            camera_list,
        )

    script.main(args, config)

    assert len(saved) == 1
    output, path, save_kwargs = saved[0]
    assert "/student_" in path and "/teacher_" not in path
    if case.script == "video":
        assert save_kwargs["fps"] == model.net.fps
    assert get_dtype(output) == torch.float32
    expected_shape = config.model.input_shape
    if case.extrapolate:
        expected_shape = [2, 3, 4, 4]
    if isinstance(output, dict):
        assert {key: list(value.shape[1:]) for key, value in output.items()} == expected_shape
    else:
        assert list(output.shape[1:]) == expected_shape

    student = model.net
    assert student.forward_records
    assert all(record["state_dtype"] == torch.float32 for record in student.forward_records)
    assert student.scale.dtype == student.precision_model == torch.bfloat16
    if case.extrapolate:
        assert student.clear_cache_calls > 0

    for preprocessor in student.preprocessors.values():
        assert preprocessor.precision_model == torch.bfloat16
        floating_state = [
            value for value in (*preprocessor.parameters(), *preprocessor.buffers()) if value.is_floating_point()
        ]
        assert floating_state and all(value.dtype == torch.bfloat16 for value in floating_state)

    if case.conditioning == "text_mask":
        condition = student.forward_records[0]["condition"]
        assert condition[0].dtype == torch.bfloat16 and condition[1].dtype == torch.long
    if case.conditioning == "tokenizer":
        condition = student.forward_records[0]["condition"]
        assert all(token_ids.dtype == torch.long for token_ids in condition["input_ids"])
        # Cosmos3-like builders receive the raw negative prompt and tokenize each branch once.
        assert student.preprocessors["text_tokenizer"].encode_calls == 2
        assert student.conditioning_records[-1]["neg_condition_raw"] == ["negative"]
    if case.conditioning == "contextual_text":
        # The script must not pre-encode a negative that a context-dependent builder cannot reuse.
        assert student.preprocessors["text_encoder"].encode_calls == 2
        assert student.conditioning_records[-1]["neg_condition_raw"] == ["negative"]


# ---------------------------------------------------------------------------
# Core student sampling
# ---------------------------------------------------------------------------


class _RecordingStudentNet(FastGenNetwork):
    def __init__(self, student_grid_cfg):
        super().__init__(
            net_pred_type="x0",
            schedule_type="rf",
            student_grid_cfg=student_grid_cfg,
        )
        self.call = None

    def forward(self, x_t, t, condition=None, fwd_pred_type=None, **kwargs):
        self.call = (t, condition, fwd_pred_type, kwargs)
        return x_t + 1


class _StudentLoopHarness(FastGenModel):
    def single_train_step(self, data, iteration):
        raise NotImplementedError


def test_student_grid_config_rejects_unknown_fields():
    with pytest.raises(TypeError):
        StudentGridConfig(sample_steps=4)
    with pytest.raises(TypeError, match="StudentGridConfig"):
        get_noise_schedule("rf", student_grid_cfg={"steps": 4})


@pytest.mark.parametrize(
    ("grid_type", "grid_kwargs"),
    [
        ("native", {"base_steps": 4, "blocks": [2, 2]}),
        ("explicit", {"timesteps": [0.999, 0.5, 0.0], "base_steps": 2}),
    ],
)
def test_removed_student_grid_kwargs_are_rejected(grid_type, grid_kwargs):
    schedule = get_noise_schedule(
        "rf",
        student_grid_cfg=StudentGridConfig(grid_type=grid_type, grid_kwargs=grid_kwargs),
    )

    with pytest.raises(ValueError, match="Unsupported"):
        schedule.student_grid()


def test_schedule_without_student_config_uses_config_default():
    schedule = get_noise_schedule("rf")

    assert schedule.default_student_steps == StudentGridConfig().steps == 1
    assert schedule.student_grid().shape == (2,)


@pytest.mark.parametrize(
    ("min_t", "max_t", "match"),
    [
        (-0.1, 0.9, "min_t must be non-negative"),
        (0.5, 0.5, "max_t must be greater than min_t"),
        (0.6, 0.5, "max_t must be greater than min_t"),
    ],
)
def test_schedule_time_range_is_validated_at_construction(min_t, max_t, match):
    with pytest.raises(ValueError, match=match):
        get_noise_schedule("rf", min_t=min_t, max_t=max_t)


@pytest.mark.parametrize("clamp_min", [0.0, -1e-6])
def test_schedule_clamp_min_is_validated_at_construction(clamp_min):
    with pytest.raises(ValueError, match="clamp_min must be positive"):
        get_noise_schedule("rf", clamp_min=clamp_min)


@pytest.mark.parametrize(
    ("schedule_name", "schedule_kwargs"),
    [
        ("edm", {}),
        ("shifted_rf", {"shift": 3.0}),
        ("trig", {}),
    ],
)
def test_checked_conversion_api_preserves_schedule_specific_kernels(schedule_name, schedule_kwargs):
    schedule = get_noise_schedule(schedule_name, **schedule_kwargs)
    t = torch.tensor([0.25, 0.5], dtype=torch.float64)
    x0 = torch.randn(2, 3, dtype=torch.float64)
    eps = torch.randn_like(x0)
    xt = schedule.forward_process(x0, eps, t)
    flow = schedule.cond_velocity(x0, eps, t)

    torch.testing.assert_close(schedule.flow_to_x0(xt, flow, t), x0)
    torch.testing.assert_close(schedule.x0_to_flow(xt, x0, t), flow)


def test_convert_model_output_checks_time_once(monkeypatch):
    schedule = get_noise_schedule("shifted_rf", shift=3.0)
    t = torch.tensor([0.25, 0.5], dtype=torch.float64)
    xt = torch.randn(2, 3, dtype=torch.float64)
    eps = torch.randn_like(xt)
    expected = schedule.x0_to_flow(xt, schedule.eps_to_x0(xt, eps, t), t)
    original_assert_t_valid = schedule._assert_t_valid
    calls = []

    def record_assert_t_valid(actual_t, name="t"):
        calls.append((actual_t, name))
        original_assert_t_valid(actual_t, name)

    monkeypatch.setattr(schedule, "_assert_t_valid", record_assert_t_valid)
    actual = schedule.convert_model_output(xt, eps, t, src_pred_type="eps", target_pred_type="flow")

    assert len(calls) == 1
    assert calls[0][0] is t and calls[0][1] == "t"
    assert torch.equal(actual, expected)


def test_v_conversion_uses_the_same_checked_and_unchecked_kernels(monkeypatch):
    schedule = get_noise_schedule(
        "alphas",
        alphas_cumprod=torch.linspace(0.999, 0.001, 1000, dtype=torch.float64),
    )
    t = torch.tensor([0.25, 0.5], dtype=torch.float64)
    xt = torch.randn(2, 3, dtype=torch.float64)
    x0 = torch.randn_like(xt)
    expected = schedule.x0_to_v(xt, x0, t)
    original_assert_t_valid = schedule._assert_t_valid
    calls = []

    def record_assert_t_valid(actual_t, name="t"):
        calls.append((actual_t, name))
        original_assert_t_valid(actual_t, name)

    monkeypatch.setattr(schedule, "_assert_t_valid", record_assert_t_valid)
    actual = schedule.convert_model_output(xt, x0, t, src_pred_type="x0", target_pred_type="v")

    assert len(calls) == 1
    assert calls[0][0] is t and calls[0][1] == "t"
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("dtype", "bound"),
    [
        (torch.float16, 0.1),
        (torch.bfloat16, 0.7),
        (torch.float32, 0.7),
        (torch.float16, 0.876543211),
        (torch.bfloat16, 0.876543211),
    ],
)
def test_safe_clamp_rounds_lower_precision_min_up(dtype, bound):
    rounded = torch.tensor(bound, dtype=dtype)
    assert rounded.to(torch.float64) < bound

    actual = BaseNoiseSchedule.safe_clamp(torch.full((2,), -torch.inf, dtype=dtype), min=bound)
    expected = torch.nextafter(rounded, torch.full_like(rounded, torch.inf))

    assert torch.equal(actual, expected.expand_as(actual))
    assert bool((actual.to(torch.float64) >= bound).all())


@pytest.mark.parametrize(
    ("dtype", "bound"),
    [
        (torch.float16, 0.7),
        (torch.bfloat16, 0.1),
        (torch.float32, 0.1),
        (torch.float16, 0.123456789),
        (torch.bfloat16, 0.123456789),
        (torch.float32, 0.123456789),
        (torch.float32, 0.876543211),
    ],
)
def test_safe_clamp_rounds_lower_precision_max_down(dtype, bound):
    rounded = torch.tensor(bound, dtype=dtype)
    assert rounded.to(torch.float64) > bound

    actual = BaseNoiseSchedule.safe_clamp(torch.full((2,), torch.inf, dtype=dtype), max=bound)
    expected = torch.nextafter(rounded, torch.full_like(rounded, -torch.inf))

    assert torch.equal(actual, expected.expand_as(actual))
    assert bool((actual.to(torch.float64) <= bound).all())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_safe_clamp_keeps_exact_lower_precision_bounds(dtype):
    values = torch.tensor([-1.0, 1.0], dtype=dtype)
    actual = BaseNoiseSchedule.safe_clamp(values, min=0.25, max=0.5)
    assert torch.equal(actual, torch.tensor([0.25, 0.5], dtype=dtype))


def test_safe_clamp_float64_skips_bound_adjustment(monkeypatch):
    values = torch.tensor([-1.0, 0.4, 1.0], dtype=torch.float64)
    expected = torch.clamp(values, min=0.1, max=0.7)

    def unexpected_nextafter(*args, **kwargs):
        raise AssertionError("float64 safe_clamp must not adjust its bounds")

    monkeypatch.setattr(torch, "nextafter", unexpected_nextafter)
    actual = BaseNoiseSchedule.safe_clamp(values, min=0.1, max=0.7)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_safe_clamp_lower_precision_does_not_read_device_scalars(dtype):
    values = torch.empty(2, dtype=dtype, device="meta")
    actual = BaseNoiseSchedule.safe_clamp(values, min=0.1, max=0.7)

    assert actual.device.type == "meta"
    assert actual.dtype == dtype
    assert actual.shape == values.shape


def test_student_grid_defaults_to_configured_steps():
    schedule = get_noise_schedule(
        "rf",
        student_grid_cfg=StudentGridConfig(steps=4),
    )

    assert schedule.default_student_steps == 4
    assert torch.equal(schedule.student_grid(), schedule.student_grid(4))


def test_native_student_grid_shift_warps_the_grid():
    schedule = get_noise_schedule(
        "rf",
        max_t=1.0,
        student_grid_cfg=StudentGridConfig(
            steps=4,
            grid_type="native",
            grid_kwargs={"shift": 5},
        ),
    )

    expected = schedule.shift_t(schedule.grid(4), shift=5)
    torch.testing.assert_close(schedule.student_grid(), expected)


def test_native_student_grid_shift_supports_an_anchored_base_grid():
    schedule = get_noise_schedule(
        "rf",
        max_t=1.0,
        student_grid_cfg=StudentGridConfig(
            steps=2,
            grid_type="native",
            grid_kwargs={"base_steps": 4, "shift": 5},
        ),
    )

    shifted_base_grid = schedule.shift_t(schedule.grid(4), shift=5)
    torch.testing.assert_close(schedule.student_grid(), shifted_base_grid[[0, 2, 4]])


@pytest.mark.parametrize("shift", [0, -1])
def test_native_student_grid_requires_a_positive_shift(shift):
    schedule = get_noise_schedule(
        "rf",
        student_grid_cfg=StudentGridConfig(
            steps=4,
            grid_type="native",
            grid_kwargs={"shift": shift},
        ),
    )

    with pytest.raises(ValueError, match="time shift must be positive"):
        schedule.student_grid()


def test_global_student_steps_override_controls_default_grid():
    schedule = get_noise_schedule(
        "rf",
        student_grid_cfg=StudentGridConfig(steps=4),
    )

    with set_temp_global_vars({"STUDENT_STEPS": 2}):
        assert schedule.default_student_steps == 2
        assert torch.equal(schedule.student_grid(), schedule.student_grid(2))

    assert schedule.default_student_steps == 4
    assert schedule.student_grid_cfg.steps == 4


def test_sample_student_timesteps_can_return_step_indices(monkeypatch):
    schedule = get_noise_schedule("rf", student_grid_cfg=StudentGridConfig(steps=3))
    expected_step_indices = torch.tensor([0, 2, 1])

    def fixed_randint(low, high, size, device=None):
        assert (low, high, size) == (0, 3, (3,))
        return expected_step_indices.to(device=device)

    monkeypatch.setattr(torch, "randint", fixed_randint)

    sampled, step_indices = schedule.sample_student_timesteps(3, return_indices=True)
    assert torch.equal(step_indices, expected_step_indices)
    torch.testing.assert_close(sampled, schedule.student_grid()[expected_step_indices])


def test_network_forward_remains_time_based():
    net = _RecordingStudentNet(
        StudentGridConfig(
            steps=3,
            grid_type="explicit",
            grid_kwargs={"timesteps": [0.9, 0.7, 0.4, 0.0]},
        )
    )
    x = torch.zeros(2, 1)
    condition = object()

    t = net.noise_scheduler.student_grid()[torch.tensor([0, 2])]
    result = net(
        x,
        t,
        condition=condition,
        fwd_pred_type="x0",
        marker=True,
    )
    torch.testing.assert_close(result, x + 1)
    seen_t, seen_condition, pred_type, kwargs = net.call
    torch.testing.assert_close(seen_t, torch.tensor([0.9, 0.4], dtype=seen_t.dtype))
    assert seen_condition is condition
    assert pred_type == "x0"
    assert kwargs == {"marker": True}


def test_base_student_loop_passes_grid_timesteps_to_network():
    class RecordingNet:
        def __init__(self):
            self.noise_scheduler = get_noise_schedule("rf")
            self.calls = []

        def __call__(self, state, t, condition=None, fwd_pred_type=None, output_dtype="same"):
            self.calls.append((t, condition, fwd_pred_type, output_dtype))
            return torch.zeros_like(state)

    model = object.__new__(_StudentLoopHarness)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(student_sample_type="ode")
    net = RecordingNet()
    grid = torch.tensor([0.9, 0.6, 0.2, 0.0])

    result = model._student_sample_loop(net, torch.ones(2, 1), grid, condition="prompt")

    torch.testing.assert_close(
        torch.stack([call[0] for call in net.calls]),
        grid[:-1, None].expand(-1, 2),
    )
    assert [call[1:] for call in net.calls] == [("prompt", "x0", None)] * 3
    torch.testing.assert_close(result, torch.zeros_like(result))


def test_base_student_loop_ends_on_its_last_x0_prediction_where_time_zero_is_noisy():
    """SD's first discrete time 0 carries noise; the loop returns the clean prediction instead of renoising to it."""

    class X0Net:
        def __init__(self):
            betas = torch.linspace(0.00085**0.5, 0.012**0.5, 1000, dtype=torch.float64) ** 2
            self.noise_scheduler = AlphasNoiseSchedule(alphas_cumprod=torch.cumprod(1 - betas, dim=0))
            self.predictions = []

        def __call__(self, state, t, condition=None, fwd_pred_type=None, output_dtype="same"):
            self.predictions.append(0.5 * state + t[:, None])
            return self.predictions[-1]

    model = object.__new__(_StudentLoopHarness)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(student_sample_type="ode")
    net = X0Net()
    grid = torch.tensor([0.9, 0.4, 0.0], dtype=torch.float64)
    assert net.noise_scheduler.sigma(grid[-1:]) > 0

    result = model._student_sample_loop(net, torch.randn(2, 1, dtype=torch.float64), grid)

    assert torch.equal(result, net.predictions[-1])


def test_student_generator_uses_the_globally_resolved_default_steps():
    class RecordingNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.noise_scheduler = get_noise_schedule(
                "rf",
                student_grid_cfg=StudentGridConfig(
                    steps=4,
                    grid_type="explicit",
                    grid_kwargs={"timesteps": [0.9, 0.7, 0.4, 0.1, 0.0]},
                ),
            )
            self.calls = []

        def forward(self, state, t, condition=None, fwd_pred_type=None, output_dtype="same"):
            self.calls.append((t.clone(), condition, fwd_pred_type, output_dtype))
            return torch.zeros_like(state)

    model = object.__new__(_StudentLoopHarness)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(student_sample_type="ode")
    model.precision_amp_infer = None
    net = RecordingNet()
    noise = torch.ones(2, 1)

    with set_temp_global_vars({"STUDENT_STEPS": 2}):
        expected_grid = net.noise_scheduler.student_grid().clone()
        result = model.generator_fn(net, noise, condition="prompt")

    assert len(net.calls) == 2
    torch.testing.assert_close(
        torch.stack([call[0] for call in net.calls]),
        expected_grid[:-1, None].expand(-1, noise.shape[0]),
    )
    assert [call[1:] for call in net.calls] == [("prompt", "x0", None)] * 2
    torch.testing.assert_close(result, torch.zeros_like(noise))


def test_base_student_step_inherits_the_callers_autograd_mode():
    net = _RecordingStudentNet(
        StudentGridConfig(
            steps=2,
            grid_type="explicit",
            grid_kwargs={"timesteps": [0.9, 0.4, 0.0]},
        )
    )
    model = object.__new__(_StudentLoopHarness)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(student_sample_type="ode")
    grid = net.noise_scheduler.student_grid()
    state = torch.ones(2, 1, requires_grad=True)

    next_state = model._advance_student_step(net, state, 0, grid)

    assert next_state.requires_grad
    next_state.sum().backward()
    assert state.grad is not None and torch.count_nonzero(state.grad)

    with torch.no_grad():
        next_state = model._advance_student_step(net, state.detach(), 0, grid)
    assert not next_state.requires_grad


def test_student_grid_indices_use_anchored_grid_contract():
    schedule = get_noise_schedule(
        "rf",
        student_grid_cfg=StudentGridConfig(
            steps=4,
            grid_type="native",
            grid_kwargs={
                "base_steps": 8,
                "indices": {4: [0, 1, 4, 6, 8]},
            },
        ),
    )

    base_grid = schedule.student_grid(8)
    assert schedule.student_base_steps == 8
    indices = schedule.student_grid_indices(target_steps=4)
    assert torch.equal(indices, torch.tensor([0, 1, 4, 6, 8]))
    torch.testing.assert_close(schedule.student_grid(4), base_grid[indices])

    with set_temp_global_vars({"STUDENT_STEPS": 2}):
        # The relation uses the configured base grid and an explicitly requested
        # student grid, not the default student NFE or its global override.
        assert torch.equal(
            schedule.student_grid_indices(target_steps=4),
            indices,
        )
        two_step_indices = schedule.student_grid_indices(target_steps=2)
        assert torch.equal(two_step_indices, torch.tensor([0, 4, 8]))
        torch.testing.assert_close(schedule.student_grid(), base_grid[two_step_indices])

    assert torch.equal(schedule.student_grid_indices(4), indices)
    with pytest.raises(TypeError):
        schedule.student_grid_indices()


def test_student_grid_indices_reject_an_unanchored_native_grid():
    unanchored = get_noise_schedule(
        "rf",
        student_grid_cfg=StudentGridConfig(steps=4, grid_type="native"),
    )
    assert unanchored.student_base_steps is None
    with pytest.raises(ValueError, match="requires an anchored student grid"):
        unanchored.student_grid_indices(target_steps=4)


@pytest.mark.parametrize(
    ("num_steps", "custom_indices", "expected"),
    [
        (4, None, [0, 1, 3, 4, 6]),
        (3, [0, 1, 4, 6], [0, 1, 4, 6]),
    ],
)
def test_student_grid_indices_support_anchored_base_grids(num_steps, custom_indices, expected):
    grid_kwargs = {"base_steps": 6}
    if custom_indices is not None:
        grid_kwargs["indices"] = {num_steps: custom_indices}
    schedule = get_noise_schedule(
        "rf",
        student_grid_cfg=StudentGridConfig(
            steps=num_steps,
            grid_type="native",
            grid_kwargs=grid_kwargs,
        ),
    )

    base_grid = schedule.student_grid(6)
    indices = schedule.student_grid_indices(target_steps=num_steps)

    assert indices.tolist() == expected
    torch.testing.assert_close(schedule.student_grid(num_steps), base_grid[indices])


def test_student_grid_indices_infer_the_base_from_an_explicit_grid():
    schedule = get_noise_schedule(
        "rf",
        student_grid_cfg=StudentGridConfig(
            steps=2,
            grid_type="explicit",
            grid_kwargs={"timesteps": [0.999, 0.75, 0.5, 0.25, 0.0]},
        ),
    )

    assert schedule.student_base_steps == 4
    assert torch.equal(schedule.student_grid_indices(target_steps=2), torch.tensor([0, 2, 4]))
    indices = schedule.student_grid_indices(target_steps=4)
    assert torch.equal(indices, torch.arange(5))


@pytest.mark.parametrize(
    ("configured_indices", "error_type", "match"),
    [
        ([0, 4], ValueError, "must have shape"),
        ([1, 2, 4], ValueError, "must start at 0 and end at 4"),
        ([0, 4, 4], ValueError, "must be strictly increasing"),
        ([0.0, 2.0, 4.0], TypeError, "must have an integer dtype"),
    ],
)
def test_student_grid_indices_reject_malformed_configured_mappings(configured_indices, error_type, match):
    schedule = get_noise_schedule(
        "rf",
        student_grid_cfg=StudentGridConfig(
            steps=2,
            grid_kwargs={
                "base_steps": 4,
                "indices": {2: configured_indices},
            },
        ),
    )

    with pytest.raises(error_type, match=match):
        schedule.student_grid_indices(target_steps=2)


def test_student_grid_indices_reject_more_target_than_base_steps():
    schedule = get_noise_schedule(
        "rf",
        student_grid_cfg=StudentGridConfig(steps=2, grid_kwargs={"base_steps": 4}),
    )

    with pytest.raises(ValueError, match="Cannot select 5 steps from a 4-step student base grid"):
        schedule.student_grid_indices(target_steps=5)


def test_grid_ignores_student_grid_cfg_policy_and_global_override():
    schedule = get_noise_schedule(
        "rf",
        student_grid_cfg=StudentGridConfig(
            steps=4,
            grid_type="explicit",
            grid_kwargs={"timesteps": [0.999, 0.9, 0.7, 0.2, 0.0]},
        ),
    )
    expected = get_noise_schedule("rf").grid(2)

    with set_temp_global_vars({"STUDENT_STEPS": 2}):
        assert torch.equal(schedule.grid(2), expected)
        assert not torch.equal(schedule.student_grid(), expected)


def test_teacher_sampler_requests_schedule_grid(monkeypatch):
    from fastgen.networks.EDM import network as edm_network

    calls = []

    class StubSchedule(EDMNoiseSchedule):
        def grid(self, num_steps, device=None):
            calls.append((num_steps, device))
            return torch.tensor([1.0, 0.5, 0.0], device=device)

        def student_grid(self, *args, **kwargs):
            raise AssertionError("teacher sampling must not request the student grid")

        @staticmethod
        def latents(noise, t_init):
            return noise * t_init

    class StubTeacher:
        schedule_type = "edm"
        noise_scheduler = StubSchedule()
        _validate_sample_kwargs = edm_network.EDMPrecond._validate_sample_kwargs

    times = []

    def predict(net, x, t, **kwargs):
        times.append(t)
        return torch.zeros_like(x)

    monkeypatch.setattr(edm_network, "predict_with_guidance", predict)
    noise = torch.ones(1, 1, 1, 1)

    result = edm_network.EDMPrecond.sample(
        StubTeacher(),
        noise,
        sample_steps=2,
        guidance_cfg=object(),
    )

    assert result.shape == noise.shape
    assert calls == [(2, noise.device)]
    torch.testing.assert_close(torch.cat(times), torch.tensor([1.0, 0.5]))


def test_edm_teacher_sampler_uses_euler_for_rf_without_r(monkeypatch):
    from fastgen.networks.EDM import network as edm_network

    calls = []
    predictions = []

    class StubSchedule(RFNoiseSchedule):
        def grid(self, num_steps, device=None):
            calls.append((num_steps, device))
            return torch.tensor([0.9, 0.4, 0.0], device=device)

        def student_grid(self, *args, **kwargs):
            raise AssertionError("teacher sampling must not request the student grid")

        @staticmethod
        def latents(noise, t_init):
            return noise * t_init

    class StubTeacher:
        schedule_type = "rf"
        noise_scheduler = StubSchedule()
        _validate_sample_kwargs = edm_network.EDMPrecond._validate_sample_kwargs

    def predict(net, x, t, **kwargs):
        predictions.append((t.clone(), kwargs))
        return torch.full_like(x, 2.0)

    monkeypatch.setattr(edm_network, "predict_with_guidance", predict)
    noise = torch.ones(1, 1, 1, 1)

    result = edm_network.EDMPrecond.sample(
        StubTeacher(),
        noise,
        sample_steps=2,
        guidance_cfg=object(),
    )

    torch.testing.assert_close(result, torch.full_like(noise, -0.9))
    assert calls == [(2, noise.device)]
    torch.testing.assert_close(torch.cat([t for t, _ in predictions]), torch.tensor([0.9, 0.4]))
    assert all(kwargs["fwd_pred_type"] == "flow" for _, kwargs in predictions)
    assert all("r" not in kwargs for _, kwargs in predictions)


def test_explicit_grid_is_balanced_subsampled_by_default():
    schedule = get_noise_schedule(
        "rf",
        student_grid_cfg=StudentGridConfig(
            steps=2,
            grid_type="explicit",
            grid_kwargs={"timesteps": [0.999, 0.75, 0.5, 0.25, 0.0]},
        ),
    )

    assert torch.equal(
        schedule.student_grid(),
        torch.tensor([0.999, 0.5, 0.0], dtype=schedule.t_precision),
    )
    with set_temp_global_vars({"STUDENT_STEPS": 3}):
        assert torch.equal(
            schedule.student_grid(),
            torch.tensor([0.999, 0.75, 0.5, 0.0], dtype=schedule.t_precision),
        )


def test_global_override_cannot_exceed_anchored_grid():
    schedule = get_noise_schedule(
        "rf",
        student_grid_cfg=StudentGridConfig(
            steps=2,
            grid_type="native",
            grid_kwargs={"base_steps": 4},
        ),
    )

    with set_temp_global_vars({"STUDENT_STEPS": 5}):
        with pytest.raises(ValueError, match="Cannot select 5 steps from a 4-step student base grid"):
            schedule.student_grid()


def test_composite_uses_primary_student_grid_cfg_policy():
    sampling = StudentGridConfig(steps=2, grid_kwargs={"base_steps": 4, "shift": 5.0})
    composite = CompositeNoiseSchedule(
        schedules={
            "video": {"name": "rf", "student_grid_cfg": sampling},
            "audio": {"name": "rf"},
        },
        primary_key="video",
    )

    assert composite.student_grid_cfg is composite.primary.student_grid_cfg
    assert composite.student_grid_cfg is sampling
    assert torch.equal(composite.student_grid(), composite.primary.student_grid())
    assert composite.sample_student_timesteps(3).shape == (3,)


@pytest.mark.parametrize("composite", [False, True], ids=["tensor", "multimodal"])
def test_step_accepts_scalar_grid_time(composite):
    if composite:
        schedule = CompositeNoiseSchedule(
            schedules={"video": {"name": "rf"}, "audio": {"name": "shifted_rf", "shift": 2.0}},
            primary_key="video",
        )
        x = {"video": torch.randn(3, 2), "audio": torch.randn(3, 4)}
    else:
        schedule = get_noise_schedule("rf")
        x = torch.randn(3, 2)
    x0 = randn_like(x)

    t = torch.tensor(0.75)
    t_next = torch.tensor(0.25)
    scalar_result = schedule.step(x, x0, t, t_next, sample_type="ode")
    batched_result = schedule.step(x, x0, t.expand(3), t_next, sample_type="ode")

    if composite:
        for key in x:
            torch.testing.assert_close(scalar_result[key], batched_result[key])
    else:
        torch.testing.assert_close(scalar_result, batched_result)
