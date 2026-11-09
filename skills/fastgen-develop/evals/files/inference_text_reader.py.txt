# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared model setup and input preparation for FastGen inference front ends."""

import argparse
import gc
from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Tuple, List, Any

import torch

from fastgen.networks.lora_utils import SharedBaseTeacher
from fastgen.utils.checkpointer import Checkpointer
from fastgen.utils.env import str2bool
from fastgen.utils.io_utils import parse_checkpoint_iteration
from fastgen.utils.tensor import unit_pixels_to_uint8
import fastgen.utils.logging_utils as logger


# -----------------------------------------------------------------------------
# Paths and line-based inputs
# -----------------------------------------------------------------------------


def expand_path(path: str | Path) -> Path:
    """Return ``path`` as is when absolute, otherwise resolved against the current working directory."""
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def load_file_entries(
    entry_file: str | Path | None,
    num: Optional[int] = None,
    drop_empty: bool = False,
) -> Optional[List[str]]:
    """Load stripped entries from a text file; ``None`` for an input that was not given.

    Args:
        entry_file: Path to a text file with one entry per line.
        num: Entry count to return. A short file repeats its last entry; a long
            one is trimmed.
        drop_empty: If ``True``, drop blank lines, for path lists where an empty
            entry is not a value the way an empty prompt is.

    Returns:
        ``None`` when ``entry_file`` is ``None``; otherwise at least one entry.
        A blank line is an empty entry unless ``drop_empty`` is set.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
        ValueError: If the file holds nothing to load, which requests nothing
            rather than requesting the empty prompt.
    """
    if entry_file is None:
        return None
    entry_path = expand_path(entry_file)
    if not entry_path.is_file():
        raise FileNotFoundError(f"Input file not found: {entry_path}")

    # Preserve blank lines for empty prompts unless the caller requests otherwise.
    with entry_path.open("r") as f:
        entries = [line.strip() for line in f.readlines()]

    if drop_empty:
        entries = [entry for entry in entries if entry]

    if len(entries) == 0:
        raise ValueError(
            f"File {entry_path} holds no entry. A blank line requests the empty prompt, as in "
            "scripts/inference/prompts/empty_prompt.txt, but a path list needs a real path."
        )

    if num is not None:
        if len(entries) < num:
            logger.info(f"Repeating the last entry of {entry_path} to reach {num} entries")
            entries = entries + [entries[-1]] * (num - len(entries))
        elif len(entries) > num:
            logger.info(f"Dropping {len(entries) - num} entries from {entry_path} to reach {num}")
            entries = entries[:num]

    logger.info(f"Loaded {len(entries)} entries from {entry_path}")
    return entries


# -----------------------------------------------------------------------------
# Prompt expansion
# -----------------------------------------------------------------------------


def expand_prompts_with_qwen(
    prompts: list[str],
    model_name: str,
    device: torch.device,
    seed: int,
    images: list[Any] | None = None,
    system_prompt: str | None = None,
) -> list[str]:
    """Expand prompts with one shared Qwen/QwenVL model instance.

    Rank 0 performs the expansion and broadcasts the resulting strings, or its
    failure, when a distributed process group exists. Vision-language expansion
    requires one image per prompt.
    """
    from fastgen.third_party.wan_prompt_expand.prompt_expand import QwenPromptExpander
    from fastgen.utils import basic_utils
    from fastgen.utils.distributed.collectives import broadcast_rank0_result
    from tqdm.auto import tqdm

    is_vl = "vl" in model_name.lower()
    if is_vl and (images is None or len(images) != len(prompts) or any(image is None for image in images)):
        raise ValueError(f"Prompt-expansion model {model_name!r} requires one image per prompt")

    def expand() -> list[str]:
        expanded_prompts = list(prompts)
        prompt_expander = QwenPromptExpander(model_name=model_name, is_vl=is_vl, device=device)
        for prompt_idx in tqdm(range(len(expanded_prompts)), disable=len(expanded_prompts) <= 1):
            logger.debug(f"Expanding prompt {expanded_prompts[prompt_idx]} with seed {seed}")
            basic_utils.set_random_seed(seed)
            result = prompt_expander(
                expanded_prompts[prompt_idx],
                system_prompt=system_prompt,
                tar_lang="en",
                seed=seed,
                image=images[prompt_idx] if is_vl else None,
            )
            expanded_prompts[prompt_idx] = result.prompt
            logger.info(f"Expanded prompt: {result.prompt}")
        del prompt_expander
        gc.collect()
        torch.cuda.empty_cache()
        return expanded_prompts

    return broadcast_rank0_result(expand, "prompt expansion")


# -----------------------------------------------------------------------------
# Image preparation
# -----------------------------------------------------------------------------


def prepare_source_image_for_net(image, net: torch.nn.Module, input_shape) -> torch.Tensor:
    """Training-align one decoded image-editing reference image."""
    from PIL import Image

    from fastgen.datasets.crop_resize import transform_image

    if not isinstance(image, Image.Image):
        image = torch.as_tensor(image).detach().cpu()
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(f"Expected one image [1, H, W, C], got {tuple(image.shape)}")
            image = image[0]
        if image.ndim == 2:
            image = image.unsqueeze(-1)
        if image.ndim != 3:
            raise ValueError(f"Expected one decoded image [H, W, C], got {tuple(image.shape)}")
        if image.is_floating_point():
            if image.min() < 0 or image.max() > 1:
                raise ValueError("Floating-point image pixels must be in [0, 1]")
            image = unit_pixels_to_uint8(image)
        else:
            image = image.clamp(0, 255).to(torch.uint8)
        if image.shape[-1] == 1:
            image = image.repeat(1, 1, 3)
        elif image.shape[-1] >= 3:
            image = image[..., :3]
        else:
            raise ValueError(f"Expected 1, 3, or 4 image channels, got {image.shape[-1]}")
        image = Image.fromarray(image.numpy()).convert("RGB")

    height, width = net.pixel_shape(input_shape)
    return transform_image(image, (width, height)).unsqueeze(0)


def load_source_image_for_net(path: str | Path, net: torch.nn.Module, input_shape) -> torch.Tensor:
    """Load and training-align an image-editing reference image."""
    from PIL import Image

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Source image not found: {path}")

    with Image.open(path) as raw:
        return prepare_source_image_for_net(raw, net, input_shape)


# -----------------------------------------------------------------------------
# Video and I2V preparation
# -----------------------------------------------------------------------------


def prepare_video_frames(frames, num_frames: int, height: int, width: int) -> dict[str, Any]:
    """Transform decoded ``[T, H, W, C]`` frames exactly like the training loader."""
    from fastgen.datasets.wds_dataloaders import transform_video

    frames = torch.as_tensor(frames).detach().cpu()
    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError(f"Expected non-empty video frames [T, H, W, C], got {tuple(frames.shape)}")
    if frames.shape[-1] == 1:
        frames = frames.repeat(1, 1, 1, 3)
    elif frames.shape[-1] >= 3:
        frames = frames[..., :3]
    else:
        raise ValueError(f"Expected 1, 3, or 4 video channels, got {frames.shape[-1]}")

    if frames.is_floating_point():
        if frames.min() < 0 or frames.max() > 1:
            raise ValueError("Floating-point video frames must be in [0, 1]")
        frames = unit_pixels_to_uint8(frames)
    else:
        frames = frames.clamp(0, 255).to(torch.uint8)

    result = transform_video(frames, sequence_length=num_frames, img_size=(width, height))
    result["real_raw"] = result["real_raw"].unsqueeze(0)
    return result


def prepare_conditioning_image_for_net(image, net: torch.nn.Module, input_shape) -> dict[str, Any]:
    """Prepare one decoded I2V image at the network's pixel resolution."""
    frames = torch.as_tensor(image)
    if frames.ndim == 2:
        frames = frames.unsqueeze(-1)
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    elif frames.ndim != 4 or frames.shape[0] != 1:
        raise ValueError(f"Expected one decoded image [H, W, C] or [1, H, W, C], got {tuple(frames.shape)}")
    video_shape = input_shape["video"] if isinstance(input_shape, Mapping) else input_shape
    _, height, width = net.pixel_shape(video_shape)
    return prepare_video_frames(frames, num_frames=1, height=height, width=width)


def load_conditioning_image_for_net(image_path: str | Path, net: torch.nn.Module, input_shape) -> dict[str, Any]:
    """Load an I2V conditioning image at the network's pixel resolution."""
    import imageio.v3 as iio

    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Conditioning image not found: {path}")
    try:
        image = iio.imread(path)
    except Exception as error:
        raise RuntimeError(f"Failed to decode conditioning image: {path}") from error
    if image is None:
        raise ValueError(f"No pixels decoded from conditioning image: {path}")
    return prepare_conditioning_image_for_net(image, net, input_shape)


def total_latent_frames(input_shape, num_segments: int, overlap_frames: int) -> int:
    """Return the latent-frame span covered by autoregressive segments."""
    video_shape = input_shape["video"] if isinstance(input_shape, Mapping) else input_shape
    if len(video_shape) != 4:
        raise ValueError(f"Video conditioning requires [C, T, H, W], got {video_shape}")
    return num_segments * video_shape[1] - (num_segments - 1) * overlap_frames


def prepare_source_video_for_net(
    frames,
    net: torch.nn.Module,
    input_shape,
    latent_frame_count: int,
    source_frame_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Prepare network-FPS source frames like a full-mode training loader."""
    if source_frame_indices is None:
        source_frame_indices = list(range(len(frames)))
    elif len(source_frame_indices) != len(frames):
        raise ValueError(f"Source video has {len(frames)} frames but {len(source_frame_indices)} source-frame indices")
    video_shape = input_shape["video"] if isinstance(input_shape, Mapping) else input_shape
    latent_shape = (video_shape[0], latent_frame_count, video_shape[2], video_shape[3])
    pixel_frames, height, width = net.pixel_shape(latent_shape)
    result = prepare_video_frames(frames, num_frames=pixel_frames, height=height, width=width)
    result["source_frame_indices"] = source_frame_indices[:pixel_frames]
    return result


def load_source_video_for_net(
    video_path: str | Path,
    net: torch.nn.Module,
    input_shape,
    latent_frame_count: int,
) -> dict[str, Any]:
    """Load a network-FPS source stream and prepare its leading window like the training loader."""
    from fastgen.datasets.decoders import decode_video_full

    path = Path(video_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Source video not found: {path}")
    decoded = decode_video_full(
        path.name,
        path.read_bytes(),
        output_format="torch",
        target_fps=net.fps,
        # Disable decode_video_full's 3 s default, sized for short VLM-judge clips; a source video may be long.
        timeout_s=None,
        return_source_frame_indices=True,
    )
    if not isinstance(decoded, Mapping):
        raise RuntimeError(f"Failed to decode source video: {path}")
    return prepare_source_video_for_net(
        decoded["frames"],
        net,
        input_shape,
        latent_frame_count,
        decoded.get("source_frame_indices"),
    )


# -----------------------------------------------------------------------------
# Depth preparation
# -----------------------------------------------------------------------------


def prepare_depth_latent(depth, latent_frame_count: int) -> torch.Tensor:
    """Slice a raw depth latent to the generated latent-frame span, which it must cover."""
    if isinstance(depth, Mapping):
        depth = depth["samples"]
    depth = torch.as_tensor(depth)
    if depth.ndim == 4:
        depth = depth.unsqueeze(0)
    elif depth.ndim != 5:
        raise ValueError(f"Expected depth latent with 4 or 5 dimensions, got {tuple(depth.shape)}")
    if depth.shape[2] < latent_frame_count:
        raise ValueError(
            f"Depth latent has {depth.shape[2]} latent frames but the generated video spans {latent_frame_count}"
        )
    return depth[:, :, :latent_frame_count]


def load_raw_depth_latent(depth_path: str | Path) -> Any:
    """Load an untrimmed precomputed depth latent safely on CPU."""
    path = Path(depth_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Depth latent not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


# -----------------------------------------------------------------------------
# Camera preparation
# -----------------------------------------------------------------------------


def load_camera_poses(camera_poses_path: str | Path | None, camera_source_video: str | Path | None) -> tuple:
    """Load raw camera poses and intrinsics from arrays or VIPE."""
    import numpy as np

    if camera_poses_path:
        path = Path(camera_poses_path)
        poses_file = path / "poses.npy"
        intrinsics_file = path / "intrinsics.npy"
        if not poses_file.is_file() or not intrinsics_file.is_file():
            raise FileNotFoundError(f"camera_poses_path missing poses.npy/intrinsics.npy: {path}")
        poses = np.load(poses_file)
        intrinsics = np.load(intrinsics_file)
    elif camera_source_video:
        source_path = Path(camera_source_video)
        if not source_path.is_file():
            raise FileNotFoundError(f"camera_source_video not found: {source_path}")
        from fastgen.utils.camera_utils import encode_video_with_vipe

        poses, intrinsics = encode_video_with_vipe(str(source_path), output_path=None, keep_temp=False)
    else:
        raise ValueError("Camera loading requires a poses folder or source video")
    return torch.from_numpy(np.asarray(poses)).float(), torch.from_numpy(np.asarray(intrinsics)).float()


def prepare_camera_poses(
    poses,
    intrinsics,
    cropping_params: Mapping[str, int | float],
    source_frame_indices: list[int] | None = None,
) -> tuple:
    """Select camera rows, apply the pixel transform's crop, and add a batch dimension."""
    from fastgen.utils.camera_utils import transform_intrinsics

    poses = torch.as_tensor(poses).float()
    intrinsics = torch.as_tensor(intrinsics).float()
    if poses.ndim == 4:
        if poses.shape[0] != 1:
            raise ValueError("Camera preprocessing currently supports one batch element")
        poses = poses[0]
    if intrinsics.ndim == 3:
        if intrinsics.shape[0] != 1:
            raise ValueError("Camera preprocessing currently supports one batch element")
        intrinsics = intrinsics[0]
    if poses.ndim != 3 or intrinsics.ndim != 2:
        raise ValueError(
            "Expected camera poses [N, 4, 4] and intrinsics [N, 4], "
            f"got {tuple(poses.shape)} and {tuple(intrinsics.shape)}"
        )
    if len(poses) != len(intrinsics):
        raise ValueError(f"Camera poses/intrinsics length mismatch: {len(poses)} != {len(intrinsics)}")

    if source_frame_indices is not None:
        if not source_frame_indices:
            raise ValueError("Camera source frame indices must not be empty")
        index = torch.as_tensor(source_frame_indices, dtype=torch.long)
        if index.min() < 0 or index.max() >= len(poses):
            raise ValueError(
                f"Camera sidecar has {len(poses)} rows but source frames span "
                f"{int(index.min())}..{int(index.max())}"
            )
        poses = poses[index]
        intrinsics = intrinsics[index]

    intrinsics = transform_intrinsics(intrinsics, dict(cropping_params))
    return poses.unsqueeze(0), intrinsics.unsqueeze(0)


# -----------------------------------------------------------------------------
# Checkpoint and model setup
# -----------------------------------------------------------------------------


def load_checkpoint(
    checkpointer: Checkpointer,
    model: Any,
    ckpt_path: str,
    strict: bool = False,
) -> Optional[int]:
    """Load an explicit checkpoint into ``model.net`` and its EMA networks, if a path is given.

    The checkpoint is read on the CPU, where meta-initialized networks are also materialized;
    ``setup_inference_modules`` places the networks afterwards.

    Args:
        checkpointer: Checkpointer instance
        model: Model whose ``net`` and EMA networks receive the weights
        ckpt_path: Explicit .pth checkpoint file or DCP completion marker
        strict: Whether checkpoint keys must match the model exactly. The default tolerates
            missing or unexpected keys, such as an optional logvar head, as long as at least
            one parameter matches.

    Returns:
        Checkpoint iteration, or None when no checkpoint path was given.

    Raises:
        ValueError: A nonempty checkpoint path does not end with ``.pth``.
        FileNotFoundError: The ``.pth`` file or the DCP shards it marks do not exist.
        RuntimeError: The checkpoint matches no parameter, or holds unexpected keys under ``strict``.
            In a distributed run, each of these failures surfaces as a ``RuntimeError`` on every rank.
    """
    if not ckpt_path:
        logger.warning("No checkpoint path provided; sampling from the pretrained network")
        return None

    # Build model dict for loading
    model_dict_infer = torch.nn.ModuleDict({"net": model.net, **model.ema_dict})
    logger.info(f"ckpt_path: {ckpt_path}")

    ckpt_iter = checkpointer.load(model_dict_infer, path=ckpt_path, strict=strict)
    logger.success(f"Loading successfully checkpoint {ckpt_iter}")
    return ckpt_iter


def default_inference_output_dir(
    save_root: str | Path,
    ckpt_path: str,
    ckpt_iter: Optional[int],
) -> Path:
    """Return a checkpoint-specific default output directory.

    Interval checkpoint names that encode the loaded iteration, such as ``0000500.pth``, keep their
    basename. Other names may be mutable aliases, so append the loaded iteration to keep evaluations distinct.
    """
    if not ckpt_path or ckpt_iter is None:
        return Path(save_root) / "inference"

    file_name = Path(str(ckpt_path).rstrip("/")).name
    checkpoint_name = file_name.removesuffix(".pth")
    iteration_name = f"{ckpt_iter:07d}"
    if not checkpoint_name:
        checkpoint_name = f"checkpoint_{iteration_name}"
    elif parse_checkpoint_iteration(file_name) != ckpt_iter and not checkpoint_name.endswith(f"_{iteration_name}"):
        checkpoint_name = f"{checkpoint_name}_{iteration_name}"
    return Path(save_root) / checkpoint_name


def setup_inference_modules(
    model: Any,
    do_teacher_sampling: bool,
    do_student_sampling: bool,
) -> Tuple[Optional[Any], Optional[Any], Any]:
    """Keep and set up only the model modules selected for inference.

    This must run after checkpoint loading: every network selected for sampling is placed as a
    frozen component, with its state stored in ``model.precision_model`` and its predictions
    returning in ``model.precision``, and the roots inference does not need are discarded before
    the preprocessors are built on a surviving one. Training-only modules (fake_score,
    discriminator, reward) are already skipped at construction time when
    ``config.model.inference=True`` (set by ``load_config(..., evaluation=True)``).

    Discarded modules are removed from ``module_dict`` and their attributes are set to None
    rather than deleted, so that model_dict property accesses (which read e.g.
    ``self.fake_score``) return None instead of raising AttributeError.

    Args:
        model: The model instance
        do_teacher_sampling: Whether to set up teacher for sampling
        do_student_sampling: Whether to set up student for sampling

    Returns:
        Tuple of (teacher, student, net). Teacher and student may be None; ``net``
        is always the surviving network that hosts the preprocessors, which callers
        need for text/image encoding and decoding via ``net.preprocessors["vae"]``.
    """
    if not do_teacher_sampling and not do_student_sampling:
        raise ValueError("At least one of teacher or student sampling must be enabled")

    # A method whose training run had no teacher (e.g. plain SFT or non-CD consistency) never
    # built one. Build and load it on demand here, before any mutation below: build_teacher
    # instantiates the teacher (or base) architecture and loads `pretrained_model_path` on top of
    # whatever weights the network already loaded itself.
    if do_teacher_sampling and getattr(model, "teacher", None) is None:
        logger.info("No teacher was built for this method; building and loading one for teacher sampling.")
        model.build_teacher()

    # The module_dict entries to preserve, starting with the roots that host the sampled
    # networks; each of those is prepared exactly once, as it is selected.
    keep = set()

    student = None
    if do_student_sampling:
        student_name = model.use_ema[0] if model.use_ema else "net"
        student = model.module_dict[student_name]
        keep.add(student_name)
        model.prepare_module(student, name=student_name)
        logger.info(f"Evaluating student with {student.noise_scheduler.default_student_steps} steps.")

    teacher = None
    if do_teacher_sampling:
        # Sampling runs through the teacher when there is one -- including a SharedBaseTeacher,
        # a parameterless view over net -- but only a separate teacher is a root of its own.
        teacher = model.teacher
        teacher_name = "net" if isinstance(teacher, SharedBaseTeacher) else "teacher"
        if teacher_name not in keep:
            keep.add(teacher_name)
            model.prepare_module(model.module_dict[teacher_name], name=teacher_name)

    # Preprocessors live outside their owner's registered module tree. Preserve any that have
    # already been initialized on a selected root; init_preprocessors will build/register missing
    # ones after the unused roots are dropped.
    for name in tuple(keep):
        keep.update(model.module_dict[name].preprocessors)

    for name, module in list(model.module_dict.items()):
        if name not in keep:
            model.module_dict.pop(name)
            if getattr(model, name, None) is module:
                setattr(model, name, None)

    net = model.init_preprocessors()

    model.apply_torch_compile()  # no-op if torch_compile_mode is None; must run after init_preprocessors

    return teacher, student, net


# -----------------------------------------------------------------------------
# Shared CLI arguments
# -----------------------------------------------------------------------------


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared between image and video inference.

    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        "--ckpt_path",
        default="",
        type=str,
        help="Path to a .pth checkpoint file or DCP marker (optional, uses pretrained if not provided)",
    )
    parser.add_argument(
        "--do_student_sampling",
        default=True,
        type=str2bool,
        help="Whether to perform student sampling",
    )
    parser.add_argument(
        "--do_teacher_sampling",
        default=False,
        type=str2bool,
        help="Whether to perform teacher sampling (a method trained without a teacher builds one from its config)",
    )
