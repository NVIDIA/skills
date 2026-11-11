# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video generation inference script.

Supports:
- Text-to-video (T2V): Wan 2.1/2.2
- Image-to-video (I2V): Wan I2V
- Video-to-video (V2V): VACE Wan, Self-Forcing
- Video2World: Cosmos Predict2
- Text-to-audio-video (T2AV): LTX-2

The config override ``model.net.fps`` sets the video frame rate for both temporal conditioning and saved
videos; also set ``model.teacher.fps`` for a separate teacher.

Examples:

    # T2V: eval teacher only (Wan)
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \\
        scripts/inference/video_model_inference.py --do_student_sampling False --do_teacher_sampling True \\
        --neg_prompt_file scripts/inference/prompts/negative_prompt.txt \\
        --config fastgen/configs/experiments/WanT2V/config_dmd2.py \\
        - trainer.seed=1 model.guidance_cfg.cfg.scale=5.0 log_config.name=wan_t2v_inference

    # I2V: image-to-video (Wan I2V)
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \\
        scripts/inference/video_model_inference.py --do_student_sampling False --do_teacher_sampling True \\
        --neg_prompt_file scripts/inference/prompts/negative_prompt.txt \\
        --input_image_file scripts/inference/prompts/source_image_paths.txt \\
        --config fastgen/configs/experiments/WanI2V/config_dmd2_14b.py \\
        - trainer.seed=1 model.guidance_cfg.cfg.scale=5.0 log_config.name=wan_i2v_inference

    # V2V: sample the fine-tuned VACE student
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \\
        scripts/inference/video_model_inference.py \\
        --neg_prompt_file scripts/inference/prompts/negative_prompt.txt \\
        --source_video_file scripts/inference/prompts/source_video_paths.txt \\
        --config fastgen/configs/experiments/WanV2V/config_sft.py \\
        - trainer.seed=1 model.guidance_cfg.cfg.scale=5.0 log_config.name=vace_wan_inference

    # Video2World: sample the fine-tuned Cosmos Predict2 student
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \\
        scripts/inference/video_model_inference.py \\
        --neg_prompt_file scripts/inference/prompts/negative_prompt_cosmos.txt \\
        --input_image_file scripts/inference/prompts/source_image_paths.txt \\
        --config fastgen/configs/experiments/CosmosPredict2/config_sft_v2w.py \\
        - log_config.name=cosmos_v2w_inference

    # Eval with skip-layer guidance (SLG)
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \\
        scripts/inference/video_model_inference.py --do_student_sampling False --do_teacher_sampling True \\
        --neg_prompt_file scripts/inference/prompts/negative_prompt.txt \\
        --config fastgen/configs/experiments/WanT2V/config_dmd2.py \\
        - trainer.seed=1 model.guidance_cfg.cfg.scale=6.0 model.guidance_cfg.cfg.skip_layers=[10] \\
        log_config.name=wan_slg_inference

    # Eval student and teacher together
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \\
        scripts/inference/video_model_inference.py --ckpt_path /path/to/checkpoint.pth \\
        --do_student_sampling True --do_teacher_sampling True \\
        --neg_prompt_file scripts/inference/prompts/negative_prompt.txt \\
        --config fastgen/configs/experiments/WanT2V/config_dmd2.py \\
        - trainer.seed=1 log_config.name=wan_student_teacher_inference

    # T2AV: text-to-audio-video (LTX-2, separate guidance scales for video and audio)
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=1 --standalone \\
        scripts/inference/video_model_inference.py \\
        --neg_prompt_file scripts/inference/prompts/negative_prompt_ltx.txt \\
        --config fastgen/configs/experiments/LTX2/config_sft.py \\
        - trainer.seed=1 'model.guidance_cfg.cfg.scale={"video": 3.0, "audio": 7.0}' \\
        log_config.name=ltx2_t2av_inference

    # Guidance-distilled networks consume no negative branch:
    # omit --neg_prompt_file and no negative prompt is built at all.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fastgen.configs.config import BaseConfig
import fastgen.utils.logging_utils as logger
from fastgen.utils.distributed.collectives import clean_up
from fastgen.configs.instantiate import instantiate
from fastgen.utils import basic_utils
from fastgen.utils.env import str2bool
from fastgen.utils.tree_utils import noise_from_shape
from fastgen.utils.media_utils import save_media
from fastgen.utils.basic_utils import cuda_timer
from fastgen.utils.scripts import parse_args, setup
from fastgen.utils.checkpointer import Checkpointer
from scripts.inference.inference_utils import (
    expand_prompts_with_qwen,
    load_conditioning_image_for_net,
    load_source_video_for_net,
    load_raw_depth_latent,
    prepare_depth_latent,
    load_camera_poses,
    prepare_camera_poses,
    load_file_entries,
    load_checkpoint,
    default_inference_output_dir,
    setup_inference_modules,
    add_common_args,
    total_latent_frames,
)


def main(args, config: BaseConfig):
    pos_prompt_set = load_file_entries(args.prompt_file)

    # Align optional inputs with the prompts; omitted inputs remain ``None``.
    neg_prompt_set = load_file_entries(args.neg_prompt_file, num=len(pos_prompt_set))
    path_list_kwargs = dict(num=len(pos_prompt_set), drop_empty=True)
    depth_latent_paths = load_file_entries(args.depth_latent_file, **path_list_kwargs)
    source_video_paths = load_file_entries(args.source_video_file, **path_list_kwargs)
    input_image_paths = load_file_entries(args.input_image_file, **path_list_kwargs)
    camera_poses_paths = load_file_entries(args.camera_poses_file, **path_list_kwargs)
    camera_source_video_paths = load_file_entries(args.camera_source_video_file, **path_list_kwargs)

    # Prompt expansion if specified; the vision-language expanders read the per-prompt input images.
    if args.prompt_expand_model:
        pos_prompt_set = expand_prompts_with_qwen(
            pos_prompt_set,
            args.prompt_expand_model,
            torch.device(config.model.device),
            args.prompt_expand_model_seed,
            images=input_image_paths,
        )

    # Fix sampling seeds
    seed = basic_utils.set_random_seed(config.trainer.seed, by_rank=True)

    # Initialize model and checkpointer (config.model.inference was set to
    # True by ``setup(..., evaluation=True)``, skipping training-only modules).
    model = instantiate(config.model_class, config=config.model)
    # Checkpoint contents select sharded loading or delegation to the basic checkpointer.
    checkpointer = Checkpointer(config.trainer.checkpointer)

    # Load checkpoint
    ckpt_iter = load_checkpoint(checkpointer, model, args.ckpt_path)

    if ckpt_iter is None and args.do_student_sampling:
        logger.warning(
            f"Performing {model.net.noise_scheduler.default_student_steps}-step generation on the non-distilled model"
        )

    # Set up save directory. Explicit destinations are caller-owned; the default includes
    # checkpoint identity so evaluating another checkpoint cannot overwrite this run.
    prompt_name = Path(args.prompt_file).stem
    if args.prompt_expand_model:
        prompt_name += f"_{args.prompt_expand_model}"
    save_root = (
        Path(args.video_save_dir)
        if args.video_save_dir
        else default_inference_output_dir(config.log_config.save_path, args.ckpt_path, ckpt_iter)
    )
    save_dir = save_root / prompt_name
    logger.info(f"video_save_dir: {save_dir}")

    # Prune and set up inference modules (also calls apply_torch_compile internally)
    teacher, student, net = setup_inference_modules(model, args.do_teacher_sampling, args.do_student_sampling)
    vae = net.preprocessors.get("vae")
    ctx = {"dtype": model.precision, "device": model.device}

    # Which sampling paths setup selected. ``setup_inference_modules`` already rejects a request
    # that selects neither, and builds a requested teacher that the method lacks.
    has_teacher_sampling = teacher is not None and hasattr(teacher, "sample")
    has_student_sampling = student is not None

    save_video_kwargs = {"precision_amp": model.precision_amp_enc, "vid_format": args.vid_format}
    if args.save_high_quality:
        save_video_kwargs.update(quality=18, preset="medium")
        save_dir = save_dir.parent / (save_dir.name + "_hq")
    if vae is not None and hasattr(vae, "vocoder_sample_rate"):
        save_video_kwargs["audio_sample_rate"] = vae.vocoder_sample_rate

    # Fix noise for all generated samples
    noise = noise_from_shape(model.input_shape, batch_size=1, **ctx)
    for i, prompt in enumerate(pos_prompt_set):
        logger.info(f"[{i+1}/{len(pos_prompt_set)}] Generating: {prompt[:80]}...")

        # Conditioning: load whatever pixel inputs the user supplied into
        # ``data["real_raw"]`` (and ``data["depth_latent"]`` for VACE), then
        # call the network's ``build_conditioning`` once.  The network is
        # the single source of truth for what data it needs; T2V networks
        # ignore ``real_raw`` and pass through, I2V/V2W/V2V networks
        # raise ``ValueError`` if required keys are missing.
        data = {"condition_raw": [prompt]}
        if neg_prompt_set is not None:
            data["neg_condition_raw"] = [neg_prompt_set[i]]
        has_image = input_image_paths is not None
        has_video = source_video_paths is not None
        has_depth = depth_latent_paths is not None
        cropping_params = None
        source_frame_indices = None
        if has_image and has_video:
            logger.warning(
                "Both --input_image_file and --source_video_file are provided; "
                "the source video takes precedence and the image is ignored."
            )
        if has_image and not has_video:
            prepared = load_conditioning_image_for_net(input_image_paths[i], net, model.input_shape)
            data["real_raw"] = prepared["real_raw"]
            cropping_params = prepared["cropping_params"]
        if has_video or has_depth:
            latent_frame_count = total_latent_frames(model.input_shape, args.num_segments, args.overlap_frames)
            if has_video:
                prepared = load_source_video_for_net(source_video_paths[i], net, model.input_shape, latent_frame_count)
                data["real_raw"] = prepared["real_raw"]
                cropping_params = prepared["cropping_params"]
                source_frame_indices = prepared["source_frame_indices"]
            if has_depth:
                data["depth_latent"] = prepare_depth_latent(
                    load_raw_depth_latent(depth_latent_paths[i]), latent_frame_count
                )
        # Camera conditioning (WanCamI2V / LingBot-world): per-prompt poses folder
        # or source video -> poses/intrinsics merged into ``data`` so
        # build_conditioning emits the Plücker embedding (same contract as
        # training).
        has_camera_poses = camera_poses_paths is not None
        has_camera_video = camera_source_video_paths is not None
        if has_camera_poses or has_camera_video:
            if cropping_params is None:
                raise ValueError("Camera conditioning requires an input image or source video")
            camera_poses, camera_intrinsics = load_camera_poses(
                camera_poses_paths[i] if has_camera_poses else None,
                camera_source_video_paths[i] if has_camera_video else None,
            )
            data["camera_poses"], data["camera_intrinsics"] = prepare_camera_poses(
                camera_poses,
                camera_intrinsics,
                cropping_params,
                source_frame_indices,
            )
        data = model.prepare_batch(data, inference_mode=True, materialize_real=False)
        # Student sampling
        if has_student_sampling:
            use_extrapolation = args.num_segments != 1 or args.overlap_frames != 0

            with cuda_timer("Student sampling time"):
                if use_extrapolation:
                    if not hasattr(model, "generator_fn_extrapolation"):
                        raise RuntimeError(
                            "Extrapolation (--num_segments/--overlap_frames) requires a method with "
                            "generator_fn_extrapolation, e.g. a CausVid or Self-Forcing config; "
                            f"{type(model).__name__} has none"
                        )
                    video_student = model.generator_fn_extrapolation(
                        student,
                        noise,
                        num_segments=args.num_segments,
                        overlap_frames=args.overlap_frames,
                        condition=data.get("condition"),
                    )
                else:
                    video_student = model.generator_fn(
                        student,
                        noise,
                        condition=data.get("condition"),
                        neg_condition=data.get("neg_condition"),
                    )

            save_path = save_dir / f"student_{i:04d}_seed{seed}.mp4"
            save_media(video_student, str(save_path), vae=vae, fps=student.fps, **save_video_kwargs)

        # Teacher sampling
        if has_teacher_sampling:
            teacher_kwargs = {
                "condition": data.get("condition"),
                "neg_condition": data.get("neg_condition"),
            }
            # Only override sample_steps when explicitly set; otherwise let net.sample()
            # use its own default.
            if args.sample_steps is not None:
                teacher_kwargs["sample_steps"] = args.sample_steps

            with cuda_timer("Teacher sampling time"):
                video_teacher = model.sample(teacher, noise, **teacher_kwargs)

            save_path = save_dir / f"teacher_{i:04d}_seed{seed}.mp4"
            save_media(video_teacher, str(save_path), vae=vae, fps=teacher.fps, **save_video_kwargs)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video model inference")

    # Add common args
    add_common_args(parser)

    # Video-specific args
    parser.add_argument(
        "--vid_format",
        default="mp4",
        choices=("mp4", "gif"),
        help="Container for saved videos (default: mp4); audio-video outputs always use mp4",
    )
    parser.add_argument(
        "--save_high_quality",
        default=False,
        type=str2bool,
        help="Whether to use a lower CRF for higher-quality MP4 output, saved to a directory with an _hq suffix "
        "(default: False)",
    )
    parser.add_argument(
        "--prompt_file",
        default="scripts/inference/prompts/video_prompts.txt",
        type=str,
        help="File containing prompts (one per line). Relative paths are resolved from the current working directory.",
    )
    parser.add_argument(
        "--neg_prompt_file",
        default=None,
        type=str,
        help="Negative prompt file for CFG, one prompt per --prompt_file prompt. Omit it for no "
        "negative conditioning, which guidance-distilled networks want, or pass "
        "scripts/inference/prompts/empty_prompt.txt for the empty negative prompt. Omitting it "
        "does not disable an active model.guidance_cfg.cfg.scale.",
    )
    parser.add_argument(
        "--prompt_expand_model",
        type=str,
        help="If specified, expand each prompt with this Qwen model before sampling. The vision-language QwenVL "
        "models also read --input_image_file. See fastgen/third_party/wan_prompt_expand/README.md.",
        choices=["QwenVL2.5_3B", "QwenVL2.5_7B", "Qwen2.5_3B", "Qwen2.5_7B", "Qwen2.5_14B"],
    )
    parser.add_argument(
        "--prompt_expand_model_seed",
        type=int,
        help="Seed for prompt expansion (default: 0).",
        default=0,
    )
    parser.add_argument(
        "--depth_latent_file",
        default=None,
        type=str,
        help="File containing precomputed depth latent paths for depth-conditioned V2V (one per line, aligned "
        "with the prompts like --input_image_file).",
    )
    parser.add_argument(
        "--source_video_file",
        default=None,
        type=str,
        help="File containing source video paths (one per line, aligned with the prompts like --input_image_file) "
        "for V2V mode. Default: None — "
        "V2V is disabled and the network falls back to T2V/I2V behavior. When set together with "
        "--input_image_file, the source video takes precedence per prompt. Videos are resampled to "
        "the inference network's FPS during decoding.",
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        default=1,
        help="Number of autoregressive segments to generate when using extrapolation (default: 1)",
    )
    parser.add_argument(
        "--overlap_frames",
        type=int,
        default=0,
        help="Number of latent frames to overlap between segments when extrapolating (default: 0). Each segment "
        "chunks its latent frames like training, with any remainder in the first chunk, so the overlap must end on a "
        "chunk boundary: 0, the first chunk's length, or that plus a multiple of the network's chunk_size.",
    )
    parser.add_argument(
        "--video_save_dir",
        type=str,
        help="Directory for generated videos, with one subdirectory per prompt file. Defaults to a directory in "
        "the run directory (log_config.save_path): inference/ without --ckpt_path, otherwise one named after "
        "the checkpoint.",
        default=None,
    )
    parser.add_argument(
        "--sample_steps",
        default=None,
        type=int,
        help="Number of teacher denoising steps, i.e. sampling-grid intervals rather than grid points (default: the "
        "network's own default; MiniMax-H3's 49 is SGLang's num_inference_steps=50). The student, including an SFT "
        "model, samples model.student_grid_cfg.steps steps instead.",
    )
    # I2V arguments
    parser.add_argument(
        "--input_image_file",
        type=str,
        default=None,
        help="File containing paths to input images (one per line) for I2V mode (or video2world mode in cosmos). "
        "Images are aligned with prompts; if fewer images than prompts, the last image is repeated. "
        "Default: None — no image conditioning (T2V networks ignore real_raw; I2V/V2W networks raise).",
    )
    # Camera conditioning (WanCamI2V / LingBot-world); per-prompt, one path per line.
    parser.add_argument(
        "--camera_poses_file",
        type=str,
        default=None,
        help="File with per-prompt camera-pose folder paths (one per line, aligned with the prompts like "
        "--input_image_file; each a folder with poses.npy + intrinsics.npy) for camera-conditioned I2V "
        "(WanCamI2V/LingBot-world).",
    )
    parser.add_argument(
        "--camera_source_video_file",
        type=str,
        default=None,
        help="File with per-prompt video paths (one per line, aligned with the prompts like --input_image_file) "
        "for on-the-fly VIPE camera-pose extraction (requires the optional vipe package).",
    )
    args = parse_args(parser)
    config = setup(args, evaluation=True)
    main(args, config)

    clean_up()

# ----------------------------------------------------------------------------
