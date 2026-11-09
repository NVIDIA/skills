# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Media I/O utilities — saving images, audio, videos, and audio-video to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from einops import rearrange
import imageio.v3 as iio
from PIL import Image

from fastgen.datasets.video_resampling import validate_frame_rate
from fastgen.utils.basic_utils import inference_mode, cuda_timer
from fastgen.utils.tensor import pixels_to_uint8, waveform_to_pcm16
from fastgen.utils.tree_utils import TensorOrDict, get_batch_size, get_device, get_shape
import fastgen.utils.logging_utils as logger

DEFAULT_AUDIO_SAMPLE_RATE = 24000


def save_audio(
    audio: torch.Tensor, path: str | Path, audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE, **kwargs
) -> str:
    """Write a mono ``[samples]`` or channel-first ``[1|2, samples]`` waveform as PCM16 WAV.

    Samples are clipped to ``[-1, 1]``. Video encoding options are ignored so the
    same media options can be used for every modality.
    """
    import wave

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2 or audio.shape[0] not in (1, 2):
        raise ValueError(f"Expected a channel-first mono or stereo waveform, got {tuple(audio.shape)}")
    if audio_sample_rate <= 0:
        raise ValueError(f"audio_sample_rate must be positive, got {audio_sample_rate}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = waveform_to_pcm16(audio.detach().cpu()).T.contiguous()
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(audio.shape[0])
        stream.setsampwidth(2)
        stream.setframerate(audio_sample_rate)
        stream.writeframes(pcm.numpy().astype("<i2", copy=False).tobytes())
    return str(path)


def save_audio_video(
    video: torch.Tensor,
    audio: torch.Tensor | None,
    path: str,
    fps: float = 16,
    audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
    quality: int = 23,
    preset: str = "medium",
    x264_options: dict[str, str] | None = None,
) -> str:
    """Mux decoded video frames and audio waveform into an MP4 file via PyAV.

    Args:
        video: Decoded video tensor ``[C, T, H, W]`` in ``[-1, 1]``.
        audio: Mono or stereo waveform, as [samples], [samples, channels], or
            [channels, samples] (optionally with a leading size-1 batch dimension).
            Trimmed to the video duration. Use *None* for video-only.
        path: Output ``.mp4`` file path.
        fps: Video frames per second.
        audio_sample_rate: Audio sample rate in Hz (default 24 kHz).
        quality: H.264 constant-rate factor (0-51; lower is better).
        preset: ``libx264`` encoding-speed preset.
        x264_options: Additional ``libx264`` options. Explicit entries override
            the options derived from ``quality`` and ``preset``.

    Returns:
        *path* for convenience.
    """
    import av as _av
    import math
    from fractions import Fraction

    fps = validate_frame_rate(fps, "fps")
    frame_rate = Fraction(str(fps)).limit_denominator()
    if not math.isclose(frame_rate, fps, rel_tol=1e-6):
        raise ValueError(f"fps={fps} cannot be stored accurately as a video frame rate (nearest is {frame_rate})")
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    video_np = pixels_to_uint8(video.permute(1, 2, 3, 0)).cpu().numpy()
    audio_t = audio.float().cpu() if audio is not None else None

    T, H, W, _ = video_np.shape
    with _av.open(path, mode="w") as container:
        v_stream = container.add_stream("libx264", rate=frame_rate)
        v_stream.width = W
        v_stream.height = H
        v_stream.pix_fmt = "yuv420p"
        resolved_x264_options = {"crf": str(quality), "preset": preset}
        resolved_x264_options.update(x264_options or {})
        v_stream.options = resolved_x264_options

        a_stream = None
        if audio_t is not None:
            a_stream = container.add_stream("aac", rate=audio_sample_rate)
            a_stream.codec_context.sample_rate = audio_sample_rate
            a_stream.codec_context.layout = "stereo"
            a_stream.codec_context.time_base = Fraction(1, audio_sample_rate)

        for frame_array in video_np:
            frame = _av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for pkt in v_stream.encode(frame):
                container.mux(pkt)
        for pkt in v_stream.encode():
            container.mux(pkt)

        if a_stream is not None and audio_t is not None:
            if audio_t.ndim == 1:
                audio_t = audio_t.unsqueeze(1).repeat(1, 2)
            elif audio_t.ndim == 2 and audio_t.shape[0] in (1, 2) and audio_t.shape[1] != 2:
                audio_t = audio_t.T
            elif audio_t.ndim == 3:
                audio_t = audio_t.squeeze(0)
                if audio_t.shape[0] <= 2 and audio_t.shape[1] > 2:
                    audio_t = audio_t.T
            if audio_t.ndim != 2 or audio_t.shape[1] not in (1, 2):
                raise ValueError(f"Expected a mono or stereo waveform, got audio of shape {tuple(audio.shape)}")
            if audio_t.shape[-1] == 1:
                audio_t = audio_t.repeat(1, 2)

            # Audio latents may outlast the picture; cut to the video duration so both streams end together.
            audio_t = audio_t[: min(audio_t.shape[0], round(T / fps * audio_sample_rate))]

            audio_int16 = waveform_to_pcm16(audio_t)
            audio_interleaved = audio_int16.contiguous().view(1, -1).numpy()

            af = _av.AudioFrame.from_ndarray(audio_interleaved, format="s16", layout="stereo")
            af.sample_rate = audio_sample_rate

            resampler = _av.audio.resampler.AudioResampler(
                format=a_stream.codec_context.format,
                layout=a_stream.codec_context.layout,
                rate=audio_sample_rate,
            )
            pts = 0
            for resampled in resampler.resample(af):
                resampled.pts = pts
                pts += resampled.samples
                for pkt in a_stream.encode(resampled):
                    container.mux(pkt)
            for pkt in a_stream.encode():
                container.mux(pkt)

    return path


def save_media(
    output: TensorOrDict,
    save_path: str,
    vae: Callable | None = None,
    precision_amp: torch.dtype | None = None,
    always_index: bool = False,
    **kwargs,
):
    """Save a tensor or multimodal output to disk.

    Args:
        output: Media tensor to save. Can be ``[B, C, H, W]`` (image),
            ``[B, C, T, H, W]`` (video, including ``T == 1``), ``[B, C, samples]``
            or ``[B, samples]`` (audio), an audio-video dict
            (``{"video": ..., "audio": ...}`` — muxed as MP4), or a
            single-modality dict (``{"video": ...}`` / ``{"image": ...}`` / ``{"audio": ...}``
            — unwrapped to the bare tensor and saved as ordinary
            video / image / audio). Audio waveforms use channel-first layout and WAV output.
        save_path: Path to save the media. If multiple instances are saved,
            an index will be added to the file name. Its suffix is replaced
            with ``.wav`` for waveforms and ``.mp4`` for audio-video outputs.
        vae: Optional VAE decoder to decode the tensor.
        precision_amp: If not None, wraps the decode in
            ``torch.amp.autocast()`` with the given precision.
        always_index: When True, append the ``_<i:03d>`` suffix even for size-1
            batches so the on-disk filename is uniform regardless of batch size.
            Default False preserves the historical "size-1 → original path"
            behavior expected by inference-script callers.
        **kwargs: Additional encoding parameters for save_image / save_video
            / save_audio / save_audio_video (``fps``, ``audio_sample_rate``, etc.).
            Audio-video outputs ignore a ``vid_format`` other than ``"mp4"``
            with a warning; ``save_audio_video`` rejects any other option it does
            not define, whereas the tensor writers ignore unrelated ones.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"🔍 Media tensor input shape: {get_shape(output)}")

    if vae is not None:
        with inference_mode(vae, precision_amp=precision_amp, device_type=get_device(output).type):
            with cuda_timer("VAE decode time"):
                output = vae.decode(output)
            logger.debug(f"🔍 Media tensor output shape after VAE decode: {get_shape(output)}")

    # Unwrap single-modality dicts for the tensor writers. Audio-video dicts
    # stay wrapped and route through the MP4 muxer below.
    if isinstance(output, dict) and len(output) == 1:
        if "video" in output:
            output = output["video"]
        elif "image" in output:
            output = output["image"]
        elif "audio" in output:
            output = output["audio"]

    is_av = isinstance(output, dict) and all(k in output for k in ("video", "audio"))

    if is_av:
        # GIF cannot carry audio, so audio-video outputs are always muxed as MP4.
        vid_format = kwargs.pop("vid_format", "mp4")
        if vid_format != "mp4":
            logger.warning(f"vid_format={vid_format!r} cannot carry audio; saving the audio-video output as mp4")
        save_path = save_path.with_suffix(".mp4")
    elif not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor, {{'video','audio'}} dict, or a single-modality dict, got {type(output)}")
    elif output.ndim in (2, 3):
        # ``save_audio`` only writes WAV.
        save_path = save_path.with_suffix(".wav")

    n_samples = get_batch_size(output)

    for i in range(n_samples):
        media_save_path = (
            save_path.parent / (save_path.stem + f"_{i:03d}" + save_path.suffix)
            if n_samples > 1 or always_index
            else save_path
        )

        if is_av:
            save_audio_video(output["video"][i], output["audio"][i], str(media_save_path), **kwargs)
        else:
            media_tensor = output[i]
            if media_tensor.ndim in (1, 2):
                save_audio(media_tensor, media_save_path, **kwargs)
            elif media_tensor.ndim == 3:
                save_image(media_tensor, media_save_path, **kwargs)
            elif media_tensor.ndim == 4:
                save_video(media_tensor, media_save_path, **kwargs)
            else:
                raise ValueError(f"Tensor has invalid shape: {media_tensor.shape}")

    if n_samples > 1 or always_index:
        logger.info(f"Saved {n_samples} {save_path.stem}_*{save_path.suffix} to {save_path.parent}")
    else:
        logger.info(f"Saved {save_path.name} to {save_path.parent}")


def save_image(tensor: torch.Tensor, save_path: str | Path, **kwargs):
    """Save a tensor image to disk.

    Args:
        tensor: Image tensor of shape [C, H, W] in range [-1, 1]
        save_path: Path to save the image
    """
    tensor = pixels_to_uint8(tensor)

    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)

    logger.debug(f"🔍 After permute shape: {tensor.shape}")
    logger.debug(f"🔍 Max/min value: {tensor.max()}, {tensor.min()}")

    image = Image.fromarray(tensor.cpu().numpy())

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)
    logger.debug(f"Saved image {save_path.name} to {save_path.parent}")


def save_video(
    frames: torch.Tensor,
    save_path: str = "sample0.mp4",
    vid_format: str = "gif",
    fps: int = 16,
    quality: int = 23,
    **kwargs,
):
    """
    Save video with basic quality control and silent encoding.

    Args:
        frames: Video frames tensor to save [C, T, H, W]
        save_path: Full path including filename
        vid_format: Container format, either ``"gif"`` or ``"mp4"``
        fps: Frames per second for playback (not frame count)
        quality: Video quality 0-51 (lower=better, default: 23)
        **kwargs: Additional encoding parameters
    """
    if vid_format not in ("gif", "mp4"):
        raise ValueError(f"save_video: vid_format must be 'gif' or 'mp4', got {vid_format!r}")
    frames = rearrange(frames, "C T H W -> T H W C")
    logger.debug(f"🔍 After rearrange shape: {frames.shape}")
    logger.debug(f"🔍 Final frame count: {frames.shape[0]} frames")
    logger.debug(f"🔍 Expected duration at {fps}fps: {frames.shape[0]/fps:.1f} seconds")

    frames = pixels_to_uint8(frames).cpu()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if vid_format == "gif":
        save_path = save_path.with_suffix(".gif")
        iio.imwrite(save_path, frames, duration=1000 / fps, loop=kwargs.get("loop", 0))
    else:
        output_params = [
            "-loglevel",
            "quiet",
            "-hide_banner",
            "-nostats",
            "-crf",
            str(quality),
            "-preset",
            kwargs.get("preset", "medium"),
        ]

        iio.imwrite(
            save_path,
            frames,
            fps=fps,
            codec="libx264",
            output_params=output_params,
        )
    logger.debug(f"Saved video {save_path.name} to {save_path.parent}")
