#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate Cosmos-RL checkpoints on WTS with bounded GB300 video decoding."""

import gc
import logging
import threading
import time
from pathlib import Path

from cosmos_rl.evaluation.base import BaseEvaluator
from qwen_vl_utils import process_vision_info, vision_process
from vllm.model_executor.layers.conv import Conv3dLayer


def _forward_cuda_with_patch_embed_fallback(self, x):
    # Qwen3-VL patch embedding uses non-overlapping kernels, so vLLM's exact
    # unfold+linear implementation is equivalent to Conv3D. cuDNN 9.20 has no
    # usable engine for this BF16 shape on GB300 in the validated test image.
    if self.enable_linear:
        return self._forward_mulmat(x)
    return self._forward_conv(x)


Conv3dLayer.forward_cuda = _forward_cuda_with_patch_embed_fallback


_video_cache = {}
_video_cache_lock = threading.Lock()
_read_video_torchvision = vision_process.VIDEO_READER_BACKENDS["torchvision"]


def _read_video_torchvision_cached(element):
    # WTS contains several questions per clip. Cache each sampled-frame tensor
    # and serialize torchvision/PyAV access for identical decode requests.
    key = (
        element.get("video"),
        element.get("video_start"),
        element.get("video_end"),
        element.get("nframes"),
        element.get("fps"),
        element.get("min_frames"),
        element.get("max_frames"),
    )
    with _video_cache_lock:
        if key not in _video_cache:
            _video_cache[key] = _read_video_torchvision(element)
        return _video_cache[key]


vision_process.VIDEO_READER_BACKENDS["torchvision"] = (
    _read_video_torchvision_cached
)


def _prepare_chunk_inputs(evaluator, input_tasks):
    """Prepare each unique video once while rendering every question prompt."""
    media_payloads = {}
    prepared = []
    for task in input_tasks:
        prompt = task.get("prompt", [])
        media_paths = task.get("media_paths", [])
        media_mode = task.get("media_mode", "image")
        if (
            len(prompt) > 1
            and prompt[1]["role"] == "user"
            and isinstance(prompt[1]["content"], str)
        ):
            content = []
            for media_path in media_paths:
                media = {"type": media_mode, media_mode: media_path}
                media.update(evaluator.vision_config)
                content.append(media)
            content.append({"type": "text", "text": prompt[1]["content"]})
            prompt[1]["content"] = content

        rendered = evaluator.processor.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        media_key = (
            media_mode,
            tuple(media_paths),
            tuple(sorted(evaluator.vision_config.items())),
        )
        if media_key not in media_payloads:
            images, videos, video_kwargs = process_vision_info(
                prompt,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if videos:
                media_payloads[media_key] = {
                    "multi_modal_data": {"video": videos},
                    "mm_processor_kwargs": video_kwargs,
                }
            elif images:
                media_payloads[media_key] = {
                    "multi_modal_data": {"image": images}
                }
            elif media_paths:
                raise RuntimeError(
                    f"Could not prepare media for evaluation task {task.get('id')}"
                )
            else:
                media_payloads[media_key] = {}
        prepared.append({"prompt": rendered, **media_payloads[media_key]})
    return prepared


def _run_evaluation_streaming(
    self,
    results_dir: Path,
    skip_saved: bool = False,
    limit: int = -1,
    total_shard: int = 1,
    shard_id: int = 0,
):
    """Evaluate in bounded chunks instead of materializing every video input."""
    started = time.time()
    self._send_status_callback("Initializing streaming evaluation pipeline...")
    self.model, self.processor = self.load_model()

    answer_type = self.eval_config.get("answer_type", "freeform")
    chunk_size = max(
        1,
        int(
            self.eval_config.get(
                "streaming_chunk_size",
                self.eval_config.get("batch_size", 8),
            )
        ),
    )

    save_folder = self.model_config.get("save_folder")
    if save_folder:
        results_output_dir = results_dir / save_folder
    else:
        model_name = self.model_config.get("model_name", "unknown_model")
        results_output_dir = results_dir / Path(model_name).name / answer_type
    results_output_dir.mkdir(parents=True, exist_ok=True)

    shard_ids = list(range(total_shard)) if total_shard > 1 else [shard_id]
    all_outputs = []
    all_predictions = []
    all_losses = []

    for current_shard_id in shard_ids:
        inputs, outputs = self.make_tasks(
            results_output_dir, total_shard, current_shard_id
        )
        if skip_saved:
            pending = [
                (item, output)
                for item, output in zip(inputs, outputs)
                if not Path(output["output_path"]).exists()
            ]
            inputs = [item for item, _ in pending]
            outputs = [output for _, output in pending]
        if limit > 0:
            inputs = inputs[:limit]
            outputs = outputs[:limit]
        if not inputs:
            continue

        shard_predictions = []
        shard_losses = []
        total = len(inputs)
        logging.info(
            "Streaming %d evaluation tasks in chunks of %d", total, chunk_size
        )
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            chunk_inputs = inputs[start:stop]
            prepared = _prepare_chunk_inputs(self, chunk_inputs)
            predictions, losses = self.run_model_inference(
                prepared, chunk_inputs, answer_type
            )
            shard_predictions.extend(predictions)
            shard_losses.extend(losses)
            self._send_status_callback(
                f"Streaming evaluation: {stop}/{total} tasks completed"
            )
            logging.info("Streaming evaluation progress: %d/%d", stop, total)
            del prepared, predictions, losses
            with _video_cache_lock:
                _video_cache.clear()
            gc.collect()

        self._current_shard_id = current_shard_id
        self._current_total_shard = total_shard
        self.save(outputs, shard_predictions)
        all_outputs.extend(outputs)
        all_predictions.extend(shard_predictions)
        all_losses.extend(shard_losses)

    if not all_outputs:
        return {"overall": {"accuracy": 0.0, "total": 0, "correct": 0}}

    metrics = self.compute_metrics(
        results_output_dir, all_outputs, all_predictions
    )
    if all_losses:
        metrics.setdefault("overall", {})["loss"] = sum(all_losses) / len(
            all_losses
        )
    elapsed = time.time() - started
    logging.info("Streaming evaluation completed in %.2f seconds", elapsed)
    self._send_status_callback(
        f"Evaluation completed successfully in {elapsed:.1f} seconds"
    )
    return metrics


BaseEvaluator.run_evaluation = _run_evaluation_streaming


def main() -> None:
    from cosmos_rl.evaluation.evaluate import main as evaluate_main

    evaluate_main()


if __name__ == "__main__":
    main()
