# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python scripts/inference/image_model_inference.py --config fastgen/configs/experiments/SD15/config_dmd2.py --ckpt_path /models/student/0000800.pth --prompt_file /inputs/prompts.txt --do_student_sampling False --do_teacher_sampling True --sample_steps 4 --image_save_dir /outputs/student-check - trainer.seed=42 model.use_ema=True
