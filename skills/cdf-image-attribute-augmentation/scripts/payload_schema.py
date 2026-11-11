# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Defaults and field allowlists for the ``image_attribute_augmentation_dag`` payload."""

from __future__ import annotations

DEFAULT_MODELS = {
    "vlm": "Qwen/Qwen3.8-27B-FP8",
    "llm": "Qwen/Qwen3.8-27B-FP8",
    "image_edit": "Qwen/Qwen-Image-Edit-2511",
}
DEFAULT_VARIABLE_DISTRIBUTION = {
    "variables": {
        "top_outer_color": {"black": 1.0},
        "top_outer_type": {"hoodie": 1.0},
        "bottom_type": {"jeans": 1.0},
        "bottom_color": {"blue": 1.0},
        "shoe_type": {"sneakers": 1.0},
        "shoe_color": {"white": 1.0},
    }
}
# The augmentation captioner template references every one of these placeholders.
REQUIRED_VARIABLES = tuple(DEFAULT_VARIABLE_DISTRIBUTION["variables"])
# The DAG deploys one shared Qwen VLM for both multimodal and text-only traffic, so there
# is no llm_service lifecycle entry.
SERVICE_KEYS = ("vlm_service", "image_edit_service")
ATTRIBUTE_SEARCH_KEY = "event_and_person_attribute_search"
DEFAULT_ATTRIBUTE_SEARCH_MODE = "image_pas"
SUPPORTED_CHECKERS = frozenset({"metropolis.attribute_verification"})

TOP_LEVEL_FIELDS = frozenset(
    {
        "input_path",
        "max_imgs",
        "output_directory",
        "external_services",
        "enable_performance_reporting",
        "service_lifecycle",
        "cosmos",
        ATTRIBUTE_SEARCH_KEY,
    }
)
SERVICE_LIFECYCLE_ENTRY_FIELDS = frozenset({"enabled", "replicas"})
COSMOS_FIELDS = frozenset(
    {
        "vlm_service_url",
        "llm_service_url",
        "image_edit_service_url",
        "cosmos_evaluator_service_url",
        "checker_overrides",
        "vlm_model",
        "llm_model",
        "image_edit_model",
        "num_augmentation",
        "external_services",
        "output_directory",
        "base_config_path",
        "variable_distribution",
    }
)
ATTRIBUTE_SEARCH_FIELDS = frozenset(
    {
        "mode",
        "person_input_dir",
        "vlm_service_url",
        "llm_service_url",
        "vlm_model",
        "llm_model",
        "external_services",
        "output_directory",
    }
)
VARIABLE_DISTRIBUTION_FIELDS = frozenset({"variables", "conditional_variables"})
CONDITIONAL_VARIABLE_FIELDS = frozenset({"depends_on", "distributions"})
UNSUPPORTED_FIELD_HINTS = {
    "service_lifecycle.llm_service": (
        "the shared VLM service also serves LLM traffic; configure "
        "service_lifecycle.vlm_service instead"
    ),
    f"{ATTRIBUTE_SEARCH_KEY}.config_file": "the DAG generates this file for each task; omit it",
}
