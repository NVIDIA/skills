# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bundled and adapted from the Apache-2.0 NVIDIA TAO Tutorials PAS DEFT utilities so the
# customer workflow does not depend on an external source checkout.

"""Bundled CLIP DEFT pipeline utilities used by this skill."""

from pas_deft.config import PasDeftConfig, config_field_metadata

__all__ = ["PasDeftConfig", "config_field_metadata"]
