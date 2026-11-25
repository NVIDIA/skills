# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/nd_render.py: YAML writer, strict inputs and the render open/close steps."""

from __future__ import annotations

import nd_common as nd
import nd_render
import pytest
from _testkit import run_cli


def test_nd_common_re_exports_the_render_helpers() -> None:
    for name in ("to_yaml", "RenderError", "whole_number", "flag_input", "open_run", "close_run"):
        assert getattr(nd, name) is getattr(nd_render, name)


def test_yaml_is_deterministic_and_quotes_when_needed() -> None:
    text = nd_render.to_yaml({"a": [{"b": 1, "c": "yes"}, "x y"], "d": {}, "e": None, "f": True})
    assert text == 'a:\n- b: 1\n  c: "yes"\n- "x y"\nd: {}\ne: null\nf: true\n'


def test_whole_number_and_flag_input() -> None:
    assert nd_render.whole_number({"n": "3"}, "n") == 3
    assert nd_render.whole_number({}, "n", 7) == 7
    for bad in ("0", "1.5", "x", True, "-2"):
        with pytest.raises(nd_render.RenderError):
            nd_render.whole_number({"n": bad}, "n")
    assert nd_render.flag_input({"x": "off"}, "x", True) is False
    with pytest.raises(nd_render.RenderError, match="expected true or false"):
        nd_render.flag_input({"x": "maybe"}, "x")


def test_library_module_prints_a_contract_object_when_run() -> None:
    code, payload = run_cli("nd_render.py")
    assert (code, payload["state"]) == (0, "library")
