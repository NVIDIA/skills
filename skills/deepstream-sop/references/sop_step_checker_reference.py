# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Module for checker service.

This service is responsible for checking SOP (Standard Operating Procedure) compliance.

It will:
- Get a request from the API server
- Receive action JSON and VLM outputs
- Process the data to check SOP compliance
- Send a message to the API server to indicate the completion of the SOP checking and the results
"""

import json
import logging
import os
import re
import time
import traceback
import uuid
from dataclasses import asdict, dataclass

from . import ds_logger
from .missing_number_detector import MissingNumberDetector

_LOGGER = ds_logger.get_logger(__name__)


@dataclass
class SopCheckerRequest:
    # The content of the actions.json
    action_json: str
    # the output of the VLM inference.
    vlm_output: str
    # If this checker should be kept and used for the next request.
    # If true, the response would contain a valid id for users to send the next request.
    # If false, the checker would be cleared.
    keep_alive: bool
    # If this ID is not empty the service would try to find the corresponding checker and continue from there.
    # Use special value "*" for the very first request.
    # Raise error if the checker is not found.
    checker_id: str
    # Options for the SOP detection.
    cycle_completion_threshold: float
    cycle_boundary_threshold_low: float
    cycle_boundary_threshold_high: float


@dataclass
class SopCheckerResponse:
    request_id: str
    error_message: str
    # If keep_alive is true, reply a valid checker_id for the next request.
    checker_id: str
    cycle: int
    missing_detected: list[int]
    misordered_detected: list[int]
    final_missing_detected: list[int]
    final_misordered_detected: list[int]
    cycle_completed: bool
    summary_cycles_detected: list[str]
    summary_cycle_analysis: list[str]

    def asdict(self):
        return asdict(self)


def read_sop_steps(content: str) -> list[str]:
    """Parse SOP steps from content string and return them as a list.

    Args:
        content (str): The content string containing SOP steps

    Returns:
        list[str]: List of SOP steps in order of appearance
    """
    # Initialize list to store SOP steps
    sop_steps = []

    # Normalize newlines in content
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # Process content directly
    step_start = 0
    while True:
        # Find next opening parenthesis
        open_paren = content.find("(", step_start)
        if open_paren == -1:
            break

        # Find closing parenthesis
        close_paren = content.find(")", open_paren)
        if close_paren == -1:
            break

        # Extract step number and content
        step_num = content[open_paren + 1 : close_paren]

        # Find start of next step or end of content
        next_step = content.find("(", close_paren + 1)
        if next_step == -1:
            step_content = content[close_paren + 1 :]
        else:
            step_content = content[close_paren + 1 : next_step]

        # Replace any newlines with spaces in the step content
        step_content = step_content.replace("\n", " ")
        # Strip the trailing space
        step_content = step_content.rstrip()

        # Add the full step to our list
        sop_steps.append(f"({step_num}){step_content}")

        # Move to next potential step
        step_start = close_paren + 1

    return sop_steps


class SopCheckerCache:
    def __init__(self):
        self._CHECKER_CACHE = {}
        self._checker = None

    def load_checker(self, checker_id: str) -> tuple[MissingNumberDetector, list[int]]:
        """
        Load the checker from the cache.
        """
        if checker_id in self._CHECKER_CACHE:
            return self._CHECKER_CACHE[checker_id]
        raise ValueError(
            f"Checker {checker_id} not found. If this is the first request, please use '*' as the checker_id."
        )

    def save_checker(self, checker: MissingNumberDetector, actions_can_be_skipped: list[int]) -> str:
        """
        Save the checker to the cache.
        """
        checker_id = f"sopchecker-{str(uuid.uuid4())}"
        self._CHECKER_CACHE[checker_id] = (checker, actions_can_be_skipped)
        _LOGGER.debug("Checker is saved to cache with id: %s", checker_id)
        return checker_id

    def try_delete_checker(self, checker_id: str):
        """
        Try to delete the checker from the cache.
        """
        if checker_id in self._CHECKER_CACHE:
            del self._CHECKER_CACHE[checker_id]
            _LOGGER.debug("Checker is deleted from cache with id: %s", checker_id)

    def process_sop_check(self, request_id: str, sop_checker_request: SopCheckerRequest) -> SopCheckerResponse:
        """
        Process SOP checking logic.

        Args:
            action_json: The content of the actions.json
            vlm_outputs: The outputs of the VLM inference
            draw_fsm: Whether to draw the FSM image

        Returns:
            tuple: (num_sop_detected, sop_final_state, fsm_image)
        """

        action_json = sop_checker_request.action_json
        vlm_output = sop_checker_request.vlm_output
        keep_alive = sop_checker_request.keep_alive
        checker_id = sop_checker_request.checker_id
        cycle_completion_threshold = sop_checker_request.cycle_completion_threshold
        cycle_boundary_threshold_low = sop_checker_request.cycle_boundary_threshold_low
        cycle_boundary_threshold_high = sop_checker_request.cycle_boundary_threshold_high

        reobj_capture_number = re.compile(r"^\((\d+)\).+")

        if checker_id == "*":
            sop_data = json.loads(action_json)

            actions = sop_data["actions"]
            action_numbers = [int(reobj_capture_number.match(action).group(1)) for action in actions]

            actions_can_be_skipped = sop_data.get("actions_can_be_skipped", [])
            actions_can_be_skipped_numbers = [
                int(reobj_capture_number.match(action).group(1)) for action in actions_can_be_skipped
            ]

            action_numbers = [num for num in action_numbers if num not in actions_can_be_skipped_numbers]
            index_to_action_number = {i + 1: action_numbers[i] for i in range(len(action_numbers))}

            _LOGGER.debug(
                "Creating checker with options: "
                "cycle_completion_threshold=%s, "
                "cycle_boundary_threshold_low=%s, "
                "cycle_boundary_threshold_high=%s",
                cycle_completion_threshold,
                cycle_boundary_threshold_low,
                cycle_boundary_threshold_high,
            )
            checker = MissingNumberDetector(
                len(action_numbers),
                index_to_action_number,
                cycle_completion_threshold=cycle_completion_threshold,
                cycle_boundary_threshold_low=cycle_boundary_threshold_low,
                cycle_boundary_threshold_high=cycle_boundary_threshold_high,
            )
            # checker is not yet cached; it will be saved at keep_alive time below

        else:
            _LOGGER.debug("Loading checker with id: %s", checker_id)
            checker, actions_can_be_skipped_numbers = self.load_checker(checker_id)

        sop_steps = read_sop_steps(vlm_output)
        _LOGGER.info("SOP steps: %s", "\n".join(sop_steps))

        response_missing_detected = []
        response_misordered_detected = []
        response_final_missing_detected = []
        response_final_misordered_detected = []
        response_cycle_completed = False
        response_cycle = checker.current_cycle
        response_summary_cycles_detected = []
        response_summary_cycle_analysis = []

        for sop_step in sop_steps:

            sop_action_number = reobj_capture_number.match(sop_step)
            if sop_action_number is None:
                _LOGGER.warning("Cannot capture action number for SOP step: %s", sop_step)
                continue

            sop_action_number = int(sop_action_number.group(1))

            if sop_action_number in actions_can_be_skipped_numbers:
                _LOGGER.info("This action can be skipped: '%s'", sop_step)
                continue

            detection_result = checker.process_number(sop_action_number)

            if not detection_result:
                _LOGGER.error("Detection result is empty for sop_step: %s", sop_step)
                continue

            response_missing_detected.extend(detection_result["missing_detected"])
            response_misordered_detected.extend(detection_result["misordered_detected"])
            response_cycle_completed = detection_result["cycle_completed"]
            response_cycle = detection_result["cycle"]
        if keep_alive and checker_id == "*":
            checker_id = self.save_checker(checker, actions_can_be_skipped_numbers)
        elif not keep_alive:
            detection_result = checker.finalize_processing()
            response_final_missing_detected = detection_result["final_missing_detected"]
            response_final_misordered_detected = detection_result["final_misordered_detected"]
            response_cycle_completed = detection_result["final_cycle_completed"]
            print_summary = checker.print_summary()
            response_summary_cycles_detected = print_summary["cycles_detected"]
            response_summary_cycle_analysis = print_summary["cycle_analysis"]

            self.try_delete_checker(checker_id)

        return SopCheckerResponse(
            request_id=request_id,
            error_message="",
            checker_id=checker_id,
            cycle=response_cycle,
            missing_detected=response_missing_detected,
            misordered_detected=response_misordered_detected,
            final_missing_detected=response_final_missing_detected,
            final_misordered_detected=response_final_misordered_detected,
            cycle_completed=response_cycle_completed,
            summary_cycles_detected=response_summary_cycles_detected,
            summary_cycle_analysis=response_summary_cycle_analysis,
        )
