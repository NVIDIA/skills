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
Missing Number Detection Algorithm - Version 11
With Missing Numbers and Mis-order Detection
"""

import logging
import time
from collections import defaultdict
from typing import List, Optional, Set, Tuple

from . import ds_logger

_LOGGER = ds_logger.get_logger(__name__)
# _LOGGER = logging.getLogger(__name__)


class MissingNumberDetector:
    """
    Detects missing numbers and mis-ordered numbers in a sequence that should follow pattern 1,2,3,...,N,1,2,3,...,N

    Features:
    - Real-time processing (one number at a time)
    - Handles duplicate numbers (not considered errors)
    - Detects missing numbers
    - Detects mis-ordered numbers
    - Natural cycle boundary detection
    """

    def __init__(
        self,
        N: int,
        index_to_action_number: dict[int, int],
        cycle_completion_threshold: float = 0.6,
        cycle_boundary_threshold_low: float = 0.3,
        cycle_boundary_threshold_high: float = 0.8,
        verbose: bool = False,
    ):
        """
        Initialize the detector

        Args:
            N: Maximum number in the sequence (1 to N)
            index_to_action_number: Mapping from index to action number
            verbose: Enable detailed logging
        """
        self.N = N
        self.index_to_action_number = index_to_action_number
        self.action_number_to_index = {v: k for k, v in index_to_action_number.items()}
        self.cycle_completion_threshold = cycle_completion_threshold
        self.cycle_boundary_threshold_low = cycle_boundary_threshold_low
        self.cycle_boundary_threshold_high = cycle_boundary_threshold_high
        self.verbose = verbose

        # State tracking
        self.current_cycle = 0
        self.seen_in_cycle = set()
        self.total_processed = 0
        self.expected_next = 1  # Track what number we expect next
        self.highest_seen_in_order = 0  # Track the highest number seen in proper order
        self.received_action_sequence = []

        # Results tracking
        self.missing_numbers = []
        self.cycle_completions = []
        self.cycle_details = []  # Track actual numbers seen in each cycle
        self.misordered_numbers = []  # Track mis-ordered numbers

        # Performance tracking
        self.processing_times = []

        if self.verbose:
            _LOGGER.info("Initialized detector: N=%d", N)

    def process_number(self, action_number: int) -> dict:
        """
        Process a single number from the sequence

        Args:
            action_number: The input action number

        Returns:
            dict: Detection results for this input
        """

        start_time = time.time()
        self.received_action_sequence.append(action_number)
        try:
            number = self.action_number_to_index[action_number]
        except KeyError:
            _LOGGER.error(
                "Action number %d not found in index_to_action_number. "
                "Maybe VLM output and action.json doesn't match. "
                "Please check if you are using the correct VLM weight and action.json.",
                action_number,
            )
            return {}

        result = {
            "number": number,
            "cycle": self.current_cycle,
            "missing_detected": [],
            "misordered_detected": [],
            "cycle_completed": False,
            "total_processed": self.total_processed + 1,
        }

        self.total_processed += 1

        # Validate input
        if number < 1 or number > self.N:
            result["error"] = f"Invalid number {number}. Expected range: 1-{self.N}"
            return result

        # Check for mis-ordering
        if number in self.seen_in_cycle:
            # Duplicate, not mis-ordered
            pass
        elif number < self.expected_next and number not in self.seen_in_cycle:
            # This number is out of order (appears too early or we've passed it)
            if number <= self.highest_seen_in_order:
                # This is truly mis-ordered
                self.misordered_numbers.append(
                    {
                        "number": number,
                        "cycle": self.current_cycle,
                        "position": self.total_processed,
                        "expected_after": self.highest_seen_in_order,
                    }
                )
                result["misordered_detected"].append(self.index_to_action_number[number])

        # Detect cycle transition
        # New cycle when we see a duplicate number that would indicate restart
        if len(self.seen_in_cycle) > 0:
            max_seen = max(self.seen_in_cycle)

            # Check if this looks like a new cycle starting
            is_new_cycle = False

            # Case 1: We've already seen this number and we've seen higher numbers
            if number in self.seen_in_cycle and max_seen > number:
                # Check if we've seen enough of the cycle to consider this a restart
                if len(self.seen_in_cycle) >= self.N * self.cycle_completion_threshold:
                    is_new_cycle = True

            # Case 2: Low number after seeing high numbers (potential cycle restart)
            elif (
                number <= self.N * self.cycle_boundary_threshold_low
                and max_seen >= self.N * self.cycle_boundary_threshold_high
            ):
                is_new_cycle = True

            if is_new_cycle:
                # Complete the previous cycle first
                missing_in_previous, misordered_in_previous = self._complete_cycle()
                result["missing_detected"] = [self.index_to_action_number[missing] for missing in missing_in_previous]
                result["misordered_detected"] = [
                    self.index_to_action_number[misordered] for misordered in misordered_in_previous
                ]
                if self.verbose:
                    _LOGGER.debug(
                        "Cycle %d completed: missing %s, misordered %s",
                        self.current_cycle - 1,
                        missing_in_previous,
                        misordered_in_previous,
                    )

                # Start new cycle
                self.seen_in_cycle = set()
                self.expected_next = 1
                self.highest_seen_in_order = 0
                result["cycle"] = self.current_cycle

        # Add to current cycle (duplicates are ignored - only add if not already seen)
        self.seen_in_cycle.add(number)

        # Update expected next number
        if number >= self.expected_next:
            self.expected_next = number + 1
            if number > self.highest_seen_in_order:
                self.highest_seen_in_order = number

        # Check if cycle is complete (all numbers 1 to N seen) but don't end it yet
        if len(self.seen_in_cycle) == self.N and not result.get("missing_detected"):
            result["cycle_completed"] = True
            if self.verbose:
                _LOGGER.debug("Cycle %d is complete but continues until natural boundary", self.current_cycle)

        # Record processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        result["processing_time_ms"] = processing_time * 1000

        return result

    def _complete_cycle(self) -> Tuple[List[int], List[int]]:
        """
        Complete the current cycle and detect any missing numbers and collect mis-ordered numbers

        Returns:
            Tuple of (missing numbers, mis-ordered numbers) in the completed cycle
        """
        missing_in_cycle = []

        if len(self.seen_in_cycle) < self.N:
            for i in range(1, self.N + 1):
                if i not in self.seen_in_cycle:
                    missing_in_cycle.append(int(i))

            # Record missing numbers
            self.missing_numbers.extend(
                [
                    {
                        "number": int(missing),
                        "cycle": int(self.current_cycle),
                        "detected_at_position": int(self.total_processed),
                    }
                    for missing in missing_in_cycle
                ]
            )

        # Collect mis-ordered numbers for this cycle
        misordered_in_cycle = [m["number"] for m in self.misordered_numbers if m["cycle"] == self.current_cycle]

        # Record cycle completion
        self.cycle_completions.append(
            {
                "cycle": int(self.current_cycle),
                "completed_at_position": int(self.total_processed),
                "numbers_seen": int(len(self.seen_in_cycle)),
                "missing_count": int(len(missing_in_cycle)),
                "misordered_count": int(len(misordered_in_cycle)),
            }
        )

        # Record cycle details (actual numbers seen, missing, and misordered)
        self.cycle_details.append(
            {
                "cycle": int(self.current_cycle),
                "seen_numbers": [int(x) for x in sorted(list(self.seen_in_cycle))],
                "missing_numbers": [int(x) for x in missing_in_cycle],
                "misordered_numbers": [int(x) for x in misordered_in_cycle],
            }
        )

        if self.verbose:
            _LOGGER.debug(
                "Cycle %d completed. Seen: %s, Missing: %s, Misordered: %s",
                self.current_cycle,
                sorted(self.seen_in_cycle),
                missing_in_cycle,
                misordered_in_cycle,
            )

        # Reset for next cycle
        self.seen_in_cycle.clear()
        self.current_cycle += 1

        return missing_in_cycle, misordered_in_cycle

    def get_statistics(self) -> dict:
        """Get comprehensive statistics about the detection process"""
        # Check for missing numbers in current incomplete cycle
        current_missing = []
        if len(self.seen_in_cycle) > 0 and len(self.seen_in_cycle) < self.N:
            for i in range(1, self.N + 1):
                if i not in self.seen_in_cycle:
                    current_missing.append(i)

        # Get mis-ordered numbers in current cycle
        current_misordered = [m["number"] for m in self.misordered_numbers if m["cycle"] == self.current_cycle]

        return {
            "total_processed": self.total_processed,
            "current_cycle": self.current_cycle,
            "total_missing": len(self.missing_numbers),
            "total_misordered": len(self.misordered_numbers),
            "cycles_completed": len(self.cycle_completions),
            "avg_processing_time_ms": (
                (sum(self.processing_times) / len(self.processing_times) * 1000) if self.processing_times else 0
            ),
            "max_processing_time_ms": (max(self.processing_times) * 1000) if self.processing_times else 0,
            "current_cycle_progress": f"{len(self.seen_in_cycle)}/{self.N}",
            "current_cycle_missing": current_missing,
            "current_cycle_misordered": current_misordered,
            "seen_in_current_cycle": sorted(list(self.seen_in_cycle)),
        }

    def get_current_cycle_status(self) -> dict:
        """Get status of the current incomplete cycle"""
        missing = []
        if len(self.seen_in_cycle) < self.N:
            for i in range(1, self.N + 1):
                if i not in self.seen_in_cycle:
                    missing.append(i)

        misordered = [m["number"] for m in self.misordered_numbers if m["cycle"] == self.current_cycle]

        return {
            "cycle": self.current_cycle,
            "seen": sorted(list(self.seen_in_cycle)),
            "missing": missing,
            "misordered": misordered,
            "progress": f"{len(self.seen_in_cycle)}/{self.N}",
            "complete": len(self.seen_in_cycle) == self.N,
        }

    def finalize_processing(self) -> dict:
        """Complete any remaining cycles"""
        result = {"final_missing_detected": [], "final_misordered_detected": [], "final_cycle_completed": False}

        # Complete current cycle if it has any data
        if len(self.seen_in_cycle) > 0:
            missing_in_final, misordered_in_final = self._complete_cycle()
            result["final_missing_detected"] = [self.index_to_action_number[int(x)] for x in missing_in_final]
            result["final_misordered_detected"] = [self.index_to_action_number[int(x)] for x in misordered_in_final]
            result["final_cycle_completed"] = len(missing_in_final) == 0

        return result

    def print_summary(self) -> dict:
        """Print a summary of the sequence processing"""
        sequence = self.received_action_sequence
        _LOGGER.info("\n--- Complete Summary ---")
        _LOGGER.info("Sequence length: %d numbers", len(sequence))

        _cycles_key = "cycles_detected"
        _cycle_analysis_key = "cycle_analysis"
        summary = {
            _cycles_key: [],
            _cycle_analysis_key: [],
        }

        # Process the sequence again to identify actual cycle boundaries
        temp_detector = MissingNumberDetector(
            N=self.N,
            index_to_action_number=self.index_to_action_number,
            cycle_completion_threshold=self.cycle_completion_threshold,
            cycle_boundary_threshold_low=self.cycle_boundary_threshold_low,
            cycle_boundary_threshold_high=self.cycle_boundary_threshold_high,
            verbose=False,
        )
        cycle_boundaries = [0]

        for i, num in enumerate(sequence):
            result = temp_detector.process_number(num)
            if not result:
                summary[_cycle_analysis_key].append(
                    f"Error: {num} not a valid action, please check if VLM output and action.json match."
                )
                continue
            if result["missing_detected"] is not None and len(result["missing_detected"]) >= 0:
                if result["cycle"] > len(cycle_boundaries) - 1:
                    cycle_boundaries.append(i)

        # Add the end of sequence as final boundary
        cycle_boundaries.append(len(sequence))

        # Print the sequence broken down by cycles
        _LOGGER.info("Actual input sequence by cycles (original order with duplicates):")
        for i in range(len(cycle_boundaries) - 1):
            start = cycle_boundaries[i]
            end = cycle_boundaries[i + 1]
            cycle_seq = sequence[start:end]
            _LOGGER.info("  Cycle %d: %s", i, cycle_seq)
            summary[_cycles_key].append(f"Cycle {i}: {cycle_seq}")

        _LOGGER.info("Cycle analysis (unique numbers sorted, duplicates removed):")
        # Print completed cycles
        for cycle_detail in self.cycle_details:
            cycle_num = int(cycle_detail["cycle"])
            seen_nums = [int(x) for x in cycle_detail["seen_numbers"]]
            missing_nums = [int(x) for x in cycle_detail["missing_numbers"]]
            misordered_nums = [int(x) for x in cycle_detail["misordered_numbers"]]

            seen_nums_mapped = [self.index_to_action_number[int(x)] for x in seen_nums]
            missing_nums_mapped = [self.index_to_action_number[int(x)] for x in missing_nums]
            misordered_nums_mapped = [self.index_to_action_number[int(x)] for x in misordered_nums]

            output_parts = []
            if missing_nums:
                output_parts.append(f"missing {missing_nums_mapped}")
            if misordered_nums:
                output_parts.append(f"misordered {misordered_nums_mapped}")

            if output_parts:
                _LOGGER.info("  Cycle %d: %s -> %s", cycle_num, seen_nums_mapped, ", ".join(output_parts))
                summary[_cycle_analysis_key].append(
                    f"Cycle {cycle_num}: {seen_nums_mapped} -> {', '.join(output_parts)}"
                )
            else:
                _LOGGER.info("  Cycle %d: %s -> no issues", cycle_num, seen_nums_mapped)
                summary[_cycle_analysis_key].append(f"Cycle {cycle_num}: {seen_nums_mapped} -> no issues")

        # Handle current incomplete cycle (if any)
        if len(self.seen_in_cycle) > 0:
            current_missing = []
            for j in range(1, self.N + 1):
                if j not in self.seen_in_cycle:
                    current_missing.append(int(j))
            current_seen = [int(x) for x in sorted(list(self.seen_in_cycle))]
            current_misordered = [m["number"] for m in self.misordered_numbers if m["cycle"] == self.current_cycle]

            current_seen_mapped = [self.index_to_action_number[int(x)] for x in current_seen]
            current_missing_mapped = [self.index_to_action_number[int(x)] for x in current_missing]
            current_misordered_mapped = [self.index_to_action_number[int(x)] for x in current_misordered]

            output_parts = []
            if current_missing:
                output_parts.append(f"missing {current_missing_mapped}")
            if current_misordered:
                output_parts.append(f"misordered {current_misordered_mapped}")

            if output_parts:
                _LOGGER.info(
                    "  Cycle %d: %s -> %s (incomplete)",
                    int(self.current_cycle),
                    current_seen_mapped,
                    ", ".join(output_parts),
                )
                summary[_cycle_analysis_key].append(
                    f"Cycle {self.current_cycle}: {current_seen_mapped} -> {', '.join(output_parts)} (incomplete)"
                )
            else:
                _LOGGER.info("  Cycle %d: %s -> no issues (incomplete)", int(self.current_cycle), current_seen_mapped)
                summary[_cycle_analysis_key].append(
                    f"Cycle {self.current_cycle}: {current_seen_mapped} -> no issues (incomplete)"
                )

        return summary

    def visualize_results(self):
        """Create visualizations of the detection results - REMOVED"""
        _LOGGER.info("Visualization feature removed for simplicity.")
        return


# Helper function to generate sequences
def generate_clean_sequence(N, num_cycles, seed, missing_strategy="random", duplicate_strategy="light"):
    """
    Helper function to generate clean sequences with regular Python integers

    Args:
        N: Range 1 to N
        num_cycles: Number of cycles to generate
        seed: Random seed for reproducibility
        missing_strategy: 'random', 'light', 'heavy', 'systematic'
        duplicate_strategy: 'light', 'medium', 'heavy'

    Returns:
        List of regular Python integers forming a sequence
    """
    import random

    random.seed(seed)

    sequence = []

    for cycle in range(num_cycles):
        # Start with complete cycle in proper order
        cycle_numbers = list(range(1, N + 1))  # [1, 2, 3, ..., N]

        # Apply missing strategy
        if missing_strategy == "light":
            remove_count = random.choice([0, 1, 2])
        elif missing_strategy == "heavy":
            remove_count = min(random.choice([3, 4, 5, 6]), N - 1)  # Keep at least one number
        elif missing_strategy == "systematic":
            if cycle % 2 == 0:
                # Remove some even numbers
                evens = [x for x in cycle_numbers if x % 2 == 0]
                remove_count = min(random.choice([1, 2, 3]), len(evens))
                for _ in range(remove_count):
                    if evens and len(cycle_numbers) > 1:
                        num_to_remove = random.choice(evens)
                        cycle_numbers.remove(num_to_remove)
                        evens.remove(num_to_remove)
            else:
                # Remove some odd numbers
                odds = [x for x in cycle_numbers if x % 2 == 1]
                remove_count = min(random.choice([1, 2, 3]), len(odds))
                for _ in range(remove_count):
                    if odds and len(cycle_numbers) > 1:
                        num_to_remove = random.choice(odds)
                        cycle_numbers.remove(num_to_remove)
                        odds.remove(num_to_remove)
            remove_count = 0  # Already handled above
        else:  # random
            remove_count = min(random.choice([1, 2, 3, 4]), N - 1)

        # Remove numbers randomly
        for _ in range(remove_count):
            if len(cycle_numbers) > 1:  # Keep at least one number
                cycle_numbers.remove(random.choice(cycle_numbers))

        # Now add duplicates while maintaining order
        final_cycle = []

        for num in cycle_numbers:
            # Add the number itself
            final_cycle.append(num)

            # Determine if we should duplicate this number
            if duplicate_strategy == "light":
                dup_chance = 0.2
                max_dups = 2
            elif duplicate_strategy == "medium":
                dup_chance = 0.4
                max_dups = 4
            elif duplicate_strategy == "heavy":
                dup_chance = 0.6
                max_dups = 6
            else:
                dup_chance = 0.3
                max_dups = 3

            # Add duplicates for this number
            if random.random() < dup_chance:
                dup_count = random.randint(1, max_dups)
                final_cycle.extend([num] * dup_count)

        sequence.extend(final_cycle)

    return sequence
