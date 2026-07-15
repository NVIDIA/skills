# § 6b — SOP Checker: MissingNumberDetector + SopCheckerCache

> **Generates**: `nvds_action_detector/missing_number_detector.py` (via § 6b-G); copies `sop_step_checker_reference.py` → `nvds_action_detector/sop_step_checker.py`
> **Pipeline stage**: Stage 4 — SOP sequence compliance (missing / mis-ordered step detection)

---

## Overview

Two modules implement SOP compliance checking.

| File | How to produce |
|------|----------------|
| `nvds_action_detector/sop_step_checker.py` | Copy verbatim from `sop_step_checker_reference.py` |
| `nvds_action_detector/missing_number_detector.py` | **Generate** using § 6b-G below when a valid `actions.json` is available. **Fallback**: copy `missing_number_detector_reference.py` verbatim when no valid `actions.json` is found or is invalid. |

---

## File 1: `missing_number_detector.py`

### Purpose

Detects **missing** and **mis-ordered** steps in a repeating SOP sequence of the form
`1, 2, 3, …, N, 1, 2, 3, …, N, …`. Processes one action number at a time (real-time streaming).
Handles duplicates (not errors), natural cycle boundary detection, and configurable thresholds.

### Class: `MissingNumberDetector`

```python
class MissingNumberDetector:
    def __init__(
        self,
        N: int,                                    # total required steps per cycle (after skippables removed)
        index_to_action_number: dict[int, int],    # 1-based index → action number (e.g. {1:1, 2:3, 3:5})
        cycle_completion_threshold: float = 0.6,   # fraction of N that must be seen to call a restart
        cycle_boundary_threshold_low: float = 0.3, # low number (≤ N*0.3) after high → new cycle
        cycle_boundary_threshold_high: float = 0.8,# high number (≥ N*0.8) required to trigger low restart
        verbose: bool = False,
    )
```

**Key internal state:**
- `seen_in_cycle: set` — action indices seen in the current cycle
- `expected_next: int` — next index expected in order
- `highest_seen_in_order: int` — highest index seen without mis-order
- `current_cycle: int` — 0-based cycle counter
- `misordered_numbers: list` — all mis-order events across all cycles

**Methods:**

```python
def process_number(self, action_number: int) -> dict:
    """
    Process a single action number. Returns detection result dict:
    {
        "number": int,           # index (1-based)
        "cycle": int,            # current cycle
        "missing_detected": [],  # action numbers missing from just-completed cycle (on cycle boundary)
        "misordered_detected": [],# action numbers detected as mis-ordered
        "cycle_completed": bool, # True when all N steps seen (but cycle continues until boundary)
        "total_processed": int,
        "processing_time_ms": float,
    }
    """

def finalize_processing(self) -> dict:
    """
    Call at end of stream to flush the last incomplete cycle.
    Returns: {"final_missing_detected": [...], "final_misordered_detected": [...], "final_cycle_completed": bool}
    """

def print_summary(self) -> dict:
    """
    Returns {"cycles_detected": [...str], "cycle_analysis": [...str]}
    Used by SopCheckerCache to populate response summary fields.
    """

def get_statistics(self) -> dict: ...
def get_current_cycle_status(self) -> dict: ...
```

### Cycle Boundary Detection Logic

A new cycle is triggered when **either** condition holds:

1. **Duplicate restart**: `number in seen_in_cycle` AND `max(seen_in_cycle) > number`
   AND `len(seen_in_cycle) >= N * cycle_completion_threshold`
2. **Low-after-high restart**: `number <= N * cycle_boundary_threshold_low`
   AND `max(seen_in_cycle) >= N * cycle_boundary_threshold_high`

When triggered, `_complete_cycle()` finalises the previous cycle (detects missing numbers),
then state resets (`seen_in_cycle`, `expected_next`, `highest_seen_in_order`).

### Mis-order Detection

A number is mis-ordered when it is **not** already in `seen_in_cycle`, is less than
`expected_next`, AND is ≤ `highest_seen_in_order` (meaning we have already passed it in order).

### `index_to_action_number` mapping

The detector works on a 1-based dense index internally. `SopCheckerCache` builds this mapping
by filtering out skippable actions from the `actions.json` list:

```python
action_numbers = [1, 3, 5, 7]           # after removing skippables
index_to_action_number = {1:1, 2:3, 3:5, 4:7}   # N=4
```

All public results (missing, misordered) are reported as original action numbers, not indices.

---

## File 2: `sop_step_checker.py`

### Purpose

Wraps `MissingNumberDetector` with:
- Pydantic-free **dataclasses** for request/response types
- A **UUID-keyed cache** enabling stateful multi-chunk streaming sessions
- A **regex step parser** for VLM output text
- Integration glue consumed by `ds_sop_process.py`

### Dataclasses

```python
@dataclass
class SopCheckerRequest:
    action_json: str              # full content of actions.json
    vlm_output: str               # raw VLM output text (e.g. "(1) wash hands (3) put on gloves")
    keep_alive: bool              # True = stateful streaming; False = finalize and return summary
    checker_id: str               # "*" for first request; UUID from prior response for subsequent
    cycle_completion_threshold: float
    cycle_boundary_threshold_low: float
    cycle_boundary_threshold_high: float

@dataclass
class SopCheckerResponse:
    request_id: str
    error_message: str
    checker_id: str               # UUID to use in next request (if keep_alive=True)
    cycle: int
    missing_detected: list[int]   # missing actions in just-completed cycle boundary
    misordered_detected: list[int]
    final_missing_detected: list[int]    # only populated when keep_alive=False
    final_misordered_detected: list[int] # only populated when keep_alive=False
    cycle_completed: bool
    summary_cycles_detected: list[str]   # only populated when keep_alive=False
    summary_cycle_analysis: list[str]    # only populated when keep_alive=False

    def asdict(self): ...         # returns dataclasses.asdict(self)
```

### `read_sop_steps(content: str) -> list[str]`

Parses VLM output into a list of SOP step strings. Expects format `(N) description text`:

```
Input:  "(1) wash hands\n(3) put on gloves\n(5) pick up tool"
Output: ["(1) wash hands", "(3) put on gloves", "(5) pick up tool"]
```

Implemented as a character-scan (not regex split) to handle multi-line step descriptions.

### Class: `SopCheckerCache`

```python
class SopCheckerCache:
    def process_sop_check(self, request_id: str, sop_checker_request: SopCheckerRequest) -> SopCheckerResponse:
        """
        Main entry point called by ds_sop_process.py per chunk.

        Flow:
          checker_id == "*"  → parse action_json, build index_to_action_number (skip skippables),
                               create MissingNumberDetector, save to cache
          checker_id != "*"  → load existing detector from cache

          For each sop_step in read_sop_steps(vlm_output):
            - skip if action number in actions_can_be_skipped
            - call checker.process_number(action_number)
            - accumulate missing_detected, misordered_detected

          keep_alive=True  + first request: save checker again under new UUID (checker_id="*" path)
          keep_alive=False: call finalize_processing() + print_summary(), delete from cache
        """
```

**Cache key lifecycle:**
- First call: `checker_id="*"` → detector created → UUID saved → returned in response
- Subsequent calls: pass returned UUID as `checker_id` → detector loaded and updated
- Final call: `keep_alive=False` → detector finalised → UUID deleted from cache

### Integration with `ds_sop_process.py`

`SOPVideoProcessor` holds a single `SopCheckerCache` instance. Per chunk:

```python
# First chunk of a new video
sop_request = SopCheckerRequest(
    action_json=self.action_json_content,
    vlm_output=vlm_chunk_output,
    keep_alive=True,
    checker_id="*",          # first request
    cycle_completion_threshold=...,
    ...
)
response = self.sop_checker_cache.process_sop_check(request_id, sop_request)
self._checker_id = response.checker_id   # save UUID for next chunk

# Subsequent chunks
sop_request = SopCheckerRequest(..., checker_id=self._checker_id, keep_alive=True)

# Final chunk
sop_request = SopCheckerRequest(..., checker_id=self._checker_id, keep_alive=False)
# response.final_missing_detected, response.summary_cycle_analysis are populated
```

### `actions.json` format expected by `SopCheckerCache`

```json
{
  "actions": [
    "(1) wash hands",
    "(2) dry hands",
    "(3) put on gloves",
    "(4) inspect parts",
    "(5) assemble component"
  ],
  "actions_can_be_skipped": [
    "(2) dry hands"
  ]
}
```

Action numbers are extracted via regex `^\((\d+)\).+`. Skippable actions are excluded from
`index_to_action_number`, so `N` = total actions minus skippables.

---

## § 6b-G — Generating `missing_number_detector.py`

### Invocation

```
# Unit-test path (skill_06b loaded in isolation):
Read skill_06b_sop_checker.md and generate missing_number_detector.py
for <path/to/actions.json>         ← explicit path, or defaults to configs/actions.json

# Real-run path (via skill.md full build):
actions.json is always at configs/actions.json relative to repo root.
No path argument needed — skill.md triggers § 6b-G automatically.
```

### G-Step 1 — Parse actions.json

Path resolution order:
1. Explicit path given in the prompt or argument.
2. `configs/actions.json` relative to repo root (`git rev-parse --show-toplevel`).
3. File not found or invalid JSON → copy `missing_number_detector_reference.py` verbatim, stop.

Parse:
- Extract action numbers with regex `^\((\d+)\)`.
- Build `required_nums` = all action numbers minus skippable numbers.
- Build `index_to_action_number = {i+1: n for i, n in enumerate(required_nums)}`.
- Build `action_number_to_index` (inverse mapping).

### G-Step 2 — Auto-deduce unordered groups from descriptions

Analyse the description text (everything after the action number). **No user input needed.**

**Group actions together when**:
- Same verb + same object class differing only by a positional/numeric suffix
  (e.g. "Unscrewing screw 1 (operator side)" ↔ "Unscrewing screw 2 (operator side)")
- Symmetric location keywords present: left/right, near/far, top/bottom, front/back, A/B, 1/2/3/…
- Same component type repeated in parallel positions (multiple identical cables, screws, boards)

**Do NOT group**:
- Actions with an explicit sequential dependency evident from the description
  (e.g. "Remove cover" must come before "Remove board beneath cover")
- Actions at clearly different assembly phases (setup vs. disassembly)
- The first and last actions of the SOP

Represent groups as **index sets** (1-based internal indices from `index_to_action_number`).
If no groups are deduced, set `UNORDERED_GROUPS = []` — the generated file will be functionally
identical to the reference copy.

### G-Step 3 — Write `nvds_action_detector/missing_number_detector.py`

Output path: `nvds_action_detector/missing_number_detector.py` inside the repo root.
Before writing, verify the path is inside the repo (`git rev-parse --show-toplevel`).

The generated class is **standalone** — it does NOT import from or subclass the reference module.
Copy `missing_number_detector_reference.py` verbatim with **two changes only**:

**Change 1 — File header and class-level constant** (insert before `class MissingNumberDetector:`):

```python
# Auto-generated from <actions.json absolute path> on <date>.
# To regenerate: load skill_06b_sop_checker.md and run § 6b-G.
# Do not edit by hand.
```

And add inside the class body, immediately after the class docstring:

```python
# Unordered group index sets (1-based internal indices).
# Members may appear in any order without triggering mis-order detection.
UNORDERED_GROUPS: list[frozenset[int]] = [
    frozenset({2, 3, 4, 5, 6, 7}),    # Group A — <brief description>
    frozenset({8, 9, 10, 11, 12, 13}), # Group B — <brief description>
]
```

(Fill in the actual index sets from G-Step 2. Use `[]` when no groups were found.)

**Change 2 — Mis-order suppression in `process_number`** (replace the inner `if` block only):

Original in reference:
```python
if number <= self.highest_seen_in_order:
    self.misordered_numbers.append({...})
    result["misordered_detected"].append(self.index_to_action_number[number])
```

Replace with:
```python
if number <= self.highest_seen_in_order:
    # Suppress only when BOTH the incoming number and the action it is out-of-order
    # against (highest_seen_in_order) belong to the same UNORDERED_GROUP.
    # Cross-group boundary violations (e.g. group member after a non-group action)
    # remain flagged.
    _same_group = any(
        number in g and self.highest_seen_in_order in g
        for g in self.UNORDERED_GROUPS
    )
    if not _same_group:
        self.misordered_numbers.append({...})
        result["misordered_detected"].append(self.index_to_action_number[number])
```

All other methods (`__init__`, `_complete_cycle`, `finalize_processing`, `print_summary`,
`get_statistics`, `get_current_cycle_status`) are copied verbatim from the reference.

### G-Step 4 — Write `tests/test_missing_number_detector.py`

Output path: `tests/test_missing_number_detector.py` inside the repo root.

All test sequences must use **concrete numbers** derived from the parsed `index_to_action_number`
and deduced groups — no `<placeholder>` values in the final file.

Required scenarios:
1. **Perfect canonical order** — feed action numbers in SOP order; assert no missing, no mis-order.
2. **Shuffled within each unordered group** — at least 2 orderings per group; assert
   `misordered_detected == []` for all results.
3. **Missing one required action** — omit one action number; call `finalize_processing()`; assert
   that action number appears in `final_missing_detected`.
4. **Cross-group boundary mis-order** — a group member fed after a later-phase non-group action
   must be flagged as mis-ordered.
5. **No groups fallback** — monkey-patch `det.UNORDERED_GROUPS = []`; feed a shuffled sequence
   that crosses group boundaries; assert mis-order IS flagged.

Test file header:
```python
# Tests for the generated MissingNumberDetector.
# Run: python3 -m pytest tests/test_missing_number_detector.py -v
import pytest
from nvds_action_detector.missing_number_detector import MissingNumberDetector
```

### G-Step 5 — Confirm

After writing both files, print:

```
Files written
─────────────
  <abs_path>/nvds_action_detector/missing_number_detector.py
  <abs_path>/tests/test_missing_number_detector.py

The original actions.json was NOT modified.

Verify:
  python3 -m pytest tests/test_missing_number_detector.py -v
  python3 -c "from nvds_action_detector.sop_step_checker import SopCheckerCache; print('ok')"
```

---

## Critical Notes

- **`ds_logger.get_logger(__name__)`** — both files use this; ensure `ds_logger.py` exports
  `get_logger` (LOGGER_EXPORT_GET_LOGGER).
- **`index_to_action_number` is 1-based dense** — the index starts at 1, not 0, and has no
  gaps after skippables are removed. This is what `MissingNumberDetector(N, ...)` expects.
- **`checker_id="*"` creates AND saves** — on the `"*"` path, `save_checker()` is called
  once during init, then again at the end if `keep_alive=True`. The first save is used to
  pass `actions_can_be_skipped_numbers` alongside the detector. Subsequent calls load this pair.
- **`keep_alive=False` deletes by the original `checker_id`** — pass the UUID received from
  the prior response, not `"*"`.
- **`UNORDERED_GROUPS` suppresses mis-order detection only within the same group** —
  a mis-order is suppressed iff both `number` and `highest_seen_in_order` belong to the
  *same* group. Cross-group violations (group member arriving after a later-phase
  non-group action, or after a member of a different group) are still flagged. Cycle
  boundary detection and missing-step detection are unaffected. Actions in a group can
  still be flagged as missing.
- **`SopCheckerCache` requires zero changes** — the generated `missing_number_detector.py` is a
  drop-in replacement with the same public interface as the reference.
- **`UNORDERED_GROUPS = []`** — when no groups are deduced, the generated file is behaviourally
  identical to the reference copy.
