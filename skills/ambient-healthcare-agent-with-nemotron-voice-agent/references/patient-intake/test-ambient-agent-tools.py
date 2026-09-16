# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ambient patient-intake confirmation."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from examples.generic import ambient_agent_tools, patient_intake_speech_guard, tool_handlers
from examples.shared.pipeline_utils import build_user_aggregator_params


def test_saved_intake_confirmation_is_speech_friendly(tmp_path):
    """The saved result should be brief and avoid repeating intake fields."""
    records_path = tmp_path / "patient_intake_records.jsonl"
    with patch.object(ambient_agent_tools, "RECORDS_PATH", records_path):
        result = ambient_agent_tools.record_patient_intake_result(
            {
                "patient_name": "Jane Johnson",
                "date_of_birth": "1975-01-06",
                "symptoms": "flu-like illness",
                "current_medications": "oseltamivir",
                "current_pharmacy": "Walgreens pharmacy in San Jose",
                "patient_confirmed": True,
            }
        )

    assert result["status"] == "saved"
    assert result["record_id"].startswith("INTAKE-")
    assert result["response_text"] == (
        f"Thank you, your information has been saved. Your confirmation number is {result['record_id']}."
    )
    assert "_" not in result["response_text"]
    assert "=" not in result["response_text"]
    assert records_path.is_file()


def test_date_formatter_uses_english_ordinals():
    """Numeric dates should use the correct English ordinal suffix."""
    expected_dates = {
        "1994-01-01": "January 1st 1994",
        "1994-01-02": "January 2nd 1994",
        "1994-01-03": "January 3rd 1994",
        "1994-01-04": "January 4th 1994",
        "1994-01-11": "January 11th 1994",
        "1994-01-12": "January 12th 1994",
        "1994-01-13": "January 13th 1994",
        "1994-01-21": "January 21st 1994",
        "1994-01-22": "January 22nd 1994",
        "1994-01-23": "January 23rd 1994",
        "1994-01-31": "January 31st 1994",
    }

    for raw_date, spoken_date in expected_dates.items():
        assert ambient_agent_tools._format_date_for_speech(raw_date) == spoken_date

    assert ambient_agent_tools._format_date_for_speech("October 1st, 1966") == "October 1st 1966"


def test_saved_confirmation_bypasses_a_second_llm_response(tmp_path):
    """A saved intake should emit its deterministic confirmation directly."""

    async def run_handler():
        """Invoke the generated Pipecat handler with test doubles."""
        params = SimpleNamespace(
            arguments={
                "patient_name": "Jane Johnson",
                "date_of_birth": "1975-01-06",
                "symptoms": "flu-like illness",
                "current_medications": "oseltamivir",
                "current_pharmacy": "Walgreens pharmacy in San Jose",
                "patient_confirmed": True,
            },
            llm=SimpleNamespace(push_frame=AsyncMock()),
            result_callback=AsyncMock(),
        )
        handler = tool_handlers._build_ambient_agent_tool_handler("record_patient_intake")
        with patch.object(ambient_agent_tools, "RECORDS_PATH", tmp_path / "patient_intake_records.jsonl"):
            await handler(params)
        return params

    params = asyncio.run(run_handler())
    frames = [call.args[0] for call in params.llm.push_frame.await_args_list]
    assert isinstance(frames[0], LLMFullResponseStartFrame)
    assert isinstance(frames[1], LLMTextFrame)
    assert frames[1].text.startswith("Thank you, your information has been saved.")
    assert "January" not in frames[1].text
    assert "_" not in frames[1].text
    assert "=" not in frames[1].text
    assert isinstance(frames[2], LLMFullResponseEndFrame)
    assert params.result_callback.await_args.kwargs["properties"].run_llm is False


def test_review_is_natural_and_does_not_persist(tmp_path):
    """The review should narrate fields, ask for confirmation, and not save."""
    records_path = tmp_path / "patient_intake_records.jsonl"
    with patch.object(ambient_agent_tools, "RECORDS_PATH", records_path):
        cold_review = ambient_agent_tools.review_patient_intake_result(
            {
                "patient_name": "Austin",
                "date_of_birth": "1994-01-06",
                "symptoms": "cold",
                "current_medications": "loratadine daily",
                "current_pharmacy": "Walgreens Pharmacy in New York",
            }
        )
        shot_review = ambient_agent_tools.review_patient_intake_result(
            {
                "patient_name": "Austin",
                "date_of_birth": "1994-01-06",
                "symptoms": "flu shot",
                "current_medications": "loratadine daily",
                "current_pharmacy": "Walgreens Pharmacy in San Jose",
            }
        )

    assert cold_review["status"] == "review_required"
    assert cold_review["response_text"] == (
        "Thank you. I have all the information I need. Your name is Austin, "
        "your date of birth is January 6th 1994, you're currently experiencing a cold, "
        "your current medications are loratadine daily, and your current pharmacy is "
        "Walgreens Pharmacy in New York. Is this all correct?"
    )
    assert "your reason for the visit is a flu shot" in shot_review["response_text"]
    assert "_" not in cold_review["response_text"]
    assert "=" not in cold_review["response_text"]
    assert ":" not in cold_review["response_text"]
    assert not records_path.exists()


def test_review_requires_current_medications(tmp_path):
    """The review must not proceed when the medications field is missing."""
    records_path = tmp_path / "patient_intake_records.jsonl"
    with patch.object(ambient_agent_tools, "RECORDS_PATH", records_path):
        result = ambient_agent_tools.review_patient_intake_result(
            {
                "patient_name": "Austin",
                "date_of_birth": "1994-01-06",
                "symptoms": "cold",
                "current_pharmacy": "Walgreens Pharmacy in New York",
            }
        )

    assert result["status"] == "needs_more_information"
    assert result["missing_fields"] == ["current_medications"]
    assert result["response_text"] == "Please share your current medications."
    assert not records_path.exists()


def test_markdown_labeled_review_is_rewritten_as_natural_speech():
    """The exact failed review format should be replaced before it reaches TTS."""
    raw_response = (
        "Yes, Walgreens in Santa Clara. Let me review all the information I've collected. "
        "**Review:** Name: Jane Johnson Date of birth: October 1st, 1964 "
        "Symptoms or visit reason: Need the second dose of my vaccination "
        "Current medications: metformin twice daily "
        "Current pharmacy: Walgreens in Santa Clara Is this all correct?"
    )

    assert patient_intake_speech_guard.sanitize_patient_intake_speech(raw_response) == (
        "Thank you. I have all the information I need. Your name is Jane Johnson, "
        "your date of birth is October 1st 1964, your reason for the visit is to receive "
        "the second dose of your vaccination, your current medications are metformin twice daily, "
        "and your current pharmacy is Walgreens in Santa Clara. "
        "Is this all correct?"
    )


def test_speech_guard_removes_a_comma_from_any_spoken_date():
    """Model-authored dates should also follow the comma-free voice contract."""
    assert patient_intake_speech_guard.sanitize_patient_intake_speech(
        "Your date of birth is October 1st, 1966."
    ) == "Your date of birth is October 1st 1966."


def test_patient_intake_welcome_is_deterministic_and_exact():
    """The opening turn should bypass model paraphrasing."""
    frames = patient_intake_speech_guard.patient_intake_welcome_frames()

    assert isinstance(frames[0], LLMFullResponseStartFrame)
    assert isinstance(frames[1], LLMTextFrame)
    assert frames[1].text == (
        "Hello and welcome to the patient intake agent. I'm here to help you get checked in for "
        "your appointment and will be asking you a few questions in order to get ready for your "
        "visit with the doctor. First, could you please tell me your full name?"
    )
    assert isinstance(frames[2], LLMFullResponseEndFrame)


def test_partial_summary_after_name_and_date_of_birth_is_replaced_with_symptoms_question():
    """A model-authored recap must never reach TTS during collection."""

    async def run_processor():
        state = patient_intake_speech_guard.PatientIntakeConversationState()
        reminder = patient_intake_speech_guard.PatientIntakeTurnReminderProcessor(state)
        reminder._reminded_frame(
            LLMContextFrame(
                context=LLMContext(
                    [
                        {"role": "assistant", "content": "Could you please tell me your full name?"},
                        {"role": "user", "content": "My name is Maya Chen."},
                        {"role": "assistant", "content": "What is your date of birth?"},
                        {"role": "user", "content": "I was born on 1988-04-12."},
                    ]
                )
            )
        )
        processor = patient_intake_speech_guard.PatientIntakeSpeechGuardProcessor(state)
        with patch.object(processor, "push_frame", new=AsyncMock()) as push_frame:
            await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            await processor.process_frame(
                LLMTextFrame(
                    text=(
                        "Thank you, Maya Chen. I have your date of birth as April 12th 1988. "
                        "What symptoms are you experiencing?"
                    )
                ),
                FrameDirection.DOWNSTREAM,
            )
            await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        return push_frame

    push_frame = asyncio.run(run_processor())
    spoken = [
        call.args[0].text
        for call in push_frame.await_args_list
        if isinstance(call.args[0], LLMTextFrame)
    ]
    assert spoken == ["What symptoms are you experiencing, or what is the reason for your visit?"]
    assert "Maya Chen" not in spoken[0]
    assert "1988" not in spoken[0]


def test_hallucinated_early_save_tool_call_is_blocked_without_persisting(tmp_path):
    """Arguments invented by the model cannot bypass conversation-state completeness."""

    async def run_handler():
        state = patient_intake_speech_guard.PatientIntakeConversationState()
        state.update_from_messages(
            [
                {"role": "assistant", "content": "Could you please tell me your full name?"},
                {"role": "user", "content": "My name is Maya Chen."},
                {"role": "assistant", "content": "What is your date of birth?"},
                {"role": "user", "content": "I was born on 1988-04-12."},
            ]
        )
        base_handler = tool_handlers._build_ambient_agent_tool_handler("record_patient_intake")
        handler = tool_handlers.build_patient_intake_guarded_handler(
            "record_patient_intake",
            base_handler,
            state,
        )
        params = SimpleNamespace(
            arguments={
                "patient_name": "Maya Chen",
                "date_of_birth": "1988-04-12",
                "symptoms": "invented symptom",
                "current_medications": "invented medication",
                "current_pharmacy": "invented pharmacy",
                "patient_confirmed": True,
            },
            llm=SimpleNamespace(push_frame=AsyncMock()),
            result_callback=AsyncMock(),
        )
        with patch.object(ambient_agent_tools, "RECORDS_PATH", tmp_path / "patient_intake_records.jsonl"):
            await handler(params)
        return params

    params = asyncio.run(run_handler())
    result = params.result_callback.await_args.args[0]
    assert result["status"] == "needs_more_information"
    assert result["missing_fields"] == ["symptoms", "current_medications", "current_pharmacy"]
    assert result["response_text"] == (
        "What symptoms are you experiencing, or what is the reason for your visit?"
    )
    assert not (tmp_path / "patient_intake_records.jsonl").exists()


def test_welcome_is_not_rewritten_by_collection_guard():
    """The deterministic opening remains intact before collection begins."""

    async def run_processor():
        processor = patient_intake_speech_guard.PatientIntakeSpeechGuardProcessor()
        with patch.object(processor, "push_frame", new=AsyncMock()) as push_frame:
            for frame in patient_intake_speech_guard.patient_intake_welcome_frames():
                await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
        return push_frame

    push_frame = asyncio.run(run_processor())
    spoken = [
        call.args[0].text
        for call in push_frame.await_args_list
        if isinstance(call.args[0], LLMTextFrame)
    ]
    assert spoken == [patient_intake_speech_guard.PATIENT_INTAKE_WELCOME_TEXT]


def test_repeated_welcome_is_replaced_with_next_field_question():
    """The opening must not restart after the patient has begun intake."""

    async def run_processor():
        state = patient_intake_speech_guard.PatientIntakeConversationState()
        processor = patient_intake_speech_guard.PatientIntakeSpeechGuardProcessor(state)
        with patch.object(processor, "push_frame", new=AsyncMock()) as push_frame:
            for frame in patient_intake_speech_guard.patient_intake_welcome_frames():
                await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

            state.update_from_messages(
                [
                    {
                        "role": "assistant",
                        "content": patient_intake_speech_guard.PATIENT_INTAKE_WELCOME_TEXT,
                    },
                    {"role": "user", "content": "My name is Maya Chen."},
                ]
            )
            for frame in patient_intake_speech_guard.patient_intake_welcome_frames():
                await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
        return push_frame

    push_frame = asyncio.run(run_processor())
    spoken = [
        call.args[0].text
        for call in push_frame.await_args_list
        if isinstance(call.args[0], LLMTextFrame)
    ]
    assert spoken == [
        patient_intake_speech_guard.PATIENT_INTAKE_WELCOME_TEXT,
        "What is your date of birth?",
    ]


def test_empty_tool_transition_is_silent_and_tool_review_is_spoken_once():
    """Empty LLM completion frames must not invent duplicate retry speech."""

    async def run_processor():
        state = patient_intake_speech_guard.PatientIntakeConversationState()
        state.update_from_messages(
            [
                {
                    "role": "user",
                    "content": (
                        "My name is John Johnson. I was born on 1989-01-07. I have a cold. "
                        "I am taking nothing. My pharmacy is CVS in Mountain View."
                    ),
                }
            ]
        )
        review = ambient_agent_tools.review_patient_intake_result(
            {
                "patient_name": "John Johnson",
                "date_of_birth": "1989-01-07",
                "symptoms": "a cold",
                "current_medications": "nothing",
                "current_pharmacy": "CVS in Mountain View",
            }
        )["response_text"]
        processor = patient_intake_speech_guard.PatientIntakeSpeechGuardProcessor(state)
        with patch.object(processor, "push_frame", new=AsyncMock()) as push_frame:
            await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
            await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            await processor.process_frame(
                patient_intake_speech_guard.PatientIntakeToolResponseFrame(text=review),
                FrameDirection.DOWNSTREAM,
            )
            await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        return push_frame, review

    push_frame, review = asyncio.run(run_processor())
    spoken = [
        call.args[0].text
        for call in push_frame.await_args_list
        if isinstance(call.args[0], LLMTextFrame)
    ]
    assert spoken == [review]
    assert "Please repeat your last response" not in spoken[0]


def test_patient_intake_welcome_can_be_interrupted():
    """Patient speech must remain unmuted while the opening welcome is playing."""
    with (
        patch.dict(os.environ, {"USE_SILERO_VAD_TURN_DETECTION": "true"}),
        patch("examples.shared.pipeline_utils.SileroVADAnalyzer", return_value=SimpleNamespace()),
    ):
        params = build_user_aggregator_params(
            welcome_enabled=True,
            interruptible_welcome=True,
        )

    assert params.user_mute_strategies == []


def test_speech_guard_forwards_interruption_and_discards_buffered_text():
    """Barge-in must cancel buffered bot speech and reach downstream processors."""

    async def run_processor():
        processor = patient_intake_speech_guard.PatientIntakeSpeechGuardProcessor()
        interruption = InterruptionFrame()
        with (
            patch.object(processor, "push_frame", new=AsyncMock()) as push_frame,
            patch.object(processor, "_start_interruption", new=AsyncMock()),
        ):
            await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            await processor.process_frame(
                LLMTextFrame(text="This response should be interrupted."),
                FrameDirection.DOWNSTREAM,
            )
            await processor.process_frame(interruption, FrameDirection.DOWNSTREAM)
        return push_frame, interruption

    push_frame, interruption = asyncio.run(run_processor())
    forwarded = [call.args[0] for call in push_frame.await_args_list]
    assert interruption in forwarded
    assert not any(isinstance(frame, LLMTextFrame) for frame in forwarded)


def test_speech_guard_discards_model_preamble_before_tool_response():
    """Only the deterministic tool-authored review should reach TTS and history."""

    async def run_processor():
        state = patient_intake_speech_guard.PatientIntakeConversationState()
        state.update_from_messages(
            [
                {
                    "role": "user",
                    "content": (
                        "My name is Jane Johnson. I was born on 1964-10-01. I have had a vaccination visit. "
                        "I take metformin twice daily. My pharmacy is Walgreens in Santa Clara."
                    ),
                }
            ]
        )
        processor = patient_intake_speech_guard.PatientIntakeSpeechGuardProcessor(state)
        with patch.object(processor, "push_frame", new=AsyncMock()) as push_frame:
            await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            await processor.process_frame(
                LLMTextFrame(text="Let me review that for you. "),
                FrameDirection.DOWNSTREAM,
            )
            await processor.process_frame(
                FunctionCallInProgressFrame(
                    function_name="review_patient_intake",
                    tool_call_id="review-call",
                    arguments={},
                ),
                FrameDirection.DOWNSTREAM,
            )
            await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            await processor.process_frame(
                patient_intake_speech_guard.PatientIntakeToolResponseFrame(
                    text=(
                        "Thank you. I have all the information I need. Your name is Jane Johnson, "
                        "your date of birth is October 1st 1964, your reason for the visit is to "
                        "receive the second dose of your vaccination, your current medications are "
                        "metformin twice daily, and your current pharmacy is Walgreens in Santa Clara. "
                        "Is this all correct?"
                    )
                ),
                FrameDirection.DOWNSTREAM,
            )
            await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
            await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        return push_frame

    push_frame = asyncio.run(run_processor())
    forwarded = [call.args[0] for call in push_frame.await_args_list]
    spoken = [frame.text for frame in forwarded if isinstance(frame, LLMTextFrame)]
    assert spoken == [
        "Thank you. I have all the information I need. Your name is Jane Johnson, "
        "your date of birth is October 1st 1964, your reason for the visit is to receive "
        "the second dose of your vaccination, your current medications are metformin twice daily, "
        "and your current pharmacy is Walgreens in Santa Clara. "
        "Is this all correct?"
    ]
    assert "Let me review" not in spoken[0]


def test_record_rejects_an_unconfirmed_intake(tmp_path):
    """The persistence tool should not write until confirmation is explicit."""
    records_path = tmp_path / "patient_intake_records.jsonl"
    with patch.object(ambient_agent_tools, "RECORDS_PATH", records_path):
        result = ambient_agent_tools.record_patient_intake_result(
            {
                "patient_name": "Austin",
                "date_of_birth": "1994-01-06",
                "symptoms": "cold",
                "current_medications": "loratadine daily",
                "current_pharmacy": "Walgreens Pharmacy in New York",
                "patient_confirmed": False,
            }
        )

    assert result["status"] == "confirmation_required"
    assert result["response_text"] == "Please confirm that the reviewed information is correct before I save it."
    assert not records_path.exists()
