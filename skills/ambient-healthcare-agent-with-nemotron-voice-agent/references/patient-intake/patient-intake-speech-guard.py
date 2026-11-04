# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime safeguards for natural patient-intake speech."""

from __future__ import annotations

import copy
import re

from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.generic.ambient_agent_tools import _review_response

PATIENT_INTAKE_WELCOME_TEXT = (
    "Hello and welcome to the patient intake agent. I'm here to help you get checked in for "
    "your appointment and will be asking you a few questions in order to get ready for your "
    "visit with the doctor. First, could you please tell me your full name?"
)

PATIENT_INTAKE_TURN_REMINDER = (
    "Patient-intake control reminder: while any required field is missing, output exactly one short "
    "question for the earliest missing field and do not repeat any known patient value. "
    "Never repeat or restart the welcome after the session's opening. "
    "Never write an intake review yourself. "
    "If this message supplies the final missing field, even if ASR phrased it as a question, "
    "call review_patient_intake silently. If it corrects a reviewed field, call review_patient_intake "
    "again. If it clearly confirms the latest review, call record_patient_intake. "
    "Never output markdown, asterisks, headings, bullets, colon-labeled fields, or a field dump."
)

PATIENT_INTAKE_FIELD_ORDER = (
    "patient_name",
    "date_of_birth",
    "symptoms",
    "current_medications",
    "current_pharmacy",
)

PATIENT_INTAKE_COLLECTION_QUESTIONS = {
    "patient_name": "Could you please tell me your full name?",
    "date_of_birth": "What is your date of birth?",
    "symptoms": "What symptoms are you experiencing, or what is the reason for your visit?",
    "current_medications": "What medications are you currently taking?",
    "current_pharmacy": "What is your current or preferred pharmacy?",
}

_MARKDOWN_PATTERN = re.compile(r"[*_#\x60]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_SPOKEN_DATE_COMMA_PATTERN = re.compile(
    r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"\d{1,2}(?:st|nd|rd|th)),\s+(\d{4})\b",
    flags=re.IGNORECASE,
)
_LABELED_REVIEW_PATTERN = re.compile(
    r"\bname\s*:\s*(?P<patient_name>.*?)\s+"
    r"date\s+of\s+birth\s*:\s*(?P<date_of_birth>.*?)\s+"
    r"(?:symptoms\s+or\s+visit\s+reason|symptoms|visit\s+reason)\s*:\s*(?P<symptoms>.*?)\s+"
    r"(?:current\s+medications?|medications?)\s*:\s*(?P<current_medications>.*?)\s+"
    r"(?:current\s+pharmacy|pharmacy)\s*:\s*(?P<current_pharmacy>.*?)"
    r"(?:\s+is\s+this\s+all\s+correct\s*\?|$)",
    flags=re.IGNORECASE | re.DOTALL,
)

_EXPLICIT_FIELD_PATTERNS = {
    "patient_name": re.compile(r"\b(?:my|patient(?:'s)?)\s+(?:full\s+)?name\s+(?:is|:)\b", re.IGNORECASE),
    "date_of_birth": re.compile(r"\b(?:born|date\s+of\s+birth|dob)\b|\b\d{4}-\d{2}-\d{2}\b", re.IGNORECASE),
    "symptoms": re.compile(
        r"\b(?:symptoms?|visit\s+reason|reason\s+for\s+(?:my|the)\s+visit|"
        r"am\s+experiencing|have\s+had)\b",
        re.IGNORECASE,
    ),
    "current_medications": re.compile(
        r"\b(?:medications?|meds?|prescriptions?|i\s+take|i\s+am\s+taking)\b",
        re.IGNORECASE,
    ),
    "current_pharmacy": re.compile(r"\bpharmac(?:y|ies)\b", re.IGNORECASE),
}

_UNANSWERED_PATTERN = re.compile(
    r"^\s*(?:i\s+do\s+not\s+know|i\s+don't\s+know|not\s+sure|skip|prefer\s+not\s+to\s+say)\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def _message_text(content: object) -> str:
    """Return plain text from an OpenAI-style message content value."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return " ".join(parts)


def _asked_field(text: str) -> str | None:
    """Identify the intake field requested by an assistant question."""
    if "?" not in text:
        return None
    normalized = text.lower()
    if "pharmacy" in normalized:
        return "current_pharmacy"
    if "medication" in normalized or " medicine" in normalized:
        return "current_medications"
    if "symptom" in normalized or "reason for your visit" in normalized:
        return "symptoms"
    if "date of birth" in normalized or "when were you born" in normalized:
        return "date_of_birth"
    if "name" in normalized:
        return "patient_name"
    return None


def _explicit_user_fields(text: str) -> set[str]:
    """Recognize only high-confidence intake fields stated directly by a user."""
    return {
        field
        for field, pattern in _EXPLICIT_FIELD_PATTERNS.items()
        if pattern.search(text)
    }


class PatientIntakeConversationState:
    """Track which intake field should be requested without storing field values."""

    def __init__(self) -> None:
        """Initialize an empty intake collection state."""
        self.known_fields: set[str] = set()
        self.next_missing_field: str | None = PATIENT_INTAKE_FIELD_ORDER[0]
        self.welcome_emitted = False

    def consume_welcome(self) -> bool:
        """Allow the welcome once, and only before collection has started."""
        if self.welcome_emitted or self.known_fields:
            return False
        self.welcome_emitted = True
        return True

    def update_from_messages(self, messages: list[dict]) -> None:
        """Infer answered fields from prompt/answer sequencing and explicit user text."""
        known_fields: set[str] = set()
        pending_field: str | None = None
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            text = _message_text(message.get("content"))
            if role == "assistant":
                pending_field = _asked_field(text)
                continue
            if role != "user":
                continue
            if pending_field and text.strip() and not _UNANSWERED_PATTERN.fullmatch(text):
                known_fields.add(pending_field)
            known_fields.update(_explicit_user_fields(text))
            pending_field = None

        self.known_fields = known_fields
        self.next_missing_field = next(
            (field for field in PATIENT_INTAKE_FIELD_ORDER if field not in known_fields),
            None,
        )

    def collection_response(self) -> str:
        """Return the deterministic next-field question while collection is active."""
        if self.next_missing_field:
            return PATIENT_INTAKE_COLLECTION_QUESTIONS[self.next_missing_field]
        return ""

    def missing_fields(self) -> list[str]:
        """Return required fields not evidenced by the conversation."""
        return [field for field in PATIENT_INTAKE_FIELD_ORDER if field not in self.known_fields]


def sanitize_patient_intake_speech(text: str) -> str:
    """Rewrite a labeled intake dump and strip markdown from other speech."""
    plain_text = _WHITESPACE_PATTERN.sub(" ", _MARKDOWN_PATTERN.sub("", text)).strip()
    plain_text = _SPOKEN_DATE_COMMA_PATTERN.sub(r"\1 \2", plain_text)
    labeled_review = _LABELED_REVIEW_PATTERN.search(plain_text)
    if not labeled_review:
        return plain_text
    return _review_response(
        {
            field: value.strip().rstrip(".")
            for field, value in labeled_review.groupdict().items()
        }
    )


def patient_intake_welcome_frames() -> list[Frame]:
    """Build the deterministic opening turn for patient intake."""
    return [
        LLMFullResponseStartFrame(),
        LLMTextFrame(text=PATIENT_INTAKE_WELCOME_TEXT),
        LLMFullResponseEndFrame(),
    ]


class PatientIntakeToolResponseFrame(LLMTextFrame):
    """Mark deterministic tool-authored speech so the guard can pass it unchanged."""


class PatientIntakeTurnReminderProcessor(FrameProcessor):
    """Attach a non-persistent routing and speech reminder to each user turn."""

    def __init__(
        self,
        conversation_state: PatientIntakeConversationState | None = None,
    ) -> None:
        """Initialize the patient-intake reminder processor."""
        super().__init__(name="patient-intake-turn-reminder")
        self._conversation_state = conversation_state or PatientIntakeConversationState()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Forward frames while augmenting outbound LLM context copies."""
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMContextFrame):
            await self.push_frame(self._reminded_frame(frame), direction)
            return
        await self.push_frame(frame, direction)

    def _reminded_frame(self, frame: LLMContextFrame) -> LLMContextFrame:
        source = frame.context
        messages = copy.deepcopy(list(source.get_messages()))
        self._conversation_state.update_from_messages(messages)
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = f"{content}\n\n{PATIENT_INTAKE_TURN_REMINDER}"
            elif isinstance(content, list):
                message["content"] = [
                    *content,
                    {"type": "text", "text": PATIENT_INTAKE_TURN_REMINDER},
                ]
            else:
                message["content"] = PATIENT_INTAKE_TURN_REMINDER
            break
        return LLMContextFrame(
            context=LLMContext(messages, tools=source.tools, tool_choice=source.tool_choice)
        )


class PatientIntakeSpeechGuardProcessor(FrameProcessor):
    """Buffer and sanitize each complete patient-intake LLM response."""

    def __init__(
        self,
        conversation_state: PatientIntakeConversationState | None = None,
    ) -> None:
        """Initialize response buffering state."""
        super().__init__(name="patient-intake-speech-guard")
        self._conversation_state = conversation_state or PatientIntakeConversationState()
        self._response_depth = 0
        self._text_parts: list[str] = []
        self._skip_tts: bool | None = None
        self._tool_response_seen = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Suppress raw streamed text and emit one speech-safe response."""
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InterruptionFrame):
            self._reset()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            if self._response_depth == 0:
                self._text_parts.clear()
                self._skip_tts = None
                self._tool_response_seen = False
                await self.push_frame(frame, direction)
            self._response_depth += 1
            return

        if self._response_depth and isinstance(frame, FunctionCallInProgressFrame):
            # A tool-authored response will follow. Discard any model preamble
            # that was streamed before the function call.
            self._text_parts.clear()
            await self.push_frame(frame, direction)
            return

        if self._response_depth and isinstance(frame, LLMTextFrame):
            self._text_parts.append(frame.text)
            self._skip_tts = frame.skip_tts
            if isinstance(frame, PatientIntakeToolResponseFrame):
                self._tool_response_seen = True
            return

        if self._response_depth and isinstance(frame, LLMFullResponseEndFrame):
            self._response_depth -= 1
            if self._response_depth:
                return
            speech = sanitize_patient_intake_speech("".join(self._text_parts))
            is_allowed_welcome = (
                speech == PATIENT_INTAKE_WELCOME_TEXT
                and self._conversation_state.consume_welcome()
            )
            if (
                speech
                and not is_allowed_welcome
                and not self._tool_response_seen
                and self._conversation_state.next_missing_field is not None
            ):
                speech = self._conversation_state.collection_response()
            elif (
                speech
                and not is_allowed_welcome
                and not self._tool_response_seen
            ):
                # Once collection is complete, only a marked tool response may
                # speak. Empty model responses are a normal tool-call transition.
                speech = ""
            if speech:
                output = LLMTextFrame(text=speech)
                output.skip_tts = self._skip_tts
                await self.push_frame(output, direction)
            self._reset()
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    def _reset(self) -> None:
        self._response_depth = 0
        self._text_parts.clear()
        self._skip_tts = None
        self._tool_response_seen = False
