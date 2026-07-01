# § 2 — Pydantic API Schemas (`api_types.py`)

> **Generates**: `nvds_action_detector/api_types.py`
> **Critical Rules**: UNIFORM_CHUNKING_BYPASSES_DDM
> **Pipeline stage**: Data contracts — defines request/response shapes for all endpoints (§ 1)

---

## Request Models

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional, Union
from typing_extensions import Annotated

# --- Input content types ---
# The chat completion endpoint (§ 1) accepts multimodal content:
#   TextContent     — user's text prompt (e.g., "Analyze SOP compliance")
#   VideoURLContent — video via HTTP URL, RTSP stream, or base64 data URI
#   VideoFileContent — previously uploaded file (via POST /v1/files)
#   CameraInputContent — live Basler industrial camera feed (§ 8)

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class VideoURL(BaseModel):
    url: str  # HTTP(S), data:video/mp4;base64,... or rtsp://

class VideoURLContent(BaseModel):
    type: Literal["video_url"] = "video_url"
    video_url: VideoURL

class VideoFileContent(BaseModel):
    type: Literal["input_video"] = "input_video"
    file_id: str

class InputCamera(BaseModel):
    """Basler industrial camera configuration.
    See § 8 for physical camera and emulation usage patterns."""
    camera_id: str
    camera_vendor: Literal["Basler"] = "Basler"
    config: Optional[str] = None             # path to .pfs file
    camera_format: Optional[Literal["RGB", "UYVY", "YUY2"]] = None
    camera_width: Optional[int] = None
    camera_height: Optional[int] = None
    camera_fps_num: Optional[int] = Field(None, ge=1, le=1e8)
    camera_fps_den: Optional[int] = Field(None, ge=1, le=1e8)

class CameraInputContent(BaseModel):
    type: Literal["input_camera"] = "input_camera"
    input_camera: InputCamera

# Discriminated union: Pydantic uses the "type" field to pick the right model
ChatMessageContent = Annotated[
    Union[TextContent, VideoURLContent, VideoFileContent, CameraInputContent],
    Field(discriminator="type")
]

class ChatCompletionMessage(BaseModel):
    role: str
    content: Union[list[ChatMessageContent], str]

# --- Chunking options ---
# Two selectable algorithms, chosen via a discriminated union on `algorithm`:
#   "ddm-net" (default) — DDM event-boundary detection (Stage 1 CV + Stage 2)
#   "uniform"           — fixed-length chunks; bypasses the DDM model entirely (see § 3, § 6)
# Both set model_config extra="forbid", so a payload that mixes the two algorithms'
# fields (e.g. `threshold` under `algorithm:"uniform"`) is rejected with HTTP 422.
# Note: internal ChunkParams.max_length_sec defaults to 10.0s,
#        but API DdmNetChunkingOptions defaults to 60.0s

class DdmNetChunkingOptions(BaseModel):
    model_config = {"extra": "forbid"}
    algorithm: Literal["ddm-net"] = "ddm-net"
    threshold: float = Field(0.8, gt=0, lt=1.0)
    min_length_sec: float = Field(1.0, gt=0)
    max_length_sec: float = Field(60.0, gt=0)

class UniformChunkingOptions(BaseModel):
    model_config = {"extra": "forbid"}
    algorithm: Literal["uniform"] = "uniform"
    chunk_length_sec: float = Field(5.0, gt=0)    # fixed chunk duration in seconds

# Discriminated union — Pydantic picks the model from the `algorithm` literal.
ChunkingOptions = Annotated[
    Union[DdmNetChunkingOptions, UniformChunkingOptions],
    Field(discriminator="algorithm"),
]

class ChatCompletionRequest(BaseModel):
    messages: list[ChatCompletionMessage]
    model: Optional[str] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_completion_tokens: Optional[int] = None
    seed: Optional[int] = None
    top_p: Optional[float] = None
    # Defaults to DDM-net when omitted; pass {"algorithm":"uniform","chunk_length_sec":N} for fixed chunks.
    chunking_options: Optional[ChunkingOptions] = Field(default_factory=DdmNetChunkingOptions)
```

---

## Response Models

```python
# --- Health check response ---
class HealthSuccessResponse(BaseModel):
    object: Literal["health.response"] = "health.response"
    message: str

# --- File management responses ---
class FileObject(BaseModel):
    id: str                          # "file-<uuid>"
    object: Literal["file"] = "file"
    bytes: int
    created_at: int                  # Unix timestamp
    filename: str
    purpose: str

class FileList(BaseModel):
    object: Literal["list"] = "list"
    data: list[FileObject]

class DeletionStatus(BaseModel):
    id: str
    object: Literal["file.deleted"] = "file.deleted"
    deleted: bool = True

# --- Streaming (SSE) response models ---
# Used by the SSE generator (§ 7) to emit chat.completion.chunk events.
# Each chunk carries one action classification result from the 4-stage pipeline.

class DeltaMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = "assistant"
    content: Optional[str] = None

class ChatCompletionResponseStreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None
    chunk_metadata: Optional[dict] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str                                          # "chatcmpl-<uuid>"
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionResponseStreamChoice]

# --- Non-streaming response models ---
# Used when stream=false; collects all chunk results before responding.

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str] = None
    chunk_metadata_list: Optional[list[dict]] = None

class ChatCompletionResponse(BaseModel):
    id: str                                          # "chatcmpl-<uuid>"
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: Optional[dict] = None
```
