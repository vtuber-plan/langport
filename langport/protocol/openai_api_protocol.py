from typing import Literal, Optional, List, Dict, Any, Union

import time

import shortuuid
from pydantic import BaseModel, Field

class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{shortuuid.random()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = True
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: str = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "langport"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = []

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class FunctionProperty(BaseModel):
    type: str
    description: Optional[str]
    enum: Optional[List[str]]

class FunctionParameters(BaseModel):
    type: str
    properties: Dict[str, FunctionProperty]
    required: List[str]

class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str]
    parameters: FunctionParameters

class FunctionEntry(BaseModel):
    name: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    functions: Optional[List[FunctionDefinition]] = None
    function_call: Optional[Union[Literal["none", "auto"], FunctionEntry]] = None
    user: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]]

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]

# TODO: Support List[int] and List[List[int]]
class EmbeddingsRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    user: Optional[str] = None
    encoding_format: Optional[Literal["float", "base64"]] = None
    dimensions: Optional[int] = None

class EmbeddingsData(BaseModel):
    object: str = "embedding"
    embedding: Union[List[float], str]
    index: int

class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingsData]
    model: str
    usage: UsageInfo


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    suffix: Optional[str] = None
    temperature: Optional[float] = 0.7
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class CompletionLogprobs(BaseModel):
    tokens: List[str]
    token_logprobs: List[float]
    top_logprobs: List[Dict[str, float]]
    text_offset: List[int]

class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogprobs] = None
    finish_reason: Optional[Literal["stop", "length"]]


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogprobs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
