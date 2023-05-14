from typing import Literal, Optional, List, Dict, Any, Union

import time

import shortuuid
from pydantic import BaseModel, Field

from langport.constants import ErrorCode


class WorkerStatus(BaseModel):
    model_name: str
    speed: int = 1
    queue_length: int

class WorkerHeartbeatPing(BaseModel):
    worker_id: str

class WorkerHeartbeatPong(BaseModel):
    exist: bool

class RegisterWorkerRequest(BaseModel):
    worker_id: str
    worker_addr: str
    worker_type: str
    check_heart_beat: bool

class RemoveWorkerRequest(BaseModel):
    worker_id: str


class WorkerAddressRequest(BaseModel):
    model_name: str
    worker_type: str

class WorkerAddressResponse(BaseModel):
    address: str

class ListModelsResponse(BaseModel):
    models: List[str]


class BaseWorkerTask(BaseModel):
    task_id: str = Field(default_factory=lambda: f"task-{shortuuid.random()}")

class EmbeddingsTask(BaseWorkerTask):
    model: str
    input: str
    user: Optional[str] = None

class GenerationTask(BaseWorkerTask):
    prompt: str
    temperature: Optional[float] = 0.7
    repetition_penalty: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 1
    max_new_tokens: Optional[int] = None
    stop: Optional[Union[List[str], str]] = None
    echo: Optional[bool] = False
    stop_token_ids: Optional[List[int]] = None

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class BaseWorkerResult(BaseModel):
    task_id: str = Field(default_factory=lambda: f"task-{shortuuid.random()}")
    created: int = Field(default_factory=lambda: int(time.time()))
    type: Literal["data", "error", "finish", "done"]
    message: Optional[str] = None
    error_code: int = ErrorCode.OK

class EmbeddingWorkerResult(BaseWorkerResult):
    embedding: List[float]
    usage: UsageInfo

class GenerationWorkerResult(BaseWorkerResult):
    text: str
    logprobs: Optional[int] = None
    finish_reason: Optional[Literal["stop", "length"]] = None
    usage: UsageInfo