from typing import Literal, Optional, List, Dict, Any, Union

import time

import shortuuid
from pydantic import BaseModel, Field

from langport.constants import ErrorCode


class WorkerStatus(BaseModel):
    model_name: str
    speed: int = 1
    queue_length: int

class WorkerHeartbeat(BaseModel):
    worker_id: str
    status: WorkerStatus

class RegisterWorkerRequest(BaseModel):
    worker_id: str
    worker_addr: str
    worker_type: str
    check_heart_beat: bool
    worker_status: Optional[WorkerStatus]

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
    temperature: float
    repetition_penalty: float
    top_p: float
    top_k: float
    max_new_tokens: int
    stop: Union[List[str], str]
    echo: bool
    stop_token_ids: List[int]

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class BaseWorkerResult(BaseModel):
    task_id: str = Field(default_factory=lambda: f"task-{shortuuid.random()}")
    created: int = Field(default_factory=lambda: int(time.time()))
    type: Literal["data", "error", "done"]
    message: Optional[str] = None
    error_code: int = ErrorCode.OK

class EmbeddingWorkerResult(BaseWorkerResult):
    embedding: List[float]
    usage: UsageInfo