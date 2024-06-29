from typing import Literal, Optional, List, Dict, Any, Union

import time

import shortuuid
from pydantic import BaseModel, Field

from langport.constants import ErrorCode

class NodeInfo(BaseModel):
    node_id: str
    node_addr: str
    check_heart_beat: bool
    refresh_time: int = Field(default_factory=lambda: int(time.time()))


class RegisterNodeRequest(BaseModel):
    node_id: str
    node_addr: str
    check_heart_beat: bool

class RegisterNodeResponse(BaseModel):
    node_id: str
    node_addr: str
    check_heart_beat: bool

class RemoveNodeRequest(BaseModel):
    node_id: str

class RemoveNodeResponse(BaseModel):
    node_id: str

class HeartbeatPing(BaseModel):
    node_id: str

class HeartbeatPong(BaseModel):
    exist: bool

class NodeListRequest(BaseModel):
    node_id: str

class NodeListResponse(BaseModel):
    nodes: List[NodeInfo]

class ListNodeStatesRequest(BaseModel):
    pass

class ListNodeStatesResponse(BaseModel):
    states: List[str]

class GetNodeStateRequest(BaseModel):
    state_name: str

class GetNodeStateResponse(BaseModel):
    state_value: str
    state_ttl: int = 60

class NodeInfoRequest(BaseModel):
    node_id: str

class NodeInfoResponse(BaseModel):
    node_info: NodeInfo

class WorkerAddressRequest(BaseModel):
    condition: str
    expression: str

class WorkerAddressResponse(BaseModel):
    id_list: List[str]
    address_list: List[str]
    values: List[str]


class BaseWorkerTask(BaseModel):
    task_id: str = Field(default_factory=lambda: f"task-{shortuuid.random()}")

class EmbeddingsTask(BaseWorkerTask):
    model: str
    input: Union[str, List[str]]
    user: Optional[str] = None
    dimensions: Optional[int] = None

class GenerationTask(BaseWorkerTask):
    prompt: str
    temperature: Optional[float] = 0.7
    repetition_penalty: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 0
    max_tokens: Optional[int] = None
    stop: Optional[Union[List[str], str]] = None
    echo: Optional[bool] = False
    stop_token_ids: Optional[List[int]] = None
    logprobs: Optional[int] = None

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

class EmbeddingsObject(BaseModel):
    embedding: List[float]
    index: int

class EmbeddingWorkerResult(BaseWorkerResult):
    embeddings: List[EmbeddingsObject]
    usage: UsageInfo = None

class GenerationWorkerLogprobs(BaseModel):
    tokens: List[str]
    token_logprobs: List[float]
    top_logprobs: List[Dict[str, float]]
    text_offset: List[int]

class GenerationWorkerResult(BaseWorkerResult):
    text: str
    logprobs: Optional[GenerationWorkerLogprobs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None
    usage: UsageInfo = None
