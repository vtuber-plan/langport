from typing import Literal, Optional, List, Dict, Any, Union

import time

import shortuuid
from pydantic import BaseModel, Field


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
    check_heart_beat: bool
    worker_status: Optional[WorkerStatus]

class RemoveWorkerRequest(BaseModel):
    worker_id: str


class WorkerAddressRequest(BaseModel):
    model_name: str

class WorkerAddressResponse(BaseModel):
    address: str

class ListModelsResponse(BaseModel):
    models: List[str]