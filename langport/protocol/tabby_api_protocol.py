from typing import Literal, Optional, List, Dict, Any, Union

import time

import shortuuid
from pydantic import BaseModel, Field

class Choice(BaseModel):
    index: int
    text: str

class CompletionRequest(BaseModel):
    language: str = "unknown"
    prompt: str

class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{shortuuid.random()}")
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[Choice] = []


class ChoiceEvent(BaseModel):
    type: Literal["completion", "view", "select"]
    completion_id: str
    choice_index: int


class CompletionEvent(BaseModel):
    type: Literal["completion", "view", "select"]
    id: str = Field(default_factory=lambda: f"modelperm-{shortuuid.random()}")
    language: str = "unknown"
    prompt: str
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[Choice] = []

class ValidationError(BaseModel):
    loc: List[Union[int, str]]
    msg: str
    type: str

class HTTPValidationError(BaseModel):
    detail: List[ValidationError] = []