from enum import Enum
from typing import Literal, Optional, List, Dict, Any, Set, Union

import time

import shortuuid
from pydantic import BaseModel, Field

# https://code.visualstudio.com/docs/languages/identifiers
class Language(str, Enum):
    UNKNOWN = "unknown"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"

class EventType(str, Enum):
    COMPLETION = "completion"
    VIEW = "view"
    SELECT = "select"

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

EventTypeMapping = {
    EventType.COMPLETION: CompletionEvent,
    EventType.VIEW: ChoiceEvent,
    EventType.SELECT: ChoiceEvent,
}


'''
Language related
'''


class LanguagePreset(BaseModel):
    max_length: int
    stop_words: List[str]
    reserved_keywords: Optional[Set]


LanguagePresets = {
    Language.UNKNOWN: LanguagePreset(
        max_length=128,
        stop_words=["\n\n"],
    ),
    Language.PYTHON: LanguagePreset(
        max_length=64,
        stop_words=["\n\n", "\ndef", "\n#", "\nimport", "\nfrom", "\nclass"],
        reserved_keywords=set(
            [
                "False",
                "class",
                "from",
                "or",
                "None",
                "continue",
                "global",
                "pass",
                "True",
                "def",
                "if",
                "raise",
                "and",
                "del",
                "import",
                "return",
                "as",
                "elif",
                "in",
                "try",
                "assert",
                "else",
                "is",
                "while",
                "async",
                "except",
                "lambda",
                "with",
                "await",
                "finally",
                "nonlocal",
                "yield",
                "break",
                "for",
                "not",
            ]
        ),
    ),
    Language.JAVASCRIPT: LanguagePreset(
        max_length=128, stop_words=["\n\n", "\nfunction", "\n//", "\nimport", "\nclass"]
    ),
    Language.TYPESCRIPT: LanguagePreset(
        max_length=128,
        stop_words=[
            "\n\n",
            "\nfunction",
            "\n//",
            "\nimport",
            "\nclass",
            "\ninterface",
            "\ntype",
        ],
    ),
}