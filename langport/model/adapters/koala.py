from typing import List, Optional
import warnings
from functools import cache

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
)

from langport.data.conversation import Conversation, get_conv_template
from langport.model.model_adapter import BaseAdapter


class KoalaAdapter(BaseAdapter):
    """The model adapter for koala"""

    def match(self, model_path: str):
        return "koala" in model_path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("koala_v1")
