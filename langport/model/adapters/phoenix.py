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

class PhoenixAdapter(BaseAdapter):
    """The model adapter for FreedomIntelligence/phoenix-inst-chat-7b"""

    def match(self, model_path: str):
        return "phoenix" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("phoenix")
