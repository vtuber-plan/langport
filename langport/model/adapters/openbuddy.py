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
    LlamaTokenizer,
    LlamaForCausalLM,
)

from langport.data.conversation import Conversation, get_conv_template
from langport.model.model_adapter import BaseAdapter

class OpenBuddyAdapter(BaseAdapter):
    """The model adapter for OpenBuddy/openbuddy-7b-v1.1-bf16-enc"""

    def match(self, model_path: str):
        return "openbuddy" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        if "-bf16" in model_path:
            from_pretrained_kwargs["torch_dtype"] = torch.bfloat16
            warnings.warn(
                "## This is a bf16(bfloat16) variant of OpenBuddy. Please make sure your GPU supports bf16."
            )
        model = LlamaForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
        )
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("openbuddy")

