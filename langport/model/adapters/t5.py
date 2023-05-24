from transformers import (
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
)

from langport.model.model_adapter import BaseAdapter


class T5Adapter(BaseAdapter):
    """The model adapter for lmsys/fastchat-t5-3b-v1.0"""

    def match(self, model_path: str):
        return "t5" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = T5Tokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
        )
        return model, tokenizer
