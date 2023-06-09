from transformers import (
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
)

from langport.model.model_adapter import BaseAdapter


class T5Adapter(BaseAdapter):
    """The model adapter for lmsys/fastchat-t5-3b-v1.0"""

    def match(self, model_path: str):
        return "t5" in model_path

