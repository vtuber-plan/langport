from transformers import AutoModelForCausalLM, AutoTokenizer
from langport.model.model_adapter import BaseAdapter

class CodeGenAdapter(BaseAdapter):
    """The model adapter for CodeGen models."""

    def match(self, model_path: str):
        return "codegen" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, **from_pretrained_kwargs)

        return model, tokenizer

class CodeGen2Adapter(BaseAdapter):
    """The model adapter for CodeGen2 models."""

    def match(self, model_path: str):
        return "codegen2" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, **from_pretrained_kwargs)

        return model, tokenizer