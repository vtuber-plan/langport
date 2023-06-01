from transformers import AutoModelForCausalLM, AutoTokenizer
from langport.model.model_adapter import BaseAdapter

class StarCoderAdapter(BaseAdapter):
    """The model adapter for starcoder models from bigcode-project."""

    def match(self, model_path: str):
        return "starcoder" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, **from_pretrained_kwargs)

        return model, tokenizer