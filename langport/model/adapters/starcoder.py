from transformers import AutoModelForCausalLM, AutoTokenizer
from langport.model.model_adapter import BaseAdapter

class StarCoderAdapter(BaseAdapter):
    """The model adapter for starcoder models from bigcode-project."""

    def match(self, model_path: str):
        return "starcoder" in model_path
