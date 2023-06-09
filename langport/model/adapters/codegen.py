from transformers import AutoModelForCausalLM, AutoTokenizer
from langport.model.model_adapter import BaseAdapter

class CodeGenAdapter(BaseAdapter):
    """The model adapter for CodeGen models."""

    def match(self, model_path: str):
        return "codegen" in model_path


class CodeGen2Adapter(BaseAdapter):
    """The model adapter for CodeGen2 models."""

    def match(self, model_path: str):
        return "codegen2" in model_path
