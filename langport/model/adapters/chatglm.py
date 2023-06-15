from transformers import (
    AutoModel,
    AutoTokenizer,
)

from langport.model.model_adapter import BaseAdapter

class ChatGLMAdapter(BaseAdapter):
    """The model adapter for THUDM/chatglm-6b"""

    def match(self, model_path: str):
        return "chatglm" in model_path
