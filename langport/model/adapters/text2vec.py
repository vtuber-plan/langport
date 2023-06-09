from transformers import BertTokenizer, BertModel
from langport.model.model_adapter import BaseAdapter

class SeberAdapter(BaseAdapter):
    """The model adapter for bert text2vec models."""

    def match(self, model_path: str):
        return "text2vec" in model_path
