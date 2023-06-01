from transformers import BertTokenizer, BertModel
from langport.model.model_adapter import BaseAdapter

class SeberAdapter(BaseAdapter):
    """The model adapter for bert text2vec models."""

    def match(self, model_path: str):
        return "text2vec" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertModel.from_pretrained(model_path, **from_pretrained_kwargs)

        return model, tokenizer