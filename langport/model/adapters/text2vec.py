from langport.model.model_adapter import BaseAdapter

class BertAdapter(BaseAdapter):
    """The model adapter for bert text2vec models."""

    def match(self, model_path: str):
        return "bert" in model_path
