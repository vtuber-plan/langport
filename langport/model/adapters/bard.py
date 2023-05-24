from typing import List, Optional
import warnings
from functools import cache

from langport.data.conversation import Conversation, get_conv_template
from langport.model.model_adapter import BaseAdapter

class BardAdapter(BaseAdapter):
    """The model adapter for Bard."""

    def match(self, model_path: str):
        return model_path == "bard"

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("bard")