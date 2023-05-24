from typing import List, Optional
import warnings
from functools import cache

from langport.data.conversation import Conversation, get_conv_template
from langport.model.model_adapter import BaseAdapter


class BiLLaAdapter(BaseAdapter):
    """The model adapter for BiLLa."""

    def match(self, model_path: str):
        return "billa" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("billa")