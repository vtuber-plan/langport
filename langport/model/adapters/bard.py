from typing import List, Optional
import warnings
from functools import cache

from langport.data.conversation import ConversationHistory, SeparatorStyle
from langport.data.conversation.conversation_settings import get_conv_settings
from langport.model.model_adapter import BaseAdapter

class BardAdapter(BaseAdapter):
    """The model adapter for Bard."""

    def match(self, model_path: str):
        return model_path == "bard"

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("bard")
        return ConversationHistory(
            system="",
            messages=(),
            offset=0,
            settings=settings,
        )