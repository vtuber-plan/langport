from typing import List, Optional
from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class BiLLaAdapter(BaseAdapter):
    """The model adapter for BiLLa."""

    def match(self, model_path: str):
        return "billa" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("bard")
        return ConversationHistory(
            system="",
            messages=(),
            offset=0,
            settings=settings,
        )