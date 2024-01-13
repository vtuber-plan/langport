from transformers import (
    AutoTokenizer,
)

from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class RwkvAdapter(BaseAdapter):
    """The model adapter for BlinkDL/RWKV-4-Raven"""

    def match(self, model_path: str):
        return "RWKV-4" in model_path

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("rwkv")
        return ConversationHistory(
            system="",
            messages=(),
            offset=0,
            settings=settings,
        )
