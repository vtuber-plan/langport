from typing import List, Optional
from langport.data.conversation import ConversationHistory, SeparatorStyle
from langport.data.conversation.conversation_settings import get_conv_settings
from langport.model.model_adapter import BaseAdapter

class BaichuanAdapter(BaseAdapter):
    """The model adapter for baichuan-inc/baichuan-7B"""

    def match(self, model_path: str):
        return "baichuan" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("baichuan")
        return ConversationHistory(
            system="",
            messages=[],
            offset=0,
            settings=settings,
        )