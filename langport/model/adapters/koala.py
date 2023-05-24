from langport.data.conversation import ConversationHistory, SeparatorStyle
from langport.data.conversation.conversation_settings import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class KoalaAdapter(BaseAdapter):
    """The model adapter for koala"""

    def match(self, model_path: str):
        return "koala" in model_path

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("dolly_v2")
        return ConversationHistory(
            system="BEGINNING OF CONVERSATION:",
            messages=(),
            offset=0,
            settings=settings,
        )
