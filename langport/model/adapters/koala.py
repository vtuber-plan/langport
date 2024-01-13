from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class KoalaAdapter(BaseAdapter):
    """The model adapter for koala"""

    def match(self, model_path: str):
        return "koala" in model_path

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("koala_v1")
        return ConversationHistory(
            system="BEGINNING OF CONVERSATION:",
            messages=[],
            offset=0,
            settings=settings,
        )
