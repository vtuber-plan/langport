from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class FalconAdapter(BaseAdapter):
    """The model adapter for falcon"""

    def match(self, model_path: str):
        return "falcon" in model_path

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("falcon")
        return ConversationHistory(
            system="The conversation between human and AI assistant.",
            messages=[],
            offset=0,
            settings=settings,
        )
