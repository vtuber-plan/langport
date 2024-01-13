from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class ChangGPTAdapter(BaseAdapter):
    """The model adapter for ChangGPT"""

    def match(self, model_path: str):
        return "changgpt" in model_path

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("changgpt")
        return ConversationHistory(
            system="",
            messages=[],
            offset=0,
            settings=settings,
        )
