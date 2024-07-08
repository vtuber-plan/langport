from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class RobinAdapter(BaseAdapter):
    """The model adapter for Robin"""

    def match(self, model_path: str):
        return "robin" in model_path

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("robin")
        return ConversationHistory(
            system="",
            messages=[],
            offset=0,
            settings=settings,
        )
