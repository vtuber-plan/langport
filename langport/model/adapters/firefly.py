from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class FireFlyAdapter(BaseAdapter):
    """The model adapter for FireFly"""

    def match(self, model_path: str):
        return "firefly-baichuan" in model_path or "firefly-bloom" in model_path or "firefly-ziya" in model_path

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("firefly")
        return ConversationHistory(
            system="The conversation between human and AI assistant.",
            messages=[],
            offset=0,
            settings=settings,
        )
