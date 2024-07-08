from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings
from langport.model.model_adapter import BaseAdapter

class ChatGLMAdapter(BaseAdapter):
    """The model adapter for THUDM/chatglm-6b"""

    def match(self, model_path: str):
        return "chatglm" in model_path

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("chatglm")
        return ConversationHistory(
            system="",
            messages=[],
            offset=0,
            settings=settings,
        )
