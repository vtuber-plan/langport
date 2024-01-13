from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class ClaudeAdapter(BaseAdapter):
    """The model adapter for Claude."""

    def match(self, model_path: str):
        return model_path == "claude-v1"

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("claude")
        return ConversationHistory(
            system="",
            messages=(),
            offset=0,
            settings=settings,
        )
