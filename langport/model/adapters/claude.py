from langport.data.conversation import ConversationHistory, SeparatorStyle
from langport.data.conversation.conversation_settings import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class ClaudeAdapter(BaseAdapter):
    """The model adapter for Claude."""

    def match(self, model_path: str):
        return model_path == "claude-v1"

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError()

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("claude")
        return ConversationHistory(
            system="",
            messages=(),
            offset=0,
            settings=settings,
        )
