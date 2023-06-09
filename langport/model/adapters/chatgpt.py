from langport.data.conversation import ConversationHistory
from langport.data.conversation.conversation_settings import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class ChatGPTAdapter(BaseAdapter):
    """The model adapter for ChatGPT."""

    def match(self, model_path: str):
        return model_path == "gpt-3.5-turbo" or model_path == "gpt-4"

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("chatgpt")
        return ConversationHistory(
            system="You are a helpful assistant.",
            messages=[],
            offset=0,
            settings=settings,
        )
