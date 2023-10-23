from langport.data.conversation import ConversationHistory, SeparatorStyle
from langport.data.conversation.conversation_settings import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class MistralAdapter(BaseAdapter):
    """The model adapter for Mistral"""

    def match(self, model_path: str):
        return model_path.lower().startswith("mistral")

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("mistral")
        return ConversationHistory(
            system="",
            messages=[],
            offset=0,
            settings=settings,
        )
