import os
from langport.data.conversation import ConversationHistory, SeparatorStyle
from langport.data.conversation.conversation_settings import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class QwenAdapter(BaseAdapter):
    """The model adapter for Robin"""

    def match(self, model_path: str):
        if model_path.endswith('/'):
            model_path = model_path[:-1]
        dir_name = os.path.basename(model_path)
        parent_name = os.path.dirname(model_path)
        return dir_name.lower().startswith('qwen') or parent_name.lower().startswith('qwen')

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("qwen")
        return ConversationHistory(
            system="",
            messages=[],
            offset=0,
            settings=settings,
        )
