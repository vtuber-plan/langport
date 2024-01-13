import os
from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings
from langport.model.model_adapter import BaseAdapter

class CodeLlamaAdapter(BaseAdapter):
    """The model adapter for codellama"""

    def match(self, model_path: str):
        if model_path.endswith('/'):
            model_path = model_path[:-1]
        dir_name = os.path.basename(model_path)
        parent_name = os.path.dirname(model_path)
        return dir_name.lower().startswith('codellama') or parent_name.lower().startswith('codellama')

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        llama_settings = get_conv_settings("llama")
        return ConversationHistory(
            system="",
            messages=[],
            offset=0,
            settings=llama_settings,
        )
