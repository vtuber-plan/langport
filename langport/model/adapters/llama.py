import os
from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings
from langport.model.model_adapter import BaseAdapter

LLAMA_CHAT_SYSTEM = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

class LlamaAdapter(BaseAdapter):
    """The model adapter for llama"""

    def match(self, model_path: str):
        if model_path.endswith('/'):
            model_path = model_path[:-1]
        dir_name = os.path.basename(model_path)
        parent_name = os.path.dirname(model_path)
        return dir_name.lower().startswith('llama') or parent_name.lower().startswith('llama')

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        llama_settings = get_conv_settings("llama")
        base_settings = get_conv_settings("zero_shot")
        if "chat" in model_path.lower():
            return ConversationHistory(
                system=LLAMA_CHAT_SYSTEM,
                messages=[],
                offset=0,
                settings=llama_settings,
            )
        else:
            return ConversationHistory(
                system=LLAMA_CHAT_SYSTEM,
                messages=[],
                offset=0,
                settings=base_settings,
            )
