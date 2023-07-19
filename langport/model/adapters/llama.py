from langport.data.conversation import ConversationHistory, SeparatorStyle
from langport.data.conversation.conversation_settings import get_conv_settings
from langport.model.model_adapter import BaseAdapter

LLAMA_CHAT_SYSTEM = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

class LlamaAdapter(BaseAdapter):
    """The model adapter for llama"""

    def match(self, model_path: str):
        return "llama" in model_path.lower()

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
