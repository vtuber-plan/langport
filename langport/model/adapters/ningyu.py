from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class NingYuAdapter(BaseAdapter):
    """The model adapter for ningyu"""

    def match(self, model_path: str):
        return "ningyu" in model_path

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("ningyu")
        return ConversationHistory(
            system="""A chat between a curious user and an artificial intelligence assistant.
The name of the assistant is NingYu (凝语).
The assistant gives helpful, detailed, and polite answers to the user's questions.""",
            messages=[],
            offset=0,
            settings=settings,
        )

