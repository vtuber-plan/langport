from typing import List, Optional
from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings
from langport.model.model_adapter import BaseAdapter

class WizardLMAdapter(BaseAdapter):
    """The model adapter for WizardLM/WizardLM-13B-V1.0"""

    def match(self, model_path: str):
        return "wizardlm" in model_path

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("wizardlm")
        return ConversationHistory(
            system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. ",
            messages=(),
            offset=0,
            settings=settings,
        )