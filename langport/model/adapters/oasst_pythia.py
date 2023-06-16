from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from langport.data.conversation import ConversationHistory, SeparatorStyle
from langport.data.conversation.conversation_settings import get_conv_settings
from langport.model.model_adapter import BaseAdapter

class OasstPythiaAdapter(BaseAdapter):
    """The model adapter for OpenAssistant/oasst-sft-1-pythia-12b"""

    def match(self, model_path: str):
        return "oasst" in model_path and "pythia" in model_path
    
    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("oasst_pythia")
        return ConversationHistory(
            system="BEGINNING OF CONVERSATION:",
            messages=(),
            offset=0,
            settings=settings,
        )
