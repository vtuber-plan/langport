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

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer
    
    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("oasst_pythia")
        return ConversationHistory(
            system="BEGINNING OF CONVERSATION:",
            messages=(),
            offset=0,
            settings=settings,
        )
