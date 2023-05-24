from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from langport.data.conversation import ConversationHistory, SeparatorStyle
from langport.data.conversation.conversation_settings import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class DollyV2Adapter(BaseAdapter):
    """The model adapter for databricks/dolly-v2-12b"""

    def match(self, model_path: str):
        return "dolly-v2" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        # 50277 means "### End"
        tokenizer.eos_token_id = 50277
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("dolly_v2")
        return ConversationHistory(
            system="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
            messages=(),
            offset=0,
            settings=settings,
        )
