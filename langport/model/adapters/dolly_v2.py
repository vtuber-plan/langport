from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class DollyV2Adapter(BaseAdapter):
    """The model adapter for databricks/dolly-v2-12b"""

    def match(self, model_path: str):
        return "dolly-v2" in model_path

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("dolly_v2")
        return ConversationHistory(
            system="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
            messages=(),
            offset=0,
            settings=settings,
        )
