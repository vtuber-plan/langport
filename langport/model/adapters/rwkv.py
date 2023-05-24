from transformers import (
    AutoTokenizer,
)

from langport.data.conversation import Conversation, SeparatorStyle
from langport.model.model_adapter import BaseAdapter


class RwkvAdapter(BaseAdapter):
    """The model adapter for BlinkDL/RWKV-4-Raven"""

    def match(self, model_path: str):
        return "RWKV-4" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        from langport.model.models.rwkv_model import RwkvModel

        model = RwkvModel(model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-160m", use_fast=True
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return Conversation(
    name="rwkv",
    system="",
    roles=("Bob", "Alice"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.RWKV,
    sep="",
    stop_str="\n\n",
)
