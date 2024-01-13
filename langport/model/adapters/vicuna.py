import warnings
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)

from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class VicunaAdapter(BaseAdapter):
    "Model adapater for vicuna-v1.1"

    def match(self, model_path: str):
        return "vicuna" in model_path

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("vicuna_v1.1")
        return ConversationHistory(
            system="A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.",
            messages=[],
            offset=0,
            settings=settings,
        )

    def raise_warning_for_old_weights(self, model):
        if isinstance(model, LlamaForCausalLM) and model.model.vocab_size > 32000:
            warnings.warn(
                "\nYou are probably using the old Vicuna-v0 model, "
                "which will generate unexpected results with the "
                "current fastchat.\nYou can try one of the following methods:\n"
                "1. Upgrade your weights to the new Vicuna-v1.1: https://github.com/lm-sys/FastChat#vicuna-weights.\n"
                "2. Use the old conversation template by `python3 -m fastchat.serve.cli --model-path /path/to/vicuna-v0 --conv-template conv_one_shot`\n"
                "3. Downgrade fschat to fschat==0.1.10 (Not recommonded).\n"
            )
