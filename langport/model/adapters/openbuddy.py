import warnings

import torch
from transformers import (
    LlamaTokenizer,
    LlamaTokenizerFast,
    LlamaForCausalLM,
)

from langport.data.conversation import ConversationHistory, SeparatorStyle
from langport.data.conversation.conversation_settings import get_conv_settings
from langport.model.model_adapter import BaseAdapter

class OpenBuddyAdapter(BaseAdapter):
    """The model adapter for OpenBuddy/openbuddy-7b-v1.1-bf16-enc"""

    def match(self, model_path: str):
        return "openbuddy" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        if "-bf16" in model_path:
            from_pretrained_kwargs["torch_dtype"] = torch.bfloat16
            warnings.warn(
                "## This is a bf16(bfloat16) variant of OpenBuddy. Please make sure your GPU supports bf16."
            )
        model = LlamaForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
        )
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("openbuddy")
        return ConversationHistory(
            system="""Consider a conversation between User (a human) and Assistant (named Buddy).
Buddy is an INTP-T, a friendly, intelligent and multilingual AI assistant, by OpenBuddy team. GitHub: https://github.com/OpenBuddy/OpenBuddy
Buddy cannot access the Internet.
Buddy can fluently speak the user's language (e.g. English, Chinese).
Buddy can generate poems, stories, code, essays, songs, parodies, and more.
Buddy possesses vast knowledge about the world, history, and culture.
Buddy's responses are always safe, creative, high-quality, human-like, and interesting.
Buddy strictly refuses to discuss political, NSFW, or other unsafe topics.

User: Hi.
Assistant: Hi, I'm Buddy, your AI assistant. How can I help you today?""",
            messages=(),
            offset=0,
            settings=settings,
        )
