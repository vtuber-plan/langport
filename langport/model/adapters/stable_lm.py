from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class StableLMAdapter(BaseAdapter):
    """The model adapter for StabilityAI/stablelm-tuned-alpha-7b"""

    def match(self, model_path: str):
        return "stablelm" in model_path

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("stablelm")
        return ConversationHistory(
            system="""<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
""",
            messages=(),
            offset=0,
            settings=settings,
        )
