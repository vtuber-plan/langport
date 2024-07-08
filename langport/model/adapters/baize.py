from chatproto.conversation.history import ConversationHistory
from langport.model.model_adapter import BaseAdapter

from chatproto.registry import get_conv_settings

class BaizeAdapter(BaseAdapter):
    """The model adapter for project-baize/baize-lora-7B"""

    def match(self, model_path: str):
        return "baize" in model_path

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("baize")
        return ConversationHistory(
            system="The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.",
            messages=(
                ("[|Human|]", "Hello!"),
                ("[|AI|]", "Hi!"),
            ),
            offset=2,
            settings=settings,
        )
