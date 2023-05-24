from langport.data.conversation import Conversation, SeparatorStyle
from langport.model.model_adapter import BaseAdapter


class BaizeAdapter(BaseAdapter):
    """The model adapter for project-baize/baize-lora-7B"""

    def match(self, model_path: str):
        return "baize" in model_path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return Conversation(
    name="baize",
    system="The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.",
    roles=("[|Human|]", "[|AI|]"),
    messages=[
        ["[|Human|]", "Hello!"],
        ["[|AI|]", "Hi!"],
    ],
    offset=2,
    sep_style=SeparatorStyle.BAIZE,
    sep="[|Human|]",
    stop_str="[|Human|]",
)
