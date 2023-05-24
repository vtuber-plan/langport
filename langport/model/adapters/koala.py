from langport.data.conversation import Conversation, SeparatorStyle
from langport.model.model_adapter import BaseAdapter


class KoalaAdapter(BaseAdapter):
    """The model adapter for koala"""

    def match(self, model_path: str):
        return "koala" in model_path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return Conversation(
    name="koala_v1",
    system="BEGINNING OF CONVERSATION:",
    roles=("USER", "GPT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_TWO,
    sep=" ",
    sep2="</s>",
)
