from langport.data.conversation import Conversation, SeparatorStyle
from langport.model.model_adapter import BaseAdapter


class ClaudeAdapter(BaseAdapter):
    """The model adapter for Claude."""

    def match(self, model_path: str):
        return model_path == "claude-v1"

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return Conversation(
    name="claude",
    system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n\n",
)

