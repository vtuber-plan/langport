from langport.data.conversation import Conversation
from langport.model.model_adapter import BaseAdapter


class ChatGPTAdapter(BaseAdapter):
    """The model adapter for ChatGPT."""

    def match(self, model_path: str):
        return model_path == "gpt-3.5-turbo" or model_path == "gpt-4"

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return Conversation(
    name="chatgpt",
    system="You are a helpful assistant.",
    roles=("user", "assistant"),
    messages=[],
    offset=0,
    sep_style=None,
    sep=None,
)
