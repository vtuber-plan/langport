from langport.data.conversation import (
    Conversation,
    SeparatorStyle,
)

# ChatGPT default template
chatgpt = Conversation(
    name="chatgpt",
    system="You are a helpful assistant.",
    roles=("user", "assistant"),
    messages=(),
    offset=0,
    sep_style=None,
    sep=None,
)