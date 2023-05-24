from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)

# ChatGPT default template
chatgpt = ConversationSettings(
    name="chatgpt",
    roles=("user", "assistant"),
    sep_style=None,
    sep=None,
)