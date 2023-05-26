from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)

# ChatGPT default template
chatgpt = ConversationSettings(
    name="chatgpt",
    roles=("user", "assistant"),
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n### ",
    stop_str="###",
)