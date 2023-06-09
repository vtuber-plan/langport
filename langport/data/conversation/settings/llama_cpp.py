from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)

# llama_cpp default template
llama_cpp = ConversationSettings(
    name="llama_cpp",
    roles=("User", "Assistant"),
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n",
)
