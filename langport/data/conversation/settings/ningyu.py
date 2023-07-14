from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)

# Ningyu default template
ningyu = ConversationSettings(
    name="ningyu",
    roles=("user", "assistant"),
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n### ",
    stop_str="###",
)
