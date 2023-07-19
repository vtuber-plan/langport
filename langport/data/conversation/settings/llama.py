from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# Llama default template
llama = ConversationSettings(
    name="llama",
    roles=("user", "assistant"),
    sep_style=SeparatorStyle.LLAMA,
    sep="",
    stop_str=["[/INST]", "[INST]"]
)
