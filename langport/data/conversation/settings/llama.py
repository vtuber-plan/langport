from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# Llama default template
llama = ConversationSettings(
    name="llama",
    system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
    roles=("user", "assistant"),
    sep_style=SeparatorStyle.LLAMA,
    sep="",
    stop_str=["[/INST]", "[INST]"]
)
