from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# internlm default template
internlm = ConversationSettings(
    name="internlm",
    roles=("<|User|>", "<|Bot|>"),
    sep_style=SeparatorStyle.ADD_COLON_TWO,
    round_sep="<s>",
    system_sep="",
    sep="<eoh>\n",
    sep2="<eoa>\n",
    stop_str="<eoa>"
)
