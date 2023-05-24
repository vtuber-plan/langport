from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


redpajama = ConversationSettings(
    name="redpajama-incite",
    roles=("<human>", "<bot>"),
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n",
    stop_str="<human>",
)