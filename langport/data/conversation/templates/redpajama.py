from langport.data.conversation import (
    Conversation,
    SeparatorStyle,
)


redpajama = Conversation(
    name="redpajama-incite",
    system="",
    roles=("<human>", "<bot>"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n",
    stop_str="<human>",
)