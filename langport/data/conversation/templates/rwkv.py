from langport.data.conversation import (
    Conversation,
    SeparatorStyle,
)


# RWKV-4-Raven default template
rwkv = Conversation(
    name="rwkv",
    system="",
    roles=("Bob", "Alice"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.RWKV,
    sep="",
    stop_str="\n\n",
)