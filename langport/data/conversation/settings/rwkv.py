from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# RWKV-4-Raven default template
rwkv = ConversationSettings(
    name="rwkv",
    roles=("Bob", "Alice"),
    sep_style=SeparatorStyle.RWKV,
    sep="",
    stop_str="\n\n",
)