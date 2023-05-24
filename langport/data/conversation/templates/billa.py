from langport.data.conversation import (
    Conversation,
    SeparatorStyle,
)


billa = Conversation(
    name="billa",
    system="",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
    sep="\n",
    stop_str="Human:",
)