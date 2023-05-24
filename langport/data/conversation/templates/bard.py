from langport.data.conversation import (
    Conversation,
    SeparatorStyle,
)


bard = Conversation(
    name="bard",
    system="",
    roles=("0", "1"),
    messages=(),
    offset=0,
    sep_style=None,
    sep=None,
)