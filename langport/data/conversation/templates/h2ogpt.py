from langport.data.conversation import (
    Conversation,
    SeparatorStyle,
)


h2ogpt = Conversation(
    name="h2ogpt",
    system="",
    roles=("<|prompt|>", "<|answer|>"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.NO_COLON_SINGLE,
    sep="</s>",
)