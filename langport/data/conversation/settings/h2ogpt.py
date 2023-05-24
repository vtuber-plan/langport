from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


h2ogpt = ConversationSettings(
    name="h2ogpt",
    roles=("<|prompt|>", "<|answer|>"),
    sep_style=SeparatorStyle.NO_COLON_SINGLE,
    sep="</s>",
)