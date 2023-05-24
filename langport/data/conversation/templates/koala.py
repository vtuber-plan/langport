from langport.data.conversation import (
    Conversation,
    SeparatorStyle,
)


# Koala default template
koala_v1 = Conversation(
    name="koala_v1",
    system="BEGINNING OF CONVERSATION:",
    roles=("USER", "GPT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_TWO,
    sep=" ",
    sep2="</s>",
)
