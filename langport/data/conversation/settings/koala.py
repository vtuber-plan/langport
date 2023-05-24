from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# Koala default template
koala_v1 = ConversationSettings(
    name="koala_v1",
    roles=("USER", "GPT"),
    sep_style=SeparatorStyle.ADD_COLON_TWO,
    sep=" ",
    sep2="</s>",
)
