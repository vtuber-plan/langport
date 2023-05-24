from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# Vicuna v1.1 template
vicuna_v1_1 = ConversationSettings(
    name="vicuna_v1.1",
    roles=("USER", "ASSISTANT"),
    sep_style=SeparatorStyle.ADD_COLON_TWO,
    sep=" ",
    sep2="</s>",
)