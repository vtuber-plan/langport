from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# Baize default template
baize = ConversationSettings(
    name="baichuan",
    roles=(" <reserved_102> ", " <reserved_103> "),
    sep_style=SeparatorStyle.NO_COLON_TWO,
    sep="",
    sep2="</s>",
    stop_token_ids=[2, 195],
)