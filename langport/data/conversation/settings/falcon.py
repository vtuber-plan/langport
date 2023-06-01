from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# Falcon default template
falcon = ConversationSettings(
    name="falcon",
    roles=("[|Human|]", "[|AI|]"),
    sep_style=SeparatorStyle.NO_COLON_SINGLE,
    sep="\n",
    stop_str="[|Human|]",
    stop_token_ids=[193],
)