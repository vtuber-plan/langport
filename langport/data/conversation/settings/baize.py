from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# Baize default template
baize = ConversationSettings(
    name="baize",
    roles=("[|Human|]", "[|AI|]"),
    sep_style=SeparatorStyle.NO_COLON_SINGLE,
    sep="[|Human|]",
    stop_str="[|Human|]",
)