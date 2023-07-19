from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# FireFly default template
firefly = ConversationSettings(
    name="firefly",
    roles=("", ""),
    sep_style=SeparatorStyle.NO_COLON_SINGLE,
    sep="</s>",
    stop_str="</s>",
)