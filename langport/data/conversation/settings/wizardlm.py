from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# Vicuna v1.1 template
wizardlm = ConversationSettings(
    name="wizardlm",
    roles=("USER", "ASSISTANT"),
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep=" ",
)