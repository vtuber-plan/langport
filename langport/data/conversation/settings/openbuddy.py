from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)

# Buddy default template
openbuddy = ConversationSettings(
    name="openbuddy",
    roles=("User", "Assistant"),
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    system_template="{system_message}\n",
    sep="\n",
    stop_str="\n</s>",
)
