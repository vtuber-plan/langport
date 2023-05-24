from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)

# Buddy default template
openbuddy = ConversationSettings(
    name="openbuddy",
    roles=("User", "Assistant"),
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n",
)
