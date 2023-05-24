from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


billa = ConversationSettings(
    name="billa",
    roles=("Human", "Assistant"),
    sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
    sep="\n",
    stop_str="Human:",
)