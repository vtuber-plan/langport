from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


one_shot = ConversationSettings(
    name="one_shot",
    roles=("Human", "Assistant"),
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n### ",
    stop_str="###",
)