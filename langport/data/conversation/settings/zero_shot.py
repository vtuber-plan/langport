
from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)

zero_shot = ConversationSettings(
    name="zero_shot",
    roles=("Human", "Assistant"),
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n### ",
    stop_str="###",
)
