from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


one_shot = ConversationSettings(
    name="qwen",
    roles=("user", "assistant"),
    sep_style=SeparatorStyle.CHATLM,
    sep="\n",
    stop_str="<|im_end|>",
)