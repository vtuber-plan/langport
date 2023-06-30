from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)

# Chatglm default template
chatglm = ConversationSettings(
    name="chatglm",
    roles=("问", "答"),
    sep_style=SeparatorStyle.CHATGLM,
    sep="\n\n",
    stop_str="\n\n",
)