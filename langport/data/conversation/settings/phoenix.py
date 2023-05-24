from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# Phoenix default template
phoenix = ConversationSettings(
    name="phoenix",
    roles=("Human", "Assistant"),
    sep_style=SeparatorStyle.PHOENIX,
    sep="</s>",
)