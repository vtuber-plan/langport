from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# Claude default template
claude = ConversationSettings(
    name="claude",
    roles=("Human", "Assistant"),
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n\n",
)
