from langport.data.conversation import (
    Conversation,
    SeparatorStyle,
)


# Claude default template
claude = Conversation(
    name="claude",
    system="",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n\n",
)
