from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# Dolly V2 default template
dolly_v2 = ConversationSettings(
    name="dolly_v2",
    roles=("### Instruction", "### Response"),
    sep_style=SeparatorStyle.DOLLY,
    sep="\n\n",
    sep2="### End",
)