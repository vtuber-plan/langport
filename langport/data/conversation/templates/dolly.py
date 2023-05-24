from langport.data.conversation import (
    Conversation,
    SeparatorStyle,
)


# Dolly V2 default template
dolly_v2 = Conversation(
    name="dolly_v2",
    system="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
    roles=("### Instruction", "### Response"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.DOLLY,
    sep="\n\n",
    sep2="### End",
)