from langport.data.conversation import (
    Conversation,
    SeparatorStyle,
)


# Phoenix default template
phoenix = Conversation(
    name="phoenix",
    system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PHOENIX,
    sep="</s>",
)