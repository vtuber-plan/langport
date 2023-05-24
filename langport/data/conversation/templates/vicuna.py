from langport.data.conversation import (
    Conversation,
    SeparatorStyle,
)


# Vicuna v1.1 template
vicuna_v1_1 = Conversation(
    name="vicuna_v1.1",
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_TWO,
    sep=" ",
    sep2="</s>",
)