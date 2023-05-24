from langport.data.conversation import (
    Conversation,
    SeparatorStyle,
)

# OpenAssistant Pythia default template
oasst_pythia = Conversation(
    name="oasst_pythia",
    system="",
    roles=("<|prompter|>", "<|assistant|>"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.NO_COLON_SINGLE,
    sep="<|endoftext|>",
)
