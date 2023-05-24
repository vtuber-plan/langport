from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)

# OpenAssistant Pythia default template
oasst_pythia = ConversationSettings(
    name="oasst_pythia",
    roles=("<|prompter|>", "<|assistant|>"),
    sep_style=SeparatorStyle.NO_COLON_SINGLE,
    sep="<|endoftext|>",
)
