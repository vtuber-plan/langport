from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# Mistral default template
mistral = ConversationSettings(
    name="mistral",
    system_template="[INST]{system_message}\n",
    roles=("[INST]", "[/INST]"),
    sep_style=SeparatorStyle.LLAMA,
    sep=" ",
    sep2="</s>",
)
