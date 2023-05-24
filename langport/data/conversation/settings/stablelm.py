from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# StableLM Alpha default template
stablelm = ConversationSettings(
    name="stablelm",
    roles=("<|USER|>", "<|ASSISTANT|>"),
    sep_style=SeparatorStyle.NO_COLON_SINGLE,
    sep="",
    stop_token_ids=[50278, 50279, 50277, 1, 0],
)
