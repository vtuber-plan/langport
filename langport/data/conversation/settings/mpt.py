from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# MPT default template
mpt = ConversationSettings(
    name="mpt",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
    sep="<|im_end|>\n",
    stop_token_ids=[50278, 0],
)
