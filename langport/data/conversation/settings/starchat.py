from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)


# StarChat default template
starchat = ConversationSettings(
    name="starchat",
    system_template="<|system|>\n{system_message}",
    roles=("<|user|>", "<|assistant|>"),
    sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
    sep="<|end|>\n",
    stop_token_ids=[0, 49155],
    stop_str="<|end|>",
)
