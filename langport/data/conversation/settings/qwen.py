from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)

"""
https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen_generation_utils.py#L119
"""
qwen = ConversationSettings(
    name="qwen",
    roles=("user", "assistant"),
    sep_style=SeparatorStyle.CHATLM,
    system_template="<|im_start|>system\n{system_message}<|im_end|>",
    sep="\n",
    stop_str="<|im_end|>",
)
