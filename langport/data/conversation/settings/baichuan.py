from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)

"""
https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/modeling_baichuan.py#L555
"""
# Baichuan default template
baichuan = ConversationSettings(
    name="baichuan",
    roles=(" <reserved_102> ", " <reserved_103> "),
    sep_style=SeparatorStyle.NO_COLON_TWO,
    sep="",
    sep2="</s>",
    stop_token_ids=[2, 195],
)