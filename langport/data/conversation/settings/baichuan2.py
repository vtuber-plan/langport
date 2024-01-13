from langport.data.conversation import (
    ConversationSettings,
    SeparatorStyle,
)

"""
https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/blob/main/modeling_baichuan.py#L555
"""
# Baichuan default template
baichuan = ConversationSettings(
    name="baichuan2",
    roles=("<reserved_106>", "<reserved_107>"),
    sep_style=SeparatorStyle.NO_COLON_SINGLE,
    sep="",
    stop_token_ids=[2, 195],
)