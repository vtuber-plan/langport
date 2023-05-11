# A template with one conversation example
from langport.data.conversation import (
    Conversation,
    SeparatorStyle,
    register_conv_template,
)

one_shot = Conversation(
    name="one_shot",
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        (
            "Human",
            "What are the key differences between renewable and non-renewable energy sources?",
        ),
        (
            "Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.",
        ),
    ),
    offset=2,
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n### ",
    stop_str="###",
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


# Koala default template
koala_v1 = Conversation(
    name="koala_v1",
    system="BEGINNING OF CONVERSATION:",
    roles=("USER", "GPT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_TWO,
    sep=" ",
    sep2="</s>",
)

# Dolly V2 default template
dolly_v2 = Conversation(
    name="dolly_v2",
    system="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
    roles=("### Instruction", "### Response"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.DOLLY,
    sep="\n\n",
    sep2="### End",
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


# StableLM Alpha default template
stablelm = Conversation(
    name="stablelm",
    system="""<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
""",
    roles=("<|USER|>", "<|ASSISTANT|>"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.NO_COLON_SINGLE,
    sep="",
    stop_token_ids=[50278, 50279, 50277, 1, 0],
)


# Baize default template
baize = Conversation(
    name="baize",
    system="The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.",
    roles=("[|Human|]", "[|AI|]"),
    messages=(
        ("[|Human|]", "Hello!"),
        ("[|AI|]", "Hi!"),
    ),
    offset=2,
    sep_style=SeparatorStyle.BAIZE,
    sep="[|Human|]",
    stop_str="[|Human|]",
)

# RWKV-4-Raven default template
rwkv = Conversation(
    name="rwkv",
    system="",
    roles=("Bob", "Alice"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.RWKV,
    sep="",
    stop_str="\n\n",
)

# Buddy default template
openbuddy = Conversation(
    name="openbuddy",
    system="""Consider a conversation between User (a human) and Assistant (named Buddy).
Buddy is an INTP-T, a friendly, intelligent and multilingual AI assistant, by OpenBuddy team. GitHub: https://github.com/OpenBuddy/OpenBuddy
Buddy cannot access the Internet.
Buddy can fluently speak the user's language (e.g. English, Chinese).
Buddy can generate poems, stories, code, essays, songs, parodies, and more.
Buddy possesses vast knowledge about the world, history, and culture.
Buddy's responses are always safe, creative, high-quality, human-like, and interesting.
Buddy strictly refuses to discuss political, NSFW, or other unsafe topics.

User: Hi.
Assistant: Hi, I'm Buddy, your AI assistant. How can I help you today?""",
    roles=("User", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n",
)


# Phoenix default template
phoenix = Conversation(
    name="phoenix",
    system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PHOENIX,
    sep="</s>",
)

# ChatGPT default template
chatgpt = Conversation(
    name="chatgpt",
    system="You are a helpful assistant.",
    roles=("user", "assistant"),
    messages=(),
    offset=0,
    sep_style=None,
    sep=None,
)


# Claude default template
claude = Conversation(
    name="claude",
    system="",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n\n",
)
