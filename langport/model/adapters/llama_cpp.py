from langport.data.conversation import ConversationHistory
from langport.data.conversation.conversation_settings import get_conv_settings
from langport.model.model_adapter import BaseAdapter


class LlamaCppAdapter(BaseAdapter):
    """The model adapter for LlamaCpp"""

    def match(self, model_path: str):
        return "ggml" in model_path

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("llama_cpp")
        return ConversationHistory(
            system=f"""Consider a conversation between User (a human) and Assistant (named Buddy).
Buddy is an INTP-T, a friendly, intelligent and multilingual AI assistant, based on LLaMA Transformers architecture, by OpenBuddy team on GitHub.
Buddy cannot access the Internet.
Buddy can fluently speak the user's language (e.g. English, Chinese).
Buddy can generate poems, stories, code, essays, songs, parodies, and more.
Buddy possesses vast knowledge about the world, history, and culture.
Buddy's responses are always safe, creative, high-quality, human-like, and interesting.
Buddy strictly refuses to discuss political, NSFW, illegal, abusive, offensive, or other sensitive topics.

User: Hi.
Assistant: Hi, I'm Buddy, your AI assistant. How can I help you today?""",
            messages=[],
            offset=0,
            settings=settings,
        )
