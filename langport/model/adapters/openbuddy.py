import os
from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings
from langport.model.model_adapter import BaseAdapter

class OpenBuddyAdapter(BaseAdapter):
    """The model adapter for OpenBuddy"""

    def match(self, model_path: str):
        if model_path.endswith('/'):
            model_path = model_path[:-1]
        dir_name = os.path.basename(model_path)
        parent_name = os.path.dirname(model_path)
        return dir_name.startswith('openbuddy') or parent_name.startswith('openbuddy')

    def get_default_conv_template(self, model_path: str) -> ConversationHistory:
        settings = get_conv_settings("openbuddy")
        return ConversationHistory(
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
            messages=[],
            offset=0,
            settings=settings,
        )
