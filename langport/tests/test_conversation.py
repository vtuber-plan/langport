
import unittest

from langport.data.conversation.conversation_settings import ConversationHistory
from langport.data.conversation.settings.baichuan import baichuan
from langport.data.conversation.settings.chatglm import chatglm
from langport.data.conversation.settings.chatgpt import chatgpt
from langport.data.conversation.settings.openbuddy import openbuddy
from langport.data.conversation.settings.qwen import qwen

class TestBaiChuanMethods(unittest.TestCase):

    def test_conv(self):
        history = ConversationHistory(
            "SYSTEM_MESSAGE",
            messages=[
                (baichuan.roles[0], "aaa"),
                (baichuan.roles[1], "bbb"),
            ],
            offset=0,
            settings=baichuan
        )
        self.assertEqual(history.get_prompt(), "SYSTEM_MESSAGE <reserved_102> aaa <reserved_103> bbb</s>")


class TestChatGLMMethods(unittest.TestCase):

    def test_conv(self):
        history = ConversationHistory(
            "SYSTEM_MESSAGE",
            messages=[
                (chatglm.roles[0], "aaa"),
                (chatglm.roles[1], "bbb"),
            ],
            offset=0,
            settings=chatglm
        )
        self.assertEqual(history.get_prompt(), "SYSTEM_MESSAGE\n\n[Round 1]\n\n问：aaa\n\n答：bbb\n\n")

class TestChatGPTMethods(unittest.TestCase):

    def test_conv(self):
        history = ConversationHistory(
            "SYSTEM_MESSAGE",
            messages=[
                (chatgpt.roles[0], "aaa"),
                (chatgpt.roles[1], "bbb"),
            ],
            offset=0,
            settings=chatgpt
        )
        self.assertEqual(history.get_prompt(), "SYSTEM_MESSAGE\n### user: aaa\n### assistant: bbb\n### ")


class TestOpenbuddyMethods(unittest.TestCase):

    def test_conv(self):
        history = ConversationHistory(
            "SYSTEM_MESSAGE",
            messages=[
                (openbuddy.roles[0], "aaa"),
                (openbuddy.roles[1], "bbb"),
            ],
            offset=0,
            settings=openbuddy
        )
        self.assertEqual(history.get_prompt(), """SYSTEM_MESSAGE

User: aaa
Assistant: bbb
""")

class TestQwenMethods(unittest.TestCase):

    def test_conv(self):
        history = ConversationHistory(
            "SYSTEM_MESSAGE",
            messages=[
                (qwen.roles[0], "aaa"),
                (qwen.roles[1], "bbb"),
            ],
            offset=0,
            settings=qwen
        )
        self.assertEqual(history.get_prompt(), """<|im_start|>system
SYSTEM_MESSAGE<|im_end|>
<|im_start|>user
aaa<|im_end|>
<|im_start|>assistant
bbb<|im_end|>
""")

if __name__ == '__main__':
    unittest.main()