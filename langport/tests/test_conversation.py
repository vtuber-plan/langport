
from typing import List
import unittest

from chatproto.conversation.history import ConversationHistory
from chatproto.conversation.models.baichuan import baichuan2
from chatproto.conversation.models.chatglm import chatglm
from chatproto.conversation.models.chatgpt import chatgpt
from chatproto.conversation.models.llama import llama
from chatproto.conversation.models.openbuddy import openbuddy
from chatproto.conversation.models.qwen import qwen
from chatproto.conversation.models.starchat import starchat

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


class TestStarChatMethods(unittest.TestCase):

    def test_conv(self):
        history = ConversationHistory(
            "SYSTEM_MESSAGE",
            messages=[
                (starchat.roles[0], "aaa"),
                (starchat.roles[1], "bbb"),
            ],
            offset=0,
            settings=starchat
        )
        self.assertEqual(history.get_prompt(), "<|system|>\nSYSTEM_MESSAGE<|end|>\n<|user|>\naaa<|end|>\n<|assistant|>\nbbb<|end|>\n")
    
    def test_conv_question(self):
        history = ConversationHistory(
            "SYSTEM_MESSAGE",
            messages=[
                (starchat.roles[0], "aaa"),
                (starchat.roles[1], None),
            ],
            offset=0,
            settings=starchat
        )
        self.assertEqual(history.get_prompt(), "<|system|>\nSYSTEM_MESSAGE<|end|>\n<|user|>\naaa<|end|>\n<|assistant|>\n")

if __name__ == '__main__':
    unittest.main()