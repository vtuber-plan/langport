
from typing import List
import unittest

from langport.data.conversation.conversation_settings import ConversationHistory
from langport.data.conversation.settings.baichuan import baichuan
from langport.data.conversation.settings.chatglm import chatglm
from langport.data.conversation.settings.chatgpt import chatgpt
from langport.data.conversation.settings.llama import llama
from langport.data.conversation.settings.openbuddy import openbuddy
from langport.data.conversation.settings.qwen import qwen
from langport.data.conversation.settings.starchat import starchat

class TestLlamaMethods(unittest.TestCase):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
    UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

    def get_llama_prompt(self, dialogs):
        unsafe_requests = []
        prompt_tokens = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in self.SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": self.B_SYS
                        + dialog[0]["content"]
                        + self.E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: str = "".join([
                f"{self.B_INST} {(prompt['content']).strip()} {self.E_INST} {(answer['content']).strip()} "
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
            ])
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += f"{self.B_INST} {(dialog[-1]['content']).strip()} {self.E_INST}"
            prompt_tokens.append(dialog_tokens)
        return prompt_tokens

    def test_conv(self):
        history = ConversationHistory(
            "SYSTEM_MESSAGE",
            messages=[
                (llama.roles[0], "aaa"),
                (llama.roles[1], "bbb"),
                (llama.roles[0], "ccc"),
            ],
            offset=0,
            settings=llama
        )
        my_out = history.get_prompt()
        llama_out = self.get_llama_prompt([[
            {"role": "system", "content": "SYSTEM_MESSAGE"},
            {"role": llama.roles[0], "content": "aaa"},
            {"role": llama.roles[1], "content": "bbb"},
            {"role": llama.roles[0], "content": "ccc"},
        ]])[0]
        self.assertEqual(my_out, llama_out)
    
    def test_conv2(self):
        history = ConversationHistory(
            "SYSTEM_MESSAGE",
            messages=[
                (llama.roles[0], "aaa"),
                (llama.roles[1], "bbb"),
                (llama.roles[0], "ccc"),
                (llama.roles[1], None),
            ],
            offset=0,
            settings=llama
        )
        my_out = history.get_prompt()
        llama_out = self.get_llama_prompt([[
            {"role": "system", "content": "SYSTEM_MESSAGE"},
            {"role": llama.roles[0], "content": "aaa"},
            {"role": llama.roles[1], "content": "bbb"},
            {"role": llama.roles[0], "content": "ccc"},
        ]])[0]
        print(my_out)
        print(llama_out)
        self.assertEqual(my_out, llama_out)


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