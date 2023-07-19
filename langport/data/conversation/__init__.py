"""
Conversation prompt templates.
From FastChat: https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
"""

import dataclasses
from enum import auto, Enum
from typing import List, Any, Optional, Tuple


class SeparatorStyle(Enum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    ADD_NEW_LINE_SINGLE = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    CHATGLM = auto()
    LLAMA = auto()


@dataclasses.dataclass
class ConversationSettings:
    # The name of this settings
    name: str
    # Two roles
    roles: List[str]
    # Separators
    sep_style: SeparatorStyle
    sep: str
    sep2: Optional[str] = None
    system_sep: Optional[str] = None
    round_sep: Optional[str] = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def copy(self):
        return ConversationSettings(
            name=self.name,
            roles=self.roles,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            system_sep=self.system_sep,
            round_sep=self.round_sep,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )


@dataclasses.dataclass
class ConversationHistory:
    """A class that keeps all conversation history."""

    # System prompts
    system: str
    # All messages
    messages: List[Tuple[str, str]]
    # Offset of few shot examples
    offset: int

    settings: ConversationSettings
    
    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        if self.settings.round_sep is not None:
            round_sep = self.settings.round_sep
        else:
            round_sep = ""
        if self.settings.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            if self.settings.system_sep is not None:
                ret = self.system + self.settings.system_sep
            else:
                ret = self.system + self.settings.sep
            
            for i, (role, message) in enumerate(self.messages):
                if i % len(self.settings.roles) == 0:
                    ret += round_sep
                if message:
                    ret += role + ": " + message + self.settings.sep
                else:
                    ret += role + ": "
            return ret
        elif self.settings.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.settings.sep, self.settings.sep2]
            if self.settings.system_sep is not None:
                ret = self.system + self.settings.system_sep
            else:
                ret = self.system + seps[0]
            
            for i, (role, message) in enumerate(self.messages):
                if i % len(self.settings.roles) == 0:
                    ret += round_sep
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ": "
            return ret
        elif self.settings.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            if self.settings.system_sep is not None:
                ret = self.system + self.settings.system_sep
            else:
                ret = self.system
            
            for i, (role, message) in enumerate(self.messages):
                if i % len(self.settings.roles) == 0:
                    ret += round_sep
                if message:
                    ret += role + message + self.settings.sep
                else:
                    ret += role
            return ret
        elif self.settings.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            if self.settings.system_sep is not None:
                ret = self.system + self.settings.system_sep
            else:
                ret = self.system + self.settings.sep

            for i, (role, message) in enumerate(self.messages):
                if i % len(self.settings.roles) == 0:
                    ret += round_sep
                
                if message:
                    ret += role + "\n" + message + self.settings.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.settings.sep_style == SeparatorStyle.DOLLY:
            seps = [self.settings.sep, self.settings.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.settings.sep_style == SeparatorStyle.RWKV:
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ": "
                        + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.settings.sep_style == SeparatorStyle.PHOENIX:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.settings.sep_style == SeparatorStyle.CHATGLM:
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i % 2 == 0:
                        ret += f"[Round {i+1}]\n\n"
                    ret += role + "：" + message + self.settings.sep
                else:
                    ret += role + "："
            return ret
        elif self.settings.sep_style == SeparatorStyle.LLAMA:
            B_INST, E_INST = "[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
            system_q = B_SYS + self.system + E_SYS
            system_a = ""
            ret = f"{B_INST} {system_q} {E_INST} {system_a}"
            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    inst = B_INST
                else:
                    inst = E_INST
                if message:
                    ret += inst + " " + message.strip() + " "
                else:
                    ret += inst + " "
            return ret
        else:
            raise ValueError(f"Invalid style: {self.settings.sep_style}")

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        """Convert the history to gradio chatbot format"""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return ConversationHistory(
            system=self.system,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            settings=self.settings
        )

    def dict(self):
        return {
            "system": self.system,
            "messages": self.messages,
            "offset": self.offset,
            "settings": self.settings,
        }

