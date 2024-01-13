
from typing import List, Optional
from chatproto.registry import get_conv_settings
from langport.model.executor.base import RemoteModelExecutor
from langport.model.executor.generation import GenerationExecutor
import openai
from langport.protocol.worker_protocol import BaseWorkerResult, GenerationWorkerResult

from langport.workers.generation_worker import GenerationModelWorker

import tiktoken
chatgpt_tokenizer = tiktoken.get_encoding('cl100k_base')
p50k_base_tokenizer = tiktoken.get_encoding('p50k_base')

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    encoding = chatgpt_tokenizer

    if model == "gpt-3.5-turbo":
        # print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        # print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

class ChatGPTGenerationExecutor(RemoteModelExecutor):
    def __init__(
        self,
        model_name: str,
        api_url: str,
        api_key: str,
    ) -> None:
        super(ChatGPTGenerationExecutor, self).__init__(
            model_name=model_name,
            api_url=api_url,
            api_key=api_key,
        )

        openai.api_base = api_url
        openai.api_key = api_key
        
        self._context_len = 2048

    @property
    def context_length(self) -> int:
        return self._context_len

    def tokenize(self, text: str) -> List[int]:
        return chatgpt_tokenizer.encode(text)
    
    def inference(self, worker: "GenerationModelWorker"):
        if not worker.online:
            return

        tasks = worker.fetch_tasks()
        batch_size = len(tasks)
        if batch_size == 0:
            return
        
        settings = get_conv_settings("chatgpt")

        for task in tasks:
            prompt = task.prompt
            sections = prompt.split(settings.sep)
            messages = []
            for i, section in enumerate(sections):
                if i == 0:
                    role = "system"
                else:
                    if i % 2 == 1:
                        role = "user"
                    else:
                        role = "assistant"
                messages.append({
                    "role": role,
                    "content": section
                })
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                max_tokens=self.context_length,
            )

            text_data = {}
            for chunk in response:
                choice = chunk.choices[0]
                if "content" not in choice.delta:
                    continue
                if task.task_id not in text_data:
                    text_data[task.task_id] = ""
                else:
                    text_data[task.task_id] += str(choice.delta.content)
                worker.push_task_result(task.task_id, GenerationWorkerResult(
                    task_id=task.task_id,
                    type="data",
                    text=text_data[task.task_id],
                    finish_reason=None,
                ))

        for task in tasks:
            worker.push_task_result(
                task.task_id, BaseWorkerResult(task_id=task.task_id, type="done")
            )