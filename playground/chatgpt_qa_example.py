import json
import traceback
import openai

openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://localhost:8000/v1"


def chat(model: str = "CodeLlama-7b-Instruct-hf", stream: bool = False, max_tokens: int = 1024):
    messages = [
        {"role":"user","content":"Hello! Who are you?"}
    ]

    # create a chat completion
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        stream=stream,
        max_tokens=max_tokens,
    )
    # print the completion
    try:
        if stream:
            out = ""
            for chunk in response:
                out += str(chunk)
        else:
            out = response.choices[0].message.content
            total_tokens = response.usage.total_tokens
            completion_tokens = response.usage.completion_tokens
    except Exception as e:
        traceback.print_exc()
    
    return out

out = chat()
print(out)