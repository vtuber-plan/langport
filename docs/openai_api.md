OpenAI-compatible API Gateway
===

LangPort provides OpenAI-compatible APIs for its supported models.
The following OpenAI APIs are supported:
- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)
- Embeddings. (Reference: https://platform.openai.com/docs/api-reference/embeddings)

### Curl Test
```shell
curl -X 'POST' \
  'http://localhost:8000/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openbuddy",
    "stream": true,
    "top_p": 0.5,
    "temperature": 0.5,
    "messages": [
        {
            "role": "user",
            "content": "Hello! What is your name?"
        }
    ]
}'
```

### OpenAI Official SDK
The goal of `openai_api_gateway.py` is to implement a fully OpenAI-compatible API server, so the models can be used directly with [openai-python](https://github.com/openai/openai-python) library.

First, install openai-python:
```bash
pip install --upgrade openai
```

Then, interact with model vicuna:
```python
import openai
openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://localhost:8000/v1"

model = "vicuna-7b-v1.1"
prompt = "Once upon a time"

# create a completion
completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=64)
# print the completion
print(prompt + completion.choices[0].text)

# create a chat completion
completion = openai.ChatCompletion.create(
  model=model,
  messages=[{"role": "user", "content": "Hello! What is your name?"}]
)
# print the completion
print(completion.choices[0].message.content)
```


### Completions
Here we list the parameter compatibility of completions API.

|    Parameter    | LangPort | OpenAI | Default Value | Maximum Value |
|       ---       |   ---    | --- | --- | --- |
| `model`         | ● | ● | - | - |
| `prompt`        | ● | ● | `""` | `COMPLETION_MAX_PROMPT` |
| `suffix`        | ○ | ● | - | - |
| `min_tokens`    | ○ | ○ | `0` | `COMPLETION_MAX_TOKENS` |
| `max_tokens`    | ● | ● | `4096` | `COMPLETION_MAX_TOKENS` |
| `temperature`   | ● | ● | `1.0` | - |
| `top_p`         | ● | ● | `1.0` | - |
| `n`             | ● | ● | `1` | `COMPLETION_MAX_N` |
| `stream`        | ● | ● | `false` | - |
| `logprobs`      | ○ | ● | `0` | `COMPLETION_MAX_LOGPROBS` |
| `echo`          | ● | ● | `false` | - |
| `stop`          | ● | ● | - | - |
| `presence_penalty`  | ○ | ● | - | - |
| `frequency_penalty` | ○ | ● | - | - |
| `best_of`       | ○ | ● | - | - |
| `logit_bias`    | ○ | ● | - | - |
| `user`          | ○ | ● | - | - |


## Roadmap

- [x] API
    - [x] Models
        - [x] List models
        - [x] Retrieve model
    - [x] Embeddings
        - [x] Create embeddings
    - [x] Completions
        - [x] Create completion
    - [x] Chat
        - [x] Create chat completion
    - [ ] Authentication
        - [ ] Forward key
- [x] Model
    - [x] Architectures
        - [x] Encoder-only
        - [x] Encoder-decoder
        - [x] Decoder-only
    - [x] Decoding strategies
        - [x] Random sampling with temperature
        - [x] Nucleus-sampling (top-p)
        - [x] Stop sequences
        - [x] Presence and frequency penalties
