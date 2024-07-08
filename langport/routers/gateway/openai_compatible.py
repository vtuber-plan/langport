import asyncio

import asyncio
import base64
import json

from typing import AsyncGenerator, Coroutine, Generator, Optional, Union, Dict, List, Any

from fastapi.responses import StreamingResponse
import httpx
import shortuuid

import numpy as np

from langport.constants import WORKER_API_TIMEOUT, ErrorCode
from langport.model.model_adapter import get_conversation_template
from langport.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    CompletionLogprobs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    EmbeddingsData,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
)
from langport.protocol.worker_protocol import (
    BaseWorkerResult,
    EmbeddingWorkerResult,
    GenerationWorkerResult,
)
from langport.core.dispatch import DispatchMethod
from langport.routers.gateway.common import (
    LANGPORT_HEADER,
    AppSettings,
    _get_worker_address,
    _list_models,
    check_model,
    check_requests,
    create_bad_request_response
)

def clean_system_prompts(messages: List[Dict[str, str]]):
    system_prompt = ""
    result = []
    for i, message in enumerate(messages):
        if len(system_prompt) == 0 and message["role"] == "system":
            system_prompt = message["content"]
            continue
        if message["role"] in ["user", "assistant"]:
            result.append(message)
    # system_prompt = system_prompt.rstrip("\n")
    result.insert(0, {"role": "system", "content": system_prompt})
    return result

def get_gen_params(
    model_name: str,
    messages: Union[str, List[Dict[str, str]]],
    *,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    echo: Optional[bool],
    stream: Optional[bool],
    stop: Optional[Union[str, List[str]]],
    logprobs: Optional[int]=None,
    presence_penalty: Optional[float]=0.0,
    frequency_penalty: Optional[float]=0.0,
) -> Dict[str, Any]:
    # is_chatglm = "chatglm" in model_name.lower()
    conv = get_conversation_template(model_name)
    if isinstance(messages, str):
        prompt = messages
    else:
        clean_messages = clean_system_prompts(messages)
        for message in clean_messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.settings.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.settings.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.settings.roles[1], None)
        prompt = conv.get_prompt()

    if max_tokens is None:
        max_tokens = 512

    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "echo": echo,
        "stream": stream,
        "logprobs": logprobs,
        "stop_token_ids": conv.settings.stop_token_ids,
    }

    stop_str = []
    conv_stop_str = conv.settings.stop_str
    if isinstance(conv_stop_str, str):
        stop_str.append(conv_stop_str)
    elif isinstance(conv_stop_str, list) or isinstance(conv_stop_str, tuple):
        if len(conv_stop_str) > 0 and not isinstance(conv_stop_str[0], str):
            raise Exception("The type of stop_str shall be str or list of str")
        stop_str = conv_stop_str
    else:
        raise Exception("The type of stop_str shall be str or list of str")
    print(stop_str)
    if stop is None:
        gen_params.update(
            {"stop": stop_str}
        )
    elif isinstance(stop, str):
        gen_params.update({"stop": [stop,] + stop_str})
    elif isinstance(stop, list) or isinstance(stop, tuple):
        gen_params.update({"stop": stop + stop_str})
    else:
        raise Exception(f"The type of stop shall be str or list of str, the type of stop is {type(stop)}")
    
    if presence_penalty is not None:
        gen_params["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        gen_params["frequency_penalty"] = frequency_penalty

    return gen_params

async def api_models(app_settings: AppSettings):
    async with httpx.AsyncClient() as client:
        generation_models = await _list_models(app_settings, "generation", client)
    async with httpx.AsyncClient() as client:
        embedding_models = await _list_models(app_settings, "embedding", client)
    models = generation_models + embedding_models
    models.sort()
    # TODO: return real model permission details
    model_cards = []
    for m in models:
        model_cards.append(ModelCard(id=m, root=m, permission=[ModelPermission()]))
    return ModelList(data=model_cards)

async def generate_completion_stream_generator(app_settings: AppSettings, payload: Dict[str, Any], n: int):
    model_name = payload["model"]
    id = f"cmpl-{shortuuid.random()}"
    finish_stream_events = []

    for i in range(n):
        previous_text = ""
        async for content in generate_completion_stream(app_settings, "/completion_stream", payload):
            if content.error_code != ErrorCode.OK:
                yield f"data: {json.dumps(content.dict(), ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            decoded_unicode = content.text.replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text) :]
            previous_text = decoded_unicode

            if content.logprobs is None:
                logprobs = None
            else:
                logprobs = CompletionLogprobs.parse_obj(content.logprobs.dict())
            choice_data = CompletionResponseStreamChoice(
                index=i,
                text=delta_text,
                logprobs=logprobs,
                finish_reason=content.finish_reason,
            )
            chunk = CompletionStreamResponse(
                id=id, object="text_completion", choices=[choice_data], model=model_name
            )
            if len(delta_text) == 0:
                if content.finish_reason is not None:
                    finish_stream_events.append(chunk)
                continue
            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


async def generate_completion_stream(app_settings: AppSettings, url: str, payload: Dict[str, Any]) -> AsyncGenerator[GenerationWorkerResult, None]:
    async with httpx.AsyncClient() as client:
        try:
            worker_addr = await _get_worker_address(app_settings, payload["model"], "generation", client, DispatchMethod.LOTTERY)
        except:
            yield BaseWorkerResult(
                type="error",
                message=f"No available worker running {payload['model']} for generation",
                error_code=ErrorCode.INVALID_MODEL
            )
            return

        delimiter = b"\0"
        try:
            async with client.stream(
                "POST",
                worker_addr + url,
                headers=LANGPORT_HEADER,
                json=payload,
                timeout=WORKER_API_TIMEOUT,
            ) as response:
                async for raw_chunk in response.aiter_raw():
                    for chunk in raw_chunk.split(delimiter):
                        if not chunk:
                            continue
                        try:
                            data = json.loads(chunk.decode())
                        except json.JSONDecodeError:
                            yield BaseWorkerResult(
                                type="error",
                                message=chunk.decode(),
                                error_code=ErrorCode.ENGINE_OVERLOADED
                            )
                            break
                        if data["type"] == "error":
                            yield BaseWorkerResult.parse_obj(data)
                            break
                        yield GenerationWorkerResult.parse_obj(data)
        except httpx.ReadTimeout:
            yield BaseWorkerResult(
                type="error",
                message="Server is overloading",
                error_code=ErrorCode.ENGINE_OVERLOADED
            )
        

async def single_completions_non_stream(app_settings: AppSettings, payload: Dict[str, Any]) -> Optional[GenerationWorkerResult]:
    ret = None
    async for content in generate_completion_stream(app_settings, "/completion_stream", payload):
        ret = content
    return ret

async def chat_completion_stream_generator(
    app_settings: AppSettings, payload: Dict[str, Any], n: int
) -> AsyncGenerator[str, None]:
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    id = f"chatcmpl-{shortuuid.random()}"
    finish_stream_events = []
    for i in range(n):
        # First chunk with role
        choice_data = ChatCompletionResponseStreamChoice(
            index=i,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None,
        )
        chunk = ChatCompletionStreamResponse(
            id=id, choices=[choice_data], model=payload["model"]
        )
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        previous_text = ""
        async for content in generate_completion_stream(app_settings, "/chat_stream", payload):
            if content.error_code != ErrorCode.OK:
                yield f"data: {json.dumps(content.dict(), ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            decoded_unicode = content.text.replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text) :]
            previous_text = decoded_unicode

            if len(delta_text) == 0:
                delta_text = None
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(content=delta_text),
                finish_reason=content.finish_reason,
            )
            chunk = ChatCompletionStreamResponse(
                id=id, choices=[choice_data], model=payload["model"]
            )
            if delta_text is None:
                if content.finish_reason is not None:
                    finish_stream_events.append(chunk)
                continue
            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_none=True, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


async def single_chat_completions_non_stream(
    app_settings: AppSettings, payload: Dict[str, Any]
) -> Optional[GenerationWorkerResult]:
    ret = None
    async for content in generate_completion_stream(app_settings, "/chat_stream", payload):
        ret = content
    return ret


async def get_embedding(app_settings: AppSettings, payload: Dict[str, Any]) -> EmbeddingWorkerResult:
    model_name = payload["model"]
    async with httpx.AsyncClient() as client:
        worker_addr = await _get_worker_address(app_settings, model_name, "embedding", client, DispatchMethod.LOTTERY)

        response = await client.post(
            worker_addr + "/embeddings",
            headers=LANGPORT_HEADER,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
        )
        if response.json()["type"] == "error":
            error_message = BaseWorkerResult.parse_obj(response.json())
            return error_message
        return EmbeddingWorkerResult.parse_obj(response.json())


async def completions_stream(app_settings: AppSettings, payload: Dict[str, Any], request: CompletionRequest):
    generator = generate_completion_stream_generator(app_settings, payload, request.n)
    return StreamingResponse(generator, media_type="text/event-stream")

async def completions_non_stream(app_settings: AppSettings, payload: Dict[str, Any], request: CompletionRequest):
    completions = []
    for i in range(request.n):
        content = asyncio.create_task(single_completions_non_stream(app_settings, payload))
        completions.append(content)

    choices = []
    usage = UsageInfo()
    for i, content_task in enumerate(completions):
        content = await content_task
        if content.error_code != ErrorCode.OK:
            return create_bad_request_response(content.error_code, content.message)
        if content.logprobs is None:
            logprobs = None
        else:
            logprobs = CompletionLogprobs.parse_obj(content.logprobs.dict())
        choices.append(
            CompletionResponseChoice(
                index=i,
                text=content.text,
                logprobs=logprobs,
                finish_reason=content.finish_reason,
            )
        )
        task_usage = UsageInfo.parse_obj(content.usage)
        for usage_key, usage_value in task_usage.dict().items():
            setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return CompletionResponse(
        model=request.model, choices=choices, usage=UsageInfo.parse_obj(usage)
    )

async def chat_completions_stream(app_settings: AppSettings, payload: Dict[str, Any], request: ChatCompletionRequest):
    generator = chat_completion_stream_generator(
        app_settings, payload, request.n
    )
    return StreamingResponse(generator, media_type="text/event-stream")


async def chat_completions_non_stream(app_settings: AppSettings, payload: Dict[str, Any], request: ChatCompletionRequest):
    choices = []
    chat_completions = []
    for i in range(request.n):
        content = asyncio.create_task(single_chat_completions_non_stream(app_settings, payload))
        chat_completions.append(content)

    usage = UsageInfo()
    for i, content_task in enumerate(chat_completions):
        content = await content_task
        if content is None:
            return create_bad_request_response(ErrorCode.INTERNAL_ERROR, "Server internal error")
        if content.error_code != ErrorCode.OK:
            return create_bad_request_response(content.error_code, content.message)
        choices.append(
            ChatCompletionResponseChoice(
                index=i,
                message=ChatMessage(role="assistant", content=content.text),
                finish_reason=content.finish_reason,
            )
        )
        if content.usage is None:
            continue
        task_usage = UsageInfo.parse_obj(content.usage)
        for usage_key, usage_value in task_usage.dict().items():
            setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)


async def api_completions(app_settings: AppSettings, request: CompletionRequest):
    error_check_ret = await check_model(app_settings, request, "generation", request.model)
    if error_check_ret is not None:
        return error_check_ret
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    payload = get_gen_params(
        request.model,
        request.prompt,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        echo=request.echo,
        stream=request.stream,
        stop=request.stop,
        logprobs=request.logprobs,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
    )

    if request.stream:
        return await completions_stream(app_settings, payload, request)
    else:
        return await completions_non_stream(app_settings, payload, request)


async def api_chat_completions(app_settings: AppSettings, request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    error_check_ret = await check_model(app_settings, request, "generation", request.model)
    if error_check_ret is not None:
        return error_check_ret
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    payload = get_gen_params(
        request.model,
        request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        echo=False,
        stream=request.stream,
        stop=request.stop,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
    )
    if request.stream:
        return await chat_completions_stream(app_settings, payload, request)
    else:
        return await chat_completions_non_stream(app_settings, payload, request)

async def api_embeddings(app_settings: AppSettings, request: EmbeddingsRequest):
    """Creates embeddings for the text"""
    error_check_ret = await check_model(app_settings, request, "embedding", request.model)
    if error_check_ret is not None:
        return error_check_ret
    payload = {
        "model": request.model,
        "input": request.input,
        "dimensions": request.dimensions,
    }

    response = await get_embedding(app_settings, payload)
    if response.type == "error":
        return create_bad_request_response(ErrorCode.INTERNAL_ERROR, response.message)
    
    if request.encoding_format is None or request.encoding_format == "float":
        return EmbeddingsResponse(
            data=[EmbeddingsData(embedding=each.embedding, index=each.index) for each in response.embeddings],
            model=request.model,
            usage=UsageInfo(
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
                completion_tokens=None,
            ),
        ).dict(exclude_none=True)
    elif request.encoding_format == "base64":
        return EmbeddingsResponse(
            data=[EmbeddingsData(
                embedding=base64.b64encode(np.array(each.embedding, dtype="float32").tobytes()).decode("utf-8"),
                index=each.index
                ) for each in response.embeddings
            ],
            model=request.model,
            usage=UsageInfo(
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
                completion_tokens=None,
            ),
        ).dict(exclude_none=True)
    else:
        raise Exception("Invalid encoding_format param.")