import asyncio

import argparse
import asyncio
import json
import logging

import os
import random
import traceback
from typing import Generator, Optional, Union, Dict, List, Any

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, DispatchFunction
import httpx
import numpy as np
import shortuuid
from starlette.types import ASGIApp
from tenacity import retry, stop_after_attempt
import uvicorn
from pydantic import BaseSettings

from langport.constants import WORKER_API_TIMEOUT, ErrorCode
from langport.model.model_adapter import get_conversation_template
from fastapi.exceptions import RequestValidationError
from langport.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    EmbeddingsData,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
)
from langport.protocol.worker_protocol import (
    BaseWorkerResult,
    EmbeddingWorkerResult,
    GenerationWorkerResult,
    WorkerAddressRequest,
    WorkerAddressResponse,
)
from langport.core.dispatch import DispatchMethod

logger = logging.getLogger(__name__)


class AppSettings(BaseSettings):
    # The address of the model controller.
    controller_address: str = "http://localhost:21001"


app_settings = AppSettings()

app = fastapi.FastAPI(debug=True)
headers = {"User-Agent": "Langport API Server"}

class BaseAuthorizationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, sk:str, dispatch: Optional[DispatchFunction] = None) -> None:
        super().__init__(app, dispatch)
        self.sk = sk

    async def dispatch(self, request, call_next):
        authorization = request.headers.get("Authorization")
        if request.url.path not in ["/docs","/redoc","/openapi.json"] and (
            not authorization 
            or authorization.split(" ")[0].lower() != "bearer" 
            or authorization.split(" ")[1] != self.sk
        ):
            return JSONResponse(
                status_code=401,
                content={"msg":"Not authenticated"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, code=code).dict(), status_code=500
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))


async def check_model(request, feature: str, model_name: str) -> Optional[JSONResponse]:
    ret = None
    async with httpx.AsyncClient() as client:
        try:
            models = await _list_models(feature, client)
        except Exception as e:
            ret = create_error_response(
                ErrorCode.INVALID_MODEL,
                str(e),
            )
            return ret
        if len(models) == 0 or model_name not in models:
            ret = create_error_response(
                ErrorCode.INVALID_MODEL,
                f"Only {'&&'.join(models)} allowed now, your model {request.model}",
            )
    return ret


async def check_length(request, request_type: str, prompt, max_tokens):
    async with httpx.AsyncClient() as client:
        worker_addr = await _get_worker_address(request.model, request_type, client, DispatchMethod.LOTTERY)

        response = await client.post(
            worker_addr + "/model_details",
            headers=headers,
            json={},
            timeout=WORKER_API_TIMEOUT,
        )
        context_len = response.json()["context_length"]

        response = await client.post(
            worker_addr + "/count_token",
            headers=headers,
            json={"prompt": prompt},
            timeout=WORKER_API_TIMEOUT,
        )
        token_num = response.json()["count"]

    max_new_tokens = max_tokens
    context_len = 2048

    if token_num + max_new_tokens > context_len:
        return create_error_response(
            ErrorCode.CONTEXT_OVERFLOW,
            f"This model's maximum context length is {context_len} tokens. "
            f"However, you requested {max_new_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{max_new_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return None


def check_requests(request) -> Optional[JSONResponse]:
    # Check all params
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.n is not None and request.n <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    if request.stop is not None and (
        not isinstance(request.stop, str) and not isinstance(request.stop, list)
    ):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )

    return None


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
) -> Dict[str, Any]:
    is_chatglm = "chatglm" in model_name.lower()
    conv = get_conversation_template(model_name)

    if isinstance(messages, str):
        prompt = messages
    else:
        for message in messages:
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

        if is_chatglm:
            prompt = conv.messages[conv.offset :]
        else:
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
    }

    if stop is None:
        gen_params.update(
            {"stop": conv.settings.stop_str, "stop_token_ids": conv.settings.stop_token_ids}
        )
    else:
        gen_params.update({"stop": stop})

    logger.debug(f"==== request ====\n{gen_params}")
    return gen_params


# @retry(stop=stop_after_attempt(5))
async def _get_worker_address(
    model_name: str, feature: str, client: httpx.AsyncClient, dispatch: Union[str, DispatchMethod]
) -> str:
    """
    Get worker address based on the requested model

    :param model_name: The worker's model name
    :param feature: The worker's feature
    :param client: The httpx client to use
    :return: Worker address from the controller
    :raises: :class:`ValueError`: No available worker for requested model
    """
    controller_address = app_settings.controller_address
    if isinstance(dispatch, str):
        dispatch = DispatchMethod.from_str(dispatch)
    if dispatch == DispatchMethod.LOTTERY:
        payload = WorkerAddressRequest(
            condition=f"{{model_name}}=='{model_name}' and '{feature}' in {{features}}", expression="1 / 0.01 + {speed}"
        )
    elif dispatch == DispatchMethod.SHORTEST_QUEUE:
        payload = WorkerAddressRequest(
            condition=f"{{model_name}}=='{model_name}' and '{feature}' in {{features}}", expression="{queue_length}/{speed}"
        )
    else:
        raise Exception("Error dispatch method.")
    ret = await client.post(
        controller_address + "/get_worker_address",
        json=payload.dict(),
    )
    response = WorkerAddressResponse.parse_obj(ret.json())
    address_list = response.address_list
    values = [json.loads(obj) for obj in response.values]

    # sort
    sorted_result = sorted(zip(address_list, values), key=lambda x: x[1])
    address_list = [x[0] for x in sorted_result]
    values = [x[1] for x in sorted_result]

    # No available worker
    if address_list == []:
        raise ValueError(f"No available worker for {model_name} and {feature}")
    if dispatch == DispatchMethod.LOTTERY:
        node_speeds = np.array(values, dtype=np.float32)
        norm = np.sum(node_speeds)
        if norm < 1e-4:
            return ""
        node_speeds = node_speeds / norm
        pt = np.random.choice(np.arange(len(address_list)), p=node_speeds)
        worker_addr = address_list[pt]
    elif dispatch == DispatchMethod.SHORTEST_QUEUE:
        worker_addr = address_list[0]
    else:
        raise Exception("Error dispatch method.")
    logger.debug(f"model_name: {model_name}, feature: {feature}, worker_addr: {worker_addr}")
    return worker_addr


# @retry(stop=stop_after_attempt(5))
async def _list_models(feature: Optional[str], client: httpx.AsyncClient) -> str:
    controller_address = app_settings.controller_address

    if feature is None:
        condition = "True"
    else:
        condition=f"'{feature}' in {{features}}"
    payload = WorkerAddressRequest(
        condition=condition, expression="{model_name}"
    )

    ret = await client.post(
        controller_address + "/get_worker_address",
        json=payload.dict(),
    )
    if ret.status_code != 200:
        return []
    response = WorkerAddressResponse.parse_obj(ret.json())
    
    address_list = response.address_list
    models = [json.loads(obj) for obj in response.values]
    # No available worker
    if address_list == []:
        raise ValueError(f"No available worker for feature {feature}")

    return models


@app.get("/v1/models")
async def show_available_models():
    async with httpx.AsyncClient() as client:
        models = await _list_models(None, client)
    models.sort()
    # TODO: return real model permission details
    model_cards = []
    for m in models:
        model_cards.append(ModelCard(id=m, root=m, permission=[ModelPermission()]))
    return ModelList(data=model_cards)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    error_check_ret = await check_model(request, "generation", request.model)
    if error_check_ret is not None:
        return error_check_ret
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    gen_params = get_gen_params(
        request.model,
        request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        echo=False,
        stream=request.stream,
        stop=request.stop,
    )
    if request.stream:
        generator = chat_completion_stream_generator(
            request.model, gen_params, request.n
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    choices = []
    chat_completions = []
    for i in range(request.n):
        content = asyncio.create_task(chat_completion(request.model, gen_params))
        chat_completions.append(content)

    usage = UsageInfo()
    for i, content_task in enumerate(chat_completions):
        content = await content_task
        if content is None:
            return create_error_response(ErrorCode.INTERNAL_ERROR, "Server internal error")
        if content.error_code != ErrorCode.OK:
            return create_error_response(content.error_code, content.message)
        choices.append(
            ChatCompletionResponseChoice(
                index=i,
                message=ChatMessage(role="assistant", content=content.text),
                finish_reason=content.finish_reason,
            )
        )
        task_usage = UsageInfo.parse_obj(content.usage)
        for usage_key, usage_value in task_usage.dict().items():
            setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)


async def chat_completion_stream_generator(
    model_name: str, gen_params: Dict[str, Any], n: int
) -> Generator[str, Any, None]:
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
            id=id, choices=[choice_data], model=model_name
        )
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        previous_text = ""
        async for content in generate_completion_stream("/chat_stream", gen_params):
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
                id=id, choices=[choice_data], model=model_name
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


async def chat_completion(
    model_name: str, gen_params: Dict[str, Any]
) -> Optional[GenerationWorkerResult]:
    ret = None
    async for content in generate_completion_stream("/chat_stream", gen_params):
        ret = content
    return ret


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    error_check_ret = await check_model(request, "generation", request.model)
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
    )

    if request.stream:
        generator = generate_completion_stream_generator(payload, request.n)
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        completions = []
        for i in range(request.n):
            content = asyncio.create_task(generate_completion(payload))
            completions.append(content)

        choices = []
        usage = UsageInfo()
        for i, content_task in enumerate(completions):
            content = await content_task
            if content.error_code != ErrorCode.OK:
                return create_error_response(content.error_code, content.message)
            choices.append(
                CompletionResponseChoice(
                    index=i,
                    text=content.text,
                    logprobs=content.logprobs,
                    finish_reason=content.finish_reason,
                )
            )
            task_usage = UsageInfo.parse_obj(content.usage)
            for usage_key, usage_value in task_usage.dict().items():
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

        return CompletionResponse(
            model=request.model, choices=choices, usage=UsageInfo.parse_obj(usage)
        )


async def generate_completion_stream_generator(payload: Dict[str, Any], n: int):
    model_name = payload["model"]
    id = f"cmpl-{shortuuid.random()}"
    finish_stream_events = []

    for i in range(n):
        previous_text = ""
        async for content in generate_completion_stream("/completion_stream", payload):
            if content.error_code != ErrorCode.OK:
                yield f"data: {json.dumps(content.dict(), ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            decoded_unicode = content.text.replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text) :]
            previous_text = decoded_unicode

            choice_data = CompletionResponseStreamChoice(
                index=i,
                text=delta_text,
                logprobs=content.logprobs,
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


async def generate_completion_stream(url: str, payload: Dict[str, Any]) -> Generator[GenerationWorkerResult, Any, None]:
    async with httpx.AsyncClient() as client:
        try:
            worker_addr = await _get_worker_address(payload["model"], "generation", client, DispatchMethod.LOTTERY)
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
                headers=headers,
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
        

async def generate_completion(payload: Dict[str, Any]) -> Optional[GenerationWorkerResult]:
    ret = None
    async for content in generate_completion_stream("/completion_stream", payload):
        ret = content
    return ret


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingsRequest):
    """Creates embeddings for the text"""
    error_check_ret = await check_model(request, "embedding", request.model)
    if error_check_ret is not None:
        return error_check_ret
    payload = {
        "model": request.model,
        "input": request.input,
    }

    response = await get_embedding(payload)
    if response.type == "error":
        return create_error_response(ErrorCode.INTERNAL_ERROR, response.message)
    return EmbeddingsResponse(
        data=[EmbeddingsData(embedding=response.embedding, index=0)],
        model=request.model,
        usage=UsageInfo(
            prompt_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens,
            completion_tokens=None,
        ),
    ).dict(exclude_none=True)


async def get_embedding(payload: Dict[str, Any]) -> EmbeddingWorkerResult:
    controller_address = app_settings.controller_address
    model_name = payload["model"]
    async with httpx.AsyncClient() as client:
        worker_addr = await _get_worker_address(model_name, "embedding", client, DispatchMethod.LOTTERY)

        response = await client.post(
            worker_addr + "/embeddings",
            headers=headers,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
        )
        if response.json()["type"] == "error":
            error_message = BaseWorkerResult.parse_obj(response.json())
            return error_message
        return EmbeddingWorkerResult.parse_obj(response.json())


if __name__ in ["__main__", "langport.service.gateway.openai_api"]:
    parser = argparse.ArgumentParser(
        description="Langport ChatGPT-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--sk", type=str, default=None, help="security key")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )
    if args.sk is not None:
        app.add_middleware(
            BaseAuthorizationMiddleware,
            sk=args.sk,
        )
    app_settings.controller_address = args.controller_address

    logger.debug(f"==== args ====\n{args}")

    # don't delete this line, otherwise the middleware won't work with reload==True
    if __name__ == "__main__":
        uvicorn.run(
            "langport.service.gateway.openai_api:app",
            host=args.host,
            port=args.port,
            log_level="info",
            reload=True,
        )
