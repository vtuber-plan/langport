import asyncio

import argparse
import asyncio
import json
import logging

import os
from typing import Generator, Optional, Union, Dict, List, Any

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import shortuuid
from tenacity import retry, stop_after_attempt
import uvicorn
from pydantic import BaseSettings

from langport.constants import WORKER_API_TIMEOUT, ErrorCode
from langport.model.model_adapter import get_conversation_template
from fastapi.exceptions import RequestValidationError
from langport.protocol.openai_api_protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
)
from langport.protocol.worker_protocol import BaseWorkerResult, GenerationWorkerResult, WorkerAddressRequest

logger = logging.getLogger(__name__)


class AppSettings(BaseSettings):
    # The address of the model controller.
    controller_address: str = "http://localhost:21001"


app_settings = AppSettings()

app = fastapi.FastAPI(debug=True)
headers = {"User-Agent": "FastChat API Server"}


def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, code=code).dict(), status_code=500
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))



async def check_model(request, request_type: str) -> Optional[JSONResponse]:
    controller_address = app_settings.controller_address
    ret = None
    async with httpx.AsyncClient() as client:
        try:
            _worker_addr = await _get_worker_address(
                request.model, request_type, client
            )
        except:
            models_ret = await client.post(controller_address + "/list_models")
            models = models_ret.json()["models"]
            ret = create_error_response(
                ErrorCode.INVALID_MODEL,
                f"Only {'&&'.join(models)} allowed now, your model {request.model}",
            )
    return ret


async def check_length(request, request_type: str, prompt, max_tokens):
    async with httpx.AsyncClient() as client:
        worker_addr = await _get_worker_address(request.model, request_type, client)

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
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)

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
        "max_new_tokens": max_tokens,
        "echo": echo,
        "stream": stream,
    }

    if stop is None:
        gen_params.update(
            {"stop": conv.stop_str, "stop_token_ids": conv.stop_token_ids}
        )
    else:
        gen_params.update({"stop": stop})

    logger.debug(f"==== request ====\n{gen_params}")
    return gen_params



@retry(stop=stop_after_attempt(5))
async def _get_worker_address(
    model_name: str, worker_type: str, client: httpx.AsyncClient
) -> str:
    """
    Get worker address based on the requested model

    :param model_name: The worker's model name
    :param client: The httpx client to use
    :return: Worker address from the controller
    :raises: :class:`ValueError`: No available worker for requested model
    """
    controller_address = app_settings.controller_address

    ret = await client.post(
        controller_address + "/get_worker_address",
        json=WorkerAddressRequest(
            model_name=model_name, worker_type=worker_type
        ).dict(),
    )
    worker_addr = ret.json()["address"]
    # No available worker
    if worker_addr == "":
        raise ValueError(f"No available worker for {model_name}")

    logger.debug(f"model_name: {model_name}, worker_addr: {worker_addr}")
    return worker_addr



@app.get("/v1/models")
async def show_available_models():
    controller_address = app_settings.controller_address
    async with httpx.AsyncClient() as client:
        ret = await client.post(controller_address + "/refresh_all_workers")
        ret = await client.post(controller_address + "/list_models")
    models = ret.json()["models"]
    models.sort()
    # TODO: return real model permission details
    model_cards = []
    for m in models:
        model_cards.append(ModelCard(id=m, root=m, permission=[ModelPermission()]))
    return ModelList(data=model_cards)


@app.post("/v1/engines/codegen/completions")
@app.post("/v1/engines/copilot-codex/completions")
@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    error_check_ret = await check_model(request, "generation")
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
    controller_address = app_settings.controller_address
    async with httpx.AsyncClient() as client:
        worker_addr = await _get_worker_address(payload["model"], "generation", client)

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
                        data = json.loads(chunk.decode())
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastChat FauxPilot-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8005, help="port number")
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
    app_settings.controller_address = args.controller_address

    logger.debug(f"==== args ====\n{args}")

    uvicorn.run(
        "langport.service.fauxpilot_api:app",
        host=args.host,
        port=args.port,
        log_level="info",
        reload=True,
    )
