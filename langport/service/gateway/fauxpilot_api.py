import asyncio

import argparse
import asyncio
import json
import logging

import os
import random
from typing import Generator, Optional, Union, Dict, List, Any

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import numpy as np
import shortuuid
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
from langport.service.gateway.openai_api import check_model, check_requests, create_error_response, generate_completion, generate_completion_stream_generator, get_gen_params

logger = logging.getLogger(__name__)


class AppSettings(BaseSettings):
    # The address of the model controller.
    controller_address: str = "http://localhost:21001"


app_settings = AppSettings()

app = fastapi.FastAPI(debug=True)
headers = {"User-Agent": "LangPort API Server"}

@app.post("/v1/engines/codegen/completions")
@app.post("/v1/engines/copilot-codex/completions")
async def create_codegen_completion(request: CompletionRequest):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Langport ChatGPT-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
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
        "langport.service.gateway.fauxpilot_api:app",
        host=args.host,
        port=args.port,
        log_level="info",
        reload=True,
    )
