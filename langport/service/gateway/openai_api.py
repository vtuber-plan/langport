import argparse
import datetime
import json
import logging
import os

from typing import Optional

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, DispatchFunction
from starlette.types import ASGIApp
from starlette.requests import Request
import uvicorn

from langport.constants import LOGDIR, ErrorCode
from fastapi.exceptions import RequestValidationError
from langport.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingsRequest,
)
from langport.routers.gateway.common import AppSettings, create_bad_request_response
from langport.routers.gateway.openai_compatible import api_chat_completions, api_completions, api_embeddings, api_models
from langport.utils import build_logger

# current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = build_logger("openai_api", "openai_api.log")
app = fastapi.FastAPI(debug=False)

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

redirect_rules = None
def redirect_model_name(model: str):
    if redirect_rules is not None:
        for rule in redirect_rules:
            from_model_name, to_model_name = rule.split(":")
            if model == from_model_name:
                logger.debug(f"Redirect model {from_model_name} to {to_model_name}")
                model = to_model_name
                break
    return model

def check_and_log_response(response):
    if isinstance(response, JSONResponse):
        response_body = json.loads(response.body)
        if "object" in response_body and response_body["object"] == "error":
            if "object" in response_body and "message" in response_body and "code" in response_body:
                logger.error(f"[{response_body['object']}] [{response_body['code']}] - {response_body['message']}")
            else:
                logger.error(response.body)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc):
    logger.error(f"Invalid request: {(await request.body()).decode('utf-8')}")
    response = create_bad_request_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))
    check_and_log_response(response)
    return response

@app.get("/v1/models")
async def models():
    response = await api_models(app.app_settings)
    check_and_log_response(response)
    return response

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    logger.info(json.dumps(json.loads(request.model_dump_json()), ensure_ascii=False))
    request.model = redirect_model_name(request.model)
    response = await api_chat_completions(app.app_settings, request)
    check_and_log_response(response)
    return response

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    logger.info(json.dumps(json.loads(request.model_dump_json()), ensure_ascii=False))
    request.model = redirect_model_name(request.model)
    response = await api_completions(app.app_settings, request)
    check_and_log_response(response)
    return response

@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingsRequest):
    logger.info(json.dumps(json.loads(request.model_dump_json()), ensure_ascii=False))
    request.model = redirect_model_name(request.model)
    response = await api_embeddings(app.app_settings, request)
    check_and_log_response(response)
    return response

if __name__ in ["__main__", "langport.service.gateway.openai_api"]:
    parser = argparse.ArgumentParser(
        description="Langport openai-compatible RESTful API server."
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
    parser.add_argument("--redirect", action="append", required=False, help="redirect model_name to another model_name, e.g. --redirect model_name1:model_name2")
    parser.add_argument("--ssl-key", type=str, default=None)
    parser.add_argument("--ssl-cert", type=str, default=None)
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )
    if args.redirect is not None:
        redirect_rules = args.redirect
    if args.sk is not None:
        app.add_middleware(
            BaseAuthorizationMiddleware,
            sk=args.sk,
        )
    app.app_settings = AppSettings(controller_address=args.controller_address)

    logger.debug(f"==== args ====\n{args}")

    # don't delete this line, otherwise the middleware won't work with reload==True
    if __name__ == "__main__":
        uvicorn.run(
            "langport.service.gateway.openai_api:app",
            host=args.host,
            port=args.port,
            log_level="info",
            reload=False,
            ssl_keyfile=args.ssl_key,
            ssl_certfile=args.ssl_cert,
        )
