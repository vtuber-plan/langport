import argparse
import json
import logging

from typing import Optional, Union

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, DispatchFunction
from starlette.types import ASGIApp
import uvicorn

from langport.constants import ErrorCode
from fastapi.exceptions import RequestValidationError
from langport.protocol.tabby_api_protocol import (
    Choice,
    CompletionRequest,
    CompletionResponse,
    ChoiceEvent,
    CompletionEvent,
    EventTypeMapping,
    HTTPValidationError,
    LanguagePresets,
)
from langport.protocol.openai_api_protocol import CompletionRequest as OpenAICompletionRequest

from langport.routers.gateway.common import AppSettings, check_model, create_bad_request_response
from langport.routers.gateway.openai_compatible import completions_non_stream, get_gen_params

logger = logging.getLogger(__name__)

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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_bad_request_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))

def trim_with_stop_words(output: str, stopwords: list) -> str:
    for w in sorted(stopwords, key=len, reverse=True):
        index = output.find(w)
        if index != -1:
            output = output[:index]
    return output

@app.post("/v1/events")
async def events(e: Union[ChoiceEvent, CompletionEvent]):
    if isinstance(e, EventTypeMapping[e.type]):
        # events_lib.log_event(e)
        return JSONResponse(content="ok")
    else:
        print(type(e))
        return JSONResponse(content="invalid event", status_code=422)


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    error_check_ret = await check_model(app.app_settings, request, "generation", app.model_name)
    if error_check_ret is not None:
        return error_check_ret
    
    preset = LanguagePresets.get(request.language, None)
    if preset is None:
        return CompletionResponse()

    # print(request.prompt)
    payload = get_gen_params(
        app.model_name,
        request.prompt,
        temperature=0.7,
        top_p=1.0,
        max_tokens=preset.max_length,
        echo=False,
        stream=False,
        stop=preset.stop_words,
    )
    N = 1
    response = await completions_non_stream(
        app.app_settings,
        payload,
        OpenAICompletionRequest(
            model=app.model_name,
            prompt=request.prompt,
            n=N,
        )
    )
    if isinstance(response, JSONResponse):
        print(response.body)
        return CompletionResponse()

    # print(response.choices[0].text.replace("\n", "\\n"))
    return CompletionResponse(choices=[
        Choice(
            index=i,
            text=response.choices[i].text
        )
        for i in range(len(response.choices))
    ])

if __name__ in ["__main__", "langport.service.gateway.tabby_api"]:
    parser = argparse.ArgumentParser(
        description="Langport tabby-compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--model-name", type=str, default="J-350M", help="default tabby model name")
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
    if args.sk is not None:
        app.add_middleware(
            BaseAuthorizationMiddleware,
            sk=args.sk,
        )
    app.model_name = args.model_name
    app.app_settings = AppSettings(controller_address=args.controller_address)

    logger.debug(f"==== args ====\n{args}")

    # don't delete this line, otherwise the middleware won't work with reload==True
    if __name__ == "__main__":
        uvicorn.run(
            "langport.service.gateway.tabby_api:app",
            host=args.host,
            port=args.port,
            log_level="info",
            reload=True,
            ssl_keyfile=args.ssl_key,
            ssl_certfile=args.ssl_cert,
        )
