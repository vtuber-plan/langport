import argparse
import json
import logging

from typing import Optional
import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, DispatchFunction
from starlette.types import ASGIApp
import uvicorn

from langport.constants import ErrorCode
from fastapi.exceptions import RequestValidationError
from langport.protocol.huggingface_api_protocol import Details, FinishReason, Request, Response
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


@app.post("/")
async def root(request: Request):
    pass


@app.post("/generate")
async def generate(request: Request):
    error_check_ret = await check_model(app.app_settings, request, "generation", app.model_name)
    if error_check_ret is not None:
        return error_check_ret
    
    temperature = 0.7
    top_p = 1.0
    top_k = 1.0
    max_tokens = 2048
    stop = []
    if request.parameters is not None:
        temperature = request.parameters.temperature
        top_p = request.parameters.top_p
        top_k = request.parameters.top_k
        max_tokens = request.parameters.max_new_tokens
        stop = request.parameters.stop
    payload = get_gen_params(
        app.model_name,
        request.inputs,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        echo=False,
        stream=False,
        stop=stop,
    )
    N = 1
    response = await completions_non_stream(
        app.app_settings,
        payload,
        OpenAICompletionRequest(
            model=app.model_name,
            prompt=request.inputs,
            n=N,
        )
    )
    if isinstance(response, JSONResponse):
        return Response(generated_text="", details=Details(
            finish_reason=FinishReason.EndOfSequenceToken,
            generated_tokens=0,
        ))

    # print(response.choices[0].text.replace("\n", "\\n"))
    return Response(generated_text=response.choices[0].text,
            details=Details(
                finish_reason=FinishReason.EndOfSequenceToken,
                generated_tokens=response.usage.completion_tokens,
            )
        )
    


@app.post("/generate_stream")
async def generate_stream(request: Request):
    error_check_ret = await check_model(app.app_settings, request, "generation", app.model_name)
    if error_check_ret is not None:
        return error_check_ret


if __name__ in ["__main__", "langport.service.gateway.huggingface_api"]:
    parser = argparse.ArgumentParser(
        description="Langport huggingface-compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--sk", type=str, default=None, help="security key")
    parser.add_argument("--model-name", type=str, default="J-350M", help="default tabby model name")
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
    app.model_name = args.model_name
    app.app_settings = AppSettings(controller_address=args.controller_address)

    logger.debug(f"==== args ====\n{args}")

    # don't delete this line, otherwise the middleware won't work with reload==True
    if __name__ == "__main__":
        uvicorn.run(
            "langport.service.gateway.huggingface_api:app",
            host=args.host,
            port=args.port,
            log_level="info",
            reload=True,
        )
