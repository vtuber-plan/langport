import argparse
import json
import logging

from typing import Optional

import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, DispatchFunction
from starlette.types import ASGIApp

from langport.constants import ErrorCode
from fastapi.exceptions import RequestValidationError
from langport.protocol.openai_api_protocol import (
    CompletionRequest,
)
from langport.routers.gateway.common import AppSettings, create_bad_request_response
from langport.routers.gateway.openai_compatible import api_completions, api_models


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


@app.get("/v1/models")
async def models():
    return await api_models(app.app_settings)

@app.post("/v1/engines/codegen/completions")
@app.post("/v1/engines/copilot-codex/completions")
@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    return await api_completions(app.app_settings, request)

if __name__ in ["__main__", "langport.service.gateway.fauxpilot_api"]:
    parser = argparse.ArgumentParser(
        description="Langport fauxpilot-compatible RESTful API server."
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
    app.app_settings = AppSettings(controller_address=args.controller_address)

    logger.debug(f"==== args ====\n{args}")

    # don't delete this line, otherwise the middleware won't work with reload==True
    if __name__ == "__main__":
        uvicorn.run(
            "langport.service.gateway.fauxpilot_api:app",
            host=args.host,
            port=args.port,
            log_level="info",
            reload=True,
        )
