import argparse
import asyncio
from typing import List, Union
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import numpy as np
import requests
import uvicorn

from langport.core.controller import Controller
from langport.protocol.worker_protocol import (
    ListModelsResponse,
    RegisterWorkerRequest,
    RemoveWorkerRequest,
    WorkerAddressRequest,
    WorkerAddressResponse,
    WorkerHeartbeatPing,
    WorkerHeartbeatPong,
)
from langport.utils import build_logger


logger = build_logger("langport.service.controller", "controller.log")

app = FastAPI()


@app.post("/register_worker")
async def register_worker(request: RegisterWorkerRequest):
    app.controller.register_worker(request)


@app.post("/remove_worker")
async def remove_worker(request: RemoveWorkerRequest):
    app.controller.remove_worker(request)


@app.post("/list_models")
async def list_models():
    models = app.controller.list_models()
    return ListModelsResponse(models=models)


@app.post("/get_worker_address")
async def get_worker_address(request: WorkerAddressRequest):
    addr = app.controller.get_worker_address(request.model_name, request.worker_type)
    return WorkerAddressResponse(address=addr)


@app.post("/receive_heart_beat")
async def receive_heart_beat(request: WorkerHeartbeatPing):
    exist = app.controller.receive_heart_beat(request)
    return WorkerHeartbeatPong(
        exist=exist
    ).dict()


@app.post("/get_worker_status")
async def api_get_worker_status(request: Request):
    return app.controller.api_get_worker_status()


@app.on_event("startup")
async def startup_event():
    app.controller.start()


@app.on_event("shutdown")
def shutdown_event():
    app.controller.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21001)
    parser.add_argument(
        "--dispatch-method",
        type=str,
        choices=["lottery", "shortest_queue"],
        default="shortest_queue",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    controller_id = str(uuid.uuid4())

    app.controller = Controller(f"http://{args.host}:{args.port}", controller_id, args.dispatch_method, logger=logger)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
