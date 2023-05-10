import argparse
import asyncio
from typing import List, Union

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import numpy as np
import requests
import uvicorn

from langport.core.controller import Controller
from langport.protocol.worker_protocol import RegisterWorkerRequest, RemoveWorkerRequest
from langport.utils import build_logger


logger = build_logger("langport.service.controller", "controller.log")

app = FastAPI()

@app.post("/register_worker")
async def register_worker(request: RegisterWorkerRequest):
    app.controller.register_worker(request)

@app.post("/remove_worker")
async def remove_worker(request: RemoveWorkerRequest):
    app.controller.remove_worker(request)


@app.post("/refresh_all_workers")
async def refresh_all_workers():
    print(app)
    app.controller.refresh_all_workers()


@app.post("/list_models")
async def list_models():
    models = app.controller.list_models()
    return {"models": models}


@app.post("/get_worker_address")
async def get_worker_address(request: Request):
    data = await request.json()
    addr = app.controller.get_worker_address(data["model"])
    return {"address": addr}


@app.post("/receive_heart_beat")
async def receive_heart_beat(request: Request):
    data = await request.json()
    exist = app.controller.receive_heart_beat(data["worker_name"], data["queue_length"])
    return {"exist": exist}


@app.post("/worker_generate_stream")
async def worker_api_generate_stream(request: Request):
    params = await request.json()
    generator = app.controller.worker_api_generate_stream(params)
    return StreamingResponse(generator)


@app.post("/worker_generate_completion")
async def worker_api_generate_completion(request: Request):
    params = await request.json()
    output = app.controller.worker_api_generate_completion(params)
    return output


@app.post("/worker_get_embeddings")
async def worker_api_embeddings(request: Request):
    params = await request.json()
    output = app.controller.worker_api_embeddings(params)
    return output


@app.post("/worker_get_status")
async def worker_api_get_status(request: Request):
    return app.controller.worker_api_get_status()

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
    parser.add_argument("--controller-addr", type=str, default="http://localhost:21001")
    parser.add_argument(
        "--dispatch-method",
        type=str,
        choices=["lottery", "shortest_queue"],
        default="shortest_queue",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    app.controller = Controller(args.dispatch_method, logger=logger)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
