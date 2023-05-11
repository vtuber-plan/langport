import argparse
import asyncio
import os
import random
import time
from typing import List, Union
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests

from langport.core.worker import ModelWorker

import uvicorn

from langport.model.model_adapter import add_model_args
from langport.utils import build_logger

app = FastAPI()


def release_model_semaphore():
    app.worker.model_semaphore.release()


def acquire_model_semaphore():
    app.worker.global_counter += 1
    if app.worker.model_semaphore is None:
        app.worker.model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    return app.worker.model_semaphore.acquire()


def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return background_tasks


@app.post("/chat_stream")
async def api_chat_stream(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    generator = app.worker.generate_stream(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)


@app.post("/chat")
async def api_chat(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    output = app.worker.generate(params)
    release_model_semaphore()
    return JSONResponse(output)


@app.post("/completion_stream")
async def api_completion_stream(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    generator = app.worker.generate_stream(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)


@app.post("/completion")
async def api_completion(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    completion = app.worker.generate(params)
    background_tasks = create_background_tasks()
    return JSONResponse(content=completion, background=background_tasks)


@app.post("/embeddings")
async def api_embeddings(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    embedding = app.worker.get_embeddings(params)
    background_tasks = create_background_tasks()
    return JSONResponse(content=embedding, background=background_tasks)


@app.get("/get_worker_status")
async def api_get_status(request: Request):
    return app.worker.get_status()


@app.on_event("startup")
async def startup_event():
    app.worker.start()


@app.on_event("shutdown")
def shutdown_event():
    app.worker.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--worker-address", type=str, default=None)
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    add_model_args(parser)
    parser.add_argument("--model-name", type=str, help="Optional display name")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker_id = str(uuid.uuid4())
    logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
    logger.info(f"args: {args}")

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.port is None:
        args.port = random.randint(21001, 29001)

    if args.worker_address is None:
        args.worker_address = f"http://{args.host}:{args.port}"

    app.worker = ModelWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        model_path=args.model_path,
        model_name=args.model_name,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        limit_model_concurrency=args.limit_model_concurrency,
        stream_interval=args.stream_interval,
        logger=logger,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
