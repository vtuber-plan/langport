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
from langport.core.embedding_worker import EmbeddingModelWorker

import uvicorn

from langport.model.model_adapter import add_model_args
from langport.protocol.worker_protocol import EmbeddingsTask
from langport.utils import build_logger

app = FastAPI()


def create_background_tasks(worker):
    background_tasks = BackgroundTasks()
    background_tasks.add_task(lambda: worker.release_model_semaphore())
    return background_tasks

@app.post("/embeddings")
async def api_embeddings(request: EmbeddingsTask):
    await app.worker.acquire_model_semaphore()
    embedding = app.worker.get_embeddings(request)
    background_tasks = create_background_tasks(app.worker)
    return JSONResponse(content=embedding.dict(), background=background_tasks)


@app.post("/get_worker_status")
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
    parser.add_argument("--batch", type=int, default=4)
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
    
    if args.model_name is None:
        args.model_name = os.path.basename(os.path.normpath(args.model_path))

    app.worker = EmbeddingModelWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        worker_type="embedding",
        model_path=args.model_path,
        model_name=args.model_name,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        limit_model_concurrency=args.limit_model_concurrency,
        max_batch=args.batch,
        stream_interval=args.stream_interval,
        logger=logger,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
