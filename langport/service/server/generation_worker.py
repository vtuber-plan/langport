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

from langport.core.generation_worker import GenerationModelWorker

import uvicorn

from langport.model.model_adapter import add_model_args
from langport.protocol.worker_protocol import GenerationTask
from langport.utils import build_logger

from .generation_node import app


# We suggest that concurrency == batch * thread
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
    parser.add_argument("--limit-model-concurrency", type=int, default=16)
    parser.add_argument("--batch", type=int, default=8)
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

    app.worker = GenerationModelWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        worker_type="generation",
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
