import argparse
import asyncio
from typing import List, Union
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import numpy as np
import requests
import uvicorn

from langport.core.cluster_node import ClusterNode
from langport.core.cluster_worker import ClusterWorker
from langport.protocol.worker_protocol import (
    GetNodeStateRequest,
    HeartbeatPing,
    HeartbeatPong,
    NodeInfoRequest,
    NodeListRequest,
    RegisterNodeRequest,
    RemoveNodeRequest,
)
from langport.utils import build_logger
from .core_node import app, create_background_tasks

logger = build_logger("langport.service.dummy_worker", "dummy_worker.log")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21001)
    parser.add_argument("--neighbors", type=str, nargs="*", default=[])
    args = parser.parse_args()
    logger.info(f"args: {args}")

    node_id = str(uuid.uuid4())
    node_addr = f"http://{args.host}:{args.port}"
    app.node = ClusterWorker(
        node_addr=node_addr,
        node_id=node_id,
        init_neighborhoods_addr=args.neighbors,
        limit_model_concurrency=32,
        max_batch=1,
        stream_interval=2,
        logger=logger,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
