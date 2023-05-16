import argparse
import asyncio
from typing import List, Union
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import numpy as np
import requests
import uvicorn

from langport.core.worker_node import WorkerNode
from langport.protocol.worker_protocol import (
    HeartbeatPing,
    HeartbeatPong,
    ListModelsResponse,
    RegisterNodeRequest,
    RemoveNodeRequest,
    WorkerAddressRequest,
    WorkerAddressResponse,

)
from langport.utils import build_logger


logger = build_logger("langport.service.test_node", "test_node.log")

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await app.node.start()


@app.on_event("shutdown")
async def shutdown_event():
    await app.node.stop()

@app.post("/register_node")
async def register_node(request: RegisterNodeRequest):
    response = await app.node.api_register_node(request)
    return response.dict()


@app.post("/remove_node")
async def remove_node(request: RemoveNodeRequest):
    response = await app.node.api_remove_node(request)
    return response.dict()

@app.post("/receive_heartbeat")
async def receive_heartbeat(request: HeartbeatPing):
    response = await app.node.api_receive_heartbeat(request)
    return response.dict()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21001)
    parser.add_argument("--neighbors", type=str, nargs="*", default=[])
    args = parser.parse_args()
    logger.info(f"args: {args}")

    node_id = str(uuid.uuid4())
    node_addr = f"http://{args.host}:{args.port}"
    app.node = WorkerNode(node_addr, node_id, args.neighbors, logger=logger)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
