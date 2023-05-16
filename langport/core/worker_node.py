import argparse
import asyncio
from collections import defaultdict
import dataclasses
import logging
import json
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union
import threading
import queue
import uuid

import httpx
import requests
from tenacity import retry, stop_after_attempt
from langport.core.base_node import BaseNode

from langport.protocol.worker_protocol import (
    NodeInfo,
    RegisterNodeRequest,
    RegisterNodeResponse,
    RemoveNodeRequest,
    RemoveNodeResponse,
    HeartbeatPing,
    HeartbeatPong,
    WorkerStatus,
)

from langport.constants import (
    WORKER_API_TIMEOUT,
    WORKER_HEART_BEAT_INTERVAL,
    ErrorCode,
)
from langport.utils.interval_timer import IntervalTimer


class WorkerNode(BaseNode):
    def __init__(
        self,
        node_addr: str,
        node_id: str,
        init_neighborhoods_addr: List[str],
        logger: logging.Logger,
    ):
        super(WorkerNode, self).__init__(
            node_addr=node_addr,
            node_id=node_id,
            logger=logger,
        )
        self.init_neighborhoods_addr: List[str] = init_neighborhoods_addr
        self.neighborhoods: Dict[str, NodeInfo] = {}
        self.headers = {"User-Agent": "LangPort nodes"}

        # timers
        self.add_timer(
            "heartbeat",
            WORKER_HEART_BEAT_INTERVAL,
            self.send_heartbeat_broadcast,
            args=None,
            kwargs=None,
            workers=1,
        )

        # start and stop
        self.on_start("register_node_broadcast", self.register_node_broadcast)
        self.on_stop("remove_node_broadcast", self.remove_node_broadcast)

    async def register_node(self, node_addr: str) -> bool:
        self.logger.info(f"Register to node {node_addr}")

        data = RegisterNodeRequest(
            node_id=self.node_id,
            node_addr=self.node_addr,
            check_heart_beat=True,
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                node_addr + "/register_node",
                headers=self.headers,
                json=data.dict(),
                timeout=WORKER_API_TIMEOUT,
            )
            ret = RegisterNodeResponse.parse_obj(response.json())
  
        return True

    async def register_node_broadcast(self):
        self.logger.info(f"Register node broadcast. node_id: {self.node_id}, node_addr: {self.node_addr}")

        for neighborhood in self.init_neighborhoods_addr:
            await self.register_node(neighborhood)

    async def remove_node(self, node_addr: str) -> bool:
        self.logger.info("Remove node")

        data = RemoveNodeRequest(
            node_id=self.node_id,
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                node_addr + "/remove_node",
                headers=self.headers,
                json=data.dict(),
                timeout=WORKER_API_TIMEOUT,
            )
            ret = RemoveNodeResponse.parse_obj(response.json())
  
        return True

    async def remove_node_broadcast(self):
        self.logger.info(f"Remove node broadcast. node_id: {self.node_id}, node_addr: {self.node_addr}")
        for node_id, node_info in self.neighborhoods.items():
            await self.remove_node(node_info.node_addr)


    async def send_heartbeat(self, node_addr: str):
        data = HeartbeatPing(
            node_id=self.node_id,
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                node_addr + "/heartbeat",
                headers=self.headers,
                json=data.dict(),
                timeout=WORKER_API_TIMEOUT,
            )
            ret = HeartbeatPong.parse_obj(response.json())
  
        return ret

    async def send_heartbeat_broadcast(self):
        self.logger.info(
            f"Send heartbeat. Worker: {self.node_id}; Address: {self.node_addr}."
        )

        self.logger.info(
            f"Neighborhoods: {self.neighborhoods}."
        )

        for node_id, node_info in self.neighborhoods.items():
            await self.send_heartbeat(node_info.node_addr)

    async def api_register_node(self, request: RegisterNodeRequest) -> RegisterNodeResponse:
        print(request)
        if request.node_id not in self.neighborhoods:
            self.logger.info(f"Register {request.node_id} on {self.node_id}")
            self.neighborhoods[request.node_id] = NodeInfo(
                node_id=request.node_id,
                node_addr=request.node_addr,
                check_heart_beat=request.check_heart_beat,
                refresh_time=int(time.time())
            )

            # broacast again
            for node_id, node_info in self.neighborhoods.items():
                if node_id == request.node_id:
                    continue
                await self.register_node(node_info.node_addr)
        
        return RegisterNodeResponse(node_id=self.node_id)

    async def api_remove_node(self, request: RemoveNodeRequest) -> RemoveNodeResponse:
        if request.node_id in self.neighborhoods:
            self.logger.info(f"Remove {request.node_id} from {self.node_id}")
            del self.neighborhoods[request.node_id]

            # broacast again
            for node_id, node_info in self.neighborhoods.items():
                if node_id == request.node_id:
                    continue
                await self.remove_node_broadcast(node_info.node_addr)
        return RemoveNodeResponse(node_id=self.node_id)
    
    async def api_receive_heartbeat(self, request: HeartbeatPing) -> HeartbeatPong:
        if request.node_id in self.neighborhoods:
            self.neighborhoods[request.node_id].refresh_time = int(time.time())
        else:
            self.logger.info(f"Invalid ping packet from {request.node_id}.")
        return HeartbeatPong(exist=True)