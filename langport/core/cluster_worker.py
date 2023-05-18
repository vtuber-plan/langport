import argparse
import asyncio
from collections import defaultdict
import dataclasses
from functools import partial
import logging
import json
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union
import threading
import queue
import uuid


from enum import Enum, auto
import httpx
import numpy as np
import requests
from tenacity import retry, stop_after_attempt
from langport.core.base_node import BaseNode
from langport.core.cluster_node import ClusterNode

from langport.protocol.worker_protocol import (
    BaseWorkerResult,
    BaseWorkerTask,
    NodeInfo,
    NodeInfoRequest,
    NodeInfoResponse,
    NodeListRequest,
    NodeListResponse,
    RegisterNodeRequest,
    RegisterNodeResponse,
    RemoveNodeRequest,
    RemoveNodeResponse,
    HeartbeatPing,
    HeartbeatPong,
    WorkerAddressRequest,
    WorkerAddressResponse,
)

from langport.constants import (
    HEART_BEAT_EXPIRATION,
    WORKER_API_TIMEOUT,
    WORKER_HEART_BEAT_INTERVAL,
    ErrorCode,
)
from langport.utils.interval_timer import IntervalTimer


class DispatchMethod(Enum):
    LOTTERY = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, name):
        if name == "lottery":
            return cls.LOTTERY
        elif name == "shortest_queue":
            return cls.SHORTEST_QUEUE
        else:
            raise ValueError(f"Invalid dispatch method")


class ClusterWorker(ClusterNode):
    def __init__(
        self,
        node_addr: str,
        node_id: str,
        init_neighborhoods_addr: List[str],
        dispatch_method: str,
        limit_model_concurrency: int,
        max_batch: int,
        stream_interval: int,
        logger: logging.Logger,
    ):
        super(ClusterWorker, self).__init__(
            node_addr=node_addr,
            node_id=node_id,
            init_neighborhoods_addr=init_neighborhoods_addr,
            logger=logger,
        )

        self.limit_model_concurrency = limit_model_concurrency
        self.max_batch = max_batch
        self.stream_interval = stream_interval
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)

        self.global_counter = 0
        self.model_semaphore = None

        self.task_queue: queue.Queue[BaseWorkerTask] = queue.Queue()
        self.output_queue: Dict[str, queue.Queue[BaseWorkerResult]] = defaultdict(
            queue.Queue
        )

        self.on_start("set_queue_state", self.set_queue_state)
        self.on_start("set_features", self.set_features)

    async def set_queue_state(self):
        await self.set_local_state("queue_length", self.get_queue_length())

    async def set_features(self):
        await self.set_local_state("features", [])

    def release_model_semaphore(self):
        self.model_semaphore.release()

    def acquire_model_semaphore(self):
        self.global_counter += 1
        if self.model_semaphore is None:
            self.model_semaphore = asyncio.Semaphore(self.limit_model_concurrency)
        return self.model_semaphore.acquire()

    def get_queue_length(self):
        if (
            self.model_semaphore is None
            or self.model_semaphore._value is None
            or self.model_semaphore._waiters is None
        ):
            return 0
        else:
            return (
                self.limit_model_concurrency
                - self.model_semaphore._value
                + len(self.model_semaphore._waiters)
            )

    def add_task(self, task: BaseWorkerTask):
        self.task_queue.put(task, block=True, timeout=WORKER_API_TIMEOUT)
        self.output_queue[task.task_id] = queue.Queue()

    def fetch_task_result(self, task_id: str):
        result_queue = self.output_queue[task_id]
        retry_counter = 0
        while True:
            try:
                event = result_queue.get(block=False, timeout=None)
            except queue.Empty:
                time.sleep(0.01)
                retry_counter += 1
                # If client disconnected, stop to wait queue.
                if retry_counter > 2000:
                    break
                else:
                    continue
            retry_counter = 0
            if event.type == "done":
                break
            elif event.type == "error":
                yield event
                break
            elif event.type == "finish":
                yield event
                break
            elif event.type == "data":
                yield event
            else:
                raise ValueError("Bad chunk type.")

        del self.output_queue[task_id]

    def get_num_tasks(self) -> int:
        return self.task_queue.qsize()

    def fetch_tasks(self) -> List[BaseWorkerResult]:
        task_batch = []
        while len(task_batch) <= self.max_batch:
            try:
                task = self.task_queue.get(block=False, timeout=None)
            except queue.Empty:
                break
            task_batch.append(task)
        return task_batch

    def push_task_result(self, task_id: str, response: BaseWorkerResult):
        self.output_queue[task_id].put(response, block=True, timeout=WORKER_API_TIMEOUT)

    async def get_worker_address(self, model_name: str, feature_tag: str) -> Optional[str]:
        if self.dispatch_method == DispatchMethod.LOTTERY:
            worker_ids = []
            worker_speeds = []
            for w_id, w_info in self.neighborhoods.items():
                w_model_name = await self.get_node_state(w_info.node_id, "model_name")
                w_features = await self.get_node_state(w_info.node_id, "features")
                w_speed = await self.get_node_state(w_info.node_id, "speed")
                if model_name == w_model_name and feature_tag in w_features:
                    worker_ids.append(w_id)
                    worker_speeds.append(w_speed)
            worker_speeds = np.array(worker_speeds, dtype=np.float32)
            norm = np.sum(worker_speeds)
            if norm < 1e-4:
                return ""
            worker_speeds = worker_speeds / norm
            pt = np.random.choice(np.arange(len(worker_ids)), p=worker_speeds)
            node_id = worker_ids[pt]
            return self.neighborhoods[node_id].node_addr

        elif self.dispatch_method == DispatchMethod.SHORTEST_QUEUE:
            worker_ids = []
            worker_qlen = []
            for w_id, w_info in self.neighborhoods.items():
                w_model_name = await self.get_node_state(w_info.node_id, "model_name")
                w_features = await self.get_node_state(w_info.node_id, "features")
                w_speed = await self.get_node_state(w_info.node_id, "speed")
                w_queue_length = await self.get_node_state(w_info.node_id, "queue_length")
                if model_name == w_model_name and feature_tag in w_features:
                    worker_ids.append(w_id)
                    worker_qlen.append(w_queue_length / w_speed)
            if len(worker_ids) == 0:
                return ""
            min_index = np.argmin(worker_qlen)
            w_id = worker_ids[min_index]
            return self.neighborhoods[w_id].node_addr
        else:
            raise ValueError(f"Invalid dispatch method: {self.dispatch_method}")

    async def api_get_worker_address(self, request: WorkerAddressRequest) -> WorkerAddressResponse:
        node_address = await self.get_worker_address(request.model_name, request.feature)
        return WorkerAddressResponse(
            node_address=node_address
        )