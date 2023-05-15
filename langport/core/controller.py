import argparse
import asyncio
import dataclasses
from enum import Enum, auto
import json
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union
import threading

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import numpy as np
import requests
import uvicorn

from langport.constants import (
    CONTROLLER_HEART_BEAT_EXPIRATION,
    CONTROLLER_STATUS_FRESH_INTERVAL,
    WORKER_API_TIMEOUT,
)
from langport.core.base_worker import BaseWorker
from langport.protocol.worker_protocol import (
    RemoveWorkerRequest,
    RegisterWorkerRequest,
    WorkerHeartbeatPing,
    WorkerHeartbeatPong,
    WorkerStatus,
)
from langport.utils import server_error_msg


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


@dataclasses.dataclass
class WorkerInfo:
    worker_id: str
    worker_type: str
    worker_addr: str
    check_heart_beat: bool
    status: Optional[WorkerStatus]
    update_time: int

class Controller(BaseWorker):
    def __init__(
        self,
        controller_addr: str,
        controller_id: str,
        dispatch_method: str,
        logger: logging.Logger,
    ):
        super(Controller, self).__init__(
            controller_addr=None,
            worker_addr=controller_addr,
            worker_id=controller_id,
            worker_type="controller",
            logger=logger,
        )
        self.worker_info: Dict[str, WorkerInfo] = {}
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)

        self.logger = logger
        self.logger.info("Init controller")

        # startup
        self.add_timer(
            "remove_stable_workers",
            CONTROLLER_HEART_BEAT_EXPIRATION,
            self.remove_stable_workers_by_expiration,
            workers=1,
        )

        self.add_timer(
            "request_worker_status",
            CONTROLLER_STATUS_FRESH_INTERVAL,
            self.request_worker_status,
            workers=1,
        )

    def register_worker(self, request: RegisterWorkerRequest) -> bool:
        worker_id = request.worker_id
        worker_type = request.worker_type
        worker_addr = request.worker_addr
        if worker_id not in self.worker_info:
            self.logger.info(f"Register a new worker: {worker_id}")
        else:
            self.logger.info(f"Register an existing worker: {worker_id}")

        self.worker_info[worker_id] = WorkerInfo(
            worker_id=request.worker_id,
            worker_type=request.worker_type,
            worker_addr=request.worker_addr,
            check_heart_beat=request.check_heart_beat,
            status=None,
            update_time=time.time(),
        )

        self.logger.info(f"Register done: {worker_id}")
        return True

    def remove_worker(self, request: RemoveWorkerRequest) -> bool:
        worker_id = request.worker_id
        if worker_id not in self.worker_info:
            self.logger.info(f"Remove a non-existent worker: {worker_id}")
        else:
            self.logger.info(f"Remove an existing worker: {worker_id}")
            del self.worker_info[worker_id]

        self.logger.info(f"Remove done: {worker_id}")
        return True

    def get_worker_status(self, worker_addr: str) -> Optional[WorkerStatus]:
        try:
            r = requests.get(worker_addr + "/get_worker_status", timeout=5)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Get worker status fails: {worker_addr}, {e}")
            return None

        if r.status_code != 200:
            self.logger.error(f"Get worker status fails: {worker_addr}, {r}")
            return None

        return WorkerStatus.parse_obj(r.json())

    def list_models(self):
        model_names = set()

        for w_id, w_info in self.worker_info.items():
            if w_info.status is None:
                w_info.status = self.request_single_worker_status(w_info.worker_addr)
            model_names.add(w_info.status.model_name)

        return list(model_names)

    def get_worker_address(self, model_name: str, worker_type: str) -> str:
        if self.dispatch_method == DispatchMethod.LOTTERY:
            worker_names = []
            worker_speeds = []
            for w_name, w_info in self.worker_info.items():
                if (
                    model_name == w_info.status.model_name
                    and worker_type == w_info.worker_type
                ):
                    worker_names.append(w_name)
                    worker_speeds.append(w_info.status.speed)
            worker_speeds = np.array(worker_speeds, dtype=np.float32)
            norm = np.sum(worker_speeds)
            if norm < 1e-4:
                return ""
            worker_speeds = worker_speeds / norm
            pt = np.random.choice(np.arange(len(worker_names)), p=worker_speeds)
            worker_name = worker_names[pt]
            return worker_name

        elif self.dispatch_method == DispatchMethod.SHORTEST_QUEUE:
            worker_ids = []
            worker_qlen = []
            for w_id, w_info in self.worker_info.items():
                if (
                    model_name == w_info.status.model_name
                    and worker_type == w_info.worker_type
                ):
                    worker_ids.append(w_id)
                    worker_qlen.append(w_info.status.queue_length / w_info.status.speed)
            if len(worker_ids) == 0:
                return ""
            min_index = np.argmin(worker_qlen)
            w_id = worker_ids[min_index]
            self.worker_info[w_id].status.queue_length += 1
            self.logger.info(
                f"names: {worker_ids}, queue_lens: {worker_qlen}, ret: {w_id}"
            )
            return self.worker_info[w_id].worker_addr
        else:
            raise ValueError(f"Invalid dispatch method: {self.dispatch_method}")

    def receive_heart_beat(self, heartbeat: WorkerHeartbeatPing):
        worker_id = heartbeat.worker_id

        if worker_id not in self.worker_info:
            self.logger.info(f"Receive unknown heart beat. {worker_id}")
            return False

        self.logger.info(f"Receive heart beat. {worker_id}")
        return True

    def remove_stable_workers_by_expiration(self):
        expire = time.time() - CONTROLLER_HEART_BEAT_EXPIRATION
        to_delete = []
        for worker_id, w_info in self.worker_info.items():
            if w_info.check_heart_beat and w_info.update_time < expire:
                self.logger.info(f"Worker hearbeat expiration: {worker_id}")
                to_delete.append(worker_id)

        for worker_id in to_delete:
            self.remove_worker(RemoveWorkerRequest(worker_id=worker_id))

    def request_single_worker_status(self, worker_addr: str):
        url = worker_addr + "/get_worker_status"

        ret = requests.post(
            url,
            json={},
            timeout=WORKER_API_TIMEOUT,
        )
        status = WorkerStatus.parse_obj(ret.json())
        return status


    def request_worker_status(self):
        self.logger.info(f"Update workers status.")
        for worker_id, worker_info in self.worker_info.items():
            worker_info.status = self.request_single_worker_status(worker_info.worker_addr)
            worker_info.update_time = time.time()