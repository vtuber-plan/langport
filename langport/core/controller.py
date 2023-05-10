import argparse
import asyncio
import dataclasses
from enum import Enum, auto
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union
import threading

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import numpy as np
import requests
import uvicorn

from langport.constants import CONTROLLER_HEART_BEAT_EXPIRATION, CONTROLLER_HEART_BEAT_CHECK_INTERVAL
from langport.protocol.worker_protocol import RemoveWorkerRequest, RegisterWorkerRequest, WorkerHeartbeat, WorkerStatus
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
    worker_addr: str
    status: Optional[WorkerStatus]
    update_time: int

def heart_beat_controller(controller: "Controller"):
    last_time = time.time()
    while controller.online:
        time.sleep(CONTROLLER_HEART_BEAT_CHECK_INTERVAL)
        now_time = time.time()
        if now_time - last_time > CONTROLLER_HEART_BEAT_EXPIRATION:
            controller.remove_stable_workers_by_expiration()
            last_time = now_time

class Controller(object):
    def __init__(self, dispatch_method: str, logger):
        self.worker_info: Dict[str: WorkerInfo] = {}
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)
        
        self.logger = logger
        self.logger.info("Init controller")

        self.online = False
    
    def start(self):
        if self.online:
            return
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_controller, args=(self,)
        )
        self.heart_beat_thread.start()
        self.online = True

    def stop(self):
        if not self.online:
            return
        self.online = False

    def register_worker(self, request: RegisterWorkerRequest) -> bool:
        worker_id = request.worker_id
        worker_addr = request.worker_addr
        if worker_id not in self.worker_info:
            self.logger.info(f"Register a new worker: {worker_id}")
        else:
            self.logger.info(f"Register an existing worker: {worker_id}")

        self.worker_info[worker_id] = WorkerInfo(
            worker_id=request.worker_id,
            worker_addr=request.worker_addr,
            status=request.worker_status,
            update_time=time.time()
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

    def refresh_all_workers(self):
        old_info = dict(self.worker_info)
        self.worker_info = {}

        for w_name, w_info in old_info.items():
            if not self.register_worker(w_name, w_info.check_heart_beat, None):
                self.logger.info(f"Remove stale worker: {w_name}")

    def list_models(self):
        model_names = set()

        for w_id, w_info in self.worker_info.items():
            model_names.add(w_info.status.model_name)

        return list(model_names)

    def get_worker_address(self, model_name: str) -> str:
        if self.dispatch_method == DispatchMethod.LOTTERY:
            worker_names = []
            worker_speeds = []
            for w_name, w_info in self.worker_info.items():
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_speeds.append(w_info.status.speed)
            worker_speeds = np.array(worker_speeds, dtype=np.float32)
            norm = np.sum(worker_speeds)
            if norm < 1e-4:
                return ""
            worker_speeds = worker_speeds / norm
            if True:  # Directly return address
                pt = np.random.choice(np.arange(len(worker_names)), p=worker_speeds)
                worker_name = worker_names[pt]
                return worker_name

            # Check status before returning
            while True:
                pt = np.random.choice(np.arange(len(worker_names)), p=worker_speeds)
                worker_name = worker_names[pt]

                if self.get_worker_status(worker_name):
                    break
                else:
                    self.remove_worker(worker_name)
                    worker_speeds[pt] = 0
                    norm = np.sum(worker_speeds)
                    if norm < 1e-4:
                        return ""
                    worker_speeds = worker_speeds / norm
                    continue
            return worker_name
        elif self.dispatch_method == DispatchMethod.SHORTEST_QUEUE:
            worker_ids = []
            worker_qlen = []
            for w_id, w_info in self.worker_info.items():
                if model_name == w_info.status.model_name:
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

    def receive_heart_beat(self, heartbeat: WorkerHeartbeat):
        worker_id = heartbeat.worker_id

        if worker_id not in self.worker_info:
            self.logger.info(f"Receive unknown heart beat. {worker_id}")
            return False

        self.worker_info[worker_id].status = heartbeat.status
        self.worker_info[worker_id].update_time = time.time()
        self.logger.info(f"Receive heart beat. {worker_id}")
        return True

    def remove_stable_workers_by_expiration(self):
        expire = time.time() - CONTROLLER_HEART_BEAT_EXPIRATION
        to_delete = []
        for worker_id, w_info in self.worker_info.items():
            if w_info.check_heart_beat and w_info.last_heart_beat < expire:
                self.logger.info(f"Worker hearbeat expiration: {worker_id}")
                to_delete.append(worker_id)

        for worker_id in to_delete:
            self.remove_worker(RemoveWorkerRequest(worker_id=worker_id))

    def handle_no_worker(params, server_error_msg):
        self.logger.info(f"no worker: {params['model']}")
        ret = {
            "text": server_error_msg,
            "error_code": 2,
        }
        return json.dumps(ret).encode() + b"\0"

    def handle_worker_timeout(worker_address, server_error_msg):
        self.logger.info(f"worker timeout: {worker_address}")
        ret = {
            "text": server_error_msg,
            "error_code": 3,
        }
        return json.dumps(ret).encode() + b"\0"

    def worker_api_generate_stream(self, params):
        worker_addr = self.get_worker_address(params["model"])
        if not worker_addr:
            yield self.handle_no_worker(params, server_error_msg)

        try:
            response = requests.post(
                worker_addr + "/worker_generate_stream",
                json=params,
                stream=True,
                timeout=15,
            )
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    yield chunk + b"\0"
        except requests.exceptions.RequestException as e:
            yield self.handle_worker_timeout(worker_addr, server_error_msg)

    def worker_api_generate_completion(self, params):
        worker_addr = self.get_worker_address(params["model"])
        if not worker_addr:
            return self.handle_no_worker(params, server_error_msg)

        try:
            response = requests.post(
                worker_addr + "/worker_generate_completion",
                json=params,
                timeout=15,
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            return self.handle_worker_timeout(worker_addr, server_error_msg)

    def worker_api_embeddings(self, params):
        worker_addr = self.get_worker_address(params["model"])
        if not worker_addr:
            return self.handle_no_worker(params, server_error_msg)

        try:
            response = requests.post(
                worker_addr + "/worker_get_embeddings",
                json=params,
                timeout=15,
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            return self.handle_worker_timeout(worker_addr, server_error_msg)

    # Let the controller act as a worker to achieve hierarchical
    # management. This can be used to connect isolated sub networks.
    def worker_api_get_status(self):
        model_names = set()
        speed = 0
        queue_length = 0

        for w_name in self.worker_info:
            worker_status = self.get_worker_status(w_name)
            if worker_status is not None:
                model_names.update(worker_status["model_names"])
                speed += worker_status["speed"]
                queue_length += worker_status["queue_length"]

        return {
            "model_names": list(model_names),
            "speed": speed,
            "queue_length": queue_length,
        }

